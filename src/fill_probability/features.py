from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .config import ExperimentConfig, LONG_FEATURES, depth_token
from .labeling import crossing_fill_times
from .raw_io import decimal_string_to_float, read_outer_zstd_parquet


def _trade_frame(config: ExperimentConfig, day: str) -> pd.DataFrame:
    files = sorted(
        (config.raw_root / day).glob(f"*/{config.symbol}_trades.parquet*")
    )
    if len(files) != 24:
        raise FileNotFoundError(f"Expected 24 trade files for {day}; found {len(files)}")
    pieces = []
    for path in files:
        table = read_outer_zstd_parquet(
            path,
            columns=["trade_time", "price", "quantity", "is_buyer_maker"],
        )
        pieces.append(
            pd.DataFrame(
                {
                    "trade_time_ms": table["trade_time"]
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False)
                    .astype(np.int64),
                    "price_tick": np.rint(
                        decimal_string_to_float(table, "price") * config.price_scale
                    ).astype(np.int32),
                    "quantity": decimal_string_to_float(table, "quantity").astype(
                        np.float64
                    ),
                    "is_buyer_maker": table["is_buyer_maker"]
                    .combine_chunks()
                    .to_numpy(zero_copy_only=False)
                    .astype(np.bool_),
                }
            )
        )
    frame = pd.concat(pieces, ignore_index=True)
    return frame.sort_values("trade_time_ms", kind="stable").reset_index(drop=True)


def _load_gap_intervals(config: ExperimentConfig, horizon_seconds: int) -> list[tuple[int, int]]:
    if config.audit_json is None or not config.audit_json.exists():
        return []
    audit = json.loads(config.audit_json.read_text(encoding="utf-8"))
    intervals: list[tuple[int, int]] = []
    for gap in audit.get("trades", {}).get("gap_details", []):
        start = pd.Timestamp(gap["previous_trade_time_utc"]).value // 1_000_000
        end = pd.Timestamp(gap["next_trade_time_utc"]).value // 1_000_000
        intervals.append((int(start - horizon_seconds * 1000), int(end)))
    for gap in audit.get("orderbook", {}).get("snapshot_bridges", []):
        if gap.get("pre_snapshot_event_gap_ms", 0) <= 1000:
            continue
        start = pd.Timestamp(gap["previous_update_time_utc"]).value // 1_000_000
        end = pd.Timestamp(gap["snapshot_time_utc"]).value // 1_000_000
        intervals.append((int(start - horizon_seconds * 1000), int(end)))
    return sorted(intervals)


def _gap_valid_mask(times_ms: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    valid = np.ones(times_ms.size, dtype=np.bool_)
    for start, end in intervals:
        valid &= ~((times_ms >= start) & (times_ms <= end))
    return valid


def engineer_state_features(state: pd.DataFrame, price_scale: int) -> pd.DataFrame:
    state = state.sort_values("timestamp_ms").drop_duplicates("timestamp_ms", keep="last")
    state = state.reset_index(drop=True)
    state["timestamp"] = pd.to_datetime(state["timestamp_ms"], unit="ms", utc=True)
    state["mid"] = (
        state["best_bid_tick"].astype(np.float64)
        + state["best_ask_tick"].astype(np.float64)
    ) / (2.0 * price_scale)
    state["relative_spread_bps"] = (
        (state["best_ask_tick"] - state["best_bid_tick"]) / price_scale
    ) / state["mid"] * 10_000.0
    denominator = state["best_bid_qty"] + state["best_ask_qty"]
    state["imbalance"] = np.where(
        denominator > 0,
        (state["best_bid_qty"] - state["best_ask_qty"]) / denominator,
        0.0,
    )
    state["abs_imbalance"] = state["imbalance"].abs()

    log_mid = np.log(state["mid"])
    state["ret_1s_bps"] = log_mid.diff() * 10_000.0
    for seconds in (10, 30, 60, 300, 1800):
        state[f"ret_{seconds}s_bps"] = log_mid.diff(seconds) * 10_000.0
    for seconds in (30, 60, 300, 1800):
        state[f"vol_{seconds}s_bps"] = state["ret_1s_bps"].rolling(
            seconds, min_periods=seconds
        ).std(ddof=1)

    state["micro_macro_vol_ratio"] = state["vol_60s_bps"] / (
        state["vol_1800s_bps"] + 1e-8
    )
    # The paper names this feature but does not publish its formula.  We compare
    # one-minute momentum with the per-minute rate of the five-minute return.
    state["momentum_acceleration"] = (
        state["ret_60s_bps"] - state["ret_300s_bps"] / 5.0
    )
    seconds_of_day = (
        state["timestamp"].dt.hour * 3600
        + state["timestamp"].dt.minute * 60
        + state["timestamp"].dt.second
    )
    angle = 2.0 * np.pi * seconds_of_day / 86_400.0
    state["hour_sin"] = np.sin(angle)
    state["hour_cos"] = np.cos(angle)

    session = state["timestamp"].dt.strftime("%Y-%m-%d")
    state["session_volatility_bps"] = (
        state.groupby(session, sort=False)["ret_1s_bps"]
        .expanding(min_periods=60)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    running_high = state.groupby(session, sort=False)["mid"].cummax()
    running_low = state.groupby(session, sort=False)["mid"].cummin()
    state["running_daily_range_bps"] = (
        (running_high - running_low) / state["mid"] * 10_000.0
    )
    return state


def _write_side(
    writer: pq.ParquetWriter | None,
    config: ExperimentConfig,
    day: str,
    base: pd.DataFrame,
    side: int,
    trades: pd.DataFrame,
    intervals: list[tuple[int, int]],
) -> pq.ParquetWriter:
    depths = np.asarray(config.final_depths_bps, dtype=np.float32)
    n_depth = depths.size
    n_time = len(base)
    times = np.repeat(base["timestamp_ms"].to_numpy(np.int64), n_depth)
    depth_flat = np.tile(depths, n_time)

    label = "bid" if side == 1 else "ask"
    price_matrix = np.column_stack(
        [
            base[f"{label}_price_{depth_token(float(depth))}"].to_numpy(np.int32)
            for depth in depths
        ]
    )
    queue_matrix = np.column_stack(
        [
            base[f"{label}_queue_{depth_token(float(depth))}"].to_numpy(np.float32)
            for depth in depths
        ]
    )
    price_tick = price_matrix.reshape(-1)
    queue = queue_matrix.reshape(-1)

    aggressive_buy = ~trades["is_buyer_maker"].to_numpy(np.bool_)
    relevant = ~aggressive_buy if side == 1 else aggressive_buy
    trade_price = trades.loc[relevant, "price_tick"].to_numpy(np.int32)
    trade_time = trades.loc[relevant, "trade_time_ms"].to_numpy(np.int64)
    trade_qty = trades.loc[relevant, "quantity"].to_numpy(np.float64)
    candidate_key = price_tick if side == 1 else -price_tick
    trade_key = trade_price if side == 1 else -trade_price
    fill_time = crossing_fill_times(
        candidate_key.astype(np.int32),
        times,
        queue.astype(np.float64),
        config.horizon_seconds * 1000,
        trade_key.astype(np.int32),
        trade_time,
        trade_qty,
    )
    filled = fill_time >= 0

    mid_t = np.repeat(base["mid"].to_numpy(np.float64), n_depth)
    mid_future = np.repeat(base["mid_future"].to_numpy(np.float64), n_depth)
    order_price = price_tick.astype(np.float64) / config.price_scale
    if side == 1:
        gross_bps = (mid_future - order_price) / mid_t * 10_000.0
    else:
        gross_bps = (order_price - mid_future) / mid_t * 10_000.0
    net_bps = np.where(filled, gross_bps - config.explicit_cost_bps, 0.0)
    adverse_selection = -side * (mid_future - mid_t) / mid_t * 10_000.0

    data: dict[str, np.ndarray] = {
        "date": np.repeat(np.asarray([day], dtype=object), times.size),
        "timestamp_ms": times,
        "side": np.full(times.size, side, dtype=np.int8),
        "depth_bps": depth_flat,
        "log_depth": np.log1p(depth_flat).astype(np.float32),
        "imbalance": np.repeat(base["imbalance"].to_numpy(np.float32), n_depth),
        "abs_imbalance": np.repeat(base["abs_imbalance"].to_numpy(np.float32), n_depth),
        "relative_spread_bps": np.repeat(
            base["relative_spread_bps"].to_numpy(np.float32), n_depth
        ),
        "l1_size": np.repeat(
            base["best_bid_qty" if side == 1 else "best_ask_qty"].to_numpy(np.float32),
            n_depth,
        ),
        "queue_ahead": queue.astype(np.float32),
        "order_price": order_price,
        "mid_t": mid_t,
        "mid_future": mid_future,
        "filled": filled.astype(np.int8),
        "fill_time_ms": fill_time,
        "net_outcome_bps": net_bps.astype(np.float32),
        "adverse_selection_bps": adverse_selection.astype(np.float32),
    }
    data["log_l1_size"] = np.log1p(data["l1_size"]).astype(np.float32)
    data["log_queue_ahead"] = np.log1p(data["queue_ahead"]).astype(np.float32)

    repeated_columns = [
        "ret_10s_bps",
        "ret_30s_bps",
        "ret_60s_bps",
        "vol_30s_bps",
        "vol_60s_bps",
        "ret_300s_bps",
        "ret_1800s_bps",
        "vol_300s_bps",
        "vol_1800s_bps",
        "micro_macro_vol_ratio",
        "momentum_acceleration",
        "hour_sin",
        "hour_cos",
        "session_volatility_bps",
        "running_daily_range_bps",
    ]
    for column in repeated_columns:
        data[column] = np.repeat(base[column].to_numpy(np.float32), n_depth)
    data["momentum_x_side"] = (data["ret_60s_bps"] * side).astype(np.float32)
    data["depth_x_volatility"] = (
        depth_flat * data["vol_60s_bps"]
    ).astype(np.float32)
    data["depth_x_5m_volatility"] = (
        depth_flat * data["vol_300s_bps"]
    ).astype(np.float32)

    valid = (
        (price_tick > 0)
        & np.isfinite(queue)
        & np.isfinite(net_bps)
        & _gap_valid_mask(times, intervals)
    )
    for feature in LONG_FEATURES:
        valid &= np.isfinite(data[feature])

    frame = pd.DataFrame({key: value[valid] for key, value in data.items()})
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(
            config.sample_dir / f"{day}.parquet",
            table.schema,
            compression="zstd",
        )
    writer.write_table(table, row_group_size=100_000)
    return writer


def build_daily_samples(config: ExperimentConfig, day: str) -> Path:
    current_path = config.state_dir / f"{day}.parquet"
    if not current_path.exists():
        raise FileNotFoundError(current_path)
    day_ts = pd.Timestamp(day, tz="UTC")
    previous_day = (day_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    previous_path = config.state_dir / f"{previous_day}.parquet"
    paths = [path for path in (previous_path, current_path) if path.exists()]
    state = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    state = engineer_state_features(state, config.price_scale)

    day_start_ms = day_ts.value // 1_000_000
    day_end_ms = (day_ts + pd.Timedelta(days=1)).value // 1_000_000
    latest_decision = day_end_ms - config.horizon_seconds * 1000
    current = state.loc[
        (state["timestamp_ms"] >= day_start_ms)
        & (state["timestamp_ms"] <= latest_decision)
    ].copy()
    current["mid_future"] = current["mid"].shift(-config.horizon_seconds)
    # Because current is consecutive one-second data and stops T seconds before
    # midnight, shift(-T) is the horizon-end midprice within the same day.
    full_current_mid = state.loc[
        (state["timestamp_ms"] >= day_start_ms)
        & (state["timestamp_ms"] < day_end_ms),
        ["timestamp_ms", "mid"],
    ].set_index("timestamp_ms")["mid"]
    future_times = current["timestamp_ms"] + config.horizon_seconds * 1000
    current["mid_future"] = full_current_mid.reindex(future_times.to_numpy()).to_numpy()
    current = current.dropna(
        subset=["mid_future", "ret_1800s_bps", "vol_1800s_bps"]
    )

    trades = _trade_frame(config, day)
    intervals = _load_gap_intervals(config, config.horizon_seconds)
    config.sample_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.sample_dir / f"{day}.parquet"
    if output_path.exists():
        output_path.unlink()
    writer: pq.ParquetWriter | None = None
    try:
        for side in (1, -1):
            writer = _write_side(writer, config, day, current, side, trades, intervals)
    finally:
        if writer is not None:
            writer.close()
    print(f"saved {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build candidate orders and labels")
    parser.add_argument("--state-dir", type=Path, default=Path("data/processed/market_state_1s"))
    parser.add_argument("--sample-dir", type=Path, default=Path("data/processed/samples_T300"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--horizon-seconds", type=int, default=300)
    parser.add_argument("--depths-bps", type=float, nargs="+", default=None)
    parser.add_argument("--audit-json", type=Path)
    args = parser.parse_args()
    base = ExperimentConfig()
    depths = tuple(args.depths_bps) if args.depths_bps else base.all_depths_bps
    config = ExperimentConfig(
        raw_root=args.raw_root or base.raw_root,
        state_dir=args.state_dir,
        sample_dir=args.sample_dir,
        audit_json=args.audit_json,
        horizon_seconds=args.horizon_seconds,
        all_depths_bps=depths,
        final_depths_bps=depths,
    )
    dates = tuple(args.dates) if args.dates else config.paper_style_dates
    for day in tqdm(dates, desc="Building daily labeled samples", unit="day"):
        build_daily_samples(config, day)


if __name__ == "__main__":
    main()
