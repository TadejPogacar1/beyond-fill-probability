from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
from tqdm.auto import tqdm

from .config import ExperimentConfig, depth_token
from .raw_io import decimal_string_to_float, hourly_files, read_outer_zstd_parquet


@njit(cache=True)
def _set_level(
    side_code: int,
    tick: int,
    quantity: float,
    bid_book: np.ndarray,
    ask_book: np.ndarray,
    best_bid: int,
    best_ask: int,
) -> tuple[int, int]:
    if tick < 0 or tick >= bid_book.size:
        return best_bid, best_ask

    if side_code == 1:
        bid_book[tick] = quantity
        if quantity > 0.0 and tick > best_bid:
            best_bid = tick
        elif tick == best_bid and quantity <= 0.0:
            while best_bid >= 0 and bid_book[best_bid] <= 0.0:
                best_bid -= 1
    else:
        ask_book[tick] = quantity
        if quantity > 0.0 and tick < best_ask:
            best_ask = tick
        elif tick == best_ask and quantity <= 0.0:
            while best_ask < ask_book.size and ask_book[best_ask] <= 0.0:
                best_ask += 1
    return best_bid, best_ask


@njit(cache=True)
def apply_hour_and_sample(
    event_ms: np.ndarray,
    is_snapshot: np.ndarray,
    side_code: np.ndarray,
    price_tick: np.ndarray,
    quantity: np.ndarray,
    sample_ms: np.ndarray,
    depths_bps: np.ndarray,
    price_scale: int,
    bid_book: np.ndarray,
    ask_book: np.ndarray,
    best_bid: int,
    best_ask: int,
    initialized: bool,
    last_snapshot_ms: int,
):
    n_sample = sample_ms.size
    n_depth = depths_bps.size
    out_bid = np.full(n_sample, -1, dtype=np.int32)
    out_ask = np.full(n_sample, -1, dtype=np.int32)
    out_bid_qty = np.full(n_sample, np.nan, dtype=np.float32)
    out_ask_qty = np.full(n_sample, np.nan, dtype=np.float32)
    out_valid = np.zeros(n_sample, dtype=np.bool_)
    out_bid_price = np.full((n_sample, n_depth), -1, dtype=np.int32)
    out_ask_price = np.full((n_sample, n_depth), -1, dtype=np.int32)
    out_bid_queue = np.full((n_sample, n_depth), np.nan, dtype=np.float32)
    out_ask_queue = np.full((n_sample, n_depth), np.nan, dtype=np.float32)

    row = 0
    n_rows = event_ms.size

    for sample_index in range(n_sample):
        decision_ms = sample_ms[sample_index]
        while row < n_rows and event_ms[row] <= decision_ms:
            if is_snapshot[row] and event_ms[row] != last_snapshot_ms:
                bid_book.fill(0.0)
                ask_book.fill(0.0)
                best_bid = -1
                best_ask = ask_book.size
                initialized = True
                last_snapshot_ms = event_ms[row]

            if initialized:
                best_bid, best_ask = _set_level(
                    side_code[row],
                    price_tick[row],
                    quantity[row],
                    bid_book,
                    ask_book,
                    best_bid,
                    best_ask,
                )
            row += 1

        if initialized and best_bid >= 0 and best_ask < ask_book.size:
            if best_bid < best_ask:
                out_valid[sample_index] = True
                out_bid[sample_index] = best_bid
                out_ask[sample_index] = best_ask
                out_bid_qty[sample_index] = bid_book[best_bid]
                out_ask_qty[sample_index] = ask_book[best_ask]
                mid_price = (best_bid + best_ask) / (2.0 * price_scale)
                for depth_index in range(n_depth):
                    fraction = depths_bps[depth_index] / 10_000.0
                    bid_tick = int(np.floor(mid_price * (1.0 - fraction) * price_scale + 1e-9))
                    ask_tick = int(np.ceil(mid_price * (1.0 + fraction) * price_scale - 1e-9))
                    if bid_tick >= 0 and bid_tick < bid_book.size:
                        out_bid_price[sample_index, depth_index] = bid_tick
                        out_bid_queue[sample_index, depth_index] = bid_book[bid_tick]
                    if ask_tick >= 0 and ask_tick < ask_book.size:
                        out_ask_price[sample_index, depth_index] = ask_tick
                        out_ask_queue[sample_index, depth_index] = ask_book[ask_tick]

    # Events after xx:59:59 still determine the starting state of the next file.
    while row < n_rows:
        if is_snapshot[row] and event_ms[row] != last_snapshot_ms:
            bid_book.fill(0.0)
            ask_book.fill(0.0)
            best_bid = -1
            best_ask = ask_book.size
            initialized = True
            last_snapshot_ms = event_ms[row]
        if initialized:
            best_bid, best_ask = _set_level(
                side_code[row],
                price_tick[row],
                quantity[row],
                bid_book,
                ask_book,
                best_bid,
                best_ask,
            )
        row += 1

    return (
        out_bid,
        out_ask,
        out_bid_qty,
        out_ask_qty,
        out_valid,
        out_bid_price,
        out_ask_price,
        out_bid_queue,
        out_ask_queue,
        best_bid,
        best_ask,
        initialized,
        last_snapshot_ms,
    )


def _flush_day(day: str, pieces: list[pd.DataFrame], output_dir: Path) -> None:
    if not pieces:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(pieces, ignore_index=True)
    frame = frame.loc[frame["book_valid"]].drop(columns="book_valid")
    path = output_dir / f"{day}.parquet"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(table, path, compression="zstd", row_group_size=50_000)
    print(f"saved {path} ({len(frame):,} decision seconds)")


def build_market_state(
    config: ExperimentConfig,
    start_hour: str,
    end_hour_exclusive: str,
) -> None:
    """Reconstruct a one-second L2 state without loading the full sample at once."""

    max_tick = config.max_price_tick
    bid_book = np.zeros(max_tick + 1, dtype=np.float32)
    ask_book = np.zeros(max_tick + 1, dtype=np.float32)
    best_bid = -1
    best_ask = max_tick + 1
    initialized = False
    last_snapshot_ms = -1
    depths = np.asarray(config.all_depths_bps, dtype=np.float64)

    files = list(
        hourly_files(
            config.raw_root,
            start_hour,
            end_hour_exclusive,
            "orderbook",
            config.symbol,
        )
    )
    current_day: str | None = None
    day_pieces: list[pd.DataFrame] = []

    for hour, path in tqdm(files, desc="Reconstructing L2 book", unit="hour"):
        hour_text = np.datetime_as_string(hour, unit="h")
        day = hour_text[:10]
        if current_day is not None and day != current_day:
            _flush_day(current_day, day_pieces, config.state_dir)
            day_pieces = []
        current_day = day

        table = read_outer_zstd_parquet(
            path,
            columns=["event_time", "event_type", "side", "price", "quantity"],
        )
        event_ms = table["event_time"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        event_type = table["event_type"].combine_chunks().to_numpy(zero_copy_only=False)
        sides = table["side"].combine_chunks().to_numpy(zero_copy_only=False)
        prices = decimal_string_to_float(table, "price")
        quantities = decimal_string_to_float(table, "quantity").astype(np.float32)
        ticks = np.rint(prices * config.price_scale).astype(np.int32)
        # Binance books occasionally contain extremely distant, even obviously
        # stale price levels (for example an ask several hundred thousand dollars
        # above the touch).  The study only needs at most 30 bps around mid, so
        # the compiled state update safely ignores levels outside this generous
        # $0..max_price_usd range instead of allocating hundreds of MB more RAM.
        is_snapshot = np.asarray(event_type == "snapshot", dtype=np.bool_)
        side_code = np.where(sides == "bid", 1, -1).astype(np.int8)
        hour_start_ms = int(hour.astype("datetime64[ms]").astype(np.int64))
        sample_ms = hour_start_ms + np.arange(0, 3_600_000, 1_000, dtype=np.int64)

        result = apply_hour_and_sample(
            event_ms,
            is_snapshot,
            side_code,
            ticks,
            quantities,
            sample_ms,
            depths,
            config.price_scale,
            bid_book,
            ask_book,
            best_bid,
            best_ask,
            initialized,
            last_snapshot_ms,
        )
        (
            out_bid,
            out_ask,
            out_bid_qty,
            out_ask_qty,
            out_valid,
            out_bid_price,
            out_ask_price,
            out_bid_queue,
            out_ask_queue,
            best_bid,
            best_ask,
            initialized,
            last_snapshot_ms,
        ) = result

        frame_data: dict[str, np.ndarray] = {
            "timestamp_ms": sample_ms,
            "best_bid_tick": out_bid,
            "best_ask_tick": out_ask,
            "best_bid_qty": out_bid_qty,
            "best_ask_qty": out_ask_qty,
            "book_valid": out_valid,
        }
        for index, depth in enumerate(depths):
            token = depth_token(float(depth))
            frame_data[f"bid_price_{token}"] = out_bid_price[:, index]
            frame_data[f"ask_price_{token}"] = out_ask_price[:, index]
            frame_data[f"bid_queue_{token}"] = out_bid_queue[:, index]
            frame_data[f"ask_queue_{token}"] = out_ask_queue[:, index]
        day_pieces.append(pd.DataFrame(frame_data))

    if current_day is not None:
        _flush_day(current_day, day_pieces, config.state_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct one-second BTCUSDT L2 state")
    parser.add_argument("--raw-root", type=Path, default=ExperimentConfig().raw_root)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/market_state_1s"))
    parser.add_argument("--start-hour", required=True)
    parser.add_argument("--end-hour-exclusive", required=True)
    args = parser.parse_args()

    config = ExperimentConfig(
        raw_root=args.raw_root,
        state_dir=args.output_dir,
    )
    build_market_state(config, args.start_hour, args.end_hour_exclusive)


if __name__ == "__main__":
    main()
