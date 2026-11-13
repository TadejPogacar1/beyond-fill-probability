from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from .config import ALL_DEPTHS_BPS, ExperimentConfig, LONG_FEATURES, SHORT_FEATURES
from .loo import SIMULATION_COLUMNS, _end_of_day_mid
from .simulator import simulate_single_level_day
from .utils import atomic_json, read_dates


OBJECTIVES = ("fill_only", "two_stage", "unified_short", "unified_long")
TARGET_COLUMNS = ["filled", "net_outcome_bps", "adverse_selection_bps"]
LABEL_VERSION = "integer_quantity_and_strict_time_v1"
EXPERIMENT_VERSION = "three_horizons_all12_train42_test7_v1"


def sample_dir(sample_root: Path, horizon: int) -> Path:
    return sample_root / f"T{horizon}"


def model_paths(model_root: Path, horizon: int) -> dict[str, Path]:
    root = model_root / f"T{horizon}"
    return {
        "fill": root / "fill_classifier.joblib",
        "adverse": root / "adverse_regressor.joblib",
        "short": root / "unified_short_regressor.joblib",
        "long": root / "unified_long_regressor.joblib",
        "metadata": root / "metadata.json",
    }


def load_models(
    model_root: Path,
    horizon: int,
    expected_train_dates: list[str] | None = None,
) -> tuple[dict[str, object], dict]:
    paths = model_paths(model_root, horizon)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {missing}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("horizon_seconds") != horizon:
        raise ValueError(f"Model horizon mismatch in {paths['metadata']}")
    if expected_train_dates and metadata.get("train_dates") != expected_train_dates:
        raise ValueError(f"Training date mismatch in {paths['metadata']}")
    label_version = metadata.get("label_version", metadata.get("label_fix_version"))
    if label_version != LABEL_VERSION:
        raise ValueError(f"Label version mismatch in {paths['metadata']}")

    models = {
        "fill": joblib.load(paths["fill"]),
        "adverse": joblib.load(paths["adverse"]),
        "short": joblib.load(paths["short"]),
        "long": joblib.load(paths["long"]),
    }
    expected_features = {
        "fill": len(SHORT_FEATURES),
        "adverse": len(SHORT_FEATURES),
        "short": len(SHORT_FEATURES),
        "long": len(LONG_FEATURES),
    }
    for name, expected in expected_features.items():
        observed = int(models[name].n_features_in_)
        if observed != expected:
            raise ValueError(
                f"{name} expects {observed} features; repository expects {expected}"
            )
    return models, metadata


def scores_and_metrics(
    test: pd.DataFrame,
    models: dict[str, object],
    config: ExperimentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    x_short = test[SHORT_FEATURES].to_numpy(np.float32)
    probability = models["fill"].predict_proba(x_short)[:, 1]
    filled = test["filled"].to_numpy(np.int8)
    fill_metrics = {
        "auc": float(roc_auc_score(filled, probability)),
        "brier": float(brier_score_loss(filled, probability)),
    }

    depth = test["depth_bps"].to_numpy(np.float64)
    fill_only = probability * (depth - config.explicit_cost_bps)
    predicted_adverse = models["adverse"].predict(x_short)
    two_stage = probability * (
        depth - predicted_adverse - config.explicit_cost_bps
    )

    target = test["net_outcome_bps"].to_numpy(np.float32)
    unified_short = models["short"].predict(x_short)
    unified_long = models["long"].predict(
        test[LONG_FEATURES].to_numpy(np.float32)
    )
    short_metrics = {
        "mae_bps": float(mean_absolute_error(target, unified_short)),
        "rmse_bps": float(np.sqrt(mean_squared_error(target, unified_short))),
    }
    long_metrics = {
        "mae_bps": float(mean_absolute_error(target, unified_long)),
        "rmse_bps": float(np.sqrt(mean_squared_error(target, unified_long))),
    }
    scores = {
        "fill_only": fill_only,
        "two_stage": two_stage,
        "unified_short": unified_short,
        "unified_long": unified_long,
    }
    metrics = {
        "fill_only": fill_metrics,
        "two_stage": fill_metrics,
        "unified_short": short_metrics,
        "unified_long": long_metrics,
    }
    return scores, metrics


def assert_causal_labels(frame: pd.DataFrame, horizon: int) -> None:
    timestamp = frame["timestamp_ms"].to_numpy(np.int64)
    fill_time = frame["fill_time_ms"].to_numpy(np.int64)
    filled = fill_time >= 0
    if np.any(filled & (fill_time <= timestamp)):
        raise AssertionError("At least one fill occurs at or before order placement")
    if np.any(filled & (fill_time > timestamp + horizon * 1_000)):
        raise AssertionError("At least one fill occurs after the configured horizon")
    observed = np.sort(frame["depth_bps"].unique().astype(np.float64))
    expected = np.sort(np.asarray(ALL_DEPTHS_BPS, dtype=np.float64))
    if observed.shape != expected.shape or not np.allclose(
        observed,
        expected,
        rtol=0.0,
        atol=1e-6,
    ):
        raise AssertionError(f"Depth grid mismatch: {observed}")


def summarize(
    daily: pd.DataFrame,
    objective: str,
    config: ExperimentConfig,
    train_dates: list[str],
    scope: str,
) -> dict:
    pnl = daily["pnl_usd"].to_numpy(np.float64)
    standard_deviation = float(np.std(pnl, ddof=1))
    cumulative = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[0.0, cumulative])[-len(cumulative) :]
    drawdown = cumulative - peaks
    fills = int(daily["fills"].sum())
    submitted = int(daily["submitted_orders"].sum())
    sharpe = (
        float(
            pnl.mean()
            / standard_deviation
            * np.sqrt(config.annualization_days)
        )
        if standard_deviation > 0
        else float("nan")
    )
    return {
        "horizon_seconds": config.horizon_seconds,
        "objective": objective,
        "data_scope": scope,
        "train_days": len(train_dates),
        "test_days": int(len(daily)),
        "total_pnl_usd": float(pnl.sum()),
        "mean_pnl_per_day_usd": float(pnl.mean()),
        "median_pnl_per_day_usd": float(np.median(pnl)),
        "pnl_standard_deviation_usd": standard_deviation,
        "annualized_sharpe": sharpe,
        "winning_days": int(np.count_nonzero(pnl > 0)),
        "fills_per_day": float(daily["fills"].mean()),
        "submitted_orders_per_day": float(daily["submitted_orders"].mean()),
        "fill_rate_pct": float(100 * fills / submitted) if submitted else float("nan"),
        "pnl_per_fill_usd": float(pnl.sum() / fills) if fills else float("nan"),
        "max_drawdown_usd": float(drawdown.min()),
        "score_threshold_bps": config.score_threshold_bps,
    }


def coverage_map(path: Path | None, test_dates: list[str]) -> dict[str, str]:
    if path is None:
        return {day: "unknown" for day in test_dates}
    coverage = pd.read_csv(path)
    if not {"date", "data_quality"}.issubset(coverage.columns):
        raise ValueError("Coverage CSV must contain date and data_quality columns")
    mapping = coverage.set_index("date")["data_quality"].to_dict()
    return {day: str(mapping.get(day, "unknown")) for day in test_dates}


def evaluate_horizon(
    sample_root: Path,
    state_dir: Path,
    model_root: Path,
    result_root: Path,
    horizon: int,
    train_dates: list[str],
    test_dates: list[str],
    quality: dict[str, str],
    force: bool,
) -> dict[tuple[int, str], pd.DataFrame]:
    models, _ = load_models(model_root, horizon, train_dates)
    config = replace(
        ExperimentConfig(),
        sample_dir=sample_dir(sample_root, horizon),
        state_dir=state_dir,
        result_dir=result_root / f"T{horizon}",
        horizon_seconds=horizon,
        all_depths_bps=ALL_DEPTHS_BPS,
        final_depths_bps=ALL_DEPTHS_BPS,
        max_train_rows_per_day=0,
    )
    columns = list(dict.fromkeys(SIMULATION_COLUMNS + LONG_FEATURES + TARGET_COLUMNS))
    fold_root = config.result_dir / "daily"

    for number, day in enumerate(test_dates, start=1):
        outputs = {
            objective: fold_root / objective / f"{day}.json"
            for objective in OBJECTIVES
        }
        if not force and all(path.exists() for path in outputs.values()):
            rows = [json.loads(path.read_text(encoding="utf-8")) for path in outputs.values()]
            if all(row.get("experiment_version") == EXPERIMENT_VERSION for row in rows):
                print(f"T={horizon} [{number:02d}/{len(test_dates)}] {day}: complete")
                continue

        started = time.perf_counter()
        test_path = config.sample_dir / f"{day}.parquet"
        if not test_path.exists():
            raise FileNotFoundError(test_path)
        test = pq.read_table(test_path, columns=columns).to_pandas()
        assert_causal_labels(test, horizon)
        scores, metrics = scores_and_metrics(test, models, config)
        end_mid = _end_of_day_mid(state_dir, day, config.price_scale)

        print(
            f"T={horizon} [{number:02d}/{len(test_dates)}] {day} "
            f"quality={quality[day]}",
            flush=True,
        )
        for objective in OBJECTIVES:
            simulation = simulate_single_level_day(
                test,
                scores[objective],
                end_mid,
                config,
            )
            row = {
                "date": day,
                "horizon_seconds": horizon,
                "data_quality": quality[day],
                "label_version": LABEL_VERSION,
                "experiment_version": EXPERIMENT_VERSION,
                "train_days": len(train_dates),
                "test_candidates": int(len(test)),
                **metrics[objective],
                **simulation,
            }
            atomic_json(outputs[objective], row)
            print(
                f"  {objective:15s} PnL=${simulation['pnl_usd']:9.2f}, "
                f"fills={simulation['fills']:4d}",
                flush=True,
            )
        print(f"  elapsed={(time.perf_counter() - started) / 60:.1f} min")
        del test, scores
        gc.collect()

    frames: dict[tuple[int, str], pd.DataFrame] = {}
    for objective in OBJECTIVES:
        records = [
            json.loads((fold_root / objective / f"{day}.json").read_text(encoding="utf-8"))
            for day in test_dates
        ]
        frames[(horizon, objective)] = pd.DataFrame(records).sort_values("date")
    return frames


def save_results(
    result_root: Path,
    frames: dict[tuple[int, str], pd.DataFrame],
    train_dates: list[str],
    complete_dates: list[str],
) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    complete_summaries = []
    for (horizon, objective), daily in frames.items():
        output = result_root / f"T{horizon}" / objective
        output.mkdir(parents=True, exist_ok=True)
        daily.to_csv(output / "daily_results_all.csv", index=False)
        complete = daily[daily["date"].isin(complete_dates)].copy()
        complete.to_csv(output / "daily_results_complete.csv", index=False)
        config = replace(ExperimentConfig(), horizon_seconds=horizon)
        all_summaries.append(
            summarize(daily, objective, config, train_dates, "all")
        )
        complete_summaries.append(
            summarize(complete, objective, config, train_dates, "complete")
        )

    all_table = pd.DataFrame(all_summaries).sort_values(
        ["horizon_seconds", "objective"]
    )
    complete_table = pd.DataFrame(complete_summaries).sort_values(
        ["horizon_seconds", "objective"]
    )
    comparison = result_root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    all_table.to_csv(comparison / "model_summary_all.csv", index=False)
    complete_table.to_csv(comparison / "model_summary_complete.csv", index=False)
    pnl_wide = pd.DataFrame(
        {
            f"T{horizon}_{objective}": daily.set_index("date")["pnl_usd"]
            for (horizon, objective), daily in frames.items()
        }
    )
    pnl_wide.to_csv(comparison / "daily_pnl.csv")

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    horizons = sorted({horizon for horizon, _ in frames})
    for axis, horizon in zip(axes.flat[:3], horizons):
        for objective in OBJECTIVES:
            daily = frames[(horizon, objective)].set_index("date")["pnl_usd"]
            axis.plot(
                pd.to_datetime(daily.index),
                daily.cumsum(),
                marker="o",
                markersize=3,
                label=objective,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"T={horizon}s: cumulative frozen-holdout PnL")
        axis.set_ylabel("USD")
        axis.tick_params(axis="x", rotation=35)
        axis.legend()

    axis = axes.flat[3]
    labels = [
        f"T{int(row.horizon_seconds)}\n{row.objective}"
        for row in all_table.itertuples()
    ]
    x = np.arange(len(labels))
    width = 0.40
    axis.bar(x - width / 2, all_table["total_pnl_usd"], width, label="all days")
    axis.bar(
        x + width / 2,
        complete_table["total_pnl_usd"],
        width,
        label="complete days",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_title("All strategy configurations")
    axis.set_ylabel("Total PnL (USD)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(comparison / "holdout_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen models on chronological holdout dates"
    )
    parser.add_argument(
        "--sample-root", type=Path, default=Path("data/processed/samples")
    )
    parser.add_argument(
        "--state-dir", type=Path, default=Path("data/processed/market_state_1s")
    )
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--result-root", type=Path, default=Path("results/generated/holdout")
    )
    parser.add_argument(
        "--train-dates-file", type=Path, default=Path("configs/train_dates.txt")
    )
    parser.add_argument(
        "--test-dates-file", type=Path, default=Path("configs/test_dates.txt")
    )
    parser.add_argument(
        "--complete-dates-file",
        type=Path,
        default=Path("configs/complete_test_dates.txt"),
    )
    parser.add_argument("--coverage-csv", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+", default=[60, 300, 600])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_dates = read_dates(args.train_dates_file)
    test_dates = read_dates(args.test_dates_file)
    complete_dates = read_dates(args.complete_dates_file)
    if not set(complete_dates).issubset(test_dates):
        raise ValueError("Complete dates must be a subset of test dates")
    quality = coverage_map(args.coverage_csv, test_dates)

    frames: dict[tuple[int, str], pd.DataFrame] = {}
    for horizon in args.horizons:
        frames.update(
            evaluate_horizon(
                args.sample_root,
                args.state_dir,
                args.model_root,
                args.result_root,
                horizon,
                train_dates,
                test_dates,
                quality,
                args.force,
            )
        )
    save_results(args.result_root, frames, train_dates, complete_dates)
    print(f"Saved results to {args.result_root}")


if __name__ == "__main__":
    main()
