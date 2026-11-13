from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from tqdm.auto import tqdm

from .config import ExperimentConfig, LONG_FEATURES, SHORT_FEATURES
from .simulator import simulate_single_level_day


SIMULATION_COLUMNS = [
    "date",
    "timestamp_ms",
    "side",
    "depth_bps",
    "order_price",
    "fill_time_ms",
    "mid_t",
]


def _feature_set(objective: str) -> list[str]:
    if objective == "unified_long":
        return LONG_FEATURES
    if objective in {"unified_short", "fill_only", "two_stage"}:
        return SHORT_FEATURES
    raise ValueError(f"Unknown objective: {objective}")


def _sample_table(
    path: Path,
    columns: list[str],
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    table = pq.read_table(path, columns=columns)
    if max_rows > 0 and table.num_rows > max_rows:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(table.num_rows, max_rows, replace=False))
        table = table.take(pa.array(indices))
    return table.to_pandas()


def _regressor(config: ExperimentConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=config.random_seed,
    )


def _classifier(config: ExperimentConfig) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=config.random_seed,
    )


def _fit_and_score(
    objective: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict]:
    x_train = train[features].to_numpy(np.float32)
    x_test = test[features].to_numpy(np.float32)
    metrics: dict[str, float] = {}

    if objective in {"unified_short", "unified_long"}:
        model = _regressor(config)
        model.fit(x_train, train["net_outcome_bps"].to_numpy(np.float32))
        score = model.predict(x_test)
        target = test["net_outcome_bps"].to_numpy(np.float32)
        metrics["mae_bps"] = float(mean_absolute_error(target, score))
        metrics["rmse_bps"] = float(np.sqrt(mean_squared_error(target, score)))
        metrics["iterations"] = int(model.n_iter_)
        return score.astype(np.float64), metrics

    classifier = _classifier(config)
    classifier.fit(x_train, train["filled"].to_numpy(np.int8))
    probability = classifier.predict_proba(x_test)[:, 1]
    fill_target = test["filled"].to_numpy(np.int8)
    metrics["auc"] = float(roc_auc_score(fill_target, probability))
    metrics["brier"] = float(brier_score_loss(fill_target, probability))
    metrics["classifier_iterations"] = int(classifier.n_iter_)

    if objective == "fill_only":
        score = probability * (
            test["depth_bps"].to_numpy(np.float64) - config.explicit_cost_bps
        )
        return score, metrics

    filled_train = train["filled"].to_numpy(np.int8) == 1
    adverse_model = _regressor(config)
    adverse_model.fit(
        x_train[filled_train],
        train.loc[filled_train, "adverse_selection_bps"].to_numpy(np.float32),
    )
    predicted_adverse = adverse_model.predict(x_test)
    score = probability * (
        test["depth_bps"].to_numpy(np.float64)
        - predicted_adverse
        - config.explicit_cost_bps
    )
    metrics["as_regressor_iterations"] = int(adverse_model.n_iter_)
    return score, metrics


def _end_of_day_mid(state_dir: Path, day: str, price_scale: int) -> float:
    table = pq.read_table(
        state_dir / f"{day}.parquet",
        columns=["best_bid_tick", "best_ask_tick"],
    )
    bid = table["best_bid_tick"][-1].as_py()
    ask = table["best_ask_tick"][-1].as_py()
    return (bid + ask) / (2.0 * price_scale)


def run_leave_one_day_out(
    config: ExperimentConfig,
    dates: tuple[str, ...],
    objective: str,
    save_oof: bool = False,
) -> dict:
    features = _feature_set(objective)
    target_columns = ["filled", "net_outcome_bps", "adverse_selection_bps"]
    sample_columns = list(dict.fromkeys(features + target_columns))
    sampled: dict[str, pd.DataFrame] = {}
    for index, day in enumerate(tqdm(dates, desc="Caching daily training samples", unit="day")):
        path = config.sample_dir / f"{day}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        sampled[day] = _sample_table(
            path,
            sample_columns,
            config.max_train_rows_per_day,
            config.random_seed + index,
        )

    config.result_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    for fold_index, held_out_day in enumerate(
        tqdm(dates, desc=f"LOO folds ({objective})", unit="fold")
    ):
        train = pd.concat(
            [sampled[day] for day in dates if day != held_out_day],
            ignore_index=True,
        )
        test_columns = list(dict.fromkeys(SIMULATION_COLUMNS + sample_columns))
        test = pq.read_table(
            config.sample_dir / f"{held_out_day}.parquet",
            columns=test_columns,
        ).to_pandas()
        score, model_metrics = _fit_and_score(
            objective, train, test, features, config
        )
        simulation = simulate_single_level_day(
            test,
            score,
            _end_of_day_mid(config.state_dir, held_out_day, config.price_scale),
            config,
        )
        fold_row = {
            "date": held_out_day,
            "train_days": len(dates) - 1,
            "test_candidates": len(test),
            **model_metrics,
            **simulation,
        }
        fold_rows.append(fold_row)

        if save_oof:
            oof = test[SIMULATION_COLUMNS].copy()
            oof["predicted_score_bps"] = score.astype(np.float32)
            pq.write_table(
                pa.Table.from_pandas(oof, preserve_index=False),
                config.result_dir / f"oof_{held_out_day}.parquet",
                compression="zstd",
            )

        del train, test, score

    daily = pd.DataFrame(fold_rows).sort_values("date")
    daily.to_csv(config.result_dir / "daily_results.csv", index=False)
    pnl = daily["pnl_usd"].to_numpy(np.float64)
    standard_deviation = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else float("nan")
    sharpe = (
        float(np.mean(pnl) / standard_deviation * np.sqrt(config.annualization_days))
        if standard_deviation > 0
        else float("nan")
    )
    summary = {
        "objective": objective,
        "dates": list(dates),
        "folds": len(dates),
        "total_pnl_usd": float(np.sum(pnl)),
        "mean_pnl_per_day_usd": float(np.mean(pnl)),
        "pnl_standard_deviation_usd": standard_deviation,
        "annualization_days": config.annualization_days,
        "annualized_sharpe": sharpe,
        "winning_days": int(np.count_nonzero(pnl > 0)),
        "fills_per_day": float(daily["fills"].mean()),
        "max_train_rows_per_day": config.max_train_rows_per_day,
        "score_threshold_bps": config.score_threshold_bps,
        "important_note": (
            "LOO models are separately fitted but not statistically independent; "
            "their training sets overlap heavily."
        ),
    }
    (config.result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 21-day leave-one-day-out experiment")
    parser.add_argument(
        "--objective",
        choices=["fill_only", "two_stage", "unified_short", "unified_long"],
        default="unified_long",
    )
    parser.add_argument("--sample-dir", type=Path, default=Path("data/processed/samples_T300"))
    parser.add_argument("--state-dir", type=Path, default=Path("data/processed/market_state_1s"))
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--dates", nargs="*")
    parser.add_argument("--max-train-rows-per-day", type=int, default=100_000)
    parser.add_argument("--score-threshold-bps", type=float, default=0.0)
    parser.add_argument("--save-oof", action="store_true")
    args = parser.parse_args()

    base = ExperimentConfig()
    result_dir = args.result_dir or Path(f"results/{args.objective}_loo21")
    config = replace(
        base,
        sample_dir=args.sample_dir,
        state_dir=args.state_dir,
        result_dir=result_dir,
        max_train_rows_per_day=args.max_train_rows_per_day,
        score_threshold_bps=args.score_threshold_bps,
    )
    dates = tuple(args.dates) if args.dates else config.paper_style_dates
    summary = run_leave_one_day_out(config, dates, args.objective, args.save_oof)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
