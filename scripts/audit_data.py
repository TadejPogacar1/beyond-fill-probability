#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from fill_probability.config import ALL_DEPTHS_BPS
from fill_probability.utils import read_dates


def audit_state(state_dir: Path, dates: list[str]) -> pd.DataFrame:
    rows = []
    for day in dates:
        path = state_dir / f"{day}.parquet"
        if not path.exists():
            rows.append({"date": day, "exists": False})
            continue
        table = pq.read_table(path, columns=["timestamp_ms"])
        timestamp = table["timestamp_ms"].combine_chunks().to_numpy().astype(np.int64)
        ordered = np.sort(timestamp)
        complete = (
            len(ordered) == 86_400
            and len(np.unique(ordered)) == 86_400
            and np.all(np.diff(ordered) == 1_000)
        )
        rows.append(
            {
                "date": day,
                "exists": True,
                "state_seconds": int(len(ordered)),
                "coverage_pct": 100.0 * len(ordered) / 86_400,
                "first_timestamp_utc": pd.to_datetime(
                    ordered.min(), unit="ms", utc=True
                ),
                "last_timestamp_utc": pd.to_datetime(
                    ordered.max(), unit="ms", utc=True
                ),
                "data_quality": "complete" if complete else "incomplete",
            }
        )
    return pd.DataFrame(rows)


def audit_samples(
    sample_root: Path,
    dates: list[str],
    horizons: list[int],
) -> pd.DataFrame:
    expected_depths = np.sort(np.asarray(ALL_DEPTHS_BPS, dtype=np.float64))
    rows = []
    for horizon in horizons:
        for day in tqdm(dates, desc=f"Audit T={horizon}s", unit="day"):
            path = sample_root / f"T{horizon}" / f"{day}.parquet"
            if not path.exists():
                rows.append(
                    {
                        "horizon_seconds": horizon,
                        "date": day,
                        "exists": False,
                    }
                )
                continue
            table = pq.read_table(
                path,
                columns=["timestamp_ms", "fill_time_ms", "filled", "depth_bps"],
            )
            timestamp = table["timestamp_ms"].combine_chunks().to_numpy().astype(np.int64)
            fill_time = table["fill_time_ms"].combine_chunks().to_numpy().astype(np.int64)
            filled = table["filled"].combine_chunks().to_numpy().astype(bool)
            depths = np.sort(
                np.unique(
                    table["depth_bps"].combine_chunks().to_numpy().astype(np.float64)
                )
            )
            rows.append(
                {
                    "horizon_seconds": horizon,
                    "date": day,
                    "exists": True,
                    "rows": int(len(timestamp)),
                    "fills": int(filled.sum()),
                    "past_or_equal": int(
                        np.count_nonzero(filled & (fill_time <= timestamp))
                    ),
                    "late": int(
                        np.count_nonzero(
                            filled & (fill_time > timestamp + horizon * 1_000)
                        )
                    ),
                    "flag_mismatch": int(
                        np.count_nonzero(filled != (fill_time >= 0))
                    ),
                    "depths_match": bool(
                        depths.shape == expected_depths.shape
                        and np.allclose(
                            depths,
                            expected_depths,
                            rtol=0.0,
                            atol=1e-6,
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit state and labeled samples")
    parser.add_argument(
        "--state-dir", type=Path, default=Path("data/processed/market_state_1s")
    )
    parser.add_argument(
        "--sample-root", type=Path, default=Path("data/processed/samples")
    )
    parser.add_argument("--dates-file", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[60, 300, 600])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/generated/audit")
    )
    args = parser.parse_args()

    dates = read_dates(args.dates_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = audit_state(args.state_dir, dates)
    samples = audit_samples(args.sample_root, dates, args.horizons)
    state.to_csv(args.output_dir / "state_coverage.csv", index=False)
    samples.to_csv(args.output_dir / "sample_integrity.csv", index=False)
    print(state.to_string(index=False))
    print(samples.groupby("horizon_seconds").agg(days=("date", "count")))

    failures = samples.loc[
        (~samples["exists"].fillna(False))
        | (samples["past_or_equal"].fillna(1) != 0)
        | (samples["late"].fillna(1) != 0)
        | (samples["flag_mismatch"].fillna(1) != 0)
        | (~samples["depths_match"].fillna(False))
    ]
    if not failures.empty:
        raise RuntimeError(f"Sample audit failed for {len(failures)} files")


if __name__ == "__main__":
    main()
