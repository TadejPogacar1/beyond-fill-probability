from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .config import ALL_DEPTHS_BPS, ExperimentConfig, LONG_FEATURES, SHORT_FEATURES
from .loo import _classifier, _regressor
from .utils import atomic_joblib, atomic_json, read_dates


CACHE_VERSION = "full_float32_quantized_to_float64_v1"
LABEL_VERSION = "integer_quantity_and_strict_time_v1"
TARGET_COLUMNS = ["filled", "net_outcome_bps", "adverse_selection_bps"]


def model_paths(model_root: Path, horizon: int) -> dict[str, Path]:
    root = model_root / f"T{horizon}"
    return {
        "fill": root / "fill_classifier.joblib",
        "adverse": root / "adverse_regressor.joblib",
        "short": root / "unified_short_regressor.joblib",
        "long": root / "unified_long_regressor.joblib",
        "metadata": root / "metadata.json",
    }


def cache_paths(cache_root: Path, horizon: int) -> dict[str, Path]:
    root = cache_root / f"T{horizon}"
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "x_long": root / "x_long_float64.dat",
        "filled": root / "filled_uint8.dat",
        "net": root / "net_outcome_float32.dat",
        "adverse": root / "adverse_all_float32.dat",
        "x_adverse": root / "x_adverse_short_float64.dat",
        "y_adverse": root / "y_adverse_float32.dat",
        "adverse_manifest": root / "adverse_manifest.json",
    }


def sample_dir(sample_root: Path, horizon: int) -> Path:
    return sample_root / f"T{horizon}"


def source_signature(
    sample_root: Path,
    horizon: int,
    train_dates: list[str],
) -> list[dict]:
    signature = []
    for day in train_dates:
        path = sample_dir(sample_root, horizon) / f"{day}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        signature.append(
            {
                "date": day,
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "rows": pq.ParquetFile(path).metadata.num_rows,
            }
        )
    return signature


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def base_cache_valid(
    sample_root: Path,
    cache_root: Path,
    horizon: int,
    train_dates: list[str],
    n_rows: int,
) -> bool:
    paths = cache_paths(cache_root, horizon)
    manifest = load_manifest(paths["manifest"])
    if not manifest:
        return False
    expected = (
        manifest.get("version") == CACHE_VERSION
        and manifest.get("horizon_seconds") == horizon
        and manifest.get("train_dates") == train_dates
        and manifest.get("features") == LONG_FEATURES
        and manifest.get("n_rows") == n_rows
        and manifest.get("source_signature")
        == source_signature(sample_root, horizon, train_dates)
    )
    if not expected:
        return False
    expected_sizes = {
        "x_long": n_rows * len(LONG_FEATURES) * np.dtype(np.float64).itemsize,
        "filled": n_rows * np.dtype(np.uint8).itemsize,
        "net": n_rows * np.dtype(np.float32).itemsize,
        "adverse": n_rows * np.dtype(np.float32).itemsize,
    }
    return all(
        paths[name].exists() and paths[name].stat().st_size == size
        for name, size in expected_sizes.items()
    )


def build_base_cache(
    sample_root: Path,
    cache_root: Path,
    horizon: int,
    train_dates: list[str],
    batch_size: int,
    force: bool,
) -> dict:
    paths = cache_paths(cache_root, horizon)
    paths["root"].mkdir(parents=True, exist_ok=True)
    signatures = source_signature(sample_root, horizon, train_dates)
    n_rows = sum(item["rows"] for item in signatures)
    if (
        not force
        and base_cache_valid(
            sample_root,
            cache_root,
            horizon,
            train_dates,
            n_rows,
        )
    ):
        print(f"T={horizon}s: reusing the complete memory-mapped cache")
        manifest = load_manifest(paths["manifest"])
        assert manifest is not None
        return manifest

    temporary = {
        name: paths[name].with_suffix(paths[name].suffix + ".building")
        for name in ("x_long", "filled", "net", "adverse")
    }
    x_map = np.memmap(
        temporary["x_long"],
        dtype=np.float64,
        mode="w+",
        shape=(n_rows, len(LONG_FEATURES)),
    )
    y_filled = np.memmap(
        temporary["filled"], dtype=np.uint8, mode="w+", shape=(n_rows,)
    )
    y_net = np.memmap(
        temporary["net"], dtype=np.float32, mode="w+", shape=(n_rows,)
    )
    y_adverse = np.memmap(
        temporary["adverse"], dtype=np.float32, mode="w+", shape=(n_rows,)
    )

    columns = LONG_FEATURES + TARGET_COLUMNS
    offset = 0
    started = time.perf_counter()
    for number, day in enumerate(train_dates, start=1):
        path = sample_dir(sample_root, horizon) / f"{day}.parquet"
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=True,
        ):
            length = batch.num_rows
            x_chunk = np.empty((length, len(LONG_FEATURES)), dtype=np.float32)
            for column, feature in enumerate(LONG_FEATURES):
                values = batch.column(
                    batch.schema.get_field_index(feature)
                ).to_numpy(zero_copy_only=False)
                x_chunk[:, column] = values.astype(np.float32, copy=False)
            x_map[offset : offset + length] = x_chunk
            y_filled[offset : offset + length] = batch.column(
                batch.schema.get_field_index("filled")
            ).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
            y_net[offset : offset + length] = batch.column(
                batch.schema.get_field_index("net_outcome_bps")
            ).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            y_adverse[offset : offset + length] = batch.column(
                batch.schema.get_field_index("adverse_selection_bps")
            ).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            offset += length
            del x_chunk, batch
        print(
            f"T={horizon}s training day {number:02d}/{len(train_dates)}: "
            f"{day}; cumulative rows={offset:,}",
            flush=True,
        )

    if offset != n_rows:
        raise RuntimeError(f"Cache row mismatch: wrote {offset}, expected {n_rows}")
    for array in (x_map, y_filled, y_net, y_adverse):
        array.flush()
    del x_map, y_filled, y_net, y_adverse
    gc.collect()
    for name in temporary:
        os.replace(temporary[name], paths[name])

    manifest = {
        "version": CACHE_VERSION,
        "horizon_seconds": horizon,
        "train_dates": train_dates,
        "features": LONG_FEATURES,
        "short_feature_count": len(SHORT_FEATURES),
        "n_rows": n_rows,
        "source_signature": signatures,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_minutes": (time.perf_counter() - started) / 60,
        "input_quantization": "Parquet -> float32 -> float64",
    }
    atomic_json(paths["manifest"], manifest)
    return manifest


def adverse_cache_valid(
    cache_root: Path,
    horizon: int,
    base_manifest: dict,
) -> bool:
    paths = cache_paths(cache_root, horizon)
    manifest = load_manifest(paths["adverse_manifest"])
    if not manifest:
        return False
    n_filled = manifest.get("n_filled", -1)
    return (
        manifest.get("version") == CACHE_VERSION
        and manifest.get("base_created_at") == base_manifest.get("created_at")
        and paths["x_adverse"].exists()
        and paths["y_adverse"].exists()
        and paths["x_adverse"].stat().st_size
        == n_filled * len(SHORT_FEATURES) * np.dtype(np.float64).itemsize
        and paths["y_adverse"].stat().st_size
        == n_filled * np.dtype(np.float32).itemsize
    )


def build_adverse_cache(
    cache_root: Path,
    horizon: int,
    base_manifest: dict,
    force: bool,
) -> dict:
    paths = cache_paths(cache_root, horizon)
    if not force and adverse_cache_valid(cache_root, horizon, base_manifest):
        print(f"T={horizon}s: reusing the filled-order subset cache")
        manifest = load_manifest(paths["adverse_manifest"])
        assert manifest is not None
        return manifest

    n_rows = base_manifest["n_rows"]
    x_long = np.memmap(
        paths["x_long"],
        dtype=np.float64,
        mode="r",
        shape=(n_rows, len(LONG_FEATURES)),
    )
    filled = np.memmap(
        paths["filled"], dtype=np.uint8, mode="r", shape=(n_rows,)
    )
    adverse = np.memmap(
        paths["adverse"], dtype=np.float32, mode="r", shape=(n_rows,)
    )
    n_filled = int(np.count_nonzero(filled))

    x_temporary = paths["x_adverse"].with_suffix(
        paths["x_adverse"].suffix + ".building"
    )
    y_temporary = paths["y_adverse"].with_suffix(
        paths["y_adverse"].suffix + ".building"
    )
    x_output = np.memmap(
        x_temporary,
        dtype=np.float64,
        mode="w+",
        shape=(n_filled, len(SHORT_FEATURES)),
    )
    y_output = np.memmap(
        y_temporary, dtype=np.float32, mode="w+", shape=(n_filled,)
    )

    write_at = 0
    chunk_rows = 1_000_000
    for start in tqdm(
        range(0, n_rows, chunk_rows),
        desc=f"T={horizon}s filled subset",
        unit="chunk",
    ):
        stop = min(start + chunk_rows, n_rows)
        mask = filled[start:stop].astype(bool, copy=False)
        count = int(mask.sum())
        if count:
            x_output[write_at : write_at + count] = x_long[
                start:stop, : len(SHORT_FEATURES)
            ][mask]
            y_output[write_at : write_at + count] = adverse[start:stop][mask]
            write_at += count
    if write_at != n_filled:
        raise RuntimeError(
            f"Filled subset mismatch: wrote {write_at}, expected {n_filled}"
        )
    x_output.flush()
    y_output.flush()
    del x_output, y_output, x_long, filled, adverse
    gc.collect()
    os.replace(x_temporary, paths["x_adverse"])
    os.replace(y_temporary, paths["y_adverse"])
    manifest = {
        "version": CACHE_VERSION,
        "horizon_seconds": horizon,
        "base_created_at": base_manifest["created_at"],
        "n_filled": n_filled,
    }
    atomic_json(paths["adverse_manifest"], manifest)
    return manifest


def valid_model(path: Path, expected_features: int, force: bool) -> object | None:
    if force or not path.exists():
        return None
    try:
        model = joblib.load(path)
        if int(model.n_features_in_) != expected_features:
            return None
        print(f"Reusing model: {path}")
        return model
    except Exception:
        return None


def timed_fit(
    label: str,
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
) -> tuple[object, float]:
    started = time.perf_counter()
    print(f"Training {label}: {len(y):,} rows x {x.shape[1]} features", flush=True)
    model.fit(x, y)
    minutes = (time.perf_counter() - started) / 60
    atomic_joblib(model, output_path)
    print(f"Completed {label} in {minutes:.1f} minutes", flush=True)
    return model, minutes


def train_horizon(
    sample_root: Path,
    cache_root: Path,
    model_root: Path,
    horizon: int,
    train_dates: list[str],
    batch_size: int,
    force_cache: bool,
    force_models: bool,
) -> dict:
    paths = model_paths(model_root, horizon)
    models = {
        "fill": valid_model(paths["fill"], len(SHORT_FEATURES), force_models),
        "adverse": valid_model(
            paths["adverse"], len(SHORT_FEATURES), force_models
        ),
        "short": valid_model(paths["short"], len(SHORT_FEATURES), force_models),
        "long": valid_model(paths["long"], len(LONG_FEATURES), force_models),
    }
    if all(model is not None for model in models.values()):
        metadata = load_manifest(paths["metadata"])
        if metadata is None:
            raise RuntimeError(f"Models exist without metadata: {paths['metadata']}")
        return metadata

    base_manifest = build_base_cache(
        sample_root,
        cache_root,
        horizon,
        train_dates,
        batch_size,
        force_cache,
    )
    cache = cache_paths(cache_root, horizon)
    n_rows = base_manifest["n_rows"]
    config = replace(
        ExperimentConfig(),
        horizon_seconds=horizon,
        max_train_rows_per_day=0,
    )
    timings: dict[str, float] = {}

    if models["fill"] is None:
        x = np.memmap(
            cache["x_long"],
            dtype=np.float64,
            mode="r",
            shape=(n_rows, len(LONG_FEATURES)),
        )[:, : len(SHORT_FEATURES)]
        y = np.memmap(cache["filled"], dtype=np.uint8, mode="r", shape=(n_rows,))
        models["fill"], timings["fill_minutes"] = timed_fit(
            f"T={horizon} fill classifier",
            _classifier(config),
            x,
            y,
            paths["fill"],
        )
        del x, y
        gc.collect()

    if models["adverse"] is None:
        adverse_manifest = build_adverse_cache(
            cache_root,
            horizon,
            base_manifest,
            force_cache,
        )
        n_filled = adverse_manifest["n_filled"]
        x = np.memmap(
            cache["x_adverse"],
            dtype=np.float64,
            mode="r",
            shape=(n_filled, len(SHORT_FEATURES)),
        )
        y = np.memmap(
            cache["y_adverse"],
            dtype=np.float32,
            mode="r",
            shape=(n_filled,),
        )
        models["adverse"], timings["adverse_minutes"] = timed_fit(
            f"T={horizon} adverse-selection regressor",
            _regressor(config),
            x,
            y,
            paths["adverse"],
        )
        del x, y
        gc.collect()

    if models["short"] is None:
        x = np.memmap(
            cache["x_long"],
            dtype=np.float64,
            mode="r",
            shape=(n_rows, len(LONG_FEATURES)),
        )[:, : len(SHORT_FEATURES)]
        y = np.memmap(cache["net"], dtype=np.float32, mode="r", shape=(n_rows,))
        models["short"], timings["short_minutes"] = timed_fit(
            f"T={horizon} unified-short regressor",
            _regressor(config),
            x,
            y,
            paths["short"],
        )
        del x, y
        gc.collect()

    if models["long"] is None:
        x = np.memmap(
            cache["x_long"],
            dtype=np.float64,
            mode="r",
            shape=(n_rows, len(LONG_FEATURES)),
        )
        y = np.memmap(cache["net"], dtype=np.float32, mode="r", shape=(n_rows,))
        models["long"], timings["long_minutes"] = timed_fit(
            f"T={horizon} unified-long regressor",
            _regressor(config),
            x,
            y,
            paths["long"],
        )
        del x, y
        gc.collect()

    metadata = {
        "train_dates": train_dates,
        "train_rows": n_rows,
        "max_train_rows_per_day": 0,
        "score_threshold_bps": config.score_threshold_bps,
        "random_seed": config.random_seed,
        "model_iterations": {
            "fill": int(models["fill"].n_iter_),
            "adverse": int(models["adverse"].n_iter_),
            "unified_short": int(models["short"].n_iter_),
            "unified_long": int(models["long"].n_iter_),
        },
        "horizon_seconds": horizon,
        "depths_bps": list(ALL_DEPTHS_BPS),
        "label_version": LABEL_VERSION,
        "training_storage": CACHE_VERSION,
        "training_minutes": timings,
    }
    atomic_json(paths["metadata"], metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train four full-data models with memory-mapped arrays"
    )
    parser.add_argument(
        "--sample-root", type=Path, default=Path("data/processed/samples")
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("artifacts/training_cache")
    )
    parser.add_argument("--model-root", type=Path, default=Path("artifacts/models"))
    parser.add_argument(
        "--train-dates-file",
        type=Path,
        default=Path("configs/train_dates.txt"),
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[60, 300, 600])
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--force-models", action="store_true")
    args = parser.parse_args()

    train_dates = read_dates(args.train_dates_file)
    for horizon in args.horizons:
        metadata = train_horizon(
            args.sample_root,
            args.cache_root,
            args.model_root,
            horizon,
            train_dates,
            args.batch_size,
            args.force_cache,
            args.force_models,
        )
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
