from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zstandard as zstd


def read_outer_zstd_parquet(path: Path, columns: list[str] | None = None) -> pa.Table:
    """Read either ordinary Parquet or a whole-file .parquet.zst object."""

    if path.suffix == ".zst":
        raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
        return pq.read_table(io.BytesIO(raw), columns=columns)
    return pq.read_table(path, columns=columns)


def decimal_string_to_float(table: pa.Table, column: str) -> np.ndarray:
    return pc.cast(table[column], pa.float64()).combine_chunks().to_numpy(
        zero_copy_only=False
    )


def utc_hour_from_path(path: Path) -> np.datetime64:
    day = path.parts[-3]
    hour = path.parts[-2]
    return np.datetime64(f"{day}T{hour}:00:00", "ms")


def hourly_files(
    root: Path,
    start_hour: str,
    end_hour_exclusive: str,
    kind: str,
    symbol: str = "BTCUSDT",
):
    """Yield hourly files in [start_hour, end_hour_exclusive)."""

    start = np.datetime64(start_hour, "h")
    end = np.datetime64(end_hour_exclusive, "h")
    current = start
    while current < end:
        text = np.datetime_as_string(current, unit="h")
        day, hour = text.split("T")
        folder = root / day / hour
        matches = sorted(folder.glob(f"{symbol}_{kind}.parquet*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {kind} file for {text}, found {len(matches)}"
            )
        yield current, matches[0]
        current += np.timedelta64(1, "h")
