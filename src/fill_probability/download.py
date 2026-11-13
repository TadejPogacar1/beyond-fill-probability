from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests
from tqdm import tqdm


API = "https://api.cryptohftdata.com/v1/download"
ZSTD_MAGIC = bytes.fromhex("28b52ffd")
PARQUET_MAGIC = b"PAR1"


def parse_day(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def file_locations(
    root: Path,
    hour: datetime,
    symbol: str,
    kind: str,
) -> tuple[Path, Path, Path]:
    folder = root / hour.strftime("%Y-%m-%d") / hour.strftime("%H")
    parquet = folder / f"{symbol}_{kind}.parquet"
    return parquet, Path(f"{parquet}.zst"), Path(f"{parquet}.part")


def signature(path: Path) -> str:
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic == ZSTD_MAGIC:
        return "zstd"
    if magic == PARQUET_MAGIC:
        return "parquet"
    return "unknown"


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def download_one(
    root: Path,
    hour: datetime,
    exchange: str,
    symbol: str,
    kind: str,
    api_key: str | None,
    retries: int,
) -> tuple[Path, bool]:
    parquet, compressed, partial = file_locations(root, hour, symbol, kind)
    for existing in (parquet, compressed):
        if (
            existing.exists()
            and existing.stat().st_size > 4
            and signature(existing) != "unknown"
        ):
            return existing, False

    partial.parent.mkdir(parents=True, exist_ok=True)
    key = f"{exchange}/{hour:%Y-%m-%d}/{hour:%H}/{symbol}_{kind}.parquet"
    parameters = {"file": key}
    if api_key:
        parameters["api_key"] = api_key
    url = API + "?" + urlencode(parameters)
    label = f"{hour:%Y-%m-%d %H}:00 {kind}"

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {
            "User-Agent": "fill-probability-replication/1.0",
            "Accept-Encoding": "identity",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(20, 300),
            ) as response:
                if response.status_code == 416 and partial.exists():
                    partial.unlink()
                    raise RuntimeError("Invalid resume position")
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if not append:
                    offset = 0
                remaining = int(response.headers.get("Content-Length", "0") or 0)
                total = offset + remaining if remaining else None
                with partial.open("ab" if append else "wb") as output, tqdm(
                    total=total,
                    initial=offset,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=label,
                    leave=False,
                    dynamic_ncols=True,
                ) as progress:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
                            progress.update(len(chunk))
            file_type = signature(partial)
            if file_type == "zstd":
                destination = compressed
            elif file_type == "parquet":
                destination = parquet
            else:
                raise RuntimeError("Downloaded file is neither Zstandard nor Parquet")
            os.replace(partial, destination)
            return destination, True
        except (requests.RequestException, OSError, RuntimeError) as error:
            if attempt == retries:
                raise
            delay = min(60, 2**attempt)
            tqdm.write(f"{label}: {error}; retrying in {delay}s")
            time.sleep(delay)
    raise AssertionError("Unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hourly order-book and trade files from CryptoHFTData"
    )
    parser.add_argument("--start", required=True, help="First UTC date, YYYY-MM-DD")
    parser.add_argument(
        "--end-exclusive",
        required=True,
        help="Exclusive UTC end date, YYYY-MM-DD",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/cryptohftdata/binance_spot/BTCUSDT"),
    )
    parser.add_argument("--exchange", default="binance_spot")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["orderbook", "trades"],
        default=["orderbook", "trades"],
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CRYPTOHFTDATA_API_KEY"),
        help="Optional; CRYPTOHFTDATA_API_KEY is used when omitted",
    )
    parser.add_argument("--min-request-interval", type=float, default=1.05)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    start = parse_day(args.start)
    end = parse_day(args.end_exclusive)
    if end <= start:
        raise ValueError("--end-exclusive must be later than --start")

    tasks = [
        (start + timedelta(hours=offset), kind)
        for offset in range(int((end - start).total_seconds() // 3600))
        for kind in args.kinds
    ]
    manifest_path = args.output / "download_manifest.json"
    manifest = {
        "source": "CryptoHFTData",
        "exchange": args.exchange,
        "symbol": args.symbol,
        "start_utc": start.isoformat(),
        "end_exclusive_utc": end.isoformat(),
        "completed": {},
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed"] = previous.get("completed", {})

    failures: list[tuple[str, str]] = []
    overall = tqdm(tasks, desc="All files", unit="file", dynamic_ncols=True)
    for hour, kind in overall:
        request_started = time.monotonic()
        label = f"{hour:%Y-%m-%d %H}:00 {kind}"
        overall.set_postfix_str(label)
        made_request = False
        try:
            path, made_request = download_one(
                args.output,
                hour,
                args.exchange,
                args.symbol,
                kind,
                args.api_key,
                args.retries,
            )
            manifest["completed"][f"{hour:%Y-%m-%dT%H}:00Z/{kind}"] = {
                "path": str(path.relative_to(args.output)),
                "bytes": path.stat().st_size,
                "format": signature(path),
            }
            write_json_atomic(manifest_path, manifest)
        except Exception as error:
            failures.append((label, str(error)))
            tqdm.write(f"FAILED {label}: {error}")
        if made_request:
            elapsed = time.monotonic() - request_started
            time.sleep(max(0.0, args.min_request_interval - elapsed))

    missing: list[str] = []
    malformed: list[str] = []
    total_bytes = 0
    for hour, kind in tasks:
        parquet, compressed, _ = file_locations(
            args.output,
            hour,
            args.symbol,
            kind,
        )
        path = compressed if compressed.exists() else parquet
        if not path.exists():
            missing.append(f"{hour:%Y-%m-%d %H}:00 {kind}")
            continue
        total_bytes += path.stat().st_size
        if signature(path) == "unknown":
            malformed.append(str(path))

    print(
        f"expected={len(tasks)}, missing={len(missing)}, "
        f"malformed={len(malformed)}"
    )
    print(f"compressed_size_gib={total_bytes / 1024**3:.3f}")
    if failures or missing or malformed:
        raise RuntimeError(
            "Incomplete download: "
            f"failures={len(failures)}, missing={len(missing)}, "
            f"malformed={len(malformed)}"
        )
    final_day = end - timedelta(days=1)
    print(f"Verified complete: {start:%Y-%m-%d} through {final_day:%Y-%m-%d}")


if __name__ == "__main__":
    main()
