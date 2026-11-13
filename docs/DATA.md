# Data

## Source

The experiment uses hourly Binance Spot BTCUSDT files from
[CryptoHFTData](https://www.cryptohftdata.com/docs):

- full L2 order-book snapshots and incremental updates;
- tick-by-tick trades with price, quantity, timestamp, and aggressor side.

Raw market data are not redistributed. Users must comply with the provider's and
exchange's terms.

## Study dates

| Component | Dates |
|---|---|
| Book bootstrap | 2026-08-01 23:03 UTC onward |
| Full-data training | 2026-08-02 through 2026-09-12, 42 days |
| Frozen holdout | 2026-09-13 through 2026-09-19, 7 days |

Every holdout date contains 86,400 one-second market states. The exact date lists
are version controlled in `configs/`.

## Download

```bash
python scripts/download_data.py \
  --start 2026-08-01 \
  --end-exclusive 2026-09-20 \
  --output data/raw/cryptohftdata/binance_spot/BTCUSDT
```

The downloader resumes partial files, verifies file signatures, records a
manifest, retries failures, and checks the final file inventory.

## Local layout

```text
data/
  raw/cryptohftdata/binance_spot/BTCUSDT/
    YYYY-MM-DD/HH/BTCUSDT_orderbook.parquet.zst
    YYYY-MM-DD/HH/BTCUSDT_trades.parquet.zst
  processed/
    market_state_1s/YYYY-MM-DD.parquet
    samples/T60/YYYY-MM-DD.parquet
    samples/T300/YYYY-MM-DD.parquet
    samples/T600/YYYY-MM-DD.parquet
```

## Integrity controls

The pipeline checks:

- hourly order-book and trade file presence;
- continuous one-second reconstructed states;
- fill times strictly after placement;
- fill times no later than the horizon;
- exact agreement between the fill flag and fill timestamp;
- all 12 quote depths in every sample file.

Audited raw-gap timestamps are stored in `configs/gap_intervals.json`. Local paths
and machine-specific metadata were removed.
