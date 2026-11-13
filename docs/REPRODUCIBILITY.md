# Reproducibility

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

No GPU is required.

## Download raw data

```bash
python scripts/download_data.py \
  --start 2026-08-01 \
  --end-exclusive 2026-09-20 \
  --output data/raw/cryptohftdata/binance_spot/BTCUSDT
```

## Reconstruct one-second L2 states

```bash
python scripts/build_market_state.py \
  --raw-root data/raw/cryptohftdata/binance_spot/BTCUSDT \
  --output-dir data/processed/market_state_1s \
  --start-hour 2026-08-01T23 \
  --end-hour-exclusive 2026-09-20T00
```

## Build samples

```bash
DATES=$(cat configs/train_dates.txt configs/test_dates.txt)
for T in 60 300 600; do
  python scripts/build_labeled_samples.py \
    --raw-root data/raw/cryptohftdata/binance_spot/BTCUSDT \
    --state-dir data/processed/market_state_1s \
    --sample-dir "data/processed/samples/T${T}" \
    --horizon-seconds "$T" \
    --audit-json configs/gap_intervals.json \
    --depths-bps 0.1 0.25 0.5 1 2 3 5 7 10 15 20 30 \
    --dates $DATES
done
```

## Audit labels

```bash
cat configs/train_dates.txt configs/test_dates.txt > /tmp/holdout_dates.txt
python scripts/audit_data.py \
  --dates-file /tmp/holdout_dates.txt \
  --state-dir data/processed/market_state_1s \
  --sample-root data/processed/samples \
  --output-dir results/generated/audit
```

## Train full-data models

Frozen models are included. To retrain all 12 artifacts:

```bash
python scripts/train_full_models.py \
  --sample-root data/processed/samples \
  --cache-root artifacts/training_cache \
  --model-root artifacts/models \
  --train-dates-file configs/train_dates.txt
```

The memory-mapped cache requires roughly 20 GB per horizon.

## Evaluate the holdout

```bash
python scripts/evaluate_frozen_models.py \
  --sample-root data/processed/samples \
  --state-dir data/processed/market_state_1s \
  --model-root artifacts/models \
  --result-root results/generated/holdout \
  --train-dates-file configs/train_dates.txt \
  --test-dates-file configs/test_dates.txt \
  --complete-dates-file configs/complete_test_dates.txt \
  --coverage-csv results/published/data_coverage.csv
```

Daily outputs are atomic and resumable.

## Validate the repository

```bash
python scripts/validate_repository.py
ruff check src scripts tests
pytest
```
