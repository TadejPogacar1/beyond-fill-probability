from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published"


def test_published_summary_has_all_twelve_configurations():
    summary = pd.read_csv(PUBLISHED / "model_summary.csv")
    assert len(summary) == 12
    assert set(summary["horizon_seconds"]) == {60, 300, 600}
    assert set(summary["objective"]) == {
        "fill_only",
        "two_stage",
        "unified_short",
        "unified_long",
    }


def test_only_positive_configuration_is_t600_unified_long():
    summary = pd.read_csv(PUBLISHED / "model_summary.csv")
    positive = summary.loc[summary["total_pnl_usd"] > 0]
    assert len(positive) == 1
    row = positive.iloc[0]
    assert int(row["horizon_seconds"]) == 600
    assert row["objective"] == "unified_long"
    assert row["total_pnl_usd"] == pytest.approx(43.873092, abs=1e-6)
    assert int(row["winning_days"]) == 6


def test_best_strategy_daily_rows_reconcile_to_summary():
    daily = pd.read_csv(PUBLISHED / "best_strategy_daily.csv")
    summary = pd.read_csv(PUBLISHED / "model_summary.csv")
    best = summary.query("horizon_seconds == 600 and objective == 'unified_long'").iloc[0]
    assert len(daily) == 7
    assert daily["pnl_usd"].sum() == pytest.approx(best["total_pnl_usd"], abs=1e-5)
    assert int(daily["fills"].sum()) == 212
    assert int((daily["pnl_usd"] > 0).sum()) == 6


def test_model_artifact_manifest_matches_files():
    manifest_path = ROOT / "artifacts" / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scikit_learn_version"] == "1.9.0"
    assert len(manifest["models"]) == 12
    for relative_path, metadata in manifest["models"].items():
        artifact = ROOT / relative_path
        assert artifact.stat().st_size == metadata["bytes"]
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        assert digest == metadata["sha256"]
