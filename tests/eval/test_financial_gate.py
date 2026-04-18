"""
Phase 8 Layer 2 — Financial eval gate.

Runs holistic_eval.py on the stratified set and asserts that the AI pipeline
meets or exceeds the legacy LightGBM baseline.

Thresholds are set from the Phase 6 baseline run (holistic_eval_results.csv).
Update THRESHOLDS below after Phase 6 completes.

Requires InfluxDB access (real data, not mocked).
Run with: pytest tests/eval/ -v
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = ROOT / "eval" / "results" / "holistic_eval_results.csv"
INDEX_FILE   = ROOT / "eval" / "results" / "holistic_eval_index.parquet"


# ── Thresholds (set from Phase 6 baseline run) ────────────────────────────────
# TODO: Fill these after holistic_eval.py completes.
# These are placeholders. See eval/results/holistic_eval_results.csv for baseline.
THRESHOLDS = {
    # AI pipeline (Tier 1+2) must meet or exceed:
    "all":    {"min_vs_lgbm_pct": 0.0},    # not worse than baseline overall
    "spike":  {"min_vs_lgbm_pct": -5.0},   # at most 5% worse on spike events
    "low":    {"min_vs_lgbm_pct": -2.0},   # at most 2% worse on low-price events
    "normal": {"min_vs_lgbm_pct": -2.0},   # at most 2% worse in normal conditions
}


@pytest.mark.skipif(
    not RESULTS_FILE.exists(),
    reason="Phase 6 baseline results not generated yet — run eval/holistic_eval.py first",
)
def test_financial_gate_lgbm_baseline_loaded():
    """Sanity check: baseline results file exists and has expected columns."""
    df = pd.read_csv(RESULTS_FILE)
    assert "source" in df.columns
    assert "stratum" in df.columns
    assert "mean_per_day" in df.columns
    assert "vs_lgbm_pct" in df.columns
    # Baseline (lgbm_legacy) must be present in each stratum
    lgbm_rows = df[df["source"] == "lgbm_legacy"]
    assert len(lgbm_rows) >= 3, "Expected lgbm_legacy rows for each stratum"


@pytest.mark.skip(
    reason="Phase 6 thresholds not yet set — fill THRESHOLDS dict after holistic_eval completes"
)
def test_ai_pipeline_meets_financial_gate():
    """
    AI pipeline (Tier 1+2) $/day must meet thresholds vs legacy LightGBM baseline.

    This test will be enabled after:
    1. holistic_eval.py runs with TFT AI source added
    2. THRESHOLDS dict is filled from Phase 6 results
    3. Phase 8 is declared complete
    """
    df = pd.read_csv(RESULTS_FILE)
    ai_rows = df[df["source"] == "tier1_tier2_ai"]
    if ai_rows.empty:
        pytest.skip("Tier 1+2 AI source not yet in holistic_eval results")

    for stratum, threshold in THRESHOLDS.items():
        row = ai_rows[ai_rows["stratum"] == stratum]
        if row.empty:
            continue
        vs_pct = row["vs_lgbm_pct"].iloc[0]
        assert vs_pct >= threshold["min_vs_lgbm_pct"], (
            f"AI pipeline {stratum} stratum: {vs_pct:.1f}% vs lgbm_legacy, "
            f"threshold is {threshold['min_vs_lgbm_pct']:.1f}%"
        )
