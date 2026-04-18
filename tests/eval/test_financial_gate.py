"""
Phase 8 Layer 2 — Financial eval gate.

Asserts that the AI pipeline (Tier 1+2) meets or exceeds the legacy LightGBM
baseline established in the Phase 6 holistic dispatch simulation.

Thresholds derived from Phase 6 full run (811 windows, July 2025–March 2026,
price-only LP MPC):
  lgbm_legacy: spike $6.82/day, low $0.89/day, normal $0.52/day, all $2.99/day

Gate: AI pipeline must not regress below legacy LightGBM by more than:
  overall: 0% (must match or beat)
  spike:   5% (spikes have highest variance, allow small miss)
  low:     2%
  normal:  2%

Requires InfluxDB access. Run with: pytest tests/eval/ -v
To refresh baseline: run eval/holistic_eval.py, then update LGBM_BASELINE below.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = ROOT / "eval" / "results" / "holistic_eval_results.csv"


# ── Phase 6 baseline (July 2025–March 2026, price-only LP MPC, 811 windows) ──
LGBM_BASELINE = {
    "spike":  6.8201,   # $/day
    "low":    0.8912,
    "normal": 0.5196,
    "all":    2.9877,
}

# Gate: AI pipeline must achieve ≥ baseline × (1 + tolerance)
TOLERANCE = {
    "all":    0.00,   # must not be worse than baseline overall
    "spike":  -0.05,  # at most 5% worse (high variance)
    "low":    -0.02,
    "normal": -0.02,
}

THRESHOLDS = {s: LGBM_BASELINE[s] * (1 + TOLERANCE[s]) for s in LGBM_BASELINE}


@pytest.mark.skipif(
    not RESULTS_FILE.exists(),
    reason="Phase 6 baseline results not generated yet — run eval/holistic_eval.py first",
)
def test_financial_gate_baseline_sanity():
    """Sanity check: baseline results file has expected columns and lgbm rows."""
    df = pd.read_csv(RESULTS_FILE)
    for col in ("source", "stratum", "mean_per_day", "vs_lgbm_pct"):
        assert col in df.columns, f"Missing column: {col}"
    lgbm_rows = df[df["source"] == "lgbm_legacy"]
    assert len(lgbm_rows) >= 3, "Expected lgbm_legacy rows for each stratum"


@pytest.mark.skipif(
    not RESULTS_FILE.exists(),
    reason="Phase 6 baseline results not generated yet",
)
def test_lgbm_baseline_matches_known_values():
    """
    Verify the stored Phase 6 baseline matches LGBM_BASELINE constants in this file.
    If holistic_eval.py is re-run, update LGBM_BASELINE in this test file.
    """
    df = pd.read_csv(RESULTS_FILE)
    lgbm = df[df["source"] == "lgbm_legacy"].set_index("stratum")["mean_per_day"]
    for stratum, expected in LGBM_BASELINE.items():
        actual = lgbm.get(stratum)
        if actual is None:
            continue
        assert abs(actual - expected) < 0.01, (
            f"lgbm_legacy {stratum}: stored={actual:.4f}, expected={expected:.4f}. "
            "Re-run holistic_eval.py changed baseline — update LGBM_BASELINE."
        )


@pytest.mark.skip(
    reason="AI pipeline source not yet in holistic_eval — enable after Tier 1+2 TFT added"
)
def test_ai_pipeline_meets_financial_gate():
    """
    AI pipeline (Tier 1+2) $/day must meet thresholds vs legacy LightGBM baseline.

    Enable after holistic_eval.py is extended with a 'tier1_tier2_ai' source.
    """
    df = pd.read_csv(RESULTS_FILE)
    ai_rows = df[df["source"] == "tier1_tier2_ai"]
    if ai_rows.empty:
        pytest.skip("tier1_tier2_ai source not yet in holistic_eval results")

    for stratum, threshold in THRESHOLDS.items():
        row = ai_rows[ai_rows["stratum"] == stratum]
        if row.empty:
            continue
        mean_val = row["mean_per_day"].iloc[0]
        assert mean_val >= threshold, (
            f"AI pipeline {stratum}: ${mean_val:.4f}/day < threshold ${threshold:.4f}/day"
        )
