"""
Phase 8 Layer 2 — Financial eval gate.

Baseline: Amber APF + LGBM extrapolation (as-run production model, July 2025–March 2026).
  'amber_apf_lgbm' in results: Amber APF seeds first ~14-28h, LightGBM extrapolates to 72h.
  This is NOT a pure LightGBM model — Amber's commercial forecast drives the short horizon.

Gate: tier1_tier2_hybrid (Tier 1 LGBM q50 for 0–60min, TFT q50 for 1h–72h) must not
regress below Amber APF + LGBM by more than:
  overall: 0% (must match or beat)
  spike:   5% (high variance — allow small miss)
  low:     2%
  normal:  2%

tier1_tier2_hybrid results (811 windows, price-only LP MPC, July 2025–March 2026):
  overall: +5.5%  ✅  ($3.15/day vs $2.99 baseline)
  spike:   +5.8%  ✅  ($7.22/day vs $6.82 baseline)
  low:     +17.1% ✅  ($1.04/day vs $0.89 baseline)
  normal:  -27.8% ❌  ($0.38/day vs $0.52 baseline)
    Root cause: TFT q50 overestimates prices in flat-price regimes (142/144 steps
    from TFT); spike-heavy training + log-scaling bias. Tier 1 only covers 2 steps.
    Blocks Phase 5 remainder until resolved.

For reference: tft_tier2_q50 standalone (archived in _ai parquet checkpoints):
  overall: +6.6%, spike: +5.8%, low: +23.6%, normal: -21.1%

Requires InfluxDB access. Run with: pytest tests/eval/ -v
To refresh: nice -n 19 python eval/holistic_eval.py --hybrid-source --price-only --workers 12
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_FILE = ROOT / "eval" / "results" / "holistic_eval_results.csv"


# ── Phase 6 baseline (Amber APF + LGBM, July 2025–March 2026, price-only LP MPC) ──
AMBER_APF_BASELINE = {
    "spike":  6.8201,   # $/day
    "low":    0.8912,
    "normal": 0.5196,
    "all":    2.9877,
}

# Gate tolerances
TOLERANCE = {
    "all":    0.00,
    "spike":  -0.05,
    "low":    -0.02,
    "normal": -0.02,
}

THRESHOLDS = {s: AMBER_APF_BASELINE[s] * (1 + TOLERANCE[s]) for s in AMBER_APF_BASELINE}


@pytest.mark.skipif(
    not RESULTS_FILE.exists(),
    reason="Phase 6 results not generated yet — run eval/holistic_eval.py first",
)
def test_financial_gate_baseline_sanity():
    """Sanity check: results file has expected columns and amber_apf_lgbm rows."""
    df = pd.read_csv(RESULTS_FILE)
    for col in ("source", "stratum", "mean_per_day", "vs_amber_apf_pct"):
        assert col in df.columns, f"Missing column: {col}"
    amber_rows = df[df["source"] == "amber_apf_lgbm"]
    assert len(amber_rows) >= 3, "Expected amber_apf_lgbm rows for each stratum"


@pytest.mark.skipif(
    not RESULTS_FILE.exists(),
    reason="Phase 6 results not generated yet",
)
def test_amber_apf_baseline_matches_known_values():
    """
    Verify stored Amber APF + LGBM baseline matches AMBER_APF_BASELINE constants.
    If holistic_eval.py is re-run, update AMBER_APF_BASELINE in this file.
    """
    df = pd.read_csv(RESULTS_FILE)
    amber = df[df["source"] == "amber_apf_lgbm"].set_index("stratum")["mean_per_day"]
    for stratum, expected in AMBER_APF_BASELINE.items():
        actual = amber.get(stratum)
        if actual is None:
            continue
        assert abs(actual - expected) < 0.01, (
            f"amber_apf_lgbm {stratum}: stored={actual:.4f}, expected={expected:.4f}. "
            "Re-run holistic_eval.py changed baseline — update AMBER_APF_BASELINE."
        )


def test_ai_pipeline_meets_financial_gate():
    """
    tier1_tier2_hybrid $/day must meet thresholds vs Amber APF + LGBM baseline.
    Hybrid: Tier 1 LGBM q50 for steps 0-1 (0-60 min), TFT q50 for steps 2-143 (1h-72h).

    To regenerate results:
        nice -n 19 python eval/holistic_eval.py --hybrid-source --price-only --workers 12

    Current status (811 windows, July 2025–March 2026):
      overall: $3.15/day (+5.5%)  ✅
      spike:   $7.22/day (+5.8%)  ✅
      low:     $1.04/day (+17.1%) ✅
      normal:  $0.38/day (-27.8%) ❌  FAILS — TFT q50 ~2× actual in flat-price windows;
                                       Tier 1 only covers 2/144 steps, TFT bias dominates.
    """
    if not RESULTS_FILE.exists():
        pytest.skip("Results file not found — run holistic_eval.py --hybrid-source first")
    df = pd.read_csv(RESULTS_FILE)
    ai_rows = df[df["source"] == "tier1_tier2_hybrid"]
    if ai_rows.empty:
        pytest.skip("tier1_tier2_hybrid not in results — run with --hybrid-source")

    for stratum, threshold in THRESHOLDS.items():
        row = ai_rows[ai_rows["stratum"] == stratum]
        if row.empty:
            continue
        mean_val = row["mean_per_day"].iloc[0]
        assert mean_val >= threshold, (
            f"tier1_tier2_hybrid {stratum}: ${mean_val:.4f}/day < threshold ${threshold:.4f}/day"
        )
