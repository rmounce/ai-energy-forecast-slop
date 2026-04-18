"""
Tests for Tier 2 TFT price forecast output shape and decoder invariants.
Uses frozen fixtures to avoid InfluxDB dependency.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
TFT_FIXTURE = FIXTURES / "tft_price_output.json"

EXPECTED_STEPS = 144   # 72h at 30-min resolution
EXPECTED_QUANTILES = 3  # q30, q50, q70


@pytest.fixture()
def tft_output():
    if not TFT_FIXTURE.exists():
        pytest.skip("TFT price fixture not captured yet — run tests/fixtures/capture_fixtures.py")
    with open(TFT_FIXTURE) as f:
        return json.load(f)


def test_tft_has_required_keys(tft_output):
    """TFT output must have tft_price, tft_price_q30, tft_price_q70."""
    assert "tft_price" in tft_output
    assert "tft_price_q30" in tft_output
    assert "tft_price_q70" in tft_output


def test_tft_shape_144_steps(tft_output):
    """TFT price forecast must have 144 steps (30-min × 72h)."""
    for key in ("tft_price", "tft_price_q30", "tft_price_q70"):
        data = tft_output[key]
        assert len(data) == EXPECTED_STEPS, f"{key}: expected {EXPECTED_STEPS} steps, got {len(data)}"


def test_tft_no_nan(tft_output):
    """No NaN in TFT wholesale_price outputs."""
    for key in ("tft_price", "tft_price_q30", "tft_price_q70"):
        values = [row["wholesale_price"] for row in tft_output[key]]
        assert all(v == v for v in values), f"NaN found in {key}"


def test_tft_quantile_ordering(tft_output):
    """q30 ≤ q50 ≤ q70 at every step (model sorts quantiles post-inference)."""
    q30 = [row["wholesale_price"] for row in tft_output["tft_price_q30"]]
    q50 = [row["wholesale_price"] for row in tft_output["tft_price"]]
    q70 = [row["wholesale_price"] for row in tft_output["tft_price_q70"]]
    for i, (lo, mid, hi) in enumerate(zip(q30, q50, q70)):
        assert lo <= mid + 1e-9, f"Step {i}: q30={lo:.6f} > q50={mid:.6f}"
        assert mid <= hi + 1e-9, f"Step {i}: q50={mid:.6f} > q70={hi:.6f}"


def test_tft_prices_in_reasonable_range(tft_output):
    """Wholesale prices in $/kWh should be in plausible range (−0.5 to 5.0)."""
    for key in ("tft_price", "tft_price_q30", "tft_price_q70"):
        values = [row["wholesale_price"] for row in tft_output[key]]
        for i, v in enumerate(values):
            assert -0.5 <= v <= 5.0, f"{key}[{i}]: implausible price {v:.4f} $/kWh"


def test_tft_decoder_feature_count_from_source():
    """Verify decoder has exactly 15 input features (static check from source)."""
    import ast

    lines = (ROOT / "forecast.py").read_text().splitlines()
    dec_cont = None
    collecting, collected = False, []
    for line in lines:
        if line.strip().startswith("DEC_CONT = ["):
            collecting = True
        if collecting:
            collected.append(line.strip())
            try:
                dec_cont = ast.literal_eval(" ".join(collected)[len("DEC_CONT = "):])
                break
            except (SyntaxError, ValueError):
                continue
    assert dec_cont is not None, "DEC_CONT not found in forecast.py"

    time_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    extra_cols = ["horizon_norm", "covar_missing"]
    total = len(dec_cont) + len(time_cols) + len(extra_cols)
    assert total == 15, (
        f"Decoder has {total} features (DEC_CONT={len(dec_cont)}, "
        f"TIME={len(time_cols)}, EXTRA={len(extra_cols)}); expected 15"
    )
