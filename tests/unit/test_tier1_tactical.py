"""
Tests for Tier 1 LightGBM tactical prediction output shape and invariants.
Uses frozen fixtures to avoid InfluxDB dependency.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
TACTICAL_FIXTURE = FIXTURES / "tactical_output.json"


@pytest.fixture()
def tactical_output():
    if not TACTICAL_FIXTURE.exists():
        pytest.skip("Tactical fixture not captured yet — run tests/fixtures/capture_fixtures.py")
    with open(TACTICAL_FIXTURE) as f:
        return json.load(f)


def test_tier1_has_required_keys(tactical_output):
    """Tier 1 output must have p5min_price, q05, q95."""
    assert "p5min_price" in tactical_output
    assert "p5min_price_q05" in tactical_output
    assert "p5min_price_q95" in tactical_output


def test_tier1_shape_12_steps(tactical_output):
    """Tier 1 forecast must have exactly 12 steps (5-min × 60 min)."""
    for key in ("p5min_price", "p5min_price_q05", "p5min_price_q95"):
        data = tactical_output[key]
        assert len(data) == 12, f"{key}: expected 12 steps, got {len(data)}"


def test_tier1_no_nan(tactical_output):
    """No NaN in Tier 1 wholesale_price column."""
    for key in ("p5min_price", "p5min_price_q05", "p5min_price_q95"):
        values = [row["wholesale_price"] for row in tactical_output[key]]
        assert all(v == v for v in values), f"NaN found in {key}"


def test_tier1_quantile_ordering(tactical_output):
    """q05 ≤ q50 ≤ q95 at every step."""
    q05 = [row["wholesale_price"] for row in tactical_output["p5min_price_q05"]]
    q50 = [row["wholesale_price"] for row in tactical_output["p5min_price"]]
    q95 = [row["wholesale_price"] for row in tactical_output["p5min_price_q95"]]
    for i, (lo, mid, hi) in enumerate(zip(q05, q50, q95)):
        assert lo <= mid + 1e-9, f"Step {i}: q05={lo:.6f} > q50={mid:.6f}"
        assert mid <= hi + 1e-9, f"Step {i}: q50={mid:.6f} > q95={hi:.6f}"


def test_tier1_prices_in_reasonable_range(tactical_output):
    """Wholesale prices in $/kWh should be in plausible range (−0.5 to 5.0)."""
    for key in ("p5min_price", "p5min_price_q05", "p5min_price_q95"):
        values = [row["wholesale_price"] for row in tactical_output[key]]
        for v in values:
            assert -0.5 <= v <= 5.0, f"{key}: implausible price {v:.4f} $/kWh"
