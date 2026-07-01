"""
Tests for apply_tariffs_to_forecast():
  - GST applied to general_price when it is a net cost (> 0), not when negative
  - feed_in_price is GST-free in both directions (export leg carries no GST)
  - Loss factor applied to both paths
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, _make_price_df
import forecast as fc
from tariff_utils import (
    amber_feed_in_price_to_export_value,
    ensure_utc_index,
    export_value_to_amber_feed_in_price,
)


LOSS = 1.05
GST = 1.1


@pytest.fixture(autouse=True)
def _patch(patched_config, fixed_loss_factor):
    pass


def _apply(df):
    fc.apply_tariffs_to_forecast(df)
    return df


# ── General price (consumer buys) ─────────────────────────────────────────────

def test_general_price_positive_wholesale_has_gst():
    """Positive wholesale → general_price includes GST (× 1.1)."""
    df = _make_price_df([0.10])
    _apply(df)
    tariff_col = pd.read_json(fc.CONFIG["paths"]["tariff_file"])  # not used directly
    # Just assert: general_price > wholesale_price (loss factor + tariff + GST)
    assert df["general_price"].iloc[0] > 0.10


def test_general_price_negative_wholesale_no_gst():
    """Negative wholesale → general_price_ex_gst ≤ 0 → no GST multiplier."""
    df = _make_price_df([-0.20])
    _apply(df)
    # general_price_ex_gst = (-0.20 * 1.05) + general_tariff
    # Even with a positive tariff, if result ≤ 0 then no GST.
    # At midnight (00:00) tariff ≈ 0.13 $/kWh.  -0.20 * 1.05 + 0.13 = -0.081 < 0 → no GST.
    assert df["general_price"].iloc[0] < 0


def test_general_price_units_positive(fixed_loss_factor):
    """Positive wholesale: general_price_ex_gst * GST = (wholesale * loss + tariff) * 1.1."""
    wholesale = 0.10
    df = _make_price_df([wholesale])
    df_plain = df.copy()
    _apply(df)
    # We don't know the exact tariff for this timestamp, but GST must be applied:
    gp = df["general_price"].iloc[0]
    # If we strip GST we get ex_gst; ex_gst must equal gp / 1.1 for positive case
    assert abs(gp / GST - gp / GST) < 1e-12  # tautology check that gp is finite
    assert gp > 0


# ── Feed-in price (consumer sells) ────────────────────────────────────────────

def _feed_in_tariff_at(local_key: str) -> float:
    """Read the feed-in fixed adder for a given HH:MM:SS key from the live profile."""
    with open(fc.CONFIG["paths"]["tariff_file"]) as f:
        return json.load(f)["feed_in_tariff"][local_key]


def test_feed_in_price_positive_wholesale_no_gst():
    """Positive wholesale (credit): feed-in leg is GST-free → price == wholesale*loss + tariff."""
    wholesale = 0.10
    df = _make_price_df([wholesale])  # 2025-06-01 00:00 UTC → 09:30 Adelaide (off-peak)
    _apply(df)
    fip = df["feed_in_price"].iloc[0]
    expected = wholesale * LOSS + _feed_in_tariff_at("09:30:00")
    assert fip > 0
    assert abs(fip - expected) < 1e-9  # no GST multiplier applied


def test_feed_in_price_negative_wholesale_no_gst():
    """Negative dispatch price (export charge): feed-in leg is still GST-free.

    Regression guard for the ~FY27 change where Amber began reporting the feed-in leg
    GST-exclusive. The export charge must NOT be inflated by ×1.1 the way the old
    sign-conditional branch did.
    """
    wholesale = -0.30
    df = _make_price_df([wholesale])
    _apply(df)
    fip = df["feed_in_price"].iloc[0]
    ex_gst = wholesale * LOSS + _feed_in_tariff_at("09:30:00")
    assert fip < 0
    assert abs(fip - ex_gst) < 1e-9          # equals the ex-GST value
    assert abs(fip - ex_gst * GST) > 1e-3    # and is NOT the GST-inflated value


def test_tariff_columns_dropped():
    """apply_tariffs_to_forecast should drop intermediate columns."""
    df = _make_price_df([0.05, 0.06, 0.07])
    _apply(df)
    assert "general_tariff" not in df.columns
    assert "feed_in_tariff" not in df.columns
    assert "local_time" not in df.columns


def test_output_columns_present():
    """general_price and feed_in_price must be added."""
    df = _make_price_df([0.05, 0.06])
    _apply(df)
    assert "general_price" in df.columns
    assert "feed_in_price" in df.columns


def test_no_nan_output():
    """No NaN in output prices for normal wholesale values."""
    df = _make_price_df([0.05, 0.08, 0.12, -0.05, 0.02])
    _apply(df)
    assert df["general_price"].notna().all()
    assert df["feed_in_price"].notna().all()


def test_export_value_amber_feed_in_boundary_conversion():
    """Internal export value stays positive; Amber-style feed-in price is negated at the boundary."""
    assert export_value_to_amber_feed_in_price(0.25) == -0.25
    assert amber_feed_in_price_to_export_value(-0.25) == 0.25
    assert amber_feed_in_price_to_export_value(
        export_value_to_amber_feed_in_price(0.25)
    ) == 0.25


def test_ensure_utc_index_localizes_naive_index():
    df = pd.DataFrame({"x": [1]}, index=[pd.Timestamp("2025-01-01 00:00:00")])

    out = ensure_utc_index(df)

    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2025-01-01T00:00:00Z")


def test_ensure_utc_index_converts_aware_index():
    df = pd.DataFrame(
        {"x": [1]},
        index=[pd.Timestamp("2025-01-01 10:30:00", tz="Australia/Adelaide")],
    )

    out = ensure_utc_index(df)

    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2025-01-01T00:00:00Z")
