"""
Tests for _build_combined_forecast_items() and _publish_combined_price_forecasts():
  - Tier 1 items have 5-min intervals
  - Tier 2 items have 30-min intervals
  - Tier 2 can be expanded to six 5-min items per 30-min source step
  - feed-in sign convention preserved
  - total items in combined output
"""

from datetime import timedelta

import pandas as pd
import pytest

import forecast as fc
from conftest import _make_price_df


@pytest.fixture(autouse=True)
def _patch(patched_config, fixed_loss_factor):
    pass


# ── Interval length ───────────────────────────────────────────────────────────

def test_tier1_item_interval_is_5min(tier1_dfs):
    gen_items, _ = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    for item in gen_items:
        start = pd.Timestamp(item["start_time"])
        end = pd.Timestamp(item["end_time"])
        assert (end - start) == pd.Timedelta(minutes=5), (
            f"Tier 1 interval should be 5 min, got {end - start}"
        )


def test_tier2_item_interval_is_30min(tier2_dfs):
    gen_items, _ = fc._build_combined_forecast_items(
        tier2_dfs["p50"], tier2_dfs["low"], tier2_dfs["high"], 30
    )
    for item in gen_items:
        start = pd.Timestamp(item["start_time"])
        end = pd.Timestamp(item["end_time"])
        assert (end - start) == pd.Timedelta(minutes=30), (
            f"Tier 2 interval should be 30 min, got {end - start}"
        )


def test_tier2_items_can_publish_as_5min(tier2_dfs):
    gen_items, _ = fc._build_combined_forecast_items(
        tier2_dfs["p50"], tier2_dfs["low"], tier2_dfs["high"], 30, 5
    )
    assert len(gen_items) == 144 * 6

    first = gen_items[0]
    seventh = gen_items[6]
    first_start = pd.Timestamp(first["start_time"])
    first_end = pd.Timestamp(first["end_time"])
    seventh_start = pd.Timestamp(seventh["start_time"])

    assert (first_end - first_start) == pd.Timedelta(minutes=5)
    assert seventh_start == first_start + pd.Timedelta(minutes=30)
    for offset in range(6):
        item = gen_items[offset]
        assert item["per_kwh"] == first["per_kwh"]
        assert pd.Timestamp(item["start_time"]) == first_start + pd.Timedelta(minutes=5 * offset)


# ── Item counts ───────────────────────────────────────────────────────────────

def test_tier1_item_count(tier1_dfs):
    gen_items, fin_items = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    assert len(gen_items) == 12
    assert len(fin_items) == 12


def test_tier2_item_count(tier2_dfs):
    gen_items, _ = fc._build_combined_forecast_items(
        tier2_dfs["p50"], tier2_dfs["low"], tier2_dfs["high"], 30
    )
    assert len(gen_items) == 144


# ── Feed-in sign in combined output ──────────────────────────────────────────

def test_combined_feed_in_sign_positive_wholesale(tier1_dfs):
    """All Tier 1 feed-in items have per_kwh ≤ 0 when wholesale is positive."""
    _, fin_items = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    for item in fin_items:
        assert item["per_kwh"] <= 0


# ── Required fields ───────────────────────────────────────────────────────────

def test_general_item_required_fields(tier1_dfs):
    gen_items, _ = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    required = {"start_time", "end_time", "advanced_price_predicted",
                "advanced_price_high", "advanced_price_low", "per_kwh"}
    for item in gen_items:
        assert required.issubset(item.keys())


def test_feed_in_item_required_fields(tier1_dfs):
    _, fin_items = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    required = {"start_time", "end_time", "advanced_price_predicted",
                "advanced_price_high", "advanced_price_low", "per_kwh"}
    for item in fin_items:
        assert required.issubset(item.keys())


# ── No NaN in output ──────────────────────────────────────────────────────────

def test_no_nan_in_general_items(tier1_dfs):
    gen_items, _ = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    for item in gen_items:
        for k in ("per_kwh", "advanced_price_predicted", "advanced_price_high", "advanced_price_low"):
            assert item[k] == item[k], f"NaN in general item field {k}"


def test_no_nan_in_feed_in_items(tier1_dfs):
    _, fin_items = fc._build_combined_forecast_items(
        tier1_dfs["p50"], tier1_dfs["low"], tier1_dfs["high"], 5
    )
    for item in fin_items:
        for k in ("per_kwh", "advanced_price_predicted", "advanced_price_high", "advanced_price_low"):
            assert item[k] == item[k], f"NaN in feed-in item field {k}"
