"""
Regression guard: feed-in sign convention in _build_combined_forecast_items.

Amber convention: negative per_kwh = consumer earns money.
For positive wholesale prices, the consumer earns when exporting — so per_kwh must be negative.

This test caught a sign inversion bug in April 2026 where feed-in items had positive
per_kwh during positive-price periods, causing EMHASS to incorrectly value exports.
"""

import pytest
import forecast as fc
from conftest import _make_price_df


@pytest.fixture(autouse=True)
def _patch(patched_config, fixed_loss_factor):
    pass


def test_feed_in_per_kwh_negative_for_positive_wholesale():
    """Positive wholesale prices → all feed-in per_kwh must be ≤ 0."""
    prices = [0.05, 0.08, 0.15, 0.30, 0.50]  # all positive
    p50 = _make_price_df(prices)
    low = _make_price_df([v * 0.8 for v in prices])
    high = _make_price_df([v * 1.2 for v in prices])

    _, feed_in_items = fc._build_combined_forecast_items(p50, low, high, 30)

    for item in feed_in_items:
        assert item["per_kwh"] <= 0, (
            f"feed-in per_kwh={item['per_kwh']:.6f} for positive wholesale — "
            "should be ≤ 0 (consumer earns)"
        )


def test_feed_in_per_kwh_positive_for_negative_wholesale():
    """Strongly negative wholesale (curtailment event) → feed-in per_kwh positive (cost to export)."""
    prices = [-0.30, -0.50, -0.20]  # negative = curtailment / oversupply
    p50 = _make_price_df(prices)
    low = _make_price_df([v * 0.8 for v in prices])
    high = _make_price_df([v * 1.2 for v in prices])

    _, feed_in_items = fc._build_combined_forecast_items(p50, low, high, 30)

    for item in feed_in_items:
        assert item["per_kwh"] >= 0, (
            f"feed-in per_kwh={item['per_kwh']:.6f} for negative wholesale — "
            "should be ≥ 0 (cost to export during curtailment)"
        )


def test_general_per_kwh_positive_for_positive_wholesale():
    """Positive wholesale → general_price per_kwh positive (consumer pays to import)."""
    prices = [0.10, 0.15]
    p50 = _make_price_df(prices)
    low = _make_price_df([v * 0.8 for v in prices])
    high = _make_price_df([v * 1.2 for v in prices])

    gen_items, _ = fc._build_combined_forecast_items(p50, low, high, 30)

    for item in gen_items:
        assert item["per_kwh"] > 0, (
            f"general per_kwh={item['per_kwh']:.6f} should be > 0 for positive wholesale"
        )


def test_advanced_price_high_low_order_feed_in():
    """
    Feed-in high > low: higher wholesale → more negative → more earning.
    After negation: advanced_price_high should be LESS negative than advanced_price_low
    ... wait, Amber convention: high=best earning opportunity.
    So for positive wholesale: high = -feed_in_high (most negative) is the best offer.
    Let's just verify high ≤ low (both negative, high is more negative = better earning).
    """
    low_prices = [0.05] * 3
    high_prices = [0.15] * 3  # high quantile = higher wholesale = better earning on sell
    p50_prices = [0.10] * 3

    p50 = _make_price_df(p50_prices)
    low = _make_price_df(low_prices)
    high = _make_price_df(high_prices)

    _, feed_in_items = fc._build_combined_forecast_items(p50, low, high, 30)

    for item in feed_in_items:
        # higher wholesale → higher earn → more negative → advanced_price_high < advanced_price_low
        assert item["advanced_price_high"] <= item["advanced_price_low"], (
            f"feed-in: high={item['advanced_price_high']:.6f} should be ≤ low={item['advanced_price_low']:.6f}"
        )


def test_feed_in_negation_is_exact():
    """_build_combined_forecast_items negates feed_in_price exactly (no scaling)."""
    prices = [0.10]
    p50 = _make_price_df(prices)
    low = p50.copy()
    high = p50.copy()

    # Apply tariffs directly to get expected feed_in_price
    df_ref = p50.copy()
    fc.apply_tariffs_to_forecast(df_ref)
    expected_fip = df_ref["feed_in_price"].iloc[0]

    _, feed_in_items = fc._build_combined_forecast_items(p50, low, high, 30)

    assert abs(feed_in_items[0]["per_kwh"] - (-expected_fip)) < 1e-9
