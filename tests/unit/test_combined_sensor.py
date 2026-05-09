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


# ── HAEO/HAFO canonical forecast output ───────────────────────────────────────

def test_haeo_price_forecast_items_use_datetime_native_value(tier1_dfs):
    import_items, export_items = fc._build_haeo_price_forecast_items(
        tier1_dfs["p50"], 5, 5
    )

    assert len(import_items) == 12
    assert len(export_items) == 12
    assert set(import_items[0]) == {"datetime", "native_value"}
    assert set(export_items[0]) == {"datetime", "native_value"}
    assert pd.Timestamp(import_items[0]["datetime"]).tzinfo is not None


def test_haeo_export_value_positive_for_positive_wholesale(tier1_dfs):
    _, export_items = fc._build_haeo_price_forecast_items(tier1_dfs["p50"], 5, 5)

    assert all(item["native_value"] >= 0 for item in export_items)


def test_haeo_export_value_negative_when_export_costs_money():
    prices = [-0.30, -0.50, -0.20]
    p50 = _make_price_df(prices)

    _, export_items = fc._build_haeo_price_forecast_items(p50, 5, 5)

    assert all(item["native_value"] <= 0 for item in export_items)


def test_haeo_tier2_can_publish_as_5min(tier2_dfs):
    import_items, export_items = fc._build_haeo_price_forecast_items(
        tier2_dfs["p50"], 30, 5
    )

    assert len(import_items) == 144 * 6
    assert len(export_items) == 144 * 6
    first = pd.Timestamp(import_items[0]["datetime"])
    second = pd.Timestamp(import_items[1]["datetime"])
    seventh = pd.Timestamp(import_items[6]["datetime"])
    assert second == first + pd.Timedelta(minutes=5)
    assert seventh == first + pd.Timedelta(minutes=30)


def test_publish_haeo_forecast_sensor_payload(monkeypatch):
    calls = []
    items = [{"datetime": "2025-01-01T00:00:00+00:00", "native_value": 0.25}]

    def fake_call(method, endpoint, payload=None):
        calls.append((method, endpoint, payload))
        return {}

    monkeypatch.setattr(fc, "call_ha_api", fake_call)

    fc._publish_haeo_forecast_sensor(
        "sensor.test_import_price_forecast",
        items,
        "Test Import Price Forecast",
        interval_minutes=5,
    )

    assert calls[0][0] == "POST"
    assert calls[0][1] == "states/sensor.test_import_price_forecast"
    payload = calls[0][2]
    assert payload["state"] == 0.25
    assert payload["attributes"]["forecast"] == items
    assert payload["attributes"]["unit_of_measurement"] == "$/kWh"
    assert payload["attributes"]["forecast_convention"] == "haeo_positive_import_export"


def test_canonical_tier2_cache_round_trip(monkeypatch, tmp_path, tier2_dfs):
    cfg = {
        **fc.CONFIG,
        "paths": {
            **fc.CONFIG["paths"],
            "canonical_tier2_cache_file": str(tmp_path / "tier2_cache.parquet"),
        },
    }
    monkeypatch.setattr(fc, "CONFIG", cfg)

    fc._save_canonical_tier2_cache(
        {"pd_direct_price": tier2_dfs["p50"]},
        tier2_price_key="pd_direct_price",
        tier2_label="PD-direct",
    )
    loaded, key, label = fc._load_canonical_tier2_cache(max_age_minutes=60)

    assert key == "cached_tier2_price"
    assert label == "cached PD-direct"
    assert loaded is not None
    pd.testing.assert_frame_equal(
        loaded[key],
        tier2_dfs["p50"][["wholesale_price"]],
        check_freq=False,
        check_names=False,
    )
