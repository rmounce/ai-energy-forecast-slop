import hwc_cop_analysis as hca
import pandas as pd


def test_series_query_can_read_aggregate_weather_without_entity_tag():
    q = hca._series_query(
        "humidity_adelaide",
        days=12,
        since="2026-05-28",
        field="mean_value",
        rp="rp_30m",
    )

    assert q == (
        'SELECT "mean_value" FROM "rp_30m"."humidity_adelaide" '
        "WHERE time >= '2026-05-27T14:30:00Z' AND time> now()-12d"
    )


def test_series_query_keeps_entity_filter_for_raw_ha_sensors():
    q = hca._series_query(
        "sensor__temperature",
        eid="heat_pump_temperature",
        days=3,
        since="2026-05-28",
    )

    assert q == (
        'SELECT "value" FROM "sensor__temperature" '
        "WHERE entity_id='heat_pump_temperature' "
        "AND time >= '2026-05-27T14:30:00Z' AND time> now()-3d"
    )


def test_series_query_defaults_to_install_date_bound():
    q = hca._series_query("sensor__power", eid="remaining_power_load")

    assert q == (
        'SELECT "value" FROM "sensor__power" '
        "WHERE entity_id='remaining_power_load' AND time >= '2026-05-27T14:30:00Z'"
    )


def test_first_rise_minutes_uses_fraction_of_observed_probe_lift():
    idx = pd.date_range("2026-06-02T00:00:00Z", periods=5, freq="30min")
    series = pd.Series([45.0, 45.5, 48.0, 55.0, 60.0], index=idx)

    assert hca._first_rise_minutes(series, idx[0], 45.0, 60.0, 0.10) == 60
    assert hca._first_rise_minutes(series, idx[0], 45.0, 60.0, 0.50) == 90
    assert hca._first_rise_minutes(series, idx[0], 45.0, 60.0, 0.90) == 120
