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


def test_series_query_can_apply_until_bound():
    q = hca._series_query(
        "sensor__power",
        eid="remaining_power_load",
        since="2026-06-03",
        until="2026-06-04",
    )

    assert q == (
        'SELECT "value" FROM "sensor__power" '
        "WHERE entity_id='remaining_power_load' "
        "AND time >= '2026-06-02T14:30:00Z' "
        "AND time <= '2026-06-03T14:30:00Z'"
    )


def test_first_rise_minutes_uses_fraction_of_observed_probe_lift():
    idx = pd.date_range("2026-06-02T00:00:00Z", periods=5, freq="30min")
    series = pd.Series([45.0, 45.5, 48.0, 55.0, 60.0], index=idx)

    assert hca._first_rise_minutes(series, idx[0], 45.0, 60.0, 0.10) == 60
    assert hca._first_rise_minutes(series, idx[0], 45.0, 60.0, 0.50) == 90
    assert hca._first_rise_minutes(series, idx[0], 45.0, 60.0, 0.90) == 120


def test_merge_cycle_tables_replaces_duplicate_start_and_sorts():
    existing = pd.DataFrame(
        [
            {"start": "2026-06-02 10:07", "cop": 2.02, "clean": True},
            {"start": "2026-06-03 10:24", "cop": 2.10, "clean": False},
        ]
    )
    new = pd.DataFrame(
        [
            {"start": "2026-06-03 10:24", "cop": 2.24, "clean": True},
            {"start": "2026-06-04 10:11", "cop": 2.30, "clean": True},
        ]
    )

    merged = hca.merge_cycle_tables(existing, new)

    assert merged["start"].tolist() == [
        "2026-06-02 10:07",
        "2026-06-03 10:24",
        "2026-06-04 10:11",
    ]
    assert merged.loc[merged["start"] == "2026-06-03 10:24", "cop"].item() == 2.24
    assert merged.loc[merged["start"] == "2026-06-03 10:24", "clean"].item() is True
