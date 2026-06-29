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


def test_cycle_is_clean_ignores_baseline_drift_on_counter_path():
    # Today's 12:51 case: big pre/post baseline drift (laggy on-edge caught spin-up in the
    # pre-window), but elec came from the energy_2 counter, so the drift is irrelevant → clean.
    assert hca.cycle_is_clean(b_pre=485, b_post=4, hp_p95_w=659, cop=2.53,
                              elec_source="counter") is True


def test_cycle_is_clean_keeps_baseline_drift_on_integration_path():
    # Same drift, but elec is the baseline-subtracted power integral → the drift does contaminate
    # the COP, so the gate must still reject it.
    assert hca.cycle_is_clean(b_pre=485, b_post=4, hp_p95_w=659, cop=2.53,
                              elec_source="power_integration") is False
    # small drift on the integration path is fine
    assert hca.cycle_is_clean(b_pre=200, b_post=240, hp_p95_w=659, cop=2.53,
                              elec_source="power_integration") is True


def test_cycle_is_clean_always_gates_power_and_cop_band():
    # Peak power and COP band apply regardless of elec source.
    assert hca.cycle_is_clean(0, 0, hp_p95_w=1200, cop=2.5, elec_source="counter") is False
    assert hca.cycle_is_clean(0, 0, hp_p95_w=500, cop=3.8, elec_source="counter") is False   # > ceiling
    assert hca.cycle_is_clean(0, 0, hp_p95_w=500, cop=0.5, elec_source="counter") is False   # < floor
    assert hca.cycle_is_clean(0, 0, hp_p95_w=500, cop=float("nan"), elec_source="counter") is False


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
