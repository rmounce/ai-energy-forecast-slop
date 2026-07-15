import hwc_cop_analysis as hca
import numpy as np
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


def test_anchor_query_fetches_last_value_before_with_entity_filter():
    q = hca._anchor_query("sensor__temperature", eid="heat_pump_temperature",
                          since="2026-06-30 12:28")
    assert q == (
        'SELECT "value" FROM "sensor__temperature" '
        "WHERE entity_id='heat_pump_temperature' AND time < '2026-06-30T02:58:00Z' "
        "ORDER BY time DESC LIMIT 1"
    )


def test_anchor_query_supports_aggregate_rp_without_entity_tag():
    q = hca._anchor_query("humidity_adelaide", since="2026-06-30 12:28",
                          field="mean_value", rp="rp_30m")
    assert q == (
        'SELECT "mean_value" FROM "rp_30m"."humidity_adelaide" '
        "WHERE time < '2026-06-30T02:58:00Z' ORDER BY time DESC LIMIT 1"
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


# ── cycle_metrics: the shared trace->summary brain (docs/hwc/local_store.md) ──


def _trace(minutes=60, tank0=45.0, tank1=56.0, power_w=1000.0, energy0=100.0,
           energy_total=1.0, start="2026-06-30T01:00:00Z", step_s=30, cols=None, drop=()):
    n = int(minutes * 60 / step_s) + 1
    idx = pd.date_range(start, periods=n, freq=f"{step_s}s", tz="UTC")
    frac = np.linspace(0.0, 1.0, n)
    df = pd.DataFrame(
        {
            "tank": tank0 + (tank1 - tank0) * frac,
            "power_w": float(power_w),
            "energy_kwh": energy0 + energy_total * frac,
            "ambient": 14.0,
            "humidity": 70.0,
            "element": 0,
            "defrost": 0,
            "four_way": 0,
            "exhaust": 20.0 + 30.0 * frac,
            "coil": 5.0,
            "return_air": 18.0,
            "inlet": 15.0,
        },
        index=idx,
    )
    for k, v in (cols or {}).items():
        df[k] = v
    return df.drop(columns=list(drop))


def test_cycle_metrics_counter_path_cop_and_clean():
    m = hca.cycle_metrics(_trace(tank1=56.0, energy_total=1.0))
    assert m["elec_source"] == "counter"
    assert abs(m["elec_kwh"] - 1.0) < 1e-6
    # tank_start/tank_end are the min/max trace reading over the cycle; this trace rises
    # monotonically so that's simply the first (45.0) and last (56.0) sample;
    # therm ≈ 225*4.186*(56-45.0)/3600 + 0.12*1h ≈ 3.00 kWh, elec 1.0 -> COP ≈ 3.00.
    assert m["tank_start"] == 45.0 and m["tank_end"] == 56.0
    assert abs(m["cop"] - 3.00) < 0.03
    assert m["clean"] is True
    assert m["hp_mean_w"] == 1000 and m["hp_p95_w"] == 1000
    assert m["dur_min"] == 60
    assert m["start_local"] == "2026-06-30 10:30"  # 01:00Z -> ACST +9:30
    assert m["element_on"] is False and m["four_way_on"] is False
    assert m["exhaust_start"] == 20.0 and m["exhaust_max"] == 50.0


def test_cycle_metrics_falls_back_to_power_integration_without_counter():
    m = hca.cycle_metrics(_trace(drop=["energy_kwh"]))
    assert m["elec_source"] == "power_integration"
    # 1000 W held across the cycle integrates to ~1 kWh (rectangular, endpoint-inclusive)
    assert abs(m["elec_kwh"] - 1.0) < 0.02


def test_cycle_metrics_rejects_implausible_counter_delta():
    m = hca.cycle_metrics(_trace(energy_total=9.0))  # > COUNTER_MAX_CYCLE_KWH
    assert m["elec_source"] == "power_integration"


def test_cycle_metrics_edge_snapshots_override_trace_boundaries():
    trace = _trace()
    edges = {
        "cs": trace.index[0], "ce": trace.index[-1],
        "tank_start": 44.0, "tank_end": 60.0,
        "energy_start": 100.0, "energy_end": 101.5,
    }
    m = hca.cycle_metrics(trace, edges=edges)
    assert abs(m["elec_kwh"] - 1.5) < 1e-6
    assert m["tank_start"] == 44.0 and m["tank_end"] == 60.0


def test_cycle_metrics_window_from_edges_subsets_a_wider_trace():
    trace = _trace(minutes=90)
    cs = trace.index[20]   # 10 min in
    ce = trace.index[-21]  # 10 min before end -> 70 min span
    m = hca.cycle_metrics(trace, edges={"cs": cs, "ce": ce})
    assert m["dur_min"] == 70


def test_cycle_metrics_tolerates_missing_optional_columns():
    m = hca.cycle_metrics(_trace(drop=["ambient", "humidity", "exhaust", "coil",
                                       "return_air", "inlet", "four_way"]))
    assert np.isnan(m["wet_bulb"])
    assert np.isnan(m["exhaust_start"])
    assert m["four_way_on"] is False
    assert m["cop"] is not None


def test_cycle_metrics_flags_element_and_defrost_when_on():
    m = hca.cycle_metrics(_trace(cols={"element": 1, "defrost": 1}))
    assert m["element_on"] is True and m["defrost_on"] is True


def test_cycle_metrics_band_cop_measures_only_the_band_traversal():
    # Linear 45→60 over 60 min: probe hits 54.0 at sample 72/120 and 60.0 at 120/120, so the
    # band consumes (120-72)/120 of the 1.0 kWh counter delta; therm credits the 6 °C band plus
    # standing loss over the 24 min traversal.
    m = hca.cycle_metrics(_trace(tank1=60.0, energy_total=1.0))
    elec = 1.0 * (120 - 72) / 120
    dur_h = (120 - 72) * 30 / 3600
    therm = 225 * 4.186 * 6.0 / 3600 + 0.12 * dur_h
    assert abs(m["band_cop"] - therm / elec) < 0.01


def test_cycle_metrics_band_cop_ignores_pre_band_probe_lag_energy():
    # Same band traversal, but preceded by a 60 min flat probe-lag phase burning another 1.0 kWh:
    # full-cycle COP halves, band_cop must not move.
    plain = hca.cycle_metrics(_trace(tank1=60.0, energy_total=1.0))
    lag = _trace(minutes=60, tank0=45.0, tank1=45.0, energy0=99.0, energy_total=1.0,
                 start="2026-06-30T00:00:00Z")
    rise = _trace(tank1=60.0, energy_total=1.0)  # starts 01:00Z, energy0=100.0 continues the meter
    lagged = hca.cycle_metrics(pd.concat([lag.iloc[:-1], rise]))
    assert lagged["cop"] < plain["cop"] - 0.5
    assert abs(lagged["band_cop"] - plain["band_cop"]) < 0.01


def test_cycle_metrics_band_cop_nan_without_a_full_traversal_from_below():
    assert np.isnan(hca.cycle_metrics(_trace(tank1=59.5))["band_cop"])       # never reaches 60
    assert np.isnan(hca.cycle_metrics(_trace(tank0=55.0, tank1=60.0))["band_cop"])  # starts in-band


def test_cycle_metrics_band_cop_uses_settle_window_for_the_final_tick():
    # Local Tuya can report the compressor-off edge before the final tick to 60: with ce pinned
    # one sample early the last in-cycle reading is 59.875, and the 60.0 lands in the settle
    # window — the traversal must still complete rather than go NaN.
    trace = _trace(tank1=60.0)
    edges = {"cs": trace.index[0], "ce": trace.index[-2]}
    m = hca.cycle_metrics(trace, edges=edges)
    assert not np.isnan(m["band_cop"])


def test_cycle_metrics_band_cop_nan_on_mid_band_probe_dip():
    trace = _trace(tank1=60.0)
    dip = (trace["tank"] >= 55.5) & (trace["tank"] <= 56.5)
    trace.loc[dip, "tank"] = trace.loc[dip, "tank"] - 1.0  # a draw hits the probe mid-band
    assert np.isnan(hca.cycle_metrics(trace)["band_cop"])


def test_cycle_metrics_none_for_empty_or_untemperatured_trace():
    assert hca.cycle_metrics(pd.DataFrame()) is None
    assert hca.cycle_metrics(None) is None
    blind = _trace()
    blind["tank"] = np.nan
    assert hca.cycle_metrics(blind) is None
