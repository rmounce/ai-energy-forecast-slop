import pytest

import hwc_stratified_model as sm


def test_heat_grows_hot_layer_before_probe_moves():
    params = sm.StratifiedTankParams(
        hot_target_c=60.0,
        probe_height_fraction=0.5,
        thermocline_width_fraction=0.1,
        standing_loss_ua_kw_per_c=0.0,
    )
    state = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=45.0, hot_fraction=0.0)
    warmed = sm.apply_heat(state, params, heat_kwh=1.0)

    assert warmed.hot_fraction > 0.0
    assert warmed.hot_temp_c == pytest.approx(60.0)
    assert sm.probe_temp_c(warmed, params) == pytest.approx(45.0)


def test_probe_sees_thermocline_after_hot_layer_reaches_probe():
    params = sm.StratifiedTankParams(
        probe_height_fraction=0.5,
        thermocline_width_fraction=0.2,
    )

    below = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=60.0, hot_fraction=0.35)
    centred = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=60.0, hot_fraction=0.50)
    above = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=60.0, hot_fraction=0.65)

    assert sm.probe_temp_c(below, params) == pytest.approx(45.0)
    assert sm.probe_temp_c(centred, params) == pytest.approx(52.5)
    assert sm.probe_temp_c(above, params) == pytest.approx(60.0)


def test_draw_off_consumes_hot_layer_before_cooling_cold_layer():
    params = sm.StratifiedTankParams()
    state = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=60.0, hot_fraction=0.5)
    drawn = sm.apply_draw_off(state, params, draw_kwh=0.5)

    assert drawn.hot_fraction < state.hot_fraction
    assert drawn.cold_temp_c == pytest.approx(state.cold_temp_c)
    assert drawn.hot_temp_c == pytest.approx(state.hot_temp_c)


def test_idle_loss_reduces_both_layer_temperatures():
    params = sm.StratifiedTankParams(standing_loss_ua_kw_per_c=0.0025)
    state = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=60.0, hot_fraction=0.5)
    cooled = sm.apply_idle_loss(state, params, ambient_c=15.0, step_h=1.0)

    assert cooled.cold_temp_c < state.cold_temp_c
    assert cooled.hot_temp_c < state.hot_temp_c
    assert cooled.hot_fraction == pytest.approx(state.hot_fraction)


def test_simulate_reports_probe_before_each_interval_update():
    params = sm.StratifiedTankParams(
        hot_target_c=60.0,
        probe_height_fraction=0.5,
        thermocline_width_fraction=0.1,
        standing_loss_ua_kw_per_c=0.0,
    )
    state = sm.StratifiedTankState(cold_temp_c=45.0, hot_temp_c=45.0, hot_fraction=0.0)

    probes, states = sm.simulate_probe_temperatures(
        schedule_heat_kwh=[1.0, 1.0],
        draw_off_kwh=[0.0, 0.0],
        ambient_c=[15.0, 15.0],
        initial_state=state,
        params=params,
        step_h=0.5,
    )

    assert probes[0] == pytest.approx(45.0)
    assert states[1].hot_fraction > states[0].hot_fraction
