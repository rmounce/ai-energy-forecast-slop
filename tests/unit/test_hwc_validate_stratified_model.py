import math

import pandas as pd

import hwc_validate_stratified_model as val


def _config():
    return {
        "hwc": {
            "optimization_time_step": 30,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0025,
                "desired_temp": 60,
                "max_temp": 60,
                "nominal_power_w": 790,
                "heat_rate_c_per_hour": 6.6,
            },
        }
    }


def _cycles():
    return pd.DataFrame(
        [
            {
                "start": "2026-06-01 10:00",
                "dur_min": 120,
                "tank_start": 47.0,
                "tank_end": 60.0,
                "ambient": 15.0,
                "wet_bulb": 12.0,
                "hp_mean_w": 760,
                "elec_kwh": 1.5,
                "therm_kwh": 3.6,
                "cop": 2.4,
                "clean": True,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
                "probe_rise_10_min": 40.0,
                "probe_rise_50_min": 75.0,
                "probe_rise_90_min": 110.0,
            },
            {
                "start": "2026-06-02 10:00",
                "dur_min": 60,
                "tank_start": 54.0,
                "tank_end": 60.0,
                "ambient": 16.0,
                "wet_bulb": 13.0,
                "hp_mean_w": 810,
                "elec_kwh": 0.8,
                "therm_kwh": 1.7,
                "cop": 2.1,
                "clean": True,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
                "probe_rise_10_min": 15.0,
                "probe_rise_50_min": 35.0,
                "probe_rise_90_min": 55.0,
            },
        ]
    )


def test_first_reach_minutes_returns_nan_when_threshold_is_not_reached():
    assert math.isnan(
        val._first_reach_minutes([45.0, 46.0], start=45.0, end=60.0, pct=90, step_min=1.0)
    )


def test_validate_cycles_reports_single_node_and_stratified_errors():
    cycles = val.load_validation_cycles_from_frame(_cycles())

    report, params = val.validate_cycles(cycles, _config(), step_min=5.0)

    assert len(report) == 2
    assert params.hot_target_c == 60
    assert {"block_end_err", "strat_end_err", "block_rise_50_err", "strat_rise_50_err"}.issubset(
        report.columns
    )
