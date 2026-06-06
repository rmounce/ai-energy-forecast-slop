import pandas as pd

import hwc_validate_block_model as val


def _config():
    return {
        "hwc": {
            "optimization_time_step": 5,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "desired_temp": 60,
                "max_temp": 60,
                "nominal_power_w": 780,
                "heat_rate_c_per_hour": 6.0,
                "top_up_start_temp_c": 53.0,
                "top_up_heat_rate_c_per_hour": 5.0,
            },
        }
    }


def _cycles():
    return pd.DataFrame(
        [
            {
                "start": "2026-06-01 10:00",
                "dur_min": 120,
                "tank_start": 48.0,
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
            },
            {
                "start": "2026-06-02 10:00",
                "dur_min": 60,
                "tank_start": 55.0,
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
            },
        ]
    )


def test_validate_cycles_reports_duration_power_and_end_temp_errors():
    cycles = val.load_validation_cycles_from_frame(_cycles())
    report = val.validate_cycles(cycles, _config(), step_min=5.0)

    assert len(report) == 2
    assert {
        "predicted_duration_min",
        "duration_err_min",
        "predicted_elec_kwh",
        "elec_err_kwh",
        "predicted_end_temp_at_observed_runtime",
        "end_temp_err_c",
    }.issubset(report.columns)
    assert report.loc[0, "predicted_duration_min"] == 120.0
    assert report.loc[0, "predicted_end_temp_at_observed_runtime"] == 60.0


def test_summary_splits_by_cycle_class():
    cycles = val.load_validation_cycles_from_frame(_cycles())
    report = val.validate_cycles(cycles, _config(), step_min=5.0)
    summary = val._summary(report)

    assert set(summary["class"]) == {"all", "full_reheat", "top_up"}
