import pandas as pd
import pytest

import hwc_fit_thermal_model as fit


def _cycles():
    return pd.DataFrame(
        [
            {
                "start": "2026-06-01 10:00",
                "dur_min": 120,
                "tank_start": 47.0,
                "tank_end": 60.0,
                "hp_mean_w": 770,
                "elec_kwh": 1.5,
                "cop": 2.4,
                "clean": True,
                "probe_lag_min": 22.0,
                "probe_rise_10_min": 50.0,
                "probe_rise_50_min": 80.0,
                "probe_rise_90_min": 110.0,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
            },
            {
                "start": "2026-06-02 10:00",
                "dur_min": 60,
                "tank_start": 54.0,
                "tank_end": 60.0,
                "hp_mean_w": 810,
                "elec_kwh": 0.85,
                "cop": 2.0,
                "clean": True,
                "probe_lag_min": 15.0,
                "probe_rise_10_min": 12.0,
                "probe_rise_50_min": 35.0,
                "probe_rise_90_min": 55.0,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
            },
            {
                "start": "2026-06-03 10:00",
                "dur_min": 100,
                "tank_start": 49.0,
                "tank_end": 60.0,
                "hp_mean_w": 130,
                "elec_kwh": 0.2,
                "cop": 12.0,
                "clean": False,
                "probe_lag_min": 20.0,
                "probe_rise_10_min": 20.0,
                "probe_rise_50_min": 50.0,
                "probe_rise_90_min": 90.0,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
            },
        ]
    )


def test_prepare_cycles_classifies_full_reheat_and_top_up():
    prepared = fit.prepare_cycles(_cycles(), top_up_start_c=53.0)

    assert prepared.loc[0, "cycle_class"] == "full_reheat"
    assert prepared.loc[1, "cycle_class"] == "top_up"
    assert prepared.loc[0, "heat_rate_c_per_hour"] == pytest.approx(6.5)
    assert prepared.loc[1, "heat_rate_c_per_hour"] == pytest.approx(6.0)


def test_fit_parameters_uses_clean_compressor_only_target_cycles():
    result = fit.fit_parameters(_cycles(), top_up_start_c=53.0)

    assert result["usable_cycle_count"] == 2
    assert result["suggestions"]["nominal_power_w"] == 790
    assert result["suggestions"]["heat_rate_c_per_hour"] == 6.5
    assert result["suggestions"]["top_up_heat_rate_c_per_hour"] == 6.0
    assert result["suggestions"]["mean_clean_cop"] == 2.2
    assert result["stratified_hints"]["probe_height_fraction"] == pytest.approx(0.62, abs=0.01)
    assert result["stratified_hints"]["thermocline_width_fraction"] == pytest.approx(0.61)


def test_fit_parameters_rejects_no_usable_cycles():
    df = _cycles()
    df["clean"] = False

    with pytest.raises(ValueError, match="no clean compressor-only cycles"):
        fit.fit_parameters(df)
