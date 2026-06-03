import pandas as pd

import hwc_validate_cycle_traces as traces


def _config():
    return {
        "hwc": {
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
                "dur_min": 60,
                "tank_start": 50.0,
                "tank_end": 60.0,
                "ambient": 15.0,
                "wet_bulb": 12.0,
                "hp_mean_w": 760,
                "elec_kwh": 1.2,
                "therm_kwh": 2.8,
                "cop": 2.3,
                "clean": True,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
                "probe_rise_10_min": 15.0,
                "probe_rise_50_min": 35.0,
                "probe_rise_90_min": 55.0,
            },
            {
                "start": "2026-06-02 10:00",
                "dur_min": 50,
                "tank_start": 55.0,
                "tank_end": 60.0,
                "ambient": 16.0,
                "wet_bulb": 13.0,
                "hp_mean_w": 810,
                "elec_kwh": 0.8,
                "therm_kwh": 1.5,
                "cop": 1.9,
                "clean": True,
                "element_on": False,
                "defrost_on": False,
                "four_way_on": False,
                "probe_rise_10_min": 8.0,
                "probe_rise_50_min": 25.0,
                "probe_rise_90_min": 45.0,
            },
        ]
    )


def test_cycle_bounds_parse_local_times_to_utc():
    row = pd.Series({"start": "2026-06-01 10:00", "dur_min": 60})

    start, end = traces._cycle_bounds_utc(row)

    assert start.isoformat() == "2026-06-01T00:30:00+00:00"
    assert end.isoformat() == "2026-06-01T01:30:00+00:00"


def test_validate_trace_cycles_writes_report_and_trace_rows():
    cycles = traces.load_cycles_from_frame(_cycles(), target_c=60)

    def loader(row):
        idx = traces._trace_index(row, step_seconds=600)
        tank = pd.Series(
            pd.Series(range(len(idx)), index=idx, dtype=float)
            / max(1, len(idx) - 1)
            * (float(row["tank_end"]) - float(row["tank_start"]))
            + float(row["tank_start"])
        )
        return {
            "tank": tank,
            "exhaust": tank + 20.0,
            "power": pd.Series(800.0, index=idx),
            "compressor": pd.Series(1.0, index=idx),
        }

    report, trace_rows, params = traces.validate_trace_cycles(
        cycles, _config(), step_seconds=600, series_loader=loader,
    )

    assert len(report) == 2
    assert len(trace_rows) == 13
    assert params.hot_target_c == 60
    assert {"single_node_mae_c", "stratified_mae_c", "exhaust_max"}.issubset(report.columns)
    assert {"observed_tank_c", "single_node_tank_c", "stratified_tank_c"}.issubset(
        trace_rows.columns
    )
