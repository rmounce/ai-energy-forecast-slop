"""Policy edge cases independent of HA, models, config and household telemetry."""
from datetime import datetime, timedelta, timezone
from energy_pipeline.payloads import (Inputs, boundary, dh_soc, export_allowance,
    hwc_averages, mpc_soc, native_hwc, scale_power, smooth_power, target_soc_offset)

START = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def inputs(now=START, actual=55, prior=None, same_block=False):
    def entity(state='unknown', **attrs):
        return {'state': str(state), 'attributes': attrs}
    return Inputs({
        'sensor.sigen_plant_battery_state_of_charge_derived': entity(actual),
        'input_number.dh_last_soc_init': entity(50),
        'input_number.battery_soc_min_target': entity(10),
        'input_number.emhass_target_soc_offset': entity(2),
        'input_number.sapn_free_exports': entity(1),
        'input_text.dh_last_reground_block': entity(boundary(now, 30).strftime('dh-%Y%m%dT%H%MZ') if same_block else ''),
        'sensor.dh_soc_batt_forecast': entity(battery_scheduled_soc=prior or []),
    }, now)


def test_sparse_hwc_counts_present_samples_and_repeats_only_dh_tail():
    rows = [{'date': (START+timedelta(minutes=5)).isoformat(), 'hwc_power_plan': 601},
            {'date': (START+timedelta(hours=24)).isoformat(), 'hwc_power_plan': 900}]
    assert hwc_averages(rows, START)[0] == 601
    assert hwc_averages(rows, START, True)[96] == 900
    assert hwc_averages(rows, START)[96] == 0
    assert native_hwc(rows, START)[:3] == [0, 601, 0]


def test_reground_breaks_feedback_chain_and_full_guard():
    prior = [{'date': START.isoformat(), 'dh_soc_batt_forecast': 60}]
    at = START+timedelta(minutes=15)
    assert dh_soc(inputs(at, prior=prior)).soc_init_pct == 55
    same = dh_soc(inputs(at, prior=prior, same_block=True))
    assert same.deviation_pct == 0
    assert same.soc_init_pct == 50
    assert mpc_soc(inputs(at, actual=100, prior=prior)).soc_init_pct == 100


def test_negative_mpc_deviation_does_not_reduce_terminal_soc():
    prior = [{'date': START.isoformat(), 'dh_soc_batt_forecast': 60}]
    result = mpc_soc(inputs(START+timedelta(minutes=15), actual=40, prior=prior))
    assert result.deviation_pct == -15
    assert result.soc_init_pct == 40
    assert result.soc_final_pct == 60


def test_dh_final_endpoint_has_no_deviation_segment():
    prior = [{'date': START.isoformat(), 'dh_soc_batt_forecast': 60}]
    result = dh_soc(inputs(START+timedelta(minutes=30), prior=prior, same_block=True))
    assert result.deviation_pct == 0


def test_scaling_preserves_live_prefix_and_native_hwc_even_when_no_base():
    assert scale_power([11, 12, 0, 0], 100, [999, 999, 30, 40]) == [11, 12, 30, 40]
    assert scale_power([10, 10, 40, 40], 120, [0, 0, 20, 20]) == [10, 10, 50, 50]
    assert scale_power([100, 100, 10, 10], 0) == [100, 100, 0, 0]


def test_smoothing_preserves_block_mean_before_integer_rounding():
    result = smooth_power([600, 1200, 600])
    assert len(result) == 18
    assert [sum(result[i:i+6]) for i in (0, 6, 12)] == [3600, 7200, 3600]


def test_export_allowance_tracks_adelaide_dst_and_exclusive_end():
    source = inputs()
    assert export_allowance(source, '2026-10-03T00:30:00Z') == .01 # 10:00 standard
    assert export_allowance(source, '2026-10-04T00:00:00Z') == .01 # 10:30 daylight
    assert export_allowance(source, '2026-10-04T05:30:00Z') == 0 # 16:00 daylight
    assert boundary(datetime.fromisoformat('2026-10-04T03:01:00+10:30'), 30).isoformat() == '2026-10-03T16:30:00+00:00'


def test_offset_only_uses_far_tail():
    rows = [{'date': (START+timedelta(hours=h)).isoformat(), 'dh_soc_batt_forecast': soc}
            for h,soc in [(47.5,100), (48,90), (49,95)]]
    assert target_soc_offset(inputs(prior=rows)) == 4.5
