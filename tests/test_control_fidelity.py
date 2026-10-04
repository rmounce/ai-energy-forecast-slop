from copy import deepcopy
import json

import pandas as pd
import pytest

from eval.audit_control_fidelity import asof, parse_curve, recorded_states, audit, archived_checkpoint_runner


def test_asof_ignores_future_and_refuses_stale_or_missing():
    past = {'time': '2026-10-03T02:00Z', 'value': 5}
    future = {'time': '2026-10-03T02:01Z', 'value': 999}
    assert asof([future, past], '2026-10-03T02:00:30Z', 60) == past
    with pytest.raises(ValueError, match='stale'):
        asof([future, past], '2026-10-03T02:03Z', 60)
    with pytest.raises(ValueError, match='no causal'):
        asof([future], '2026-10-03T02:00Z', 60)


def test_published_python_and_json_curves_keep_endpoint_labels():
    curve = [{'date': '2026-10-03T02:00Z', 'power': '-1000'},
        {'date': '2026-10-03T02:05Z', 'power': '200'}]
    assert parse_curve(repr(curve), 'power') == curve
    assert parse_curve(json.dumps(curve), 'power') == curve
    assert parse_curve(repr(curve), 'power')[0]['date'] == '2026-10-03T02:00Z'


@pytest.mark.parametrize('curve', [[], [{'date': '2026-10-03T02:00', 'power': 1}],
    [{'date': '2026-10-03T02:00Z', 'power': 'nan'}],
    [{'date': '2026-10-03T02:05Z', 'power': 1}, {'date': '2026-10-03T02:00Z', 'power': 2}]])
def test_invalid_published_curves_fail(curve):
    with pytest.raises(ValueError): parse_curve(repr(curve), 'power')


def history():
    at = '2026-10-03T02:00Z'
    result = {key: [{'time': at, 'value': 50}] for key in ('soc', 'pv', 'load', 'loss', 'dh_anchor')}
    result['mode'] = [{'time': at, 'state': 'measured'}]
    for key, field, value_key in (
        ('dh_soc', 'battery_scheduled_soc_str', 'dh_soc_batt_forecast'),
        ('dh_load', 'forecasts_str', 'dh_p_load_forecast'),
        ('dh_pv', 'forecasts_str', 'dh_p_pv_forecast'),
        ('hwc', 'deferrables_schedule_json_str', 'hwc_power_plan')):
        result[key] = [{'time': at, field: repr([{'date': at, value_key: 50}])}]
    return result


def test_new_anchor_without_published_parent_rejected():
    raw = history()
    raw['dh_anchor'][0]['time'] = '2026-10-03T02:00:01Z'
    with pytest.raises(ValueError, match='anchor advanced'):
        recorded_states(raw, {}, '2026-10-03T02:00:02Z', with_parent=True)


def test_recorded_telemetry_excludes_future_parent_and_preserves_knobs():
    raw = history()
    future = deepcopy(raw['dh_soc'][0])
    future['time'] = '2026-10-03T02:01Z'
    raw['dh_soc'].append(future)
    parent = {'input_number.emhass_weight_pv_forecast': {'state': '.5'}}
    states, refs = recorded_states(raw, parent, '2026-10-03T02:00:30Z', with_parent=True)
    assert refs['dh_soc'] == '2026-10-03T02:00Z'
    assert states['input_number.emhass_weight_pv_forecast'] == parent['input_number.emhass_weight_pv_forecast']
    assert parent == {'input_number.emhass_weight_pv_forecast': {'state': '.5'}}


def test_nonfinite_control_sample_not_silently_coerced_to_zero():
    raw = history()
    raw['pv'][0]['value'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite control telemetry'):
        recorded_states(raw, {}, '2026-10-03T02:00:30Z', with_parent=True)


def test_archive_plan_disagreement_fails_before_solver_diagnostics():
    origins = pd.date_range('2026-10-03T02:00Z', periods=2, freq='5min')
    bundle = {'actuals': [{'time': at.isoformat()} for at in origins]}
    actual = pd.DataFrame({'battery_charge_w': [1000., 1000.],
        'planned_battery_discharge_w': [-999., -999.], 'soc_pct_end': [50., 50.]}, index=origins)
    raw = {'mpc_battery': [{'time': at.isoformat(), 'value': -1000.}
        for at in pd.date_range(origins[0]-pd.Timedelta(minutes=1), periods=11, freq='1min')]}
    with pytest.raises(ValueError, match='does not reproduce'):
        audit(raw, bundle, {}, actual)


def test_checkpoint_cache_requires_exact_request_and_image_and_owns_result():
    request = {'request_id': 'recorded', 'payload': {'soc_init': .5}}
    artifact = {'request': request, 'image': 'pinned', 'result': {'status': 'Optimal'}}
    runner = archived_checkpoint_runner({'core_checkpoints': [{'artifact': artifact}]})
    copied = runner(request, 'pinned')
    copied['result']['status'] = 'changed'
    assert artifact['result']['status'] == 'Optimal'
    with pytest.raises(ValueError, match='differs'):
        runner(request | {'payload': {'soc_init': .6}}, 'pinned')
    with pytest.raises(ValueError, match='differs'):
        runner(request, 'other-image')
    with pytest.raises(ValueError, match='differs'):
        runner(request | {'request_id': 'unseen'}, 'pinned')
