from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from eval.minute_core_replay import execution_segments, advance, policy_payload, simulate
from eval.sequential_core_replay import snapshot_at, execute
from scripts.replay_energy_solves import run_batch
from test_sequential_core_replay import plant, toy_bundle


def minute_bundle(plant):
    source = toy_bundle(plant)
    steps = []
    for origin in pd.date_range('2026-10-03T02:00:20Z', periods=2, freq='1min'):
        states, _ = snapshot_at(source['parents']['baseline'], .5, source['past'], origin,
            source['apf_revisions'], source['quote_rows'])
        states.pop('sensor.sigen_plant_battery_state_of_charge_derived')
        actual = {'pv_dc_w': 0., 'load_site_w': 1000., 'general_rate': .2, 'feed_rate': .1}
        steps.append({'origin': origin.isoformat(), 'activation': (origin+pd.Timedelta(seconds=10)).isoformat(),
            'end': (origin+pd.Timedelta(minutes=1)).isoformat(), 'states': states, 'input_receipts': {},
            'published_battery_discharge_w': 0., 'before_activation': [actual | {'duration_seconds': 10}],
            'after_activation': [actual | {'duration_seconds': 50}]})
    return {'steps': steps, 'initial_soc': .5,
        'initial_command': {'battery_w': -6000., 'curtail_w': 0., 'export_limit_w': 10000.},
        'observed': {'final_soc': .9}, 'configuration': source['configuration'],
        'optimization_sha256': source['optimization_sha256'], 'source_publication_id': 'history'}


def feasible_solve(request):
    """Constant DC battery trajectory that reaches the exact requested endpoint."""
    p, plant = request['payload'], request['configuration']['plant_conf']
    count = p['prediction_horizon']
    delta = p['soc_final']-p['soc_init']
    factor = plant['battery_charge_efficiency'] if delta >= 0 else 1/plant['battery_discharge_efficiency']
    batt = -delta*plant['battery_nominal_energy_capacity']/(count*5/60*factor)
    assert p['pv_power_forecast'] == [0]*count
    ac = batt*plant['inverter_efficiency_dc_ac'] if batt >= 0 else batt/plant['inverter_efficiency_ac_dc']
    frame = pd.DataFrame({'P_Load': p['load_power_forecast'], 'P_PV': p['pv_power_forecast'],
        'P_batt': [batt]*count, 'P_grid_pos': np.asarray(p['load_power_forecast'])-ac,
        'P_grid_neg': [0.]*count, 'SOC_opt': p['soc_init']+np.arange(1,count+1)*delta/count,
        'P_hybrid_inverter': [ac]*count, 'unit_load_cost': p['load_cost_forecast'],
        'unit_prod_price': p['prod_price_forecast']}, index=pd.date_range(request['forecast_start'], periods=count, freq='5min'))
    return {'request_id': request['request_id'], 'optimization_sha256': request['optimization_sha256'],
        'status': 'Optimal', 'solve_seconds': 0., 'columns': list(frame), 'values': frame.to_numpy().tolist(),
        'targets': [stamp.isoformat() for stamp in frame.index]}


def test_own_inventory_survives_decisions_and_latency_retains_previous_command(plant):
    bundle = minute_bundle(plant)
    report = simulate(bundle, feasible_solve)
    rows = [row for row in report['steps'] if row['arm'] == 'baseline']
    expected = .5+6000*.99*10/3600/40300
    assert rows[0]['activation_soc'] == pytest.approx(expected)
    assert rows[1]['initial_soc'] == pytest.approx(rows[0]['end_soc'])
    assert rows[1]['initial_soc'] != .5
    assert rows[1]['initial_soc'] != bundle['observed']['final_soc']
    assert report['summary']['baseline']['grid_import_kwh'] > 1000*120/3_600_000


def test_future_execution_change_cannot_change_current_request(plant):
    original = minute_bundle(plant)
    changed = deepcopy(original)
    changed['steps'][0]['after_activation'][0]['load_site_w'] = 5000
    changed['steps'][0]['after_activation'][0]['general_rate'] = 999
    a, b = simulate(original, feasible_solve), simulate(changed, feasible_solve)
    assert a['solves'][0]['request'] == b['solves'][0]['request']
    assert a['summary']['baseline']['variable_cost_aud'] != b['summary']['baseline']['variable_cost_aud']


def test_observed_soc_cannot_reground_policy_payload(plant):
    states = minute_bundle(plant)['steps'][0]['states']
    changed = deepcopy(states)
    changed['sensor.sigen_plant_battery_state_of_charge_derived'] = {'state': '99'}
    assert policy_payload(states, '2026-10-03T02:00:20Z', .52, 'baseline') == policy_payload(changed, '2026-10-03T02:00:20Z', .52, 'baseline')
    a = policy_payload(states, '2026-10-03T02:00:20Z', .52, 'baseline')
    b = policy_payload(states, '2026-10-03T02:00:20Z', .52, 'without_positive_lockin')
    assert a['soc_final'] == .52 and b['soc_final'] == .5
    assert {key: value for key, value in a.items() if key != 'soc_final'} == {key: value for key, value in b.items() if key != 'soc_final'}


def test_subinterval_energy_matches_five_minute_execution_when_unclipped(plant):
    long = execute(plant, .5, -1000., 0., 100., 1000., 10000.)
    soc = .5
    segments = [{'duration_seconds': 1., 'pv_dc_w': 100., 'load_site_w': 1000., 'general_rate': .2, 'feed_rate': .1}]*300
    soc, result = advance(plant, soc, {'battery_w': -1000., 'curtail_w': 0., 'export_limit_w': 10000.}, segments)
    assert soc == pytest.approx(long['end_soc'])
    assert result['grid_import_kwh'] == pytest.approx(long['grid_import_kwh'])
    assert result['battery_throughput_kwh'] == pytest.approx(long['battery_throughput_kwh'])


@pytest.mark.parametrize('duration', [0, -1, 301, float('nan')])
def test_invalid_execution_duration_fails(plant, duration):
    with pytest.raises(ValueError): execute(plant, .5, 0., 0., 0., 1000., 10000., duration_seconds=duration)


def test_raw_execution_preserves_sign_changes_and_refuses_unsupported_gaps():
    start, end = pd.Timestamp('2026-10-03T02:00Z'), pd.Timestamp('2026-10-03T02:01Z')
    history = {key: [{'time': start.isoformat(), 'value': 1000.},
        {'time': (start+pd.Timedelta(seconds=30)).isoformat(), 'value': -1000. if key in ('battery','grid') else 1000.}]
        for key in ('pv', 'load', 'battery', 'grid')}
    rates = {key: pd.DataFrame({'rate': [.1]}, index=pd.DatetimeIndex([start])) for key in ('general','feed')}
    segments = execution_segments(history, start, end, rates)
    assert [row['observed_grid_import_w'] for row in segments] == [1000., -1000.]
    assert sum(max(row['observed_grid_import_w'], 0)*row['duration_seconds']/3_600_000 for row in segments) == pytest.approx(1/120)
    with pytest.raises(ValueError, match='incomplete held'):
        execution_segments(history, start, start+pd.Timedelta(minutes=4), rates)


def test_mutated_solver_request_rejected(plant):
    def bad(request):
        result = feasible_solve(request)
        request['payload']['soc_init'] = .1
        return result
    with pytest.raises(ValueError, match='mutated frozen'):
        simulate(minute_bundle(plant), bad)


def test_batch_worker_and_paths_cannot_escape_selected_replay_code():
    with pytest.raises(ValueError, match='unsupported batch worker'):
        run_batch({}, 'arbitrary.py', ['arbitrary.py'])
    with pytest.raises(ValueError, match='unsupported staged code path'):
        run_batch({}, 'eval/minute_core_replay.py', ['eval/minute_core_replay.py', '../config.secrets.yaml'])
