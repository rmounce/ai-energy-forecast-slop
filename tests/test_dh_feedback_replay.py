from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from energy_pipeline.solver_chain import project_dh_entities
from eval.dh_feedback_replay import OWN, overlay, between, simulate
from test_dh_source_admission import source_fixture
from test_minute_core_replay import minute_bundle, feasible_solve
from test_sequential_core_replay import plant


def feedback_bundle(plant):
    bundle = minute_bundle(plant)
    first = bundle['steps'][0]['states']
    first['input_text.dh_last_reground_block'] = {'state':'previous'}
    first['input_number.emhass_target_soc_offset'] = {'state':'0'}
    bundle['initial_parent'] = {key:deepcopy(first[key]) for key in OWN}
    for step in bundle['steps']:
        origin = pd.Timestamp(step['origin'])
        for key in ('before_activation','after_activation'):
            left,right = (origin,pd.Timestamp(step['activation'])) if key == 'before_activation' else (
                pd.Timestamp(step['activation']),pd.Timestamp(step['end']))
            step[key][0].update(start=left.isoformat(),end=right.isoformat())
        for key in OWN: step['states'].pop(key,None)
    states,_ = source_fixture()
    for row in states['sensor.solcast_pv_forecast_forecast_today']['attributes']['detailedForecast']:
        row.update(pv_estimate=0,pv_estimate10=0,pv_estimate90=0)
    states.update({'sensor.sigen_plant_rated_energy_capacity':{'state':'40.3'},
        'sensor.sigen_plant_battery_state_of_health':{'state':'100'},
        'input_number.battery_soc_min_target':{'state':'15'}})
    bundle['dh_events'] = [dict(origin='2026-10-03T02:00:25Z',ready=False,reasons=['load shifted']),
        dict(origin='2026-10-03T02:00:40Z',activation='2026-10-03T02:00:50Z',
            ready=True,reasons=[],states=states,input_receipts={}),
        dict(origin='2026-10-03T02:01:40Z',activation='2026-10-03T02:01:50Z',
            ready=True,reasons=[],states=deepcopy(states),input_receipts={})]
    return bundle


def solver(request):
    if request['kind'] == 'mpc': return feasible_solve(request)
    p = request['payload']
    # Neutral DH trajectory; exact endpoint unchanged in first acceptance.
    n,delta = p['prediction_horizon'],p['soc_final']-p['soc_init']
    plant = request['configuration']['plant_conf']
    factor = plant['battery_charge_efficiency'] if delta >= 0 else 1/plant['battery_discharge_efficiency']
    batt = -delta*plant['battery_nominal_energy_capacity']/(n*.5*factor)
    ac = batt*plant['inverter_efficiency_dc_ac'] if batt >= 0 else batt/plant['inverter_efficiency_ac_dc']
    frame = pd.DataFrame({'P_Load':p['load_power_forecast'],'P_PV':p['pv_power_forecast'],
        'P_batt':batt,'P_grid_pos':np.asarray(p['load_power_forecast'])-ac,'P_grid_neg':0.,
        'SOC_opt':p['soc_init']+np.arange(1,n+1)*delta/n,'P_hybrid_inverter':ac,
        'unit_load_cost':p['load_cost_forecast'],'unit_prod_price':p['prod_price_forecast']},
        index=pd.date_range(request['forecast_start'],periods=n,freq='30min'))
    return dict(request_id=request['request_id'],optimization_sha256=request['optimization_sha256'],
        status='Optimal',solve_seconds=0.,columns=list(frame),values=frame.to_numpy().tolist(),
        targets=[at.isoformat() for at in frame.index],projected_dh_entities=project_dh_entities(frame))


def test_rejected_sources_hold_parent_and_acceptance_changes_only_after_activation(plant):
    report = simulate(feedback_bundle(plant),solver)
    rows = [row for row in report['events'] if row['arm']=='baseline']
    rejected = next(row for row in rows if row['kind']=='dh' and row.get('accepted') is False)
    assert rejected['dh_parent_revision'].startswith('initial_archived_parent:')
    first_dh = next(row for row in rows if row['kind']=='dh' and 'request_id' in row)
    acceptance = next(row for row in rows if row['kind']=='dh_activation')
    assert acceptance['dh_parent_revision'] == first_dh['dh_parent_revision']
    mpc = [row for row in rows if row['kind']=='mpc']
    assert mpc[1]['dh_parent_revision'] == first_dh['request_id']
    assert len(report['solves']) == 8


def test_each_arm_has_own_parent_and_soc_never_reseeded_from_archived_events(plant):
    bundle = feedback_bundle(plant)
    changed = deepcopy(bundle)
    for event in changed['dh_events']:
        if event['ready']:
            event['states'].update({key:{'state':'99'} for key in OWN})
            event['states']['sensor.sigen_plant_battery_state_of_charge_derived'] = {'state':'99'}
    a,b = simulate(bundle,solver),simulate(changed,solver)
    assert a == b
    for arm in ('baseline','without_positive_lockin'):
        dh = [r for r in a['events'] if r['arm']==arm and r['kind']=='dh' and 'request_id' in r]
        assert dh[1]['dh_parent_revision'] == dh[0]['request_id']
        assert dh[0]['soc'] != .5


def test_future_measured_prices_do_not_enter_current_solver_inputs(plant):
    a = feedback_bundle(plant)
    b = deepcopy(a)
    b['steps'][-1]['after_activation'][0]['general_rate'] = 999
    assert [s['request'] for s in simulate(a,solver)['solves']] == [s['request'] for s in simulate(b,solver)['solves']]


def test_missing_or_mutated_admitted_sources_fail_before_dh_solve(plant):
    bundle = feedback_bundle(plant)
    bundle['dh_events'][1]['states']['sensor.ai_load_forecast_high']['attributes']['forecasts'][0]['timestamp'] = '2026-10-03T01:30Z'
    with pytest.raises(ValueError,match='admitted DH sources changed'): simulate(bundle,solver)


def test_timeline_split_conserves_duration_and_refuses_gaps():
    rows = [dict(start='2026-10-03T02:00Z',end='2026-10-03T02:01Z',duration_seconds=60)]
    assert between(rows,'2026-10-03T02:00:10Z','2026-10-03T02:00:30Z')[0]['duration_seconds'] == 20
    with pytest.raises(ValueError,match='gap'): between(rows,'2026-10-03T01:59Z','2026-10-03T02:00:30Z')


def test_owned_overlay_cannot_share_mutable_parent():
    parent = {key:{'state':'50'} for key in OWN}
    states = overlay({},parent,.6)
    states[OWN[0]]['state'] = '0'
    assert parent[OWN[0]]['state'] == '50'


def test_runtime_capacity_controls_initial_execution_and_inventory_not_base_placeholder(plant):
    bundle = feedback_bundle(plant)
    bundle['execution_plant'] = deepcopy(plant)
    bundle['configuration']['plant_conf']['battery_nominal_energy_capacity'] = 420000
    report = simulate(bundle,solver)
    baseline = report['summary']['baseline']
    assert baseline['ending_inventory_kwh'] == pytest.approx(baseline['final_soc']*40.3)
    first_dh = next(r for r in report['events'] if r['arm']=='baseline' and r['kind']=='dh' and 'request_id' in r)
    assert first_dh['soc'] > .5+6000*.99*10/3600/40300


def test_diverging_commands_produce_distinct_dh_inventory_and_parents(plant):
    bundle = feedback_bundle(plant)
    bundle['initial_soc'] = .8
    # Let distinct commands persist past the fractional-SoC rounding threshold.
    bundle['dh_events'][1].update(origin='2026-10-03T02:00:55Z',activation='2026-10-03T02:01:05Z')
    for row in bundle['initial_parent']['sensor.dh_p_load_forecast']['attributes']['forecasts']:
        row['dh_p_load_forecast'] = 2000
    report = simulate(bundle,solver)
    dh = {arm:[r for r in report['events'] if r['arm']==arm and r['kind']=='dh' and 'request_id' in r]
        for arm in ('baseline','without_positive_lockin')}
    a,b = [rows[0] for rows in dh.values()]
    assert a['soc'] != b['soc']
    assert a['soc_init'] != b['soc_init']
    for rows in dh.values():
        assert rows[1]['dh_parent_revision'] == rows[0]['request_id']
