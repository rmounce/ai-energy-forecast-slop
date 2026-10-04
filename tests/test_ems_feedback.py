from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from eval.dh_feedback_replay import simulate
from eval.ems_feedback_policy import project, selected, activate, validate_guard_archive
from eval.feedback_checkpoint import validate_resume
from test_dh_feedback_replay import feedback_bundle, solver
from test_sequential_core_replay import plant


def timing_bundle(plant):
    bundle = feedback_bundle(plant)
    shift = lambda stamp: (pd.Timestamp(stamp)+pd.Timedelta(minutes=4)).isoformat()
    for step in bundle['steps']:
        for key in ('origin','activation','end'): step[key] = shift(step[key])
        for key in ('before_activation','after_activation'):
            for row in step[key]:
                for field in ('start','end'): row[field] = shift(row[field])
    for event in bundle['dh_events']:
        for key in ('origin','activation'):
            if key in event: event[key] = shift(event[key])
    receipt = '2026-10-03T02:04:10Z'
    bundle.update(experiment='ems_timing',execution_plant=deepcopy(plant),
        ems_execution={'schema':1,'dc_fixed_loss_w':0.,'minimum_export_soc':.05,
            'guards': {'effective_feed':[{'time':receipt,'value':.1}],
                       'flexible_export_limit':[{'time':receipt,'value':10}],
                       'grid_status':[{'time':receipt,'state':'On Grid'}]}},
        initial_ems_plan={'accepted_at':receipt,'points':[
            {'target':'2026-10-03T02:00Z','battery':8000.,'load':1000.,'pv':0.,
             'hybrid':7600.,'grid':-6600.,'curtailment':0.},
            {'target':'2026-10-03T02:05Z','battery':1000/.95,'load':1000.,'pv':0.,
             'hybrid':1000.,'grid':0.,'curtailment':0.}]})
    return bundle


def export_solver(request):
    if request['kind'] == 'dh': return solver(request)
    p = request['payload']
    plant = request['configuration']['plant_conf']
    count = p['prediction_horizon']
    hours = p['optimization_time_step']/60
    # A valid endpoint-reaching plan whose preceding next slot self-consumes.
    batt = np.zeros(count)
    batt[0] = 8000
    batt[1] = p['load_power_forecast'][1]/plant['inverter_efficiency_dc_ac']
    total = (p['soc_init']-p['soc_final'])*plant['battery_nominal_energy_capacity']/hours
    remaining = (total-sum(batt[:2])/plant['battery_discharge_efficiency'])/(count-2)
    batt[2:] = remaining*(plant['battery_discharge_efficiency'] if remaining >= 0
                          else 1/plant['battery_charge_efficiency'])
    ac = np.where(batt >= 0,batt*.95,batt/.95)
    grid = np.asarray(p['load_power_forecast'])-ac
    change = np.where(batt >= 0,batt/.99,batt*.99)*hours/plant['battery_nominal_energy_capacity']
    frame = pd.DataFrame({'P_Load':p['load_power_forecast'],'P_PV':p['pv_power_forecast'],
        'P_batt':batt,'P_grid_pos':np.maximum(grid,0),'P_grid_neg':np.minimum(grid,0),
        'SOC_opt':p['soc_init']-np.cumsum(change),'P_hybrid_inverter':ac,
        'unit_load_cost':p['load_cost_forecast'],'unit_prod_price':p['prod_price_forecast']},
        index=pd.date_range(request['forecast_start'],periods=count,freq='5min'))
    return dict(request_id=request['request_id'],optimization_sha256=request['optimization_sha256'],
        status='Optimal',solve_seconds=0.,columns=list(frame),values=frame.to_numpy().tolist(),
        targets=[at.isoformat() for at in frame.index])


def test_own_timing_changes_later_optimizer_inputs_and_dh_feedback(plant):
    report = simulate(timing_bundle(plant),export_solver)
    requests = {arm:[a['request'] for a in report['solves'] if a['request']['counterfactual']['arm']==arm]
                for arm in report['summary']}
    base,hold = requests.values()
    mpc = lambda rows: [r for r in rows if r['kind']=='mpc']
    assert mpc(base)[0]['payload'] == mpc(hold)[0]['payload']
    assert mpc(base)[1]['payload']['soc_init'] > mpc(hold)[1]['payload']['soc_init']
    dh = lambda rows: [r for r in rows if r['kind']=='dh']
    assert dh(base)[1]['payload']['soc_init'] > dh(hold)[1]['payload']['soc_init']
    assert report['comparison']['cashflow_delta_aud'] > 0
    assert report['comparison']['ending_inventory_delta_kwh'] < 0
    tick = [r for r in report['events'] if r['kind']=='fallback']
    assert tick[0]['ems_command']['branch']=='self_consume'
    assert tick[1]['ems_command']['branch']=='partial_export'


def test_chunked_timing_carries_owned_plan_command_and_physics(plant):
    bundle = timing_bundle(plant)
    full = simulate(bundle,export_solver)
    first,second = deepcopy(bundle),deepcopy(bundle)
    first['steps'] = first['steps'][:1]
    second['steps'] = second['steps'][1:]
    boundary = pd.Timestamp(second['steps'][0]['origin'])
    first['dh_events'] = [e for e in first['dh_events'] if pd.Timestamp(e['origin']) < boundary]
    second['dh_events'] = [e for e in second['dh_events'] if pd.Timestamp(e['origin']) >= boundary]
    a = simulate(first,export_solver)
    second['resume_checkpoint'] = a['checkpoint']
    second['initial_ems_plan']['points'][0]['grid'] = 999
    b = simulate(second,export_solver)
    assert b['checkpoint'] == full['checkpoint']
    for arm in full['summary']:
        assert a['summary'][arm]['variable_cost_aud']+b['summary'][arm]['variable_cost_aud'] == pytest.approx(full['summary'][arm]['variable_cost_aud'])
    second['ems_execution']['dc_fixed_loss_w'] = 140.
    with pytest.raises(ValueError,match='contract differs'): validate_resume(second)


def test_future_scoring_prices_do_not_change_solver_requests(plant):
    a = timing_bundle(plant)
    b = deepcopy(a)
    for step in b['steps']:
        for row in step['before_activation']+step['after_activation']: row['feed_rate'] = 999.
    requests = lambda bundle: [a['request'] for a in simulate(bundle,export_solver)['solves']]
    assert requests(a) == requests(b)


def test_changed_archived_soc_and_control_trajectory_do_not_reseed(plant):
    a = timing_bundle(plant)
    b = deepcopy(a)
    b['observed']['final_soc'] = .01
    for step in b['steps']:
        step['states']['sensor.sigen_plant_battery_state_of_charge_derived'] = {'state':'1'}
    first,second = simulate(a,export_solver),simulate(b,export_solver)
    assert first['summary'] == second['summary']
    assert first['solves'] == second['solves']


def test_unsupported_counterfactual_plan_aborts_instead_of_using_recorded_control(plant):
    bundle = timing_bundle(plant)
    bundle['initial_ems_plan']['points'][0].update(battery=0.,grid=-1000.)
    with pytest.raises(ValueError,match='PV-only'): simulate(bundle,export_solver)


def test_fixed_loss_changes_stock_and_checkpoint_contract(plant):
    a = timing_bundle(plant)
    b = deepcopy(a)
    b['ems_execution']['dc_fixed_loss_w'] = 140.
    zero,loss = simulate(a,export_solver),simulate(b,export_solver)
    assert loss['summary']['baseline']['ending_inventory_kwh'] < zero['summary']['baseline']['ending_inventory_kwh']
    assert loss['checkpoint']['contract'] != zero['checkpoint']['contract']
    last = lambda report: [row['soc'] for row in report['events'] if row['arm']=='baseline' and row['kind']=='mpc'][-1]
    assert last(loss) < last(zero)  # Solver input additionally rounds fractional SOC to four decimals.


def test_updated_physical_floor_cannot_allow_unsupported_guard_crossing(plant):
    bundle = timing_bundle(plant)
    lowered = deepcopy(plant)
    lowered['battery_minimum_state_of_charge'] = .01
    with pytest.raises(ValueError,match='guard above physical'):
        activate(bundle['initial_ems_plan'],.5,bundle['ems_execution'],lowered)


def test_projection_keeps_negative_export_and_rounds_consumed_fields():
    frame = pd.DataFrame({'P_batt':[1000.123],'P_grid_pos':[0.],'P_grid_neg':[-400.126],
        'P_Load':[550.],'P_PV':[0.],'P_hybrid_inverter':[950.116]},index=pd.to_datetime(['2026-10-03T02:00Z']))
    plan = project(frame,'2026-10-03T02:00:20Z')
    point = selected(plan,'2026-10-03T02:00:30Z')
    assert point['grid']==-400.13 and point['battery']==1000.12
    with pytest.raises(ValueError,match='coverage'): selected(plan,'2026-10-03T02:02:21Z')


def test_changed_guard_archive_requires_overlap_audit(plant):
    bundle = timing_bundle(plant)
    bundle['provenance'] = {'ems_execution':{'history_sha256':'frozen'}}
    previous = deepcopy(bundle)
    validate_guard_archive(bundle,previous)
    bundle['ems_execution']['guards']['effective_feed'][0]['value'] = 0
    with pytest.raises(ValueError,match='guard archive differs'): validate_guard_archive(bundle,previous)


def test_feed_receipt_must_cover_entire_execution_segment(plant):
    bundle = timing_bundle(plant)
    bundle['ems_execution']['guards']['effective_feed'][0]['time'] = '2026-10-03T01:49:30Z'
    with pytest.raises(ValueError,match='stale before execution'): simulate(bundle,export_solver)
