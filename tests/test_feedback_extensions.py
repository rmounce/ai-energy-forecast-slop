from copy import deepcopy

import pandas as pd
import pytest

from eval.dh_feedback_replay import execute_batch, simulate
from eval.ems_feedback_policy import select_command
from eval.minute_core_replay import publication_clock
from test_audit_mpc_clock import evidence
from test_ems_feedback import timing_bundle, export_solver
from test_sequential_core_replay import plant


def test_saved_exact_requests_avoid_new_solves_without_changing_economics(plant):
    bundle = timing_bundle(plant)
    original = simulate(bundle,export_solver)
    bundle['reuse_solves'] = original['solves']
    def forbidden(request): raise AssertionError('unexpected new solve')
    result = execute_batch(bundle,forbidden)
    assert result.pop('solve_execution') == {'reused':len(original['solves']),'new':0}
    assert result == original


def test_different_inputs_require_new_solves(plant):
    bundle = timing_bundle(plant)
    bundle['reuse_solves'] = simulate(bundle,export_solver)['solves']
    bundle['initial_soc'] = .6
    result = execute_batch(bundle,export_solver)
    assert result['solve_execution']['new'] > 0


def test_mutated_cache_result_is_rejected(plant):
    bundle = timing_bundle(plant)
    bundle['reuse_solves'] = simulate(bundle,export_solver)['solves']
    bundle['reuse_solves'][0]['result']['status'] = 'Infeasible'
    with pytest.raises(ValueError,match='Optimal'): execute_batch(bundle,export_solver)


def test_duplicate_cache_identity_is_rejected(plant):
    bundle = timing_bundle(plant)
    artifact = simulate(bundle,export_solver)['solves'][0]
    bundle['reuse_solves'] = [artifact,deepcopy(artifact)]
    with pytest.raises(ValueError,match='duplicate reused'): execute_batch(bundle,export_solver)


def test_causal_price_selects_pv_only_policy_and_future_price_stays_out(plant):
    bundle = timing_bundle(plant)
    settings = bundle['ems_execution']
    settings['controller_branches'] = 'pv_export_v1'
    settings['guards'].update(effective_general=[{'time':'2026-10-03T02:04Z','value':.02},
        {'time':'2026-10-03T02:05Z','value':.5}],controller_discharge_weight=[{'time':'2026-10-03T02:04Z','value':.04}])
    plan = deepcopy(bundle['initial_ems_plan'])
    for point in plan['points']: point.update(battery=0.,pv=2000.,hybrid=1900.,grid=-900.)
    before = select_command(plan,'2026-10-03T02:04:30Z',.5,settings)
    after = select_command(plan,'2026-10-03T02:05Z',.5,settings)
    assert before['discharge_limit_kw']==0 and after['discharge_limit_kw']==24
    settings['guards']['effective_general'][0]['time'] = '2026-10-03T01:00Z'
    with pytest.raises(ValueError,match='stale'): select_command(plan,'2026-10-03T02:04:30Z',.5,settings)


@pytest.mark.parametrize('offset',[25.,25.1])
def test_explicit_clock_sensitivity_preserves_known_clocks_and_labels_inferred(offset):
    history,controls,publication,plant = evidence()
    with pytest.raises(ValueError,match='stale'): publication_clock(history,publication)
    at,ref = publication_clock(history,publication,offset=offset,controls=controls,captured={},plant=plant)
    assert at == pd.Timestamp('2026-09-30T22:41Z')+pd.Timedelta(seconds=offset)
    assert ref['kind']=='modeled_timer_clock_sensitivity_not_observed_capture'
    history['mpc_anchor'].append({'time':'2026-09-30T22:41:25.079Z','value':71.75})
    assert publication_clock(history,publication,offset=offset,controls=controls,captured={},plant=plant)==(
        pd.Timestamp('2026-09-30T22:41:25.079Z'),None)


def test_source_changes_between_candidates_or_five_minute_trigger_remain_rejected():
    history,controls,publication,plant = evidence()
    history['load'].append({'time':'2026-09-30T22:41:25.05Z','value':999})
    with pytest.raises(ValueError,match='two supported'): publication_clock(history,publication,offset=25.,controls=controls,captured={},plant=plant)
    publication['time'] = '2026-09-30T22:45:27Z'
    with pytest.raises(ValueError,match='two supported'): publication_clock(history,publication,offset=25.,controls=controls,captured={},plant=plant)


def test_clock_and_controller_contracts_prevent_incompatible_resume(plant):
    from eval.feedback_checkpoint import validate_resume
    first = timing_bundle(plant)
    first['timer_clock_seconds'] = 25.
    second = deepcopy(first)
    second['resume_checkpoint'] = simulate(first,export_solver)['checkpoint']
    second['timer_clock_seconds'] = 25.1
    with pytest.raises(ValueError,match='contract differs'): validate_resume(second)
    second['timer_clock_seconds'] = 25.
    second['ems_execution']['controller_branches'] = 'pv_export_v1'
    with pytest.raises(ValueError,match='contract differs'): validate_resume(second)


@pytest.mark.parametrize('stamp,prevent',[
    ('2026-10-01T00:29:59Z',False),('2026-10-01T00:30Z',True),
    ('2026-10-01T06:30Z',False),('2026-10-05T23:29:59Z',False),('2026-10-05T23:30Z',True)])
def test_pv_charge_uses_frozen_site_timezone_across_dst(plant,stamp,prevent):
    settings = timing_bundle(plant)['ems_execution'] | {'controller_branches':'pv_export_v2','local_time_zone':'Australia/Adelaide'}
    plan = {'accepted_at':stamp,'points':[{'target':pd.Timestamp(stamp).floor('5min').isoformat(),
        'battery':-143.47,'grid':0.,'load':238.,'pv':394.,'hybrid':238.,'curtailment':0.}]}
    command = select_command(plan,stamp,.7,settings)
    assert (command['branch']=='pv_charge_prevent_discharge') == prevent
