from copy import deepcopy
import json

import pandas as pd
import pytest

from energy_pipeline.solver_replay import digest
from eval.dh_feedback_replay import simulate
from eval.feedback_checkpoint import verified_checkpoint
from test_dh_feedback_replay import feedback_bundle, solver
from test_sequential_core_replay import plant


def split(bundle):
    first,second = deepcopy(bundle),deepcopy(bundle)
    first['steps'] = first['steps'][:1]
    second['steps'] = second['steps'][1:]
    boundary = pd.Timestamp(second['steps'][0]['origin'])
    first['dh_events'] = [e for e in first['dh_events'] if pd.Timestamp(e['origin']) < boundary]
    second['dh_events'] = [e for e in second['dh_events'] if pd.Timestamp(e['origin']) >= boundary]
    return first,second


def test_chunked_replay_matches_unbroken_requests_inventory_and_plans(plant):
    bundle = feedback_bundle(plant)
    full = simulate(bundle,solver)
    first,second = split(bundle)
    a = simulate(first,solver)
    second['resume_checkpoint'] = a['checkpoint']
    second['initial_soc'] = .99
    second['initial_command']['battery_w'] = 9999
    second['initial_parent'] = {key:{'state':'99'} for key in second['initial_parent']}
    b = simulate(second,solver)
    assert b['checkpoint'] == full['checkpoint']
    assert {s['request']['request_id'] for s in a['solves']+b['solves']} == {s['request']['request_id'] for s in full['solves']}
    for arm in full['summary']:
        assert b['summary'][arm]['initial_soc'] == a['summary'][arm]['final_soc']
        for key in ('variable_cost_aud','grid_export_kwh','grid_import_kwh','battery_throughput_kwh'):
            assert a['summary'][arm][key]+b['summary'][arm][key] == pytest.approx(full['summary'][arm][key])


def test_gap_uses_previous_own_command_and_explicit_targets(plant):
    bundle = feedback_bundle(plant)
    full = simulate(bundle,solver)
    first,second = split(bundle)
    old = first['steps'][0]['end']
    cut = (pd.Timestamp(old)-pd.Timedelta(seconds=10)).isoformat()
    first['steps'][0]['end'] = cut
    first['steps'][0]['after_activation'][0].update(end=cut,duration_seconds=40)
    a = simulate(first,solver)
    second['resume_checkpoint'] = a['checkpoint']
    with pytest.raises(ValueError,match='timeline gap'): simulate(second,solver)
    second['prelude'] = [deepcopy(first['steps'][0]['after_activation'][0]) | {
        'start':cut,'end':old,'duration_seconds':10}]
    b = simulate(second,solver)
    for arm in full['summary']:
        assert b['summary'][arm]['final_soc'] == pytest.approx(full['summary'][arm]['final_soc'])


@pytest.mark.parametrize('change',['experiment','configuration','future','long_gap'])
def test_incompatible_or_discontinuous_checkpoint_is_rejected(plant,change):
    first,second = split(feedback_bundle(plant))
    second['resume_checkpoint'] = simulate(first,solver)['checkpoint']
    if change == 'experiment': second['experiment'] = 'load_calibration'
    elif change == 'configuration': second['configuration']['plant_conf']['battery_charge_efficiency'] = .8
    else: second['resume_checkpoint']['cursor'] = '2026-10-03T02:03Z' if change=='future' else '2026-10-03T01:00Z'
    with pytest.raises(ValueError): simulate(second,solver)


def test_saved_checkpoint_requires_exact_reproduced_prior_evidence(plant,tmp_path):
    bundle = feedback_bundle(plant)
    report = simulate(bundle,solver)
    report['bundle_sha256'] = digest(bundle)
    (tmp_path/'bundle.json').write_text(json.dumps(bundle))
    (tmp_path/'report.json').write_text(json.dumps(report))
    checkpoint,_ = verified_checkpoint(tmp_path)
    assert checkpoint == report['checkpoint']
    report['checkpoint']['arms']['baseline']['soc'] = .99
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='checkpoint differs'): verified_checkpoint(tmp_path)


def test_prior_artifact_without_checkpoint_can_be_verified_and_derived(plant,tmp_path):
    bundle = feedback_bundle(plant)
    report = simulate(bundle,solver)
    expected = report.pop('checkpoint')
    report['bundle_sha256'] = digest(bundle)
    (tmp_path/'bundle.json').write_text(json.dumps(bundle))
    (tmp_path/'report.json').write_text(json.dumps(report))
    assert verified_checkpoint(tmp_path)[0] == expected


def test_chain_totals_flows_once_and_checks_saved_report_lineage(plant,tmp_path):
    from eval.summarize_feedback_chain import summarize
    first,second = split(feedback_bundle(plant))
    folders = [tmp_path/'first',tmp_path/'second']
    def save(folder,bundle):
        folder.mkdir()
        report = simulate(bundle,solver)
        report['bundle_sha256'] = digest(bundle)
        (folder/'bundle.json').write_text(json.dumps(bundle))
        (folder/'report.json').write_text(json.dumps(report))
        return report
    a = save(folders[0],first)
    checkpoint,sha = verified_checkpoint(folders[0])
    second['resume_checkpoint'] = checkpoint
    second['provenance'] = {'resume_parent_report_sha256':sha}
    b = save(folders[1],second)
    total = summarize(folders)
    for arm,values in total['summary'].items():
        assert values['ending_inventory_kwh'] == b['summary'][arm]['ending_inventory_kwh']
        assert values['initial_soc'] == a['summary'][arm]['initial_soc']
        assert values['variable_cost_aud'] == a['summary'][arm]['variable_cost_aud']+b['summary'][arm]['variable_cost_aud']
    second['provenance']['resume_parent_report_sha256'] = 'changed'
    save_path = folders[1]/'bundle.json'
    save_path.write_text(json.dumps(second))
    b['bundle_sha256'] = digest(second)
    (folders[1]/'report.json').write_text(json.dumps(b))
    with pytest.raises(ValueError,match='lineage'): summarize(folders)
    with pytest.raises(ValueError,match='initial batch'): summarize(folders[1:])
