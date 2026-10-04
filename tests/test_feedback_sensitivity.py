from copy import deepcopy

import pytest
import pandas as pd

from eval.audit_feedback_sensitivity import plan_diagnostics, forecast_diagnostics
from eval.dh_feedback_replay import simulate
from test_dh_feedback_replay import feedback_bundle, solver
from test_sequential_core_replay import plant
from test_prepare_load_feedback import calibration_bundle
from test_load_solver_sensitivity import fixture, SETTINGS
from eval.prepare_load_feedback import prepare_calibrated_events


def test_pairing_is_independent_of_artifact_order_and_labels_projections(plant):
    report = simulate(feedback_bundle(plant),solver)
    expected = plan_diagnostics(report)
    report['solves'].reverse()
    actual = plan_diagnostics(report)
    assert sorted(actual['pairs'],key=lambda r:(r['origin'],r['kind'])) == sorted(expected['pairs'],key=lambda r:(r['origin'],r['kind']))
    assert len(actual['pairs']) == 4
    assert 'not_future_execution_or_savings' in actual['mode']


@pytest.mark.parametrize('defect',['missing','duplicate','identity','targets'])
def test_unpaired_or_changed_evidence_fails(plant,defect):
    report = simulate(feedback_bundle(plant),solver)
    if defect == 'missing': report['solves'].pop()
    elif defect == 'duplicate': report['solves'].append(deepcopy(report['solves'][0]))
    elif defect == 'identity': report['solves'][0]['request']['payload']['soc_final'] = .99
    else: report['solves'][0]['result']['targets'][0] = '2026-10-03T01:30Z'
    with pytest.raises(ValueError): plan_diagnostics(report)


def test_forecast_scoring_counts_each_vintage_once_and_requires_complete_labels(plant):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    prepared = prepare_calibrated_events(bundle,rows,SETTINGS)
    actuals = pd.DataFrame({'load_base_w':400.},index=pd.date_range('2026-10-03T02:00Z',periods=12,freq='5min'))
    result = forecast_diagnostics(prepared,actuals)
    assert len(result) == 1
    first = result[0]['bands'][0]
    assert first['paired_complete_targets'] == 2
    assert first['baseline']['mae_w'] == 200
    assert first['calibrated']['mae_w'] == 100
    actuals.iloc[0,0] = float('nan')
    assert forecast_diagnostics(prepared,actuals)[0]['bands'][0]['paired_complete_targets'] == 1


def test_same_vintage_cannot_be_recalibrated_at_later_dh_time(plant):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    prepared = prepare_calibrated_events(bundle,rows,SETTINGS)
    prepared['dh_events'][2]['calibrated_load_rows'][0]['power_load'] += 1
    actuals = pd.DataFrame({'load_base_w':400.},index=pd.date_range('2026-10-03T02:00Z',periods=12,freq='5min'))
    with pytest.raises(ValueError,match='changed within issued'): forecast_diagnostics(prepared,actuals)
