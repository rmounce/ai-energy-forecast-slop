from copy import deepcopy

import pandas as pd
import pytest

from eval.prepare_load_feedback import prepare_calibrated_events, LOAD
from eval.dh_feedback_replay import simulate
from test_dh_feedback_replay import feedback_bundle, solver
from test_load_solver_sensitivity import fixture, SETTINGS
from test_sequential_core_replay import plant


def calibration_bundle(plant):
    bundle = feedback_bundle(plant)
    bundle['experiment'] = 'load_calibration'
    for event in bundle['dh_events']:
        if event['ready']:
            event['input_receipts']['load_forecast'] = '2026-10-03T01:59Z'
            for row in event['states'][LOAD]['attributes']['forecasts']: row['power_load'] = 600.
    return bundle


def test_exact_lineage_correction_changes_only_base_load_and_keeps_rejected_event(plant):
    bundle = calibration_bundle(plant)
    before = deepcopy(bundle)
    _,rows = fixture()
    prepared = prepare_calibrated_events(bundle,rows,SETTINGS)
    assert before == bundle
    assert prepared['dh_events'][0] == bundle['dh_events'][0]
    event = prepared['dh_events'][1]
    assert event['calibrated_load_rows'][0]['power_load'] == 500
    assert event['load_calibration']['forecast_creation'] == '2026-10-03T01:32:11+00:00'
    assert event['states'] == bundle['dh_events'][1]['states']
    assert event['load_calibration']['matching_vector_targets'] >= 12
    assert event['calibrated_load_rows'] == prepared['dh_events'][2]['calibrated_load_rows']


def test_later_measurements_and_unreceived_vintage_cannot_change_issued_correction(plant):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    expected = prepare_calibrated_events(bundle,rows,SETTINGS)
    changed = rows.copy()
    changed.loc[changed.forecast_target_time >= pd.Timestamp('2026-10-03T01:00Z'),'actual_w'] = 99999
    future = rows.loc[rows.forecast_creation_time == rows.forecast_creation_time.max()].copy()
    future['forecast_creation_time'] = pd.Timestamp('2026-10-03T02:00Z')
    future['prediction'] = 99999
    assert expected == prepare_calibrated_events(bundle,pd.concat([changed,future],ignore_index=True),SETTINGS)


def test_training_label_available_by_dh_time_but_not_creation_is_excluded(plant):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    rows.loc[0,'forecast_target_time'] = pd.Timestamp('2026-10-03T01:00Z')
    result = prepare_calibrated_events(bundle,rows,SETTINGS)
    assert result['dh_events'][1]['load_calibration']['corrections']['0–6h']['training_targets'] == 0
    assert result['dh_events'][1]['calibrated_load_rows'][0]['power_load'] == 600


def test_exact_logger_vector_after_receipt_is_bounded_by_decision(plant):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    for event in bundle['dh_events']:
        if event['ready']: event['input_receipts']['load_forecast'] = '2026-10-03T02:00:39Z'
    recent = rows.loc[rows.forecast_target_time >= pd.Timestamp('2026-10-03T02:30Z')].copy()
    recent['forecast_creation_time'] = pd.Timestamp('2026-10-03T02:00:39.022Z')
    recent['horizon_hours'] = (recent.forecast_target_time-recent.forecast_creation_time).dt.total_seconds()/3600
    result = prepare_calibrated_events(bundle,pd.concat([rows,recent],ignore_index=True),SETTINGS)
    assert result['dh_events'][1]['load_calibration']['logged_creation_minus_receipt_seconds'] == pytest.approx(.022)
    future = recent.copy()
    future['forecast_creation_time'] = pd.Timestamp('2026-10-03T02:00:40.5Z')
    # First decision is02:00:40: an exact matching future vector is still excluded.
    after = prepare_calibrated_events(bundle,pd.concat([rows,recent,future],ignore_index=True),SETTINGS)
    assert after['dh_events'][1] == result['dh_events'][1]


@pytest.mark.parametrize('defect',['vector','future_receipt','experiment'])
def test_invalid_lineage_or_receipt_refuses_preparation(plant,defect):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    if defect == 'vector': rows.loc[1,'prediction'] += 1
    elif defect == 'future_receipt': bundle['dh_events'][1]['input_receipts']['load_forecast'] = '2026-10-03T02:10Z'
    else: bundle['experiment'] = 'terminal_policy'
    with pytest.raises(ValueError): prepare_calibrated_events(bundle,rows,SETTINGS)


def test_load_experiment_preserves_terminal_policy_and_accepts_own_challenger_plan(plant):
    bundle = calibration_bundle(plant)
    _,rows = fixture()
    prepared = prepare_calibrated_events(bundle,rows,SETTINGS)
    report = simulate(prepared,solver)
    assert set(report['summary']) == {'baseline','calibrated_load'}
    dh = [a for a in report['solves'] if a['request']['kind']=='dh']
    a,b = dh[0]['request'],dh[2]['request']
    assert a['counterfactual']['arm'] == 'baseline'
    assert b['counterfactual']['arm'] == 'calibrated_load'
    assert a['payload']['load_power_forecast'][0] == 740
    assert b['payload']['load_power_forecast'][0] == 640
    for key in a['payload']:
        if key != 'load_power_forecast': assert a['payload'][key] == b['payload'][key]
    assert b['counterfactual']['input_receipts']['load_calibration']['model_version'] == 'v1'
