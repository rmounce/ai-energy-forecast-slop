from copy import deepcopy

import pandas as pd
import pytest

from energy_pipeline.payloads import Inputs, build_dh_payload
from eval.compare_load_solver_sensitivity import calibrated_handoff, comparison
from scripts.replay_energy_solves import run_request


ORIGIN = pd.Timestamp('2026-10-03T01:32:11Z')
SETTINGS = {'history_days': 3, 'min_targets': 1, 'availability_lag_minutes': 30}


def fixture():
    targets = pd.date_range('2026-10-03T01:30:00Z', periods=24, freq='30min')
    base = [{'timestamp': t.isoformat(), 'power_load': 600.} for t in targets]
    states = {'sensor.ai_load_forecast_high': {'attributes': {'forecasts': base}},
              'sensor.solcast_pv_forecast_forecast_today': {'attributes': {'detailedForecast': [
                  {'period_start': t.isoformat(), 'pv_estimate': 1.} for t in targets]}}}
    record = {'input_snapshot': {'states': states}, 'captured_at': '2026-10-03T01:51:09Z',
              'input_revision': 'original', 'lineage': {'mpc_dh_parent': 'unchanged'},
              'publication_id': 'historical', 'payloads': {'dh': build_dh_payload(Inputs(states, pd.Timestamp('2026-10-03T01:51:09Z').to_pydatetime()))}}
    issued = pd.DataFrame({'forecast_target_time': targets[1:], 'forecast_creation_time': ORIGIN,
                          'model_name': 'load_p65', 'model_version': 'v1', 'prediction_type': 'simple',
                          'prediction': 600., 'actual_w': 999.,
                          'horizon_hours': [(t-ORIGIN).total_seconds()/3600 for t in targets[1:]]})
    released = issued.iloc[[0]].copy()
    released['forecast_target_time'] = pd.Timestamp('2026-10-03T00:00:00Z')
    released['forecast_creation_time'] = pd.Timestamp('2026-10-02T23:00:00Z')
    released['horizon_hours'] = 1.
    released['actual_w'] = 500.
    return record, pd.concat([released, issued], ignore_index=True)


def test_past_labels_only_and_partial_interval_unchanged():
    record, rows = fixture()
    before = deepcopy(record)
    result, provenance = calibrated_handoff(record, rows, SETTINGS)
    assert record == before
    assert provenance['corrections']['0–6h']['correction_w'] == -100
    assert result['payloads']['dh']['load_power_forecast'][0] == record['payloads']['dh']['load_power_forecast'][0]
    assert result['payloads']['dh']['load_power_forecast'][1] == 500
    changed = rows.copy()
    changed.loc[changed.forecast_target_time >= ORIGIN, 'actual_w'] = -99999
    other, _ = calibrated_handoff(record, changed, SETTINGS)
    assert other['payloads'] == result['payloads']
    for key in record['payloads']['dh']:
        if key != 'load_power_forecast':
            assert result['payloads']['dh'][key] == record['payloads']['dh'][key]
    assert result['lineage'] == record['lineage']
    assert provenance['publication_authorized'] is False


def test_completed_label_receipt_lag_not_target_start():
    record, rows = fixture()
    rows.loc[0, 'forecast_target_time'] = pd.Timestamp('2026-10-03T01:00:00Z')
    _, provenance = calibrated_handoff(record, rows, SETTINGS)
    assert provenance['corrections']['0–6h']['training_targets'] == 0


def test_version_isolation_and_latest_vintage_per_training_target():
    record, rows = fixture()
    stale = rows.iloc[[0]].copy()
    stale.forecast_creation_time -= pd.Timedelta(minutes=30)
    stale.prediction = 9999
    foreign = stale.copy()
    foreign.model_version = 'another-version'
    result, provenance = calibrated_handoff(record, pd.concat([rows, stale, foreign], ignore_index=True), SETTINGS)
    assert provenance['corrections']['0–6h']['training_targets'] == 1
    assert result['payloads']['dh']['load_power_forecast'][1] == 500


@pytest.mark.parametrize('problem', ['vector', 'parity', 'duplicate'])
def test_ambiguous_or_unmatched_lineage_rejected(problem):
    record, rows = fixture()
    if problem == 'vector': rows.loc[1, 'prediction'] += 1
    elif problem == 'parity': record['payloads']['dh']['load_power_forecast'][0] += 1
    else: rows = pd.concat([rows, rows.iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError): calibrated_handoff(record, rows, SETTINGS)


@pytest.mark.parametrize('problem', ['endpoint', 'tariff', 'solar', 'origin', 'image'])
def test_economic_comparison_rejects_unmatched_conditions(problem):
    request = {'kind': 'dh', 'captured_at': 'historical', 'forecast_start': 'historical',
               'optimization_sha256': 'same', 'configuration': {},
               'payload': {'soc_final': .8, 'load_cost_forecast': [.2],
                           'pv_power_forecast': [1000.], 'load_power_forecast': [500.]}}
    baseline = {'image': 'same', 'request': request}
    challenger = deepcopy(baseline)
    if problem == 'endpoint': challenger['request']['payload']['soc_final'] = .7
    elif problem == 'tariff': challenger['request']['payload']['load_cost_forecast'] = [.3]
    elif problem == 'solar': challenger['request']['payload']['pv_power_forecast'] = [2000.]
    elif problem == 'origin': challenger['request']['forecast_start'] = 'different'
    else: challenger['image'] = 'different'
    with pytest.raises(ValueError): comparison(baseline, challenger)


def test_mutated_request_identity_rejected_before_docker(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('invalid request must not invoke Docker')
    monkeypatch.setattr('scripts.replay_energy_solves.subprocess.run', forbidden)
    with pytest.raises(ValueError, match='identity'):
        run_request({'payload': {'soc_final': .8}, 'request_id': 'stale'}, 'sha256:'+'a'*64)
