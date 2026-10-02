from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import pytest
import forecast as fc
from energy_pipeline import price_worker
from energy_pipeline.model_cache import ModelCache
from energy_pipeline.source_cache import SourceCache, SourcePolicy
from production_contract import PredictionOutcome


@pytest.fixture
def world(tmp_path, monkeypatch):
    now = pd.Timestamp.now(tz='UTC')
    start = now.floor('30min')
    config = {'paths': {'tariff_file': 'tariff.json'}, 'home_assistant': {
        'amber_billing_entity': 'sensor.apf', 'solcast_entities': ['sensor.pv']}}
    monkeypatch.setattr(fc, 'CONFIG', config)
    monkeypatch.setattr(fc, 'ROOT', tmp_path)
    monkeypatch.setattr(price_worker, 'load_config', lambda: config)
    monkeypatch.setattr('tariff_utils.load_tariff_profile', lambda *args: ({}, {}, 1))
    pointer = tmp_path/'active.json'
    monkeypatch.setattr(fc, '_production_bundle_store', lambda: SimpleNamespace(active_pointer=lambda family: pointer))
    rows = [{'entity_id': 'sensor.apf', 'attributes': {'Forecasts': [1]}},
            {'entity_id': 'sensor.pv', 'attributes': {'detailedForecast': [1]}}]
    api = Mock(side_effect=lambda *args: rows)
    monkeypatch.setattr(fc, 'call_ha_api', api)
    index = pd.date_range(start, periods=144, freq='30min')
    forecasts = {name: pd.DataFrame({name: float(i)}, index=index)
                 for i, name in enumerate(('price_p30', 'price', 'price_p70'))}
    monkeypatch.setattr(fc, 'run_predictions', lambda *args, **kwargs: PredictionOutcome(
        'price', 'test', 'bundle', forecasts))
    frame = pd.DataFrame({'value': 1.}, index=pd.date_range(start, periods=150, freq='30min'))
    monkeypatch.setattr(fc, 'get_solcast_forecast', lambda: frame.rename(columns={'value': 'power_pv'}))
    sources = SourceCache({name: SourcePolicy(('value',), 300, 1800) for name in ('aemo', 'weather', 'history')})
    for name in sources.policies:
        sources.put(name, frame, now)
    worker = price_worker.PriceWorker(sources)
    model, params = tmp_path/'model.pkl', tmp_path/'params.json'
    model.write_text('model'); params.write_text('{}')
    worker.cache._signature = ModelCache.signature({'price': {'model': model, 'params': params}})
    result = worker.predict()
    return SimpleNamespace(worker=worker, result=result, rows=rows, sources=sources,
                           frame=frame, pointer=pointer, model=model, config=config,
                           now=pd.Timestamp(result.captured_at), tariff=tmp_path/'tariff.json', api=api)


def test_matching_inputs_are_accepted_without_refetching_cached_sources(world):
    decision = world.worker.evaluate_completion(world.result, now=world.now)
    assert decision.accepted
    assert world.api.call_count == 2  # one generation snapshot; one acceptance read


@pytest.mark.parametrize('change', ['apf', 'pv', 'source', 'tariff', 'pointer', 'model', 'config'])
def test_changed_parent_is_rejected(world, change):
    if change == 'apf': world.rows[0]['attributes']['Forecasts'] = [2]
    elif change == 'pv': world.rows[1]['attributes']['detailedForecast'] = [2]
    elif change == 'source': world.sources.put('aemo', world.frame*2, world.now)
    elif change == 'tariff': world.tariff.write_text('{}')
    elif change == 'pointer': world.pointer.write_text('{"bundle_id":"new"}')
    elif change == 'model': world.model.write_text('replacement model')
    elif change == 'config': world.config['new_setting'] = True
    decision = world.worker.evaluate_completion(world.result, now=world.now)
    assert not decision.accepted
    assert any(reason.startswith('input_changed:') for reason in decision.reasons)
    assert decision.reconcile == (change != 'config')


def test_interval_rollover_discards_previous_surface(world):
    boundary = world.now.floor('30min')+pd.Timedelta(minutes=30)
    # Capture just before boundary to distinguish rollover from completion age.
    world.result = SimpleNamespace(**{**world.result.__dict__, 'captured_at': (boundary-pd.Timedelta(seconds=1)).isoformat()})
    decision = world.worker.evaluate_completion(world.result, now=boundary)
    assert decision.reasons == ('forecast_interval_changed',)
    assert decision.reconcile


def test_source_expiry_waits_for_refresh_and_invalid_surface_is_rejected(world):
    world.sources.policies['aemo'] = SourcePolicy(('value',), 300, .01)
    decision = world.worker.evaluate_completion(world.result, now=world.now+pd.Timedelta(seconds=1))
    assert decision.reasons[0].startswith('inputs_unavailable:')
    assert not decision.reconcile
    world.result.outcome.forecasts['price'].iloc[0, 0] = float('nan')
    decision = world.worker.evaluate_completion(world.result, now=world.now)
    assert decision.reasons[0].startswith('forecast_invalid:')


def test_completion_age_and_failed_validation_read_cannot_be_accepted(world):
    assert not world.worker.evaluate_completion(world.result, now=world.now+pd.Timedelta(seconds=181)).accepted
    world.api.side_effect = lambda *args: None
    with pytest.raises(RuntimeError, match='verify current HA'):
        world.worker.evaluate_completion(world.result, now=world.now)


def test_acceptance_read_crossing_boundary_cannot_accept_old_interval(world, monkeypatch):
    boundary = world.now.floor('30min')+pd.Timedelta(minutes=30)
    before = boundary-pd.Timedelta(seconds=1)
    result = SimpleNamespace(**{**world.result.__dict__, 'captured_at': before.isoformat()})
    clock = iter((before, boundary))
    monkeypatch.setattr(price_worker, 'utc_now', lambda: next(clock))
    decision = world.worker.evaluate_completion(result)
    assert decision.reasons == ('forecast_interval_changed',)
