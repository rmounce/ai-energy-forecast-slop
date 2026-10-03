import asyncio
from contextlib import suppress
from types import SimpleNamespace
import threading
from unittest.mock import patch
import pytest
from services.resident_price import ResidentPriceListener, WorkerTimeout
import services.resident_price as resident
import services.ha_listener as legacy
from energy_pipeline.acceptance import AcceptanceDecision


class HealthyPredictor:
    def evaluate_completion(self, result):
        return AcceptanceDecision()

CONFIG = {'home_assistant': {'url': 'http://ha', 'token': 'x', 'amber_billing_entity': 'sensor.apf'}}


async def wait_until(condition):
    async def poll():
        while not condition():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 2)


def completion():
    return SimpleNamespace(outcome=SimpleNamespace(run_id='run'), parent_revision='parent', rss_mib=10)


def test_worker_runs_in_one_thread_without_blocking_event_loop():
    async def run():
        calls = []
        started, release = threading.Event(), threading.Event()
        class Predictor(HealthyPredictor):
            def predict(self):
                calls.append(threading.get_ident())
                if len(calls) == 1:
                    started.set()
                    assert release.wait(2)
                return completion()
        listener = ResidentPriceListener(CONFIG, Predictor())
        try:
            task = asyncio.create_task(listener._run_predict_price())
            await wait_until(started.is_set)
            assert listener.completed is None
            assert calls[0] != threading.get_ident()
            release.set()
            await task
            await listener._run_predict_price()
            assert len(set(calls)) == 1
            assert listener.completed.parent_revision == 'parent'
            assert listener.last_run_at is not None
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_timeout_discards_late_completion_and_stops_scheduler():
    async def run():
        release = threading.Event()
        class Predictor(HealthyPredictor):
            def predict(self):
                release.wait(2)
                return completion()
        listener = ResidentPriceListener(CONFIG, Predictor())
        try:
            with patch.object(resident, 'WORKER_TIMEOUT_SECONDS', .01):
                with pytest.raises(WorkerTimeout):
                    await listener._run_predict_price()
            assert listener.shutdown.is_set()
            assert listener.completed is None
            assert listener.last_run_at is None
            assert listener._retry_task is None
            release.set()
            await asyncio.sleep(.01)
            assert listener.completed is None
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_failure_retries_without_touching_production_health():
    async def run():
        class Predictor(HealthyPredictor):
            def predict(self):
                raise ValueError('bad input')
        listener = ResidentPriceListener(CONFIG, Predictor())
        try:
            with patch.object(legacy, 'record_job_status') as health:
                await listener._run_predict_price()
                health.assert_not_called()
            assert listener._retry_not_before > 0
            assert listener.last_run_at is None
        finally:
            listener.shutdown.set()
            if listener._retry_task:
                listener._retry_task.cancel()
                with suppress(asyncio.CancelledError):
                    await listener._retry_task
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_burst_during_work_coalesces_to_one_followup():
    async def run():
        started, release = threading.Event(), threading.Event()
        calls = []
        class Predictor(HealthyPredictor):
            def predict(self):
                calls.append(1)
                if len(calls) == 1:
                    started.set()
                    assert release.wait(2)
                return completion()
        listener = ResidentPriceListener(CONFIG, Predictor())
        try:
            with patch.object(legacy, 'DEBOUNCE_SECONDS', 0):
                listener.trigger.set()
                task = asyncio.create_task(listener.worker())
                await wait_until(started.is_set)
                for _ in range(100):
                    listener.trigger.set()
                release.set()
                await wait_until(lambda: len(calls) >= 2 and not listener.run_lock.locked())
                listener.shutdown.set()
                await asyncio.wait_for(task, 1)
            assert len(calls) == 2
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_subscription_reconciles_latest_state_after_connect():
    class WS:
        async def send(self, message):
            pass
        async def recv(self):
            return '{"success": true}'
    async def run():
        listener = ResidentPriceListener(CONFIG, SimpleNamespace(predict=lambda: completion(), evaluate_completion=lambda result: AcceptanceDecision()))
        try:
            await listener._subscribe_state_changed(WS())
            assert listener.trigger.is_set()
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_shutdown_cancels_idle_ingress_without_waiting_for_another_event():
    async def run():
        listener = ResidentPriceListener(CONFIG, SimpleNamespace(predict=lambda: completion(), evaluate_completion=lambda result: AcceptanceDecision()))
        started = asyncio.Event()
        async def ingress():
            started.set()
            await asyncio.Event().wait()
        with patch.object(listener, 'consume_websocket', side_effect=ingress):
            task = asyncio.create_task(listener.run())
            await started.wait()
            listener.shutdown.set()
            await asyncio.wait_for(task, 1)
    asyncio.run(run())


def test_memory_budget_stops_worker_without_accepting_result():
    async def run():
        class Predictor(HealthyPredictor):
            def predict(self):
                result = completion()
                result.rss_mib = 3000
                return result
        listener = ResidentPriceListener(CONFIG, Predictor())
        try:
            with pytest.raises(resident.WorkerResourceLimit):
                await listener._run_predict_price()
            assert listener.shutdown.is_set()
            assert listener.completed is None
            assert listener._retry_task is None
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_obsolete_completion_keeps_previous_and_arms_one_replacement():
    async def run():
        predictor = SimpleNamespace(predict=completion, evaluate_completion=lambda result:
                                    AcceptanceDecision(('input_changed:source:aemo',), True))
        listener = ResidentPriceListener(CONFIG, predictor)
        previous = completion()
        listener.completed, listener.last_run_at = previous, 123
        try:
            await listener._run_predict_price()
            assert listener.completed is previous and listener.last_run_at == 123
            assert listener.trigger.is_set()
            assert listener._retry_task is None
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


@pytest.mark.parametrize('reconnect', [False, True])
def test_ha_change_or_reconnect_during_acceptance_discards_result(reconnect):
    async def run():
        started, release = threading.Event(), threading.Event()
        def evaluate(result):
            started.set()
            release.wait(2)
            return AcceptanceDecision()
        listener = ResidentPriceListener(CONFIG, SimpleNamespace(predict=completion, evaluate_completion=evaluate))
        class WS:
            async def send(self, message): pass
            async def recv(self): return '{"success": true}'
        try:
            task = asyncio.create_task(listener._run_predict_price())
            await wait_until(started.is_set)
            if reconnect:
                await listener._subscribe_state_changed(WS())
            else:
                listener._on_state_changed({'entity_id': 'sensor.apf'})
            release.set()
            await task
            assert listener.completed is None and listener.last_run_at is None
            assert listener.last_decision.reasons == ('ha_changed_during_acceptance',)
            assert listener.trigger.is_set()
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_acceptance_timeout_never_commits_late_result():
    async def run():
        release = threading.Event()
        def evaluate(result):
            release.wait(2)
            return AcceptanceDecision()
        listener = ResidentPriceListener(CONFIG, SimpleNamespace(predict=completion, evaluate_completion=evaluate))
        try:
            with patch.object(resident, 'WORKER_TIMEOUT_SECONDS', .02):
                with pytest.raises(WorkerTimeout):
                    await listener._run_predict_price()
            release.set()
            await asyncio.sleep(.02)
            assert listener.shutdown.is_set() and listener.completed is None
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_only_consumed_ha_entities_trigger_reconciliation():
    listener = ResidentPriceListener({'home_assistant': {**CONFIG['home_assistant'],
        'solcast_entities': ['sensor.pv'], 'solcast_last_polled_entity': 'sensor.pv_poll'}}, HealthyPredictor())
    try:
        listener._on_state_changed({'entity_id': 'sensor.unrelated'})
        assert not listener.trigger.is_set()
        for entity in ('sensor.apf', 'sensor.pv', 'sensor.pv_poll'):
            listener._on_state_changed({'entity_id': entity})
        assert listener.trigger.is_set() and listener._input_generation == 3
    finally:
        listener.executor.shutdown(wait=True)


def test_source_change_during_acceptance_discards_result():
    async def run():
        started, release = threading.Event(), threading.Event()
        sources = SimpleNamespace(generation=1)
        def evaluate(result):
            started.set()
            release.wait(2)
            return AcceptanceDecision()
        listener = ResidentPriceListener(CONFIG, SimpleNamespace(predict=completion, sources=sources,
                                                                 evaluate_completion=evaluate))
        listener.refreshers = SimpleNamespace()  # emulate owned cache tracking
        try:
            task = asyncio.create_task(listener._run_predict_price())
            await wait_until(started.is_set)
            sources.generation += 1
            release.set()
            await task
            assert listener.completed is None
            assert listener.last_decision.reasons == ('source_changed_during_acceptance',)
            assert listener.trigger.is_set()
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_recovery_does_not_restore_heartbeat_or_completed_forecast():
    recovered = SimpleNamespace(payload={'run_id': 'old'}, time_current=lambda: False)
    store = SimpleNamespace(recover=lambda: recovered)
    listener = ResidentPriceListener(CONFIG, HealthyPredictor(), store=store)
    try:
        assert listener.recovered is recovered
        assert listener.completed is None and listener.last_run_at is None
    finally:
        listener.executor.shutdown(wait=True)


@pytest.mark.parametrize('accepted', [True, False])
def test_only_accepted_results_are_persisted_before_heartbeat(accepted):
    async def run():
        saved = []
        listener = None
        def save(result, timestamp):
            assert listener.completed is None and listener.last_run_at is None
            saved.append(result)
        store = SimpleNamespace(recover=lambda: None, save=save)
        predictor = SimpleNamespace(predict=completion, evaluate_completion=lambda result:
            AcceptanceDecision() if accepted else AcceptanceDecision(('changed',), True))
        listener = ResidentPriceListener(CONFIG, predictor, store=store)
        try:
            await listener._run_predict_price()
            assert len(saved) == int(accepted)
            assert (listener.completed is not None) == accepted
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_storage_failure_stops_without_advancing_heartbeat():
    from energy_pipeline.accepted_store import StoreError
    async def run():
        def save(*args): raise StoreError('disk failure')
        listener = ResidentPriceListener(CONFIG,
            SimpleNamespace(predict=completion, evaluate_completion=lambda result: AcceptanceDecision()),
            store=SimpleNamespace(recover=lambda: None, save=save))
        try:
            with pytest.raises(StoreError): await listener._run_predict_price()
            assert listener.shutdown.is_set()
            assert listener.completed is None and listener.last_run_at is None
            assert listener._retry_task is None
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_storage_timeout_never_restores_in_memory_acceptance():
    async def run():
        release, committed = threading.Event(), threading.Event()
        def save(*args):
            release.wait(2)
            committed.set()
        listener = ResidentPriceListener(CONFIG,
            SimpleNamespace(predict=completion, evaluate_completion=lambda result: AcceptanceDecision()),
            store=SimpleNamespace(recover=lambda: None, save=save))
        try:
            with patch.object(resident, 'WORKER_TIMEOUT_SECONDS', .02):
                with pytest.raises(WorkerTimeout): await listener._run_predict_price()
            release.set()
            await wait_until(committed.is_set)
            assert listener.shutdown.is_set() and listener.completed is None
            assert listener.last_run_at is None
        finally:
            release.set()
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


@pytest.mark.parametrize('change', ['ha', 'source', 'expiry'])
def test_checkpoint_race_keeps_saved_record_historical_without_advancing(change):
    async def run():
        saved = []
        sources = SimpleNamespace(generation=1)
        listener = None
        def save(result, timestamp):
            saved.append(result)
            if change == 'ha': listener._input_generation += 1
            if change == 'source': sources.generation += 1
            return SimpleNamespace(time_current=lambda: change != 'expiry')
        listener = ResidentPriceListener(CONFIG,
            SimpleNamespace(predict=completion, sources=sources,
                            evaluate_completion=lambda result: AcceptanceDecision()),
            store=SimpleNamespace(recover=lambda: None, save=save))
        listener.refreshers = SimpleNamespace()
        try:
            await listener._run_predict_price()
            assert len(saved) == 1
            assert listener.completed is None and listener.last_run_at is None
            assert listener.trigger.is_set()
            assert not listener.last_decision.accepted
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


@pytest.mark.parametrize('outcome', ['commit', 'superseded', 'config_change', 'failure'])
def test_local_publication_gates_heartbeat_and_preserves_validation_decisions(monkeypatch, outcome):
    from energy_pipeline.accepted_store import StoreError
    async def run():
        validations = []
        def evaluate(result):
            validations.append(1)
            if outcome == 'config_change' and len(validations) > 1:
                return AcceptanceDecision(('input_changed:config',), False)
            return AcceptanceDecision()
        saved = SimpleNamespace(time_current=lambda: True)
        def execute(plan, revalidate):
            assert listener.completed is None and listener.last_run_at is None
            if outcome == 'failure': raise StoreError('injected journal error')
            return revalidate() and outcome == 'commit'
        publication = SimpleNamespace(abandon_pending=lambda: 0, execute=execute)
        store = SimpleNamespace(recover=lambda: None, save=lambda *args: saved)
        monkeypatch.setattr(resident, 'prepare_plan', lambda *args: {})
        listener = ResidentPriceListener(CONFIG,
            SimpleNamespace(predict=completion, evaluate_completion=evaluate), store=store, publication=publication)
        try:
            if outcome == 'failure':
                with pytest.raises(StoreError): await listener._run_predict_price()
                assert listener.shutdown.is_set()
            else:
                await listener._run_predict_price()
                if outcome == 'config_change':
                    assert listener.last_decision.reasons == ('input_changed:config',)
                    assert not listener.trigger.is_set() and listener._retry_task is None
            assert (listener.completed is not None) == (outcome == 'commit')
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_handoff_uses_selected_frozen_states_and_saves_before_heartbeat(monkeypatch):
    import forecast as fc
    async def run():
        captured = []
        record = {'readiness': {'dh': {'coverage_ready': True}}}
        def build(plan, snapshot):
            captured.append(snapshot)
            return record
        monkeypatch.setattr(resident, 'prepare_plan', lambda *args: {'id': 'local'})
        monkeypatch.setattr(resident, 'build_handoff', build)
        monkeypatch.setattr(fc, 'call_ha_api', lambda method, endpoint: [
            {'entity_id': 'sensor.sigen_plant_rated_energy_capacity', 'state': '50', 'attributes': {}},
            {'entity_id': 'sensor.unrelated', 'state': 'secret', 'attributes': {}}])
        saved = SimpleNamespace(time_current=lambda: True)
        persisted = []
        def save(plan, result):
            assert listener.completed is None and listener.last_run_at is None
            persisted.append(result)
        publication = SimpleNamespace(abandon_pending=lambda: 0,
            execute=lambda plan, valid: valid(), save_handoff=save)
        listener = ResidentPriceListener({**CONFIG, 'timezone': 'Australia/Adelaide'},
            SimpleNamespace(predict=completion, evaluate_completion=lambda result: AcceptanceDecision()),
            store=SimpleNamespace(recover=lambda: None, save=lambda *args: saved),
            publication=publication, handoff=True)
        try:
            await listener._run_predict_price()
            assert persisted == [record]
            assert set(captured[0]['states']) == {'sensor.sigen_plant_rated_energy_capacity'}
            assert captured[0]['capture_started_at'] <= captured[0]['captured_at']
            assert listener.completed is not None
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())
