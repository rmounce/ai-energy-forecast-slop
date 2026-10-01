import asyncio
from contextlib import suppress
from types import SimpleNamespace
import threading
from unittest.mock import patch
import pytest
from services.resident_price import ResidentPriceListener, WorkerTimeout
import services.resident_price as resident
import services.ha_listener as legacy

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
        class Predictor:
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
        class Predictor:
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
        class Predictor:
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
        class Predictor:
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
        listener = ResidentPriceListener(CONFIG)
        try:
            await listener._subscribe_state_changed(WS())
            assert listener.trigger.is_set()
        finally:
            listener.executor.shutdown(wait=True)
    asyncio.run(run())


def test_shutdown_cancels_idle_ingress_without_waiting_for_another_event():
    async def run():
        listener = ResidentPriceListener(CONFIG)
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
        class Predictor:
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
