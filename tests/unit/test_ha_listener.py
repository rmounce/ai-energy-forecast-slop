import asyncio
from contextlib import suppress
from unittest.mock import patch

from services.ha_listener import Listener
import services.ha_listener as listener_module


class Proc:
    def __init__(self):
        self.returncode = None
        self.killed = False
        self.waited = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        self.waited = True
        return self.returncode


class Output:
    async def readline(self):
        return b""


class RunProc(Proc):
    def __init__(self, returncode=0):
        super().__init__()
        self.returncode = returncode
        self.stdout = Output()

    async def wait(self):
        self.waited = True
        return self.returncode


class HangingProc(Proc):
    def __init__(self):
        super().__init__()
        self.stdout = Output()
        self.started = asyncio.Event()
        self.terminated = asyncio.Event()

    def kill(self):
        super().kill()
        self.terminated.set()

    async def wait(self):
        self.waited = True
        self.started.set()
        if self.returncode is None:
            await self.terminated.wait()
        return self.returncode


def test_terminate_child_kills_reaps_and_drains():
    async def run():
        proc = Proc()
        stream = asyncio.create_task(asyncio.sleep(60))
        await Listener._terminate_child(proc, stream)
        assert proc.killed and proc.waited and stream.done()

    asyncio.run(run())


def test_failed_retry_has_bounded_deadline_and_coalesces_trigger():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        listener.trigger.set()
        listener._schedule_failure_retry()
        assert not listener.trigger.is_set()
        assert listener._retry_not_before > 0
        listener.shutdown.set()
        if listener._retry_task:
            listener._retry_task.cancel()
            try:
                await listener._retry_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run())


def test_success_updates_last_run_and_records_local_success():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        proc = RunProc(0)
        with patch.object(listener_module.asyncio, "create_subprocess_exec", return_value=proc), \
             patch.object(listener_module, "record_job_status") as record:
            await listener._run_predict_price()
        assert listener.last_run_at is not None
        record.assert_called_once_with("price-listener", 0)

    asyncio.run(run())


def test_nonzero_child_records_failure_and_schedules_retry():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        with patch.object(listener_module.asyncio, "create_subprocess_exec", return_value=RunProc(1)), \
             patch.object(listener_module, "record_job_status") as record:
            await listener._run_predict_price()
        assert listener.last_run_at is None
        record.assert_called_once_with("price-listener", 1)
        assert listener._retry_not_before > 0
        listener.shutdown.set()
        if listener._retry_task:
            listener._retry_task.cancel()
            with suppress(asyncio.CancelledError):
                await listener._retry_task

    asyncio.run(run())


def test_timeout_kills_child_records_failure_and_schedules_retry():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        proc = HangingProc()
        with patch.object(listener_module, "SUBPROCESS_TIMEOUT_SECONDS", 0.001), \
             patch.object(listener_module.asyncio, "create_subprocess_exec", return_value=proc), \
             patch.object(listener_module, "record_job_status") as record:
            await listener._run_predict_price()
        assert proc.killed and proc.waited
        record.assert_called_once_with("price-listener", 1)
        assert listener._retry_not_before > 0
        listener.shutdown.set()
        if listener._retry_task:
            listener._retry_task.cancel()
            with suppress(asyncio.CancelledError):
                await listener._retry_task

    asyncio.run(run())


def test_cancellation_kills_and_reaps_running_child():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        proc = HangingProc()
        with patch.object(listener_module.asyncio, "create_subprocess_exec", return_value=proc):
            task = asyncio.create_task(listener._run_predict_price())
            await proc.started.wait()
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        assert proc.killed and proc.waited

    asyncio.run(run())


def test_event_during_retry_wait_coalesces_with_timer_into_one_run():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        calls = []

        async def predicted_once():
            calls.append("run")
            listener.shutdown.set()

        with patch.object(listener_module, "DEBOUNCE_SECONDS", 0), \
             patch.object(listener_module, "FAILURE_RETRY_SECONDS", 0.02), \
             patch.object(listener, "_run_predict_price", side_effect=predicted_once):
            listener._schedule_failure_retry()
            worker = asyncio.create_task(listener.worker())
            await asyncio.sleep(0.005)
            listener.trigger.set()  # APF event before the timer deadline
            await asyncio.wait_for(worker, timeout=1)
        assert calls == ["run"]
        if listener._retry_task and not listener._retry_task.done():
            listener._retry_task.cancel()
            with suppress(asyncio.CancelledError):
                await listener._retry_task

    asyncio.run(run())


def test_event_during_child_creates_exactly_one_follow_up():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = []

        async def prediction():
            calls.append("run")
            if len(calls) == 1:
                first_started.set()
                await release_first.wait()
            else:
                listener.shutdown.set()

        with patch.object(listener_module, "DEBOUNCE_SECONDS", 0), \
             patch.object(listener, "_run_predict_price", side_effect=prediction):
            listener.trigger.set()
            worker = asyncio.create_task(listener.worker())
            await first_started.wait()
            listener.trigger.set()
            listener.trigger.set()
            release_first.set()
            await asyncio.wait_for(worker, timeout=1)
        assert calls == ["run", "run"]

    asyncio.run(run())
