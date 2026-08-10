import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, patch

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


def test_success_updates_last_run_and_pings_once():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        proc = RunProc(0)
        with patch.object(listener_module.asyncio, "create_subprocess_exec", return_value=proc), \
             patch.object(listener, "_ping_healthcheck", new_callable=AsyncMock) as ping:
            await listener._run_predict_price()
        assert listener.last_run_at is not None
        ping.assert_awaited_once()

    asyncio.run(run())


def test_nonzero_child_does_not_ping_and_schedules_retry():
    async def run():
        listener = Listener({"home_assistant": {
            "url": "http://ha", "token": "x", "amber_billing_entity": "sensor.x",
        }})
        with patch.object(listener_module.asyncio, "create_subprocess_exec", return_value=RunProc(1)), \
             patch.object(listener, "_ping_healthcheck", new_callable=AsyncMock) as ping:
            await listener._run_predict_price()
        assert listener.last_run_at is None
        ping.assert_not_awaited()
        assert listener._retry_not_before > 0
        listener.shutdown.set()
        if listener._retry_task:
            listener._retry_task.cancel()
            with suppress(asyncio.CancelledError):
                await listener._retry_task

    asyncio.run(run())
