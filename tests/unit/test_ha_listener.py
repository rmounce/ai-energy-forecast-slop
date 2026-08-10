import asyncio

from services.ha_listener import Listener


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
