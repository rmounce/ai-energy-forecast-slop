#!/usr/bin/env python3
"""Event-driven trigger for `forecast.py predict-price` on Amber APF state change.

Subscribes to HA's WebSocket API, listens for state_changed events on the
configured `amber_billing_entity`, debounces a short window to coalesce
near-simultaneous updates, and shells out to `predict-price`. A 30-minute
idle heartbeat fires the same command even if no event has arrived — the
defensive fallback the user asked for in the design plan.

Design: docs/event_driven_predict_price_plan.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import requests
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

# Make repo root importable so we can reuse config_utils.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from config_utils import load_config  # noqa: E402

DEBOUNCE_SECONDS = 1.0
HEARTBEAT_SECONDS = 30 * 60
HEARTBEAT_POLL_SECONDS = 30
SUBPROCESS_TIMEOUT_SECONDS = 120
RECONNECT_BACKOFF_INITIAL = 1
RECONNECT_BACKOFF_CAP = 30
HEALTHCHECK_TIMEOUT = 5

PREDICT_PRICE_CMD = [
    str(REPO_ROOT / ".venv" / "bin" / "python"),
    str(REPO_ROOT / "forecast.py"),
    "predict-price",
    "--dynamic-handoff",
    "--publish-hass",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("ha_listener")


class Listener:
    def __init__(self, config: dict):
        ha = config["home_assistant"]
        self.ws_url = ha["url"].replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
        self.token = ha["token"]
        self.entity_id = ha["amber_billing_entity"]
        self.healthcheck_url = os.environ.get("HC_PREDICT_URL")
        self.trigger = asyncio.Event()
        self.run_lock = asyncio.Lock()
        self.last_run_at = 0.0  # monotonic
        self.shutdown = asyncio.Event()
        self._next_msg_id = 1

    def _msg_id(self) -> int:
        n = self._next_msg_id
        self._next_msg_id += 1
        return n

    # ---- WebSocket loop ----

    async def consume_websocket(self) -> None:
        backoff = RECONNECT_BACKOFF_INITIAL
        while not self.shutdown.is_set():
            try:
                log.info("Connecting to %s", self.ws_url)
                async with websockets.connect(self.ws_url, ping_interval=30, ping_timeout=10) as ws:
                    await self._authenticate(ws)
                    await self._subscribe_state_changed(ws)
                    log.info("Subscribed to state_changed; filtering for %s", self.entity_id)
                    backoff = RECONNECT_BACKOFF_INITIAL
                    await self._read_events(ws)
            except (ConnectionClosed, OSError, InvalidStatus, asyncio.TimeoutError) as e:
                log.warning("WebSocket error: %s; reconnecting in %ss", e, backoff)
            except Exception:
                log.exception("Unexpected WebSocket failure; reconnecting in %ss", backoff)
            if self.shutdown.is_set():
                break
            try:
                await asyncio.wait_for(self.shutdown.wait(), timeout=backoff)
                break  # shutdown signalled during backoff
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, RECONNECT_BACKOFF_CAP)

    async def _authenticate(self, ws) -> None:
        hello = json.loads(await ws.recv())
        if hello.get("type") != "auth_required":
            raise RuntimeError(f"Unexpected first message: {hello}")
        await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
        reply = json.loads(await ws.recv())
        if reply.get("type") != "auth_ok":
            raise RuntimeError(f"HA auth failed: {reply}")
        log.info("HA WebSocket auth OK (version=%s)", reply.get("ha_version"))

    async def _subscribe_state_changed(self, ws) -> None:
        sub_id = self._msg_id()
        await ws.send(json.dumps({
            "id": sub_id,
            "type": "subscribe_events",
            "event_type": "state_changed",
        }))
        reply = json.loads(await ws.recv())
        if not reply.get("success"):
            raise RuntimeError(f"subscribe_events failed: {reply}")

    async def _read_events(self, ws) -> None:
        async for raw in ws:
            if self.shutdown.is_set():
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                log.warning("Non-JSON message dropped")
                continue
            if msg.get("type") != "event":
                continue
            event = msg.get("event", {})
            data = event.get("data", {})
            if data.get("entity_id") != self.entity_id:
                continue
            # We don't compare old/new state — any change to the entity
            # (including attribute-only changes) is a signal that Amber
            # published fresh APF.
            log.debug("APF entity state_changed; arming trigger")
            self.trigger.set()

    # ---- Worker (debounce + subprocess) ----

    async def worker(self) -> None:
        while not self.shutdown.is_set():
            await self._wait_for_trigger_or_shutdown()
            if self.shutdown.is_set():
                return
            self.trigger.clear()
            # Debounce: hold for a moment so a burst of events coalesces.
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.shutdown.wait(), timeout=DEBOUNCE_SECONDS)
            if self.shutdown.is_set():
                return
            # Any extra events that arrived during the debounce are
            # satisfied by this run; clear before firing.
            self.trigger.clear()
            await self._run_predict_price()

    async def _wait_for_trigger_or_shutdown(self) -> None:
        trig = asyncio.create_task(self.trigger.wait())
        stop = asyncio.create_task(self.shutdown.wait())
        try:
            await asyncio.wait({trig, stop}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (trig, stop):
                if not t.done():
                    t.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await t

    async def _run_predict_price(self) -> None:
        async with self.run_lock:
            started = time.monotonic()
            log.info("Running: %s", " ".join(PREDICT_PRICE_CMD))
            try:
                # PYTHONUNBUFFERED=1 so the child's stdout is line-buffered
                # when piped; without it Python block-buffers and we only
                # see output at process exit.
                env = {**os.environ, "PYTHONUNBUFFERED": "1"}
                proc = await asyncio.create_subprocess_exec(
                    *PREDICT_PRICE_CMD,
                    cwd=str(REPO_ROOT),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                )
                stream_task = asyncio.create_task(self._stream_subprocess_output(proc))
                try:
                    await asyncio.wait_for(proc.wait(), timeout=SUBPROCESS_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    log.error("predict-price exceeded %ss; killing", SUBPROCESS_TIMEOUT_SECONDS)
                    proc.kill()
                    await proc.wait()
                    stream_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await stream_task
                    return
                # Drain any final lines buffered after process exit.
                await stream_task
                elapsed = time.monotonic() - started
                self.last_run_at = time.monotonic()
                if proc.returncode == 0:
                    log.info("predict-price succeeded in %.1fs", elapsed)
                    await self._ping_healthcheck()
                else:
                    log.error("predict-price failed rc=%s after %.1fs", proc.returncode, elapsed)
            except Exception:
                log.exception("predict-price subprocess raised")

    async def _stream_subprocess_output(self, proc) -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            log.info("[predict-price] %s", line.rstrip(b"\n").decode(errors="replace"))

    async def _ping_healthcheck(self) -> None:
        if not self.healthcheck_url:
            return
        try:
            await asyncio.to_thread(
                requests.get, self.healthcheck_url, timeout=HEALTHCHECK_TIMEOUT
            )
        except Exception as e:
            log.warning("Healthcheck ping failed (non-fatal): %s", e)

    # ---- Heartbeat ----

    async def heartbeat(self) -> None:
        # Mark startup so the first heartbeat-driven run isn't immediate;
        # the WebSocket will usually deliver an event well within the
        # heartbeat window in normal operation.
        self.last_run_at = time.monotonic()
        while not self.shutdown.is_set():
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.shutdown.wait(), timeout=HEARTBEAT_POLL_SECONDS)
            if self.shutdown.is_set():
                return
            idle = time.monotonic() - self.last_run_at
            if idle >= HEARTBEAT_SECONDS:
                log.warning(
                    "No predict-price run in %.0fs (>= %ss heartbeat); firing fallback",
                    idle, HEARTBEAT_SECONDS,
                )
                self.trigger.set()

    # ---- Orchestration ----

    async def run(self) -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.consume_websocket())
            tg.create_task(self.worker())
            tg.create_task(self.heartbeat())


def _install_signal_handlers(listener: Listener, loop: asyncio.AbstractEventLoop) -> None:
    def _signal():
        log.info("Shutdown signal received")
        listener.shutdown.set()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal)


def main() -> int:
    config = load_config(str(REPO_ROOT / "config.json"))
    if not config["home_assistant"].get("token"):
        log.error("home_assistant.token missing (config.secrets.json)")
        return 2
    listener = Listener(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(listener, loop)
    rc = 0
    try:
        loop.run_until_complete(listener.run())
    except* Exception as eg:  # TaskGroup raises ExceptionGroup
        for e in eg.exceptions:
            log.exception("Listener task failed", exc_info=e)
        rc = 1
    finally:
        loop.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
