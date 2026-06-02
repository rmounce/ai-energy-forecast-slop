#!/usr/bin/env python3
"""Event-driven HWC planner/executor daemon.

This replaces the eventual shape of separate planner/executor timers: it watches Home
Assistant for HWC-relevant state changes, replans when inputs change, and runs the
execution decision loop on both events and a short periodic cadence.

Actuation remains gated by ``hwc.actuation.enabled`` in ``config.json``. With that flag
false, the daemon can publish updated plans but the executor will not call water_heater
services.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import signal
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import hwc_executor  # noqa: E402
import hwc_planner  # noqa: E402
from config_utils import load_config  # noqa: E402

RECONNECT_BACKOFF_INITIAL = 1
RECONNECT_BACKOFF_CAP = 30

log = logging.getLogger("hwc_daemon")


@dataclass(frozen=True)
class TriggerDecision:
    replan: bool
    execute: bool
    reason: str


def _published_entity_id(prefix: str, entity_id: str) -> str:
    domain, object_id = entity_id.split(".", 1)
    return f"{domain}.{prefix}{object_id}"


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = (int(x) for x in value.split(":"))
    return hour, minute


def _daemon_state_path(config: dict) -> Path:
    configured = config["hwc"].get("daemon", {}).get("state_file", "data/hwc_daemon_state.json")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_event_time_utc(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def main_window_overlap_dates(config: dict, start_utc: datetime, end_utc: datetime) -> set[str]:
    if end_utc <= start_utc:
        return set()

    hwc = config["hwc"]
    block_cfg = hwc.get("block_planner", {})
    tz = ZoneInfo(config["timezone"])
    start_hour, start_minute = _parse_hhmm(block_cfg.get("main_window_start", "10:00"))
    end_hour, end_minute = _parse_hhmm(block_cfg.get("main_window_end", "18:00"))

    local_start = start_utc.astimezone(tz)
    local_end = end_utc.astimezone(tz)
    cursor = local_start.date()
    end_date = local_end.date()
    out: set[str] = set()
    while cursor <= end_date:
        window_start = datetime.combine(cursor, datetime_time(start_hour, start_minute), tz)
        window_end = datetime.combine(cursor, datetime_time(end_hour, end_minute), tz)
        if window_end <= window_start:
            window_end += timedelta(days=1)
        overlap_start = max(start_utc, window_start.astimezone(timezone.utc))
        overlap_end = min(end_utc, window_end.astimezone(timezone.utc))
        if overlap_start < overlap_end:
            out.add(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def watched_entities(config: dict) -> set[str]:
    hwc = config["hwc"]
    ha = config["home_assistant"]
    act = hwc.get("actuation", {})
    prefix = hwc.get("publish_prefix", "hwc_")

    entities = {
        hwc["import_price_entity"],
        hwc["tank_temp_entity"],
        ha["weather_entity"],
        _published_entity_id(prefix, hwc["power_plan_entity"]),
        _published_entity_id(prefix, hwc["predicted_temp_entity"]),
    }
    for key in ("water_heater_entity", "compressor_entity"):
        if act.get(key):
            entities.add(act[key])
    return entities


def _state_float(state: dict | None) -> float | None:
    if not state:
        return None
    try:
        return float(state.get("state"))
    except (TypeError, ValueError):
        return None


def classify_state_change(config: dict, entity_id: str, old_state: dict | None, new_state: dict | None) -> TriggerDecision:
    hwc = config["hwc"]
    act = hwc.get("actuation", {})
    daemon = hwc.get("daemon", {})
    prefix = hwc.get("publish_prefix", "hwc_")
    plan_entities = {
        _published_entity_id(prefix, hwc["power_plan_entity"]),
        _published_entity_id(prefix, hwc["predicted_temp_entity"]),
    }

    if entity_id == hwc["tank_temp_entity"]:
        old = _state_float(old_state)
        new = _state_float(new_state)
        min_delta = float(daemon.get("tank_temp_replan_delta_c", 0.3))
        if old is None or new is None or abs(new - old) >= min_delta:
            return TriggerDecision(True, False, "tank temperature changed")
        return TriggerDecision(False, False, "tank temperature change below threshold")

    if entity_id in {hwc["import_price_entity"], config["home_assistant"]["weather_entity"]}:
        return TriggerDecision(True, False, "forecast input changed")

    if entity_id in {act.get("water_heater_entity"), act.get("compressor_entity")}:
        return TriggerDecision(False, True, "equipment state changed")

    if entity_id in plan_entities:
        return TriggerDecision(False, True, "published plan changed")

    return TriggerDecision(False, False, "not watched")


def should_suppress_off_after_heat(
    *,
    decision_action: str,
    now: float,
    last_heat_command_at: float,
    grace_seconds: float,
    compressor_seen_on_since_heat: bool,
) -> bool:
    if decision_action != "off":
        return False
    if compressor_seen_on_since_heat:
        return False
    if last_heat_command_at <= 0:
        return False
    return now - last_heat_command_at < grace_seconds


class HwcDaemon:
    def __init__(self, config: dict, *, dry_run: bool = False):
        ha = config["home_assistant"]
        self.config = config
        self.dry_run = dry_run
        self.ws_url = ha["url"].replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
        self.token = ha["token"]
        self.entities = watched_entities(config)
        self.replan_trigger = asyncio.Event()
        self.execute_trigger = asyncio.Event()
        self.run_lock = asyncio.Lock()
        self.shutdown = asyncio.Event()
        self.started_at = time.monotonic()
        self.last_plan_at = 0.0
        self.last_heat_command_at = 0.0
        self.compressor_seen_on_since_heat = False
        self.compressor_on_since_utc: datetime | None = None
        self.completed_main_dates = self._load_state()
        self._next_msg_id = 1

    def _msg_id(self) -> int:
        n = self._next_msg_id
        self._next_msg_id += 1
        return n

    async def consume_websocket(self) -> None:
        backoff = RECONNECT_BACKOFF_INITIAL
        while not self.shutdown.is_set():
            try:
                log.info("Connecting to %s", self.ws_url)
                async with websockets.connect(self.ws_url, ping_interval=30, ping_timeout=10) as ws:
                    await self._authenticate(ws)
                    await self._subscribe_state_changed(ws)
                    log.info("Subscribed to state_changed; watching %s", ", ".join(sorted(self.entities)))
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
                break
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
            entity_id = data.get("entity_id")
            if entity_id not in self.entities:
                continue
            self._track_compressor_event(entity_id, data, _parse_event_time_utc(event.get("time_fired")))
            decision = classify_state_change(
                self.config,
                entity_id,
                data.get("old_state"),
                data.get("new_state"),
            )
            if decision.replan:
                log.info("%s: arming replan (%s)", entity_id, decision.reason)
                self.replan_trigger.set()
            if decision.execute:
                log.info("%s: arming executor (%s)", entity_id, decision.reason)
                self.execute_trigger.set()

    async def planning_worker(self) -> None:
        while not self.shutdown.is_set():
            await self._wait_for(self.replan_trigger)
            if self.shutdown.is_set():
                return
            self.replan_trigger.clear()
            await self._debounce()
            self.replan_trigger.clear()
            await self._respect_minimum_replan_interval()
            if self.shutdown.is_set():
                return
            await self._run_planner()
            self.execute_trigger.set()

    async def execution_worker(self) -> None:
        while not self.shutdown.is_set():
            await self._wait_for(self.execute_trigger)
            if self.shutdown.is_set():
                return
            self.execute_trigger.clear()
            await self._run_executor()

    async def periodic_execution(self) -> None:
        interval = float(self.config["hwc"].get("daemon", {}).get("execution_interval_seconds", 60))
        while not self.shutdown.is_set():
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.shutdown.wait(), timeout=interval)
            if self.shutdown.is_set():
                return
            self.execute_trigger.set()

    async def heartbeat(self) -> None:
        interval = float(self.config["hwc"].get("daemon", {}).get("heartbeat_seconds", 1800))
        while not self.shutdown.is_set():
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.shutdown.wait(), timeout=30)
            if self.shutdown.is_set():
                return
            reference = self.last_plan_at or self.started_at
            idle = time.monotonic() - reference
            if idle >= interval:
                log.warning("No HWC replan in %.0fs (>= %.0fs heartbeat); arming replan", idle, interval)
                self.replan_trigger.set()

    async def _wait_for(self, trigger: asyncio.Event) -> None:
        trig = asyncio.create_task(trigger.wait())
        stop = asyncio.create_task(self.shutdown.wait())
        try:
            await asyncio.wait({trig, stop}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (trig, stop):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await task

    async def _debounce(self) -> None:
        seconds = float(self.config["hwc"].get("daemon", {}).get("debounce_seconds", 2))
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.shutdown.wait(), timeout=seconds)

    async def _respect_minimum_replan_interval(self) -> None:
        minimum = float(self.config["hwc"].get("daemon", {}).get("minimum_replan_interval_seconds", 60))
        if self.last_plan_at <= 0:
            return
        remaining = minimum - (time.monotonic() - self.last_plan_at)
        if remaining <= 0:
            return
        log.info("Delaying HWC replan %.1fs to respect minimum interval", remaining)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.shutdown.wait(), timeout=remaining)

    async def _run_planner(self) -> None:
        async with self.run_lock:
            started = time.monotonic()
            horizon = int(self.config["hwc"].get("horizon_steps", 72))
            planner_config = copy.deepcopy(self.config)
            planner_config["hwc"].setdefault("block_planner", {})["completed_main_dates"] = sorted(
                self.completed_main_dates
            )
            try:
                await asyncio.to_thread(hwc_planner.run, planner_config, horizon, self.dry_run)
            except Exception:
                log.exception("HWC planner failed")
                return
            self.last_plan_at = time.monotonic()
            log.info("HWC planner completed in %.1fs", self.last_plan_at - started)

    async def _run_executor(self) -> None:
        async with self.run_lock:
            started = time.monotonic()
            try:
                decision = await asyncio.to_thread(hwc_executor.decide_current, self.config)
            except Exception:
                log.exception("HWC executor failed")
                return
            log.info(
                "HWC executor decision: %s (%s), setpoint=%s",
                decision.action,
                decision.reason,
                decision.setpoint_c,
            )

            if self._should_suppress_off_after_heat(decision):
                log.warning(
                    "Suppressing HWC off command %.1fs after heat command",
                    time.monotonic() - self.last_heat_command_at,
                )
                return

            act = self.config["hwc"].get("actuation", {})
            if self.dry_run or not act.get("enabled", False):
                log.info("Dry/config-disabled run; not calling water_heater services")
            else:
                try:
                    await asyncio.to_thread(hwc_executor.apply_decision, self.config, decision)
                except Exception:
                    log.exception("HWC executor apply failed")
                    return
                if decision.action == "heat":
                    self.last_heat_command_at = time.monotonic()
                    self.compressor_seen_on_since_heat = False
            log.info("HWC executor completed in %.1fs: %s", time.monotonic() - started, decision.action)

    def _should_suppress_off_after_heat(self, decision: hwc_executor.Decision) -> bool:
        grace = float(self.config["hwc"].get("daemon", {}).get("heat_command_grace_seconds", 600))
        return should_suppress_off_after_heat(
            decision_action=decision.action,
            now=time.monotonic(),
            last_heat_command_at=self.last_heat_command_at,
            grace_seconds=grace,
            compressor_seen_on_since_heat=self.compressor_seen_on_since_heat,
        )

    def _track_compressor_event(self, entity_id: str, data: dict, at_utc: datetime) -> None:
        if entity_id != self.config["hwc"].get("actuation", {}).get("compressor_entity"):
            return

        old_state = (data.get("old_state") or {}).get("state")
        new_state = (data.get("new_state") or {}).get("state")
        if new_state == "on" and old_state != "on":
            self.compressor_seen_on_since_heat = True
            self.compressor_on_since_utc = at_utc
            return

        if old_state == "on" and new_state != "on" and self.compressor_on_since_utc is not None:
            started = self.compressor_on_since_utc
            self.compressor_on_since_utc = None
            run_minutes = (at_utc - started).total_seconds() / 60.0
            minimum = float(
                self.config["hwc"].get("daemon", {}).get("main_completion_min_run_minutes", 20)
            )
            if run_minutes < minimum:
                log.info(
                    "Compressor run %.1fmin below %.1fmin main-completion threshold",
                    run_minutes,
                    minimum,
                )
                return

            added = main_window_overlap_dates(self.config, started, at_utc) - self.completed_main_dates
            if not added:
                return
            self.completed_main_dates.update(added)
            self._save_state()
            log.info("Marked HWC main block complete for %s", ", ".join(sorted(added)))

    def _load_state(self) -> set[str]:
        path = _daemon_state_path(self.config)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return set()
        except Exception:
            log.exception("Failed to load HWC daemon state from %s", path)
            return set()
        return set(data.get("completed_main_dates", []))

    def _save_state(self) -> None:
        path = _daemon_state_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"completed_main_dates": sorted(self.completed_main_dates)}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    async def run(self) -> None:
        self.replan_trigger.set()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.consume_websocket())
            tg.create_task(self.planning_worker())
            tg.create_task(self.execution_worker())
            tg.create_task(self.periodic_execution())
            tg.create_task(self.heartbeat())


def _install_signal_handlers(daemon: HwcDaemon, loop: asyncio.AbstractEventLoop) -> None:
    def _signal():
        log.info("Shutdown signal received")
        daemon.shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal)


def main() -> int:
    parser = argparse.ArgumentParser(description="HWC event-driven planner/executor daemon")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.json"))
    parser.add_argument("--dry-run", action="store_true", help="Do not publish plans or actuate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if not config["home_assistant"].get("token"):
        log.error("home_assistant.token missing (config.secrets.json)")
        return 2

    daemon = HwcDaemon(config, dry_run=args.dry_run)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(daemon, loop)
    rc = 0
    try:
        loop.run_until_complete(daemon.run())
    except* Exception as eg:
        for e in eg.exceptions:
            log.exception("HWC daemon task failed", exc_info=e)
        rc = 1
    finally:
        loop.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
