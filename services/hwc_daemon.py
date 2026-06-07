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
from datetime import datetime, time as dt_time, timezone
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


def _target_temperature_c(config: dict) -> float:
    return float(config["hwc"]["thermal"].get("desired_temp", 60))


def target_reached_local_date(config: dict, reached_at_utc: str | None) -> str | None:
    if not reached_at_utc:
        return None
    try:
        reached_at = datetime.fromisoformat(reached_at_utc.replace("Z", "+00:00"))
    except ValueError:
        return None
    return reached_at.astimezone(ZoneInfo(config["timezone"])).date().isoformat()


def price_input_entities(config: dict) -> set[str]:
    hwc = config["hwc"]
    entities = {
        hwc.get("emhass_dh_unit_load_cost_entity", "sensor.dh_unit_load_cost"),
    }
    mpc_entity = hwc.get("emhass_mpc_unit_load_cost_entity", "sensor.mpc_unit_load_cost")
    if int(hwc.get("optimization_time_step", 30)) < 30 and mpc_entity:
        entities.add(mpc_entity)
    return {entity for entity in entities if entity}


def watched_entities(config: dict) -> set[str]:
    hwc = config["hwc"]
    ha = config["home_assistant"]
    act = hwc.get("actuation", {})
    prefix = hwc.get("publish_prefix", "hwc_")

    entities = {
        hwc["tank_temp_entity"],
        ha["weather_entity"],
        _published_entity_id(prefix, hwc["power_plan_entity"]),
        _published_entity_id(prefix, hwc["predicted_temp_entity"]),
    } | price_input_entities(config)
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

    forecast_entities = price_input_entities(config) | {
        config["home_assistant"]["weather_entity"],
    }
    if entity_id in forecast_entities:
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


def _parse_hhmm_time(value: str) -> dt_time:
    hour, minute = (int(part) for part in value.split(":", 1))
    return dt_time(hour=hour, minute=minute)


def _time_in_window(value: dt_time, start: dt_time, end: dt_time) -> bool:
    if start <= end:
        return start <= value < end
    return value >= start or value < end


def fallback_decision(
    config: dict,
    *,
    now_utc: datetime,
    tank_temp_c: float,
    compressor_on: bool,
) -> hwc_executor.Decision | None:
    """Return a fixed-window safety decision when normal plan execution fails."""
    daemon = config["hwc"].get("daemon", {})
    if not daemon.get("fallback_enabled", False):
        return None

    act = config["hwc"].get("actuation", {})
    th = config["hwc"].get("thermal", {})
    local = now_utc.astimezone(ZoneInfo(config["timezone"]))
    start = _parse_hhmm_time(daemon.get("fallback_window_start", "10:00"))
    end = _parse_hhmm_time(daemon.get("fallback_window_end", "16:00"))
    in_window = _time_in_window(local.time(), start, end)
    setpoint_min = float(act.get("setpoint_min_c", 55))
    setpoint_max = float(act.get("setpoint_max_c", 60))
    fallback_setpoint = float(daemon.get("fallback_setpoint_c", th.get("desired_temp", 60)))
    setpoint = min(setpoint_max, max(setpoint_min, fallback_setpoint))
    min_temp = float(daemon.get("fallback_min_temp_c", th.get("min_temp", 45)))
    min_delta = float(act.get("min_heat_start_delta_c", 0.0))
    heat_threshold = setpoint - min_delta

    if tank_temp_c < min_temp:
        return hwc_executor.Decision(
            action="heat",
            reason=f"fallback emergency heat: tank {tank_temp_c:.1f}C below {min_temp:.1f}C",
            setpoint_c=round(setpoint, 1),
        )
    if in_window and (compressor_on or tank_temp_c < heat_threshold):
        return hwc_executor.Decision(
            action="heat",
            reason=(
                f"fallback fixed-window heat: tank {tank_temp_c:.1f}C, "
                f"window {start.strftime('%H:%M')}-{end.strftime('%H:%M')}"
            ),
            setpoint_c=round(setpoint, 1),
        )
    if compressor_on:
        return hwc_executor.Decision(action="wait", reason="fallback outside window but compressor is running")
    return hwc_executor.Decision(action="off", reason="fallback outside fixed heat conditions")


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
        self.last_reached_target_at = self._load_state()
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
            event_time = _parse_event_time_utc(event.get("time_fired"))
            self._track_compressor_latch_event(entity_id, data)
            self._track_target_temperature_event(entity_id, data, event_time)
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
            satisfied_date = target_reached_local_date(self.config, self.last_reached_target_at)
            planner_config["hwc"].setdefault("block_planner", {})["main_satisfied_dates"] = (
                [satisfied_date] if satisfied_date else []
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
                decision = await asyncio.to_thread(self._fallback_decision_current)
                if decision is None:
                    log.error("HWC fallback disabled or unavailable; no water_heater command issued")
                    return
                log.warning("Using HWC fallback decision: %s (%s)", decision.action, decision.reason)
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

    def _fallback_decision_current(self) -> hwc_executor.Decision | None:
        try:
            tank_temp = hwc_executor._tank_temperature(self.config)
            compressor_state = hwc_executor._entity_state(
                self.config,
                self.config["hwc"].get("actuation", {})["compressor_entity"],
            )
        except Exception:
            log.exception("HWC fallback could not read tank/compressor state")
            return None
        return fallback_decision(
            self.config,
            now_utc=datetime.now(timezone.utc),
            tank_temp_c=tank_temp,
            compressor_on=compressor_state.get("state") == "on",
        )

    def _should_suppress_off_after_heat(self, decision: hwc_executor.Decision) -> bool:
        grace = float(self.config["hwc"].get("daemon", {}).get("heat_command_grace_seconds", 600))
        return should_suppress_off_after_heat(
            decision_action=decision.action,
            now=time.monotonic(),
            last_heat_command_at=self.last_heat_command_at,
            grace_seconds=grace,
            compressor_seen_on_since_heat=self.compressor_seen_on_since_heat,
        )

    def _track_compressor_latch_event(self, entity_id: str, data: dict) -> None:
        if entity_id != self.config["hwc"].get("actuation", {}).get("compressor_entity"):
            return

        old_state = (data.get("old_state") or {}).get("state")
        new_state = (data.get("new_state") or {}).get("state")
        if new_state == "on" and old_state != "on":
            self.compressor_seen_on_since_heat = True

    def _track_target_temperature_event(self, entity_id: str, data: dict, at_utc: datetime) -> None:
        if entity_id != self.config["hwc"]["tank_temp_entity"]:
            return

        old_temp = _state_float(data.get("old_state"))
        new_temp = _state_float(data.get("new_state"))
        target = _target_temperature_c(self.config)
        if new_temp is None or new_temp < target:
            return
        if old_temp is not None and old_temp >= target:
            return
        self.last_reached_target_at = at_utc.isoformat()
        self._save_state()
        log.info("HWC target temperature reached: %.1fC at %s", new_temp, self.last_reached_target_at)

    def _load_state(self) -> str | None:
        path = _daemon_state_path(self.config)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except Exception:
            log.exception("Failed to load HWC daemon state from %s", path)
            return None
        return data.get("last_reached_target_at")

    def _save_state(self) -> None:
        path = _daemon_state_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_reached_target_at": self.last_reached_target_at}
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
