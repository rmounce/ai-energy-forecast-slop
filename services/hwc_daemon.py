#!/usr/bin/env python3
"""Event-driven HWC planner/executor daemon.

This replaces the eventual shape of separate planner/executor timers: it watches Home
Assistant for HWC-relevant state changes, replans when inputs change, and runs the
execution decision loop on both events and a short periodic cadence.

Actuation remains gated by ``hwc.actuation.enabled`` in ``config.yaml``. With that flag
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

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import hwc_cop_analysis  # noqa: E402
import hwc_cycle_reporter  # noqa: E402
import hwc_cycle_store  # noqa: E402
import hwc_dp_planner  # noqa: E402
import hwc_executor  # noqa: E402
import hwc_planner  # noqa: E402
import hwc_soc_tracker  # noqa: E402
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


def _coerce_float(raw) -> float | None:
    try:
        return float(raw)
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


def command_key(decision: hwc_executor.Decision):
    """Identity of an actuation command; equal keys => a redundant re-command.

    ``None`` for no-op actions (idle/wait), which are never actuated and never dedup-skipped.
    Used to avoid re-issuing an identical water_heater command every periodic tick; the
    daemon invalidates its cached key on equipment state changes so real drift re-asserts.
    """
    if decision.action not in ("off", "heat"):
        return None
    setpoint = round(decision.setpoint_c, 1) if decision.setpoint_c is not None else None
    return (decision.action, setpoint)


def should_suppress_off_after_heat(
    *,
    decision_action: str,
    now: float,
    last_heat_command_at: float,
    grace_seconds: float,
    compressor_on: bool,
) -> bool:
    """Suppress an ``off`` issued shortly after a heat command, while the freshly-commanded
    start has not yet registered.

    The Tuya ``binary_sensor.aquatech_compressor`` lags the real compressor by ~50 s on both
    edges (see ``docs/hwc_thermal_characterisation.md``), so right after an off->heat command
    the sensor still reads "off"; a replan flip to ``off`` in that window would abort the start
    on a stale reading. We therefore hold the ``off`` until the compressor is *confirmed on*
    (``compressor_on``) or the grace expires. Gating on the current observed state — rather than
    an edge latch — means an already-running compressor never blocks a genuine stop, and the
    guard can't be defeated by an intervening heat re-assertion.
    """
    if decision_action != "off":
        return False
    if compressor_on:
        return False
    if last_heat_command_at <= 0:
        return False
    return now - last_heat_command_at < grace_seconds


def should_suppress_heat_after_off(
    *,
    decision_action: str,
    now: float,
    last_off_command_at: float,
    min_off_seconds: float,
) -> bool:
    """Inhibit a ``heat`` command issued within ``min_off_seconds`` of an off command.

    Enforces a minimum compressor rest period — symmetric, hardware-protection counterpart to
    ``should_suppress_off_after_heat``. Unlike that guard this is a *pure time gate* and does not
    consult the compressor sensor: the whole point is to guarantee the compressor a rest before
    a restart regardless of what the planner wants, bounding the start rate against sensor
    jitter or any model edge-case (the 53 °C short-cycle being the motivating one). A min-*on*
    guard is deliberately not added: forcing a stop to be deferred would push the tank past its
    setpoint, whereas deferring a *start* is always safe.
    """
    if decision_action != "heat":
        return False
    if last_off_command_at <= 0:
        return False
    return now - last_off_command_at < min_off_seconds


def effective_compressor_running(
    *,
    raw_on: bool,
    last_command_action: str | None,
    tank_at_target: bool,
    now: float,
    last_heat_command_at: float,
    last_on_at: float,
    start_grace_s: float,
    defrost_grace_s: float,
) -> bool:
    """Debounced "is the compressor effectively running?" for *planner* transition accounting.

    The raw Tuya ``binary_sensor.aquatech_compressor`` lags the real compressor by ~50-60 s on
    both edges, and additionally reads "off" during a defrost while the cycle is really still in
    progress. Feeding the raw reading to the DP as ``compressor_initially_on`` causes phantom
    transition costs: the planner mis-prices "continue" vs "restart", which can truncate the
    in-progress block (sensor lag / defrost) or fail to respect a just-issued stop (off lag).

    Our own command is the timeliest, most authoritative signal — the compressor responds
    near-instantly — so we lead with command intent and use the sensor as confirmation:

    - commanded ``off``    -> not running immediately (don't wait out the off-edge lag, which
                              would risk planning a continue->restart spurious cycle);
    - raw sensor on        -> running (confirmed);
    - tank at/above target -> not running (the unit stopped on its own thermostat: a genuine
                              finish, not a transient pause);
    - within ``start_grace_s`` of a heat command -> running (started ~instantly, sensor lagging);
    - off < ``defrost_grace_s`` while below target -> running (transient pause = defrost);
    - otherwise            -> not running (genuinely off / fault / thermostat-satisfied).

    This signal is *not* used by the executor's off-suppression, which must stay on the raw
    sensor to actually bridge the start lag.
    """
    if last_command_action == "off":
        return False
    if raw_on:
        return True
    if tank_at_target:
        return False
    if last_heat_command_at > 0 and now - last_heat_command_at < start_grace_s:
        return True
    if last_on_at > 0 and now - last_on_at < defrost_grace_s:
        return True
    return False


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
    heat_threshold = setpoint

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
        self.last_off_command_at = 0.0
        self.last_applied_command = None
        # last_command_action persists across cache invalidations (unlike last_applied_command),
        # so the planner's effective-running signal can tell a commanded stop from a defrost pause.
        self.last_command_action: str | None = None
        self.compressor_last_on_at = 0.0
        self._heat_unconfirmed_warned = False
        state = self._load_state()
        self.last_reached_target_at = state.get("last_reached_target_at")
        # Two-state (V_hot, T_hot) tracker state ({v_hot, t_hot, updated_at}); None until seeded.
        self.soc: dict | None = state.get("soc")
        # Cycle-reporting state (publish-only; firewalled from the control loop). The SQLite store
        # (docs/hwc_local_store.md) is the system-of-record; the daemon keeps only the open cycle's
        # edge snapshot and a websocket-fed cache of the latest HWC sensor values.
        rep = config["hwc"].get("reporting", {})
        self._report_enabled = bool(rep.get("enabled", False))
        self._report_db_path = self._abs_path(rep.get("db_path", "data/hwc_cycles.sqlite"))
        self.report_entities: dict[str, str] = dict(rep.get("entities", {}))
        self.report_entity_role = {eid: role for role, eid in self.report_entities.items()}
        self.humidity_entity = rep.get("humidity_entity")
        # After compressor-off, wait this long before finalising so the tank probe's post-off peak
        # (it keeps rising a few seconds) lands in the cache/trace; see docs/hwc_cycle_reporting.md.
        self._close_settle_seconds = float(rep.get("close_settle_seconds", 60))
        self.report_cache: dict[str, float | bool | None] = {}
        # Open cycle {start_ts, tank_start, energy_start}; once the compressor stops it also carries
        # {closed_at, end_ts, energy_end, tank_end_edge} and finalises after the settle window.
        # Persisted so a restart resumes (still running) or finalises (settling) it.
        self.reporter_cycle: dict | None = state.get("reporter_cycle")
        self._reporter_prev_on: bool | None = None
        self._next_msg_id = 1

    def _abs_path(self, p) -> Path:
        path = Path(p)
        return path if path.is_absolute() else REPO_ROOT / path

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
            event_time = _parse_event_time_utc(event.get("time_fired"))
            # Reporter cache + compressor-edge capture (publish-only, firewalled). Runs for HWC
            # report entities regardless of whether they're in the control-watched set.
            if self._report_enabled:
                try:
                    await self._reporter_observe(
                        entity_id, data.get("new_state") or {}, event_time
                    )
                except Exception:
                    log.exception("HWC reporter observe failed; control loop unaffected")
            if entity_id not in self.entities:
                continue
            self._invalidate_command_cache_on_equipment_change(entity_id)
            self._track_compressor_run_event(entity_id, data)
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

    async def cycle_reporter(self) -> None:
        """Publish-only per-cycle reporting to ``sensor.hwc_cycles`` (docs/hwc_local_store.md).

        Firewalled from the planner/executor: its own task, every exception caught, never touches
        ``run_lock`` or the command path. The websocket feeds a live cache of the HWC sensor set and
        the compressor edges (see ``_reporter_observe``); this task samples that cache into the
        per-cycle SQLite trace every ``sample_seconds`` and refreshes the card. Off unless
        ``hwc.reporting.enabled``.
        """
        if not self._report_enabled:
            return
        interval = float(self.config["hwc"]["reporting"].get("sample_seconds", 30))
        try:
            await self._reporter_startup()
        except Exception:
            log.exception("HWC reporter startup failed; continuing best-effort")
        while not self.shutdown.is_set():
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.shutdown.wait(), timeout=interval)
            if self.shutdown.is_set():
                return
            try:
                await self._reporter_sample_tick()
            except Exception:
                log.exception("HWC cycle reporter tick failed; control loop unaffected")

    # ── store I/O (each opens a short-lived connection inside a worker thread) ───────────────────
    def _store_upsert(self, row: dict) -> None:
        conn = hwc_cycle_store.connect(self._report_db_path)
        try:
            hwc_cycle_store.upsert_cycle(conn, row)
        finally:
            conn.close()

    def _store_append_sample(self, start_ts: float, sample: dict) -> None:
        conn = hwc_cycle_store.connect(self._report_db_path)
        try:
            hwc_cycle_store.append_samples(conn, start_ts, [sample])
        finally:
            conn.close()

    def _store_delete(self, start_ts: float) -> None:
        conn = hwc_cycle_store.connect(self._report_db_path)
        try:
            hwc_cycle_store.delete_cycle(conn, start_ts)
        finally:
            conn.close()

    def _store_recent(self, n: int) -> list[dict]:
        conn = hwc_cycle_store.connect(self._report_db_path)
        try:
            return hwc_cycle_store.recent_cycles(conn, n, include_running=False)
        finally:
            conn.close()

    def _store_load_trace(self, start_ts: float):
        conn = hwc_cycle_store.connect(self._report_db_path)
        try:
            return hwc_cycle_store.load_trace(conn, start_ts)
        finally:
            conn.close()

    # ── websocket-driven cache + compressor edges ───────────────────────────────────────────────
    async def _reporter_observe(self, entity_id, new_state, event_time) -> None:
        """Update the latest-value cache from a state_changed event; detect compressor edges."""
        if not entity_id:
            return
        if entity_id == self.humidity_entity:
            self.report_cache["humidity"] = _coerce_float(
                (new_state.get("attributes") or {}).get("humidity"))
            return
        role = self.report_entity_role.get(entity_id)
        if role is None:
            return
        raw = new_state.get("state")
        if role in ("compressor", "element", "defrost", "four_way"):
            value = raw == "on"
        else:
            value = _coerce_float(raw)
        self.report_cache[role] = value
        if role == "compressor":
            await self._reporter_compressor_edge(bool(value), event_time)

    async def _reporter_compressor_edge(self, now_on: bool, event_time) -> None:
        prev = self._reporter_prev_on
        self._reporter_prev_on = now_on
        if prev is None or now_on == prev:
            return
        now_ts = event_time.timestamp()
        if now_on:
            await self._reporter_open(now_ts)
        else:
            await self._reporter_close(now_ts)

    def _reporter_sample(self, now_ts: float) -> dict:
        c = self.report_cache
        return {
            "ts": now_ts, "tank": c.get("tank"), "power_w": c.get("power"),
            "energy_kwh": c.get("energy"), "ambient": c.get("ambient"),
            "humidity": c.get("humidity"), "element": c.get("element"),
            "defrost": c.get("defrost"), "four_way": c.get("four_way"),
            "exhaust": c.get("exhaust"), "coil": c.get("coil"),
            "return_air": c.get("return_air"), "inlet": c.get("inlet"),
        }

    async def _reporter_open(self, now_ts: float) -> None:
        """Compressor off→on: open a 'running' cycle, snapshotting tank/energy at the edge."""
        # A cycle still settling (compressor flicked back on inside the settle window) finalises now.
        if self.reporter_cycle is not None and "closed_at" in self.reporter_cycle:
            await self._reporter_finalise()
        c = self.report_cache
        self.reporter_cycle = {
            "start_ts": now_ts, "tank_start": c.get("tank"), "energy_start": c.get("energy"),
        }
        tz = self.config["timezone"]
        await asyncio.to_thread(self._store_upsert, {
            "start_ts": now_ts, "start_local": hwc_cycle_reporter._local_str(now_ts, tz),
            "tank_start": c.get("tank"), "status": "running", "updated_at": time.time(),
        })
        await asyncio.to_thread(self._store_append_sample, now_ts, self._reporter_sample(now_ts))
        self._save_state()
        log.info("HWC reporter: cycle opened at %s", hwc_cycle_reporter._local_str(now_ts, tz))
        await self._reporter_publish()

    async def _reporter_close(self, now_ts: float) -> None:
        """Compressor on→off: mark the cycle settling; ``_reporter_finalise`` closes it after the
        settle window so the tank probe's post-off peak is captured (it lags compressor-off by a few
        seconds). A sub-``min_cycle_seconds`` run (defrost flicker) is discarded immediately."""
        rc = self.reporter_cycle
        if not rc or "closed_at" in rc:
            return
        cs = rc["start_ts"]
        min_s = float(self.config["hwc"]["reporting"].get("min_cycle_seconds", 300))
        if now_ts - cs < min_s:
            await asyncio.to_thread(self._store_delete, cs)
            self.reporter_cycle = None
            self._save_state()
            log.info("HWC reporter: discarded sub-%.0fs run (defrost flicker?)", min_s)
            await self._reporter_publish()
            return
        c = self.report_cache
        # Snapshot the compressor-off boundary: energy_end (elec input stops here) and the edge tank.
        rc.update(closed_at=now_ts, end_ts=now_ts,
                  energy_end=c.get("energy"), tank_end_edge=c.get("tank"))
        await asyncio.to_thread(self._store_append_sample, cs, self._reporter_sample(now_ts))
        self._save_state()
        log.info("HWC reporter: cycle settling %.0fs before finalise", self._close_settle_seconds)
        await self._reporter_publish()

    async def _reporter_finalise(self) -> None:
        """Close a settling cycle: recompute via cycle_metrics over the trace + edge snapshots, using
        the settled post-off tank as ``tank_end``. Called once the settle window elapses (or on a
        reopen / restart that lands mid-settle)."""
        rc = self.reporter_cycle
        if not rc or "closed_at" not in rc:
            return
        cs, ce = rc["start_ts"], rc["closed_at"]
        tz = self.config["timezone"]
        c = self.report_cache
        # Settled tank: the highest of the compressor-off edge and the (now risen) cached probe.
        tank_end = rc.get("tank_end_edge")
        tank_now = c.get("tank")
        if tank_now is not None and (tank_end is None or tank_now > tank_end):
            tank_end = tank_now
        trace = await asyncio.to_thread(self._store_load_trace, cs)
        edges = {
            "cs": pd.Timestamp(cs, unit="s", tz="UTC"),
            "ce": pd.Timestamp(ce, unit="s", tz="UTC"),
            "tank_start": rc.get("tank_start"), "tank_end": tank_end,
            "energy_start": rc.get("energy_start"), "energy_end": rc.get("energy_end"),
        }
        metrics = hwc_cop_analysis.cycle_metrics(trace, edges=edges) if not trace.empty else None
        if metrics is not None:
            row = {**metrics, "status": "complete", "updated_at": time.time()}
        else:
            # No usable tank trace — keep the run as a minimal complete row from the edge snapshot.
            es, en = rc.get("energy_start"), rc.get("energy_end")
            row = {
                "start_ts": cs, "start_local": hwc_cycle_reporter._local_str(cs, tz),
                "end_ts": ce, "dur_min": round((ce - cs) / 60),
                "tank_start": rc.get("tank_start"), "tank_end": tank_end,
                "elec_kwh": (round(en - es, 3) if (es is not None and en is not None) else None),
                "elec_source": "counter", "status": "complete", "updated_at": time.time(),
            }
        await asyncio.to_thread(self._store_upsert, row)
        self.reporter_cycle = None
        self._save_state()
        log.info("HWC reporter: cycle finalised %s cop=%s tank_end=%s",
                 hwc_cycle_reporter._local_str(cs, tz), row.get("cop"), row.get("tank_end"))
        await self._reporter_publish()

    async def _reporter_sample_tick(self) -> None:
        """Every ``sample_seconds``: append a trace sample while a cycle is open (this is what pulls
        the settling probe's post-off peak into the trace), finalise a settled cycle, refresh card."""
        rc = self.reporter_cycle
        if rc is not None:
            await asyncio.to_thread(
                self._store_append_sample, rc["start_ts"], self._reporter_sample(time.time()),
            )
            if "closed_at" in rc and time.time() - rc["closed_at"] >= self._close_settle_seconds:
                await self._reporter_finalise()
                return
        await self._reporter_publish()

    async def _reporter_publish(self) -> None:
        rep = self.config["hwc"]["reporting"]
        rows = await asyncio.to_thread(self._store_recent, int(rep.get("history_len", 20)))
        records = hwc_cycle_reporter.records_from_store_rows(rows)
        c = self.report_cache
        live = hwc_cycle_reporter.live_record(
            self.reporter_cycle, tank_now=c.get("tank"), energy_now=c.get("energy"),
            now_ts=time.time(), tz_name=self.config["timezone"],
        )
        today = datetime.now(ZoneInfo(self.config["timezone"])).date().isoformat()
        state_scalar, attributes = hwc_cycle_reporter.build_payload(records, live, today_local=today)
        await asyncio.to_thread(
            hwc_planner._ha_set_state, self.config, rep.get("cycles_entity", "sensor.hwc_cycles"),
            state_scalar, attributes,
        )

    async def _reporter_startup(self) -> None:
        """Seed the cache from current HA states, resolve an open cycle across a restart, publish."""
        await asyncio.to_thread(self._reporter_seed_cache)
        compressor_on = bool(self.report_cache.get("compressor"))
        self._reporter_prev_on = compressor_on
        rc = self.reporter_cycle
        if rc is not None and "closed_at" in rc:
            # Was settling when we went down — finalise from the snapshot (the probe has since settled,
            # and the cache was just reseeded from the current state).
            await self._reporter_finalise()
        elif rc is not None and not compressor_on:
            # A run ended while we were down — can't reconstruct precisely; drop it (forward-only).
            await asyncio.to_thread(self._store_delete, rc["start_ts"])
            log.info("HWC reporter: dropped an open cycle that ended during downtime")
            self.reporter_cycle = None
            self._save_state()
        elif rc is None and compressor_on:
            await self._reporter_open(time.time())
        await self._reporter_publish()

    def _reporter_seed_cache(self) -> None:
        for role, eid in self.report_entities.items():
            try:
                st = hwc_planner._ha_call(self.config, "GET", f"states/{eid}")
            except Exception:
                continue
            raw = st.get("state")
            if role in ("compressor", "element", "defrost", "four_way"):
                self.report_cache[role] = raw == "on"
            else:
                self.report_cache[role] = _coerce_float(raw)
        if self.humidity_entity:
            try:
                st = hwc_planner._ha_call(self.config, "GET", f"states/{self.humidity_entity}")
                self.report_cache["humidity"] = _coerce_float(
                    (st.get("attributes") or {}).get("humidity"))
            except Exception:
                pass

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
            planner_config["hwc"]["main_satisfied_dates"] = (
                [satisfied_date] if satisfied_date else []
            )
            planner_config["hwc"]["compressor_initially_on_override"] = (
                await asyncio.to_thread(self._effective_compressor_running)
            )
            if planner_config["hwc"].get("dp_planner", {}).get("soc_model"):
                try:
                    probe = await asyncio.to_thread(hwc_planner.get_tank_temperature, self.config)
                    seed = self._update_soc_tracker(
                        probe_c=probe,
                        heating=planner_config["hwc"]["compressor_initially_on_override"],
                    )
                    if seed is not None:
                        planner_config["hwc"].setdefault("dp_planner", {})["_soc_state0"] = list(seed)
                except Exception:
                    log.exception("HWC SoC tracker update failed; planning without a tracked seed")
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
                effective_on = await asyncio.to_thread(self._effective_compressor_running)
                decision = await asyncio.to_thread(
                    hwc_executor.decide_current, self.config, effective_compressor_on=effective_on
                )
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

            grace = float(self.config["hwc"].get("daemon", {}).get("heat_command_grace_seconds", 120))
            if (
                decision.action == "heat"
                and self.last_command_action == "heat"
                and self.last_heat_command_at > 0
                and time.monotonic() - self.last_heat_command_at > grace
                and not decision.compressor_on
                and not self._heat_unconfirmed_warned
            ):
                log.warning(
                    "HWC heat commanded %.0fs ago but compressor still reports off; "
                    "start may not have taken effect",
                    time.monotonic() - self.last_heat_command_at,
                )
                self._heat_unconfirmed_warned = True

            if self._should_suppress_off_after_heat(decision):
                log.warning(
                    "Suppressing HWC off command %.1fs after heat command",
                    time.monotonic() - self.last_heat_command_at,
                )
                return

            if self._should_suppress_heat_after_off(decision):
                log.warning(
                    "Suppressing HWC heat command %.1fs after off command (min-off guard)",
                    time.monotonic() - self.last_off_command_at,
                )
                return

            act = self.config["hwc"].get("actuation", {})
            key = command_key(decision)
            if self.dry_run or not act.get("enabled", False):
                log.info("Dry/config-disabled run; not calling water_heater services")
            elif key is None:
                log.info("HWC executor no-op (%s); no command issued", decision.action)
            elif key == self.last_applied_command:
                log.info(
                    "HWC command unchanged (%s, setpoint=%s); skipping re-command",
                    decision.action,
                    decision.setpoint_c,
                )
            else:
                try:
                    await asyncio.to_thread(hwc_executor.apply_decision, self.config, decision)
                except Exception:
                    # Leave last_applied_command unchanged so the next tick retries.
                    log.exception("HWC executor apply failed")
                    return
                self.last_applied_command = key
                self.last_command_action = decision.action
                if decision.action == "heat":
                    self.last_heat_command_at = time.monotonic()
                    self._heat_unconfirmed_warned = False
                elif decision.action == "off":
                    self.last_off_command_at = time.monotonic()
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
            compressor_on=decision.compressor_on,
        )

    def _should_suppress_heat_after_off(self, decision: hwc_executor.Decision) -> bool:
        min_off = float(self.config["hwc"].get("daemon", {}).get("min_off_seconds", 180))
        return should_suppress_heat_after_off(
            decision_action=decision.action,
            now=time.monotonic(),
            last_off_command_at=self.last_off_command_at,
            min_off_seconds=min_off,
        )

    def _invalidate_command_cache_on_equipment_change(self, entity_id: str) -> None:
        # Any change to the heater/compressor means our cached command may no longer reflect
        # reality (e.g. a manual mode change), so force the next executor pass to re-assert.
        act = self.config["hwc"].get("actuation", {})
        if entity_id in (act.get("water_heater_entity"), act.get("compressor_entity")):
            self.last_applied_command = None

    def _track_compressor_run_event(self, entity_id: str, data: dict) -> None:
        # Record when the compressor was last observed on, so the planner's effective-running
        # signal can bound a defrost pause (off-duration) without an unreliable edge latch.
        if entity_id != self.config["hwc"].get("actuation", {}).get("compressor_entity"):
            return
        if (data.get("new_state") or {}).get("state") == "on":
            self.compressor_last_on_at = time.monotonic()

    def _effective_compressor_running(self) -> bool:
        daemon = self.config["hwc"].get("daemon", {})
        start_grace = float(daemon.get("heat_command_grace_seconds", 120))
        defrost_grace = float(daemon.get("compressor_off_debounce_seconds", 600))
        try:
            raw_on = hwc_planner.compressor_is_on(self.config)
        except Exception:
            log.exception("Could not read compressor state; assuming off for planning")
            raw_on = False
        try:
            tank_at_target = (
                hwc_planner.get_tank_temperature(self.config) >= _target_temperature_c(self.config)
            )
        except Exception:
            log.exception("Could not read tank temperature; assuming below target for planning")
            tank_at_target = False
        return effective_compressor_running(
            raw_on=raw_on,
            last_command_action=self.last_command_action,
            tank_at_target=tank_at_target,
            now=time.monotonic(),
            last_heat_command_at=self.last_heat_command_at,
            last_on_at=self.compressor_last_on_at,
            start_grace_s=start_grace,
            defrost_grace_s=defrost_grace,
        )

    def _soc_draw_rate_kwh_per_s(self) -> float:
        """Conservative draw-prior rate (kWh/s of hot water) during the configured draw window.

        Pessimistic by design: a draw that stays above the probe fires no watermark, so the prior is
        the only guard (docs/hwc_2state_soc_model.md). Defaults to the planning draw total spread
        over the window; tune up via ``hwc.dp_planner.soc.draw_prior_kwh_per_day``.
        """
        hwc = self.config["hwc"]
        soc_cfg = hwc.get("dp_planner", {}).get("soc", {})
        draw_off = hwc.get("draw_off", {})
        kwh_per_day = float(soc_cfg.get("draw_prior_kwh_per_day", draw_off.get("total_kwh", 0.0)))
        if kwh_per_day <= 0:
            return 0.0
        start = _parse_hhmm_time(str(draw_off.get("window_start", "06:00")))
        end = _parse_hhmm_time(str(draw_off.get("window_end", "07:00")))
        now_local = datetime.now(ZoneInfo(self.config["timezone"])).time()
        if not _time_in_window(now_local, start, end):
            return 0.0
        window_s = ((end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)) * 60
        return kwh_per_day / (window_s if window_s > 0 else 3600)

    def _update_soc_tracker(self, *, probe_c: float, heating: bool) -> tuple[float, float] | None:
        """Advance the (V_hot, T_hot) tracker to now and return the planner seed (or None if off).

        Coarse but conservative: the current ``heating`` signal is applied over the whole elapsed
        interval; the two watermark resets correct any drift (docs/hwc_2state_soc_model.md). Pure
        math + a local state-file write — no network (probe/heating are passed in).
        """
        hwc = self.config["hwc"]
        dp_cfg = hwc.get("dp_planner", {})
        if not dp_cfg.get("soc_model"):
            return None
        th = hwc["thermal"]
        p = hwc_dp_planner._soc_params_from_cfg(th, dp_cfg)
        desired = _target_temperature_c(self.config)
        cliff = float(dp_cfg.get("soc", {}).get("cliff_probe_c", (p.t_mains_c + desired) / 2.0))
        now = time.time()
        if self.soc is None:
            st = hwc_soc_tracker.seed_state(probe_c, p)
            reason = "seed"
        else:
            dt_s = max(0.0, now - float(self.soc.get("updated_at", now)))
            prev = hwc_soc_tracker.TrackerState(float(self.soc["v_hot"]), float(self.soc["t_hot"]))
            power = hwc_planner._compressor_power_w(th, probe_c, None)
            st, reason = hwc_soc_tracker.advance(
                prev, dt_s, heating=bool(heating), probe_c=probe_c, modelled_power_w=power,
                draw_kwh_per_s=self._soc_draw_rate_kwh_per_s(), p=p,
                desired_c=desired, cliff_probe_c=cliff,
            )
        self.soc = {"v_hot": round(st.v_hot, 5), "t_hot": round(st.t_hot, 3), "updated_at": now}
        self._save_state()
        log.info("HWC SoC tracker: V_hot=%.2f T_hot=%.1f°C (%s)", st.v_hot, st.t_hot, reason)
        return (st.v_hot, st.t_hot)

    def _mark_target_reached(self, temp: float | None, at_utc: datetime) -> None:
        # Level-triggered: latch whenever the tank is observed at/above target and we have not
        # already recorded it for this local day. Edge-triggering on the upward crossing missed
        # the target whenever the crossing event was absent (daemon restart while already hot, a
        # dropped websocket event) — a false negative that drops the day's reheat obligation.
        if temp is None or temp < _target_temperature_c(self.config):
            return
        local_date = target_reached_local_date(self.config, at_utc.isoformat())
        if target_reached_local_date(self.config, self.last_reached_target_at) == local_date:
            return
        self.last_reached_target_at = at_utc.isoformat()
        self._save_state()
        log.info("HWC target temperature reached: %.1fC at %s", temp, self.last_reached_target_at)

    def _track_target_temperature_event(self, entity_id: str, data: dict, at_utc: datetime) -> None:
        if entity_id != self.config["hwc"]["tank_temp_entity"]:
            return
        self._mark_target_reached(_state_float(data.get("new_state")), at_utc)

    async def _seed_target_reached(self) -> None:
        # Catch the case where the daemon starts while the tank is already at/above target, so no
        # crossing event will arrive to latch it.
        try:
            temp = await asyncio.to_thread(hwc_planner.get_tank_temperature, self.config)
        except Exception:
            log.exception("Could not seed HWC target-reached state at startup")
            return
        self._mark_target_reached(temp, datetime.now(timezone.utc))

    def _load_state(self) -> dict:
        path = _daemon_state_path(self.config)
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return {}
        except Exception:
            log.exception("Failed to load HWC daemon state from %s", path)
            return {}

    def _save_state(self) -> None:
        path = _daemon_state_path(self.config)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_reached_target_at": self.last_reached_target_at}
        soc = getattr(self, "soc", None)  # getattr: tolerate __new__'d daemons in unit tests
        if soc is not None:
            payload["soc"] = soc
        reporter_cycle = getattr(self, "reporter_cycle", None)
        if reporter_cycle is not None:
            payload["reporter_cycle"] = reporter_cycle
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    async def run(self) -> None:
        await self._seed_target_reached()
        self.replan_trigger.set()
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.consume_websocket())
            tg.create_task(self.planning_worker())
            tg.create_task(self.execution_worker())
            tg.create_task(self.periodic_execution())
            tg.create_task(self.heartbeat())
            tg.create_task(self.cycle_reporter())


def _install_signal_handlers(daemon: HwcDaemon, loop: asyncio.AbstractEventLoop) -> None:
    def _signal():
        log.info("Shutdown signal received")
        daemon.shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal)


def main() -> int:
    parser = argparse.ArgumentParser(description="HWC event-driven planner/executor daemon")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Do not publish plans or actuate")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    if not config["home_assistant"].get("token"):
        log.error("home_assistant.token missing (config.secrets.yaml)")
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
