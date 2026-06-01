#!/usr/bin/env python3
"""Execute the published HWC block plan.

This is the actuation layer for ``hwc_planner.py``. It reads the published HWC power and
temperature plan from Home Assistant, decides whether the Aquatech should be enabled, and
uses Home Assistant's water_heater services to set mode/setpoint or turn the unit off.

The logic is deliberately simple:

- inside a planned compressor block: set the configured operation mode and the block-end
  predicted temperature as setpoint;
- just after a block, while the compressor is still running: keep the same mode/setpoint so
  the unit can finish naturally;
- outside a block, once the compressor is off: turn the water heater off, allowing the tank
  to drift below the unit's normal reheat trigger.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from config_utils import load_config


@dataclass(frozen=True)
class PlanPoint:
    at: datetime
    power_w: float
    temp_c: float


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str
    setpoint_c: float | None = None
    block_start: datetime | None = None
    block_end: datetime | None = None


def _published_entity_id(prefix: str, entity_id: str) -> str:
    domain, object_id = entity_id.split(".", 1)
    return f"{domain}.{prefix}{object_id}"


def _ha_call(cfg: dict, method: str, endpoint: str, payload: dict | None = None):
    ha = cfg["home_assistant"]
    url = f"{ha['url']}/api/{endpoint}"
    headers = {"Authorization": f"Bearer {ha['token']}", "Content-Type": "application/json"}
    resp = (
        requests.post(url, headers=headers, json=payload, timeout=30)
        if method == "POST"
        else requests.get(url, headers=headers, timeout=30)
    )
    resp.raise_for_status()
    return resp.json()


def _service_call(cfg: dict, service: str, payload: dict):
    return _ha_call(cfg, "POST", f"services/water_heater/{service}", payload)


def _entity_state(cfg: dict, entity_id: str) -> dict:
    return _ha_call(cfg, "GET", f"states/{entity_id}")


def _point_time(item: dict) -> datetime:
    return pd.to_datetime(item["date"], utc=True).to_pydatetime()


def load_plan(cfg: dict) -> list[PlanPoint]:
    hwc = cfg["hwc"]
    prefix = hwc.get("publish_prefix", "hwc_")
    power_entity = _published_entity_id(prefix, hwc["power_plan_entity"])
    temp_entity = _published_entity_id(prefix, hwc["predicted_temp_entity"])
    power_key = power_entity.split(".", 1)[1]
    temp_key = temp_entity.split(".", 1)[1]

    power_state = _entity_state(cfg, power_entity)
    temp_state = _entity_state(cfg, temp_entity)
    power_rows = power_state.get("attributes", {}).get("deferrables_schedule", []) or []
    temp_rows = temp_state.get("attributes", {}).get("predicted_temperatures", []) or []
    temps_by_time = {_point_time(row): float(row[temp_key]) for row in temp_rows}

    points = []
    for row in power_rows:
        at = _point_time(row)
        if at in temps_by_time:
            points.append(PlanPoint(at=at, power_w=float(row[power_key]), temp_c=temps_by_time[at]))
    points.sort(key=lambda p: p.at)
    if not points:
        raise RuntimeError("HWC published plan is empty or inconsistent")
    return points


def _current_index(points: list[PlanPoint], now: datetime) -> int | None:
    if now < points[0].at:
        return None
    for idx, point in enumerate(points):
        next_at = points[idx + 1].at if idx + 1 < len(points) else point.at + _step(points)
        if point.at <= now < next_at:
            return idx
    return None


def _step(points: list[PlanPoint]) -> timedelta:
    if len(points) >= 2:
        return points[1].at - points[0].at
    return timedelta(minutes=30)


def _block_bounds(points: list[PlanPoint], idx: int, threshold_w: float) -> tuple[int, int]:
    start = idx
    while start > 0 and points[start - 1].power_w > threshold_w:
        start -= 1
    end = idx
    while end + 1 < len(points) and points[end + 1].power_w > threshold_w:
        end += 1
    return start, end


def _previous_block(points: list[PlanPoint], idx: int, threshold_w: float) -> tuple[int, int] | None:
    cursor = min(idx, len(points) - 1)
    while cursor >= 0 and points[cursor].power_w <= threshold_w:
        cursor -= 1
    if cursor < 0:
        return None
    return _block_bounds(points, cursor, threshold_w)


def _block_setpoint(points: list[PlanPoint], end_idx: int, setpoint_min: float, setpoint_max: float) -> float:
    # Planner temperatures are interval-start states; after the last heating slot appears at
    # the next point when available.
    target_idx = min(end_idx + 1, len(points) - 1)
    return round(min(setpoint_max, max(setpoint_min, points[target_idx].temp_c)), 1)


def decide(
    points: list[PlanPoint],
    *,
    now: datetime,
    compressor_on: bool,
    threshold_w: float,
    setpoint_min: float,
    setpoint_max: float,
    post_block_grace: timedelta,
) -> Decision:
    idx = _current_index(points, now)
    if idx is None:
        return Decision(action="idle", reason="outside published plan")

    if points[idx].power_w > threshold_w:
        start, end = _block_bounds(points, idx, threshold_w)
        return Decision(
            action="heat",
            reason="inside planned block",
            setpoint_c=_block_setpoint(points, end, setpoint_min, setpoint_max),
            block_start=points[start].at,
            block_end=points[end].at + _step(points),
        )

    prev = _previous_block(points, idx, threshold_w)
    if prev and compressor_on:
        start, end = prev
        block_end = points[end].at + _step(points)
        if now - block_end <= post_block_grace:
            return Decision(
                action="heat",
                reason="compressor still running after planned block",
                setpoint_c=_block_setpoint(points, end, setpoint_min, setpoint_max),
                block_start=points[start].at,
                block_end=block_end,
            )

    if compressor_on:
        return Decision(action="wait", reason="outside block but compressor is running")
    return Decision(action="off", reason="outside planned block and compressor is off")


def apply_decision(cfg: dict, decision: Decision):
    act = cfg["hwc"]["actuation"]
    entity = act["water_heater_entity"]
    if decision.action == "heat":
        _service_call(
            cfg,
            "set_operation_mode",
            {
                "entity_id": entity,
                "operation_mode": act.get("operation_mode", "heat_pump"),
            },
        )
        _service_call(
            cfg,
            "set_temperature",
            {
                "entity_id": entity,
                "temperature": decision.setpoint_c,
                "operation_mode": act.get("operation_mode", "heat_pump"),
            },
        )
    elif decision.action == "off":
        _service_call(cfg, "turn_off", {"entity_id": entity})


def run(cfg: dict, *, dry_run: bool = False, force: bool = False) -> Decision:
    act = cfg["hwc"].get("actuation", {})
    points = load_plan(cfg)
    compressor_state = _entity_state(cfg, act["compressor_entity"])
    compressor_on = compressor_state.get("state") == "on"
    decision = decide(
        points,
        now=datetime.now(timezone.utc),
        compressor_on=compressor_on,
        threshold_w=float(act.get("power_on_threshold_w", 100)),
        setpoint_min=float(act.get("setpoint_min_c", 55)),
        setpoint_max=float(act.get("setpoint_max_c", 60)),
        post_block_grace=timedelta(minutes=float(act.get("post_block_grace_minutes", 90))),
    )
    logging.info("HWC executor decision: %s (%s), setpoint=%s", decision.action, decision.reason, decision.setpoint_c)

    if dry_run or (not act.get("enabled", False) and not force):
        logging.info("Dry/config-disabled run; not calling water_heater services")
        return decision
    apply_decision(cfg, decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description="HWC plan executor")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Actuate even if hwc.actuation.enabled is false")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_config(args.config), dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
