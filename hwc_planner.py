#!/usr/bin/env python3
"""Heat-pump hot water (HWC) scheduling planner — v1 (modelling only).

Optimises a heat-pump hot water unit (Aquatech RAPID X6), *separately* from the home
battery, against the import-price forecast and weather. The default planner models the
unit as a fixed-speed block heater and publishes predicted tank temperature + planned
power directly to Home Assistant. The older EMHASS ``thermal_battery`` planner is still
available via config for comparison.

This is the *modelling* phase: it produces and publishes a plan only — it does NOT
actuate the unit. See ``docs/hwc_emhass.md`` for the full design, the calibration
anchors, and the v1/v2/v3 roadmap.

Pure helpers (``stull_wet_bulb``, ``interpolate_to_grid``, ``build_draw_off_profile``,
``build_payload`` and block-planner helpers) are unit-tested in
``tests/unit/test_hwc_planner.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests

from config_utils import load_config


# ── Pure helpers (unit-tested) ──────────────────────────────────────────────


def stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """Wet-bulb temperature (°C) from dry-bulb temp (°C) and relative humidity (%).

    Stull (2011) empirical approximation — valid for roughly RH 5–99 % and
    T −20…50 °C, which comfortably covers Adelaide conditions. The heat pump's COP is
    governed by evaporator (wet-bulb) conditions, so this is what we feed EMHASS as the
    outdoor-temperature forecast. See docs/hwc_emhass.md ("Wet-bulb vs dry-bulb").
    """
    rh = max(1.0, min(100.0, float(rh_pct)))
    t = float(temp_c)
    return (
        t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )


def interpolate_to_grid(
    src_epoch: list[float], src_values: list[float], grid_epoch: list[float]
) -> list[float]:
    """Linearly interpolate ``src_values`` (sampled at ``src_epoch`` seconds) onto
    ``grid_epoch``. Ends are held flat (np.interp constant extrapolation), so a grid
    extending slightly past the weather horizon reuses the last forecast value."""
    if not src_epoch:
        raise ValueError("interpolate_to_grid: empty source series")
    order = np.argsort(src_epoch)
    xs = np.asarray(src_epoch, dtype=float)[order]
    ys = np.asarray(src_values, dtype=float)[order]
    return np.interp(np.asarray(grid_epoch, dtype=float), xs, ys).tolist()


def build_draw_off_profile(
    grid_times_utc: list[datetime],
    tz_name: str,
    window_start: str,
    window_end: str,
    total_kwh: float,
) -> list[float]:
    """Build a per-timestep hot-water draw-off profile (kWh) aligned to *clock time*.

    EMHASS tiles ``draw_off_demand`` from the start of the horizon (index 0 = "now"),
    NOT by time of day, so we must place the draw ourselves at the timesteps whose local
    clock time falls in [window_start, window_end). ``total_kwh`` is distributed evenly
    across the in-window slots, so each day in the horizon receives ``total_kwh``.
    """
    tz = pytz.timezone(tz_name)
    sh, sm = (int(x) for x in window_start.split(":"))
    eh, em = (int(x) for x in window_end.split(":"))
    start_min, end_min = sh * 60 + sm, eh * 60 + em
    slots_per_window = max(1, (end_min - start_min) // _step_minutes(grid_times_utc))
    per_slot = float(total_kwh) / slots_per_window

    profile = []
    for t in grid_times_utc:
        local = t.astimezone(tz)
        minute_of_day = local.hour * 60 + local.minute
        in_window = start_min <= minute_of_day < end_min
        profile.append(round(per_slot, 5) if in_window else 0.0)
    return profile


def add_draw_off_event(
    profile: list[float],
    grid_times_utc: list[datetime],
    tz_name: str,
    start_time: str,
    duration_min: int,
    total_kwh: float,
) -> list[float]:
    """Return ``profile`` with an extra clock-aligned draw-off event added each local day."""
    tz = pytz.timezone(tz_name)
    start_min = _parse_hhmm(start_time)
    end_min = start_min + int(duration_min)
    slots_per_event = max(1, int(math.ceil(duration_min / _step_minutes(grid_times_utc))))
    per_slot = float(total_kwh) / slots_per_event
    out = list(profile)
    for idx, t in enumerate(grid_times_utc):
        minute = _local_minute(t, tz)
        if _minute_in_window(minute, start_min, end_min % (24 * 60)):
            out[idx] = round(out[idx] + per_slot, 5)
    return out


def parse_extra_draw_off(spec: str) -> tuple[str, int, float]:
    """Parse ``HH:MM=KWH`` or ``HH:MM+MIN=KWH`` for one-off scenario runs."""
    when, energy = spec.split("=", 1)
    if "+" in when:
        start_time, duration = when.split("+", 1)
        duration_min = int(duration)
    else:
        start_time = when
        duration_min = 60
    _parse_hhmm(start_time)
    if duration_min <= 0:
        raise ValueError("extra draw-off duration must be positive")
    return start_time, duration_min, float(energy)


def _step_minutes(grid_times_utc: list[datetime]) -> int:
    """Infer the grid step in minutes (defaults to 30 if not inferable)."""
    if len(grid_times_utc) >= 2:
        return max(1, int((grid_times_utc[1] - grid_times_utc[0]).total_seconds() // 60))
    return 30


def _thermal_capacity_kwh_per_c(th: dict) -> float:
    """Return tank sensible heat capacity in kWh / °C."""
    litres = float(th["volume_l"])
    density_kg_per_l = float(th.get("density", 997)) / 1000.0
    return litres * density_kg_per_l * float(th.get("heat_capacity", 4.184)) / 3600.0


def _heat_rate_c_per_hour(th: dict, temp_c: float) -> float:
    """Return empirical compressor heat rate for the current modelled tank temp.

    The Aquatech data shows lower effective probe lift rate for near-target top-ups.
    Keep this optional so existing configs retain the original single-rate model.
    """
    base = float(th.get("heat_rate_c_per_hour", 5.2))
    top_up = th.get("top_up_heat_rate_c_per_hour")
    top_up_start = th.get("top_up_start_temp_c")
    if top_up is not None and top_up_start is not None and temp_c >= float(top_up_start):
        return float(top_up)
    return base


def _parse_hhmm(value: str) -> int:
    h, m = (int(x) for x in value.split(":"))
    return h * 60 + m


def _minute_in_window(minute: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= minute < end
    return minute >= start or minute < end


def _local_minute(t: datetime, tz) -> int:
    local = t.astimezone(tz)
    return local.hour * 60 + local.minute


def _published_entity_id(prefix: str, entity_id: str) -> str:
    domain, object_id = entity_id.split(".", 1)
    return f"{domain}.{prefix}{object_id}"


def simulate_block_temperatures(
    *,
    schedule_w: list[float],
    start_temperature: float,
    dry_bulb: list[float],
    draw_off: list[float],
    cfg: dict,
) -> tuple[list[float], float]:
    """Simulate the custom HWC block model.

    Temperatures are reported at each forecast timestamp before that interval's heat/loss/draw
    is applied. The returned terminal temperature is the state after the final interval.
    """
    th = cfg["thermal"]
    step_h = cfg.get("optimization_time_step", 30) / 60.0
    cap_kwh_per_c = _thermal_capacity_kwh_per_c(th)
    ua_kw_per_c = float(th.get("standing_loss_ua_kw_per_c", 0.0025))
    max_temp = float(th.get("max_temp", 62))
    temp = float(start_temperature)
    temps = []
    for power_w, ambient_c, draw_kwh in zip(schedule_w, dry_bulb, draw_off, strict=True):
        temps.append(round(temp, 2))
        loss_kwh = max(0.0, temp - float(ambient_c)) * ua_kw_per_c * step_h
        temp -= loss_kwh / cap_kwh_per_c
        temp -= float(draw_kwh) / cap_kwh_per_c
        if power_w > 0:
            heat_rate_c_per_h = _heat_rate_c_per_hour(th, temp)
            temp += heat_rate_c_per_h * step_h
        temp = min(max_temp, temp)
    return temps, round(temp, 2)


def _add_contiguous_heat(
    schedule_w: list[float],
    *,
    start_idx: int,
    end_idx: int,
    target_temp: float,
    start_temperature: float,
    dry_bulb: list[float],
    draw_off: list[float],
    cfg: dict,
) -> list[float]:
    """Return a copy with one contiguous heater run from start_idx until target is reached."""
    th = cfg["thermal"]
    power_w = float(th["nominal_power_w"])
    out = list(schedule_w)
    end_idx = min(end_idx, len(out))
    if start_idx >= end_idx:
        return out
    for idx in range(start_idx, end_idx):
        temps, _ = simulate_block_temperatures(
            schedule_w=out,
            start_temperature=start_temperature,
            dry_bulb=dry_bulb,
            draw_off=draw_off,
            cfg=cfg,
        )
        if temps[idx] >= target_temp:
            break
        out[idx] = power_w
    return out


def _schedule_cost_delta(
    before: list[float], after: list[float], load_cost: list[float], step_h: float
) -> float:
    return sum(
        max(0.0, new - old) / 1000.0 * float(cost) * step_h
        for old, new, cost in zip(before, after, load_cost, strict=True)
    )


def _min_block_lift_c(hwc: dict) -> float:
    block_cfg = hwc.get("block_planner", {})
    return float(block_cfg.get("min_block_lift_c", block_cfg.get("min_main_block_lift_c", 0.0)))


def _choose_daily_main_blocks(
    schedule_w: list[float],
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    start_temperature: float,
    dry_bulb: list[float],
    draw_off: list[float],
    cfg: dict,
) -> list[float]:
    tz = pytz.timezone(cfg["timezone"])
    block_cfg = cfg["hwc"].get("block_planner", {})
    main_start = _parse_hhmm(block_cfg.get("main_window_start", "10:00"))
    main_end = _parse_hhmm(block_cfg.get("main_window_end", "18:00"))
    target = float(cfg["hwc"]["thermal"].get("desired_temp", 60))
    min_temp = float(cfg["hwc"]["thermal"].get("min_temp", 45))
    min_lift_c = _min_block_lift_c(cfg["hwc"])
    step_h = cfg["hwc"].get("optimization_time_step", 30) / 60.0

    slots_by_day: dict[datetime.date, list[int]] = {}
    for idx, t in enumerate(grid_times_utc):
        minute = _local_minute(t, tz)
        if _minute_in_window(minute, main_start, main_end):
            slots_by_day.setdefault(t.astimezone(tz).date(), []).append(idx)

    satisfied_dates = set(block_cfg.get("main_satisfied_dates", []))
    out = list(schedule_w)
    for day, slots in slots_by_day.items():
        if day.isoformat() in satisfied_dates:
            continue
        best = out
        best_score = (math.inf, math.inf, math.inf)
        base_temps, _ = simulate_block_temperatures(
            schedule_w=out,
            start_temperature=start_temperature,
            dry_bulb=dry_bulb,
            draw_off=draw_off,
            cfg=cfg["hwc"],
        )
        for start_idx in slots:
            if base_temps[start_idx] > target - min_lift_c:
                candidate = out
            else:
                candidate = _add_contiguous_heat(
                    out,
                    start_idx=start_idx,
                    end_idx=slots[-1] + 1,
                    target_temp=target,
                    start_temperature=start_temperature,
                    dry_bulb=dry_bulb,
                    draw_off=draw_off,
                    cfg=cfg["hwc"],
                )
            ctemps, _ = simulate_block_temperatures(
                schedule_w=candidate,
                start_temperature=start_temperature,
                dry_bulb=dry_bulb,
                draw_off=draw_off,
                cfg=cfg["hwc"],
            )
            window_shortfall = max(0.0, min_temp - min(ctemps[slots[0] : slots[-1] + 1]))
            cost = _schedule_cost_delta(out, candidate, load_cost, step_h)
            score = (window_shortfall, cost, float(start_idx))
            if score < best_score:
                best = candidate
                best_score = score
        out = best
    return out


def _repair_min_temperature(
    schedule_w: list[float],
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    start_temperature: float,
    dry_bulb: list[float],
    draw_off: list[float],
    cfg: dict,
) -> list[float]:
    tz = pytz.timezone(cfg["timezone"])
    hwc = cfg["hwc"]
    block_cfg = hwc.get("block_planner", {})
    overnight_start = _parse_hhmm(block_cfg.get("overnight_window_start", "00:00"))
    overnight_end = _parse_hhmm(block_cfg.get("overnight_window_end", "06:00"))
    min_temp = float(hwc["thermal"].get("min_temp", 45))
    boost_target = float(block_cfg.get("boost_target_temp", min_temp + 5))
    step_h = hwc.get("optimization_time_step", 30) / 60.0
    lookback = int(round(18 / step_h))

    out = list(schedule_w)
    for _ in range(8):
        temps, _ = simulate_block_temperatures(
            schedule_w=out,
            start_temperature=start_temperature,
            dry_bulb=dry_bulb,
            draw_off=draw_off,
            cfg=hwc,
        )
        bad_idx = next((i for i, temp in enumerate(temps) if temp < min_temp), None)
        if bad_idx is None:
            return out

        lo = max(0, bad_idx - lookback)
        candidate_starts = [
            i
            for i in range(lo, bad_idx + 1)
            if _minute_in_window(_local_minute(grid_times_utc[i], tz), overnight_start, overnight_end)
        ]
        if not candidate_starts:
            candidate_starts = list(range(lo, bad_idx + 1))

        best = None
        best_score = (math.inf, math.inf)
        for start_idx in candidate_starts:
            candidate = list(out)
            for heat_idx in range(start_idx, bad_idx + 1):
                ctemps, _ = simulate_block_temperatures(
                    schedule_w=candidate,
                    start_temperature=start_temperature,
                    dry_bulb=dry_bulb,
                    draw_off=draw_off,
                    cfg=hwc,
                )
                if ctemps[bad_idx] >= min_temp and ctemps[heat_idx] >= boost_target:
                    break
                candidate[heat_idx] = float(hwc["thermal"]["nominal_power_w"])
            ctemps, _ = simulate_block_temperatures(
                schedule_w=candidate,
                start_temperature=start_temperature,
                dry_bulb=dry_bulb,
                draw_off=draw_off,
                cfg=hwc,
            )
            shortfall = max(0.0, min_temp - ctemps[bad_idx])
            cost = _schedule_cost_delta(out, candidate, load_cost, step_h)
            score = (shortfall, cost)
            if score < best_score:
                best = candidate
                best_score = score
        if best is None or best == out:
            return out
        out = best
    return out


def _repair_terminal_temperature(
    schedule_w: list[float],
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    start_temperature: float,
    dry_bulb: list[float],
    draw_off: list[float],
    cfg: dict,
) -> list[float]:
    hwc = cfg["hwc"]
    th = hwc["thermal"]
    target_setting = th.get("terminal_target", "current")
    if target_setting == "current":
        terminal_target = float(start_temperature)
    else:
        terminal_target = float(target_setting)
    _, terminal = simulate_block_temperatures(
        schedule_w=schedule_w,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        draw_off=draw_off,
        cfg=hwc,
    )
    if terminal >= terminal_target:
        return schedule_w
    min_lift_c = _min_block_lift_c(hwc)
    if terminal_target - terminal < min_lift_c:
        return schedule_w

    step_h = hwc.get("optimization_time_step", 30) / 60.0
    lookback_h = float(hwc.get("block_planner", {}).get("terminal_lookback_hours", 24))
    lo = max(0, len(schedule_w) - int(round(lookback_h / step_h)))
    best = schedule_w
    best_score = (max(0.0, terminal_target - terminal), math.inf)
    for start_idx in range(lo, len(schedule_w)):
        candidate = list(schedule_w)
        for heat_idx in range(start_idx, len(schedule_w)):
            _, cterminal = simulate_block_temperatures(
                schedule_w=candidate,
                start_temperature=start_temperature,
                dry_bulb=dry_bulb,
                draw_off=draw_off,
                cfg=hwc,
            )
            if cterminal >= terminal_target:
                break
            candidate[heat_idx] = float(hwc["thermal"]["nominal_power_w"])
        _, cterminal = simulate_block_temperatures(
            schedule_w=candidate,
            start_temperature=start_temperature,
            dry_bulb=dry_bulb,
            draw_off=draw_off,
            cfg=hwc,
        )
        score = (
            max(0.0, terminal_target - cterminal),
            _schedule_cost_delta(schedule_w, candidate, load_cost, step_h),
        )
        if score < best_score:
            best = candidate
            best_score = score
    return best


def build_block_plan(
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    dry_bulb: list[float],
    draw_off: list[float],
    start_temperature: float,
    cfg: dict,
) -> dict:
    """Build a fixed-speed HWC block plan and HA-compatible published attributes."""
    n = len(grid_times_utc)
    schedule = [0.0] * n
    schedule = _choose_daily_main_blocks(
        schedule,
        grid_times_utc=grid_times_utc,
        load_cost=load_cost,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        draw_off=draw_off,
        cfg=cfg,
    )
    schedule = _repair_min_temperature(
        schedule,
        grid_times_utc=grid_times_utc,
        load_cost=load_cost,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        draw_off=draw_off,
        cfg=cfg,
    )
    schedule = _repair_terminal_temperature(
        schedule,
        grid_times_utc=grid_times_utc,
        load_cost=load_cost,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        draw_off=draw_off,
        cfg=cfg,
    )
    temps, terminal_temp = simulate_block_temperatures(
        schedule_w=schedule,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        draw_off=draw_off,
        cfg=cfg["hwc"],
    )

    prefix = cfg["hwc"].get("publish_prefix", "hwc_")
    temp_key = _published_entity_id(prefix, cfg["hwc"]["predicted_temp_entity"]).split(".", 1)[1]
    power_key = _published_entity_id(prefix, cfg["hwc"]["power_plan_entity"]).split(".", 1)[1]
    cost_key = f"{prefix}unit_load_cost"
    return {
        "schedule_w": schedule,
        "temperatures": temps,
        "terminal_temperature": terminal_temp,
        "predicted_temperatures": [
            {"date": t.isoformat(), temp_key: f"{temp:.2f}"}
            for t, temp in zip(grid_times_utc, temps, strict=True)
        ],
        "deferrables_schedule": [
            {"date": t.isoformat(), power_key: f"{power:.1f}"}
            for t, power in zip(grid_times_utc, schedule, strict=True)
        ],
        "unit_load_cost_forecasts": [
            {"date": t.isoformat(), cost_key: f"{cost:.5f}"}
            for t, cost in zip(grid_times_utc, load_cost, strict=True)
        ],
    }


def build_payload(
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    wet_bulb: list[float],
    draw_off: list[float],
    start_temperature: float,
    cfg: dict,
) -> dict:
    """Assemble the EMHASS ``naive-mpc-optim`` payload for the HWC thermal load.

    Battery and PV are disabled (runtime-overridable via associations.csv), so the
    optimisation is HWC-only against the import-price forecast.
    """
    n = len(grid_times_utc)
    th = cfg["thermal"]
    return {
        "prediction_horizon": n,
        "optimization_time_step": cfg.get("optimization_time_step", 30),
        # HWC-only: disable battery + PV so this is fast and decoupled.
        "set_use_battery": False,
        "set_use_pv": False,
        "number_of_deferrable_loads": 1,
        "nominal_power_of_deferrable_loads": [th["nominal_power_w"]],
        "treat_deferrable_load_as_semi_cont": [False],
        "set_deferrable_load_single_constant": [False],
        "set_deferrable_startup_penalty": [0],
        # Import-price objective; export/PV/base-load irrelevant here.
        "load_cost_forecast": [round(float(c), 5) for c in load_cost],
        "prod_price_forecast": [0.0] * n,
        "load_power_forecast": [0] * n,
        # Wet-bulb drives the Carnot COP (the only place outdoor temp is used in tank mode).
        "outdoor_temperature_forecast": [round(float(w), 3) for w in wet_bulb],
        "def_load_config": [
            {
                "thermal_battery": {
                    "volume": th["volume_l"] / 1000.0,  # litres → m³
                    "density": th.get("density", 997),
                    "heat_capacity": th.get("heat_capacity", 4.184),
                    "supply_temperature": th["supply_temperature"],
                    "carnot_efficiency": th["carnot_efficiency"],
                    "thermal_loss": th["thermal_loss_kw"],
                    "start_temperature": round(float(start_temperature), 2),
                    "min_temperatures": [th["min_temp"]] * n,
                    "max_temperatures": [th["max_temp"]] * n,
                    "desired_temperatures": [th["desired_temp"]] * n,
                    "penalty_factor": th.get("penalty_factor", 15),
                    "thermal_inertia_time_constant": th.get(
                        "thermal_inertia_time_constant", 0.5
                    ),
                    "draw_off_demand": [round(float(d), 5) for d in draw_off],
                }
            }
        ],
        "custom_predicted_temperature_id": [
            {
                "entity_id": cfg["predicted_temp_entity"],
                "unit_of_measurement": "°C",
                "friendly_name": "HWC Predicted Tank Temp",
            }
        ],
        "custom_deferrable_forecast_id": [
            {
                "entity_id": cfg["power_plan_entity"],
                "unit_of_measurement": "W",
                "friendly_name": "HWC Planned Power",
            }
        ],
        "entity_save": True,
        "publish_prefix": cfg.get("publish_prefix", "hwc_"),
    }


# ── Home Assistant I/O ──────────────────────────────────────────────────────


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


def _ha_set_state(cfg: dict, entity_id: str, state, attributes: dict):
    return _ha_call(
        cfg,
        "POST",
        f"states/{entity_id}",
        {"state": state, "attributes": attributes},
    )


def get_tank_temperature(cfg: dict) -> float:
    state = _ha_call(cfg, "GET", f"states/{cfg['hwc']['tank_temp_entity']}")
    return float(state["state"])


def get_import_price_grid(cfg: dict, horizon_steps: int):
    """Return (grid_times_utc, load_cost) from the published DH import-price forecast,
    starting at the current 30-min interval."""
    entity = cfg["hwc"]["import_price_entity"]
    state = _ha_call(cfg, "GET", f"states/{entity}")
    forecast = state.get("attributes", {}).get("forecast", []) or []
    if not forecast:
        raise RuntimeError(f"{entity} has no 'forecast' attribute")

    now = datetime.now(timezone.utc)
    step = cfg["hwc"].get("optimization_time_step", 30)
    now_floor = now.replace(
        minute=(now.minute // step) * step, second=0, microsecond=0
    )
    rows = []
    for item in forecast:
        dt = pd.to_datetime(item["datetime"], utc=True).to_pydatetime()
        if dt >= now_floor:
            rows.append((dt, float(item["native_value"])))
    rows.sort(key=lambda r: r[0])
    rows = rows[:horizon_steps]
    if not rows:
        raise RuntimeError(f"{entity}: no future forecast points")
    return [r[0] for r in rows], [r[1] for r in rows]


def get_weather_series(cfg: dict):
    """Return (epoch_seconds, temp_c, rh_pct) from the BOM hourly weather forecast."""
    ha = cfg["home_assistant"]
    resp = _ha_call(
        cfg,
        "POST",
        "services/weather/get_forecasts?return_response=true",
        {"type": "hourly", "entity_id": ha["weather_entity"]},
    )
    forecast = (
        resp.get("service_response", {}).get(ha["weather_entity"], {}).get("forecast", [])
    )
    if not forecast:
        raise RuntimeError("weather.get_forecasts returned no forecast")

    tz = pytz.timezone(cfg["timezone"])
    epochs, temps, rhs = [], [], []
    for f in forecast:
        dt = pd.to_datetime(f["datetime"])
        if dt.tzinfo is None:
            dt = tz.localize(dt.to_pydatetime(), is_dst=None)
        epochs.append(dt.timestamp())
        temps.append(float(f["temperature"]))
        rhs.append(float(f.get("humidity", 60.0)))
    return epochs, temps, rhs


def _publish_block_plan(cfg: dict, plan: dict):
    hwc = cfg["hwc"]
    prefix = hwc.get("publish_prefix", "hwc_")
    temp_entity = _published_entity_id(prefix, hwc["predicted_temp_entity"])
    power_entity = _published_entity_id(prefix, hwc["power_plan_entity"])
    cost_entity = f"sensor.{prefix}unit_load_cost"

    _ha_set_state(
        cfg,
        temp_entity,
        plan["predicted_temperatures"][0][temp_entity.split(".", 1)[1]],
        {
            "unit_of_measurement": "°C",
            "friendly_name": "HWC Predicted Tank Temp",
            "predicted_temperatures": plan["predicted_temperatures"],
            "terminal_temperature": plan["terminal_temperature"],
        },
    )
    _ha_set_state(
        cfg,
        power_entity,
        plan["deferrables_schedule"][0][power_entity.split(".", 1)[1]],
        {
            "unit_of_measurement": "W",
            "friendly_name": "HWC Planned Power",
            "deferrables_schedule": plan["deferrables_schedule"],
        },
    )
    _ha_set_state(
        cfg,
        cost_entity,
        plan["unit_load_cost_forecasts"][0][cost_entity.split(".", 1)[1]],
        {
            "unit_of_measurement": "$/kWh",
            "friendly_name": "HWC Unit Load Cost",
            "unit_load_cost_forecasts": plan["unit_load_cost_forecasts"],
        },
    )
    logging.info("Published HWC block plan to HA (%s, %s, %s)", temp_entity, power_entity, cost_entity)


def _run_emhass(cfg: dict, payload: dict, dry_run: bool) -> dict:
    if dry_run:
        logging.info("Dry run — not POSTing to EMHASS. Payload:\n%s", json.dumps(payload, indent=2))
        return payload

    base = cfg["hwc"]["emhass_base_url"].rstrip("/")
    r = requests.post(f"{base}/action/naive-mpc-optim", json=payload, timeout=300)
    r.raise_for_status()
    logging.info("naive-mpc-optim OK (%s)", r.status_code)

    # Ask EMHASS to publish the plan (predicted temp + planned power) to HA.
    publish_payload = {
        "publish_prefix": cfg["hwc"].get("publish_prefix", "hwc_"),
        "custom_predicted_temperature_id": payload["custom_predicted_temperature_id"],
        "custom_deferrable_forecast_id": payload["custom_deferrable_forecast_id"],
        "def_load_config": [{"thermal_battery": {}}],
    }
    rp = requests.post(f"{base}/action/publish-data", json=publish_payload, timeout=60)
    rp.raise_for_status()
    logging.info("publish-data OK (%s)", rp.status_code)
    return payload


def run(cfg: dict, horizon_steps: int, dry_run: bool, extra_draw_off: list[str] | None = None) -> dict:
    grid_times, load_cost = get_import_price_grid(cfg, horizon_steps)
    grid_epoch = [t.timestamp() for t in grid_times]

    w_epoch, w_temp, w_rh = get_weather_series(cfg)
    dry_bulb = interpolate_to_grid(w_epoch, w_temp, grid_epoch)
    rh_grid = interpolate_to_grid(w_epoch, w_rh, grid_epoch)
    wet_bulb = [stull_wet_bulb(t, rh) for t, rh in zip(dry_bulb, rh_grid)]

    draw_off = build_draw_off_profile(
        grid_times,
        cfg["timezone"],
        cfg["hwc"]["draw_off"]["window_start"],
        cfg["hwc"]["draw_off"]["window_end"],
        cfg["hwc"]["draw_off"]["total_kwh"],
    )
    for spec in extra_draw_off or []:
        start_time, duration_min, total_kwh = parse_extra_draw_off(spec)
        draw_off = add_draw_off_event(
            draw_off,
            grid_times,
            cfg["timezone"],
            start_time,
            duration_min,
            total_kwh,
        )
        logging.info(
            "Added scenario draw-off: %s for %d min, %.2f kWh",
            start_time,
            duration_min,
            total_kwh,
        )

    start_temp = get_tank_temperature(cfg)

    logging.info(
        "HWC plan: horizon=%d steps, start_temp=%.1f°C, wet-bulb %.1f→%.1f°C, "
        "import $%.3f→$%.3f/kWh, draw-off total=%.2f kWh",
        len(grid_times),
        start_temp,
        min(wet_bulb),
        max(wet_bulb),
        min(load_cost),
        max(load_cost),
        sum(draw_off),
    )

    if cfg["hwc"].get("planner", "block") == "emhass":
        payload = build_payload(
            grid_times_utc=grid_times,
            load_cost=load_cost,
            wet_bulb=wet_bulb,
            draw_off=draw_off,
            start_temperature=start_temp,
            cfg=cfg["hwc"],
        )
        return _run_emhass(cfg, payload, dry_run)

    plan = build_block_plan(
        grid_times_utc=grid_times,
        load_cost=load_cost,
        dry_bulb=dry_bulb,
        draw_off=draw_off,
        start_temperature=start_temp,
        cfg=cfg,
    )
    starts = sum(
        1
        for prev, cur in zip([0.0] + plan["schedule_w"][:-1], plan["schedule_w"], strict=True)
        if prev <= 0 and cur > 0
    )
    logging.info(
        "Block plan: starts=%d, heat_kwh=%.2f, terminal_temp=%.1f°C, min_pred=%.1f°C",
        starts,
        sum(plan["schedule_w"]) / 1000.0 * (cfg["hwc"].get("optimization_time_step", 30) / 60.0),
        plan["terminal_temperature"],
        min(plan["temperatures"]),
    )
    if dry_run:
        logging.info("Dry run — not publishing to HA. Plan:\n%s", json.dumps(plan, indent=2))
        return plan
    _publish_block_plan(cfg, plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description="HWC planner (v1, modelling only)")
    parser.add_argument("--dry-run", action="store_true", help="Build + log the payload, don't POST")
    parser.add_argument(
        "--extra-draw-off",
        action="append",
        default=[],
        metavar="HH:MM[+MIN]=KWH",
        help="Scenario-only extra draw-off event, repeated by local day across the horizon",
    )
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon (timesteps)")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # load_config deep-merges the untracked config.secrets.json (HA token, EMHASS URL)
    # over config.json — same practice as forecast.py.
    cfg = load_config(args.config)
    horizon = args.horizon or cfg["hwc"].get("horizon_steps", 72)
    try:
        run(cfg, horizon, args.dry_run, args.extra_draw_off)
    except Exception:
        logging.exception("HWC planner failed")
        raise


if __name__ == "__main__":
    main()
