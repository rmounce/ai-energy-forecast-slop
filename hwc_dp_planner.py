"""Shadow dynamic-programming planner for the HWC heat-pump schedule.

This module is deliberately pure: no Home Assistant I/O and no dependency on
``hwc_planner.py``. V1 uses the same single-node thermal assumptions as the live
block planner so optimiser behaviour can be evaluated separately from physical
model changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DpConfig:
    step_minutes: int = 5
    temp_bin_c: float = 0.01
    min_state_temp_c: float = 35.0
    max_state_temp_c: float = 60.0
    start_penalty_aud: float = 0.05
    floor_penalty_aud_per_c2: float = 5.0
    target_miss_penalty_aud: float = 20.0
    terminal_penalty_aud_per_c2: float = 0.02
    target_tolerance_c: float = 0.05
    main_window_end: str = "18:00"


@dataclass(frozen=True)
class DpResult:
    schedule_w: list[float]
    temperatures: list[float]
    terminal_temperature: float
    objective_cost: float
    objective_breakdown: dict[str, float]
    required_target_dates: list[str]
    target_satisfied_dates: list[str]
    starts: int


StateKey = tuple[int, int, int]
Parent = tuple[StateKey, int]


def thermal_capacity_kwh_per_c(th: dict) -> float:
    litres = float(th["volume_l"])
    density_kg_per_l = float(th.get("density", 997)) / 1000.0
    return litres * density_kg_per_l * float(th.get("heat_capacity", 4.184)) / 3600.0


def heat_rate_c_per_hour(th: dict, temp_c: float, wet_bulb_c: float | None = None) -> float:
    base = float(th.get("heat_rate_c_per_hour", 5.2))
    top_up = th.get("top_up_heat_rate_c_per_hour")
    top_up_start = th.get("top_up_start_temp_c")
    if top_up is not None and top_up_start is not None and temp_c >= float(top_up_start):
        base = float(top_up)
    if wet_bulb_c is not None:
        reference_wb = th.get("heat_rate_reference_wet_bulb_c")
        slope = th.get("heat_rate_wet_bulb_slope_c_per_c")
        if reference_wb is not None and slope is not None:
            base += (float(wet_bulb_c) - float(reference_wb)) * float(slope)
    min_rate = th.get("heat_rate_min_c_per_hour")
    max_rate = th.get("heat_rate_max_c_per_hour")
    if min_rate is not None:
        base = max(float(min_rate), base)
    if max_rate is not None:
        base = min(float(max_rate), base)
    return base


def transition_temperature(
    *,
    temp_c: float,
    action_heat: bool,
    dry_bulb_c: float,
    wet_bulb_c: float | None,
    draw_off_kwh: float,
    hwc_cfg: dict,
    step_h: float,
) -> float:
    th = hwc_cfg["thermal"]
    cap_kwh_per_c = thermal_capacity_kwh_per_c(th)
    ua_kw_per_c = float(th.get("standing_loss_ua_kw_per_c", 0.0025))
    max_temp = float(th.get("max_temp", 60.0))

    temp = float(temp_c)
    loss_kwh = max(0.0, temp - float(dry_bulb_c)) * ua_kw_per_c * step_h
    temp -= loss_kwh / cap_kwh_per_c
    temp -= float(draw_off_kwh) / cap_kwh_per_c
    if action_heat:
        temp += heat_rate_c_per_hour(th, temp, wet_bulb_c) * step_h
    return min(max_temp, temp)


def required_target_dates(
    grid_times_utc: list[datetime],
    *,
    tz_name: str,
    main_window_end: str,
) -> set[str]:
    if not grid_times_utc:
        return set()
    tz = ZoneInfo(tz_name)
    hour, minute = (int(part) for part in main_window_end.split(":", 1))
    start = grid_times_utc[0]
    step = _grid_step(grid_times_utc)
    end = grid_times_utc[-1] + step

    local_start = start.astimezone(tz).date()
    local_end = end.astimezone(tz).date()
    days = (local_end - local_start).days
    out = set()
    for offset in range(days + 1):
        day = local_start + timedelta(days=offset)
        local_deadline = datetime(
            day.year, day.month, day.day, hour, minute, tzinfo=tz
        )
        deadline_utc = local_deadline.astimezone(start.tzinfo)
        if start <= deadline_utc <= end:
            out.add(day.isoformat())
    return out


def solve(
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    dry_bulb: list[float],
    wet_bulb: list[float],
    draw_off: list[float],
    start_temperature: float,
    hwc_cfg: dict,
    tz_name: str,
    already_satisfied_dates: set[str] | None = None,
    dp_cfg: DpConfig | None = None,
) -> DpResult:
    cfg = dp_cfg or DpConfig()
    n = len(grid_times_utc)
    if not (len(load_cost) == len(dry_bulb) == len(wet_bulb) == len(draw_off) == n):
        raise ValueError("DP input series must have matching lengths")
    if n == 0:
        raise ValueError("DP horizon must not be empty")

    tz = ZoneInfo(tz_name)
    th = hwc_cfg["thermal"]
    step_h = cfg.step_minutes / 60.0
    nominal_power_w = float(th["nominal_power_w"])
    min_temp = float(th.get("min_temp", 45.0))
    target_temp = float(th.get("desired_temp", 60.0))
    target_threshold = target_temp - cfg.target_tolerance_c
    satisfied_initial = set(already_satisfied_dates or set())
    required_dates = required_target_dates(
        grid_times_utc, tz_name=tz_name, main_window_end=cfg.main_window_end
    )

    start_date = grid_times_utc[0].astimezone(tz).date().isoformat()
    start_satisfied = start_date in satisfied_initial or start_temperature >= target_threshold
    start_key = (_temp_to_bin(start_temperature, cfg), 0, int(start_satisfied))
    frontier: dict[StateKey, tuple[float, dict[str, float]]] = {
        start_key: (0.0, _empty_breakdown())
    }
    parents: list[dict[StateKey, Parent]] = []

    for idx, t in enumerate(grid_times_utc):
        next_t = t + timedelta(minutes=cfg.step_minutes)
        cur_date = t.astimezone(tz).date().isoformat()
        next_date = next_t.astimezone(tz).date().isoformat()
        next_frontier: dict[StateKey, tuple[float, dict[str, float]]] = {}
        step_parents: dict[StateKey, Parent] = {}

        for key, (cost_so_far, breakdown) in frontier.items():
            temp_bin, prev_action, satisfied_flag = key
            temp = _bin_to_temp(temp_bin, cfg)
            for action in (0, 1):
                next_temp_raw = transition_temperature(
                    temp_c=temp,
                    action_heat=bool(action),
                    dry_bulb_c=dry_bulb[idx],
                    wet_bulb_c=wet_bulb[idx],
                    draw_off_kwh=draw_off[idx],
                    hwc_cfg=hwc_cfg,
                    step_h=step_h,
                )
                next_temp = _bounded_temp(next_temp_raw, cfg)
                day_satisfied = bool(satisfied_flag) or next_temp_raw >= target_threshold
                target_penalty = 0.0
                next_satisfied_flag = int(day_satisfied)
                if next_date != cur_date:
                    if cur_date in required_dates and not day_satisfied:
                        target_penalty = cfg.target_miss_penalty_aud
                    next_satisfied_flag = int(next_temp_raw >= target_threshold)

                energy_cost = (
                    nominal_power_w / 1000.0 * float(load_cost[idx]) * step_h if action else 0.0
                )
                start_penalty = (
                    cfg.start_penalty_aud if action and not prev_action else 0.0
                )
                floor_shortfall = max(0.0, min_temp - next_temp_raw)
                floor_penalty = floor_shortfall * floor_shortfall * cfg.floor_penalty_aud_per_c2

                add_cost = energy_cost + start_penalty + floor_penalty + target_penalty
                next_key = (_temp_to_bin(next_temp, cfg), action, next_satisfied_flag)
                new_breakdown = dict(breakdown)
                new_breakdown["energy_cost_aud"] += energy_cost
                new_breakdown["start_penalty_aud"] += start_penalty
                new_breakdown["floor_penalty_aud"] += floor_penalty
                new_breakdown["target_miss_penalty_aud"] += target_penalty
                new_cost = cost_so_far + add_cost

                existing = next_frontier.get(next_key)
                if existing is None or new_cost < existing[0]:
                    next_frontier[next_key] = (new_cost, new_breakdown)
                    step_parents[next_key] = (key, action)

        frontier = next_frontier
        parents.append(step_parents)

    terminal_date = (grid_times_utc[-1] + timedelta(minutes=cfg.step_minutes)).astimezone(tz).date().isoformat()
    best_key = None
    best_cost = math.inf
    best_breakdown = None
    for key, (cost, breakdown) in frontier.items():
        terminal_temp = _bin_to_temp(key[0], cfg)
        terminal_shortfall = max(0.0, float(start_temperature) - terminal_temp)
        terminal_penalty = terminal_shortfall * terminal_shortfall * cfg.terminal_penalty_aud_per_c2
        target_penalty = 0.0
        if terminal_date in required_dates and not bool(key[2]):
            target_penalty = cfg.target_miss_penalty_aud
        total = cost + terminal_penalty + target_penalty
        if total < best_cost:
            best_key = key
            best_cost = total
            best_breakdown = dict(breakdown)
            best_breakdown["terminal_penalty_aud"] += terminal_penalty
            best_breakdown["target_miss_penalty_aud"] += target_penalty

    if best_key is None or best_breakdown is None:
        raise RuntimeError("DP solver found no terminal state")

    actions = _backtrack_actions(best_key, parents)
    schedule = [nominal_power_w if action else 0.0 for action in actions]
    temperatures, terminal_temp = simulate_schedule(
        schedule_w=schedule,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        wet_bulb=wet_bulb,
        draw_off=draw_off,
        hwc_cfg=hwc_cfg,
        step_h=step_h,
    )
    satisfied_dates = _satisfied_dates(
        grid_times_utc, temperatures, terminal_temp, tz_name, target_threshold
    )
    starts = sum(1 for prev, cur in zip([0] + actions[:-1], actions, strict=True) if cur and not prev)
    best_breakdown["total_cost_aud"] = best_cost
    return DpResult(
        schedule_w=schedule,
        temperatures=temperatures,
        terminal_temperature=round(terminal_temp, 2),
        objective_cost=best_cost,
        objective_breakdown={k: round(v, 5) for k, v in best_breakdown.items()},
        required_target_dates=sorted(required_dates),
        target_satisfied_dates=sorted(satisfied_dates),
        starts=starts,
    )


def simulate_schedule(
    *,
    schedule_w: list[float],
    start_temperature: float,
    dry_bulb: list[float],
    wet_bulb: list[float],
    draw_off: list[float],
    hwc_cfg: dict,
    step_h: float,
) -> tuple[list[float], float]:
    temp = float(start_temperature)
    temps = []
    for power_w, ambient_c, heat_wb, draw_kwh in zip(
        schedule_w, dry_bulb, wet_bulb, draw_off, strict=True
    ):
        temps.append(round(temp, 2))
        temp = transition_temperature(
            temp_c=temp,
            action_heat=power_w > 0,
            dry_bulb_c=ambient_c,
            wet_bulb_c=heat_wb,
            draw_off_kwh=draw_kwh,
            hwc_cfg=hwc_cfg,
            step_h=step_h,
        )
    return temps, temp


def _grid_step(grid_times_utc: list[datetime]) -> timedelta:
    if len(grid_times_utc) >= 2:
        return grid_times_utc[1] - grid_times_utc[0]
    return timedelta(minutes=5)


def _empty_breakdown() -> dict[str, float]:
    return {
        "energy_cost_aud": 0.0,
        "start_penalty_aud": 0.0,
        "floor_penalty_aud": 0.0,
        "target_miss_penalty_aud": 0.0,
        "terminal_penalty_aud": 0.0,
    }


def _temp_to_bin(temp_c: float, cfg: DpConfig) -> int:
    bounded = _bounded_temp(temp_c, cfg)
    return int(math.floor(bounded / cfg.temp_bin_c + 0.5))


def _bin_to_temp(temp_bin: int, cfg: DpConfig) -> float:
    return temp_bin * cfg.temp_bin_c


def _bounded_temp(temp_c: float, cfg: DpConfig) -> float:
    return min(cfg.max_state_temp_c, max(cfg.min_state_temp_c, float(temp_c)))


def _backtrack_actions(final_key: StateKey, parents: list[dict[StateKey, Parent]]) -> list[int]:
    actions = []
    key = final_key
    for step_parents in reversed(parents):
        prev_key, action = step_parents[key]
        actions.append(action)
        key = prev_key
    actions.reverse()
    return actions


def _satisfied_dates(
    grid_times_utc: list[datetime],
    temperatures: list[float],
    terminal_temperature: float,
    tz_name: str,
    target_temp: float,
) -> set[str]:
    tz = ZoneInfo(tz_name)
    out = {
        t.astimezone(tz).date().isoformat()
        for t, temp in zip(grid_times_utc, temperatures, strict=True)
        if temp >= target_temp
    }
    step = _grid_step(grid_times_utc)
    temps_after = list(temperatures[1:]) + [terminal_temperature]
    out.update(
        (t + step).astimezone(tz).date().isoformat()
        for t, temp in zip(grid_times_utc, temps_after, strict=True)
        if temp >= target_temp
    )
    return out
