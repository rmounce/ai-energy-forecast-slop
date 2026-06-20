#!/usr/bin/env python3
"""Dynamic-programming HWC planner (opt-in via ``hwc.planner == "dp"``).

Chooses the compressor on/off sequence by minimising a single monetary objective
— import energy + a per-start ``transition_cost_aud`` — subject to *soft, high-penalty*
min-temperature and daily-60 °C obligations. Short cycles are discouraged purely by the
transition cost; there is no hard minimum-runtime rule (2026-06-20 decision, see
``docs/hwc_thermal_characterisation.md``).

The DP only selects the binary on/off sequence. The published power/temperature plan is
then produced by the *exact* ``hwc_planner`` thermal model
(``_refresh_planned_power`` + ``simulate_block_temperatures`` via ``assemble_plan_dict``),
so a DP plan is identically shaped, scored and comparable with a block-planner plan.
Temperature binning is therefore an internal approximation of the DP's own cost/feasibility
estimate only — it never reaches the published numbers.

State per grid position: ``(temp_bin, compressor_on, regime, satisfied_today)``.
``regime`` (full-reheat vs top-up) is carried because the heat-rate model latches on the
*block-start* temperature (a cold reheat keeps the full rate even past ``top_up_start_temp_c``).

Design notes live in ``docs/hwc_dp_planner.md``.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytz

import hwc_planner as hp

# regime codes
_OFF = 0
_FULL = 1
_TOPUP = 2


def _rate_for_regime(th: dict, regime: int, wet_bulb, temp: float) -> float:
    """Heat rate honouring a *carried* regime (not the instantaneous temp).

    ``_heat_rate_c_per_hour`` selects the top-up base when its ``heat_block_start_temp_c``
    argument is ``>= top_up_start_temp_c``. We pass a sentinel so the regime bit — not the
    current temp — decides, matching ``simulate_block_temperatures``' block-start latch.
    """
    sentinel = 1e9 if regime == _TOPUP else -1e9
    return hp._heat_rate_c_per_hour(th, temp, wet_bulb, heat_block_start_temp_c=sentinel)


def build_dp_plan(
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    dry_bulb: list[float],
    wet_bulb: list[float] | None = None,
    draw_off: list[float],
    start_temperature: float,
    cfg: dict,
    compressor_initially_on: bool = False,
) -> dict:
    """Build a DP-optimised HWC plan in the same shape as ``build_block_plan``."""
    hwc = cfg["hwc"]
    th = hwc["thermal"]
    block_cfg = hwc.get("block_planner", {})
    dp_cfg = hwc.get("dp_planner", {})
    tz = pytz.timezone(cfg["timezone"])
    n = len(grid_times_utc)

    transition_cost = float(block_cfg.get("transition_cost_aud", 0.0))

    def _finalize(schedule_w: list[float]) -> dict:
        return hp.assemble_plan_dict(
            schedule_w,
            grid_times_utc=grid_times_utc,
            load_cost=load_cost,
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            draw_off=draw_off,
            start_temperature=start_temperature,
            cfg=cfg,
            transition_cost_aud=transition_cost,
            compressor_initially_on=compressor_initially_on,
        )

    if n == 0:
        return _finalize([])

    step_h = hwc.get("optimization_time_step", 30) / 60.0
    cap = hp._thermal_capacity_kwh_per_c(th)
    ua = float(th.get("standing_loss_ua_kw_per_c", 0.0025))
    max_temp = float(th.get("max_temp", 62))
    min_temp = float(th.get("min_temp", 45))
    desired = float(th.get("desired_temp", 60))
    top_up_start = th.get("top_up_start_temp_c")
    top_up_start = float(top_up_start) if top_up_start is not None else None

    # High penalties so obligations dominate energy when physically achievable, while still
    # degrading gracefully (e.g. cold start) instead of going infeasible.
    min_temp_pen = float(dp_cfg.get("min_temp_penalty_aud_per_c", 5.0))
    desired_pen = float(dp_cfg.get("desired_penalty_aud_per_c", 1.0))
    terminal_pen = float(dp_cfg.get("terminal_penalty_aud_per_c", 0.05))
    bin_c = float(dp_cfg.get("temp_bin_c", 0.25))

    terminal_setting = th.get("terminal_target", "current")
    terminal_target = (
        float(start_temperature) if terminal_setting == "current" else float(terminal_setting)
    )

    lo = min(start_temperature, min_temp) - 5.0

    def tbin(t: float) -> int:
        return int(round((t - lo) / bin_c))

    def regime_for_start(t1: float) -> int:
        if top_up_start is not None and t1 >= top_up_start:
            return _TOPUP
        return _FULL

    wb = wet_bulb if wet_bulb is not None else [None] * n

    # Per-position local-day bookkeeping for the daily-60 obligation.
    main_end = hp._parse_hhmm(block_cfg.get("main_window_end", "18:00"))
    satisfied_dates = set(block_cfg.get("main_satisfied_dates", []))
    local_dates = [t.astimezone(tz).date() for t in grid_times_utc]
    day_ord = [d.toordinal() for d in local_dates]
    local_minute = [hp._local_minute(t, tz) for t in grid_times_utc]
    deadline_pos: dict[int, int] = {}
    for p in range(n):
        if local_minute[p] <= main_end:
            deadline_pos[day_ord[p]] = p  # last in-window position that day
    obligation_due_at = [False] * n
    for d, p in deadline_pos.items():
        if local_dates[p].isoformat() not in satisfied_dates:
            obligation_due_at[p] = True

    # Forward DP. State key -> (cost, exact_temp, prev_key, action_on). history[p] is the
    # state map *before* interval p; history[n] is the terminal map.
    init_on = bool(compressor_initially_on)
    init_regime = regime_for_start(start_temperature) if init_on else _OFF
    init_sat = start_temperature >= desired
    init_key = (tbin(start_temperature), init_on, init_regime, init_sat)
    states: dict[tuple, tuple] = {init_key: (0.0, float(start_temperature), None, None)}
    history: list[dict] = []

    for p in range(n):
        history.append(states)
        nxt: dict[tuple, tuple] = {}
        lc = float(load_cost[p])
        amb = float(dry_bulb[p])
        wbp = wb[p]
        draw_p = float(draw_off[p])
        arr = p + 1
        arr_new_day = arr < n and day_ord[arr] != day_ord[p]
        arr_min_temp = arr < n  # penalise future reported temps, not the fixed start/terminal
        arr_oblig = arr < n and obligation_due_at[arr]

        for key, (cost, temp, _prev, _act) in states.items():
            _, on_prev, regime_prev, sat_prev = key
            t1 = temp - max(0.0, temp - amb) * ua * step_h / cap - draw_p / cap
            for action_on in (False, True):
                if action_on:
                    if on_prev and regime_prev != _OFF:
                        regime = regime_prev
                    else:
                        regime = regime_for_start(t1)
                    rate = _rate_for_regime(th, regime, wbp, t1)
                    t_next = min(max_temp, t1 + rate * step_h)
                    power = hp._compressor_power_w(th, t1, wbp)
                    energy = max(0.0, power) / 1000.0 * lc * step_h
                    trans = 0.0 if on_prev else transition_cost
                    nregime = regime
                else:
                    t_next = min(max_temp, t1)
                    energy = 0.0
                    trans = 0.0
                    nregime = _OFF

                if arr_new_day:
                    sat = t_next >= desired
                else:
                    sat = sat_prev or (t_next >= desired)

                pen = 0.0
                if arr_min_temp and t_next < min_temp:
                    pen += (min_temp - t_next) * min_temp_pen
                if arr_oblig and not sat:
                    pen += max(0.0, desired - t_next) * desired_pen

                ncost = cost + energy + trans + pen
                nkey = (tbin(t_next), action_on, nregime, sat)
                cur = nxt.get(nkey)
                if cur is None or ncost < cur[0]:
                    nxt[nkey] = (ncost, t_next, key, action_on)
        states = nxt

    history.append(states)

    best_key, best_cost = None, math.inf
    for key, (cost, temp, _prev, _act) in states.items():
        total = cost + max(0.0, terminal_target - temp) * terminal_pen
        if total < best_cost:
            best_cost, best_key = total, key

    actions = [False] * n
    key = best_key
    for p in range(n, 0, -1):
        _cost, _temp, prev, act = history[p][key]
        actions[p - 1] = bool(act)
        key = prev

    binary = [1.0 if a else 0.0 for a in actions]
    schedule_w = hp._refresh_planned_power(
        binary,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        wet_bulb=wet_bulb,
        draw_off=draw_off,
        cfg=hwc,
    )
    return _finalize(schedule_w)
