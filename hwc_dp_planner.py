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

State per grid position: ``(temp_bin, compressor_on, satisfied_today)``. The heat rate is a
*continuous* function of the current modelled temperature (``hwc_planner._heat_rate_c_per_hour``
taper), so there is no FULL/TOP-UP regime to carry: with no step in the rate, stopping and
restarting buys nothing, which is what killed the 53 °C short-cycle limit cycle that the old
carried-regime workaround quarantined.

Design notes live in ``docs/hwc_dp_planner.md``.
"""

from __future__ import annotations

import math
from datetime import datetime

import pytz

import hwc_planner as hp
import hwc_soc_model as soc


def _soc_params_from_cfg(th: dict, dp_cfg: dict) -> soc.SoCParams:
    """Build the two-state model parameters from config, falling back to the model defaults.

    Only the few things the planner config already knows (tank volume, water properties) are
    threaded; the COP/geometry stay at the first-cut ``SoCParams`` anchors unless explicitly
    overridden under ``hwc.dp_planner.soc`` (see docs/hwc_2state_soc_model.md "What still needs
    fitting"). This keeps the calibrated standalone model the single source of those numbers.
    """
    base = soc.SoCParams()
    over = dict(dp_cfg.get("soc", {}))
    return soc.SoCParams(
        tank_l=float(th.get("volume_l", base.tank_l)),
        density=float(th.get("density", base.density * 1000.0)) / 1000.0
        if th.get("density") else base.density,
        heat_capacity=float(th.get("heat_capacity", base.heat_capacity)),
        t_mains_c=float(over.get("t_mains_c", base.t_mains_c)),
        cop_build=float(over.get("cop_build", base.cop_build)),
        cop_rise_at_lo=float(over.get("cop_rise_at_lo", base.cop_rise_at_lo)),
        cop_rise_at_hi=float(over.get("cop_rise_at_hi", base.cop_rise_at_hi)),
        sensor_height=float(over.get("sensor_height", base.sensor_height)),
        g_width=float(over.get("g_width", base.g_width)),
        standing_loss_w=float(over.get("standing_loss_w", base.standing_loss_w)),
    )


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
    soc_state0: tuple[float, float] | None = None,
) -> dict:
    """Build a DP-optimised HWC plan in the shape published by ``hwc_planner.run``."""
    hwc = cfg["hwc"]
    th = hwc["thermal"]
    dp_cfg = hwc.get("dp_planner", {})
    tz = pytz.timezone(cfg["timezone"])
    n = len(grid_times_utc)

    # Opt-in two-state (V_hot, T_hot) decision model (docs/hwc_2state_soc_model.md), shelved and
    # off in production: kept behind the flag for a possible future revisit. When off, the default
    # continuous-rate single-temperature path below runs. The published render is unchanged either
    # way (it is a function of the chosen binary schedule, not the DP's internal temperature).
    if soc_state0 is None:
        # The daemon injects the tracked seed here (it calls hwc_planner.run, not build_dp_plan).
        _seed = dp_cfg.get("_soc_state0")
        soc_state0 = (float(_seed[0]), float(_seed[1])) if _seed else None
    if dp_cfg.get("soc_model"):
        return _build_dp_plan_soc(
            grid_times_utc=grid_times_utc, load_cost=load_cost, dry_bulb=dry_bulb,
            wet_bulb=wet_bulb, draw_off=draw_off, start_temperature=start_temperature, cfg=cfg,
            compressor_initially_on=compressor_initially_on, soc_state0=soc_state0,
        )

    transition_cost = float(hwc.get("transition_cost_aud", 0.0))

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

    # High penalties so obligations dominate energy when physically achievable, while still
    # degrading gracefully (e.g. cold start) instead of going infeasible.
    min_temp_pen = float(dp_cfg.get("min_temp_penalty_aud_per_c", 5.0))
    desired_pen = float(dp_cfg.get("desired_penalty_aud_per_c", 1.0))
    terminal_pen = float(dp_cfg.get("terminal_penalty_aud_per_c", 0.05))
    bin_c = float(dp_cfg.get("temp_bin_c", 0.25))

    # Survivors kept per binned state. 1 = min-cost only — objectively optimal here. 2 also
    # keeps the max-temp ("run a bit longer") path. NOTE: survivors_per_state=2 is present
    # ONLY because the owner wanted it kept; the 2026-06-21 sweep measured it net-negative
    # (~1c/plan, mixed sign) on objective_cost_aud — i.e. NOT objectively helpful, just
    # warmer/safer. See docs/hwc_dp_planner.md. Safe to delete this knob if never enabled.
    survivors = 2 if int(dp_cfg.get("survivors_per_state", 1)) >= 2 else 1

    terminal_setting = th.get("terminal_target", "current")
    terminal_target = (
        float(start_temperature) if terminal_setting == "current" else float(terminal_setting)
    )

    lo = min(start_temperature, min_temp) - 5.0

    def tbin(t: float) -> int:
        return int(round((t - lo) / bin_c))

    wb = wet_bulb if wet_bulb is not None else [None] * n

    # Per-position local-day bookkeeping for the daily-60 obligation.
    main_end = hp._parse_hhmm(hwc.get("main_window_end", "18:00"))
    satisfied_dates = set(hwc.get("main_satisfied_dates", []))
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

    # Forward DP. State key -> list of records, each (cost, exact_temp, prev_key, prev_idx,
    # action_on). At most `survivors` records are kept per key. history[p] is the state map
    # *before* interval p; history[n] is the terminal map.
    init_on = bool(compressor_initially_on)
    init_sat = start_temperature >= desired
    init_key = (tbin(start_temperature), init_on, init_sat)
    states: dict[tuple, list] = {init_key: [(0.0, float(start_temperature), None, None, None)]}
    history: list[dict] = []

    def _collapse(recs: list) -> list:
        # survivors==1: min-cost only (first on ties, matching strict-< accumulation).
        # survivors==2: also keep the highest exact-temp record.
        if len(recs) <= 1:
            return recs
        best_cost = min(recs, key=lambda r: r[0])
        if survivors == 1:
            return [best_cost]
        best_temp = max(recs, key=lambda r: r[1])
        return [best_cost] if best_cost is best_temp else [best_cost, best_temp]

    for p in range(n):
        history.append(states)
        nxt: dict[tuple, list] = {}
        lc = float(load_cost[p])
        amb = float(dry_bulb[p])
        wbp = wb[p]
        draw_p = float(draw_off[p])
        arr = p + 1
        arr_new_day = arr < n and day_ord[arr] != day_ord[p]
        arr_min_temp = arr < n  # penalise future reported temps, not the fixed start/terminal
        arr_oblig = arr < n and obligation_due_at[arr]

        for key, recs in states.items():
            _, on_prev, sat_prev = key
            for idx, (cost, temp, _pk, _pi, _act) in enumerate(recs):
                t1 = temp - max(0.0, temp - amb) * ua * step_h / cap - draw_p / cap
                for action_on in (False, True):
                    if action_on:
                        # Rate is a continuous function of the current temp (no carried regime);
                        # the taper has no step for the DP to arbitrage by stopping/restarting.
                        rate = hp._heat_rate_c_per_hour(th, t1, wbp)
                        t_next = min(max_temp, t1 + rate * step_h)
                        power = hp._compressor_power_w(th, t1, wbp)
                        energy = max(0.0, power) / 1000.0 * lc * step_h
                        trans = 0.0 if on_prev else transition_cost
                    else:
                        t_next = min(max_temp, t1)
                        energy = 0.0
                        trans = 0.0

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
                    nkey = (tbin(t_next), action_on, sat)
                    nxt.setdefault(nkey, []).append((ncost, t_next, key, idx, action_on))
        states = {k: _collapse(v) for k, v in nxt.items()}

    history.append(states)

    best_key, best_idx, best_cost = None, 0, math.inf
    for key, recs in states.items():
        for idx, (cost, temp, _pk, _pi, _act) in enumerate(recs):
            total = cost + max(0.0, terminal_target - temp) * terminal_pen
            if total < best_cost:
                best_cost, best_key, best_idx = total, key, idx

    actions = [False] * n
    key, idx = best_key, best_idx
    for p in range(n, 0, -1):
        _cost, _temp, pk, pi, act = history[p][key][idx]
        actions[p - 1] = bool(act)
        key, idx = pk, pi

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


def _build_dp_plan_soc(
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    dry_bulb: list[float],
    wet_bulb: list[float] | None,
    draw_off: list[float],
    start_temperature: float,
    cfg: dict,
    compressor_initially_on: bool,
    soc_state0: tuple[float, float] | None,
) -> dict:
    """Two-state ``(V_hot, T_hot)`` DP (docs/hwc_2state_soc_model.md).

    Same contract and published render as ``build_dp_plan`` — only the internal decision model
    changes. The DP still picks the binary on/off sequence; the published power/temps come from the
    exact ``hwc_planner`` model via ``_refresh_planned_power``/``assemble_plan_dict`` exactly as
    before. Obligations are evaluated on the *probe* (``probe_temp(V_hot, T_hot)``) since that is
    what the sensor and the unit's own controller see. The chosen binary is additionally forward-
    simulated through the two-state model to attach ``soc_*`` diagnostic series (consumed by nothing
    — visual only). Scaffolding is duplicated from ``build_dp_plan``; this whole path is shelved
    (off in production) and kept only for a possible future revisit.
    """
    hwc = cfg["hwc"]
    th = hwc["thermal"]
    dp_cfg = hwc.get("dp_planner", {})
    tz = pytz.timezone(cfg["timezone"])
    n = len(grid_times_utc)
    p = _soc_params_from_cfg(th, dp_cfg)
    cap = p.cap_full_kwh_per_k

    transition_cost = float(hwc.get("transition_cost_aud", 0.0))

    def _finalize(schedule_w: list[float], diag: dict | None = None) -> dict:
        plan = hp.assemble_plan_dict(
            schedule_w, grid_times_utc=grid_times_utc, load_cost=load_cost, dry_bulb=dry_bulb,
            wet_bulb=wet_bulb, draw_off=draw_off, start_temperature=start_temperature, cfg=cfg,
            transition_cost_aud=transition_cost, compressor_initially_on=compressor_initially_on,
        )
        if diag:
            plan.update(diag)
        return plan

    if n == 0:
        return _finalize([])

    step_h = hwc.get("optimization_time_step", 30) / 60.0
    dt_s = step_h * 3600.0
    max_temp = float(th.get("max_temp", 62))
    min_temp = float(th.get("min_temp", 45))
    desired = float(th.get("desired_temp", 60))
    min_temp_pen = float(dp_cfg.get("min_temp_penalty_aud_per_c", 5.0))
    desired_pen = float(dp_cfg.get("desired_penalty_aud_per_c", 1.0))
    terminal_pen = float(dp_cfg.get("terminal_penalty_aud_per_c", 0.05))
    bin_c = float(dp_cfg.get("temp_bin_c", 0.25))
    v_bin = float(dp_cfg.get("v_hot_bin", 0.05))

    terminal_setting = th.get("terminal_target", "current")

    # Seed (V_hot, T_hot). The daemon's tracker supplies the real state (conservative draw prior +
    # watermark resets); standalone we fall back to "probe ≈ T_hot, V_hot from a conservative
    # constant" so a flag flip without the tracker still runs (just less informed).
    if soc_state0 is not None:
        v0, t0 = float(soc_state0[0]), float(soc_state0[1])
    else:
        t0 = max(float(start_temperature), p.t_mains_c + soc._MIN_LIFT_C)
        v0 = float(dp_cfg.get("soc", {}).get("seed_v_hot", 0.5))
    terminal_target_probe = (
        soc.probe_temp(v0, t0, p) if terminal_setting == "current" else float(terminal_setting)
    )

    lo_t = min(t0, min_temp) - 5.0

    def tbin(t: float) -> int:
        return int(round((t - lo_t) / bin_c))

    def vbin(v: float) -> int:
        return int(round(v / v_bin))

    def coast_draw(v: float, t: float, draw_kwh: float) -> tuple[float, float]:
        # A draw removes hot water at T_hot, replaced by mains: ΔV_hot by energy equivalence.
        lift = max(t - p.t_mains_c, soc._MIN_LIFT_C)
        return soc.apply_draw(v, t, draw_kwh / (cap * lift))

    wb = wet_bulb if wet_bulb is not None else [None] * n

    # Per-position local-day bookkeeping for the daily-60 obligation (as in build_dp_plan).
    main_end = hp._parse_hhmm(hwc.get("main_window_end", "18:00"))
    satisfied_dates = set(hwc.get("main_satisfied_dates", []))
    local_dates = [t.astimezone(tz).date() for t in grid_times_utc]
    day_ord = [d.toordinal() for d in local_dates]
    local_minute = [hp._local_minute(t, tz) for t in grid_times_utc]
    deadline_pos: dict[int, int] = {}
    for pos in range(n):
        if local_minute[pos] <= main_end:
            deadline_pos[day_ord[pos]] = pos
    obligation_due_at = [False] * n
    for d, pos in deadline_pos.items():
        if local_dates[pos].isoformat() not in satisfied_dates:
            obligation_due_at[pos] = True

    init_on = bool(compressor_initially_on)
    init_sat = soc.probe_temp(v0, t0, p) >= desired
    init_key = (vbin(v0), tbin(t0), init_on, init_sat)
    # records: (cost, V_hot, T_hot, prev_key, prev_idx, action_on)
    states: dict[tuple, list] = {init_key: [(0.0, v0, t0, None, None, None)]}
    history: list[dict] = []

    def _collapse(recs: list) -> list:
        return [min(recs, key=lambda r: r[0])] if len(recs) > 1 else recs

    for pos in range(n):
        history.append(states)
        nxt: dict[tuple, list] = {}
        lc = float(load_cost[pos])
        wbp = wb[pos]
        draw_p = float(draw_off[pos])
        arr = pos + 1
        arr_new_day = arr < n and day_ord[arr] != day_ord[pos]
        arr_min_temp = arr < n
        arr_oblig = arr < n and obligation_due_at[arr]

        for key, recs in states.items():
            _, _, on_prev, sat_prev = key  # key = (v_hot_bin, t_hot_bin, on, sat)
            for idx, (cost, v, t, _pk, _pi, _act) in enumerate(recs):
                vc, tc = coast_draw(v, t, draw_p)           # coast/draw before the action
                probe_c = soc.probe_temp(vc, tc, p)         # power model input = coasted probe
                for action_on in (False, True):
                    if action_on:
                        power = hp._compressor_power_w(th, probe_c, wbp)
                        vn, tn = soc.step(vc, tc, on=True, dt_s=dt_s, p_elec_w=power, p=p)
                        tn = min(max_temp, tn)
                        energy = max(0.0, power) / 1000.0 * lc * step_h
                        trans = 0.0 if on_prev else transition_cost
                    else:
                        vn, tn = soc.step(vc, tc, on=False, dt_s=dt_s, p_elec_w=0.0, p=p)
                        energy = 0.0
                        trans = 0.0
                    probe_n = soc.probe_temp(vn, tn, p)

                    sat = (probe_n >= desired) if arr_new_day else (sat_prev or probe_n >= desired)
                    pen = 0.0
                    if arr_min_temp and probe_n < min_temp:
                        pen += (min_temp - probe_n) * min_temp_pen
                    if arr_oblig and not sat:
                        pen += max(0.0, desired - probe_n) * desired_pen

                    ncost = cost + energy + trans + pen
                    nkey = (vbin(vn), tbin(tn), action_on, sat)
                    nxt.setdefault(nkey, []).append((ncost, vn, tn, key, idx, action_on))
        states = {k: _collapse(v) for k, v in nxt.items()}

    history.append(states)

    best_key, best_idx, best_cost = None, 0, math.inf
    for key, recs in states.items():
        for idx, (cost, v, t, _pk, _pi, _act) in enumerate(recs):
            total = cost + max(0.0, terminal_target_probe - soc.probe_temp(v, t, p)) * terminal_pen
            if total < best_cost:
                best_cost, best_key, best_idx = total, key, idx

    actions = [False] * n
    key, idx = best_key, best_idx
    for pos in range(n, 0, -1):
        _c, _v, _t, pk, pi, act = history[pos][key][idx]
        actions[pos - 1] = bool(act)
        key, idx = pk, pi

    binary = [1.0 if a else 0.0 for a in actions]
    schedule_w = hp._refresh_planned_power(
        binary, start_temperature=start_temperature, dry_bulb=dry_bulb, wet_bulb=wet_bulb,
        draw_off=draw_off, cfg=hwc,
    )

    # Diagnostics: forward-sim the chosen binary through the two-state model (visual only).
    v, t = v0, t0
    diag_v, diag_t, diag_probe = [], [], []
    for pos in range(n):
        vc, tc = coast_draw(v, t, float(draw_off[pos]))
        if actions[pos]:
            power = hp._compressor_power_w(th, soc.probe_temp(vc, tc, p), wb[pos])
            v, t = soc.step(vc, tc, on=True, dt_s=dt_s, p_elec_w=power, p=p)
            t = min(max_temp, t)
        else:
            v, t = soc.step(vc, tc, on=False, dt_s=dt_s, p_elec_w=0.0, p=p)
        diag_v.append(round(v, 4))
        diag_t.append(round(t, 2))
        diag_probe.append(round(soc.probe_temp(v, t, p), 2))
    soc_forecast = [
        {"date": gt.isoformat(), "v_hot": vv, "t_hot": tt, "probe": pr}
        for gt, vv, tt, pr in zip(grid_times_utc, diag_v, diag_t, diag_probe, strict=True)
    ]
    diag = {
        "soc_v_hot": diag_v, "soc_t_hot": diag_t, "soc_probe": diag_probe,
        "soc_state0": [round(v0, 4), round(t0, 2)], "soc_forecast": soc_forecast,
    }
    return _finalize(schedule_w, diag)
