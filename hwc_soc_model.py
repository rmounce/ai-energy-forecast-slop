#!/usr/bin/env python3
"""Standalone two-state ``(V_hot, T_hot)`` stratified-tank forward model (HWC).

Pure physics, **no wiring into the DP planner** — this is step 1 of
``docs/hwc_2state_soc_model.md`` (the validate-before-touching-production step). It replaces the
discontinuous FULL/TOP-UP heat-rate latch in ``hwc_dp_planner.py`` with a continuous two-state
tank: ``V_hot`` (fraction of the tank above the thermocline) and ``T_hot`` (that hot zone's
temperature). See ``docs/hwc_thermal_characterisation.md`` Findings 1–5 for the empirical basis.

Three pieces, all scalar and side-effect-free:

* :func:`step` — advance ``(V_hot, T_hot)`` one timestep given compressor on/off + electrical W.
  Heating fills ``V_hot`` first (the probe-blind *build* phase, ~const ``T_hot``, high COP), then
  raises ``T_hot`` (the *rise* phase, COP collapsing). Energy that crosses ``V_hot = 1`` mid-step
  is split across the two regimes so the boundary conserves energy and carries no discontinuity.
* :func:`probe_temp` — the observation map ``g(V_hot, T_hot)``: a **finite-width smooth** blend of
  ``T_hot`` (thermocline above the sensor) and ``T_mains`` (below). The smoothness is load-bearing:
  a step at ``V_hot ≈ 0.5`` would reincarnate the 53 °C latch the DP arbitraged (the short-cycle
  bug). The map lives in observation, never in cost.
* :func:`apply_draw` — a draw drops ``V_hot`` at fixed ``T_hot`` (hot off the top, mains in the
  bottom).

:func:`replay_reheat` forward-integrates a measured compressor-on run (a DataFrame with
``power_w`` + ``probe_ctrl``) so the structure can be checked against real cycles; the parameters
in :class:`SoCParams` are **first-cut** (marked ``TODO fit``) — refine them with
``hwc_soc_calibrate.py --mode replay`` as cycles accumulate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Below this hot/cold temperature gap the build-phase volume update is ill-conditioned (you cannot
# grow a "hot" zone that is not hotter than mains); clamp the denominator to keep dV finite.
_MIN_LIFT_C = 1.0


@dataclass(frozen=True)
class SoCParams:
    """First-cut parameters for the two-state tank model.

    Values marked ``TODO fit`` are placeholders pending more metered cycles (see
    ``docs/hwc_2state_soc_model.md`` "What still needs fitting"). The *structure* is settled; these
    numbers are not. Geometry/COP anchors are from ``docs/hwc_thermal_characterisation.md``.
    """

    # Tank sensible capacity: 222 L heated volume (manual), water cp.
    tank_l: float = 222.0
    density: float = 0.997          # kg/L
    heat_capacity: float = 4.186    # kJ/kg·K

    # Seasonal mains/inlet temperature — a model PARAMETER (no water-side sensor logs). Adelaide
    # winter; the draw cliff bottoming at ~35 °C with a 0.50-centred g anchors the cold leg.
    t_mains_c: float = 18.0         # TODO fit (seasonal)

    # COP indexed on regime as a first cut. Build = low condensing temp, high COP (Finding 1);
    # rise = condensing temp climbs with T_hot, COP collapses 2.3 → 1.75 over 53 → 60 (Finding 3).
    cop_build: float = 2.6          # TODO fit (needs the exhaust proxy; probe can't give it)
    cop_rise_lo_c: float = 53.0
    cop_rise_hi_c: float = 60.0
    cop_rise_at_lo: float = 2.3     # TODO fit
    cop_rise_at_hi: float = 1.75    # TODO fit
    cop_floor: float = 1.4          # clamp the linear extrapolation
    cop_ceil: float = 3.2

    # Observation map g(V_hot, T_hot): thermocline crossing height + finite transition width.
    sensor_height: float = 0.50     # manual implies 45–55 %; split the difference
    g_width: float = 0.10           # logistic scale in V_hot units (CRITICAL: > 0, finite — no step)

    # Standing loss acts mainly on T_hot (~0.3 °C/h ≈ 75 W upper bound, Finding 5).
    standing_loss_w: float = 75.0   # TODO fit (draw-confounded upper bound)

    @property
    def cap_full_kwh_per_k(self) -> float:
        """Whole-tank sensible heat capacity, kWh per K (≈ 0.257)."""
        return self.tank_l * self.density * self.heat_capacity / 3600.0


def cop_rise(t_hot_c: float, p: SoCParams) -> float:
    """Rise-phase COP, linear in T_hot between the two anchors, clamped to a sane range."""
    span = p.cop_rise_hi_c - p.cop_rise_lo_c
    frac = (t_hot_c - p.cop_rise_lo_c) / span if span else 0.0
    cop = p.cop_rise_at_lo + frac * (p.cop_rise_at_hi - p.cop_rise_at_lo)
    return min(p.cop_ceil, max(p.cop_floor, cop))


def probe_temp(v_hot: float, t_hot: float, p: SoCParams) -> float:
    """Observation map ``g(V_hot, T_hot)`` → the mid-tank control probe (°C).

    The probe reads ``T_hot`` when the thermocline is above the sensor (``V_hot`` above
    ``sensor_height``) and ``T_mains`` when below, through a **finite-width logistic** transition.
    At ``V_hot = sensor_height`` it returns the midpoint ``(T_mains + T_hot)/2`` — the ~35 °C draw
    cliff anchor (Finding 5). The finite width is the whole point: a hard threshold here is the
    53 °C FULL/TOP-UP latch reincarnated in the observation layer.
    """
    x = (v_hot - p.sensor_height) / p.g_width
    w = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))  # logistic, overflow-guarded
    return p.t_mains_c + w * (t_hot - p.t_mains_c)


def apply_draw(v_hot: float, t_hot: float, draw_frac: float) -> tuple[float, float]:
    """A hot-water draw of ``draw_frac`` of the tank: ``V_hot`` falls, ``T_hot`` unchanged."""
    return max(0.0, v_hot - draw_frac), t_hot


def step(
    v_hot: float, t_hot: float, on: bool, dt_s: float, p_elec_w: float, p: SoCParams
) -> tuple[float, float]:
    """Advance ``(V_hot, T_hot)`` by ``dt_s`` seconds.

    Standing loss always cools ``T_hot``. When ``on`` with ``p_elec_w > 0``, delivered heat
    (``COP · P_elec``) first fills ``V_hot`` toward 1 at ``cop_build`` (build), then any surplus
    raises ``T_hot`` at ``cop_rise`` (rise). A step that crosses ``V_hot = 1`` is split between the
    two so the regime boundary conserves energy and is continuous — never a latch.
    """
    cap = p.cap_full_kwh_per_k

    # Standing loss on the hot zone (acts whether heating or not; dominated while heating).
    t_hot -= (p.standing_loss_w / 1000.0) * (dt_s / 3600.0) / cap

    if on and p_elec_w > 0.0:
        elec_kwh = (p_elec_w / 1000.0) * (dt_s / 3600.0)

        # Build: grow V_hot at ~const T_hot until the hot zone fills the tank.
        if v_hot < 1.0:
            lift = max(t_hot - p.t_mains_c, _MIN_LIFT_C)
            kwh_to_full = (1.0 - v_hot) * cap * lift / p.cop_build
            if elec_kwh <= kwh_to_full:
                v_hot += elec_kwh * p.cop_build / (cap * lift)
                elec_kwh = 0.0
            else:
                v_hot = 1.0
                elec_kwh -= kwh_to_full

        # Rise: surplus heat (incl. the spill from a boundary-crossing step) raises T_hot.
        if elec_kwh > 0.0 and v_hot >= 1.0:
            t_hot += elec_kwh * cop_rise(t_hot, p) / cap

    return min(1.0, max(0.0, v_hot)), t_hot


def replay_reheat(run, p: SoCParams, v_hot0: float, t_hot0: float | None = None):
    """Forward-integrate a measured compressor-on run; return predicted probe + diagnostics.

    ``run`` is a DataFrame (local-naive or tz-aware index) with ``power_w`` and ``probe_ctrl``.
    ``t_hot0`` defaults to the first probe reading (the hot zone the sensor sees at the start).
    Returns ``(probe_pred: list, info: dict)`` where ``info`` has the modelled build-complete time,
    final probe, and probe RMSE vs the measured trace — the harness for fitting parameters and for
    checking the model reproduces the blind→rise flat-then-climb shape.
    """
    import numpy as np  # local import keeps the core model dependency-light

    idx = run.index
    power = run["power_w"].fillna(0.0).to_numpy()
    probe_obs = run["probe_ctrl"].to_numpy()
    secs = idx.to_series().diff().dt.total_seconds().fillna(0.0).to_numpy()

    v_hot, t_hot = float(v_hot0), float(t_hot0 if t_hot0 is not None else probe_obs[0])
    probe_pred, v_series = [], []
    build_done_i = None
    for i in range(len(idx)):
        if i > 0:
            v_hot, t_hot = step(v_hot, t_hot, on=True, dt_s=float(secs[i]),
                                p_elec_w=float(power[i]), p=p)
        if build_done_i is None and v_hot >= 1.0 - 1e-9:
            build_done_i = i
        probe_pred.append(probe_temp(v_hot, t_hot, p))
        v_series.append(v_hot)

    probe_pred = np.array(probe_pred)
    resid = probe_pred - probe_obs
    mask = ~np.isnan(resid)
    rmse = float(np.sqrt(np.mean(resid[mask] ** 2))) if mask.any() else float("nan")
    elapsed_min = (idx - idx[0]).total_seconds() / 60.0
    info = {
        "v_hot0": float(v_hot0),
        "t_hot0": float(t_hot),  # filled below with start value
        "build_done_min": float(elapsed_min[build_done_i]) if build_done_i is not None else None,
        "probe_pred_final": float(probe_pred[-1]),
        "probe_obs_final": float(probe_obs[-1]),
        "rmse_c": rmse,
        "v_hot_final": float(v_series[-1]),
    }
    info["t_hot0"] = float(t_hot0 if t_hot0 is not None else probe_obs[0])
    return list(probe_pred), info


def fit_v_hot0(run, p: SoCParams, t_hot0: float | None = None, n: int = 41):
    """Coarse 1-D search for the latent start ``V_hot`` that minimises probe RMSE on a run.

    ``V_hot0`` is the unobservable the whole design hinges on (the probe can't see it). Fitting it
    per cycle and then checking the *rise* phase is reproduced is the model's over-identification
    test: one free latent set by the blind phase should also get the final temperature right.
    """
    import numpy as np

    best = None
    for v0 in np.linspace(0.05, 0.95, n):
        _, info = replay_reheat(run, p, v_hot0=float(v0), t_hot0=t_hot0)
        if best is None or info["rmse_c"] < best["rmse_c"]:
            best = info
    return best
