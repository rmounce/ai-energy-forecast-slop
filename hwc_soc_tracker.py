#!/usr/bin/env python3
"""Daemon-side ``(V_hot, T_hot)`` state tracker for the two-state HWC model.

This is the production estimator that seeds the DP planner (``soc_state0``). Per the agreed design
(docs/hwc_2state_soc_model.md "Production architecture") it is deliberately **not** a state
observer — ``V_hot`` is unobservable in the interior, so an observer would drift and lie. Instead:

1. **Conservative (pessimistic) draw prior** decrements ``V_hot`` between resets — an over-estimating
   daily draw, not a historical average, because a heavier-than-average draw that stays above the
   probe is otherwise invisible.
2. **Two watermark resets** — the only two crossings where the state is exactly observable; snap to
   them and accumulated drift erases:
   - **Top** (``V_hot → 1``): the probe reaches the daily-60 target (tank full + hot).
   - **Cliff** (``V_hot → sensor_height``): the probe falls past the cold-slug cliff (a deep draw
     has pulled the thermocline down to the sensor).
3. **Standing loss** cools ``T_hot``; **active heating** advances the state through the same
   ``hwc_soc_model.step`` the planner uses (a forward-sim from the modelled compressor power, not an
   observation).

Pure and side-effect-free so it is unit-testable without the daemon; the daemon (slice 2b) owns
persistence (``data/hwc_daemon_state.json``), wall-clock ``dt``, and the live probe/compressor reads.
"""
from __future__ import annotations

from dataclasses import dataclass

import hwc_soc_model as soc


@dataclass(frozen=True)
class TrackerState:
    v_hot: float
    t_hot: float


def seed_state(probe_c: float, p: soc.SoCParams, conservative_v_hot: float = 0.5) -> TrackerState:
    """Cold-start seed (no persisted state): pessimistic ``V_hot``, ``T_hot`` ≈ the probe.

    With no history we cannot know ``V_hot``; assume the conservative middle. ``T_hot`` is taken at
    the probe (the hot zone is at least as warm as the sensor reads), floored just above mains.
    The first watermark crossing corrects both.
    """
    return TrackerState(
        v_hot=min(1.0, max(0.0, conservative_v_hot)),
        t_hot=max(float(probe_c), p.t_mains_c + soc._MIN_LIFT_C),
    )


def draw_kwh_per_s_to_dv(draw_kwh_per_s: float, t_hot: float, p: soc.SoCParams) -> float:
    """Convert a pessimistic draw rate (kWh/s of hot water) to a ``V_hot`` decrement rate (1/s)."""
    lift = max(t_hot - p.t_mains_c, soc._MIN_LIFT_C)
    return draw_kwh_per_s / (p.cap_full_kwh_per_k * lift)


def advance(
    state: TrackerState,
    dt_s: float,
    *,
    heating: bool,
    probe_c: float | None,
    modelled_power_w: float,
    draw_kwh_per_s: float,
    p: soc.SoCParams,
    desired_c: float,
    cliff_probe_c: float,
) -> tuple[TrackerState, str]:
    """Advance the tracker by ``dt_s`` and apply watermark resets. Returns (state, reason).

    ``heating`` is the effective compressor-on signal over the interval; ``probe_c`` the current
    control-probe reading (``None`` if unavailable); ``modelled_power_w`` the planner's compressor
    power estimate for this probe; ``draw_kwh_per_s`` the *conservative* draw rate (0 outside the
    draw window). ``cliff_probe_c`` is the probe level below which a deep draw is inferred.
    """
    v, t = state.v_hot, state.t_hot

    # 1. Dynamics.
    if heating:
        v, t = soc.step(v, t, on=True, dt_s=dt_s, p_elec_w=modelled_power_w, p=p)
    else:
        v, t = soc.step(v, t, on=False, dt_s=dt_s, p_elec_w=0.0, p=p)  # standing loss on T_hot
        v = max(0.0, v - draw_kwh_per_s_to_dv(draw_kwh_per_s, t, p) * dt_s)  # conservative prior

    # 2. Watermark resets (exact-observability crossings; snap and let drift erase).
    reason = "heat" if heating else "coast"
    if probe_c is not None:
        if probe_c >= desired_c:
            v, t, reason = 1.0, max(t, float(probe_c)), "watermark:full@target"
        elif probe_c <= cliff_probe_c and v > p.sensor_height:
            v, reason = p.sensor_height, "watermark:cliff"

    return TrackerState(min(1.0, max(0.0, v)), t), reason
