"""Unit tests for the standalone two-state ``(V_hot, T_hot)`` tank forward model.

These are pure/deterministic (no InfluxDB, no data files): they pin the *structural* invariants
the design depends on — a finite-width smooth observation map (the load-bearing anti-latch
property), build-before-rise ordering, energy conservation across the regime boundary, standing
loss, draws, and clamping. The data-dependent replay of real metered reheats is exercised by
``hwc_soc_calibrate.py --mode replay`` (parameters are still first-cut), not here.
"""

import math

import numpy as np

from hwc_soc_model import (
    SoCParams,
    apply_draw,
    cop_rise,
    fit_v_hot0,
    probe_temp,
    replay_reheat,
    step,
)

P = SoCParams()


# --- observation map g ---------------------------------------------------------------------

def test_g_endpoints():
    # Empty tank reads ~mains; full hot tank reads ~T_hot (within the finite-width tolerance).
    assert probe_temp(0.0, 60.0, P) < P.t_mains_c + 1.0
    assert probe_temp(1.0, 60.0, P) > 60.0 - 1.0


def test_g_midpoint_is_the_cliff_anchor():
    # At the sensor height the probe reads the midpoint of mains and T_hot — the ~35 °C draw-cliff
    # bottom for a 53 °C hot zone (Finding 5), independent of g_width.
    mid = probe_temp(P.sensor_height, 53.0, P)
    assert mid == (P.t_mains_c + 53.0) / 2.0
    assert 34.0 < mid < 37.0


def test_g_is_monotone_in_v_hot():
    vs = np.linspace(0.0, 1.0, 200)
    probe = [probe_temp(v, 60.0, P) for v in vs]
    assert all(b >= a - 1e-12 for a, b in zip(probe, probe[1:]))


def test_g_is_finite_width_not_a_step():
    # The anti-latch invariant: the transition spans a real interval, so the per-V_hot slope is
    # bounded. A hard threshold would show a near-1.0 jump in one tiny V_hot increment.
    vs = np.linspace(0.0, 1.0, 1001)
    probe = np.array([probe_temp(v, 60.0, P) for v in vs])
    dv = vs[1] - vs[0]
    max_slope = np.max(np.abs(np.diff(probe))) / dv  # °C per unit V_hot
    span = 60.0 - P.t_mains_c
    # A step would give slope ~ span/dv (~42000); a smooth logistic of width ~0.1 gives ~span/4w.
    assert max_slope < span / (2.0 * P.g_width)
    # And intermediate values genuinely exist just either side of the crossing.
    assert P.t_mains_c < probe_temp(0.45, 60.0, P) < 60.0
    assert P.t_mains_c < probe_temp(0.55, 60.0, P) < 60.0


def test_g_continuous():
    vs = np.linspace(0.0, 1.0, 2001)
    probe = np.array([probe_temp(v, 60.0, P) for v in vs])
    assert np.max(np.abs(np.diff(probe))) < 0.5  # no jump between adjacent samples


# --- COP -----------------------------------------------------------------------------------

def test_cop_rise_anchors_and_clamp():
    assert math.isclose(cop_rise(53.0, P), 2.3, abs_tol=1e-9)
    assert math.isclose(cop_rise(60.0, P), 1.75, abs_tol=1e-9)
    assert cop_rise(53.0, P) > cop_rise(60.0, P)  # collapses with condensing temp
    assert P.cop_floor <= cop_rise(80.0, P) <= P.cop_ceil  # extrapolation clamped


# --- step dynamics -------------------------------------------------------------------------

def test_build_grows_v_hot_holds_t_hot():
    v, t = step(0.5, 53.0, on=True, dt_s=60.0, p_elec_w=800.0, p=P)
    assert v > 0.5                      # hot zone grows
    assert abs(t - 53.0) < 0.05         # T_hot ~unchanged (only tiny standing loss) during build


def test_rise_raises_t_hot_holds_v_hot():
    v, t = step(1.0, 53.0, on=True, dt_s=300.0, p_elec_w=800.0, p=P)
    assert math.isclose(v, 1.0)
    assert t > 53.0


def test_build_before_rise_ordering_reproduces_flat_then_climb():
    # From a partly-charged tank, repeated heating must fill V_hot to 1 (T_hot ~flat) BEFORE T_hot
    # climbs — the measured probe-blind build then rise (Finding 4). T_hot stays flat until full.
    v, t = 0.55, 53.0
    t_while_building = []
    reached_full_at = None
    for k in range(600):  # 10 min-equivalent steps of 60 s
        v, t = step(v, t, on=True, dt_s=60.0, p_elec_w=800.0, p=P)
        if v < 1.0 - 1e-9:
            t_while_building.append(t)
        elif reached_full_at is None:
            reached_full_at = k
            t_at_full = t
    assert reached_full_at is not None
    assert max(t_while_building) - min(t_while_building) < 0.6   # ~flat through the build
    assert t > t_at_full                                         # climbs after full


def test_boundary_crossing_conserves_energy():
    # One big step that crosses V_hot=1 must land at the same state as two half-steps — i.e. the
    # build/rise split inside `step` doesn't create or destroy energy at the boundary. Standing
    # loss is zeroed so the only operator-split effect under test is the build/rise boundary
    # itself (with loss on, the T_hot decay differs trivially between one step and two; with a
    # T_hot-dependent rise COP, Euler samples it at different points — also not a boundary bug).
    p = SoCParams(standing_loss_w=0.0, cop_rise_at_lo=2.0, cop_rise_at_hi=2.0)
    v0, t0 = 0.95, 53.0  # 900 W × 1800 s crosses V_hot=1 partway through
    big = step(v0, t0, on=True, dt_s=1800.0, p_elec_w=900.0, p=p)
    assert big[0] == 1.0 and big[1] > 53.0  # genuinely crossed into the rise phase
    half = step(v0, t0, on=True, dt_s=900.0, p_elec_w=900.0, p=p)
    two = step(half[0], half[1], on=True, dt_s=900.0, p_elec_w=900.0, p=p)
    assert math.isclose(big[0], two[0], abs_tol=1e-12)
    assert math.isclose(big[1], two[1], abs_tol=1e-9)


def test_standing_loss_rate():
    # Off, full tank: T_hot should drop at ~0.3 °C/h (75 W over the tank capacity).
    v, t = step(1.0, 60.0, on=False, dt_s=3600.0, p_elec_w=0.0, p=P)
    drop = 60.0 - t
    assert math.isclose(drop, 0.075 / P.cap_full_kwh_per_k, rel_tol=1e-6)
    assert 0.25 < drop < 0.35


def test_off_does_not_heat():
    v, t = step(0.5, 53.0, on=False, dt_s=600.0, p_elec_w=800.0, p=P)
    assert math.isclose(v, 0.5)          # no compressor → no build
    assert t < 53.0                      # only standing loss


def test_v_hot_clamped():
    v, t = step(0.99, 53.0, on=True, dt_s=36000.0, p_elec_w=900.0, p=P)
    assert v == 1.0                      # cannot exceed a full tank


# --- draws ---------------------------------------------------------------------------------

def test_draw_drops_v_hot_holds_t_hot_and_clamps():
    v, t = apply_draw(0.8, 55.0, 0.3)
    assert math.isclose(v, 0.5)
    assert t == 55.0
    v2, _ = apply_draw(0.2, 55.0, 0.5)
    assert v2 == 0.0                     # cannot draw below empty


# --- synthetic replay (the structural reproduction, no real data needed) -------------------

def _synthetic_run(v_hot0, t_hot0, minutes=180, power_w=800.0):
    import pandas as pd

    idx = pd.date_range("2026-06-25 12:00", periods=minutes, freq="60s")
    v, t = v_hot0, t_hot0
    probe = []
    for i in range(minutes):
        if i > 0:
            v, t = step(v, t, on=True, dt_s=60.0, p_elec_w=power_w, p=P)
        probe.append(probe_temp(v, t, P))
    return pd.DataFrame({"power_w": power_w, "probe_ctrl": probe}, index=idx)


def test_replay_recovers_synthetic_truth():
    # Self-consistency: a run generated from a known V_hot0 is recovered by the fit, with a small
    # probe RMSE and a final probe that matches.
    run = _synthetic_run(v_hot0=0.6, t_hot0=53.0)
    # T_hot0 must be supplied: the probe reads 43.6 °C here (a 0.6-charged tank sits the sensor
    # below the thermocline), so the first probe reading is NOT T_hot — the whole point of g.
    best = fit_v_hot0(run, P, t_hot0=53.0)
    assert abs(best["v_hot0"] - 0.6) < 0.1
    assert best["rmse_c"] < 1.0
    assert abs(best["probe_pred_final"] - best["probe_obs_final"]) < 1.0


def test_replay_reproduces_blind_then_rise():
    # The probe is ~flat (build) then climbs (rise); the model's build-complete time precedes the
    # final probe climb to 60.
    run = _synthetic_run(v_hot0=0.6, t_hot0=53.0)
    _, info = replay_reheat(run, P, v_hot0=0.6, t_hot0=53.0)
    assert info["build_done_min"] is not None
    assert 0 < info["build_done_min"] < 180
    assert info["probe_pred_final"] > 53.0
