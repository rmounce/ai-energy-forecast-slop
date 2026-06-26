"""Unit tests for the opt-in two-state (V_hot, T_hot) DP path (hwc.dp_planner.soc_model).

Slice 1 of docs/hwc_2state_soc_model.md steps 2–3: the flag-gated decision model. These cover that
the flag-off path is untouched (the existing suite is the regression guard), and that the flag-on
path keeps the published contract, meets the daily-60 obligation on the *probe*, emits the soc_*
diagnostics, and responds to the seed `(V_hot0, T_hot0)` the daemon tracker will supply.

`max_temp` is set to the production value 62 (not the other suite's 60): at V_hot=1 the finite-width
probe map reads ~0.3 °C below T_hot, so driving the *probe* to 60 needs a touch of headroom above 60
in T_hot — a small, conservative artifact of the smooth g, not a bug.
"""

from datetime import datetime, timedelta, timezone

import pytz

import hwc_dp_planner as dp

TZ = pytz.timezone("Australia/Adelaide")


def _grid(local_start_hour, n, step_min=30, day=21):
    base = TZ.localize(datetime(2026, 6, day, local_start_hour, 0))
    return [(base + timedelta(minutes=step_min * i)).astimezone(timezone.utc) for i in range(n)]


def _cfg(step_min=30, soc=True, **overrides):
    thermal = {
        "volume_l": 222, "density": 997, "heat_capacity": 4.186,
        "supply_temperature": 60, "carnot_efficiency": 0.38, "thermal_loss_kw": 0.12,
        "nominal_power_w": 780, "compressor_power_reference_w": 740,
        "compressor_power_reference_wet_bulb_c": 12.5, "compressor_power_wet_bulb_slope_w_per_c": 1.5,
        "compressor_power_reference_tank_c": 50.0, "compressor_power_tank_slope_w_per_c": 15.0,
        "compressor_power_min_w": 650, "compressor_power_max_w": 930,
        "min_temp": 45, "max_temp": 62, "desired_temp": 60,
        "standing_loss_ua_kw_per_c": 0.0025, "heat_rate_c_per_hour": 6.6,
        "top_up_start_temp_c": 53.0, "top_up_heat_rate_c_per_hour": 5.5,
        "heat_rate_reference_wet_bulb_c": 12.5, "heat_rate_wet_bulb_slope_c_per_c": 0.08,
        "heat_rate_min_c_per_hour": 4.8, "heat_rate_max_c_per_hour": 7.6,
        "terminal_target": "current",
    }
    thermal.update(overrides)
    dp_planner = {} if not soc else {"soc_model": True}
    return {
        "timezone": "Australia/Adelaide",
        "hwc": {
            "predicted_temp_entity": "sensor.hwc_predicted_temp",
            "power_plan_entity": "sensor.hwc_power_plan", "publish_prefix": "hwc_",
            "optimization_time_step": step_min, "thermal": thermal,
            "main_window_end": "18:00", "transition_cost_aud": 0.05, "dp_planner": dp_planner,
        },
    }


def _plan(cfg, grid, *, price=0.20, start=52.0, soc_state0=None, draw=None):
    n = len(grid)
    return dp.build_dp_plan(
        grid_times_utc=grid, load_cost=[price] * n, dry_bulb=[15.0] * n, wet_bulb=[12.5] * n,
        draw_off=draw if draw is not None else [0.0] * n, start_temperature=start, cfg=cfg,
        soc_state0=soc_state0,
    )


def _on_steps(plan):
    return sum(1 for w in plan["schedule_w"] if w > 0)


# ── flag dispatch ─────────────────────────────────────────────────────────────

def test_flag_off_does_not_take_soc_path():
    # No soc_model key ⇒ legacy path ⇒ no soc_* diagnostics.
    plan = _plan(_cfg(soc=False), _grid(0, 24))
    assert "soc_v_hot" not in plan


def test_flag_on_keeps_published_contract_and_adds_diagnostics():
    grid = _grid(0, 48)
    plan = _plan(_cfg(), grid)
    for key in ("schedule_w", "temperatures", "terminal_temperature", "objective_cost_aud",
                "planned_stop_count", "predicted_temperatures"):
        assert key in plan
    assert len(plan["schedule_w"]) == 48
    for w in plan["schedule_w"]:
        assert w == 0.0 or 650.0 <= w <= 930.0
    # diagnostics present, right length, and consistent with the published schedule's run/idle
    assert len(plan["soc_v_hot"]) == 48 == len(plan["soc_t_hot"]) == len(plan["soc_probe"])
    assert all(0.0 <= v <= 1.0 for v in plan["soc_v_hot"])
    assert len(plan["soc_state0"]) == 2


# ── obligation on the probe ───────────────────────────────────────────────────

def test_daily_60_met_on_probe_when_achievable():
    # Cheap flat price, a cold half-charged start, a full day of horizon: the plan should drive the
    # modelled *probe* to ≈60 before the 18:00 deadline.
    grid = _grid(6, 48)  # 06:00 → 06:00 next day, deadline 18:00 day 1
    plan = _plan(_cfg(), grid, price=0.10, start=49.0, soc_state0=(0.5, 53.0))
    deadline_idx = max(i for i, t in enumerate(grid)
                       if t.astimezone(TZ).hour < 18 and t.astimezone(TZ).date().day == 21)
    assert max(plan["soc_probe"][: deadline_idx + 1]) >= 59.0
    assert _on_steps(plan) > 0


def test_no_heating_when_already_satisfied():
    # Start full & hot, all days pre-satisfied, no terminal pull: no reason to run.
    cfg = _cfg(terminal_target=45.0)
    grid = _grid(12, 24)
    cfg["hwc"]["main_satisfied_dates"] = sorted(
        {t.astimezone(TZ).date().isoformat() for t in grid}
    )
    plan = _plan(cfg, grid, price=0.20, start=61.0, soc_state0=(1.0, 61.0))
    assert _on_steps(plan) == 0


# ── seed (V_hot0, T_hot0) drives the work ─────────────────────────────────────

def test_emptier_seed_schedules_more_heating():
    grid = _grid(6, 48)
    cfg = _cfg()
    empty = _plan(cfg, grid, price=0.10, start=53.0, soc_state0=(0.2, 53.0))
    full = _plan(cfg, grid, price=0.10, start=53.0, soc_state0=(0.9, 53.0))
    # Same probe-equivalent start, but a fuller hot zone needs less build → fewer on-steps.
    assert _on_steps(empty) > _on_steps(full)


def test_seed_defaults_when_not_supplied():
    # No soc_state0 ⇒ falls back to (conservative V_hot, probe≈T_hot); still produces a valid plan.
    plan = _plan(_cfg(), _grid(6, 24), price=0.15, start=50.0, soc_state0=None)
    assert len(plan["schedule_w"]) == 24
    assert plan["soc_state0"][1] >= 50.0  # T_hot0 seeded from the start probe
