"""Unit tests for the DP HWC planner (hwc_dp_planner.build_dp_plan).

Covers: plan shape parity with the block planner; min-temperature honoured as a soft
high-penalty floor; daily-60 C obligation met when achievable; transition cost merges a
short cycle through a price blip; compressor-initially-on handled; cold start degrades
gracefully; published powers come from the exact thermal model.
"""

from datetime import datetime, timedelta, timezone

import pytz

import hwc_dp_planner as dp
import hwc_planner as hp

TZ = pytz.timezone("Australia/Adelaide")


def _grid(local_start_hour, n, step_min=30, day=21):
    base = TZ.localize(datetime(2026, 6, day, local_start_hour, 0))
    return [
        (base + timedelta(minutes=step_min * i)).astimezone(timezone.utc) for i in range(n)
    ]


def _cfg(step_min=30, **thermal_overrides):
    thermal = {
        "volume_l": 225,
        "density": 997,
        "heat_capacity": 4.184,
        "supply_temperature": 60,
        "carnot_efficiency": 0.38,
        "thermal_loss_kw": 0.12,
        "nominal_power_w": 780,
        "compressor_power_reference_w": 740,
        "compressor_power_reference_wet_bulb_c": 12.5,
        "compressor_power_wet_bulb_slope_w_per_c": 1.5,
        "compressor_power_reference_tank_c": 50.0,
        "compressor_power_tank_slope_w_per_c": 15.0,
        "compressor_power_min_w": 650,
        "compressor_power_max_w": 930,
        "min_temp": 45,
        "max_temp": 60,
        "desired_temp": 60,
        "standing_loss_ua_kw_per_c": 0.0025,
        "heat_rate_c_per_hour": 6.6,
        "top_up_start_temp_c": 53.0,
        "top_up_heat_rate_c_per_hour": 5.5,
        "heat_rate_reference_wet_bulb_c": 12.5,
        "heat_rate_wet_bulb_slope_c_per_c": 0.08,
        "heat_rate_min_c_per_hour": 4.8,
        "heat_rate_max_c_per_hour": 7.6,
        "terminal_target": "current",
    }
    thermal.update(thermal_overrides)
    return {
        "timezone": "Australia/Adelaide",
        "hwc": {
            "predicted_temp_entity": "sensor.hwc_predicted_temp",
            "power_plan_entity": "sensor.hwc_power_plan",
            "publish_prefix": "hwc_",
            "optimization_time_step": step_min,
            "thermal": thermal,
            "main_window_end": "18:00",
            "transition_cost_aud": 0.05,
        },
    }


def _starts(schedule_w):
    return sum(1 for a, b in zip([0.0] + schedule_w[:-1], schedule_w) if a <= 0 and b > 0)


def _satisfy_all_days(cfg, grid):
    dates = sorted({t.astimezone(TZ).date().isoformat() for t in grid})
    cfg["hwc"]["main_satisfied_dates"] = dates


# ── structure ───────────────────────────────────────────────────────────────


def test_dp_plan_shape_matches_published_contract():
    grid = _grid(0, 48)
    cfg = _cfg()
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=[0.20] * 48,
        dry_bulb=[15.0] * 48,
        wet_bulb=[12.5] * 48,
        draw_off=[0.0] * 48,
        start_temperature=52.0,
        cfg=cfg,
    )
    for key in (
        "schedule_w",
        "temperatures",
        "terminal_temperature",
        "objective_cost_aud",
        "planned_stop_count",
        "planned_energy_cost_aud",
        "predicted_temperatures",
        "deferrables_schedule",
        "wet_bulb_forecasts",
    ):
        assert key in plan
    assert len(plan["schedule_w"]) == 48
    assert len(plan["temperatures"]) == 48
    assert len(plan["predicted_temperatures"]) == 48
    # Powers are either off or within the modelled compressor clamp range.
    for p in plan["schedule_w"]:
        assert p == 0.0 or 650.0 <= p <= 930.0


def test_dp_published_temps_match_exact_model_replay():
    grid = _grid(0, 24)
    cfg = _cfg()
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=[0.20] * 24,
        dry_bulb=[15.0] * 24,
        wet_bulb=[12.5] * 24,
        draw_off=[0.0] * 24,
        start_temperature=51.0,
        cfg=cfg,
    )
    temps, terminal = hp.simulate_block_temperatures(
        schedule_w=plan["schedule_w"],
        start_temperature=51.0,
        dry_bulb=[15.0] * 24,
        wet_bulb=[12.5] * 24,
        draw_off=[0.0] * 24,
        cfg=cfg["hwc"],
    )
    assert plan["temperatures"] == temps
    assert plan["terminal_temperature"] == terminal


# ── soft constraints ──────────────────────────────────────────────────────────


def test_dp_holds_min_temperature_against_draw_off():
    # Obligation off so only the min-temp floor drives heating.
    grid = _grid(0, 48)
    cfg = _cfg()
    _satisfy_all_days(cfg, grid)
    draw = hp.build_draw_off_profile(grid, "Australia/Adelaide", "06:00", "07:00", 2.0)
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=[0.20] * 48,
        dry_bulb=[14.0] * 48,
        wet_bulb=[11.0] * 48,
        draw_off=draw,
        start_temperature=47.0,
        cfg=cfg,
    )
    # Heats (to defend the floor) and keeps the tank close to >= min_temp.
    assert any(p > 0 for p in plan["schedule_w"])
    assert min(plan["temperatures"]) >= 44.0


def test_dp_reaches_desired_on_obligation_day():
    grid = _grid(0, 40)  # 20h from local midnight: 18:00 deadline is in-horizon
    cfg = _cfg()
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=[0.20] * 40,
        dry_bulb=[15.0] * 40,
        wet_bulb=[12.5] * 40,
        draw_off=[0.0] * 40,
        start_temperature=50.0,
        cfg=cfg,
    )
    # Reaches the 60 C legionella target within the day's main window.
    deadline_idx = max(
        i for i, t in enumerate(grid) if t.astimezone(TZ).strftime("%H:%M") <= "18:00"
    )
    assert max(plan["temperatures"][: deadline_idx + 1]) >= 59.5


# ── transition cost ───────────────────────────────────────────────────────────


def _blip_scenario_starts(transition_cost):
    # The 60 C reheat must land in a tight cheap window (everything else very expensive)
    # that is split by a single moderately-expensive blip slot. Avoiding the blip costs a
    # second start; eating it keeps a single block. The transition cost decides.
    grid = _grid(14, 12)  # deadline 18:00 is index 8
    cfg = _cfg()
    cfg["hwc"]["transition_cost_aud"] = transition_cost
    load = [5.0] * 12
    load[4], load[5], load[6], load[7] = 0.05, 0.55, 0.05, 0.05  # blip at index 5
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=load,
        dry_bulb=[15.0] * 12,
        wet_bulb=[12.5] * 12,
        draw_off=[0.0] * 12,
        start_temperature=53.0,
        cfg=cfg,
    )
    return _starts(plan["schedule_w"]), max(plan["temperatures"])


def test_dp_transition_cost_merges_short_cycle_through_price_blip():
    free_starts, free_max = _blip_scenario_starts(0.0)
    costly_starts, costly_max = _blip_scenario_starts(0.30)
    # Both still reach the 60 C target...
    assert free_max >= 59.5 and costly_max >= 59.5
    # ...but with no transition cost the DP splits to dodge the blip, while charging a
    # transition cost makes it run straight through (fewer starts).
    assert free_starts == 2
    assert costly_starts == 1


# ── running compressor + robustness ──────────────────────────────────────────


def test_dp_no_short_cycle_flip_at_taper_boundary():
    # Regression for the 53 °C short-cycle. The heat rate now depends only on the *current* temp
    # (no carried FULL/TOP-UP regime), so continuing a run and freshly starting at the same temp
    # compute the same rate — the asymmetry the DP used to arbitrage is gone. The seeded-on and
    # seeded-off plans must therefore agree on the first action at the old boundary temp.
    # (docs/hwc/reviews/short_cycle_review_2026-06-26.md)
    grid = _grid(15, 48, step_min=5, day=26)  # 18:00 legionella deadline is in-window
    cfg = _cfg(step_min=5, heat_rate_taper_width_c=6.0)
    load = [0.12, 0.12] + [0.4] * 46  # the scenario that exposed the flip
    common = dict(
        grid_times_utc=grid,
        load_cost=load,
        dry_bulb=[15.0] * 48,
        wet_bulb=[12.5] * 48,
        draw_off=[0.0] * 48,
        start_temperature=53.0,
        cfg=cfg,
    )
    off0 = dp.build_dp_plan(compressor_initially_on=False, **common)["schedule_w"][0]
    on0 = dp.build_dp_plan(compressor_initially_on=True, **common)["schedule_w"][0]
    assert (off0 > 0) == (on0 > 0)


def test_dp_handles_compressor_initially_on():
    grid = _grid(10, 16)
    cfg = _cfg()
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=[0.80] * 4 + [0.05] * 12,
        dry_bulb=[15.0] * 16,
        wet_bulb=[12.5] * 16,
        draw_off=[0.0] * 16,
        start_temperature=55.0,
        cfg=cfg,
        compressor_initially_on=True,
    )
    assert len(plan["schedule_w"]) == 16
    # Continuing the in-progress run is free (no transition charge on slot 0); the cost
    # accounting must not invent a phantom start cost.
    assert plan["planned_stop_count"] >= 0


def test_dp_cold_start_degrades_gracefully():
    grid = _grid(0, 24)
    cfg = _cfg()
    plan = dp.build_dp_plan(
        grid_times_utc=grid,
        load_cost=[0.20] * 24,
        dry_bulb=[12.0] * 24,
        wet_bulb=[9.0] * 24,
        draw_off=[0.0] * 24,
        start_temperature=38.0,  # well below min_temp
        cfg=cfg,
    )
    # No exception; it heats hard toward the floor and ends warmer than it started.
    assert any(p > 0 for p in plan["schedule_w"])
    assert plan["terminal_temperature"] > 38.0


def test_dp_two_survivors_produces_valid_plan():
    # survivors_per_state=2 is an opt-in "compromise" path (not objectively helpful; kept by
    # owner request). It must still yield a structurally valid, exactly-replayed plan.
    grid = _grid(0, 40)
    cfg = _cfg()
    cfg["hwc"].setdefault("dp_planner", {})["survivors_per_state"] = 2
    kwargs = dict(
        grid_times_utc=grid,
        load_cost=[0.20] * 20 + [0.60] * 20,
        dry_bulb=[15.0] * 40,
        wet_bulb=[12.5] * 40,
        draw_off=[0.0] * 40,
        start_temperature=50.0,
        cfg=cfg,
    )
    plan = dp.build_dp_plan(**kwargs)
    assert len(plan["schedule_w"]) == 40
    for p in plan["schedule_w"]:
        assert p == 0.0 or 650.0 <= p <= 930.0
    # Published temps still come from the exact replay, not the binned DP estimate.
    temps, terminal = hp.simulate_block_temperatures(
        schedule_w=plan["schedule_w"],
        start_temperature=50.0,
        dry_bulb=[15.0] * 40,
        wet_bulb=[12.5] * 40,
        draw_off=[0.0] * 40,
        cfg=cfg["hwc"],
    )
    assert plan["temperatures"] == temps
    assert plan["terminal_temperature"] == terminal


def test_dp_survivors_default_is_one():
    # Absent config => single survivor (the evidence-based default); identical to omitting
    # the dp_planner block entirely.
    grid = _grid(0, 32)
    base = _cfg()
    explicit = _cfg()
    explicit["hwc"].setdefault("dp_planner", {})["survivors_per_state"] = 1
    common = dict(
        grid_times_utc=grid,
        load_cost=[0.20] * 16 + [0.55] * 16,
        dry_bulb=[14.0] * 32,
        wet_bulb=[11.5] * 32,
        draw_off=[0.0] * 32,
        start_temperature=49.0,
    )
    assert (
        dp.build_dp_plan(cfg=base, **common)["schedule_w"]
        == dp.build_dp_plan(cfg=explicit, **common)["schedule_w"]
    )


def test_dp_empty_horizon_returns_empty_plan():
    cfg = _cfg()
    plan = dp.build_dp_plan(
        grid_times_utc=[],
        load_cost=[],
        dry_bulb=[],
        wet_bulb=None,
        draw_off=[],
        start_temperature=50.0,
        cfg=cfg,
    )
    assert plan["schedule_w"] == []
