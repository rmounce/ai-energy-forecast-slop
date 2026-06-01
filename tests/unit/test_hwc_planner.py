"""Unit tests for the pure helpers in hwc_planner.py:
  - stull_wet_bulb: known reference value, bounds, RH→100 behaviour
  - interpolate_to_grid: linear interpolation + flat end extrapolation
  - build_draw_off_profile: clock-aligned placement, per-day total, zero outside window
  - build_payload: HWC-only structure (battery/PV off), thermal_battery keys, units
"""

from datetime import datetime, timedelta, timezone

import pytest
import pytz

import hwc_planner as hp


# ── stull_wet_bulb ──────────────────────────────────────────────────────────


def test_wet_bulb_reference_value():
    # Stull (2011) worked example: 20°C, 50% RH ≈ 13.7°C wet-bulb.
    assert hp.stull_wet_bulb(20.0, 50.0) == pytest.approx(13.7, abs=0.2)


def test_wet_bulb_never_exceeds_dry_bulb():
    for t in (5, 15, 25, 35):
        for rh in (20, 50, 80):
            assert hp.stull_wet_bulb(t, rh) <= t + 1e-6


def test_wet_bulb_approaches_dry_bulb_at_saturation():
    t = 22.0
    assert hp.stull_wet_bulb(t, 99.0) == pytest.approx(t, abs=1.0)


# ── interpolate_to_grid ─────────────────────────────────────────────────────


def test_interpolate_midpoint():
    src_epoch = [0.0, 100.0]
    src_vals = [10.0, 20.0]
    out = hp.interpolate_to_grid(src_epoch, src_vals, [0.0, 50.0, 100.0])
    assert out == pytest.approx([10.0, 15.0, 20.0])


def test_interpolate_flat_extrapolation_past_ends():
    src_epoch = [0.0, 100.0]
    src_vals = [10.0, 20.0]
    out = hp.interpolate_to_grid(src_epoch, src_vals, [-50.0, 150.0])
    assert out == pytest.approx([10.0, 20.0])  # held flat, not extrapolated


# ── build_draw_off_profile ──────────────────────────────────────────────────


def _adelaide_grid(local_start_hour, n, step_min=30):
    tz = pytz.timezone("Australia/Adelaide")
    base_local = tz.localize(datetime(2026, 6, 2, local_start_hour, 0))
    return [
        (base_local + timedelta(minutes=step_min * i)).astimezone(timezone.utc)
        for i in range(n)
    ]


def test_draw_off_places_energy_in_window():
    # 08:00–11:30 local, 30-min steps. Window 09:00–10:00 ⇒ slots at 09:00 and 09:30.
    grid = _adelaide_grid(8, 8)
    profile = hp.build_draw_off_profile(
        grid, "Australia/Adelaide", "09:00", "10:00", total_kwh=1.3
    )
    assert sum(profile) == pytest.approx(1.3)
    # exactly two non-zero slots, each half the total
    nonzero = [p for p in profile if p > 0]
    assert len(nonzero) == 2
    assert nonzero == pytest.approx([0.65, 0.65])
    # indices 2 (09:00) and 3 (09:30) are the window
    assert profile[2] == pytest.approx(0.65)
    assert profile[3] == pytest.approx(0.65)
    assert profile[0] == 0.0 and profile[4] == 0.0


def test_draw_off_total_per_day_repeats_across_horizon():
    # 36-hour grid spans two mornings ⇒ each gets the full total.
    grid = _adelaide_grid(8, 72)
    profile = hp.build_draw_off_profile(
        grid, "Australia/Adelaide", "09:00", "10:00", total_kwh=1.3
    )
    assert sum(profile) == pytest.approx(2.6, abs=1e-6)


def test_extra_draw_off_adds_to_matching_clock_window():
    grid = _adelaide_grid(21, 8)
    base = [0.0] * len(grid)
    profile = hp.add_draw_off_event(
        base, grid, "Australia/Adelaide", "22:00", duration_min=60, total_kwh=1.3
    )

    assert sum(profile) == pytest.approx(1.3)
    assert profile[2] == pytest.approx(0.65)
    assert profile[3] == pytest.approx(0.65)
    assert profile[1] == 0.0
    assert profile[4] == 0.0


def test_parse_extra_draw_off_default_and_custom_duration():
    assert hp.parse_extra_draw_off("22:00=1.3") == ("22:00", 60, 1.3)
    assert hp.parse_extra_draw_off("22:00+30=0.8") == ("22:00", 30, 0.8)


# ── build_payload ───────────────────────────────────────────────────────────


def _hwc_cfg():
    return {
        "predicted_temp_entity": "sensor.hwc_predicted_temp",
        "power_plan_entity": "sensor.hwc_power_plan",
        "publish_prefix": "hwc_",
        "optimization_time_step": 30,
        "thermal": {
            "volume_l": 225,
            "density": 997,
            "heat_capacity": 4.184,
            "supply_temperature": 60,
            "carnot_efficiency": 0.38,
            "thermal_loss_kw": 0.12,
            "nominal_power_w": 800,
            "min_temp": 45,
            "max_temp": 60,
            "desired_temp": 60,
            "penalty_factor": 15,
            "thermal_inertia_time_constant": 0.5,
            "standing_loss_ua_kw_per_c": 0.0025,
            "heat_rate_c_per_hour": 5.2,
            "terminal_target": "current",
        },
        "block_planner": {
            "main_window_start": "10:00",
            "main_window_end": "18:00",
            "overnight_window_start": "00:00",
            "overnight_window_end": "06:00",
            "boost_target_temp": 50,
            "terminal_lookback_hours": 24,
        },
    }


def test_build_payload_structure():
    grid = _adelaide_grid(8, 3)
    payload = hp.build_payload(
        grid_times_utc=grid,
        load_cost=[0.10, 0.20, 0.30],
        wet_bulb=[8.0, 9.0, 10.0],
        draw_off=[0.0, 0.65, 0.0],
        start_temperature=48.5,
        cfg=_hwc_cfg(),
    )
    # HWC-only: battery and PV disabled.
    assert payload["set_use_battery"] is False
    assert payload["set_use_pv"] is False
    assert payload["number_of_deferrable_loads"] == 1
    assert payload["prediction_horizon"] == 3

    # Wet-bulb is fed as the outdoor temperature forecast.
    assert payload["outdoor_temperature_forecast"] == [8.0, 9.0, 10.0]

    tb = payload["def_load_config"][0]["thermal_battery"]
    assert tb["volume"] == pytest.approx(0.225)  # litres → m³
    assert tb["start_temperature"] == pytest.approx(48.5)
    assert tb["carnot_efficiency"] == 0.38
    assert tb["thermal_loss"] == 0.12
    assert tb["draw_off_demand"] == [0.0, 0.65, 0.0]
    # per-timestep temperature arrays match the horizon length
    for key in ("min_temperatures", "max_temperatures", "desired_temperatures"):
        assert len(tb[key]) == 3
    assert tb["min_temperatures"][0] == 45
    assert tb["max_temperatures"][0] == 60

    # All forecast lists are horizon-length.
    for key in ("load_cost_forecast", "prod_price_forecast", "load_power_forecast"):
        assert len(payload[key]) == 3

    assert payload["custom_predicted_temperature_id"][0]["entity_id"] == "sensor.hwc_predicted_temp"


def test_block_planner_builds_long_horizon_with_terminal_target():
    grid = _adelaide_grid(0, 144)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    plan = hp.build_block_plan(
        grid_times_utc=grid,
        load_cost=[0.30] * 20 + [0.08] * 16 + [0.22] * 108,
        dry_bulb=[15.0] * 144,
        draw_off=hp.build_draw_off_profile(
            grid, "Australia/Adelaide", "09:00", "10:00", total_kwh=1.3
        ),
        start_temperature=55.0,
        cfg=cfg,
    )

    assert len(plan["predicted_temperatures"]) == 144
    assert len(plan["deferrables_schedule"]) == 144
    assert min(plan["temperatures"]) >= 45
    assert plan["terminal_temperature"] >= 55.0


def test_block_planner_prefers_contiguous_daytime_runs():
    grid = _adelaide_grid(0, 96)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    plan = hp.build_block_plan(
        grid_times_utc=grid,
        load_cost=[0.30] * 20 + [0.08] * 16 + [0.30] * 28 + [0.08] * 16 + [0.30] * 16,
        dry_bulb=[15.0] * 96,
        draw_off=hp.build_draw_off_profile(
            grid, "Australia/Adelaide", "09:00", "10:00", total_kwh=1.3
        ),
        start_temperature=55.0,
        cfg=cfg,
    )

    starts = sum(
        1
        for prev, cur in zip([0.0] + plan["schedule_w"][:-1], plan["schedule_w"], strict=True)
        if prev <= 0 and cur > 0
    )
    assert starts <= 3
    assert all(power in (0.0, 800.0) for power in plan["schedule_w"])
