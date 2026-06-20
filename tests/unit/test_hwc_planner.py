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


def test_centered_rolling_mean_smooths_without_time_shift():
    assert hp.centered_rolling_mean([0.0, 0.0, 9.0, 0.0, 0.0], 3) == pytest.approx(
        [0.0, 3.0, 3.0, 3.0, 0.0]
    )


def test_centered_rolling_mean_rounds_even_window_to_odd():
    assert hp.centered_rolling_mean([0.0, 0.0, 9.0, 0.0, 0.0], 2) == pytest.approx(
        [0.0, 3.0, 3.0, 3.0, 0.0]
    )


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


def test_price_grid_prefers_5min_primary_then_expands_30min_tail():
    now = datetime(2026, 6, 1, 0, 2, tzinfo=timezone.utc)
    base = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    grid, prices = hp._build_price_grid_from_rows(
        now_utc=now,
        horizon_steps=8,
        step_minutes=5,
        primary_rows=[
            (base + timedelta(minutes=5 * i), 0.10 + i * 0.01)
            for i in range(3)
        ],
        tail_rows=[
            (base, 0.30),
            (base + timedelta(minutes=30), 0.40),
        ],
        tail_step_minutes=30,
        source_name="test",
    )

    assert grid == [base + timedelta(minutes=5 * i) for i in range(8)]
    assert prices == pytest.approx([0.10, 0.11, 0.12, 0.30, 0.30, 0.30, 0.40, 0.40])


def test_unit_load_cost_rows_from_entity(monkeypatch):
    def fake_ha_call(cfg, method, endpoint, payload=None):
        assert method == "GET"
        assert endpoint == "states/sensor.mpc_unit_load_cost"
        return {
            "attributes": {
                "unit_load_cost_forecasts": [
                    {
                        "date": "2026-06-01T00:05:00+00:00",
                        "mpc_unit_load_cost": "0.2345",
                    },
                    {
                        "date": "2026-06-01T00:00:00+00:00",
                        "mpc_unit_load_cost": "0.1234",
                    },
                ]
            }
        }

    monkeypatch.setattr(hp, "_ha_call", fake_ha_call)

    rows = hp._unit_load_cost_rows_from_entity({}, "sensor.mpc_unit_load_cost")

    assert rows == [
        (datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc), 0.1234),
        (datetime(2026, 6, 1, 0, 5, tzinfo=timezone.utc), 0.2345),
    ]


def test_main_satisfied_dates_from_state_file(tmp_path):
    state_file = tmp_path / "hwc_state.json"
    state_file.write_text('{"last_reached_target_at": "2026-06-01T14:00:00+00:00"}')
    cfg = {
        "timezone": "Australia/Adelaide",
        "hwc": {
            "daemon": {"state_file": str(state_file)},
            "block_planner": {},
        },
    }

    assert hp._main_satisfied_dates_from_state(cfg) == ["2026-06-01"]


def test_simulate_uses_single_heat_rate_without_top_up_config():
    temps, terminal = hp.simulate_block_temperatures(
        schedule_w=[800.0],
        start_temperature=54.0,
        dry_bulb=[54.0],
        draw_off=[0.0],
        cfg={
            "optimization_time_step": 60,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "heat_rate_c_per_hour": 6.6,
                "max_temp": 60,
            },
        },
    )

    assert temps == [54.0]
    assert terminal == pytest.approx(60.0)


def test_simulate_uses_top_up_heat_rate_above_threshold():
    temps, terminal = hp.simulate_block_temperatures(
        schedule_w=[800.0],
        start_temperature=54.0,
        dry_bulb=[54.0],
        draw_off=[0.0],
        cfg={
            "optimization_time_step": 60,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "heat_rate_c_per_hour": 6.6,
                "top_up_heat_rate_c_per_hour": 5.5,
                "top_up_start_temp_c": 53.0,
                "max_temp": 60,
            },
        },
    )

    assert temps == [54.0]
    assert terminal == pytest.approx(59.5)


def test_simulate_keeps_full_reheat_rate_when_cold_start_crosses_top_up_threshold():
    _, terminal = hp.simulate_block_temperatures(
        schedule_w=[800.0, 800.0],
        start_temperature=50.0,
        dry_bulb=[50.0, 50.0],
        draw_off=[0.0, 0.0],
        cfg={
            "optimization_time_step": 30,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "heat_rate_c_per_hour": 6.6,
                "top_up_heat_rate_c_per_hour": 1.0,
                "top_up_start_temp_c": 53.0,
                "max_temp": 60,
            },
        },
    )

    assert terminal == pytest.approx(56.6)


def test_simulate_adjusts_heat_rate_from_wet_bulb_when_configured():
    temps, terminal = hp.simulate_block_temperatures(
        schedule_w=[800.0],
        start_temperature=50.0,
        dry_bulb=[50.0],
        wet_bulb=[15.0],
        draw_off=[0.0],
        cfg={
            "optimization_time_step": 60,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "heat_rate_c_per_hour": 6.0,
                "heat_rate_reference_wet_bulb_c": 10.0,
                "heat_rate_wet_bulb_slope_c_per_c": 0.1,
                "max_temp": 60,
            },
        },
    )

    assert temps == [50.0]
    assert terminal == pytest.approx(56.5)


def test_simulate_clamps_wet_bulb_heat_rate_adjustment():
    _, terminal = hp.simulate_block_temperatures(
        schedule_w=[800.0],
        start_temperature=50.0,
        dry_bulb=[50.0],
        wet_bulb=[25.0],
        draw_off=[0.0],
        cfg={
            "optimization_time_step": 60,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "heat_rate_c_per_hour": 6.0,
                "heat_rate_reference_wet_bulb_c": 10.0,
                "heat_rate_wet_bulb_slope_c_per_c": 0.5,
                "heat_rate_max_c_per_hour": 7.0,
                "max_temp": 60,
            },
        },
    )

    assert terminal == pytest.approx(57.0)


def test_compressor_power_defaults_to_nominal_power():
    assert hp._compressor_power_w({"nominal_power_w": 800}, 50.0, 12.5) == 800.0


def test_compressor_power_uses_wet_bulb_and_tank_temperature():
    power = hp._compressor_power_w(
        {
            "nominal_power_w": 780,
            "compressor_power_reference_w": 780,
            "compressor_power_reference_wet_bulb_c": 12.5,
            "compressor_power_wet_bulb_slope_w_per_c": 8.0,
            "compressor_power_reference_tank_c": 50.0,
            "compressor_power_tank_slope_w_per_c": 1.5,
        },
        temp_c=56.0,
        wet_bulb_c=8.5,
    )

    assert power == pytest.approx(757.0)


def test_compressor_power_clamps_to_configured_range():
    assert hp._compressor_power_w(
        {
            "nominal_power_w": 780,
            "compressor_power_reference_w": 780,
            "compressor_power_reference_tank_c": 50.0,
            "compressor_power_tank_slope_w_per_c": 50.0,
            "compressor_power_min_w": 650,
            "compressor_power_max_w": 870,
        },
        temp_c=60.0,
    ) == 870.0


def test_refresh_planned_power_keeps_binary_schedule_shape_with_modelled_watts():
    refreshed = hp._refresh_planned_power(
        [0.0, 1.0, 1.0],
        start_temperature=50.0,
        dry_bulb=[15.0, 15.0, 15.0],
        wet_bulb=[8.5, 8.5, 8.5],
        draw_off=[0.0, 0.0, 0.0],
        cfg={
            "optimization_time_step": 30,
            "thermal": {
                "volume_l": 225,
                "density": 997,
                "heat_capacity": 4.184,
                "standing_loss_ua_kw_per_c": 0.0,
                "heat_rate_c_per_hour": 6.0,
                "max_temp": 60,
                "nominal_power_w": 780,
                "compressor_power_reference_w": 780,
                "compressor_power_reference_wet_bulb_c": 12.5,
                "compressor_power_wet_bulb_slope_w_per_c": 8.0,
                "compressor_power_reference_tank_c": 50.0,
                "compressor_power_tank_slope_w_per_c": 1.5,
            },
        },
    )

    assert refreshed[0] == 0.0
    assert refreshed[1] == pytest.approx(748.0)
    assert refreshed[2] > refreshed[1]


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
            "min_block_duration_minutes": 60,
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
        wet_bulb=[12.5] * 144,
        draw_off=hp.build_draw_off_profile(
            grid, "Australia/Adelaide", "09:00", "10:00", total_kwh=1.3
        ),
        start_temperature=55.0,
        cfg=cfg,
    )

    assert len(plan["predicted_temperatures"]) == 144
    assert len(plan["deferrables_schedule"]) == 144
    assert len(plan["wet_bulb_forecasts"]) == 144
    assert plan["wet_bulb_forecasts"][0]["hwc_wet_bulb_forecast"] == "12.50"
    assert min(plan["temperatures"]) >= 45
    assert plan["terminal_temperature"] >= 55.0


def test_publish_block_plan_includes_wet_bulb_forecast(monkeypatch):
    grid = _adelaide_grid(8, 3)
    cfg = {
        "timezone": "Australia/Adelaide",
        "home_assistant": {"url": "http://ha", "token": "token"},
        "hwc": _hwc_cfg(),
    }
    plan = hp.build_block_plan(
        grid_times_utc=grid,
        load_cost=[0.20, 0.10, 0.30],
        dry_bulb=[15.0, 16.0, 17.0],
        wet_bulb=[10.0, 11.0, 12.0],
        draw_off=[0.0, 0.0, 0.0],
        start_temperature=55.0,
        cfg=cfg,
    )
    published = {}

    def fake_set_state(config, entity_id, state, attributes):
        published[entity_id] = {"state": state, "attributes": attributes}

    monkeypatch.setattr(hp, "_ha_set_state", fake_set_state)

    hp._publish_block_plan(cfg, plan)

    wet_bulb = published["sensor.hwc_wet_bulb_forecast"]
    assert wet_bulb["state"] == "10.00"
    assert wet_bulb["attributes"]["unit_of_measurement"] == "°C"
    assert wet_bulb["attributes"]["wet_bulb_forecasts"][1]["hwc_wet_bulb_forecast"] == "11.00"


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


def test_block_planner_accepts_legacy_locked_schedule_seed():
    grid = _adelaide_grid(10, 16)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["min_block_duration_minutes"] = 0
    dry_bulb = [15.0] * len(grid)
    wet_bulb = [12.5] * len(grid)
    draw_off = [0.0] * len(grid)
    locked = hp._running_compressor_locked_schedule(
        grid_times_utc=grid,
        start_temperature=55.0,
        dry_bulb=dry_bulb,
        wet_bulb=wet_bulb,
        draw_off=draw_off,
        cfg=cfg,
    )

    plan = hp.build_block_plan(
        grid_times_utc=grid,
        load_cost=[0.60] * 4 + [0.05] * 12,
        dry_bulb=dry_bulb,
        wet_bulb=wet_bulb,
        draw_off=draw_off,
        start_temperature=55.0,
        cfg=cfg,
        locked_schedule_w=locked,
    )

    assert locked[0] > 0
    assert plan["schedule_w"][0] > 0


def test_running_compressor_low_stop_cost_can_defer_to_cheaper_main_block():
    grid = _adelaide_grid(10, 16)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["stop_cost_aud"] = 0.01
    cfg["hwc"]["block_planner"]["min_block_duration_minutes"] = 0
    plan = hp.build_block_plan(
        grid_times_utc=grid,
        load_cost=[0.80] * 4 + [0.02] * 12,
        dry_bulb=[15.0] * len(grid),
        wet_bulb=[12.5] * len(grid),
        draw_off=[0.0] * len(grid),
        start_temperature=55.0,
        cfg=cfg,
        compressor_initially_on=True,
    )

    assert plan["schedule_w"][0] == 0.0
    assert any(power > 0 for power in plan["schedule_w"][4:])
    assert plan["planned_stop_count"] >= 2


def test_running_compressor_high_stop_cost_keeps_current_run():
    grid = _adelaide_grid(10, 16)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["stop_cost_aud"] = 1.0
    cfg["hwc"]["block_planner"]["min_block_duration_minutes"] = 0
    plan = hp.build_block_plan(
        grid_times_utc=grid,
        load_cost=[0.80] * 4 + [0.02] * 12,
        dry_bulb=[15.0] * len(grid),
        wet_bulb=[12.5] * len(grid),
        draw_off=[0.0] * len(grid),
        start_temperature=55.0,
        cfg=cfg,
        compressor_initially_on=True,
    )

    assert plan["schedule_w"][0] > 0
    assert plan["planned_stop_count"] == 1


def test_main_block_skips_tiny_topup_near_target():
    grid = _adelaide_grid(10, 4)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["min_block_lift_c"] = 2.0
    schedule = hp._choose_daily_main_blocks(
        [0.0] * len(grid),
        grid_times_utc=grid,
        load_cost=[0.05] * len(grid),
        start_temperature=59.0,
        dry_bulb=[15.0] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg,
    )

    assert schedule == [0.0] * len(grid)


def test_minimum_block_duration_can_be_disabled():
    cfg = _hwc_cfg()
    cfg["block_planner"]["min_block_duration_minutes"] = 0

    assert hp._min_block_steps(cfg) == 1


def test_main_block_prioritises_reaching_daily_target_over_low_total_cost():
    grid = _adelaide_grid(10, 16)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    schedule = hp._choose_daily_main_blocks(
        [0.0] * len(grid),
        grid_times_utc=grid,
        load_cost=[0.08] * 8 + [0.60] * 8,
        start_temperature=49.0,
        dry_bulb=[15.0] * len(grid),
        wet_bulb=[12.5] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg,
    )
    temps, _ = hp.simulate_block_temperatures(
        schedule_w=schedule,
        start_temperature=49.0,
        dry_bulb=[15.0] * len(grid),
        wet_bulb=[12.5] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg["hwc"],
    )

    assert max(temps) >= 60.0
    assert sum(1 for power in schedule if power > 0) >= 2
    assert any(power > 0 for power in schedule[:8])


def test_main_block_skips_satisfied_local_date():
    grid = _adelaide_grid(10, 4)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["main_satisfied_dates"] = ["2026-06-02"]
    schedule = hp._choose_daily_main_blocks(
        [0.0] * len(grid),
        grid_times_utc=grid,
        load_cost=[0.05] * len(grid),
        start_temperature=55.0,
        dry_bulb=[15.0] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg,
    )

    assert schedule == [0.0] * len(grid)


def test_main_block_prefers_later_equal_cost_run_for_end_window_reserve():
    grid = _adelaide_grid(9, 22)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["main_end_reserve_penalty_aud_per_c2"] = 0.03
    schedule = hp._choose_daily_main_blocks(
        [0.0] * len(grid),
        grid_times_utc=grid,
        load_cost=[0.10] * len(grid),
        start_temperature=48.0,
        dry_bulb=[15.0] * len(grid),
        wet_bulb=[12.5] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg,
    )

    first_heat = next(idx for idx, power in enumerate(schedule) if power > 0)
    first_local = grid[first_heat].astimezone(pytz.timezone("Australia/Adelaide"))
    assert first_local.hour >= 15


def test_terminal_repair_respects_minimum_block_duration():
    grid = _adelaide_grid(0, 8)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["thermal"]["terminal_target"] = 53.0
    schedule = hp._repair_terminal_temperature(
        [0.0] * len(grid),
        grid_times_utc=grid,
        load_cost=[0.05] * len(grid),
        start_temperature=49.0,
        dry_bulb=[49.0] * len(grid),
        wet_bulb=[12.5] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg,
    )

    assert sum(1 for power in schedule if power > 0) >= 2


def test_terminal_repair_skips_tiny_shortfall_below_min_lift():
    grid = _adelaide_grid(0, 4)
    cfg = {"timezone": "Australia/Adelaide", "hwc": _hwc_cfg()}
    cfg["hwc"]["block_planner"]["min_block_lift_c"] = 2.0
    schedule = hp._repair_terminal_temperature(
        [0.0] * len(grid),
        grid_times_utc=grid,
        load_cost=[0.05] * len(grid),
        start_temperature=60.0,
        dry_bulb=[15.0] * len(grid),
        draw_off=[0.0] * len(grid),
        cfg=cfg,
    )

    assert schedule == [0.0] * len(grid)
