from datetime import datetime, timedelta, timezone

import pytest
import pytz

import hwc_dp_planner as dp
import hwc_planner as hp


def _adelaide_grid(year, month, day, hour, minute, n, step_min=5):
    tz = pytz.timezone("Australia/Adelaide")
    base_local = tz.localize(datetime(year, month, day, hour, minute))
    return [
        (base_local + timedelta(minutes=step_min * i)).astimezone(timezone.utc)
        for i in range(n)
    ]


def _hwc_cfg():
    return {
        "optimization_time_step": 5,
        "thermal": {
            "volume_l": 225,
            "density": 997,
            "heat_capacity": 4.184,
            "nominal_power_w": 800,
            "min_temp": 45,
            "max_temp": 60,
            "desired_temp": 55,
            "standing_loss_ua_kw_per_c": 0.0025,
            "heat_rate_c_per_hour": 12.0,
            "heat_rate_reference_wet_bulb_c": 10.0,
            "heat_rate_wet_bulb_slope_c_per_c": 0.1,
            "heat_rate_min_c_per_hour": 5.0,
            "heat_rate_max_c_per_hour": 20.0,
        },
    }


def test_transition_matches_single_node_simulator_for_one_step():
    hwc = _hwc_cfg()
    hwc["thermal"]["desired_temp"] = 60
    hwc["thermal"]["heat_rate_c_per_hour"] = 6.0
    hwc["thermal"]["top_up_start_temp_c"] = 53.0
    hwc["thermal"]["top_up_heat_rate_c_per_hour"] = 5.5

    next_temp = dp.transition_temperature(
        temp_c=54.0,
        action_heat=True,
        dry_bulb_c=12.0,
        wet_bulb_c=14.0,
        draw_off_kwh=0.1,
        hwc_cfg=hwc,
        step_h=5 / 60,
    )
    _, terminal = hp.simulate_block_temperatures(
        schedule_w=[800.0],
        start_temperature=54.0,
        dry_bulb=[12.0],
        wet_bulb=[14.0],
        draw_off=[0.1],
        cfg=hwc,
    )

    assert next_temp == pytest.approx(terminal, abs=0.01)


def test_required_target_dates_skip_partial_tail_day():
    grid = _adelaide_grid(2026, 6, 5, 15, 0, 48 * 12, step_min=5)

    assert dp.required_target_dates(
        grid,
        tz_name="Australia/Adelaide",
        main_window_end="18:00",
    ) == {"2026-06-05", "2026-06-06"}


def test_start_penalty_discourages_fragmented_cheap_slots():
    grid = _adelaide_grid(2026, 6, 5, 9, 0, 6, step_min=5)
    hwc = _hwc_cfg()
    hwc["thermal"]["desired_temp"] = 54.0
    hwc["thermal"]["heat_rate_c_per_hour"] = 60.0
    hwc["thermal"]["heat_rate_min_c_per_hour"] = 60.0
    hwc["thermal"]["heat_rate_max_c_per_hour"] = 60.0
    load_cost = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    common = dict(
        grid_times_utc=grid,
        load_cost=load_cost,
        dry_bulb=[45.0] * len(grid),
        wet_bulb=[12.0] * len(grid),
        draw_off=[0.0] * len(grid),
        start_temperature=45.0,
        hwc_cfg=hwc,
        tz_name="Australia/Adelaide",
        dp_cfg=dp.DpConfig(
            step_minutes=5,
            temp_bin_c=0.25,
            start_penalty_aud=0.0,
            main_window_end="09:30",
            terminal_penalty_aud_per_c2=0.0,
        ),
    )

    no_penalty = dp.solve(**common)
    with_penalty = dp.solve(
        **{
            **common,
            "dp_cfg": dp.DpConfig(
                step_minutes=5,
                temp_bin_c=0.25,
                start_penalty_aud=0.5,
                main_window_end="09:30",
                terminal_penalty_aud_per_c2=0.0,
            ),
        }
    )

    assert no_penalty.starts == 2
    assert with_penalty.starts == 1
    assert with_penalty.objective_breakdown["start_penalty_aud"] == pytest.approx(0.5)
