#!/usr/bin/env python3
"""Print a compact live block-vs-DP HWC planning diagnostic."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from config_utils import load_config
import hwc_dp_planner as dp
import hwc_planner as hp


@dataclass(frozen=True)
class Block:
    start_idx: int
    end_idx: int


def _daemon_satisfied_date(cfg: dict) -> str | None:
    state_file = cfg["hwc"].get("daemon", {}).get("state_file", "data/hwc_daemon_state.json")
    path = Path(state_file)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        reached_at = json.loads(path.read_text()).get("last_reached_target_at")
    except FileNotFoundError:
        return None
    if not reached_at:
        return None
    try:
        dt = datetime.fromisoformat(reached_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(ZoneInfo(cfg["timezone"])).date().isoformat()


def _with_daemon_satisfied_date(cfg: dict) -> dict:
    satisfied_date = _daemon_satisfied_date(cfg)
    if not satisfied_date:
        return cfg
    out = copy.deepcopy(cfg)
    out["hwc"].setdefault("block_planner", {})["main_satisfied_dates"] = [satisfied_date]
    return out


def _blocks(schedule_w: list[float]) -> list[Block]:
    out = []
    in_block = False
    start_idx = 0
    for idx, power in enumerate(schedule_w + [0.0]):
        if power > 0 and not in_block:
            start_idx = idx
            in_block = True
        elif in_block and power <= 0:
            out.append(Block(start_idx=start_idx, end_idx=idx))
            in_block = False
    return out


def _summary(
    *,
    label: str,
    schedule_w: list[float],
    temperatures: list[float],
    terminal_temperature: float,
    step_minutes: int,
) -> str:
    starts = len(_blocks(schedule_w))
    heat_kwh = sum(schedule_w) / 1000.0 * (step_minutes / 60.0)
    return (
        f"{label}: starts={starts}, heat_kwh={heat_kwh:.2f}, "
        f"min={min(temperatures):.1f}C, terminal={terminal_temperature:.1f}C"
    )


def _block_rows(
    *,
    grid_times,
    schedule_w: list[float],
    temperatures: list[float],
    terminal_temperature: float,
    load_cost: list[float],
    dry_bulb: list[float],
    wet_bulb: list[float],
    draw_off: list[float],
    step_minutes: int,
    tz_name: str,
    min_temp: float,
    target_temp: float,
    terminal_target: float,
    hwc_cfg: dict,
    required_target_dates: set[str],
) -> list[dict]:
    tz = ZoneInfo(tz_name)
    temps_after = list(temperatures[1:]) + [terminal_temperature]
    rows = []
    blocks = _blocks(schedule_w)
    for block_idx, block in enumerate(blocks):
        idxs = list(range(block.start_idx, block.end_idx))
        next_start = blocks[block_idx + 1].start_idx if block_idx + 1 < len(blocks) else len(schedule_w)
        local_start = grid_times[block.start_idx].astimezone(tz)
        local_end = grid_times[block.end_idx - 1].astimezone(tz)
        pre_temp = temperatures[block.start_idx]
        post_temp = temps_after[block.end_idx - 1]
        max_temp = max(temps_after[i] for i in idxs)
        next_min = min(temperatures[block.end_idx:next_start] or [post_temp])
        block_cost = sum(schedule_w[i] / 1000.0 * load_cost[i] * (step_minutes / 60.0) for i in idxs)
        heat_kwh = sum(schedule_w[i] for i in idxs) / 1000.0 * (step_minutes / 60.0)
        reason_flags = []
        if pre_temp <= min_temp + 1.0 or next_min <= min_temp + 1.0:
            reason_flags.append("floor")
        if pre_temp < target_temp <= max_temp:
            reason_flags.append("target")
        if block_idx == len(blocks) - 1 and post_temp >= terminal_target - 0.25:
            reason_flags.append("terminal")
        if not reason_flags:
            reason_flags.append("cost/reserve")
        removal = _removal_effect(
            schedule_w=schedule_w,
            block=block,
            start_temperature=temperatures[0],
            dry_bulb=dry_bulb,
            wet_bulb=wet_bulb,
            draw_off=draw_off,
            load_cost=load_cost,
            step_minutes=step_minutes,
            grid_times=grid_times,
            tz_name=tz_name,
            hwc_cfg=hwc_cfg,
            required_target_dates=required_target_dates,
            min_temp=min_temp,
            terminal_target=terminal_target,
        )
        rows.append(
            {
                "start": local_start,
                "end": local_end,
                "slots": len(idxs),
                "heat_kwh": heat_kwh,
                "cost": block_cost,
                "avg_price": sum(load_cost[i] for i in idxs) / len(idxs),
                "avg_wet_bulb": sum(wet_bulb[i] for i in idxs) / len(idxs),
                "pre_temp": pre_temp,
                "post_temp": post_temp,
                "next_min": next_min,
                "reasons": ",".join(reason_flags),
                "removal": removal,
            }
        )
    return rows


def _print_rows(label: str, rows: list[dict]) -> None:
    print(f"\n{label} blocks")
    if not rows:
        print("  none")
        return
    print("  start           end    slots  kWh   cost  avg$/kWh  wbC  preC  postC nextMin reason       remove")
    for row in rows:
        print(
            f"  {row['start']:%a %H:%M}  {row['end']:%H:%M}"
            f"  {row['slots']:>5}  {row['heat_kwh']:>4.2f}"
            f"  ${row['cost']:>5.3f}    {row['avg_price']:>5.3f}"
            f"  {row['avg_wet_bulb']:>4.1f}"
            f"  {row['pre_temp']:>4.1f}  {row['post_temp']:>5.1f}"
            f"   {row['next_min']:>5.1f} {row['reasons']:<12} {row['removal']}"
        )


def _price_source_counts(cfg: dict, dp_grid_times: list[datetime]) -> tuple[int, int]:
    dp_cfg = cfg["hwc"].get("dp_planner", {})
    mpc_entity = dp_cfg.get("mpc_import_price_entity", "sensor.ai_mpc_import_price_forecast")
    mpc_times = {
        dt for dt, _ in hp._forecast_rows_from_entity(cfg, mpc_entity)
    }
    mpc_count = sum(1 for t in dp_grid_times if t in mpc_times)
    return mpc_count, len(dp_grid_times) - mpc_count


def _removal_effect(
    *,
    schedule_w: list[float],
    block: Block,
    start_temperature: float,
    dry_bulb: list[float],
    wet_bulb: list[float],
    draw_off: list[float],
    load_cost: list[float],
    step_minutes: int,
    grid_times: list[datetime],
    tz_name: str,
    hwc_cfg: dict,
    required_target_dates: set[str],
    min_temp: float,
    terminal_target: float,
) -> str:
    removed = list(schedule_w)
    for idx in range(block.start_idx, block.end_idx):
        removed[idx] = 0.0
    replay_cfg = copy.deepcopy(hwc_cfg)
    replay_cfg["optimization_time_step"] = step_minutes
    temps, terminal = hp.simulate_block_temperatures(
        schedule_w=removed,
        start_temperature=start_temperature,
        dry_bulb=dry_bulb,
        wet_bulb=wet_bulb,
        draw_off=draw_off,
        cfg=replay_cfg,
    )
    step = dp._grid_step(grid_times)
    temps_after = list(temps[1:]) + [terminal]
    target_threshold = float(hwc_cfg["thermal"].get("desired_temp", 60)) - float(
        hwc_cfg.get("dp_planner", {}).get("target_tolerance_c", 0.05)
    )
    satisfied = {
        t.astimezone(ZoneInfo(tz_name)).date().isoformat()
        for t, temp in zip(grid_times, temps, strict=True)
        if temp >= target_threshold
    }
    satisfied.update(
        (t + step).astimezone(ZoneInfo(tz_name)).date().isoformat()
        for t, temp in zip(grid_times, temps_after, strict=True)
        if temp >= target_threshold
    )
    missing = sorted(required_target_dates - satisfied)
    flags = []
    if min(temps) < min_temp:
        flags.append(f"floor->{min(temps):.1f}")
    if missing:
        flags.append("miss_target=" + ",".join(missing))
    if terminal < terminal_target:
        flags.append(f"term->{terminal:.1f}")
    if not flags:
        removed_cost = sum(removed[i] / 1000.0 * load_cost[i] * (step_minutes / 60.0) for i in range(len(removed)))
        flags.append(f"still_ok cost=${removed_cost:.2f}")
    return ";".join(flags)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print live HWC block-vs-DP diagnostic")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = _with_daemon_satisfied_date(load_config(args.config))
    tz = ZoneInfo(cfg["timezone"])
    start_temp = hp.get_tank_temperature(cfg)

    block_grid, block_prices = hp.get_import_price_grid(cfg, int(cfg["hwc"].get("horizon_steps", 144)))
    block_step = int(cfg["hwc"].get("optimization_time_step", 30))
    block_dry, block_wet = hp.build_weather_grid(cfg, block_grid, step_minutes=block_step)
    block_draw = hp.build_draw_off_profile(
        block_grid,
        cfg["timezone"],
        cfg["hwc"]["draw_off"]["window_start"],
        cfg["hwc"]["draw_off"]["window_end"],
        cfg["hwc"]["draw_off"]["total_kwh"],
    )
    block_plan = hp.build_block_plan(
        grid_times_utc=block_grid,
        load_cost=block_prices,
        dry_bulb=block_dry,
        wet_bulb=block_wet,
        draw_off=block_draw,
        start_temperature=start_temp,
        cfg=cfg,
    )

    dp_grid, dp_prices = hp.get_dp_import_price_grid(cfg)
    dp_step = int(cfg["hwc"].get("dp_planner", {}).get("internal_step_minutes", 5))
    dp_dry, dp_wet = hp.build_weather_grid(cfg, dp_grid, step_minutes=dp_step)
    dp_draw = hp.build_draw_off_profile(
        dp_grid,
        cfg["timezone"],
        cfg["hwc"]["draw_off"]["window_start"],
        cfg["hwc"]["draw_off"]["window_end"],
        cfg["hwc"]["draw_off"]["total_kwh"],
    )
    dp_plan = hp.build_dp_shadow_plan(
        grid_times_utc=dp_grid,
        load_cost=dp_prices,
        dry_bulb=dp_dry,
        wet_bulb=dp_wet,
        draw_off=dp_draw,
        start_temperature=start_temp,
        cfg=cfg,
    )

    min_temp = float(cfg["hwc"]["thermal"].get("min_temp", 45))
    target_temp = float(cfg["hwc"]["thermal"].get("desired_temp", 60))
    terminal_target = start_temp
    mpc_count, dh_count = _price_source_counts(cfg, dp_grid)
    block_required_targets = dp.required_target_dates(
        block_grid,
        tz_name=cfg["timezone"],
        main_window_end=cfg["hwc"].get("block_planner", {}).get("main_window_end", "18:00"),
    )
    dp_required_targets = set(dp_plan["required_target_dates"])

    print(f"generated={datetime.now(tz).isoformat(timespec='seconds')}")
    print(f"start_temp={start_temp:.1f}C satisfied_dates={cfg['hwc'].get('block_planner', {}).get('main_satisfied_dates', [])}")
    print(f"DP price source slots: mpc_5m={mpc_count}, dh_tail_5m={dh_count}")
    print(_summary(
        label="block",
        schedule_w=block_plan["schedule_w"],
        temperatures=block_plan["temperatures"],
        terminal_temperature=block_plan["terminal_temperature"],
        step_minutes=block_step,
    ))
    print(_summary(
        label="dp",
        schedule_w=dp_plan["schedule_w"],
        temperatures=dp_plan["temperatures"],
        terminal_temperature=dp_plan["terminal_temperature"],
        step_minutes=dp_step,
    ))
    print(f"dp objective=${dp_plan['objective_cost']:.3f} breakdown={dp_plan['objective_breakdown']}")
    print(f"dp required_targets={dp_plan['required_target_dates']} satisfied={dp_plan['target_satisfied_dates']}")

    _print_rows(
        "Block",
        _block_rows(
            grid_times=block_grid,
            schedule_w=block_plan["schedule_w"],
            temperatures=block_plan["temperatures"],
            terminal_temperature=block_plan["terminal_temperature"],
            load_cost=block_prices,
            dry_bulb=block_dry,
            wet_bulb=block_wet,
            draw_off=block_draw,
            step_minutes=block_step,
            tz_name=cfg["timezone"],
            min_temp=min_temp,
            target_temp=target_temp,
            terminal_target=terminal_target,
            hwc_cfg=cfg["hwc"],
            required_target_dates=block_required_targets,
        ),
    )
    _print_rows(
        "DP",
        _block_rows(
            grid_times=dp_grid,
            schedule_w=dp_plan["schedule_w"],
            temperatures=dp_plan["temperatures"],
            terminal_temperature=dp_plan["terminal_temperature"],
            load_cost=dp_prices,
            dry_bulb=dp_dry,
            wet_bulb=dp_wet,
            draw_off=dp_draw,
            step_minutes=dp_step,
            tz_name=cfg["timezone"],
            min_temp=min_temp,
            target_temp=target_temp,
            terminal_target=terminal_target,
            hwc_cfg=cfg["hwc"],
            required_target_dates=dp_required_targets,
        ),
    )


if __name__ == "__main__":
    main()
