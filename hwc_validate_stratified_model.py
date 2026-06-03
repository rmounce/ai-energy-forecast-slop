#!/usr/bin/env python3
"""Validate HWC tank models against extracted compressor-cycle summaries.

This is an offline shadow diagnostic. It uses the cycle-level calibration table,
not raw HA curves, so it compares end state and observed probe-rise milestones
rather than every point on the tank-temperature trace.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import hwc_fit_thermal_model as fit
import hwc_planner
import hwc_stratified_model as strat
from config_utils import load_config

DEFAULT_CYCLES_CSV = "data/hwc_cop_cycles.csv"
DEFAULT_REPORT_CSV = "data/hwc_stratified_validation.csv"
DEFAULT_REPORT_MD = "docs/hwc_stratified_validation.md"
MILESTONES = (10, 50, 90)


def _thermal_params(config: dict, fit_result: dict) -> strat.StratifiedTankParams:
    th = config["hwc"]["thermal"]
    hints = fit_result.get("stratified_hints", {})
    return strat.StratifiedTankParams(
        volume_l=float(th.get("volume_l", 225.0)),
        density_kg_per_m3=float(th.get("density", 997.0)),
        heat_capacity_kj_per_kg_c=float(th.get("heat_capacity", 4.184)),
        standing_loss_ua_kw_per_c=float(th.get("standing_loss_ua_kw_per_c", 0.0025)),
        hot_target_c=float(th.get("desired_temp", 60.0)),
        probe_height_fraction=float(hints.get("probe_height_fraction", 0.55)),
        thermocline_width_fraction=float(hints.get("thermocline_width_fraction", 0.18)),
    )


def _cycle_step_minutes(duration_min: float, requested_step_min: float) -> tuple[int, float]:
    steps = max(1, int(math.ceil(float(duration_min) / float(requested_step_min))))
    return steps, float(duration_min) / steps


def _first_reach_minutes(values: list[float], *, start: float, end: float, pct: int, step_min: float) -> float:
    lift = float(end) - float(start)
    if lift <= 0:
        return np.nan
    threshold = float(start) + lift * pct / 100.0
    for idx, value in enumerate(values):
        if value >= threshold:
            return round(idx * step_min, 1)
    return np.nan


def _simulate_block_cycle(row: pd.Series, config: dict, *, step_min: float) -> dict:
    steps, actual_step_min = _cycle_step_minutes(row["dur_min"], step_min)
    hwc_cfg = dict(config["hwc"])
    hwc_cfg["optimization_time_step"] = actual_step_min
    schedule = [float(hwc_cfg["thermal"]["nominal_power_w"])] * steps
    temps, terminal = hwc_planner.simulate_block_temperatures(
        schedule_w=schedule,
        start_temperature=float(row["tank_start"]),
        dry_bulb=[float(row["ambient"])] * steps,
        draw_off=[0.0] * steps,
        cfg=hwc_cfg,
    )
    series = temps + [terminal]
    return {
        "end_temp": terminal,
        **{
            f"rise_{pct}_min": _first_reach_minutes(
                series,
                start=float(row["tank_start"]),
                end=float(row["tank_end"]),
                pct=pct,
                step_min=actual_step_min,
            )
            for pct in MILESTONES
        },
    }


def _simulate_stratified_cycle(
    row: pd.Series,
    params: strat.StratifiedTankParams,
    *,
    step_min: float,
) -> dict:
    steps, actual_step_min = _cycle_step_minutes(row["dur_min"], step_min)
    heat_per_step = float(row["therm_kwh"]) / steps
    probes, states = strat.simulate_probe_temperatures(
        schedule_heat_kwh=[heat_per_step] * steps,
        draw_off_kwh=[0.0] * steps,
        ambient_c=[float(row["ambient"])] * steps,
        initial_state=strat.StratifiedTankState(
            cold_temp_c=float(row["tank_start"]),
            hot_temp_c=float(row["tank_start"]),
            hot_fraction=0.0,
        ),
        params=params,
        step_h=actual_step_min / 60.0,
    )
    final_state = states[-1] if states else strat.StratifiedTankState(
        cold_temp_c=float(row["tank_start"]),
        hot_temp_c=float(row["tank_start"]),
        hot_fraction=0.0,
    )
    # Apply the last interval once more to report terminal state after the whole cycle.
    final_state = strat.apply_idle_loss(
        final_state, params, ambient_c=float(row["ambient"]), step_h=actual_step_min / 60.0
    )
    final_state = strat.apply_heat(final_state, params, heat_kwh=heat_per_step)
    terminal = round(strat.probe_temp_c(final_state, params), 2)
    series = probes + [terminal]
    return {
        "end_temp": terminal,
        "hot_fraction_end": round(final_state.hot_fraction, 3),
        "mean_temp_end": round(strat.mean_temp_c(final_state), 2),
        **{
            f"rise_{pct}_min": _first_reach_minutes(
                series,
                start=float(row["tank_start"]),
                end=float(row["tank_end"]),
                pct=pct,
                step_min=actual_step_min,
            )
            for pct in MILESTONES
        },
    }


def _error(predicted: float, observed: float) -> float:
    if pd.isna(predicted) or pd.isna(observed):
        return np.nan
    return round(float(predicted) - float(observed), 2)


def load_validation_cycles_from_frame(df: pd.DataFrame, *, target_c: float = 60.0) -> pd.DataFrame:
    cycles = fit.prepare_cycles(df, target_c=target_c)
    return cycles[cycles["clean"] & cycles["compressor_only"] & cycles["to_target"]].copy()


def load_validation_cycles(path: str, *, target_c: float = 60.0) -> pd.DataFrame:
    return load_validation_cycles_from_frame(pd.read_csv(path), target_c=target_c)


def validate_cycles(
    cycles: pd.DataFrame,
    config: dict,
    *,
    step_min: float = 1.0,
) -> tuple[pd.DataFrame, strat.StratifiedTankParams]:
    fit_result = fit.fit_parameters(cycles)
    params = _thermal_params(config, fit_result)
    rows = []
    for _, row in cycles.iterrows():
        block = _simulate_block_cycle(row, config, step_min=step_min)
        layered = _simulate_stratified_cycle(row, params, step_min=step_min)
        out = {
            "start": row["start"],
            "cycle_class": row["cycle_class"],
            "dur_min": int(row["dur_min"]),
            "tank_start": row["tank_start"],
            "tank_end_obs": row["tank_end"],
            "ambient": row["ambient"],
            "wet_bulb": row["wet_bulb"],
            "cop": row["cop"],
            "block_end": block["end_temp"],
            "block_end_err": _error(block["end_temp"], row["tank_end"]),
            "strat_end": layered["end_temp"],
            "strat_end_err": _error(layered["end_temp"], row["tank_end"]),
            "strat_hot_fraction_end": layered["hot_fraction_end"],
            "strat_mean_temp_end": layered["mean_temp_end"],
        }
        for pct in MILESTONES:
            observed = row[f"probe_rise_{pct}_min"]
            out[f"rise_{pct}_obs"] = observed
            out[f"block_rise_{pct}"] = block[f"rise_{pct}_min"]
            out[f"block_rise_{pct}_err"] = _error(block[f"rise_{pct}_min"], observed)
            out[f"strat_rise_{pct}"] = layered[f"rise_{pct}_min"]
            out[f"strat_rise_{pct}_err"] = _error(layered[f"rise_{pct}_min"], observed)
        rows.append(out)
    return pd.DataFrame(rows), params


def _mae(series: pd.Series) -> float:
    valid = series.dropna()
    return round(float(valid.abs().mean()), 2) if not valid.empty else np.nan


def _metric_summary(series: pd.Series) -> str:
    valid = series.dropna()
    misses = int(series.isna().sum())
    mae = round(float(valid.abs().mean()), 2) if not valid.empty else np.nan
    return f"MAE `{mae}`; reached `{len(valid)}/{len(series)}`; misses `{misses}`"


def write_markdown(report: pd.DataFrame, params: strat.StratifiedTankParams, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HWC Stratified Model Validation",
        "",
        "Offline validation against clean compressor-only cycles in `data/hwc_cop_cycles.csv`.",
        "",
        "This report uses cycle-level probe-rise milestones, not raw HA temperature curves.",
        "The stratified model is therefore a shape diagnostic only; it is not yet a live-control model.",
        "The stratified replay uses observed per-cycle thermal input, so its end-temperature error is",
        "not a fair forecast benchmark; the milestone timing errors are the more useful signal.",
        "",
        "## Parameters",
        "",
        "| parameter | value |",
        "| --- | --- |",
        f"| `probe_height_fraction` | `{params.probe_height_fraction}` |",
        f"| `thermocline_width_fraction` | `{params.thermocline_width_fraction}` |",
        f"| `hot_target_c` | `{params.hot_target_c}` |",
        f"| `standing_loss_ua_kw_per_c` | `{params.standing_loss_ua_kw_per_c}` |",
        "",
        "## Aggregate Error",
        "",
        "| metric | single_node | stratified |",
        "| --- | --- | --- |",
        f"| end temp C | `{_mae(report['block_end_err'])}` | `{_mae(report['strat_end_err'])}` |",
    ]
    for pct in MILESTONES:
        lines.append(
            f"| rise {pct}% min | {_metric_summary(report[f'block_rise_{pct}_err'])} | "
            f"{_metric_summary(report[f'strat_rise_{pct}_err'])} |"
        )

    display_cols = [
        "start", "cycle_class", "dur_min", "tank_start", "tank_end_obs",
        "block_end", "block_end_err", "strat_end", "strat_end_err",
        "rise_10_obs", "block_rise_10", "block_rise_10_err", "strat_rise_10",
        "strat_rise_10_err", "rise_50_obs", "block_rise_50", "block_rise_50_err",
        "strat_rise_50", "strat_rise_50_err", "rise_90_obs", "block_rise_90",
        "block_rise_90_err", "strat_rise_90", "strat_rise_90_err",
    ]
    lines += [
        "",
        "## Cycle Detail",
        "",
        "| " + " | ".join(display_cols) + " |",
        "| " + " | ".join(["---"] * len(display_cols)) + " |",
    ]
    table = report[display_cols].astype(str)
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display_cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-csv", default=DEFAULT_CYCLES_CSV)
    parser.add_argument("--report-csv", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--step-min", type=float, default=1.0)
    args = parser.parse_args()

    config = load_config()
    cycles = load_validation_cycles(
        args.cycles_csv,
        target_c=float(config["hwc"]["thermal"].get("desired_temp", 60.0)),
    )
    report, params = validate_cycles(cycles, config, step_min=args.step_min)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_csv, index=False)
    write_markdown(report, params, args.report_md)
    print(f"validated cycles: {len(report)}")
    print(f"single-node end MAE C: {_mae(report['block_end_err'])}")
    print(f"stratified end MAE C: {_mae(report['strat_end_err'])}")
    print(f"wrote {args.report_csv}")
    print(f"wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
