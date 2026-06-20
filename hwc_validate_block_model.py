#!/usr/bin/env python3
"""Validate the live HWC block transition model against extracted cycles.

This report answers a narrower question than the stratified validation: if the live
block scheduler says "run the fixed-speed compressor from this starting state", how
well does its single-node transition predict end temperature, duration to 60 C, and
electrical kWh?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import hwc_fit_thermal_model as fit
import hwc_planner
from config_utils import load_config

DEFAULT_CYCLES_CSV = "data/hwc_cop_cycles.csv"
DEFAULT_REPORT_CSV = "data/hwc_block_model_validation.csv"
DEFAULT_REPORT_MD = "docs/hwc_block_model_validation.md"


def load_validation_cycles_from_frame(df: pd.DataFrame, *, target_c: float = 60.0) -> pd.DataFrame:
    cycles = fit.prepare_cycles(df, target_c=target_c)
    return cycles[cycles["clean"] & cycles["compressor_only"] & cycles["to_target"]].copy()


def load_validation_cycles(path: str, *, target_c: float = 60.0) -> pd.DataFrame:
    return load_validation_cycles_from_frame(pd.read_csv(path), target_c=target_c)


def _cycle_step_minutes(duration_min: float, requested_step_min: float) -> tuple[int, float]:
    steps = max(1, int(math.ceil(float(duration_min) / float(requested_step_min))))
    return steps, float(duration_min) / steps


def _block_cfg_for_step(config: dict, step_min: float) -> dict:
    hwc_cfg = dict(config["hwc"])
    hwc_cfg["optimization_time_step"] = float(step_min)
    return hwc_cfg


def _simulate_observed_runtime(row: pd.Series, config: dict, *, step_min: float) -> dict:
    steps, actual_step_min = _cycle_step_minutes(row["dur_min"], step_min)
    hwc_cfg = _block_cfg_for_step(config, actual_step_min)
    schedule = [float(hwc_cfg["thermal"]["nominal_power_w"])] * steps
    temps, terminal = hwc_planner.simulate_block_temperatures(
        schedule_w=schedule,
        start_temperature=float(row["tank_start"]),
        dry_bulb=[float(row["ambient"])] * steps,
        wet_bulb=[float(row["wet_bulb"])] * steps,
        draw_off=[0.0] * steps,
        cfg=hwc_cfg,
    )
    return {
        "step_min": actual_step_min,
        "end_temp": terminal,
        "series": temps + [terminal],
    }


def _duration_to_target(
    row: pd.Series,
    config: dict,
    *,
    target_c: float,
    step_min: float,
    max_duration_min: float,
) -> tuple[float, float]:
    hwc_cfg = _block_cfg_for_step(config, step_min)
    max_steps = max(1, int(math.ceil(max_duration_min / step_min)))
    temp = float(row["tank_start"])
    for steps in range(1, max_steps + 1):
        _, temp = hwc_planner.simulate_block_temperatures(
            schedule_w=[float(hwc_cfg["thermal"]["nominal_power_w"])] * steps,
            start_temperature=float(row["tank_start"]),
            dry_bulb=[float(row["ambient"])] * steps,
            wet_bulb=[float(row["wet_bulb"])] * steps,
            draw_off=[0.0] * steps,
            cfg=hwc_cfg,
        )
        if temp >= target_c:
            minutes = steps * step_min
            kwh = float(hwc_cfg["thermal"]["nominal_power_w"]) / 1000.0 * minutes / 60.0
            return round(minutes, 1), round(kwh, 3)
    return np.nan, np.nan


def _error(predicted: float, observed: float) -> float:
    if pd.isna(predicted) or pd.isna(observed):
        return np.nan
    return round(float(predicted) - float(observed), 3)


def validate_cycles(
    cycles: pd.DataFrame,
    config: dict,
    *,
    target_c: float = 60.0,
    step_min: float = 5.0,
    max_duration_min: float = 240.0,
) -> pd.DataFrame:
    rows = []
    nominal_power_kw = float(config["hwc"]["thermal"]["nominal_power_w"]) / 1000.0
    for _, row in cycles.iterrows():
        observed = _simulate_observed_runtime(row, config, step_min=step_min)
        pred_min, pred_kwh = _duration_to_target(
            row,
            config,
            target_c=target_c,
            step_min=step_min,
            max_duration_min=max_duration_min,
        )
        observed_model_kwh = nominal_power_kw * float(row["dur_min"]) / 60.0
        rows.append(
            {
                "start": row["start"],
                "cycle_class": row["cycle_class"],
                "tank_start": round(float(row["tank_start"]), 2),
                "tank_end_obs": round(float(row["tank_end"]), 2),
                "ambient": round(float(row["ambient"]), 2),
                "wet_bulb": round(float(row["wet_bulb"]), 2),
                "observed_duration_min": round(float(row["dur_min"]), 1),
                "predicted_duration_min": pred_min,
                "duration_err_min": _error(pred_min, row["dur_min"]),
                "observed_elec_kwh": round(float(row["elec_kwh"]), 3),
                "predicted_elec_kwh": pred_kwh,
                "elec_err_kwh": _error(pred_kwh, row["elec_kwh"]),
                "nominal_elec_for_observed_runtime_kwh": round(observed_model_kwh, 3),
                "runtime_power_proxy_err_kwh": _error(observed_model_kwh, row["elec_kwh"]),
                "predicted_end_temp_at_observed_runtime": observed["end_temp"],
                "end_temp_err_c": _error(observed["end_temp"], row["tank_end"]),
                "cop": round(float(row["cop"]), 2),
            }
        )
    return pd.DataFrame(rows)


def _mae(series: pd.Series) -> float:
    valid = series.dropna()
    return round(float(valid.abs().mean()), 3) if not valid.empty else np.nan


def _bias(series: pd.Series) -> float:
    valid = series.dropna()
    return round(float(valid.mean()), 3) if not valid.empty else np.nan


def _summary(report: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in [("all", report), *report.groupby("cycle_class")]:
        rows.append(
            {
                "class": label,
                "n": int(len(frame)),
                "duration_mae_min": _mae(frame["duration_err_min"]),
                "duration_bias_min": _bias(frame["duration_err_min"]),
                "end_temp_mae_c": _mae(frame["end_temp_err_c"]),
                "end_temp_bias_c": _bias(frame["end_temp_err_c"]),
                "elec_mae_kwh": _mae(frame["elec_err_kwh"]),
                "elec_bias_kwh": _bias(frame["elec_err_kwh"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(report: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = _summary(report)
    detail_cols = [
        "start",
        "cycle_class",
        "tank_start",
        "tank_end_obs",
        "wet_bulb",
        "observed_duration_min",
        "predicted_duration_min",
        "duration_err_min",
        "observed_elec_kwh",
        "predicted_elec_kwh",
        "elec_err_kwh",
        "predicted_end_temp_at_observed_runtime",
        "end_temp_err_c",
    ]
    lines = [
        "# HWC Block Model Validation",
        "",
        "Offline validation of the live block scheduler's single-node transition model against",
        "clean compressor-only cycles in `data/hwc_cop_cycles.csv`.",
        "",
        "This evaluates the model used by `hwc_planner.py`, not the offline stratified shape",
        "diagnostic. It is intended to show whether block scheduling errors are coming from",
        "duration/heat-rate assumptions, nominal electrical power, or end-temperature replay.",
        "",
        "## Aggregate Error",
        "",
        "| " + " | ".join(summary.columns) + " |",
        "| " + " | ".join(["---"] * len(summary.columns)) + " |",
    ]
    for _, row in summary.astype(str).iterrows():
        lines.append("| " + " | ".join(row[col] for col in summary.columns) + " |")
    lines += [
        "",
        "## Cycle Detail",
        "",
        "| " + " | ".join(detail_cols) + " |",
        "| " + " | ".join(["---"] * len(detail_cols)) + " |",
    ]
    for _, row in report[detail_cols].astype(str).iterrows():
        lines.append("| " + " | ".join(row[col] for col in detail_cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-csv", default=DEFAULT_CYCLES_CSV)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--target-c", type=float, default=60.0)
    parser.add_argument("--step-min", type=float, default=5.0)
    parser.add_argument("--max-duration-min", type=float, default=240.0)
    parser.add_argument("--report-csv", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    args = parser.parse_args()

    report = validate_cycles(
        load_validation_cycles(args.cycles_csv, target_c=args.target_c),
        load_config(args.config),
        target_c=args.target_c,
        step_min=args.step_min,
        max_duration_min=args.max_duration_min,
    )
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.report_csv, index=False)
    write_markdown(report, args.report_md)
    print(f"wrote {args.report_csv} and {args.report_md} ({len(report)} cycles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
