#!/usr/bin/env python3
"""Validate HWC tank models against raw per-cycle Home Assistant traces.

This pulls tank probe, exhaust, compressor, and house-load proxy traces from
InfluxDB for the clean cycles in ``data/hwc_cop_cycles.csv``. The raw trace CSV
is intended as a local/ignored diagnostic artifact; the compact report is small
enough to commit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import hwc_cop_analysis as cop
import hwc_planner
import hwc_stratified_model as strat
import hwc_validate_stratified_model as summary_val
from config_utils import load_config

DEFAULT_CYCLES_CSV = "data/hwc_cop_cycles.csv"
DEFAULT_TRACE_CSV = "data/hwc_cycle_traces.csv"
DEFAULT_REPORT_CSV = "data/hwc_cycle_trace_validation.csv"
DEFAULT_REPORT_MD = "docs/hwc_cycle_trace_validation.md"
DEFAULT_LOO_REPORT_CSV = "data/hwc_cycle_trace_loo_validation.csv"
DEFAULT_LOO_REPORT_MD = "docs/hwc_cycle_trace_loo_validation.md"
LOCAL_TZ = cop.LOCAL_TZ


def _local_to_utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TZ)
    return ts.tz_convert("UTC")


def _cycle_bounds_utc(row: pd.Series) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = _local_to_utc(str(row["start"]))
    return start, start + pd.Timedelta(minutes=float(row["dur_min"]))


def _query_cycle_series(row: pd.Series, *, padding_min: float = 10.0) -> dict[str, pd.Series]:
    start_utc, end_utc = _cycle_bounds_utc(row)
    client = cop._client()
    since_utc = start_utc - pd.Timedelta(minutes=float(padding_min))
    until_utc = end_utc + pd.Timedelta(minutes=float(padding_min))
    since = since_utc.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    until = until_utc.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")
    return {
        "tank": cop._series(
            client, "sensor__temperature", "heat_pump_temperature",
            since=since, until=until,
        ),
        "exhaust": cop._series(
            client, "sensor__temperature", "aquatech_exhaust_temperature",
            since=since, until=until,
        ),
        "power": cop._series(
            client, "sensor__power", "remaining_power_load",
            since=since, until=until,
        ),
        "compressor": cop._series(
            client, "binary_sensor__running", "aquatech_compressor",
            since=since, until=until,
        ),
    }


def _trace_index(row: pd.Series, *, step_seconds: int) -> pd.DatetimeIndex:
    start_utc, end_utc = _cycle_bounds_utc(row)
    return pd.date_range(start_utc, end_utc, freq=f"{step_seconds}s", tz="UTC")


def _single_node_trace(row: pd.Series, config: dict, *, steps: int, step_h: float) -> list[float]:
    hwc_cfg = dict(config["hwc"])
    hwc_cfg["optimization_time_step"] = step_h * 60.0
    temps, terminal = hwc_planner.simulate_block_temperatures(
        schedule_w=[float(hwc_cfg["thermal"]["nominal_power_w"])] * steps,
        start_temperature=float(row["tank_start"]),
        dry_bulb=[float(row["ambient"])] * steps,
        draw_off=[0.0] * steps,
        cfg=hwc_cfg,
    )
    return temps + [terminal]


def _stratified_trace(
    row: pd.Series,
    params: strat.StratifiedTankParams,
    *,
    steps: int,
    step_h: float,
) -> list[float]:
    heat_kwh = float(row["therm_kwh"]) / steps if steps > 0 else 0.0
    state = strat.StratifiedTankState(
        cold_temp_c=float(row["tank_start"]),
        hot_temp_c=float(row["tank_start"]),
        hot_fraction=0.0,
    )
    values = []
    for _ in range(steps):
        values.append(round(strat.probe_temp_c(state, params), 2))
        state = strat.apply_idle_loss(state, params, ambient_c=float(row["ambient"]), step_h=step_h)
        state = strat.apply_heat(state, params, heat_kwh=heat_kwh)
    values.append(round(strat.probe_temp_c(state, params), 2))
    return values


def _model_traces(row: pd.Series, config: dict, params: strat.StratifiedTankParams, *, idx_len: int) -> dict:
    steps = max(1, idx_len - 1)
    step_h = float(row["dur_min"]) / 60.0 / steps
    return {
        "single_node_temp": _single_node_trace(row, config, steps=steps, step_h=step_h),
        "stratified_temp": _stratified_trace(row, params, steps=steps, step_h=step_h),
    }


def _fit_params(cycles: pd.DataFrame, config: dict) -> strat.StratifiedTankParams:
    fit_result = summary_val.fit.fit_parameters(cycles)
    return summary_val._thermal_params(config, fit_result)


def _mae(predicted: pd.Series, observed: pd.Series) -> float:
    err = predicted - observed
    return round(float(err.abs().mean()), 3)


def _max_abs_error(predicted: pd.Series, observed: pd.Series) -> float:
    err = predicted - observed
    return round(float(err.abs().max()), 3)


def _initial_flat_error_minutes(predicted: pd.Series, observed: pd.Series, start_temp: float) -> float:
    threshold = float(start_temp) + 0.5
    obs = observed[observed >= threshold]
    pred = predicted[predicted >= threshold]
    if obs.empty or pred.empty:
        return np.nan
    return round((pred.index[0] - obs.index[0]).total_seconds() / 60.0, 1)


def _validate_one_trace_cycle(
    row: pd.Series,
    config: dict,
    params: strat.StratifiedTankParams,
    *,
    step_seconds: int,
    series: dict[str, pd.Series],
) -> tuple[dict | None, list[dict]]:
    idx = _trace_index(row, step_seconds=step_seconds)
    observed = cop._interp_to_idx(series["tank"], idx)
    traces = _model_traces(row, config, params, idx_len=len(idx))
    block = pd.Series(traces["single_node_temp"], index=idx)
    layered = pd.Series(traces["stratified_temp"], index=idx)
    exhaust = cop._interp_to_idx(series["exhaust"], idx)
    power = cop._interp_to_idx(series["power"], idx)
    compressor = cop._state_to_idx(series["compressor"], idx)
    valid = observed.dropna()
    if valid.empty:
        return None, []

    start_local = str(row["start"])
    report = {
        "start": start_local,
        "cycle_class": row["cycle_class"],
        "points": int(observed.notna().sum()),
        "duration_min": float(row["dur_min"]),
        "observed_start": round(float(valid.iloc[0]), 2),
        "observed_end": round(float(valid.iloc[-1]), 2),
        "single_node_mae_c": _mae(block, observed),
        "single_node_max_err_c": _max_abs_error(block, observed),
        "single_node_end_err_c": round(float(block.iloc[-1] - valid.iloc[-1]), 3),
        "single_node_flat_lag_err_min": _initial_flat_error_minutes(
            block, observed, float(row["tank_start"])
        ),
        "stratified_mae_c": _mae(layered, observed),
        "stratified_max_err_c": _max_abs_error(layered, observed),
        "stratified_end_err_c": round(float(layered.iloc[-1] - valid.iloc[-1]), 3),
        "stratified_flat_lag_err_min": _initial_flat_error_minutes(
            layered, observed, float(row["tank_start"])
        ),
        "exhaust_start": round(float(exhaust.dropna().iloc[0]), 2) if exhaust.notna().any() else np.nan,
        "exhaust_end": round(float(exhaust.dropna().iloc[-1]), 2) if exhaust.notna().any() else np.nan,
        "exhaust_max": round(float(exhaust.max()), 2) if exhaust.notna().any() else np.nan,
    }
    trace_rows = []
    for ts in idx:
        trace_rows.append({
            "cycle_start": start_local,
            "time": ts.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
            "minute": round((ts - idx[0]).total_seconds() / 60.0, 2),
            "observed_tank_c": round(float(observed.loc[ts]), 3) if pd.notna(observed.loc[ts]) else np.nan,
            "single_node_tank_c": round(float(block.loc[ts]), 3),
            "stratified_tank_c": round(float(layered.loc[ts]), 3),
            "exhaust_c": round(float(exhaust.loc[ts]), 3) if pd.notna(exhaust.loc[ts]) else np.nan,
            "remaining_power_w": round(float(power.loc[ts]), 3) if pd.notna(power.loc[ts]) else np.nan,
            "compressor_on": bool(compressor.loc[ts]),
        })
    return report, trace_rows


def validate_trace_cycles(
    cycles: pd.DataFrame,
    config: dict,
    *,
    step_seconds: int = 60,
    series_loader=_query_cycle_series,
) -> tuple[pd.DataFrame, pd.DataFrame, strat.StratifiedTankParams]:
    params = _fit_params(cycles, config)
    trace_rows = []
    report_rows = []
    for _, row in cycles.iterrows():
        report, rows = _validate_one_trace_cycle(
            row, config, params, step_seconds=step_seconds, series=series_loader(row)
        )
        if report is None:
            continue
        report_rows.append(report)
        trace_rows.extend(rows)
    return pd.DataFrame(report_rows), pd.DataFrame(trace_rows), params


def validate_leave_one_out_trace_cycles(
    cycles: pd.DataFrame,
    config: dict,
    *,
    step_seconds: int = 60,
    series_loader=_query_cycle_series,
) -> pd.DataFrame:
    report_rows = []
    for idx, row in cycles.iterrows():
        train = cycles.drop(index=idx)
        if train.empty:
            continue
        params = _fit_params(train, config)
        report, _ = _validate_one_trace_cycle(
            row, config, params, step_seconds=step_seconds, series=series_loader(row)
        )
        if report is None:
            continue
        report.update({
            "train_cycles": int(len(train)),
            "loo_probe_height_fraction": params.probe_height_fraction,
            "loo_thermocline_width_fraction": params.thermocline_width_fraction,
        })
        report_rows.append(report)
    return pd.DataFrame(report_rows)


def load_cycles(path: str, *, target_c: float) -> pd.DataFrame:
    return summary_val.load_validation_cycles(path, target_c=target_c)


def load_cycles_from_frame(df: pd.DataFrame, *, target_c: float) -> pd.DataFrame:
    return summary_val.load_validation_cycles_from_frame(df, target_c=target_c)


def _summary_metric(report: pd.DataFrame, column: str) -> float:
    return round(float(report[column].mean()), 3) if not report.empty else np.nan


def write_markdown(report: pd.DataFrame, params: strat.StratifiedTankParams, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HWC Cycle Trace Validation",
        "",
        "Raw-trace validation against clean compressor-only cycles from `data/hwc_cop_cycles.csv`.",
        "",
        "The stratified trace still uses observed per-cycle thermal input, so it is a probe-shape",
        "diagnostic rather than a forecast-ready model. The trace CSV is local diagnostic data;",
        "this report captures the compact metrics.",
        "",
        "## Parameters",
        "",
        "| parameter | value |",
        "| --- | --- |",
        f"| `probe_height_fraction` | `{params.probe_height_fraction}` |",
        f"| `thermocline_width_fraction` | `{params.thermocline_width_fraction}` |",
        f"| `hot_target_c` | `{params.hot_target_c}` |",
        "",
        "## Aggregate Trace Error",
        "",
        "| metric | single_node | stratified |",
        "| --- | --- | --- |",
        f"| mean trace MAE C | `{_summary_metric(report, 'single_node_mae_c')}` | "
        f"`{_summary_metric(report, 'stratified_mae_c')}` |",
        f"| mean max abs error C | `{_summary_metric(report, 'single_node_max_err_c')}` | "
        f"`{_summary_metric(report, 'stratified_max_err_c')}` |",
        f"| mean end error C | `{_summary_metric(report, 'single_node_end_err_c')}` | "
        f"`{_summary_metric(report, 'stratified_end_err_c')}` |",
        f"| mean +0.5C flat-lag error min | "
        f"`{_summary_metric(report, 'single_node_flat_lag_err_min')}` | "
        f"`{_summary_metric(report, 'stratified_flat_lag_err_min')}` |",
        "",
        "## Cycle Detail",
        "",
    ]
    cols = [
        "start", "cycle_class", "points", "duration_min", "observed_start", "observed_end",
        "single_node_mae_c", "single_node_max_err_c", "single_node_end_err_c",
        "single_node_flat_lag_err_min", "stratified_mae_c", "stratified_max_err_c",
        "stratified_end_err_c", "stratified_flat_lag_err_min", "exhaust_start",
        "exhaust_end", "exhaust_max",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in report[cols].astype(str).iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_loo_markdown(report: pd.DataFrame, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HWC Cycle Trace Leave-One-Out Validation",
        "",
        "Held-out raw-trace validation against clean compressor-only cycles from",
        "`data/hwc_cop_cycles.csv`.",
        "",
        "For each row, the stratified probe-shape parameters are fitted from the other",
        "clean cycles, then replayed against the held-out cycle. The stratified trace still",
        "uses observed per-cycle thermal input, so this validates probe-shape transfer, not",
        "forecast-ready heat input.",
        "",
        "## Aggregate Held-Out Trace Error",
        "",
        "| metric | single_node | stratified_loo |",
        "| --- | --- | --- |",
        f"| mean trace MAE C | `{_summary_metric(report, 'single_node_mae_c')}` | "
        f"`{_summary_metric(report, 'stratified_mae_c')}` |",
        f"| mean max abs error C | `{_summary_metric(report, 'single_node_max_err_c')}` | "
        f"`{_summary_metric(report, 'stratified_max_err_c')}` |",
        f"| mean end error C | `{_summary_metric(report, 'single_node_end_err_c')}` | "
        f"`{_summary_metric(report, 'stratified_end_err_c')}` |",
        f"| mean +0.5C flat-lag error min | "
        f"`{_summary_metric(report, 'single_node_flat_lag_err_min')}` | "
        f"`{_summary_metric(report, 'stratified_flat_lag_err_min')}` |",
        "",
        "## Cycle Detail",
        "",
    ]
    cols = [
        "start", "cycle_class", "train_cycles", "loo_probe_height_fraction",
        "loo_thermocline_width_fraction", "points", "duration_min", "observed_start",
        "observed_end", "single_node_mae_c", "single_node_max_err_c",
        "single_node_end_err_c", "single_node_flat_lag_err_min", "stratified_mae_c",
        "stratified_max_err_c", "stratified_end_err_c", "stratified_flat_lag_err_min",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in report[cols].astype(str).iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-csv", default=DEFAULT_CYCLES_CSV)
    parser.add_argument("--trace-csv", default=DEFAULT_TRACE_CSV)
    parser.add_argument("--report-csv", default=DEFAULT_REPORT_CSV)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--loo-report-csv", default=DEFAULT_LOO_REPORT_CSV)
    parser.add_argument("--loo-report-md", default=DEFAULT_LOO_REPORT_MD)
    parser.add_argument("--step-seconds", type=int, default=60)
    args = parser.parse_args()

    config = load_config()
    cycles = load_cycles(
        args.cycles_csv,
        target_c=float(config["hwc"]["thermal"].get("desired_temp", 60.0)),
    )
    if args.leave_one_out:
        report = validate_leave_one_out_trace_cycles(
            cycles, config, step_seconds=args.step_seconds,
        )
        Path(args.loo_report_csv).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.loo_report_csv, index=False)
        write_loo_markdown(report, args.loo_report_md)
        print(f"validated held-out cycles: {len(report)}")
        print(f"mean single-node trace MAE C: {_summary_metric(report, 'single_node_mae_c')}")
        print(f"mean stratified LOO trace MAE C: {_summary_metric(report, 'stratified_mae_c')}")
        print(f"wrote {args.loo_report_csv}")
        print(f"wrote {args.loo_report_md}")
        return 0

    report, traces, params = validate_trace_cycles(
        cycles, config, step_seconds=args.step_seconds,
    )
    Path(args.trace_csv).parent.mkdir(parents=True, exist_ok=True)
    traces.to_csv(args.trace_csv, index=False)
    report.to_csv(args.report_csv, index=False)
    write_markdown(report, params, args.report_md)
    print(f"validated cycles: {len(report)}")
    print(f"trace rows: {len(traces)}")
    print(f"mean single-node trace MAE C: {_summary_metric(report, 'single_node_mae_c')}")
    print(f"mean stratified trace MAE C: {_summary_metric(report, 'stratified_mae_c')}")
    print(f"wrote {args.trace_csv}")
    print(f"wrote {args.report_csv}")
    print(f"wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
