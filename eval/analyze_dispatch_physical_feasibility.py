#!/usr/bin/env python3
"""
Audit rolling MPC raw outputs for simple physical-feasibility artifacts.

This is a guardrail for tariffed site-flow evals. It does not prove the LP is a
perfect inverter model; it flags the common loopholes that would make a
profit/path-quality conclusion suspect, such as simultaneous import/export,
simultaneous charge/discharge, curtailing more PV than is available, SoC balance
drift, or grid-flow balance residuals.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
INTERVAL_H = 5.0 / 60.0
EFF_C = 0.95
EFF_D = 0.95
CAPACITY_KWH = 40.0
MAX_POWER_KW = 10.0


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find parquet: {path_arg}")


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(["source", "time"], kind="stable").reset_index(drop=True)


def _violation_amount(series: pd.Series, threshold: float) -> pd.Series:
    return (series.astype(float) - float(threshold)).clip(lower=0.0)


def _flag_summary(
    df: pd.DataFrame,
    *,
    flag: str,
    amount: str,
    group_cols: list[str],
) -> pd.DataFrame:
    if flag not in df.columns or amount not in df.columns:
        return pd.DataFrame()
    out = (
        df.groupby(group_cols, as_index=False)
        .agg(
            rows=("time", "size"),
            violations=(flag, "sum"),
            max_amount=(amount, "max"),
            mean_amount=(amount, "mean"),
        )
        .sort_values(group_cols, kind="stable")
        .reset_index(drop=True)
    )
    out["violation_rate"] = out["violations"] / out["rows"]
    out.insert(0, "check", flag)
    return out


def build_audit(df: pd.DataFrame, *, tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    required = [
        "time",
        "source",
        "charge_kw",
        "discharge_kw",
        "grid_import_kw",
        "grid_export_kw",
        "realized_grid_import_kw",
        "realized_grid_export_kw",
        "curtail_kw",
        "actual_net_load_kw",
        "soc_prev_kwh",
        "soc_kwh",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Raw parquet missing required columns: {missing}")

    has_split = "actual_load_kw" in out.columns and "actual_pv_kw" in out.columns
    if has_split:
        base_grid_kw = out["actual_load_kw"].astype(float) - out["actual_pv_kw"].astype(float)
        available_pv_kw = out["actual_pv_kw"].astype(float).clip(lower=0.0)
        import_bound_kw = MAX_POWER_KW + out["actual_load_kw"].astype(float).clip(lower=0.0)
        export_bound_kw = MAX_POWER_KW * EFF_D + available_pv_kw
    else:
        base_grid_kw = out["actual_net_load_kw"].astype(float)
        available_pv_kw = (-out["actual_net_load_kw"].astype(float)).clip(lower=0.0)
        import_bound_kw = MAX_POWER_KW + out["actual_net_load_kw"].astype(float).clip(lower=0.0)
        export_bound_kw = MAX_POWER_KW * EFF_D + available_pv_kw

    out["planned_grid_balance_residual_kw"] = (
        out["grid_import_kw"].astype(float)
        - out["grid_export_kw"].astype(float)
        - (
            base_grid_kw
            + out["charge_kw"].astype(float)
            - out["discharge_kw"].astype(float) * EFF_D
            + out["curtail_kw"].astype(float)
        )
    ).abs()
    out["realized_grid_balance_residual_kw"] = (
        out["realized_grid_import_kw"].astype(float)
        - out["realized_grid_export_kw"].astype(float)
        - (
            base_grid_kw
            + out["charge_kw"].astype(float)
            - out["discharge_kw"].astype(float) * EFF_D
            + out["curtail_kw"].astype(float)
        )
    ).abs()
    out["soc_transition_residual_kwh"] = (
        out["soc_kwh"].astype(float)
        - (
            out["soc_prev_kwh"].astype(float)
            + (out["charge_kw"].astype(float) * EFF_C - out["discharge_kw"].astype(float)) * INTERVAL_H
        )
    ).abs()

    out["simultaneous_charge_discharge_kw"] = np.minimum(out["charge_kw"].astype(float), out["discharge_kw"].astype(float))
    out["simultaneous_grid_import_export_kw"] = np.minimum(
        out["grid_import_kw"].astype(float),
        out["grid_export_kw"].astype(float),
    )
    out["simultaneous_realized_import_export_kw"] = np.minimum(
        out["realized_grid_import_kw"].astype(float),
        out["realized_grid_export_kw"].astype(float),
    )
    out["curtail_over_available_pv_kw"] = (out["curtail_kw"].astype(float) - available_pv_kw).clip(lower=0.0)
    out["charge_over_max_kw"] = _violation_amount(out["charge_kw"], MAX_POWER_KW)
    out["discharge_over_max_kw"] = _violation_amount(out["discharge_kw"], MAX_POWER_KW)
    out["grid_import_over_bound_kw"] = (out["grid_import_kw"].astype(float) - import_bound_kw).clip(lower=0.0)
    out["grid_export_over_bound_kw"] = (out["grid_export_kw"].astype(float) - export_bound_kw).clip(lower=0.0)
    out["soc_below_zero_kwh"] = (-out["soc_kwh"].astype(float)).clip(lower=0.0)
    out["soc_above_capacity_kwh"] = (out["soc_kwh"].astype(float) - CAPACITY_KWH).clip(lower=0.0)

    checks = {
        "planned_grid_balance_residual": "planned_grid_balance_residual_kw",
        "realized_grid_balance_residual": "realized_grid_balance_residual_kw",
        "soc_transition_residual": "soc_transition_residual_kwh",
        "simultaneous_charge_discharge": "simultaneous_charge_discharge_kw",
        "simultaneous_grid_import_export": "simultaneous_grid_import_export_kw",
        "simultaneous_realized_import_export": "simultaneous_realized_import_export_kw",
        "curtail_over_available_pv": "curtail_over_available_pv_kw",
        "charge_over_max": "charge_over_max_kw",
        "discharge_over_max": "discharge_over_max_kw",
        "grid_import_over_bound": "grid_import_over_bound_kw",
        "grid_export_over_bound": "grid_export_over_bound_kw",
        "soc_below_zero": "soc_below_zero_kwh",
        "soc_above_capacity": "soc_above_capacity_kwh",
    }

    for check, amount in checks.items():
        out[check] = out[amount] > tolerance

    summaries = [
        _flag_summary(out, flag=check, amount=amount, group_cols=["source"])
        for check, amount in checks.items()
    ]
    summary = pd.concat([s for s in summaries if not s.empty], ignore_index=True)
    summary = summary.sort_values(["violations", "max_amount", "check", "source"], ascending=[False, False, True, True])

    top_rows: list[pd.DataFrame] = []
    cols = [
        "check",
        "amount",
        "time",
        "source",
        "charge_kw",
        "discharge_kw",
        "grid_import_kw",
        "grid_export_kw",
        "realized_grid_import_kw",
        "realized_grid_export_kw",
        "curtail_kw",
        "actual_net_load_kw",
        "soc_prev_kwh",
        "soc_kwh",
        "actual_general_price_mwh",
        "actual_feed_in_price_mwh",
    ]
    if has_split:
        cols.insert(cols.index("actual_net_load_kw"), "actual_load_kw")
        cols.insert(cols.index("actual_net_load_kw"), "actual_pv_kw")
    for check, amount in checks.items():
        bad = out[out[check]].copy()
        if bad.empty:
            continue
        bad["check"] = check
        bad["amount"] = bad[amount]
        top_rows.append(bad.sort_values("amount", ascending=False).head(20)[cols])
    examples = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame(columns=cols)
    return summary, examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Rolling raw parquet path or filename under eval/results")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    raw_path = _resolve_path(args.raw)
    df = _load_raw(raw_path)
    summary, examples = build_audit(df, tolerance=float(args.tolerance))

    summary_path = RESULTS_DIR / f"{args.output_prefix}_physical_feasibility_summary.csv"
    examples_path = RESULTS_DIR / f"{args.output_prefix}_physical_feasibility_examples.csv"
    summary.to_csv(summary_path, index=False)
    examples.to_csv(examples_path, index=False)

    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {examples_path}")
    if summary.empty:
        print("No checks produced summary rows.")
    else:
        print(summary.to_string(index=False))
    if not examples.empty:
        print("\nTop examples:")
        print(examples.head(40).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
