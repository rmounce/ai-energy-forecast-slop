#!/usr/bin/env python3
"""
Attribute forced-prefix regret differences to short-horizon path behavior.

This companion to build_forced_prefix_regret_dataset.py joins an existing forced-prefix
regret dataset back to the rolling MPC raw closed-loop paths. It avoids rerunning the
expensive oracle LP solves while answering the next tactical question: when Amber's
multi-step prefix has lower regret, is the prefix storing more, exporting more, importing
less, or simply ending at a different SoC?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
DT_HOURS = 5.0 / 60.0


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find path: {path_arg}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _bucket_rows(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("all", pd.Series(True, index=df.index)),
        ("fit_lt_300", df["actual_feed_in_price_mwh"] < 300.0),
        ("fit_gte_300", df["actual_feed_in_price_mwh"] >= 300.0),
        (
            "fit_lt_300_negload",
            (df["actual_feed_in_price_mwh"] < 300.0) & (df["actual_net_load_kw"] < 0.0),
        ),
        (
            "fit_lt_300_nonnegload",
            (df["actual_feed_in_price_mwh"] < 300.0) & (df["actual_net_load_kw"] >= 0.0),
        ),
        (
            "fit_gte_300_negload",
            (df["actual_feed_in_price_mwh"] >= 300.0) & (df["actual_net_load_kw"] < 0.0),
        ),
        (
            "fit_gte_300_nonnegload",
            (df["actual_feed_in_price_mwh"] >= 300.0) & (df["actual_net_load_kw"] >= 0.0),
        ),
    ]


def _kw_col(df: pd.DataFrame, realized_col: str, planned_col: str) -> str:
    if realized_col in df.columns:
        return realized_col
    return planned_col


def _prefix_metrics_for_source(
    src_df: pd.DataFrame,
    *,
    start_time: pd.Timestamp,
    steps: int,
    import_col: str,
    export_col: str,
) -> dict[str, float]:
    prefix = src_df[src_df["time"] >= start_time].head(steps)
    if prefix.empty:
        return {
            "prefix_rows": 0.0,
            "prefix_charge_kwh": np.nan,
            "prefix_discharge_kwh": np.nan,
            "prefix_import_kwh": np.nan,
            "prefix_export_kwh": np.nan,
            "prefix_curtail_kwh": np.nan,
            "prefix_import_cost": np.nan,
            "prefix_export_revenue": np.nan,
            "prefix_step_pnl": np.nan,
            "prefix_soc_delta_kwh": np.nan,
            "prefix_end_soc_kwh": np.nan,
        }

    charge_kwh = float(prefix["charge_kw"].sum() * DT_HOURS)
    discharge_kwh = float(prefix["discharge_kw"].sum() * DT_HOURS)
    curtail_kwh = float(prefix["curtail_kw"].sum() * DT_HOURS) if "curtail_kw" in prefix.columns else 0.0
    import_kwh = float(prefix[import_col].sum() * DT_HOURS)
    export_kwh = float(prefix[export_col].sum() * DT_HOURS)
    import_cost = float((prefix[import_col] * prefix["actual_general_price_mwh"]).sum() * DT_HOURS / 1000.0)
    export_revenue = float((prefix[export_col] * prefix["actual_feed_in_price_mwh"]).sum() * DT_HOURS / 1000.0)
    step_pnl = float(prefix["step_pnl"].sum()) if "step_pnl" in prefix.columns else export_revenue - import_cost
    start_soc = float(prefix["soc_prev_kwh"].iloc[0])
    end_soc = float(prefix["soc_kwh"].iloc[-1])
    return {
        "prefix_rows": float(len(prefix)),
        "prefix_charge_kwh": charge_kwh,
        "prefix_discharge_kwh": discharge_kwh,
        "prefix_import_kwh": import_kwh,
        "prefix_export_kwh": export_kwh,
        "prefix_curtail_kwh": curtail_kwh,
        "prefix_import_cost": import_cost,
        "prefix_export_revenue": export_revenue,
        "prefix_step_pnl": step_pnl,
        "prefix_soc_delta_kwh": end_soc - start_soc,
        "prefix_end_soc_kwh": end_soc,
    }


def build_attribution(
    raw_df: pd.DataFrame,
    prefix_df: pd.DataFrame,
    *,
    source_a: str | None = None,
    source_b: str | None = None,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    prefix_df = prefix_df.copy()
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    prefix_df["time"] = pd.to_datetime(prefix_df["time"], utc=True)

    if source_a is None:
        source_a = str(prefix_df["a_source"].dropna().iloc[0])
    if source_b is None:
        source_b = str(prefix_df["b_source"].dropna().iloc[0])

    a_df = raw_df[raw_df["source"] == source_a].sort_values("time", kind="stable").reset_index(drop=True)
    b_df = raw_df[raw_df["source"] == source_b].sort_values("time", kind="stable").reset_index(drop=True)
    if a_df.empty:
        raise ValueError(f"No rows found in raw data for source A: {source_a}")
    if b_df.empty:
        raise ValueError(f"No rows found in raw data for source B: {source_b}")

    import_col = _kw_col(raw_df, "realized_grid_import_kw", "grid_import_kw")
    export_col = _kw_col(raw_df, "realized_grid_export_kw", "grid_export_kw")

    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    for row in prefix_df.itertuples(index=False):
        start_time = getattr(row, "time")
        steps = int(getattr(row, "prefix_steps_actual"))
        a = _prefix_metrics_for_source(a_df, start_time=start_time, steps=steps, import_col=import_col, export_col=export_col)
        b = _prefix_metrics_for_source(b_df, start_time=start_time, steps=steps, import_col=import_col, export_col=export_col)
        out: dict[str, float | int | str | pd.Timestamp] = {
            "time": start_time,
            "prefix_steps_requested": int(getattr(row, "prefix_steps_requested")),
            "prefix_steps_actual": steps,
            "actual_feed_in_price_mwh": float(getattr(row, "actual_feed_in_price_mwh")),
            "actual_net_load_kw": float(getattr(row, "actual_net_load_kw")),
            "a_source": source_a,
            "b_source": source_b,
            "a_minus_b_forced_prefix_regret": float(getattr(row, "a_minus_b_forced_prefix_regret")),
            "a_forced_prefix_regret": float(getattr(row, "a_forced_prefix_regret")),
            "b_forced_prefix_regret": float(getattr(row, "b_forced_prefix_regret")),
        }
        for key in a:
            out[f"a_{key}"] = a[key]
            out[f"b_{key}"] = b[key]
            out[f"a_minus_b_{key}"] = float(a[key] - b[key])
        rows.append(out)

    return pd.DataFrame(rows)


def summarize(attr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "a_minus_b_forced_prefix_regret",
        "a_minus_b_prefix_charge_kwh",
        "a_minus_b_prefix_discharge_kwh",
        "a_minus_b_prefix_import_kwh",
        "a_minus_b_prefix_export_kwh",
        "a_minus_b_prefix_curtail_kwh",
        "a_minus_b_prefix_import_cost",
        "a_minus_b_prefix_export_revenue",
        "a_minus_b_prefix_step_pnl",
        "a_minus_b_prefix_soc_delta_kwh",
        "a_prefix_charge_kwh",
        "b_prefix_charge_kwh",
        "a_prefix_discharge_kwh",
        "b_prefix_discharge_kwh",
        "a_prefix_import_kwh",
        "b_prefix_import_kwh",
        "a_prefix_export_kwh",
        "b_prefix_export_kwh",
        "a_prefix_curtail_kwh",
        "b_prefix_curtail_kwh",
        "a_prefix_soc_delta_kwh",
        "b_prefix_soc_delta_kwh",
        "a_prefix_step_pnl",
        "b_prefix_step_pnl",
    ]
    for bucket, mask in _bucket_rows(attr_df):
        sub = attr_df[mask]
        if sub.empty:
            continue
        row: dict[str, float | int | str] = {
            "bucket": bucket,
            "n_rows": int(len(sub)),
            "prefix_steps_requested": int(sub["prefix_steps_requested"].iloc[0]),
            "mean_actual_feed_in_price_mwh": float(sub["actual_feed_in_price_mwh"].mean()),
            "mean_actual_net_load_kw": float(sub["actual_net_load_kw"].mean()),
            "p_a_lower_regret": float((sub["a_minus_b_forced_prefix_regret"] < -1e-9).mean()),
            "p_b_lower_regret": float((sub["a_minus_b_forced_prefix_regret"] > 1e-9).mean()),
        }
        for col in metric_cols:
            row[f"mean_{col}"] = float(sub[col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Rolling raw parquet path or filename under eval/results")
    parser.add_argument("--prefix", required=True, help="Forced-prefix regret CSV/parquet path or filename")
    parser.add_argument("--source-a", default=None)
    parser.add_argument("--source-b", default=None)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    raw_df = pd.read_parquet(_resolve_path(args.raw))
    prefix_df = _read_table(_resolve_path(args.prefix))
    attr = build_attribution(raw_df, prefix_df, source_a=args.source_a, source_b=args.source_b)
    summary = summarize(attr)

    detail_csv = RESULTS_DIR / f"{args.output_prefix}_forced_prefix_path_attribution.csv"
    summary_csv = RESULTS_DIR / f"{args.output_prefix}_forced_prefix_path_attribution_summary.csv"
    attr.to_csv(detail_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"[done] wrote {detail_csv}")
    print(f"[done] wrote {summary_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
