#!/usr/bin/env python3
"""
Analyze a tariffed rolling MPC raw parquet and summarize source deltas.

This is intended for short-window residual diagnostics after running
`rolling_mpc_eval.py --economic-mode netload_tariffed`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
INTERVAL_H = 5 / 60
EFF_C = 0.95
DEG_PER_KWH = 0.05


def _resolve_path(raw_arg: str) -> Path:
    p = Path(raw_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / raw_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find parquet: {raw_arg}")


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(["source", "time"], kind="stable").reset_index(drop=True)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["time"].dt.date
    out["charge_energy_kwh"] = out["charge_kw"] * INTERVAL_H
    out["discharge_energy_kwh"] = out["discharge_kw"] * INTERVAL_H
    out["grid_import_energy_kwh"] = out["realized_grid_import_kw"] * INTERVAL_H
    out["grid_export_energy_kwh"] = out["realized_grid_export_kw"] * INTERVAL_H
    out["import_cost"] = out["grid_import_energy_kwh"] * out["actual_general_price_mwh"] / 1000.0
    out["export_revenue"] = out["grid_export_energy_kwh"] * out["actual_feed_in_price_mwh"] / 1000.0
    out["degradation_cost"] = (out["charge_kw"] * EFF_C + out["discharge_kw"]) * DEG_PER_KWH * INTERVAL_H
    out["charge_price_weight"] = out["actual_general_price_mwh"] * out["charge_energy_kwh"]
    out["discharge_price_weight"] = out["actual_feed_in_price_mwh"] * out["discharge_energy_kwh"]
    return out


def build_daily_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["source", "date"], as_index=False)
        .agg(
            steps=("time", "size"),
            total_pnl=("step_pnl", "sum"),
            charge_energy_kwh=("charge_energy_kwh", "sum"),
            discharge_energy_kwh=("discharge_energy_kwh", "sum"),
            grid_import_energy_kwh=("grid_import_energy_kwh", "sum"),
            grid_export_energy_kwh=("grid_export_energy_kwh", "sum"),
            import_cost=("import_cost", "sum"),
            export_revenue=("export_revenue", "sum"),
            degradation_cost=("degradation_cost", "sum"),
            soc_open_kwh=("soc_prev_kwh", "first"),
            soc_close_kwh=("soc_kwh", "last"),
            charge_price_weight=("charge_price_weight", "sum"),
            discharge_price_weight=("discharge_price_weight", "sum"),
        )
        .sort_values(["date", "source"], kind="stable")
        .reset_index(drop=True)
    )
    out["avg_charge_price_mwh"] = out["charge_price_weight"] / out["charge_energy_kwh"]
    out["avg_discharge_price_mwh"] = out["discharge_price_weight"] / out["discharge_energy_kwh"]
    out["soc_delta_kwh"] = out["soc_close_kwh"] - out["soc_open_kwh"]
    return out.drop(columns=["charge_price_weight", "discharge_price_weight"])


def build_delta_vs_baseline(daily_df: pd.DataFrame, baseline_source: str) -> pd.DataFrame:
    baseline = (
        daily_df[daily_df["source"] == baseline_source]
        .drop(columns=["source"])
        .rename(columns=lambda c: c if c == "date" else f"{c}_baseline")
    )
    merged = daily_df.merge(baseline, on="date", how="left")
    numeric_cols = [
        "total_pnl",
        "charge_energy_kwh",
        "discharge_energy_kwh",
        "grid_import_energy_kwh",
        "grid_export_energy_kwh",
        "import_cost",
        "export_revenue",
        "degradation_cost",
        "soc_close_kwh",
        "soc_delta_kwh",
        "avg_charge_price_mwh",
        "avg_discharge_price_mwh",
    ]
    for col in numeric_cols:
        merged[f"{col}_delta_vs_{baseline_source}"] = merged[col] - merged[f"{col}_baseline"]
    return merged


def build_overall_source_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("source", as_index=False)
        .agg(
            total_pnl=("step_pnl", "sum"),
            charge_energy_kwh=("charge_energy_kwh", "sum"),
            discharge_energy_kwh=("discharge_energy_kwh", "sum"),
            grid_import_energy_kwh=("grid_import_energy_kwh", "sum"),
            grid_export_energy_kwh=("grid_export_energy_kwh", "sum"),
            import_cost=("import_cost", "sum"),
            export_revenue=("export_revenue", "sum"),
            degradation_cost=("degradation_cost", "sum"),
            soc_open_kwh=("soc_prev_kwh", "first"),
            soc_close_kwh=("soc_kwh", "last"),
        )
        .sort_values("source", kind="stable")
        .reset_index(drop=True)
    )
    out["soc_delta_kwh"] = out["soc_close_kwh"] - out["soc_open_kwh"]
    return out


def build_same_target_daily_delta(
    daily_df: pd.DataFrame,
    *,
    amber_tactical_source: str,
    hybrid_tactical_source: str,
) -> pd.DataFrame:
    amber = (
        daily_df[daily_df["source"] == amber_tactical_source]
        .drop(columns=["source"])
        .rename(columns=lambda c: c if c == "date" else f"{c}_amber_tactical")
    )
    hybrid = (
        daily_df[daily_df["source"] == hybrid_tactical_source]
        .drop(columns=["source"])
        .rename(columns=lambda c: c if c == "date" else f"{c}_hybrid_tactical")
    )
    merged = amber.merge(hybrid, on="date", how="inner")
    metric_roots = [
        "total_pnl",
        "charge_energy_kwh",
        "discharge_energy_kwh",
        "grid_import_energy_kwh",
        "grid_export_energy_kwh",
        "import_cost",
        "export_revenue",
        "degradation_cost",
        "soc_close_kwh",
        "soc_delta_kwh",
        "avg_charge_price_mwh",
        "avg_discharge_price_mwh",
    ]
    for root in metric_roots:
        merged[f"{root}_delta_amber_minus_hybrid_tactical"] = (
            merged[f"{root}_amber_tactical"] - merged[f"{root}_hybrid_tactical"]
        )
    return merged.sort_values("date", kind="stable").reset_index(drop=True)


def build_missed_export_report(
    df: pd.DataFrame,
    *,
    amber_source: str,
    hybrid_source: str,
    top_n: int,
) -> pd.DataFrame:
    amber = (
        df[df["source"] == amber_source]
        .copy()
        .rename(columns=lambda c: c if c == "time" else f"{c}_amber")
    )
    hybrid = (
        df[df["source"] == hybrid_source]
        .copy()
        .rename(columns=lambda c: c if c == "time" else f"{c}_hybrid")
    )
    merged = amber.merge(hybrid, on="time", how="inner")
    missed = merged[
        (merged["realized_grid_export_kw_amber"] > 0.0)
        & (merged["realized_grid_export_kw_hybrid"] <= 0.0)
    ].copy()
    if missed.empty:
        return missed
    missed["export_gap_kw"] = (
        missed["realized_grid_export_kw_amber"] - missed["realized_grid_export_kw_hybrid"]
    )
    missed["step_pnl_gap_amber_minus_hybrid"] = (
        missed["step_pnl_amber"] - missed["step_pnl_hybrid"]
    )
    preferred_cols = [
        "time",
        "actual_feed_in_price_mwh_amber",
        "actual_general_price_mwh_amber",
        "actual_net_load_kw_amber",
        "soc_prev_kwh_amber",
        "soc_prev_kwh_hybrid",
        "strategic_soc_target_kwh_amber",
        "strategic_soc_target_kwh_hybrid",
        "forecast_step0_mwh_amber",
        "forecast_step0_mwh_hybrid",
        "forecast_mean_next_1h_mwh_amber",
        "forecast_mean_next_1h_mwh_hybrid",
        "forecast_mean_next_4h_mwh_amber",
        "forecast_mean_next_4h_mwh_hybrid",
        "forecast_mean_next_14h_mwh_amber",
        "forecast_mean_next_14h_mwh_hybrid",
        "realized_grid_export_kw_amber",
        "realized_grid_export_kw_hybrid",
        "discharge_kw_amber",
        "discharge_kw_hybrid",
        "step_pnl_amber",
        "step_pnl_hybrid",
        "export_gap_kw",
        "step_pnl_gap_amber_minus_hybrid",
    ]
    existing = [c for c in preferred_cols if c in missed.columns]
    return (
        missed.nlargest(top_n, "step_pnl_gap_amber_minus_hybrid")[existing]
        .reset_index(drop=True)
    )


def top_events(df: pd.DataFrame, by: str, source: str, top_n: int) -> pd.DataFrame:
    cols = [
        "time",
        by,
        "actual_general_price_mwh",
        "actual_feed_in_price_mwh",
        "actual_net_load_kw",
        "soc_prev_kwh",
        "soc_kwh",
        "realized_grid_import_kw",
        "realized_grid_export_kw",
        "step_pnl",
    ]
    existing = [c for c in cols if c in df.columns]
    return (
        df[df["source"] == source]
        .nlargest(top_n, by)[existing]
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw parquet path or filename under eval/results")
    parser.add_argument("--baseline-source", required=True, help="Baseline source for delta reporting")
    parser.add_argument("--top-n", type=int, default=10, help="Top charge/discharge events per source")
    parser.add_argument(
        "--amber-source",
        default="amber_apf_lgbm",
        help="Amber-like source label for missed-export tactical comparisons.",
    )
    parser.add_argument(
        "--hybrid-source",
        default="model_a_hybrid",
        help="Hybrid-like source label for missed-export tactical comparisons.",
    )
    parser.add_argument(
        "--amber-tactical-source",
        default="amber_tactical_hybrid_strategic",
        help="Source label used for the same-strategic-target / Amber-tactical daily delta table.",
    )
    parser.add_argument(
        "--hybrid-tactical-source",
        default="model_a_hybrid",
        help="Source label used for the same-strategic-target / Hybrid-tactical daily delta table.",
    )
    parser.add_argument("--output-prefix", default="", help="Optional prefix for CSV outputs under eval/results")
    args = parser.parse_args()

    raw_path = _resolve_path(args.raw)
    raw = _load_raw(raw_path)
    prepped = _prep(raw)

    daily = build_daily_source_summary(prepped)
    daily_delta = build_delta_vs_baseline(daily, args.baseline_source)
    overall = build_overall_source_summary(prepped)
    same_target_daily_delta = build_same_target_daily_delta(
        daily,
        amber_tactical_source=args.amber_tactical_source,
        hybrid_tactical_source=args.hybrid_tactical_source,
    )
    missed_export = build_missed_export_report(
        prepped,
        amber_source=args.amber_source,
        hybrid_source=args.hybrid_source,
        top_n=args.top_n,
    )

    print("Rolling MPC tariffed diagnostics")
    print(f"  raw: {raw_path.relative_to(ROOT)}")
    print(f"  baseline: {args.baseline_source}")
    print("\nOverall summary")
    print(overall.to_string(index=False))
    print("\nDaily summary vs baseline")
    print(daily_delta.to_string(index=False))
    if not same_target_daily_delta.empty:
        print("\nSame strategic target, different tactical curve — daily delta")
        print(same_target_daily_delta.to_string(index=False))
    if not missed_export.empty:
        print("\nTop missed export intervals — Amber exports, Hybrid does not")
        print(missed_export.to_string(index=False))

    for source in overall["source"]:
        print(f"\nTop charge events — {source}")
        print(top_events(prepped, "charge_kw", source, args.top_n).to_string(index=False))
        print(f"\nTop discharge events — {source}")
        print(top_events(prepped, "discharge_kw", source, args.top_n).to_string(index=False))

    if args.output_prefix:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        overall_path = RESULTS_DIR / f"{args.output_prefix}_overall.csv"
        daily_path = RESULTS_DIR / f"{args.output_prefix}_daily.csv"
        same_target_path = RESULTS_DIR / f"{args.output_prefix}_same_target_daily_delta.csv"
        missed_export_path = RESULTS_DIR / f"{args.output_prefix}_missed_export.csv"
        overall.to_csv(overall_path, index=False)
        daily_delta.to_csv(daily_path, index=False)
        same_target_daily_delta.to_csv(same_target_path, index=False)
        missed_export.to_csv(missed_export_path, index=False)
        print(f"\nSaved overall -> {overall_path.relative_to(ROOT)}")
        print(f"Saved daily -> {daily_path.relative_to(ROOT)}")
        print(f"Saved same-target daily delta -> {same_target_path.relative_to(ROOT)}")
        print(f"Saved missed export -> {missed_export_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
