#!/usr/bin/env python3
"""
Compare tactical forward-curve persistence between two rolling-eval sources.

This is a lightweight diagnostic for the question:
"Does Hybrid get step 0 roughly right but mean-revert too quickly over the next hour?"

It works from raw rolling-eval parquet outputs that already contain step-0 and horizon-summary
columns such as:
- forecast_step0_mwh
- forecast_mean_next_1h_mwh / 4h / 14h
- forecast_feed_in_step0_mwh
- forecast_feed_in_mean_next_1h_mwh / 4h / 14h
- charge_kw / discharge_kw / grid_export_kw
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find parquet: {path_arg}")


def _rename_with_source(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    key_cols = {"time"}
    renamed = {}
    for col in df.columns:
        if col not in key_cols:
            renamed[col] = f"{prefix}_{col}"
    return df.rename(columns=renamed)


def build_comparison(
    raw_df: pd.DataFrame,
    *,
    source_a: str,
    source_b: str,
    min_feed_in: float = 0.0,
    require_a_export_gt_b: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = raw_df.copy()
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)

    cols = [
        "time",
        "source",
        "actual_feed_in_price_mwh",
        "actual_general_price_mwh",
        "actual_net_load_kw",
        "forecast_step0_mwh",
        "forecast_mean_next_1h_mwh",
        "forecast_mean_next_4h_mwh",
        "forecast_mean_next_14h_mwh",
        "forecast_feed_in_step0_mwh",
        "forecast_feed_in_mean_next_1h_mwh",
        "forecast_feed_in_mean_next_4h_mwh",
        "forecast_feed_in_mean_next_14h_mwh",
        "charge_kw",
        "discharge_kw",
        "grid_export_kw",
        "step_pnl",
    ]
    missing = [c for c in cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    a = raw_df[raw_df["source"] == source_a][cols].copy().drop(columns=["source"])
    b = raw_df[raw_df["source"] == source_b][cols].copy().drop(columns=["source"])
    if a.empty or b.empty:
        raise ValueError("Requested sources not found in raw parquet")

    merged = _rename_with_source(a, "a").merge(
        _rename_with_source(b, "b"),
        on="time",
        how="inner",
        validate="one_to_one",
    )

    merged["a_decay_1h_mwh"] = merged["a_forecast_step0_mwh"] - merged["a_forecast_mean_next_1h_mwh"]
    merged["b_decay_1h_mwh"] = merged["b_forecast_step0_mwh"] - merged["b_forecast_mean_next_1h_mwh"]
    merged["a_feed_in_decay_1h_mwh"] = merged["a_forecast_feed_in_step0_mwh"] - merged["a_forecast_feed_in_mean_next_1h_mwh"]
    merged["b_feed_in_decay_1h_mwh"] = merged["b_forecast_feed_in_step0_mwh"] - merged["b_forecast_feed_in_mean_next_1h_mwh"]
    merged["a_decay_4h_mwh"] = merged["a_forecast_step0_mwh"] - merged["a_forecast_mean_next_4h_mwh"]
    merged["b_decay_4h_mwh"] = merged["b_forecast_step0_mwh"] - merged["b_forecast_mean_next_4h_mwh"]
    merged["a_feed_in_decay_4h_mwh"] = merged["a_forecast_feed_in_step0_mwh"] - merged["a_forecast_feed_in_mean_next_4h_mwh"]
    merged["b_feed_in_decay_4h_mwh"] = merged["b_forecast_feed_in_step0_mwh"] - merged["b_forecast_feed_in_mean_next_4h_mwh"]
    merged["a_minus_b_step_pnl"] = merged["a_step_pnl"] - merged["b_step_pnl"]
    merged["a_minus_b_export_kw"] = merged["a_grid_export_kw"] - merged["b_grid_export_kw"]
    merged["a_minus_b_discharge_kw"] = merged["a_discharge_kw"] - merged["b_discharge_kw"]
    merged["a_minus_b_feed_in_mean_next_1h_mwh"] = (
        merged["a_forecast_feed_in_mean_next_1h_mwh"] - merged["b_forecast_feed_in_mean_next_1h_mwh"]
    )
    merged["a_minus_b_feed_in_mean_next_4h_mwh"] = (
        merged["a_forecast_feed_in_mean_next_4h_mwh"] - merged["b_forecast_feed_in_mean_next_4h_mwh"]
    )

    filtered = merged[merged["a_actual_feed_in_price_mwh"] >= min_feed_in].copy()
    if require_a_export_gt_b:
        filtered = filtered[filtered["a_minus_b_export_kw"] > 0.0].copy()

    summary = pd.DataFrame(
        [
            {
                "n_rows": len(filtered),
                "mean_actual_feed_in_price_mwh": filtered["a_actual_feed_in_price_mwh"].mean(),
                "mean_a_export_kw": filtered["a_grid_export_kw"].mean(),
                "mean_b_export_kw": filtered["b_grid_export_kw"].mean(),
                "mean_a_discharge_kw": filtered["a_discharge_kw"].mean(),
                "mean_b_discharge_kw": filtered["b_discharge_kw"].mean(),
                "mean_a_feed_in_step0_mwh": filtered["a_forecast_feed_in_step0_mwh"].mean(),
                "mean_b_feed_in_step0_mwh": filtered["b_forecast_feed_in_step0_mwh"].mean(),
                "mean_a_feed_in_1h_mwh": filtered["a_forecast_feed_in_mean_next_1h_mwh"].mean(),
                "mean_b_feed_in_1h_mwh": filtered["b_forecast_feed_in_mean_next_1h_mwh"].mean(),
                "mean_a_feed_in_4h_mwh": filtered["a_forecast_feed_in_mean_next_4h_mwh"].mean(),
                "mean_b_feed_in_4h_mwh": filtered["b_forecast_feed_in_mean_next_4h_mwh"].mean(),
                "mean_a_feed_in_decay_1h_mwh": filtered["a_feed_in_decay_1h_mwh"].mean(),
                "mean_b_feed_in_decay_1h_mwh": filtered["b_feed_in_decay_1h_mwh"].mean(),
                "mean_a_feed_in_decay_4h_mwh": filtered["a_feed_in_decay_4h_mwh"].mean(),
                "mean_b_feed_in_decay_4h_mwh": filtered["b_feed_in_decay_4h_mwh"].mean(),
                "mean_a_minus_b_feed_in_mean_next_1h_mwh": filtered["a_minus_b_feed_in_mean_next_1h_mwh"].mean(),
                "mean_a_minus_b_feed_in_mean_next_4h_mwh": filtered["a_minus_b_feed_in_mean_next_4h_mwh"].mean(),
                "mean_a_minus_b_export_kw": filtered["a_minus_b_export_kw"].mean(),
                "mean_a_minus_b_step_pnl": filtered["a_minus_b_step_pnl"].mean(),
            }
        ]
    )

    interesting_cols = [
        "time",
        "a_actual_feed_in_price_mwh",
        "a_actual_net_load_kw",
        "a_grid_export_kw",
        "b_grid_export_kw",
        "a_discharge_kw",
        "b_discharge_kw",
        "a_forecast_feed_in_step0_mwh",
        "b_forecast_feed_in_step0_mwh",
        "a_forecast_feed_in_mean_next_1h_mwh",
        "b_forecast_feed_in_mean_next_1h_mwh",
        "a_forecast_feed_in_mean_next_4h_mwh",
        "b_forecast_feed_in_mean_next_4h_mwh",
        "a_feed_in_decay_1h_mwh",
        "b_feed_in_decay_1h_mwh",
        "a_feed_in_decay_4h_mwh",
        "b_feed_in_decay_4h_mwh",
        "a_minus_b_export_kw",
        "a_minus_b_step_pnl",
    ]
    top = filtered.sort_values(
        ["a_minus_b_step_pnl", "a_actual_feed_in_price_mwh"],
        ascending=[False, False],
        kind="stable",
    )[interesting_cols]

    return summary, top


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw parquet path or filename under eval/results")
    parser.add_argument("--source-a", default="amber_tactical_hybrid_strategic")
    parser.add_argument("--source-b", default="model_a_hybrid")
    parser.add_argument("--min-feed-in", type=float, default=300.0)
    parser.add_argument("--require-a-export-gt-b", action="store_true")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    raw_path = _resolve_path(args.raw)
    raw_df = pd.read_parquet(raw_path)
    summary, top = build_comparison(
        raw_df,
        source_a=args.source_a,
        source_b=args.source_b,
        min_feed_in=float(args.min_feed_in),
        require_a_export_gt_b=bool(args.require_a_export_gt_b),
    )

    summary_csv = RESULTS_DIR / f"{args.output_prefix}_curve_persistence_summary.csv"
    top_csv = RESULTS_DIR / f"{args.output_prefix}_curve_persistence_top.csv"
    summary.to_csv(summary_csv, index=False)
    top.head(max(0, int(args.top_k))).to_csv(top_csv, index=False)

    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {top_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
