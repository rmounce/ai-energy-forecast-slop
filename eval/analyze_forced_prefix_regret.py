#!/usr/bin/env python3
"""
Summarize forced-prefix regret datasets by simple tariffed-state buckets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    raise FileNotFoundError(f"Could not find prefix-regret dataset: {path_arg}")


def _bucket_rows(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("all", pd.Series(True, index=df.index)),
        ("fit_lt_300", df["actual_feed_in_price_mwh"] < 300.0),
        ("fit_gte_300", df["actual_feed_in_price_mwh"] >= 300.0),
        ("fit_lt_300_negload", (df["actual_feed_in_price_mwh"] < 300.0) & (df["actual_net_load_kw"] < 0.0)),
        ("fit_lt_300_posload", (df["actual_feed_in_price_mwh"] < 300.0) & (df["actual_net_load_kw"] >= 0.0)),
        ("fit_gte_300_negload", (df["actual_feed_in_price_mwh"] >= 300.0) & (df["actual_net_load_kw"] < 0.0)),
        ("fit_gte_300_posload", (df["actual_feed_in_price_mwh"] >= 300.0) & (df["actual_net_load_kw"] >= 0.0)),
    ]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    out = []
    for name, mask in _bucket_rows(df):
        sub = df[mask].copy()
        if sub.empty:
            continue
        out.append(
            {
                "bucket": name,
                "n_rows": int(len(sub)),
                "prefix_steps_requested": int(sub["prefix_steps_requested"].iloc[0]),
                "mean_actual_feed_in_price_mwh": float(sub["actual_feed_in_price_mwh"].mean()),
                "mean_actual_net_load_kw": float(sub["actual_net_load_kw"].mean()),
                "mean_a_forced_prefix_regret": float(sub["a_forced_prefix_regret"].mean()),
                "mean_b_forced_prefix_regret": float(sub["b_forced_prefix_regret"].mean()),
                "mean_a_minus_b_forced_prefix_regret": float(sub["a_minus_b_forced_prefix_regret"].mean()),
                "p_a_better": float((sub["a_minus_b_forced_prefix_regret"] < -1e-9).mean()),
                "p_b_better": float((sub["a_minus_b_forced_prefix_regret"] > 1e-9).mean()),
                "mean_a_prefix_export_kw_sum": float(sub["a_prefix_export_kw_sum"].mean()),
                "mean_b_prefix_export_kw_sum": float(sub["b_prefix_export_kw_sum"].mean()),
                "mean_a_prefix_discharge_kw_sum": float(sub["a_prefix_discharge_kw_sum"].mean()),
                "mean_b_prefix_discharge_kw_sum": float(sub["b_prefix_discharge_kw_sum"].mean()),
            }
        )
    return pd.DataFrame(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV path or filename under eval/results")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    df = pd.read_csv(_resolve_path(args.csv))
    summary = summarize(df)
    out_csv = RESULTS_DIR / f"{args.output_prefix}_forced_prefix_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
