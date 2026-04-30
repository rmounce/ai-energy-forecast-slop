#!/usr/bin/env python3
"""
Summarize state-transition label datasets by horizon.

This is the lightweight readout companion to build_state_transition_label_dataset.py.
It focuses on the control-relevant question: compared with the Hybrid target path,
does the oracle or comparator carry more/less SoC, churn more/less, import/export
more/less, and earn better/worse prefix PnL over the 30-60 minute prefix?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


KEY_DELTAS = [
    "soc_delta_kwh",
    "throughput_kwh",
    "charge_kwh",
    "discharge_kwh",
    "import_kwh",
    "export_kwh",
    "step_pnl",
]


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find label dataset: {path_arg}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _safe_quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q)) if len(series) else float("nan")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    for horizon, sub in df.groupby("horizon_steps", sort=True):
        row: dict[str, float | int | str | bool] = {
            "horizon_steps": int(horizon),
            "horizon_minutes": int(horizon) * 5,
            "n_rows": int(len(sub)),
            "target_source": str(sub["target_source"].iloc[0]) if "target_source" in sub else "",
            "comparator_source": str(sub["comparator_source"].iloc[0]) if "comparator_source" in sub else "",
            "all_fit_lt_300_negload": bool(sub["bucket_fit_lt_300_negload"].all())
            if "bucket_fit_lt_300_negload" in sub
            else False,
            "mean_actual_feed_in_price_mwh": _safe_mean(sub["actual_feed_in_price_mwh"]),
            "mean_actual_net_load_kw": _safe_mean(sub["actual_net_load_kw"]),
            "mean_future_feed_in_price_mwh": _safe_mean(sub["future_mean_feed_in_price_mwh"]),
            "mean_future_net_load_kw": _safe_mean(sub["future_mean_net_load_kw"]),
        }

        for prefix in ["oracle_minus_target", "comparator_minus_target", "oracle_minus_comparator"]:
            for metric in KEY_DELTAS:
                col = f"{prefix}_{metric}"
                if col not in sub:
                    continue
                row[f"mean_{col}"] = _safe_mean(sub[col])
                row[f"median_{col}"] = _safe_quantile(sub[col], 0.5)
                row[f"p25_{col}"] = _safe_quantile(sub[col], 0.25)
                row[f"p75_{col}"] = _safe_quantile(sub[col], 0.75)

        for prefix in ["oracle_minus_target", "comparator_minus_target", "oracle_minus_comparator"]:
            throughput = f"{prefix}_throughput_kwh"
            soc_delta = f"{prefix}_soc_delta_kwh"
            step_pnl = f"{prefix}_step_pnl"
            export = f"{prefix}_export_kwh"
            if throughput in sub:
                row[f"p_{prefix}_less_churn"] = float((sub[throughput] < -1e-9).mean())
                row[f"p_{prefix}_more_churn"] = float((sub[throughput] > 1e-9).mean())
            if soc_delta in sub:
                row[f"p_{prefix}_lower_soc_delta"] = float((sub[soc_delta] < -1e-9).mean())
                row[f"p_{prefix}_higher_soc_delta"] = float((sub[soc_delta] > 1e-9).mean())
            if step_pnl in sub:
                row[f"p_{prefix}_better_prefix_pnl"] = float((sub[step_pnl] > 1e-9).mean())
                row[f"p_{prefix}_worse_prefix_pnl"] = float((sub[step_pnl] < -1e-9).mean())
            if export in sub:
                row[f"p_{prefix}_less_export"] = float((sub[export] < -1e-9).mean())
                row[f"p_{prefix}_more_export"] = float((sub[export] > 1e-9).mean())

        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="State-transition label CSV/parquet path or filename")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    labels_path = _resolve_path(args.labels)
    df = _read_table(labels_path)
    summary = summarize(df)
    out_csv = RESULTS_DIR / f"{args.output_prefix}_state_transition_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
