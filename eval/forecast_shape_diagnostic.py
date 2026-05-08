#!/usr/bin/env python3
"""Forecast shape diagnostic — Phase α-prime Step 5a.

The 2026-05-09 architectural-rethink reviewer call surfaced that the dispatch eval
averages step-by-step decisions and rewards extracted spread; it cannot detect
forecast-curve pathology like cap-induced step-functions or non-physical
oscillation in the back half. PD-direct's published forecast tail (30h+) has 41% of
steps pinned at the $300 PD7Day cap, producing square-wave artefacts that the user
detected visually. The dispatch eval did not flag this.

This script computes per-source per-horizon-bucket physical-plausibility metrics
from the published forecast logs:

  - cap_hit_rate: fraction of steps at or near a known cap value
  - step_diff_mean / step_diff_max: step-to-step absolute change ($/MWh)
  - sign_changes: direction flips in the first difference (oscillation count)
  - local_extrema: count of peaks + troughs (rough wiggle metric)
  - splice_jump_28h: absolute price jump at the PREDISPATCH→PD7Day boundary
    (~step 56). Should be small for any source that is internally coherent;
    large values indicate seam pathology
  - tail_spread: max - min over the 30h+ tail (what the strategic LP sees as
    arbitrage opportunity in the back half)

Output: a tidy CSV with one row per (source, run_creation_time, horizon_bucket).
Plus an aggregate summary across runs at the bottom (per source × horizon_bucket).

Usage:
  python eval/forecast_shape_diagnostic.py
  python eval/forecast_shape_diagnostic.py --since 2026-05-08T00:00:00Z
  python eval/forecast_shape_diagnostic.py --output eval/results/shape_diag_$(date +%Y%m%d).csv

Per the reviewer's reframe: this metric layer is a prerequisite for any
architectural decision about PD-direct's tail. The follow-up steps (hybrid-tail
probe, cap-materialisation analysis) should join their outputs to this script's
output so we can correlate shape pathology with dispatch outcomes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

LOG_FILES = {
    "tft":         ROOT / "tft_price_forecast_log.csv",
    "pd_direct":   ROOT / "pd_direct_forecast_log.csv",
    "incumbent":   ROOT / "price_forecast_log.csv",
    "tier1":       ROOT / "p5min_forecast_log.csv",
}

# Source unit conventions. TFT log writes raw $/MWh; the others write $/kWh.
# Multiplier converts each source's `prediction` column into $/MWh for comparison.
SOURCE_TO_MWH_MULT = {
    "tft":       1.0,
    "pd_direct": 1000.0,
    "incumbent": 1000.0,
    "tier1":     1000.0,
}

# Known cap values per source ($/MWh). PD-direct hard-caps PD7Day at $300; others
# don't have explicit caps. Detect cap-hits as values within ±0.5 $/MWh of the
# nominal cap.
SOURCE_CAP_MWH = {
    "pd_direct": 300.0,
}

# Horizon buckets (minutes from forecast_creation_time). Edges chosen to reflect
# the layered structure of PD-direct: <30h is debiased PREDISPATCH; 30h+ is
# PD7Day/seasonal. Splice diagnostic is anchored at the 28h boundary.
HORIZON_BUCKETS = [
    (0,    360,  "0-6h"),
    (360,  840,  "6-14h"),
    (840,  1680, "14-28h"),
    (1680, 4320, "28-72h"),
    (0,    4320, "ALL"),
]
SPLICE_BOUNDARY_MIN = 1680  # ~28h, last full PREDISPATCH coverage step


def _bucket_label(horizon_min: float) -> str:
    for lo, hi, label in HORIZON_BUCKETS:
        if label == "ALL":
            continue
        if lo <= horizon_min < hi:
            return label
    return "unbucketed"


def load_forecast_log(source: str, path: Path,
                      since: pd.Timestamp | None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(
        path,
        usecols=["forecast_creation_time", "forecast_target_time", "prediction"],
    )
    df["forecast_creation_time"] = pd.to_datetime(
        df["forecast_creation_time"], utc=True, format="ISO8601"
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="ISO8601"
    )
    if since is not None:
        df = df[df["forecast_creation_time"] >= since]
    df = df.dropna(subset=["prediction"])
    if df.empty:
        return df
    df["source"] = source
    df["prediction_mwh"] = df["prediction"] * SOURCE_TO_MWH_MULT.get(source, 1000.0)
    df["horizon_min"] = (
        (df["forecast_target_time"] - df["forecast_creation_time"]).dt.total_seconds() / 60.0
    )
    return df


def compute_run_metrics(run_df: pd.DataFrame, source: str) -> list[dict]:
    """Compute shape metrics for one forecast run, one row per horizon bucket.

    Expects `run_df` to have columns prediction_mwh, horizon_min, sorted by
    horizon_min ascending.
    """
    rows = []
    cap_value = SOURCE_CAP_MWH.get(source)

    for lo, hi, label in HORIZON_BUCKETS:
        sub = run_df[(run_df["horizon_min"] >= lo) & (run_df["horizon_min"] < hi)]
        n = len(sub)
        if n == 0:
            continue
        vals = sub["prediction_mwh"].to_numpy(dtype=np.float64)
        diffs = np.diff(vals) if n > 1 else np.array([])

        cap_hit_rate = (
            float(np.mean(np.abs(vals - cap_value) < 0.5))
            if cap_value is not None else float("nan")
        )

        if len(diffs) > 0:
            step_diff_mean = float(np.mean(np.abs(diffs)))
            step_diff_max = float(np.max(np.abs(diffs)))
        else:
            step_diff_mean = step_diff_max = float("nan")

        if len(diffs) > 1:
            # Sign changes: direction flips in the first-difference series.
            # Exclude zero-diffs to avoid spurious flips at the cap plateau.
            nonzero = diffs[np.abs(diffs) > 1e-9]
            if len(nonzero) > 1:
                sign_changes = int(np.sum(np.sign(nonzero[:-1]) * np.sign(nonzero[1:]) < 0))
            else:
                sign_changes = 0
        else:
            sign_changes = 0

        # Local extrema: a step is a local extremum if both neighbours have the
        # opposite sign of difference. Cheap approximation; close enough for a
        # shape diagnostic.
        if n >= 3:
            local_extrema = 0
            for i in range(1, n - 1):
                left = vals[i] - vals[i-1]
                right = vals[i+1] - vals[i]
                if left * right < 0:
                    local_extrema += 1
        else:
            local_extrema = 0

        rows.append({
            "source":          source,
            "horizon_bucket":  label,
            "n_steps":         n,
            "mean_mwh":        float(np.mean(vals)),
            "min_mwh":         float(np.min(vals)),
            "max_mwh":         float(np.max(vals)),
            "spread_mwh":      float(np.max(vals) - np.min(vals)),
            "std_mwh":         float(np.std(vals)),
            "cap_hit_rate":    cap_hit_rate,
            "step_diff_mean":  step_diff_mean,
            "step_diff_max":   step_diff_max,
            "sign_changes":    sign_changes,
            "local_extrema":   local_extrema,
        })

    # Splice jump: absolute change at the PREDISPATCH→tail boundary (~28h).
    # Take the last step before SPLICE_BOUNDARY_MIN and the first step ≥ it.
    before = run_df[run_df["horizon_min"] < SPLICE_BOUNDARY_MIN]
    after  = run_df[run_df["horizon_min"] >= SPLICE_BOUNDARY_MIN]
    if not before.empty and not after.empty:
        last_before = before.iloc[-1]["prediction_mwh"]
        first_after = after.iloc[0]["prediction_mwh"]
        splice_jump = float(abs(first_after - last_before))
    else:
        splice_jump = float("nan")

    # Splice gets attached as its own row so it doesn't pollute per-bucket means.
    rows.append({
        "source":          source,
        "horizon_bucket":  "SPLICE_28h",
        "n_steps":         (1 if not (before.empty or after.empty) else 0),
        "mean_mwh":        float("nan"),
        "min_mwh":         float("nan"),
        "max_mwh":         float("nan"),
        "spread_mwh":      float("nan"),
        "std_mwh":         float("nan"),
        "cap_hit_rate":    float("nan"),
        "step_diff_mean":  splice_jump,
        "step_diff_max":   splice_jump,
        "sign_changes":    0,
        "local_extrema":   0,
    })

    return rows


def aggregate_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across runs: per (source, horizon_bucket) → mean/median of metrics."""
    if per_run.empty:
        return per_run
    metric_cols = ["n_steps", "mean_mwh", "spread_mwh", "std_mwh",
                   "cap_hit_rate", "step_diff_mean", "step_diff_max",
                   "sign_changes", "local_extrema"]
    g = per_run.groupby(["source", "horizon_bucket"], observed=True)
    out = g[metric_cols].agg(["mean", "median"]).reset_index()
    out.columns = ["source", "horizon_bucket"] + [
        f"{c}_{stat}" for c in metric_cols for stat in ("mean", "median")
    ]
    out["n_runs"] = g.size().reset_index(drop=True).values
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", default="tft,pd_direct,incumbent",
        help="Comma-separated source labels. Available: " + ",".join(LOG_FILES),
    )
    parser.add_argument(
        "--since", default="",
        help="Only include forecasts created on/after this UTC timestamp",
    )
    parser.add_argument(
        "--output", default="",
        help="CSV output path. Default: print summary to stdout",
    )
    parser.add_argument(
        "--per-run-output", default="",
        help="Optional path to write the full per-run table (one row per "
             "source × run × horizon_bucket). Default: not written",
    )
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in LOG_FILES]
    if unknown:
        parser.error(f"Unknown sources: {unknown}")

    since = pd.Timestamp(args.since, tz="UTC") if args.since else None

    all_rows = []
    for src in sources:
        df = load_forecast_log(src, LOG_FILES[src], since)
        if df.empty:
            print(f"  [{src}] no rows", file=sys.stderr)
            continue
        n_runs = df["forecast_creation_time"].nunique()
        print(f"  [{src}] {len(df):,} forecasts across {n_runs} runs",
              file=sys.stderr)
        for run_t, run_df in df.groupby("forecast_creation_time", sort=True):
            run_df = run_df.sort_values("horizon_min").reset_index(drop=True)
            for row in compute_run_metrics(run_df, src):
                row["run_creation_time"] = run_t
                all_rows.append(row)

    if not all_rows:
        print("No data to summarise.", file=sys.stderr)
        return 0

    per_run = pd.DataFrame(all_rows)
    cols = ["run_creation_time", "source", "horizon_bucket"] + [
        c for c in per_run.columns
        if c not in ("run_creation_time", "source", "horizon_bucket")
    ]
    per_run = per_run[cols].sort_values(
        ["source", "run_creation_time", "horizon_bucket"]
    ).reset_index(drop=True)

    if args.per_run_output:
        out_path = Path(args.per_run_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_run.to_csv(out_path, index=False)
        print(f"Wrote per-run table ({len(per_run)} rows) → {out_path}",
              file=sys.stderr)

    summary = aggregate_summary(per_run)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"Wrote summary ({len(summary)} rows) → {out_path}",
              file=sys.stderr)
    else:
        # Print a focused subset for human reading.
        cols_to_show = [
            "source", "horizon_bucket", "n_runs",
            "n_steps_mean",
            "spread_mwh_mean", "step_diff_mean_mean",
            "sign_changes_mean", "local_extrema_mean",
            "cap_hit_rate_mean",
        ]
        view = summary[cols_to_show].copy()
        for c in view.columns:
            if c.endswith("_mean") or c.endswith("_median"):
                view[c] = view[c].astype(float).round(2)
        print()
        print(view.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main() or 0)
