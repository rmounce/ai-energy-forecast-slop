#!/usr/bin/env python3
"""
Pass A — 5-min tactical eval: Tier 1 LGBM forecast accuracy per horizon step h0-h11.

For each P5MIN run_time in the eval period (Jul 2025–Mar 2026), compares:
  tier1_q50   — Tier 1 LGBM q50 corrected forecast (12 × 5-min values)
  p5min_direct — raw P5MIN rrp[h0..h11] unchanged
  p5min_naive  — persistence: h0 price held constant for all 12 horizons
  oracle       — actual 5-min rrp at each interval

Stratum classification per window (based on actual rrp over h0-h11):
  spike  — any actual >= 300 $/MWh
  low    — any actual <= -50 $/MWh (no spike in same window)
  normal — all other windows

Metrics output: MAE per horizon step h0-h11, skill score vs p5min_naive.

Output files:
  eval/results/tier1_accuracy_by_horizon.csv  — MAE per (stratum, horizon, source)
  eval/results/tier1_accuracy_summary.csv     — mean MAE across horizons per (stratum, source)

Usage:
    source .venv/bin/activate
    nice -n 19 python eval/eval_tier1_accuracy.py [--eval-start YYYY-MM-DD] [--eval-end YYYY-MM-DD]
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.retro_tier1_inference import (
    load_actuals_from_parquet,
    load_p5min_from_parquet,
    build_features,
)

RESULTS_DIR     = ROOT / "eval" / "results"
PARQUET_DIR     = ROOT / "data" / "parquet"
DEFAULT_MODEL_DIR = ROOT / "models" / "lgbm_tactical"
P5MIN_PARQUET   = PARQUET_DIR / "aemo_p5min_sa1.parquet"
ACTUALS_PARQUET = PARQUET_DIR / "actuals_sa1_5m.parquet"

OUT_HORIZON = RESULTS_DIR / "tier1_accuracy_by_horizon.csv"
OUT_SUMMARY = RESULTS_DIR / "tier1_accuracy_summary.csv"

OUTPUT_STEPS = 12
SPIKE_THRESH = 300.0
LOW_THRESH   = -50.0

SOURCES = ["tier1_q50", "p5min_direct", "p5min_naive"]


def classify_window(actuals: np.ndarray) -> str:
    if np.any(actuals >= SPIKE_THRESH):
        return "spike"
    if np.any(actuals <= LOW_THRESH):
        return "low"
    return "normal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-start", default="2025-07-01",
                        help="Eval period start (UTC date, default: 2025-07-01)")
    parser.add_argument("--eval-end", default="2026-04-01",
                        help="Eval period end exclusive (UTC date, default: 2026-04-01)")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing Tier 1 tactical LightGBM artifacts.",
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end   = pd.Timestamp(args.eval_end,   tz="UTC")

    if not (model_dir / "lgbm_q50.pkl").exists():
        print(f"ERROR: model not found at {model_dir}/lgbm_q50.pkl")
        sys.exit(1)

    print("Loading Tier 1 LGBM q50 model...")
    q50_model = joblib.load(model_dir / "lgbm_q50.pkl")

    p5min_runs = load_p5min_from_parquet()
    act_df     = load_actuals_from_parquet()

    # Dict for O(1) actual rrp lookup by interval_dt
    # Normalize to UTC, round to 5-min so lookups are reliable
    print("Building actuals lookup dict...")
    act_rrp_dict = {}
    for ts, row in act_df["rrp"].items():
        key = ts.replace(second=0, microsecond=0)
        act_rrp_dict[key] = float(row)

    # Filter run_times to eval period
    run_times_all = sorted(p5min_runs.keys())
    run_times_eval = [t for t in run_times_all if eval_start <= t < eval_end]
    print(f"\nEval period: {eval_start.date()} – {eval_end.date()}")
    print(f"P5MIN run_times in eval period: {len(run_times_eval):,}")

    # Build fast prev-run lookup
    run_time_set = set(run_times_all)

    # Accumulators: {(stratum, horizon_step, source): [abs_error, ...]}
    errors: dict = defaultdict(list)
    n_skip_no_actuals = 0
    n_skip_partial    = 0
    t0 = time.time()
    dot_interval = max(1, len(run_times_eval) // 50)

    pv_empty = pd.Series(dtype=float, name="pv")

    for i, run_time in enumerate(run_times_eval):
        if i % dot_interval == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            eta  = (len(run_times_eval) - i - 1) / max(rate, 0.001)
            print(f"  {i+1:>6}/{len(run_times_eval)}  {rate:.0f} wins/s  ETA {eta:.0f}s", end="\r")

        p5min_rrp = p5min_runs[run_time]   # list[12]

        # Actual rrp at each of the 12 intervals (h0=run_time, h1=+5min … h11=+55min)
        # Target definition from build_tactical_dataset.py: t_target = run_time + h*5min
        intervals = [run_time + pd.Timedelta(minutes=5 * h) for h in range(OUTPUT_STEPS)]
        actual_rrp = np.array([
            act_rrp_dict.get(iv.replace(second=0, microsecond=0), np.nan)
            for iv in intervals
        ], dtype=np.float64)

        n_missing = int(np.sum(np.isnan(actual_rrp)))
        if n_missing == OUTPUT_STEPS:
            n_skip_no_actuals += 1
            continue
        if n_missing > 0:
            n_skip_partial += 1
            # Don't skip — accumulate per-step errors only where actual is available

        stratum = classify_window(actual_rrp)

        # Tier 1 inference (price-only; pv_series left empty → residual_demand = total_demand)
        prev_rt = run_time - pd.Timedelta(minutes=5)
        prev_p5min_h0 = p5min_runs[prev_rt][0] if prev_rt in run_time_set else float("nan")

        feats = build_features(run_time, p5min_rrp, prev_p5min_h0, act_df, pv_empty)

        X_long = np.column_stack([
            np.tile(feats, (OUTPUT_STEPS, 1)),
            np.arange(OUTPUT_STEPS, dtype=np.float32).reshape(-1, 1),
        ])
        tier1_preds = q50_model.predict(X_long).astype(np.float64)   # shape (12,)

        p5min_arr = np.array(p5min_rrp, dtype=np.float64)
        naive_arr = np.full(OUTPUT_STEPS, p5min_arr[0], dtype=np.float64)

        preds_by_source = {
            "tier1_q50":    tier1_preds,
            "p5min_direct": p5min_arr,
            "p5min_naive":  naive_arr,
        }

        for h in range(OUTPUT_STEPS):
            if np.isnan(actual_rrp[h]):
                continue
            for src, pred_arr in preds_by_source.items():
                errors[(stratum, h, src)].append(abs(pred_arr[h] - actual_rrp[h]))
            errors[(stratum, h, "oracle")].append(0.0)

    elapsed = time.time() - t0
    total = len(run_times_eval)
    n_processed = total - n_skip_no_actuals
    print(f"\nDone: {n_processed:,}/{total:,} windows in {elapsed:.1f}s  "
          f"(skipped {n_skip_no_actuals} no-actuals; {n_skip_partial} partial — "
          f"per-step NaN excluded)")

    # Build MAE table
    strata = ["all", "spike", "low", "normal"]
    all_sources = SOURCES + ["oracle"]
    rows_horizon = []
    rows_summary_acc = defaultdict(list)

    for h in range(OUTPUT_STEPS):
        for src in all_sources:
            # Per-stratum
            for stratum in ["spike", "low", "normal"]:
                key = (stratum, h, src)
                vals = errors.get(key, [])
                mae  = float(np.mean(vals)) if vals else np.nan
                n    = len(vals)
                rows_horizon.append({"stratum": stratum, "horizon": h, "source": src,
                                     "mae": round(mae, 4), "n": n})
                if not np.isnan(mae):
                    rows_summary_acc[(stratum, src)].append(mae)

            # "all" strata combined
            all_vals = []
            for stratum in ["spike", "low", "normal"]:
                all_vals.extend(errors.get((stratum, h, src), []))
            mae_all = float(np.mean(all_vals)) if all_vals else np.nan
            rows_horizon.append({"stratum": "all", "horizon": h, "source": src,
                                  "mae": round(mae_all, 4), "n": len(all_vals)})
            if not np.isnan(mae_all):
                rows_summary_acc[("all", src)].append(mae_all)

    df_horizon = pd.DataFrame(rows_horizon)

    # Summary: mean MAE over horizons h1-h11 (skip h0 — most sources tied on persistence)
    rows_summary = []
    for stratum in strata:
        naive_mean = float(np.mean(rows_summary_acc.get((stratum, "p5min_naive"), [np.nan])))
        for src in all_sources:
            vals = rows_summary_acc.get((stratum, src), [])
            mean_mae_all  = float(np.mean(vals)) if vals else np.nan
            # h1-h11 only
            h1_vals = []
            for h in range(1, OUTPUT_STEPS):
                h1_vals.extend(errors.get((stratum if stratum != "all" else "spike", h, src), []))
                if stratum == "all":
                    h1_vals.extend(errors.get(("low",    h, src), []))
                    h1_vals.extend(errors.get(("normal", h, src), []))
            mean_mae_h1_11 = float(np.mean(h1_vals)) if h1_vals else np.nan
            skill = float(1.0 - mean_mae_all / naive_mean) if naive_mean > 0 else np.nan
            rows_summary.append({
                "stratum": stratum, "source": src,
                "mae_all_horizons": round(mean_mae_all, 4),
                "mae_h1_h11": round(mean_mae_h1_11, 4),
                "skill_vs_naive": round(skill, 4),
            })
    df_summary = pd.DataFrame(rows_summary)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_horizon.to_csv(OUT_HORIZON, index=False)
    df_summary.to_csv(OUT_SUMMARY, index=False)
    print(f"\nSaved → {OUT_HORIZON.relative_to(ROOT)}")
    print(f"Saved → {OUT_SUMMARY.relative_to(ROOT)}")

    # Console summary table
    print("\n=== MAE Summary (mean over h0-h11) ===")
    pivot = df_summary.pivot(index="source", columns="stratum", values="mae_all_horizons")
    print(pivot.to_string())
    print("\n=== Skill vs p5min_naive (h0-h11 MAE, positive = better) ===")
    pivot_skill = df_summary.pivot(index="source", columns="stratum", values="skill_vs_naive")
    print(pivot_skill.to_string())

    # Per-horizon detail for "all" stratum
    print("\n=== MAE by horizon step (stratum=all) ===")
    df_all_h = df_horizon[df_horizon["stratum"] == "all"].pivot(
        index="horizon", columns="source", values="mae"
    )[all_sources]
    print(df_all_h.to_string())


if __name__ == "__main__":
    main()
