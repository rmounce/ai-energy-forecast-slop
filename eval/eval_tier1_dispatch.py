#!/usr/bin/env python3
"""
Pass B — 5-min tactical eval: battery dispatch value of Tier 1 LGBM.

Stratified sample of ~300 windows/stratum (spike/low/normal) from the
Jul 2025–Mar 2026 eval period. Each window = one P5MIN run_time, 12 steps
(5–60 min).

For each window, runs rolling LP MPC (reusing dispatch_simulator.simulate_mpc)
under four dispatch strategies:
  tier1_q50   — Tier 1 LGBM q50 forecast
  p5min_direct — raw P5MIN rrp[h0..h11] unchanged
  p5min_naive  — persistence: h0 price held constant for all 12 steps
  oracle       — actual rrp at each interval (perfect foresight)

Revenue is always booked against actual prices. Differences are purely due to
forecast quality. NaN actuals are filled with P5MIN fallback (consistent with
dispatch_simulator.py Phase 3 mode).

Pass B gate: Tier 1 dispatch revenue >= p5min_naive (no regression).
No hard threshold yet — first run establishes the baseline.

Output files:
  eval/results/tier1_dispatch_results.csv  — $/window + $/day by (stratum, strategy)

Usage:
    source .venv/bin/activate
    nice -n 19 python eval/eval_tier1_dispatch.py [--sample 300] [--seed 42]
"""

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytz

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.dispatch_simulator import simulate_mpc
from eval.retro_tier1_inference import (
    load_actuals_from_parquet,
    load_p5min_from_parquet,
    build_features,
)

RESULTS_DIR     = ROOT / "eval" / "results"
PARQUET_DIR     = ROOT / "data" / "parquet"
DEFAULT_MODEL_DIR = ROOT / "models" / "lgbm_tactical"
OUT_FILE        = RESULTS_DIR / "tier1_dispatch_results.csv"

OUTPUT_STEPS = 12
SPIKE_THRESH = 300.0
LOW_THRESH   = -50.0
INTERVAL_H   = 5 / 60.0   # hours per step
HOURS_PER_DAY = 24.0


def classify_window(actuals: np.ndarray) -> str:
    valid = actuals[~np.isnan(actuals)]
    if len(valid) == 0:
        return "unknown"
    if np.any(valid >= SPIKE_THRESH):
        return "spike"
    if np.any(valid <= LOW_THRESH):
        return "low"
    return "normal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-start", default="2025-07-01")
    parser.add_argument("--eval-end",   default="2026-04-01")
    parser.add_argument("--sample",     type=int, default=300,
                        help="Max windows per stratum (default 300)")
    parser.add_argument("--seed",       type=int, default=42)
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

    print("Building actuals lookup dict...")
    act_rrp_dict: dict = {}
    for ts, val in act_df["rrp"].items():
        act_rrp_dict[ts.replace(second=0, microsecond=0)] = float(val)

    run_times_all  = sorted(p5min_runs.keys())
    run_times_eval = [t for t in run_times_all if eval_start <= t < eval_end]
    run_time_set   = set(run_times_all)
    print(f"\nEval period: {eval_start.date()} – {eval_end.date()}")
    print(f"P5MIN run_times in eval period: {len(run_times_eval):,}")

    pv_empty = pd.Series(dtype=float, name="pv")

    # ── Classify all eval windows ───────────────────────────────────────────
    print("Classifying windows by stratum...")
    window_strata: dict[str, list] = {"spike": [], "low": [], "normal": []}
    t0 = time.time()

    for run_time in run_times_eval:
        intervals = [run_time + pd.Timedelta(minutes=5 * h) for h in range(OUTPUT_STEPS)]
        actual_rrp = np.array([
            act_rrp_dict.get(iv.replace(second=0, microsecond=0), np.nan)
            for iv in intervals
        ], dtype=np.float64)

        if np.all(np.isnan(actual_rrp)):
            continue

        stratum = classify_window(actual_rrp)
        if stratum in window_strata:
            window_strata[stratum].append(run_time)

    for s, lst in window_strata.items():
        print(f"  {s}: {len(lst):,} windows available")

    # ── Stratified sample ───────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    sampled: dict[str, list] = {}
    for s, lst in window_strata.items():
        arr = np.array(lst)
        n   = min(args.sample, len(arr))
        idx = rng.choice(len(arr), size=n, replace=False)
        idx.sort()
        sampled[s] = arr[idx].tolist()
        print(f"  Sampled {n} {s} windows")

    total_windows = sum(len(v) for v in sampled.values())

    # ── Run dispatch simulation ─────────────────────────────────────────────
    print(f"\nRunning LP MPC dispatch ({total_windows} windows × 4 strategies)...")
    strategies = ["tier1_q50", "p5min_direct", "p5min_naive", "oracle"]
    rows = []
    t0 = time.time()
    done = 0

    for stratum, run_times in sampled.items():
        for run_time in run_times:
            p5min_rrp = p5min_runs[run_time]

            intervals = [run_time + pd.Timedelta(minutes=5 * h) for h in range(OUTPUT_STEPS)]
            actual_rrp = np.array([
                act_rrp_dict.get(iv.replace(second=0, microsecond=0), np.nan)
                for iv in intervals
            ], dtype=np.float64)

            p5min_arr = np.array(p5min_rrp, dtype=np.float64)
            # Fill NaN actuals with P5MIN fallback for revenue calculation
            actual_for_revenue = np.where(np.isnan(actual_rrp), p5min_arr, actual_rrp)

            # Tier 1 inference
            prev_rt = run_time - pd.Timedelta(minutes=5)
            prev_p5min_h0 = p5min_runs[prev_rt][0] if prev_rt in run_time_set else float("nan")
            feats = build_features(run_time, p5min_rrp, prev_p5min_h0, act_df, pv_empty)
            X_long = np.column_stack([
                np.tile(feats, (OUTPUT_STEPS, 1)),
                np.arange(OUTPUT_STEPS, dtype=np.float32).reshape(-1, 1),
            ])
            tier1_pred = q50_model.predict(X_long).astype(np.float64)

            forecasts = {
                "tier1_q50":    tier1_pred,
                "p5min_direct": p5min_arr,
                "p5min_naive":  np.full(OUTPUT_STEPS, p5min_arr[0]),
                "oracle":       actual_for_revenue,
            }

            for src in strategies:
                result = simulate_mpc(forecasts[src], actual_for_revenue)
                revenue_window = result["total_pnl"]   # $ per 60-min window
                revenue_day    = revenue_window * HOURS_PER_DAY  # $/day (normalised)
                rows.append({
                    "stratum": stratum,
                    "run_time": run_time,
                    "source": src,
                    "revenue_window": round(revenue_window, 6),
                    "revenue_day": round(revenue_day, 4),
                })

            done += 1
            if done % 50 == 0 or done == total_windows:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 0.001)
                eta  = (total_windows - done) / max(rate, 0.001)
                print(f"  {done}/{total_windows}  {rate:.0f} wins/s  ETA {eta:.0f}s", end="\r")

    elapsed = time.time() - t0
    print(f"\nDone: {done} windows in {elapsed:.1f}s")

    df = pd.DataFrame(rows)

    # ── Summary table ───────────────────────────────────────────────────────
    strata_order = ["all", "spike", "low", "normal"]
    summary_rows = []

    print("\n=== Dispatch Revenue Summary ($/day normalised) ===")
    for stratum in strata_order:
        if stratum == "all":
            sub = df
        else:
            sub = df[df["stratum"] == stratum]

        if len(sub) == 0:
            continue

        naive_mean = sub[sub["source"] == "p5min_naive"]["revenue_day"].mean()
        print(f"\n  [{stratum.upper()}]  n={sub['source'].value_counts().iloc[0]}")
        print(f"  {'Source':<16}  {'Mean $/day':>12}  {'vs naive':>10}  {'Skill':>8}")
        print(f"  {'-'*50}")

        for src in strategies:
            sub_src = sub[sub["source"] == src]
            mean_day = sub_src["revenue_day"].mean()
            vs_naive = mean_day - naive_mean
            skill    = vs_naive / max(abs(naive_mean), 1e-6) if naive_mean != 0 else float("nan")
            gate_str = ""
            if src == "tier1_q50" and stratum != "all":
                gate_str = " ✅" if mean_day >= naive_mean else " ❌"
            print(f"  {src:<16}  {mean_day:>12.4f}  {vs_naive:>+10.4f}  {skill:>+7.1%}{gate_str}")
            summary_rows.append({
                "stratum": stratum, "source": src,
                "mean_revenue_day": round(mean_day, 4),
                "vs_naive": round(vs_naive, 4),
                "skill_vs_naive": round(skill, 4),
                "n_windows": len(sub_src),
            })

    df_summary = pd.DataFrame(summary_rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    df_summary.to_csv(RESULTS_DIR / "tier1_dispatch_summary.csv", index=False)
    print(f"\nSaved → {OUT_FILE.relative_to(ROOT)}")
    print(f"Saved → {(RESULTS_DIR / 'tier1_dispatch_summary.csv').relative_to(ROOT)}")

    # Gate summary
    print("\n=== Pass B Gate: Tier 1 revenue >= p5min_naive (no regression) ===")
    for stratum in ["spike", "low", "normal"]:
        row_t1    = df_summary[(df_summary["stratum"] == stratum) & (df_summary["source"] == "tier1_q50")]
        row_naive = df_summary[(df_summary["stratum"] == stratum) & (df_summary["source"] == "p5min_naive")]
        if row_t1.empty or row_naive.empty:
            continue
        t1_val    = row_t1["mean_revenue_day"].iloc[0]
        naive_val = row_naive["mean_revenue_day"].iloc[0]
        pct       = (t1_val - naive_val) / max(abs(naive_val), 1e-6) * 100
        status    = "✅ PASS" if t1_val >= naive_val else "❌ FAIL"
        print(f"  {stratum:<8}  tier1={t1_val:.4f} vs naive={naive_val:.4f}  ({pct:+.1f}%)  {status}")


if __name__ == "__main__":
    main()
