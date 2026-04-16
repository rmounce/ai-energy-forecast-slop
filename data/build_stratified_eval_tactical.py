#!/usr/bin/env python3
"""
Build a stratified evaluation benchmark for Tier 1 tactical LightGBM.

Mirrors the approach of build_stratified_eval.py (TFT/Tier 2) but uses the
P5MIN tactical dataset (5-min intervals, 12-step / 60-min horizon).

A time-ordered val split (last 60 days) varies in composition across rebuilds
— a mild autumn window has far fewer spikes than a summer window, making
absolute model numbers incomparable. A fixed stratified set removes this.

Strata:
  spike  — runs where max actual RRP in 12-step window ≥ spike_threshold
           (default: $300/MWh; absolute, not percentile, since market cap is
           well-defined — $300 is roughly "stress pricing" in SA1)
  low    — runs where min actual RRP < --low-threshold (default $0, negatives)
  normal — remaining runs, balanced by Southern Hemisphere season

Outputs:
  data/parquet/stratified_eval_run_times_tactical.npy  — datetime64[ns]
    Used by build_tactical_dataset.py to exclude these from train/val.
    Used by evaluate_lgbm_tactical.py for reporting.

Notes:
  - Uses actual RRP targets (y_tactical.npy), not P5MIN forecasts, for scoring.
  - Samples with no valid targets are excluded from all strata.
  - File is written once and treated as stable; --force regenerates it.

Usage:
    python data/build_stratified_eval_tactical.py
    python data/build_stratified_eval_tactical.py --force
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"

SEASONS = ["summer", "autumn", "winter", "spring"]


def get_season(month: int) -> str:
    if month in (12, 1, 2):  return "summer"
    if month in (3, 4, 5):   return "autumn"
    if month in (6, 7, 8):   return "winter"
    return "spring"


def main():
    parser = argparse.ArgumentParser(
        description="Build stratified eval benchmark for Tier 1 tactical LightGBM")
    parser.add_argument("--spike-threshold", type=float, default=300.0,
                        help="Absolute RRP threshold for spike stratum (default: $300/MWh)")
    parser.add_argument("--low-threshold", type=float, default=0.0,
                        help="Min RRP below this → low/negative stratum (default: $0)")
    parser.add_argument("--n-spike", type=int, default=500,
                        help="Max samples from spike stratum (default: 500)")
    parser.add_argument("--n-low", type=int, default=300,
                        help="Max samples from low/negative stratum (default: 300)")
    parser.add_argument("--normal-per-season", type=int, default=200,
                        help="Normal samples per season (default: 200, total 800)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing file (changes benchmark — use with care)")
    args = parser.parse_args()

    out_path = PARQUET_DIR / "stratified_eval_run_times_tactical.npy"

    if out_path.exists() and not args.force:
        existing = np.load(out_path, allow_pickle=True)
        print(f"Tactical stratified eval set already exists: {out_path}")
        print(f"  {len(existing):,} run_times. Use --force to regenerate.")
        return

    rng = np.random.default_rng(args.seed)

    print("Loading tactical dataset arrays...")
    y_raw     = np.load(PARQUET_DIR / "y_tactical.npy")       # [N, 12] float32
    y_mask    = np.load(PARQUET_DIR / "y_tactical_mask.npy")  # [N, 12] bool
    run_times = np.load(PARQUET_DIR / "run_times_tactical.npy")  # [N] datetime64[ns]
    N = len(run_times)
    print(f"  {N:,} samples, {y_mask.sum():,} valid steps")

    # ── Score each sample across its full 12-step window
    has_valid    = y_mask.any(axis=1)
    spike_scores = np.full(N, np.nan, dtype=np.float32)
    min_scores   = np.full(N, np.nan, dtype=np.float32)

    scoreable = np.where(has_valid)[0]
    # Vectorised scoring: mask invalid steps before max/min
    y_masked = np.where(y_mask, y_raw, np.nan)
    spike_scores[scoreable] = np.nanmax(y_masked[scoreable], axis=1)
    min_scores[scoreable]   = np.nanmin(y_masked[scoreable], axis=1)

    print(f"  Scoreable: {len(scoreable):,}")
    print(f"  Spike score (max RRP): "
          f"p50={np.nanpercentile(spike_scores, 50):.0f}  "
          f"p95={np.nanpercentile(spike_scores, 95):.0f}  "
          f"p99={np.nanpercentile(spike_scores, 99):.0f}  "
          f"max={np.nanmax(spike_scores):.0f} $/MWh")
    n_above = (spike_scores >= args.spike_threshold).sum()
    print(f"  Runs with max RRP ≥ ${args.spike_threshold:.0f}: {n_above:,} "
          f"({n_above / len(scoreable):.1%} of scoreable)")

    # ── Season labels (AEST)
    rts_dt  = pd.DatetimeIndex(run_times).tz_localize("UTC").tz_convert("Australia/Brisbane")
    seasons = np.array([get_season(dt.month) for dt in rts_dt])

    # ── Spike stratum
    spike_cands = scoreable[spike_scores[scoreable] >= args.spike_threshold]
    n_spike     = min(args.n_spike, len(spike_cands))
    spike_idx   = (rng.choice(spike_cands, size=n_spike, replace=False)
                   if n_spike > 0 else np.array([], dtype=np.intp))
    print(f"\nSpike stratum (max RRP ≥ ${args.spike_threshold:.0f}):")
    print(f"  Candidates: {len(spike_cands):,}  →  selected: {n_spike}")
    if n_spike:
        print(f"  RRP range: "
              f"{spike_scores[spike_idx].min():.0f} – {spike_scores[spike_idx].max():.0f} $/MWh")

    # ── Low/negative stratum
    low_cands = scoreable[min_scores[scoreable] < args.low_threshold]
    low_cands = np.setdiff1d(low_cands, spike_idx)
    n_low     = min(args.n_low, len(low_cands))
    low_idx   = (rng.choice(low_cands, size=n_low, replace=False)
                 if n_low > 0 else np.array([], dtype=np.intp))
    print(f"\nLow/negative stratum (min RRP < ${args.low_threshold:.0f}):")
    print(f"  Candidates: {len(low_cands):,} (after removing spike overlap)  →  selected: {n_low}")

    # ── Normal stratum (seasonal balance)
    excluded    = np.union1d(spike_idx, low_idx)
    normal_pool = np.setdiff1d(scoreable, excluded)
    normal_parts = []
    print(f"\nNormal stratum ({args.normal_per_season}/season):")
    for season in SEASONS:
        cands = normal_pool[seasons[normal_pool] == season]
        n     = min(args.normal_per_season, len(cands))
        if n > 0:
            normal_parts.append(rng.choice(cands, size=n, replace=False))
        print(f"  {season:6s}: {len(cands):,} candidates → selected {n}")
    normal_idx = (np.concatenate(normal_parts)
                  if normal_parts else np.array([], dtype=np.intp))

    # ── Combine and save
    all_idx = np.union1d(np.union1d(spike_idx, low_idx), normal_idx)
    selected_run_times = run_times[all_idx]

    print(f"\n── Tactical stratified eval summary ──")
    print(f"  Total: {len(all_idx):,}  "
          f"(spike={len(spike_idx)}, low={len(low_idx)}, normal={len(normal_idx)})")
    rt_dt = pd.DatetimeIndex(selected_run_times).tz_localize("UTC")
    print(f"  Date range: {rt_dt.min().date()} → {rt_dt.max().date()}")
    print(f"  Season mix: { {s: int((seasons[all_idx] == s).sum()) for s in SEASONS} }")
    print(f"  As fraction of dataset: {len(all_idx) / N:.1%}")

    np.save(out_path, selected_run_times)
    print(f"\nSaved: {out_path}")
    print("Next: rebuild tactical dataset (build_tactical_dataset.py) to exclude these from train/val.")


if __name__ == "__main__":
    main()
