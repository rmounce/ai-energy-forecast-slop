#!/usr/bin/env python3
"""
Build a stratified evaluation benchmark for durable TFT run comparisons.

The current val set (last N days) varies in composition run-to-run — volatile summer
windows produce higher nMAPE than mild autumn windows, making absolute numbers
incomparable. A fixed stratified set removes this ambiguity.

Strata (data-driven, no hardcoded dates):
  spike  — top --spike-pct% by max actual RRP in first 28h (56 decoder steps)
  low    — samples where min actual RRP < --low-threshold (default $0, negative prices)
  normal — remaining samples, stratified by Southern Hemisphere season,
            --normal-per-season samples each

Outputs:
  data/parquet/stratified_eval_run_times.npy  — datetime64[ns] timestamps of selected runs
    Used by build_training_dataset.py to exclude these from train/val on next rebuild.
    Used by evaluate_tft.py (--eval-set stratified) to select the eval samples.

Design notes:
  - Scoring uses first 28h (56 steps) only — the most commercially important window
    and the one with highest mask coverage (~48–98% of samples).
  - Samples with no valid steps in the first 28h are excluded from scoring and
    cannot be selected (they can't be meaningfully evaluated anyway).
  - The file is written once and treated as stable. Re-running with --force will
    regenerate and change the benchmark — don't do this between comparable runs.
  - Random seed --seed (default 42) ensures reproducibility.

Usage:
    python data/build_stratified_eval.py
    python data/build_stratified_eval.py --spike-pct 5 --n-spike 300 --normal-per-season 100
    python data/build_stratified_eval.py --force   # regenerate (changes benchmark!)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"

SPIKE_STEPS = 56   # first 28h for scoring (highest PREDISPATCH coverage)
SEASONS = ["summer", "autumn", "winter", "spring"]


def get_season(month: int) -> str:
    """Southern Hemisphere seasons by calendar month."""
    if month in (12, 1, 2):
        return "summer"
    if month in (3, 4, 5):
        return "autumn"
    if month in (6, 7, 8):
        return "winter"
    return "spring"


def main():
    parser = argparse.ArgumentParser(
        description="Build stratified eval benchmark (run once, then treat as stable)"
    )
    parser.add_argument("--spike-pct", type=float, default=5.0,
                        help="Top percentile of samples by max RRP → spike stratum (default: 5)")
    parser.add_argument("--low-threshold", type=float, default=0.0,
                        help="Min RRP below this → low/negative stratum (default: 0, i.e. negative prices)")
    parser.add_argument("--n-spike", type=int, default=300,
                        help="Samples to draw from spike stratum (default: 300)")
    parser.add_argument("--n-low", type=int, default=200,
                        help="Max samples from low/negative stratum (default: 200)")
    parser.add_argument("--normal-per-season", type=int, default=100,
                        help="Normal samples per season (default: 100, total 400)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing file (changes the benchmark — use with care)")
    args = parser.parse_args()

    out_path = PARQUET_DIR / "stratified_eval_run_times.npy"

    if out_path.exists() and not args.force:
        existing = np.load(out_path, allow_pickle=True)
        print(f"Stratified eval set already exists: {out_path}")
        print(f"  {len(existing):,} eval run_times. Use --force to regenerate.")
        return

    rng = np.random.default_rng(args.seed)

    # ── Load arrays
    print("Loading arrays...")
    y_raw     = np.load(PARQUET_DIR / "y_targets_raw.npy")   # [N, 144]
    y_mask    = np.load(PARQUET_DIR / "y_mask.npy")          # [N, 144] bool
    run_times = np.load(PARQUET_DIR / "run_times.npy")       # [N] datetime64[ns]
    N = len(run_times)
    print(f"  {N:,} samples")

    # ── Score each sample on first 28h only
    short_mask = y_mask[:, :SPIKE_STEPS]      # [N, 56] bool
    has_valid  = short_mask.any(axis=1)        # [N] — can be scored

    spike_scores = np.full(N, np.nan, dtype=np.float32)
    min_scores   = np.full(N, np.nan, dtype=np.float32)

    scoreable = np.where(has_valid)[0]
    for i in scoreable:
        vals = y_raw[i, :SPIKE_STEPS][short_mask[i]]
        spike_scores[i] = vals.max()
        min_scores[i]   = vals.min()

    print(f"  Scoreable samples: {len(scoreable):,} (have ≥1 valid step in first 28h)")
    print(f"  Spike score (max RRP in 28h): p50={np.nanpercentile(spike_scores, 50):.0f}, "
          f"p95={np.nanpercentile(spike_scores, 95):.0f}, "
          f"p99={np.nanpercentile(spike_scores, 99):.0f}, "
          f"max={np.nanmax(spike_scores):.0f} $/MWh")

    # ── Season labels from run_times
    rts_dt = pd.DatetimeIndex(run_times).tz_localize("UTC").tz_convert("Australia/Adelaide")
    seasons = np.array([get_season(dt.month) for dt in rts_dt])

    # ── Spike stratum
    threshold = np.nanpercentile(spike_scores, 100 - args.spike_pct)
    spike_cands = scoreable[spike_scores[scoreable] >= threshold]
    n_spike = min(args.n_spike, len(spike_cands))
    spike_idx = rng.choice(spike_cands, size=n_spike, replace=False)
    print(f"\nSpike stratum:")
    print(f"  Top {args.spike_pct:.0f}%: {len(spike_cands):,} candidates "
          f"(max RRP ≥ {threshold:.0f} $/MWh)")
    print(f"  Selected: {n_spike}")
    print(f"  RRP range in selection: "
          f"{spike_scores[spike_idx].min():.0f} – {spike_scores[spike_idx].max():.0f} $/MWh")

    # ── Low/negative price stratum (excluding already-selected spike samples)
    low_cands = scoreable[min_scores[scoreable] < args.low_threshold]
    low_cands = np.setdiff1d(low_cands, spike_idx)
    n_low = min(args.n_low, len(low_cands))
    low_idx = (rng.choice(low_cands, size=n_low, replace=False)
               if n_low > 0 else np.array([], dtype=np.intp))
    print(f"\nLow/negative price stratum:")
    print(f"  min RRP < {args.low_threshold:.0f}: {len(low_cands) + len(np.intersect1d(low_cands, spike_idx)):,} "
          f"candidates ({len(low_cands):,} after removing spike overlap)")
    print(f"  Selected: {n_low}")

    # ── Normal stratum (seasonal, from remaining scoreable samples)
    excluded_so_far = np.union1d(spike_idx, low_idx)
    normal_pool = np.setdiff1d(scoreable, excluded_so_far)
    normal_idx_parts = []
    print(f"\nNormal stratum ({args.normal_per_season} per season):")
    for season in SEASONS:
        season_cands = normal_pool[seasons[normal_pool] == season]
        n = min(args.normal_per_season, len(season_cands))
        if n > 0:
            normal_idx_parts.append(rng.choice(season_cands, size=n, replace=False))
        print(f"  {season:6s}: {len(season_cands):,} candidates → selected {n}")
    normal_idx = (np.concatenate(normal_idx_parts)
                  if normal_idx_parts else np.array([], dtype=np.intp))

    # ── Combine and save
    all_idx = np.union1d(np.union1d(spike_idx, low_idx), normal_idx)
    selected_run_times = run_times[all_idx]

    print(f"\n── Stratified eval set summary ──")
    print(f"  Total samples: {len(all_idx):,}")
    print(f"    spike={len(spike_idx)}, low={len(low_idx)}, normal={len(normal_idx)}")
    rt_dt = pd.DatetimeIndex(selected_run_times).tz_localize("UTC")
    print(f"  Date range: {rt_dt.min().date()} → {rt_dt.max().date()}")
    season_counts = {s: (seasons[all_idx] == s).sum() for s in SEASONS}
    print(f"  Season mix: {season_counts}")

    np.save(out_path, selected_run_times)
    print(f"\nSaved: {out_path}")
    print("Next: rebuild training dataset (build_training_dataset.py) to exclude these from train/val.")
    print("      evaluate_tft.py --eval-set stratified will use this set.")


if __name__ == "__main__":
    main()
