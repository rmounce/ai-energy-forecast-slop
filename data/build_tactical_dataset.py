#!/usr/bin/env python3
"""
Build Tier 1 tactical training dataset for multi-output LightGBM.

For each P5MIN run_time, produces a single sample with:

  X  [N, F]  — flat feature vector computed at run_time
  y  [N, 12] — actual RRP at each of the 12 forecast intervals
              (run_time, run_time+5, ..., run_time+55 min)

Intervals where actuals are unavailable (near data end) get y=NaN; the
corresponding mask entry is False so those steps are excluded from training.

Feature set (F = 12 + 8 + 1 + 3 + 2 + 1 + 4 + 1 = 32):
  p5min_rrp_h{0..11}  — P5MIN forecast for each target interval  [12]
  eff_import_price_*  — current-tariff effective import price summaries [4]
  eff_feed_in_price_* — current-tariff effective feed-in price summaries [4]
  aemo_divergence_t1  — actual[t-1] minus p5min_self_forecast[t-1] [1]
  actual_rrp_t1/t2/t6 — lag actual prices at -5, -10, -30 min    [3]
  rolling_1h_std      — std of actuals in last 12 × 5-min (1h)    [1]
  rolling_3h_max      — max of actuals in last 36 × 5-min (3h)    [1]
  residual_demand_t1  — total_demand[t-1] minus power_pv[t-1]     [1]
                        (PV forward-filled from 30-min actuals)
  hour_sin/cos        — cyclic hour-of-day                         [2]
  dow_sin/cos         — cyclic day-of-week                         [2]
  is_imputed_p5min    — 1 if P5MIN value forward-filled, else 0   [1]

Notes:
  - is_intervention is not yet available in actuals_sa1_5m.parquet;
    aemo_divergence_t1 is currently unmasked. Add intervention masking
    once is_intervention is exported from InfluxDB.
  - residual_demand_t1 falls back to total_demand_t1 where power_pv
    is unavailable (pre-2025-03-30).
  - p5min_rrp_h0 is the forecast for run_time itself (the dispatch
    interval currently settling). All 12 intervals are in [run_time,
    run_time+55min].

Outputs (data/parquet/):
  X_tactical.npy            [N, 32]  — raw features (NOT normalised)
  y_tactical.npy            [N, 12]  — raw target RRP (NaN masked)
  y_tactical_mask.npy       [N, 12]  — bool: True where target exists
  run_times_tactical.npy    [N]      — int64 nanosecond run_time per sample
  tactical_meta.json                 — shapes, feature names, split info
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
sys.path.insert(0, str(ROOT))

from tariff_utils import load_tariff_profile, tariffed_price_frame_from_wholesale_mwh

OUTPUT_STEPS = 12   # 12 × 5-min = 60-min horizon
VAL_DAYS     = 60   # last N days of run_times → validation split
TRAIN_GAP_H  = 1    # hours gap between train and val to avoid leakage

from config_utils import load_config
CONFIG = load_config(ROOT / "config.yaml")

GENERAL_TARIFF_MAP, FEED_IN_TARIFF_MAP, NETWORK_LOSS_FACTOR = load_tariff_profile(CONFIG, ROOT)

# Legacy feature names for the original 24-feature Tier 1 contract.
LEGACY_FEATURE_NAMES = (
    [f"p5min_rrp_h{h}" for h in range(OUTPUT_STEPS)]   # 12
    + ["aemo_divergence_t1"]                             # 1
    + ["actual_rrp_t1", "actual_rrp_t2", "actual_rrp_t6"]  # 3
    + ["rolling_1h_std", "rolling_3h_max"]               # 2
    + ["residual_demand_t1"]                              # 1
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]     # 4
    + ["is_imputed_p5min"]                                # 1
)

# Current feature names — must stay in sync with array construction below.
FEATURE_NAMES = (
    [f"p5min_rrp_h{h}" for h in range(OUTPUT_STEPS)]   # 12
    + [
        "eff_import_price_h0",
        "eff_feed_in_price_h0",
        "eff_import_price_1h_mean",
        "eff_import_price_1h_max",
        "eff_import_price_1h_spread",
        "eff_feed_in_price_1h_mean",
        "eff_feed_in_price_1h_max",
        "eff_feed_in_price_1h_spread",
    ]                                                  # 8
    + ["aemo_divergence_t1"]                             # 1
    + ["actual_rrp_t1", "actual_rrp_t2", "actual_rrp_t6"]  # 3
    + ["rolling_1h_std", "rolling_3h_max"]               # 2
    + ["residual_demand_t1"]                              # 1
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]     # 4
    + ["is_imputed_p5min"]                                # 1
)
N_FEATURES = len(FEATURE_NAMES)  # 32


def _time_encodings(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos for hour-of-day and day-of-week (AEST, no DST)."""
    t = timestamps.tz_convert("Australia/Brisbane")
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * t.hour / 24),
        "hour_cos": np.cos(2 * np.pi * t.hour / 24),
        "dow_sin":  np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos":  np.cos(2 * np.pi * t.dayofweek / 7),
    }, index=timestamps)


def _build_tariff_feature_block(
    run_times: pd.DatetimeIndex,
    p5_rrp_flat: np.ndarray,
) -> np.ndarray:
    dt5 = pd.Timedelta(minutes=5)
    horizon_offsets = np.arange(OUTPUT_STEPS, dtype=np.int64) * dt5.value
    run_ns = run_times.view(np.int64)
    interval_ns = (run_ns[:, None] + horizon_offsets[None, :]).reshape(-1)
    interval_idx = pd.DatetimeIndex(interval_ns, tz="UTC")

    wholesale_flat = p5_rrp_flat.reshape(-1)
    tariffed = tariffed_price_frame_from_wholesale_mwh(
        pd.Series(wholesale_flat, index=interval_idx, dtype=np.float64),
        timezone=CONFIG["timezone"],
        general_tariff_map=GENERAL_TARIFF_MAP,
        feed_in_tariff_map=FEED_IN_TARIFF_MAP,
        network_loss_factor=NETWORK_LOSS_FACTOR,
        gst_rate=CONFIG["gst_rate"],
    )
    import_curve = tariffed["general_price_mwh"].to_numpy(dtype=np.float32, copy=False).reshape(len(run_times), OUTPUT_STEPS)
    export_curve = tariffed["feed_in_price_mwh"].to_numpy(dtype=np.float32, copy=False).reshape(len(run_times), OUTPUT_STEPS)

    return np.column_stack([
        import_curve[:, 0],
        export_curve[:, 0],
        import_curve.mean(axis=1),
        import_curve.max(axis=1),
        np.ptp(import_curve, axis=1),
        export_curve.mean(axis=1),
        export_curve.max(axis=1),
        np.ptp(export_curve, axis=1),
    ]).astype(np.float32)


def build_dataset(p5min: pd.DataFrame,
                  actuals_5m: pd.DataFrame,
                  actuals_30m: pd.DataFrame | None = None,
                  dry_run: bool = False) -> tuple:
    """
    Build (X, y, mask, run_times) arrays.

    p5min      : columns [run_time, interval_dt, rrp]; exactly 12 per run
    actuals_5m : time-indexed DataFrame with column 'rrp', 'total_demand'
    actuals_30m: time-indexed DataFrame with 'power_pv', 'total_demand';
                 used only for PV to compute residual_demand.
                 Forward-filled to 5-min resolution before joining.
    """
    dt5 = pd.Timedelta(minutes=5)

    # ── Sort P5MIN and extract run_times in order ─────────────────────────────
    print("  Sorting P5MIN by (run_time, interval_dt)...")
    p5_sorted = p5min.sort_values(["run_time", "interval_dt"]).reset_index(drop=True)

    # Verify exactly 12 intervals per run
    run_counts = p5_sorted.groupby("run_time").size()
    assert (run_counts == 12).all(), "Not all P5MIN runs have 12 intervals"
    print(f"  {len(run_counts):,} runs × 12 intervals confirmed")

    all_run_times = pd.DatetimeIndex(run_counts.index)
    N = len(all_run_times)
    if dry_run:
        N = min(N, 500)
        all_run_times = all_run_times[:N]

    # Reshape P5MIN rrp to [N, 12] — safe because exactly 12 per run in sorted order
    p5_rrp_flat = p5_sorted["rrp"].values[:N * 12].reshape(N, 12)

    # Verify first interval == run_time for each run (sanity check)
    first_intervals = p5_sorted["interval_dt"].values[::12][:N]
    rt_arr = p5_sorted["run_time"].values[::12][:N]
    assert (first_intervals == rt_arr).all(), \
        "First P5MIN interval_dt does not equal run_time — data ordering issue"

    # ── Tariff-aware effective import/export summaries ───────────────────────
    print("  Deriving tariff-aware import/export summaries from the P5MIN curve...")
    tariff_feature_block = _build_tariff_feature_block(all_run_times, p5_rrp_flat)

    # ── P5MIN self-forecast series (first interval of each run) ──────────────
    # self_forecast[T] = P5MIN forecast for interval T issued at run_time T
    p5_self = pd.Series(p5_rrp_flat[:, 0], index=all_run_times)

    # ── Actual 5-min data ─────────────────────────────────────────────────────
    act = actuals_5m.sort_index()

    # Lag times
    t_minus_1 = all_run_times - dt5
    t_minus_2 = all_run_times - 2 * dt5
    t_minus_6 = all_run_times - 6 * dt5

    print("  Computing actual lag features...")
    actual_t1 = act["rrp"].reindex(t_minus_1).values.astype(np.float32)
    actual_t2 = act["rrp"].reindex(t_minus_2).values.astype(np.float32)
    actual_t6 = act["rrp"].reindex(t_minus_6).values.astype(np.float32)

    # ── AEMO divergence at t-1 ────────────────────────────────────────────────
    # = actual[run_time-5min] − P5MIN self-forecast at (run_time-5min)
    # The self-forecast at t-1 is p5_self[t-1] (the previous run's h0 value).
    print("  Computing AEMO divergence (actual t-1 minus P5MIN self-forecast t-1)...")
    p5_t1 = p5_self.reindex(t_minus_1).values.astype(np.float32)
    divergence_t1 = actual_t1 - p5_t1
    # Where either is NaN (no prior run or no actual), divergence becomes NaN → 0-filled

    # ── Rolling volatility features (on the full actuals series) ─────────────
    print("  Computing rolling volatility features (1h std, 3h max)...")
    rrp_roll = act["rrp"].copy()
    # Compute rolling on a regular time series; reindex to get values at t-1
    roll_1h_std = rrp_roll.rolling("1h", min_periods=6).std()
    roll_3h_max = rrp_roll.rolling("3h", min_periods=18).max()
    rolling_1h_std = roll_1h_std.reindex(t_minus_1).values.astype(np.float32)
    rolling_3h_max = roll_3h_max.reindex(t_minus_1).values.astype(np.float32)

    # ── Residual demand at t-1 ────────────────────────────────────────────────
    # residual = total_demand - power_pv (solar subtraction)
    # PV comes from 30-min actuals, forward-filled to 5-min resolution.
    print("  Computing residual demand at t-1...")
    if actuals_30m is not None and "power_pv" in actuals_30m.columns:
        # Forward-fill PV to 5-min actuals index
        pv_series = actuals_30m["power_pv"].sort_index()
        # Reindex to full 5-min actuals index and forward-fill (max 6 steps = 30 min)
        pv_5m = pv_series.reindex(act.index.union(pv_series.index)).sort_index() \
                         .ffill(limit=6).reindex(act.index)
        total_demand_5m = act["total_demand"]
        residual_demand = (total_demand_5m - pv_5m.fillna(0)).reindex(t_minus_1).values.astype(np.float32)
        n_pv_valid = pv_5m.notna().sum()
        print(f"    PV coverage: {n_pv_valid:,}/{len(pv_5m):,} 5-min steps "
              f"({pv_5m.notna().mean():.1%})")
    else:
        print("    WARNING: power_pv unavailable — residual_demand = total_demand")
        residual_demand = act["total_demand"].reindex(t_minus_1).values.astype(np.float32)

    # ── Time encodings at run_time ────────────────────────────────────────────
    print("  Computing time encodings...")
    time_enc = _time_encodings(all_run_times)

    # ── is_imputed_p5min ──────────────────────────────────────────────────────
    # All runs in the backfill have real P5MIN data (no imputation).
    # Live ingest will set this to 1 for forward-filled fallback intervals.
    is_imputed = np.zeros(N, dtype=np.float32)

    # ── Assemble X [N, 32] ────────────────────────────────────────────────────
    print("  Assembling feature matrix...")
    X = np.column_stack([
        p5_rrp_flat,                                              # [N, 12]
        tariff_feature_block,                                     # [N,  8]
        np.nan_to_num(divergence_t1, nan=0.0).reshape(-1, 1),    # [N,  1]
        np.nan_to_num(actual_t1,     nan=0.0).reshape(-1, 1),    # [N,  1]
        np.nan_to_num(actual_t2,     nan=0.0).reshape(-1, 1),    # [N,  1]
        np.nan_to_num(actual_t6,     nan=0.0).reshape(-1, 1),    # [N,  1]
        np.nan_to_num(rolling_1h_std, nan=0.0).reshape(-1, 1),   # [N,  1]
        np.nan_to_num(rolling_3h_max, nan=0.0).reshape(-1, 1),   # [N,  1]
        np.nan_to_num(residual_demand, nan=0.0).reshape(-1, 1),  # [N,  1]
        time_enc["hour_sin"].values.reshape(-1, 1),              # [N,  1]
        time_enc["hour_cos"].values.reshape(-1, 1),              # [N,  1]
        time_enc["dow_sin"].values.reshape(-1, 1),               # [N,  1]
        time_enc["dow_cos"].values.reshape(-1, 1),               # [N,  1]
        is_imputed.reshape(-1, 1),                               # [N,  1]
    ]).astype(np.float32)

    assert X.shape == (N, N_FEATURES), f"X shape mismatch: {X.shape} != {(N, N_FEATURES)}"

    # ── Targets y [N, 12] ─────────────────────────────────────────────────────
    print("  Building target matrix (actual RRP at each of 12 intervals)...")
    y = np.full((N, OUTPUT_STEPS), np.nan, dtype=np.float32)
    y_mask = np.zeros((N, OUTPUT_STEPS), dtype=bool)
    for h in range(OUTPUT_STEPS):
        t_target = all_run_times + h * dt5
        vals = act["rrp"].reindex(t_target).values.astype(np.float32)
        valid = ~np.isnan(vals)
        y[:, h] = np.where(valid, vals, 0.0)
        y_mask[:, h] = valid

    cov = y_mask.mean(axis=0)
    print(f"  Target coverage by horizon:")
    print(f"    h0  (  0 min): {cov[0]:.1%}")
    print(f"    h5  ( 25 min): {cov[5]:.1%}")
    print(f"    h11 ( 55 min): {cov[11]:.1%}")

    # ── Feature missingness summary ───────────────────────────────────────────
    n_div_missing = np.isnan(divergence_t1).sum()
    n_lag_missing = np.isnan(actual_t1).sum()
    print(f"\n  Feature gaps (out of {N:,} runs):")
    print(f"    divergence_t1:   {n_div_missing:,} NaN → 0-filled")
    print(f"    actual lags:     {n_lag_missing:,} NaN → 0-filled")

    run_times_out = np.array(all_run_times, dtype="datetime64[ns]")
    return X, y, y_mask, run_times_out


def split_by_time(run_times_ns: np.ndarray,
                  val_days: int = VAL_DAYS,
                  train_gap_hours: int = TRAIN_GAP_H):
    rts = pd.DatetimeIndex(run_times_ns)
    cutoff_val   = rts.max() - pd.Timedelta(days=val_days)
    cutoff_train = cutoff_val - pd.Timedelta(hours=train_gap_hours)
    return rts < cutoff_train, rts >= cutoff_val


def main():
    parser = argparse.ArgumentParser(
        description="Build Tier 1 tactical dataset for LightGBM (0–60 min, 5-min resolution)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 500 runs and print shapes; do not save")
    args = parser.parse_args()

    print("=== Building Tier 1 tactical dataset ===")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading parquet files...")
    p5min = pd.read_parquet(PARQUET_DIR / "aemo_p5min_sa1.parquet")
    for col in ["run_time", "interval_dt"]:
        p5min[col] = pd.to_datetime(p5min[col], utc=True)
    print(f"  P5MIN:       {len(p5min):,} rows, {p5min.run_time.nunique():,} runs "
          f"({p5min.run_time.min().date()} → {p5min.run_time.max().date()})")

    actuals_5m = pd.read_parquet(PARQUET_DIR / "actuals_sa1_5m.parquet")
    actuals_5m.index = pd.to_datetime(actuals_5m.index, utc=True)
    print(f"  Actuals 5m:  {len(actuals_5m):,} rows "
          f"({actuals_5m.index.min().date()} → {actuals_5m.index.max().date()})")

    actuals_30m_path = PARQUET_DIR / "actuals_sa1.parquet"
    if actuals_30m_path.exists():
        actuals_30m = pd.read_parquet(actuals_30m_path)
        actuals_30m["time"] = pd.to_datetime(actuals_30m["time"], utc=True)
        actuals_30m = actuals_30m.set_index("time").sort_index()
        print(f"  Actuals 30m: {len(actuals_30m):,} rows (power_pv for residual demand)")
    else:
        actuals_30m = None
        print("  WARNING: actuals_sa1.parquet not found — residual_demand = total_demand")

    # ── Build ─────────────────────────────────────────────────────────────────
    print("\nBuilding samples...")
    X, y, y_mask, run_times = build_dataset(
        p5min, actuals_5m, actuals_30m, dry_run=args.dry_run
    )

    N = len(run_times)
    print(f"\nDataset shapes:")
    print(f"  X:      {X.shape}  ({N_FEATURES} features)")
    print(f"  y:      {y.shape}  ({OUTPUT_STEPS} horizon steps)")
    print(f"  y_mask: {y_mask.shape}, total valid: {y_mask.sum():,}")
    print(f"  Feature names: {FEATURE_NAMES}")

    if args.dry_run:
        print("\n[dry-run] Shapes look good. Remove --dry-run to build full dataset.")
        return

    # ── Train/val split ───────────────────────────────────────────────────────
    train_mask, val_mask = split_by_time(run_times)

    # ── Stratified eval hold-out (excluded from train and val) ───────────────
    strat_path = PARQUET_DIR / "stratified_eval_run_times_tactical.npy"
    test_mask  = np.zeros(len(run_times), dtype=bool)
    if strat_path.exists():
        strat_rt    = np.load(strat_path)                        # datetime64[ns]
        strat_set   = set(strat_rt.view(np.int64).tolist())
        rt_int64    = run_times.view(np.int64)
        for i, rt in enumerate(rt_int64):
            if int(rt) in strat_set:
                test_mask[i]  = True
                train_mask[i] = False
                val_mask[i]   = False
        n_found   = test_mask.sum()
        n_missing = len(strat_rt) - n_found
        print(f"\nStratified eval hold-out: {n_found:,} samples excluded from train/val")
        if n_missing:
            print(f"  ({n_missing} run_times in stratified file not found in dataset)")
    else:
        print(f"\nNo stratified eval file found ({strat_path.name}).")
        print("  Run data/build_stratified_eval_tactical.py, then rebuild to exclude hold-out.")

    n_train, n_val = train_mask.sum(), val_mask.sum()
    rts = pd.DatetimeIndex(run_times)
    print(f"\nTrain/val split (last {VAL_DAYS} days = val, {TRAIN_GAP_H}h gap):")
    print(f"  Train: {n_train:,}  ({rts[train_mask].min().date()} → {rts[train_mask].max().date()})")
    print(f"  Val:   {n_val:,}  ({rts[val_mask].min().date()} → {rts[val_mask].max().date()})")

    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]

    # ── Target stats ──────────────────────────────────────────────────────────
    valid_train_rrp = y[train_mask][y_mask[train_mask]]
    print(f"\nTrain target RRP stats ($/MWh, {len(valid_train_rrp):,} valid steps):")
    print(f"  p1={np.percentile(valid_train_rrp, 1):.1f}  p50={np.percentile(valid_train_rrp, 50):.1f}  "
          f"p99={np.percentile(valid_train_rrp, 99):.1f}  max={valid_train_rrp.max():.1f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving arrays...")
    np.save(PARQUET_DIR / "X_tactical.npy",          X)
    np.save(PARQUET_DIR / "y_tactical.npy",          y)
    np.save(PARQUET_DIR / "y_tactical_mask.npy",     y_mask)
    np.save(PARQUET_DIR / "run_times_tactical.npy",  run_times)
    np.savez(PARQUET_DIR / "split_indices_tactical.npz",
             train=train_idx, val=val_idx, test=test_idx)

    meta = {
        "output_steps":   OUTPUT_STEPS,
        "n_features":     N_FEATURES,
        "feature_names":  FEATURE_NAMES,
        "n_samples":      int(N),
        "n_train":        int(n_train),
        "n_val":          int(n_val),
        "n_test_stratified": int(test_mask.sum()),
        "val_days":       VAL_DAYS,
        "train_gap_hours": TRAIN_GAP_H,
        "run_time_min":   str(pd.Timestamp(run_times[0])),
        "run_time_max":   str(pd.Timestamp(run_times[-1])),
        "target_coverage": {
            f"h{h}": float(y_mask[:, h].mean()) for h in range(OUTPUT_STEPS)
        },
        "notes": [
            "is_intervention masking not yet applied to aemo_divergence_t1",
            "residual_demand uses 30-min power_pv forward-filled to 5-min resolution",
            "p5min_rrp_h0 is the forecast for run_time itself (h=0 = current dispatch interval)",
            "tariff-derived features are reconstructed from the current tariff contract "
            "(current-tariff backtest assumption)",
        ],
    }
    with open(PARQUET_DIR / "tactical_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    n_test = test_mask.sum()
    test_s = f", {n_test:,} stratified eval" if n_test else ""
    print(f"\n=== Tactical dataset build complete ===")
    print(f"  {N:,} samples ({n_train:,} train, {n_val:,} val{test_s})")
    print(f"  Files saved to {PARQUET_DIR}/")
    print(f"    X_tactical.npy          {X.nbytes / 1e6:.1f} MB")
    print(f"    y_tactical.npy          {y.nbytes / 1e6:.1f} MB")
    print(f"    y_tactical_mask.npy")
    print(f"    run_times_tactical.npy")
    print(f"    split_indices_tactical.npz")
    print(f"    tactical_meta.json")


if __name__ == "__main__":
    main()
