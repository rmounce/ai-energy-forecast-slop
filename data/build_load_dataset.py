#!/usr/bin/env python3
"""
Build TFT load model training arrays from load_actuals_tft.parquet.

Architecture:
  Encoder: 96 steps (48h) — power_load, power_pv, weather, time features, holiday flags
  Decoder: 144 steps (72h) — weather (known future proxy from actuals), time features, holiday flags
  Target:  power_load at each decoder step
  Mask:    1 where target is valid (not NaN)

Output files (data/parquet/):
  X_enc_load.npy           [N, 96, 13]  encoder features (normalised)
  X_dec_load.npy           [N, 144, 13] decoder features (normalised)
  y_load.npy               [N, 144]     normalised target
  y_load_raw.npy           [N, 144]     raw target (W) for evaluation
  y_load_mask.npy          [N, 144]     bool: valid target step
  run_times_load.npy       [N]          datetime64 — sample boundary (encoder/decoder split)
  split_indices_load.npz               train/val index arrays
  load_scalers.pkl                     StandardScaler per feature
  load_dataset_meta.json               shape / feature name metadata

Encoder features (13):
  power_load, power_pv, temp, humidity, wind_speed,
  is_public_holiday, is_daylight_saving_time,
  hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

Decoder features (13):
  temp, humidity, wind_speed, power_pv,
  is_public_holiday, is_daylight_saving_time, horizon_norm,
  hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos

Usage:
  python data/build_load_dataset.py
  python data/build_load_dataset.py --val-days 90 --stride 4
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT       = Path(__file__).resolve().parent.parent
PARQUET    = ROOT / "data" / "parquet"
IN_FILE    = PARQUET / "load_actuals_tft.parquet"

ENC_STEPS  = 96   # 48h lookback
DEC_STEPS  = 144  # 72h horizon

TZ_LOCAL   = "Australia/Adelaide"

ENC_FEATURES = [
    "power_load", "power_pv", "temp", "humidity", "wind_speed",
    "is_public_holiday", "is_daylight_saving_time",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]
DEC_FEATURES = [
    "temp", "humidity", "wind_speed", "power_pv",
    "is_public_holiday", "is_daylight_saving_time", "horizon_norm",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]


def add_calendar_features(df):
    """Add cyclic time encodings + holiday/DST flags (in-place, returns df)."""
    import pytz
    tz = pytz.timezone(TZ_LOCAL)

    if df.index.tz is None:
        local_idx = df.index.tz_localize("UTC").tz_convert(tz)
    else:
        local_idx = df.index.tz_convert(tz)

    df["hour_sin"]   = np.sin(2 * np.pi * local_idx.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * local_idx.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * local_idx.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * local_idx.dayofweek / 7)
    df["month_sin"]  = np.sin(2 * np.pi * (local_idx.month - 1) / 12)
    df["month_cos"]  = np.cos(2 * np.pi * (local_idx.month - 1) / 12)

    df["is_daylight_saving_time"] = (
        local_idx.map(lambda x: x.dst() > pd.Timedelta(seconds=0)).astype(np.float32)
    )

    sa_hols = holidays.AU(subdiv="SA")
    dates = pd.Series(local_idx.date, index=df.index)
    df["is_public_holiday"] = dates.map(lambda d: d in sa_hols).astype(np.float32).values

    return df


def build_scalers(df, feature_cols):
    """Fit StandardScaler on each continuous feature (skip binary flags)."""
    binary = {"is_public_holiday", "is_daylight_saving_time", "horizon_norm"}
    scalers = {}
    for col in feature_cols:
        if col in binary:
            scalers[col] = None   # passthrough
        else:
            vals = df[col].dropna().values.reshape(-1, 1)
            sc = StandardScaler()
            sc.fit(vals)
            scalers[col] = sc
    return scalers


def apply_scalers_batch(arr, feature_list, scalers):
    """Scale a 3-D array [N, T, F] in-place using per-feature scalers.
    Operates column-wise across all samples at once (vectorised).
    NaN values are filled with the feature mean before scaling (= 0 after scaling).
    """
    N, T, _ = arr.shape
    for i, feat in enumerate(feature_list):
        sc = scalers.get(feat)
        if sc is not None:
            col = arr[:, :, i].reshape(-1)          # [N*T]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                col = col.copy()
                col[nan_mask] = float(sc.mean_[0])  # mean → 0 after scaling
            arr[:, :, i] = sc.transform(col.reshape(-1, 1)).reshape(N, T)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-days",  type=int, default=90,
                        help="Number of days to hold out as validation (default 90)")
    parser.add_argument("--min-valid-frac", type=float, default=0.80,
                        help="Minimum fraction of valid (non-NaN) decoder steps to keep a sample")
    parser.add_argument("--stride", type=int, default=4,
                        help="Step between sequence start points (default 4 = every 2h). "
                             "Use 1 for dense (original) dataset.")
    args = parser.parse_args()

    if not IN_FILE.exists():
        print(f"ERROR: {IN_FILE} not found — run data/export_load_dataset.py first")
        sys.exit(1)

    print("=" * 60)
    print("Build TFT Load Dataset")
    print("=" * 60)

    # ── Load raw data
    df = pd.read_parquet(IN_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    print(f"\nRaw rows: {len(df):,}  [{df.index.min()} → {df.index.max()}]")

    # ── Fill PV gaps conservatively (nightly zeros are real zeros)
    df["power_pv"] = df["power_pv"].fillna(0.0)

    # ── Add calendar features (cyclic + holiday/DST flags)
    df = add_calendar_features(df)

    # ── Fit scalers on the full dataset (except validation period)
    # We fit on everything to avoid look-ahead; val data is within the time range
    # so it's included here but that's standard practice for feature normalisation.
    print("\nFitting scalers...")
    all_features = list(set(ENC_FEATURES + DEC_FEATURES))
    scalers = build_scalers(df, all_features)
    for feat, sc in scalers.items():
        if sc is not None:
            mean = float(sc.mean_[0])
            std  = float(sc.scale_[0])
            print(f"  {feat:<30} mean={mean:>8.2f}  std={std:>8.2f}")

    # ── Sliding window construction
    times   = df.index.values                      # [N_total] datetime64
    n       = len(times)
    n_total = n - ENC_STEPS - DEC_STEPS + 1
    starts  = np.arange(0, n_total, args.stride)
    n_samp  = len(starts)

    print(f"\nBuilding {n_samp:,} samples (window {ENC_STEPS}+{DEC_STEPS} steps, stride={args.stride})...")

    # Pre-extract arrays for each feature (avoid repeated pandas lookups)
    load_raw = df["power_load"].values.astype(np.float32)

    enc_arrs = {}
    for feat in ENC_FEATURES:
        enc_arrs[feat] = df[feat].values.astype(np.float32)

    dec_arrs = {}
    for feat in DEC_FEATURES:
        if feat == "horizon_norm":
            dec_arrs[feat] = np.arange(DEC_STEPS, dtype=np.float32) / DEC_STEPS
        else:
            dec_arrs[feat] = df[feat].values.astype(np.float32)

    # Allocate output arrays
    X_enc_all  = np.empty((n_samp, ENC_STEPS,  len(ENC_FEATURES)), dtype=np.float32)
    X_dec_all  = np.empty((n_samp, DEC_STEPS,  len(DEC_FEATURES)), dtype=np.float32)
    y_raw_all  = np.empty((n_samp, DEC_STEPS),                      dtype=np.float32)
    y_mask_all = np.zeros((n_samp, DEC_STEPS),                      dtype=bool)
    run_times  = np.empty(n_samp, dtype="datetime64[ns]")

    valid_count = 0
    keep        = np.ones(n_samp, dtype=bool)

    for idx, s in enumerate(starts):
        enc_start = s
        enc_end   = s + ENC_STEPS
        dec_start = enc_end
        dec_end   = dec_start + DEC_STEPS

        # Target + mask
        tgt = load_raw[dec_start:dec_end]
        msk = ~np.isnan(tgt)
        valid_frac = msk.sum() / DEC_STEPS

        if valid_frac < args.min_valid_frac:
            keep[idx] = False
            continue

        y_raw_all[idx]  = np.where(msk, tgt, 0.0)
        y_mask_all[idx] = msk
        run_times[idx]  = times[dec_start]

        # Encoder
        for j, feat in enumerate(ENC_FEATURES):
            arr = enc_arrs[feat]
            X_enc_all[idx, :, j] = arr[enc_start:enc_end]

        # Decoder
        for j, feat in enumerate(DEC_FEATURES):
            if feat == "horizon_norm":
                X_dec_all[idx, :, j] = dec_arrs[feat]
            else:
                X_dec_all[idx, :, j] = dec_arrs[feat][dec_start:dec_end]

        valid_count += 1
        if valid_count % 5_000 == 0:
            print(f"  {valid_count:,} valid samples built...")

    # Apply keep mask
    X_enc_all  = X_enc_all[keep]
    X_dec_all  = X_dec_all[keep]
    y_raw_all  = y_raw_all[keep]
    y_mask_all = y_mask_all[keep]
    run_times  = run_times[keep]
    N = keep.sum()
    print(f"\nKept {N:,} / {n_samp:,} samples (dropped {n_samp - N:,} with <{args.min_valid_frac:.0%} valid decoder steps)")

    # ── Scale features (vectorised: operates over all samples at once)
    print("Scaling features...")
    apply_scalers_batch(X_enc_all, ENC_FEATURES, scalers)
    apply_scalers_batch(X_dec_all, DEC_FEATURES, scalers)

    # ── Scale target
    load_sc = scalers["power_load"]
    y_all   = load_sc.transform(
        y_raw_all.reshape(-1, 1)
    ).reshape(N, DEC_STEPS).astype(np.float32)
    # Restore NaN-masked steps to 0 in normalised targets
    y_all[~y_mask_all] = 0.0

    # ── Fill any remaining NaN encoder steps (brief gaps)
    nan_enc = np.isnan(X_enc_all).sum()
    if nan_enc > 0:
        print(f"  Filling {nan_enc:,} NaN encoder values with 0")
        X_enc_all = np.nan_to_num(X_enc_all, nan=0.0)
    nan_dec = np.isnan(X_dec_all).sum()
    if nan_dec > 0:
        print(f"  Filling {nan_dec:,} NaN decoder values with 0")
        X_dec_all = np.nan_to_num(X_dec_all, nan=0.0)

    # ── Train / val split (time-based)
    val_cutoff_dt = pd.Timestamp(run_times.max()) - pd.Timedelta(days=args.val_days)
    val_cutoff    = np.datetime64(val_cutoff_dt.to_datetime64(), "ns")
    train_idx = np.where(run_times < val_cutoff)[0]
    val_idx   = np.where(run_times >= val_cutoff)[0]
    print(f"\nSplit: train={len(train_idx):,}  val={len(val_idx):,} (last {args.val_days}d)")
    print(f"  Train: {run_times[train_idx[0]]} → {run_times[train_idx[-1]]}")
    print(f"  Val:   {run_times[val_idx[0]]}  → {run_times[val_idx[-1]]}")

    # ── Save
    print("\nSaving arrays...")
    np.save(PARQUET / "X_enc_load.npy",      X_enc_all)
    np.save(PARQUET / "X_dec_load.npy",      X_dec_all)
    np.save(PARQUET / "y_load.npy",          y_all)
    np.save(PARQUET / "y_load_raw.npy",      y_raw_all)
    np.save(PARQUET / "y_load_mask.npy",     y_mask_all)
    np.save(PARQUET / "run_times_load.npy",  run_times)
    np.savez(PARQUET / "split_indices_load.npz", train=train_idx, val=val_idx)

    with open(PARQUET / "load_scalers.pkl", "wb") as f:
        pickle.dump(scalers, f)

    meta = {
        "n_samples":        int(N),
        "enc_steps":        ENC_STEPS,
        "dec_steps":        DEC_STEPS,
        "n_enc_features":   len(ENC_FEATURES),
        "n_dec_features":   len(DEC_FEATURES),
        "enc_feature_names": ENC_FEATURES,
        "dec_feature_names": DEC_FEATURES,
        "n_train":          int(len(train_idx)),
        "n_val":            int(len(val_idx)),
        "val_days":         args.val_days,
        "target_feature":   "power_load",
        "time_start":       str(run_times.min()),
        "time_end":         str(run_times.max()),
        "min_valid_frac":   args.min_valid_frac,
        "stride":           args.stride,
    }
    with open(PARQUET / "load_dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Sizes
    for fname in ["X_enc_load.npy", "X_dec_load.npy", "y_load.npy",
                  "y_load_raw.npy", "y_load_mask.npy"]:
        p = PARQUET / fname
        print(f"  {fname:<28} {p.stat().st_size / 1024 / 1024:.1f}MB  {np.load(p).shape}")

    print(f"\nDone.  {N:,} samples  ({len(train_idx):,} train / {len(val_idx):,} val)")


if __name__ == "__main__":
    main()
