#!/usr/bin/env python3
"""
Build run-aligned training dataset for TFT price forecasting.

Implements Option B (run-aligned) covariate construction from Sinclair et al. 2026:
  For each PREDISPATCH run at time T:
    encoder: actuals[T - 48h+30min : T]  → [96, n_enc]
    decoder: predispatch_run_T[T+30min : T+28h]  → [56, n_dec]
    target:  actuals_rrp[T+30min : T+28h]  → [56]

This matches inference exactly: at prediction time T, the encoder sees the last 48h
of actuals, and the decoder sees the PREDISPATCH run issued at T.

Normalisation:
  QuantileTransformer(n_quantiles=2000, output_distribution='normal') per feature,
  fitted on train split only (last 30 days of run_times = validation).
  Confirmed approach from Sinclair et al. training.py.

Outputs (data/parquet/):
  X_encoder.npy        [N, 96, N_ENC_FEATURES]  — normalised
  X_decoder.npy        [N, 56, N_DEC_FEATURES]  — normalised prices/demand + raw time encodings
  y_targets.npy        [N, 56]                  — normalised actual RRP
  y_targets_raw.npy    [N, 56]                  — raw actual RRP in $/MWh (for evaluation)
  run_times.npy        [N]                       — Unix timestamps of each run_time (int64, UTC)
  scalers.pkl                                    — dict of feature → fitted QuantileTransformer
  dataset_meta.json                              — shapes, feature names, split info
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"

INPUT_LENGTH = 96    # encoder steps = 2 days of 30-min intervals

# OUTPUT_LENGTH tradeoff (PREDISPATCH future interval availability):
#   32 (16h) → 97.9% of runs qualify — paper's exact choice (Sinclair et al.)
#   48 (24h) → 64.6% of runs qualify
#   56 (28h) → 47.9% of runs qualify — but excludes all morning-AEST runs
#              (UTC 15:00-02:00 = AEST 01:00-12:00), creating severe time-of-day bias.
#
# DEFAULT: 32 (matches paper, no bias). Change here to experiment with longer horizons.
# Note: at inference time, TFT can forecast longer than OUTPUT_LENGTH via autoregression,
# but for initial training the paper's 32-step window is the validated choice.
OUTPUT_LENGTH = 32   # decoder steps = 16h; matches Sinclair et al. 2026

VAL_DAYS = 30        # last N days of training data → validation split


# ─── Feature definitions ────────────────────────────────────────────────────

# Encoder features: past actuals (available at T, the PREDISPATCH run time)
ENC_CONTINUOUS = ["rrp", "total_demand", "net_interchange",
                  "power_load", "power_pv", "temp", "humidity", "wind_speed"]

# Decoder features: PREDISPATCH forecast + time encodings
DEC_CONTINUOUS = ["pd_rrp", "pd_demand", "pd_net_interchange"]

# Time encoding features (appended to both encoder and decoder)
TIME_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                 "month_sin", "month_cos"]

ENC_FEATURES = ENC_CONTINUOUS + TIME_FEATURES   # 14 total
DEC_FEATURES = DEC_CONTINUOUS + TIME_FEATURES   # 9 total


# ─── Time encoding helpers ──────────────────────────────────────────────────

def time_encodings(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos encodings for hour, day-of-week, month."""
    t = timestamps.tz_convert("Australia/Brisbane")
    df = pd.DataFrame(index=timestamps)
    df["hour_sin"] = np.sin(2 * np.pi * t.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * t.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * t.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * t.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (t.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (t.month - 1) / 12)
    return df


# ─── Dataset construction ───────────────────────────────────────────────────

def build_samples(actuals: pd.DataFrame, predispatch: pd.DataFrame,
                  dry_run: bool = False, output_length: int = OUTPUT_LENGTH):
    """
    Build (encoder, decoder, target_raw, run_time) tuples following Option B.

    actuals: indexed by UTC datetime, columns include ENC_CONTINUOUS
    predispatch: columns [interval_dt, run_time, rrp, total_demand, net_interchange]
    """
    # Index actuals by time for fast .loc slicing
    actuals = actuals.set_index("time").sort_index()

    # Build lookup: run_time → DataFrame of sorted forecast rows
    print("  Grouping PREDISPATCH by run_time...")
    pd_grouped = (predispatch
                  .sort_values(["run_time", "interval_dt"])
                  .groupby("run_time", sort=False))

    all_run_times = sorted(predispatch["run_time"].unique())
    print(f"  Total PREDISPATCH runs: {len(all_run_times):,}")

    dt30 = pd.Timedelta(minutes=30)

    enc_list, dec_list, target_list, run_time_list = [], [], [], []
    skipped_enc = skipped_dec = skipped_target = 0

    for i, run_t in enumerate(all_run_times):
        if dry_run and i >= 200:
            break

        # ── Encoder window: actuals[T-48h+30min : T] (96 intervals)
        t_enc_start = run_t - pd.Timedelta(hours=48) + dt30
        t_enc_end   = run_t
        enc_actuals = actuals.loc[t_enc_start:t_enc_end, ENC_CONTINUOUS]

        # Forward-fill small gaps (e.g. missing 1-2 intervals in HA data)
        enc_actuals = enc_actuals.reindex(
            pd.date_range(t_enc_start, t_enc_end, freq="30min", tz="UTC")
        ).ffill().bfill()

        if len(enc_actuals) < INPUT_LENGTH:
            skipped_enc += 1
            continue
        enc_actuals = enc_actuals.iloc[-INPUT_LENGTH:]

        # Add time encodings for encoder
        enc_time = time_encodings(enc_actuals.index)
        enc_full = pd.concat([enc_actuals, enc_time], axis=1)[ENC_FEATURES]

        if enc_full.isna().any().any():
            skipped_enc += 1
            continue

        # ── Decoder window: PREDISPATCH run at T, covering T+30min to T+28h (56 steps)
        try:
            run_df = pd_grouped.get_group(run_t)
        except KeyError:
            skipped_dec += 1
            continue

        dec_intervals = pd.date_range(run_t + dt30, periods=output_length,
                                      freq="30min", tz="UTC")
        run_df = (run_df
                  .set_index("interval_dt")
                  .reindex(dec_intervals)
                  [["rrp", "total_demand", "net_interchange"]])

        if run_df.isna().any().any():
            skipped_dec += 1
            continue

        dec_cont = run_df.rename(columns={
            "rrp": "pd_rrp",
            "total_demand": "pd_demand",
            "net_interchange": "pd_net_interchange",
        })
        dec_time = time_encodings(dec_cont.index)
        dec_full = pd.concat([dec_cont, dec_time], axis=1)[DEC_FEATURES]

        # ── Target: actual RRP for T+30min to T+output_length*30min
        target_actuals = actuals.loc[
            run_t + dt30 : run_t + output_length * dt30, "rrp"
        ]
        target_actuals = target_actuals.reindex(dec_intervals)

        if target_actuals.isna().any():
            skipped_target += 1
            continue

        # All checks passed — append this sample
        enc_list.append(enc_full.values.astype(np.float32))
        dec_list.append(dec_full.values.astype(np.float32))
        target_list.append(target_actuals.values.astype(np.float32))
        run_time_list.append(run_t)

    print(f"  Valid samples: {len(enc_list):,}")
    print(f"  Skipped — encoder gaps: {skipped_enc}, "
          f"decoder missing: {skipped_dec}, target missing: {skipped_target}")

    return (np.stack(enc_list),       # [N, 96, N_ENC]
            np.stack(dec_list),       # [N, 56, N_DEC]
            np.stack(target_list),    # [N, 56]
            np.array(run_time_list, dtype="datetime64[ns]"))


# ─── Train/val split ─────────────────────────────────────────────────────────

def split_by_time(run_times: np.ndarray, val_days: int = VAL_DAYS):
    """Return (train_mask, val_mask) boolean arrays."""
    rts = pd.DatetimeIndex(run_times)
    cutoff = rts.max() - pd.Timedelta(days=val_days)
    train_mask = rts < cutoff
    val_mask   = rts >= cutoff
    return train_mask, val_mask


# ─── Normalisation ───────────────────────────────────────────────────────────

def fit_scalers(X_enc_train, X_dec_train, y_train):
    """
    Fit one QuantileTransformer per continuous feature, on train split only.
    Time encoding features (sin/cos) are left unnormalised.

    Returns dict: feature_name → fitted QuantileTransformer
    """
    scalers = {}

    def fit_feature(name, values_2d):
        """values_2d: [N*T, 1] flattened values."""
        qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal",
                                 random_state=42)
        qt.fit(values_2d.reshape(-1, 1))
        scalers[name] = qt

    # Encoder continuous features
    for j, feat in enumerate(ENC_CONTINUOUS):
        vals = X_enc_train[:, :, j].reshape(-1, 1)
        fit_feature(feat, vals)
        print(f"    Fitted scaler for {feat}: "
              f"p50={np.nanpercentile(vals, 50):.2f}, "
              f"p95={np.nanpercentile(vals, 95):.2f}, "
              f"p99={np.nanpercentile(vals, 99):.2f}")

    # Decoder continuous features
    for j, feat in enumerate(DEC_CONTINUOUS):
        vals = X_dec_train[:, :, j].reshape(-1, 1)
        fit_feature(feat, vals)
        print(f"    Fitted scaler for {feat}: "
              f"p50={np.nanpercentile(vals, 50):.2f}, "
              f"p95={np.nanpercentile(vals, 95):.2f}, "
              f"p99={np.nanpercentile(vals, 99):.2f}")

    # Target (rrp in $/MWh — same distribution as ENC 'rrp' but future steps)
    fit_feature("target_rrp", y_train.reshape(-1, 1))
    print(f"    Fitted scaler for target_rrp: "
          f"p50={np.nanpercentile(y_train, 50):.2f}, "
          f"p99={np.nanpercentile(y_train, 99):.2f}, "
          f"max={np.nanpercentile(y_train, 100):.2f}")

    return scalers


def apply_scalers(X_enc, X_dec, y_raw, scalers):
    """Apply fitted scalers. Time encoding features pass through unchanged."""
    X_enc_norm = X_enc.copy()
    X_dec_norm = X_dec.copy()
    y_norm = y_raw.copy()

    n_enc_cont = len(ENC_CONTINUOUS)
    n_dec_cont = len(DEC_CONTINUOUS)

    for j, feat in enumerate(ENC_CONTINUOUS):
        qt = scalers[feat]
        flat = X_enc_norm[:, :, j].reshape(-1, 1)
        X_enc_norm[:, :, j] = qt.transform(flat).reshape(X_enc_norm[:, :, j].shape)

    for j, feat in enumerate(DEC_CONTINUOUS):
        qt = scalers[feat]
        flat = X_dec_norm[:, :, j].reshape(-1, 1)
        X_dec_norm[:, :, j] = qt.transform(flat).reshape(X_dec_norm[:, :, j].shape)

    qt_target = scalers["target_rrp"]
    y_norm = qt_target.transform(y_raw.reshape(-1, 1)).reshape(y_raw.shape)

    return X_enc_norm, X_dec_norm, y_norm


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build run-aligned TFT training dataset from Parquet files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 200 runs only; print shapes and exit")
    parser.add_argument("--no-normalise", action="store_true",
                        help="Skip normalisation (saves raw arrays for inspection)")
    parser.add_argument("--output-length", type=int, default=OUTPUT_LENGTH,
                        help=f"Decoder steps (default: {OUTPUT_LENGTH}). "
                             "32=16h/97.9%%, 48=24h/64.6%%, 56=28h/47.9%% of runs qualify")
    args = parser.parse_args()

    # Allow CLI override of output length
    out_len = args.output_length
    if out_len != OUTPUT_LENGTH:
        print(f"  [--output-length={out_len}] overriding default of {OUTPUT_LENGTH}")

    print("=== Building training dataset ===")

    # ── Load Parquet files
    print("\nLoading Parquet files...")
    predispatch = pd.read_parquet(PARQUET_DIR / "aemo_predispatch_sa1.parquet")
    actuals     = pd.read_parquet(PARQUET_DIR / "actuals_sa1.parquet")

    # Ensure UTC datetimes
    predispatch["interval_dt"] = pd.to_datetime(predispatch["interval_dt"], utc=True)
    predispatch["run_time"]    = pd.to_datetime(predispatch["run_time"],    utc=True)
    actuals["time"]            = pd.to_datetime(actuals["time"],            utc=True)

    print(f"  PREDISPATCH: {len(predispatch):,} rows, "
          f"{predispatch.run_time.nunique():,} unique run_times")
    print(f"  Actuals: {len(actuals):,} rows, "
          f"{actuals.time.min()} → {actuals.time.max()}")

    # Handle NaN in actuals (forward-fill minor sensor gaps)
    actuals_numeric = actuals.select_dtypes(include=[np.number])
    gap_counts = actuals_numeric.isna().sum()
    if gap_counts.sum() > 0:
        print(f"  Actuals NaN counts (will forward-fill): {gap_counts[gap_counts > 0].to_dict()}")
    for col in ENC_CONTINUOUS:
        if col in actuals.columns:
            actuals[col] = actuals[col].ffill().bfill()

    # ── Build samples
    print(f"\nBuilding run-aligned samples (output_length={out_len})...")
    X_enc, X_dec, y_raw, run_times = build_samples(
        actuals, predispatch, args.dry_run, output_length=out_len)

    print(f"\nDataset shapes:")
    print(f"  X_encoder:  {X_enc.shape}  (N, {INPUT_LENGTH}, {len(ENC_FEATURES)})")
    print(f"  X_decoder:  {X_dec.shape}  (N, {OUTPUT_LENGTH}, {len(DEC_FEATURES)})")
    print(f"  y_targets:  {y_raw.shape}  (N, {OUTPUT_LENGTH})")
    print(f"  run_times:  {run_times.shape}")

    if args.dry_run:
        print("\n[dry-run] Stopping after shape check. Remove --dry-run to build full dataset.")
        return

    # ── Train/val split
    train_mask, val_mask = split_by_time(run_times)
    n_train, n_val = train_mask.sum(), val_mask.sum()
    print(f"\nTrain/val split (last {VAL_DAYS} days = val):")
    print(f"  Train: {n_train:,} samples")
    print(f"  Val:   {n_val:,} samples")

    # ── Normalisation
    if not args.no_normalise:
        print("\nFitting QuantileTransformer scalers on train split...")
        scalers = fit_scalers(X_enc[train_mask], X_dec[train_mask], y_raw[train_mask])

        print("\nApplying scalers to all splits...")
        X_enc_norm, X_dec_norm, y_norm = apply_scalers(X_enc, X_dec, y_raw, scalers)

        # Save scalers
        scalers_path = PARQUET_DIR / "scalers.pkl"
        with open(scalers_path, "wb") as f:
            pickle.dump(scalers, f)
        print(f"  Scalers saved: {scalers_path}")
    else:
        X_enc_norm, X_dec_norm, y_norm = X_enc, X_dec, y_raw
        scalers = {}

    # ── Save arrays
    print("\nSaving arrays...")
    np.save(PARQUET_DIR / "X_encoder.npy",    X_enc_norm)
    np.save(PARQUET_DIR / "X_decoder.npy",    X_dec_norm)
    np.save(PARQUET_DIR / "y_targets.npy",    y_norm)
    np.save(PARQUET_DIR / "y_targets_raw.npy", y_raw)
    np.save(PARQUET_DIR / "run_times.npy",    run_times)

    # Save split indices for reproducibility
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    np.savez(PARQUET_DIR / "split_indices.npz", train=train_idx, val=val_idx)

    # ── Save metadata
    meta = {
        "input_length": INPUT_LENGTH,
        "output_length": out_len,
        "enc_features": ENC_FEATURES,
        "dec_features": DEC_FEATURES,
        "enc_continuous": ENC_CONTINUOUS,
        "dec_continuous": DEC_CONTINUOUS,
        "time_features": TIME_FEATURES,
        "n_enc_features": len(ENC_FEATURES),
        "n_dec_features": len(DEC_FEATURES),
        "n_samples": int(len(run_times)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "val_days": VAL_DAYS,
        "run_time_min": str(pd.Timestamp(run_times.min()).isoformat()),
        "run_time_max": str(pd.Timestamp(run_times.max()).isoformat()),
        "normalised": not args.no_normalise,
    }
    with open(PARQUET_DIR / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Summary
    print("\n=== Dataset build complete ===")
    print(f"  Samples: {len(run_times):,} total ({n_train:,} train, {n_val:,} val)")
    print(f"  Encoder: {X_enc_norm.shape}  ({len(ENC_FEATURES)} features)")
    print(f"  Decoder: {X_dec_norm.shape}  ({len(DEC_FEATURES)} features)")
    print(f"  Target:  {y_norm.shape}")

    if not args.no_normalise:
        print(f"\n  y_targets_raw stats ($/MWh):")
        print(f"    mean={y_raw.mean():.1f}, p50={np.percentile(y_raw, 50):.1f}, "
              f"p95={np.percentile(y_raw, 95):.1f}, max={y_raw.max():.1f}")
        print(f"  y_targets normalised stats:")
        print(f"    mean={y_norm.mean():.3f}, std={y_norm.std():.3f}, "
              f"min={y_norm.min():.3f}, max={y_norm.max():.3f}")

    print(f"\nFiles in {PARQUET_DIR}:")
    all_output_files = (list(PARQUET_DIR.glob("*.npy")) + list(PARQUET_DIR.glob("*.npz")) +
                        list(PARQUET_DIR.glob("*.pkl")) + list(PARQUET_DIR.glob("*.json")))
    for f in sorted(set(all_output_files)):
        print(f"  {f.name}: {f.stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
