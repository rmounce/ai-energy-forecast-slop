#!/usr/bin/env python3
"""
Build run-aligned training dataset for TFT price forecasting.

Implements Option B (run-aligned) covariate construction from Sinclair et al. 2026:
  For each PREDISPATCH run at time T:
    encoder: actuals[T - 48h+30min : T]      → [96,  N_ENC]
    decoder: forecast covariates[T+30min : T+72h] → [144, N_DEC]
    target:  actuals_rrp[T+30min : T+72h]   → [144]
    mask:    valid[T+30min : T+72h]          → [144] bool

Decoder covariate sources (Option B for PREDISPATCH, Option A for PD7Day):
  Steps   1–56  (T+30min to T+28h):  PREDISPATCH run issued AT T
  Steps  57–144 (T+28.5h to T+72h):  Most recent PD7Day run ≤ T
    (PD7Day is 3×/day; nearest run is at most ~8h stale; horizon bias is small
     relative to the 7-day PD7Day horizon — fully acceptable for the long tail.)

Masked loss: mask[h] = 1 where BOTH forecast covariate AND actual target exist.
  - PREDISPATCH steps may be unavailable for runs near the PREDISPATCH horizon cutoff
    (see OUTPUT_LENGTH analysis below; now handled gracefully rather than skipping runs)
  - PD7Day only available from 2026-02-09; earlier runs have mask=0 for steps 57–144
  - Future targets unavailable for most recent runs (~72h from data end)
  - A sample must have ≥ MIN_VALID_STEPS valid steps to be included

Why 144 steps (72h)?
  - 56 steps = PREDISPATCH coverage (28h, 47.9% of runs qualify due to AEMO horizon cycle)
  - 144 steps = adds PD7Day for 28h–72h window
  - With only ~60 days of PD7Day (as of 2026-04-11), ~1,000 samples cover steps 57–144
  - The model learns the short horizon from 15K samples and the long horizon from ~1K,
    improving passively as PD7Day accumulates (weekly retraining)

Decoder feature 'horizon_norm':
  = (h-1)/143, range [0,1] across decoder steps h=1..144
  Sinclair et al. 2026 (SHAP analysis, Table 3) identify "hours to delivery" as the
  second most important decoder feature. It teaches the model which data source to trust
  at each step (PREDISPATCH for small h, PD7Day for large h; zeros for missing data).

References:
  Sinclair, Shepley, Hajati (2026) "Learning the Grid: Transformer Architectures for
  Electricity Price Forecasting in the Australian National Market". Appl. Sci. 16(1), 75.
  Code: github.com/redaxe101/TransformerApplicationNEM

Normalisation:
  QuantileTransformer(n_quantiles=2000, output_distribution='normal') per continuous
  feature, fitted on TRAIN split only. Sinclair et al. confirmed use in training.py.
  Time features (sin/cos, horizon_norm) are not normalised (already bounded).

Outputs (data/parquet/):
  X_encoder.npy        [N, 96,  N_ENC]  — normalised
  X_decoder.npy        [N, 144, N_DEC]  — normalised prices/demand + raw time/horizon
  y_targets.npy        [N, 144]         — normalised actual RRP (zeros where mask=0)
  y_targets_raw.npy    [N, 144]         — raw actual RRP in $/MWh
  y_mask.npy           [N, 144]         — bool: 1 where covariate + target both valid
  run_times.npy        [N]              — int64 nanosecond timestamps of each run_time
  scalers.pkl          — dict: feature_name → fitted QuantileTransformer
  dataset_meta.json    — shapes, feature names, split info
"""

import argparse
import bisect
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"

INPUT_LENGTH    = 96    # encoder steps = 2 days at 30-min resolution
OUTPUT_LENGTH   = 144   # decoder steps = 72h at 30-min resolution
MIN_VALID_STEPS = 32    # minimum valid (masked) steps to include a sample
VAL_DAYS        = 30    # last N days of run_times → validation split

# PREDISPATCH horizon coverage by OUTPUT_LENGTH:
#   32  (16h) → 97.9% of runs — paper's choice (Sinclair et al.)
#   48  (24h) → 64.6% of runs
#   56  (28h) → 47.9% of runs (time-of-day bias if used as hard cutoff)
#   144 (72h) → masked loss handles PREDISPATCH gaps; PD7Day fills 28h–72h
# With masked loss, using OUTPUT_LENGTH=144 means:
#   Steps  1–32:  ~15K samples contribute (97.9% of 17K runs)
#   Steps 33–56:  ~8.5K samples contribute (47.9% of runs, PREDISPATCH coverage)
#   Steps 57–144: ~1K samples contribute (PD7Day backfill window, growing)


# ─── Feature definitions ────────────────────────────────────────────────────

# Encoder: past actuals (available at T)
ENC_CONTINUOUS = ["rrp", "total_demand", "net_interchange",
                  "power_load", "power_pv", "temp", "humidity", "wind_speed"]
TIME_FEATURES  = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                  "month_sin", "month_cos"]
ENC_FEATURES   = ENC_CONTINUOUS + TIME_FEATURES          # 14 features

# Decoder: forecast covariates (PREDISPATCH/PD7Day) + time + horizon
DEC_CONTINUOUS = ["pd_rrp", "pd_demand", "pd_net_interchange"]
DEC_FEATURES   = DEC_CONTINUOUS + TIME_FEATURES + ["horizon_norm"]  # 10 features


# ─── Time encoding helpers ──────────────────────────────────────────────────

def time_encodings(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos encodings for hour, day-of-week, month."""
    t = timestamps.tz_convert("Australia/Brisbane")
    df = pd.DataFrame(index=timestamps)
    df["hour_sin"]   = np.sin(2 * np.pi * t.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * t.hour / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * t.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * t.dayofweek / 7)
    df["month_sin"]  = np.sin(2 * np.pi * (t.month - 1) / 12)
    df["month_cos"]  = np.cos(2 * np.pi * (t.month - 1) / 12)
    return df


# ─── Dataset construction ───────────────────────────────────────────────────

def build_samples(actuals: pd.DataFrame,
                  predispatch: pd.DataFrame,
                  pd7day: pd.DataFrame,
                  dry_run: bool = False,
                  output_length: int = OUTPUT_LENGTH,
                  min_valid_steps: int = MIN_VALID_STEPS):
    """
    Build (encoder, decoder, target_raw, mask, run_time) tuples using masked loss.

    actuals:     columns include ENC_CONTINUOUS; indexed by UTC datetime after set_index
    predispatch: columns [interval_dt, run_time, rrp, total_demand, net_interchange]
    pd7day:      columns [interval_dt, run_time, rrp]
    """
    actuals = actuals.set_index("time").sort_index()

    print("  Grouping PREDISPATCH by run_time...")
    pd_grouped = (predispatch
                  .sort_values(["run_time", "interval_dt"])
                  .groupby("run_time", sort=False))

    # PD7Day lookup: sorted list of run_times for bisect
    pd7day_run_times_sorted = sorted(pd7day["run_time"].unique())
    print(f"  Grouping PD7Day by run_time ({len(pd7day_run_times_sorted)} runs)...")
    pd7_grouped = (pd7day
                   .sort_values(["run_time", "interval_dt"])
                   .groupby("run_time", sort=False))

    all_run_times = sorted(predispatch["run_time"].unique())
    print(f"  Total PREDISPATCH runs: {len(all_run_times):,}")

    dt30      = pd.Timedelta(minutes=30)
    all_steps = pd.RangeIndex(output_length)     # 0..143

    (enc_list, dec_list, target_list,
     mask_list, covar_mask_list, run_time_list) = [], [], [], [], [], []
    stats = dict(skipped_encoder=0, skipped_min_valid=0, total=0)

    for i, run_t in enumerate(all_run_times):
        if dry_run and i >= 200:
            break
        stats["total"] += 1

        # ── Encoder: actuals[T-48h+30min : T] (96 steps)
        t_enc_start = run_t - pd.Timedelta(hours=48) + dt30
        enc_actuals = actuals.loc[t_enc_start:run_t, ENC_CONTINUOUS]
        enc_actuals = enc_actuals.reindex(
            pd.date_range(t_enc_start, run_t, freq="30min", tz="UTC")
        ).ffill().bfill()

        if len(enc_actuals) < INPUT_LENGTH or enc_actuals.isna().any().any():
            stats["skipped_encoder"] += 1
            continue
        enc_actuals = enc_actuals.iloc[-INPUT_LENGTH:]
        enc_time = time_encodings(enc_actuals.index)
        enc_full = pd.concat([enc_actuals, enc_time], axis=1)[ENC_FEATURES]

        # ── Decoder: 144-step window with per-step mask
        dec_intervals = pd.date_range(run_t + dt30, periods=output_length,
                                      freq="30min", tz="UTC")
        dec_arr  = np.zeros((output_length, len(DEC_FEATURES)), dtype=np.float32)
        mask_arr = np.zeros(output_length, dtype=bool)
        targ_arr = np.zeros(output_length, dtype=np.float32)

        # Time encodings + horizon_norm are always valid (constant for each step)
        dec_time_enc = time_encodings(dec_intervals)  # [144, 6]
        horizon_norm = np.arange(output_length, dtype=np.float32) / max(output_length - 1, 1)  # [144]

        # Fill time + horizon into decoder array (features DEC_CONTINUOUS onwards)
        n_cont = len(DEC_CONTINUOUS)
        dec_arr[:, n_cont:n_cont + 6] = dec_time_enc.values
        dec_arr[:, n_cont + 6]        = horizon_norm

        # -- PREDISPATCH (indices 0..55 = steps h=1..56)
        try:
            pd_run = pd_grouped.get_group(run_t)
            pd_sub = (pd_run.set_index("interval_dt")
                      .reindex(dec_intervals[:56])
                      [["rrp", "total_demand", "net_interchange"]])
            valid_pd = ~pd_sub["rrp"].isna()
            dec_arr[:56, 0] = pd_sub["rrp"].fillna(0.0).values
            dec_arr[:56, 1] = pd_sub["total_demand"].fillna(0.0).values
            dec_arr[:56, 2] = pd_sub["net_interchange"].fillna(0.0).values
            mask_arr[:56]   = valid_pd.values
        except KeyError:
            pass   # no PREDISPATCH run at T — indices 0..55 stay masked

        # -- PD7Day (indices 56..143 = steps h=57..144)
        bisect_idx = bisect.bisect_right(pd7day_run_times_sorted, run_t) - 1
        if bisect_idx >= 0:
            pd7day_run_t = pd7day_run_times_sorted[bisect_idx]
            try:
                pd7_run = pd7_grouped.get_group(pd7day_run_t)
                pd7_sub = (pd7_run.set_index("interval_dt")
                           .reindex(dec_intervals[56:])["rrp"])
                valid_pd7 = ~pd7_sub.isna()
                dec_arr[56:, 0] = pd7_sub.fillna(0.0).values
                mask_arr[56:]   = valid_pd7.values
            except KeyError:
                pass

        covar_mask = mask_arr.copy()

        # -- Targets: actual RRP (sets mask to False where actuals unavailable)
        target_actual = actuals.loc[
            run_t + dt30 : run_t + output_length * dt30, "rrp"
        ].reindex(dec_intervals)
        for hi in range(output_length):
            v = target_actual.iloc[hi] if hi < len(target_actual) else np.nan
            if pd.isna(v):
                mask_arr[hi] = False  # covariate may exist but no target yet
            else:
                targ_arr[hi] = float(v)
                # mask_arr[hi] stays True only if covariate was also set above

        valid_count = mask_arr.sum()
        if valid_count < min_valid_steps:
            stats["skipped_min_valid"] += 1
            continue

        enc_list.append(enc_full.values.astype(np.float32))
        dec_list.append(dec_arr)
        target_list.append(targ_arr)
        mask_list.append(mask_arr)
        covar_mask_list.append(covar_mask)
        run_time_list.append(run_t)

    print(f"  Valid samples:  {len(enc_list):,}")
    print(f"  Skipped — encoder gaps:   {stats['skipped_encoder']}")
    print(f"  Skipped — <{min_valid_steps} valid steps: {stats['skipped_min_valid']}")

    X_enc  = np.stack(enc_list)     # [N, 96,  N_ENC]
    X_dec  = np.stack(dec_list)     # [N, 144, N_DEC]
    y_raw  = np.stack(target_list)  # [N, 144]
    y_mask = np.stack(mask_list)    # [N, 144] bool
    y_covar_mask = np.stack(covar_mask_list) # [N, 144] bool

    # Print coverage stats
    cov_by_step = y_mask.mean(axis=0)
    print(f"  Mask coverage by horizon:")
    print(f"    Steps  1–32  (16h):  {cov_by_step[:32].mean():.1%}")
    print(f"    Steps  1–56  (28h):  {cov_by_step[:56].mean():.1%}")
    print(f"    Steps 57–144 (72h):  {cov_by_step[56:].mean():.1%}")

    return (X_enc,
            X_dec,
            y_raw,
            y_mask,
            y_covar_mask,
            np.array(run_time_list, dtype="datetime64[ns]"))


# ─── Train/val split ─────────────────────────────────────────────────────────

def split_by_time(run_times: np.ndarray, val_days: int = VAL_DAYS, train_gap_hours: int = 72):
    rts = pd.DatetimeIndex(run_times)
    cutoff_val = rts.max() - pd.Timedelta(days=val_days)
    cutoff_train = cutoff_val - pd.Timedelta(hours=train_gap_hours)
    return rts < cutoff_train, rts >= cutoff_val


# ─── Normalisation ───────────────────────────────────────────────────────────

def fit_scalers(X_enc_train, X_dec_train, y_train, y_mask_train, y_covar_mask_train):
    """
    Fit QuantileTransformer per continuous feature on train split only.
    Time encoding and horizon_norm features are NOT normalised (already bounded).
    """
    scalers = {}

    def fit_one(name, values):
        qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal",
                                 random_state=42)
        qt.fit(values.reshape(-1, 1))
        scalers[name] = qt
        v = values
        print(f"    {name}: p50={np.nanpercentile(v, 50):.2f}, "
              f"p95={np.nanpercentile(v, 95):.2f}, max={np.nanmax(v):.2f}")

    for j, feat in enumerate(ENC_CONTINUOUS):
        fit_one(feat, X_enc_train[:, :, j])
    for j, feat in enumerate(DEC_CONTINUOUS):
        valid_covars = X_dec_train[:, :, j][y_covar_mask_train]
        fit_one(feat, valid_covars)

    # Target: fit on VALID train steps only to avoid fitting on zero-padded values
    valid_targets = y_train[y_mask_train]
    fit_one("target_rrp", valid_targets)

    return scalers


def apply_scalers(X_enc, X_dec, y_raw, y_mask, y_covar_mask, scalers):
    """Apply fitted scalers. Non-continuous features pass through unchanged."""
    X_enc_n = X_enc.copy()
    X_dec_n = X_dec.copy()
    y_norm  = np.zeros_like(y_raw)

    for j, feat in enumerate(ENC_CONTINUOUS):
        flat = X_enc_n[:, :, j].reshape(-1, 1)
        X_enc_n[:, :, j] = scalers[feat].transform(flat).reshape(X_enc_n[:, :, j].shape)

    for j, feat in enumerate(DEC_CONTINUOUS):
        view = X_dec_n[:, :, j]
        valid_flat = view[y_covar_mask].reshape(-1, 1)
        view[y_covar_mask] = scalers[feat].transform(valid_flat).reshape(-1)

    qt = scalers["target_rrp"]
    # Only transform valid targets; leave zero-padded positions as 0
    valid_flat = y_raw[y_mask].reshape(-1, 1)
    y_norm_valid = qt.transform(valid_flat).reshape(-1)
    y_norm[y_mask] = y_norm_valid

    return X_enc_n, X_dec_n, y_norm


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build run-aligned TFT training dataset (144-step / 72h)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 200 runs; print shapes and exit")
    parser.add_argument("--no-normalise", action="store_true",
                        help="Skip normalisation (saves raw arrays for inspection)")
    parser.add_argument("--output-length", type=int, default=OUTPUT_LENGTH,
                        help=f"Decoder steps (default: {OUTPUT_LENGTH} = 72h)")
    parser.add_argument("--min-valid-steps", type=int, default=MIN_VALID_STEPS,
                        help=f"Minimum valid masked steps to include a sample (default: {MIN_VALID_STEPS})")
    args = parser.parse_args()
    out_len = args.output_length

    print("=== Building training dataset ===")
    print(f"  output_length={out_len}, min_valid_steps={args.min_valid_steps}")

    # ── Load Parquet files
    print("\nLoading Parquet files...")
    predispatch = pd.read_parquet(PARQUET_DIR / "aemo_predispatch_sa1.parquet")
    pd7day      = pd.read_parquet(PARQUET_DIR / "aemo_pd7day_sa1.parquet")
    actuals     = pd.read_parquet(PARQUET_DIR / "actuals_sa1.parquet")

    for df in [predispatch, pd7day]:
        for col in ["interval_dt", "run_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
    actuals["time"] = pd.to_datetime(actuals["time"], utc=True)

    print(f"  PREDISPATCH: {len(predispatch):,} rows, "
          f"{predispatch.run_time.nunique():,} unique run_times")
    print(f"  PD7Day:      {len(pd7day):,} rows, "
          f"{pd7day.run_time.nunique():,} unique run_times "
          f"({pd7day.run_time.min().date()} → {pd7day.run_time.max().date()})")
    print(f"  Actuals:     {len(actuals):,} rows, "
          f"{actuals.time.min()} → {actuals.time.max()}")

    # Forward-fill minor actuals gaps
    gap_counts = actuals[ENC_CONTINUOUS].isna().sum()
    if gap_counts.sum() > 0:
        print(f"  Actuals NaN counts (forward-filling): "
              f"{gap_counts[gap_counts > 0].to_dict()}")
    for col in ENC_CONTINUOUS:
        if col in actuals.columns:
            actuals[col] = actuals[col].ffill().bfill()

    # ── Build samples
    print("\nBuilding run-aligned samples...")
    X_enc, X_dec, y_raw, y_mask, y_covar_mask, run_times = build_samples(
        actuals, predispatch, pd7day,
        dry_run=args.dry_run,
        output_length=out_len,
        min_valid_steps=args.min_valid_steps,
    )

    print(f"\nDataset shapes:")
    print(f"  X_encoder: {X_enc.shape}  (N, {INPUT_LENGTH}, {len(ENC_FEATURES)} enc features)")
    print(f"  X_decoder: {X_dec.shape}  (N, {out_len}, {len(DEC_FEATURES)} dec features)")
    print(f"  y_targets: {y_raw.shape}")
    print(f"  y_mask:    {y_mask.shape}, total valid steps: {y_mask.sum():,}")

    if args.dry_run:
        print("\n[dry-run] Stopping. Remove --dry-run to build full dataset.")
        return

    # ── Train/val split
    train_mask, val_mask = split_by_time(run_times)
    n_train, n_val = train_mask.sum(), val_mask.sum()
    print(f"\nTrain/val split (last {VAL_DAYS} days = val):")
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    # ── Normalisation
    if not args.no_normalise:
        print("\nFitting scalers on train split...")
        scalers = fit_scalers(X_enc[train_mask], X_dec[train_mask],
                              y_raw[train_mask], y_mask[train_mask], y_covar_mask[train_mask])
        print("\nApplying scalers...")
        X_enc_n, X_dec_n, y_norm = apply_scalers(X_enc, X_dec, y_raw, y_mask, y_covar_mask, scalers)
        with open(PARQUET_DIR / "scalers.pkl", "wb") as f:
            pickle.dump(scalers, f)
        print(f"  Scalers saved: {PARQUET_DIR}/scalers.pkl")
    else:
        X_enc_n, X_dec_n, y_norm = X_enc, X_dec, y_raw
        scalers = {}

    # ── Save arrays
    print("\nSaving arrays...")
    np.save(PARQUET_DIR / "X_encoder.npy",    X_enc_n)
    np.save(PARQUET_DIR / "X_decoder.npy",    X_dec_n)
    np.save(PARQUET_DIR / "y_targets.npy",    y_norm)
    np.save(PARQUET_DIR / "y_targets_raw.npy", y_raw)
    np.save(PARQUET_DIR / "y_mask.npy",       y_mask)
    np.save(PARQUET_DIR / "y_covar_mask.npy", y_covar_mask)
    np.save(PARQUET_DIR / "run_times.npy",    run_times)
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    np.savez(PARQUET_DIR / "split_indices.npz", train=train_idx, val=val_idx)

    # ── Save metadata
    meta = {
        "input_length": INPUT_LENGTH,
        "output_length": out_len,
        "min_valid_steps": args.min_valid_steps,
        "enc_features": ENC_FEATURES,
        "dec_features": DEC_FEATURES,
        "enc_continuous": ENC_CONTINUOUS,
        "dec_continuous": DEC_CONTINUOUS,
        "n_enc_features": len(ENC_FEATURES),
        "n_dec_features": len(DEC_FEATURES),
        "n_samples": int(len(run_times)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "val_days": VAL_DAYS,
        "normalised": not args.no_normalise,
        "mask_coverage": {
            "steps_1_32":   float(y_mask[:, :32].mean()),
            "steps_1_56":   float(y_mask[:, :56].mean()),
            "steps_57_144": float(y_mask[:, 56:].mean()),
        },
    }
    with open(PARQUET_DIR / "dataset_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    valid_rrp = y_raw[y_mask]
    print(f"\n=== Dataset build complete ===")
    print(f"  {len(run_times):,} samples ({n_train:,} train, {n_val:,} val)")
    print(f"  Mask coverage: 1–32h {meta['mask_coverage']['steps_1_32']:.1%}, "
          f"1–28h {meta['mask_coverage']['steps_1_56']:.1%}, "
          f"28–72h {meta['mask_coverage']['steps_57_144']:.1%}")
    print(f"  Valid RRP stats ($/MWh): mean={valid_rrp.mean():.1f}, "
          f"p50={np.percentile(valid_rrp, 50):.1f}, max={valid_rrp.max():.1f}")


if __name__ == "__main__":
    main()
