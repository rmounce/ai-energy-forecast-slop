#!/usr/bin/env python3
"""
Build run-aligned training dataset for TFT price forecasting.

Implements Option B (run-aligned) covariate construction from Sinclair et al. 2026:
  For each PREDISPATCH run at time T:
    encoder: actuals[T - 48h+30min : T]      → [96,  N_ENC]
    decoder: forecast covariates[T+30min : T+72h] → [144, N_DEC]
    target:  actuals_rrp[T+30min : T+72h]   → [144]
    mask:    valid[T+30min : T+72h]          → [144] bool

Decoder covariate sources — Phase 7 Enhanced Input (parallel signals):
  pd_rrp / pd_demand / pd_net_interchange / vic1_pd_rrp / nsw1_pd_rrp:
    Steps   1–56  (T+30min to T+28h):  PREDISPATCH run issued AT T
    Steps  57–144 (T+28.5h to T+72h):  0-filled (PREDISPATCH-only features)
  pd7_rrp:
    Steps   1–144 (T+30min to T+72h):  Most recent PD7Day run ≤ T (all steps)
    Provides a continuous price baseline signal even where PREDISPATCH is absent.
  predispatch_active:
    1 where PREDISPATCH data is present for that step, 0 otherwise.
    Teaches the TFT when to trust pd_rrp vs fall back to pd7_rrp.
  pd7_generation_hour:
    Hour of day the PD7Day run was published (7, 12, or 18 AEST → normalised /23).
    Encodes diurnal bias between the three daily PD7Day publications.
  pd7_available:
    1 if a PD7Day run existed at or before T, 0 otherwise (pre-2026-02 training data).

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
  X_encoder.npy        [N, 96,  N_ENC]  — normalised base(8) + 5m(3) + time(6) + missing_flag(1) = 18
  X_decoder.npy        [N, 144, N_DEC]  — normalised prices/demand (SA1+VIC1+NSW1+SDO) + raw time/horizon
  y_targets.npy        [N, 144]         — normalised actual RRP (zeros where mask=0)
  y_targets_raw.npy    [N, 144]         — raw actual RRP in $/MWh
  y_mask.npy           [N, 144]         — bool: 1 where covariate + target both valid
  sdo_covar_mask.npy   [N, 144]         — bool: 1 where SevenDayOutlook data was available
  run_times.npy        [N]              — int64 nanosecond timestamps of each run_time
  scalers.pkl          — dict: feature_name → fitted QuantileTransformer
  dataset_meta.json    — shapes, feature names, split info

Run 011 changes:
  - pd_rrp (steps 0–55) substituted with OOF-debiased PREDISPATCH RRP where available
    (raw pd_rrp kept as fallback when OOF data absent for a run)
  - sd_demand + sd_net_interchange added as decoder features (SevenDayOutlook, all 144 steps)
    Scalers fitted on SDO-valid steps only; 0-filled where SDO unavailable (pre-2025-03)
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
VAL_DAYS        = 60    # last N days of run_times → validation split

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
ENC_CONT_BASE  = ["rrp", "total_demand", "net_interchange",
                  "power_load", "power_pv", "temp", "humidity", "wind_speed"]

# 5-min volatility aggregates — only available from ~2025-03-30; NaN-filled earlier.
# Steps where data is absent get rrp_5m_missing=1; the 3 values are 0-filled.
ENC_5M_CONTINUOUS   = ["rrp_5m_max", "rrp_5m_std", "rrp_persistence", "rrp_volatility_30m"]
ENC_5M_FEATURES_SET = set(ENC_5M_CONTINUOUS)

ENC_CONTINUOUS = ENC_CONT_BASE + ENC_5M_CONTINUOUS + ["rrp_log_momentum"]  # 13 scaled features
TIME_FEATURES  = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                  "month_sin", "month_cos"]
ENC_FEATURES   = ENC_CONTINUOUS + TIME_FEATURES + ["rrp_5m_missing"]   # 20 features
ENC_5M_MISSING_IDX = len(ENC_CONTINUOUS) + len(TIME_FEATURES)          # index of rrp_5m_missing = 19

LOG_SCALE_FACTOR = 60.0  # reference price for log-scaling: log1p(x/60)

# Decoder: forecast covariates (PREDISPATCH/PD7Day/SDO) + time + horizon
# Phase 7: pd7_rrp added as parallel PD7Day signal across all 144 steps.
# pd_rrp is now PREDISPATCH-only (0-filled steps 56-143); pd7_rrp carries PD7Day.
DEC_CONTINUOUS = ["pd_rrp", "pd_demand", "pd_net_interchange",
                  "vic1_pd_rrp", "nsw1_pd_rrp",
                  "pd7_rrp",                        # NEW: PD7Day rrp, all 144 steps
                  "sd_demand", "sd_net_interchange"]
DEC_FEATURES   = DEC_CONTINUOUS + TIME_FEATURES + [
    "horizon_norm",
    "predispatch_active",   # 1 where PREDISPATCH present (replaces covar_missing, flipped)
    "pd7_generation_hour",  # PD7Day run hour normalised /23 (0 if no PD7Day)
    "pd7_available",        # 1 if a PD7Day run existed at T, 0 for pre-2026-02 samples
]  # 18 features total

# VIC1/NSW1 data only covers steps 0-55 (PREDISPATCH horizon); steps 56-143 are 0-filled.
ADJ_REGION_FEATURES = {"vic1_pd_rrp", "nsw1_pd_rrp"}
# SDO features available from ~2025-03-22; 0-filled for earlier runs.
SDO_FEATURES        = {"sd_demand", "sd_net_interchange"}
# PD7Day price feature: log-scaled like pd_rrp; fitted on PD7Day-available steps only.
PD7_FEATURES        = {"pd7_rrp"}


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
                  vic1_pd: pd.DataFrame | None = None,
                  nsw1_pd: pd.DataFrame | None = None,
                  sdo: pd.DataFrame | None = None,
                  oof_grouped=None,
                  dry_run: bool = False,
                  output_length: int = OUTPUT_LENGTH,
                  min_valid_steps: int = MIN_VALID_STEPS):
    """
    Build (encoder, decoder, target_raw, mask, run_time) tuples using masked loss.

    actuals:     columns include ENC_CONTINUOUS; indexed by UTC datetime after set_index
    predispatch: columns [interval_dt, run_time, rrp, total_demand, net_interchange]
    pd7day:      columns [interval_dt, run_time, rrp]
    vic1_pd:     columns [interval_dt, run_time, rrp] — adjacent region, steps 0-55 only
    nsw1_pd:     columns [interval_dt, run_time, rrp] — adjacent region, steps 0-55 only
    sdo:         columns [interval_dt, run_time, scheduled_demand, net_interchange]
                 SevenDayOutlook — fills sd_demand/sd_net_interchange for all 144 steps
    oof_grouped: GroupBy object from debiased_pd_rrp_oof.parquet; substitutes raw pd_rrp
                 at steps 0-55 where OOF values are available (raw pd_rrp kept as fallback)
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

    # Adjacent-region PREDISPATCH (VIC1/NSW1) — steps 0-55 only; None → all 0-filled
    adj_grouped = {}
    for name, df in (("vic1", vic1_pd), ("nsw1", nsw1_pd)):
        if df is not None:
            adj_grouped[name] = (df.sort_values(["run_time", "interval_dt"])
                                   .groupby("run_time", sort=False))
            print(f"  Grouping {name.upper()} PREDISPATCH by run_time "
                  f"({df.run_time.nunique():,} runs)...")
        else:
            adj_grouped[name] = None

    # SevenDayOutlook lookup (bisect on run_times, like PD7Day)
    if sdo is not None:
        sdo_run_times_sorted = sorted(sdo["run_time"].unique())
        sdo_grouped_by_rt = sdo.groupby("run_time", sort=False)
        print(f"  Grouping SevenDayOutlook by run_time ({len(sdo_run_times_sorted)} runs, "
              f"{sdo['run_time'].min().date()} → {sdo['run_time'].max().date()})...")
    else:
        sdo_run_times_sorted = []
        sdo_grouped_by_rt = None
        print(f"  WARNING: sdo=None — sd_demand/sd_net_interchange will be 0-filled")

    # Column indices for new SDO features
    sd_demand_idx = DEC_CONTINUOUS.index("sd_demand")
    sd_ni_idx     = DEC_CONTINUOUS.index("sd_net_interchange")

    all_run_times = sorted(predispatch["run_time"].unique())
    print(f"  Total PREDISPATCH runs: {len(all_run_times):,}")

    dt30      = pd.Timedelta(minutes=30)
    all_steps = pd.RangeIndex(output_length)     # 0..143

    (enc_list, dec_list, target_list,
     mask_list, covar_mask_list, sdo_covar_mask_list, run_time_list) = [], [], [], [], [], [], []
    stats = dict(skipped_encoder=0, skipped_min_valid=0, total=0)

    for i, run_t in enumerate(all_run_times):
        if dry_run and i >= 200:
            break
        stats["total"] += 1

        # ── Encoder: actuals[T-48h+30min : T] (96 steps)
        t_enc_start = run_t - pd.Timedelta(hours=48) + dt30
        enc_grid = pd.date_range(t_enc_start, run_t, freq="30min", tz="UTC")

        # Base actuals (ffill gaps — price/demand/weather)
        enc_base = (actuals.loc[t_enc_start:run_t, ENC_CONT_BASE]
                    .reindex(enc_grid).ffill().bfill())
        if len(enc_base) < INPUT_LENGTH or enc_base.isna().any().any():
            stats["skipped_encoder"] += 1
            continue
        enc_base = enc_base.iloc[-INPUT_LENGTH:]

        # 5m volatility features (NO ffill — absent before 5m era; flagged explicitly)
        enc_5m = (actuals.loc[t_enc_start:run_t, ENC_5M_CONTINUOUS]
                  .reindex(enc_grid))  # NaN where 5m data unavailable
        missing_5m = enc_5m.isna().any(axis=1).values[-INPUT_LENGTH:]  # [96] bool
        enc_5m = enc_5m.fillna(0.0).iloc[-INPUT_LENGTH:]

        enc_time = time_encodings(enc_base.index)

        # Add log-momentum and volatility features
        # Log-momentum: slope of log-price over last 4 steps (2h)
        rrp_raw = enc_base["rrp"].values.astype(np.float32)
        rrp_log = np.log1p(np.maximum(0, rrp_raw) / LOG_SCALE_FACTOR)
        # Simple slope estimate: current - 4 steps ago
        log_mom = np.zeros_like(rrp_log)
        log_mom[4:] = rrp_log[4:] - rrp_log[:-4]

        # Assemble [96, 20]: base(8) + 5m(4) + log_mom(1) + time(6) + missing_flag(1)
        enc_full = np.concatenate([
            enc_base.values.astype(np.float32),                      # [96, 8]
            enc_5m.values.astype(np.float32),                        # [96, 4]
            log_mom.reshape(-1, 1),                                  # [96, 1]
            enc_time.values.astype(np.float32),                      # [96, 6]
            missing_5m.astype(np.float32).reshape(-1, 1),            # [96, 1]
        ], axis=1)

        # ── Decoder: 144-step window with per-step mask
        dec_intervals = pd.date_range(run_t + dt30, periods=output_length,
                                      freq="30min", tz="UTC")
        dec_arr      = np.zeros((output_length, len(DEC_FEATURES)), dtype=np.float32)
        mask_arr     = np.zeros(output_length, dtype=bool)
        sdo_mask_arr = np.zeros(output_length, dtype=bool)
        targ_arr     = np.zeros(output_length, dtype=np.float32)

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

        # -- OOF debiased pd_rrp substitution (indices 0..55 = steps h=1..56)
        # Replaces raw pd_rrp with the OOF-debiased value where available.
        # Fallback: raw pd_rrp kept if run_t not in OOF (e.g. edge of training window).
        if oof_grouped is not None:
            try:
                oof_run = oof_grouped.get_group(run_t)
                oof_sub = (oof_run.set_index("interval_dt")
                           .reindex(dec_intervals[:56])["oof_debiased_rrp"])
                valid_oof = ~oof_sub.isna()
                dec_arr[:56, 0] = np.where(
                    valid_oof.values,
                    oof_sub.fillna(0.0).values,
                    dec_arr[:56, 0],   # keep raw pd_rrp where OOF absent
                )
            except KeyError:
                pass   # no OOF for this run — raw pd_rrp stays

        # -- VIC1/NSW1 PREDISPATCH (indices 0..55 = steps h=1..56, rrp only)
        for adj_name, adj_col_idx in (("vic1", DEC_CONTINUOUS.index("vic1_pd_rrp")),
                                       ("nsw1", DEC_CONTINUOUS.index("nsw1_pd_rrp"))):
            grp = adj_grouped[adj_name]
            if grp is None:
                continue
            try:
                adj_run = grp.get_group(run_t)
                adj_sub = (adj_run.set_index("interval_dt")
                           .reindex(dec_intervals[:56])["rrp"])
                dec_arr[:56, adj_col_idx] = adj_sub.fillna(0.0).values
            except KeyError:
                pass   # no matching run — stays 0

        # -- PD7Day: pd7_rrp for ALL 144 steps (parallel to pd_rrp; pd_rrp stays 0 for 56-143)
        pd7_rrp_idx    = DEC_CONTINUOUS.index("pd7_rrp")
        pd7_gen_hr_idx = len(DEC_CONTINUOUS) + len(TIME_FEATURES) + 2  # after horizon_norm+predispatch_active
        pd7_avail_idx  = len(DEC_CONTINUOUS) + len(TIME_FEATURES) + 3
        bisect_idx = bisect.bisect_right(pd7day_run_times_sorted, run_t) - 1
        if bisect_idx >= 0:
            pd7day_run_t = pd7day_run_times_sorted[bisect_idx]
            try:
                pd7_run = pd7_grouped.get_group(pd7day_run_t)
                pd7_sub = (pd7_run.set_index("interval_dt")
                           .reindex(dec_intervals)["rrp"])   # all 144 steps
                valid_pd7 = ~pd7_sub.isna()
                dec_arr[:, pd7_rrp_idx] = pd7_sub.fillna(0.0).values
                mask_arr[56:]           = valid_pd7.values[56:]  # PREDISPATCH mask owns 0-55
                gen_hour = pd7day_run_t.tz_convert("Australia/Brisbane").hour
                dec_arr[:, pd7_gen_hr_idx] = np.float32(gen_hour) / 23.0
                dec_arr[:, pd7_avail_idx]  = np.float32(1.0)
            except KeyError:
                pass
        # pd7_available=0 and pd7_generation_hour=0 stay at default for pre-PD7Day samples

        # -- SevenDayOutlook (all 144 steps): sd_demand + sd_net_interchange
        # Uses most recent SDO run ≤ run_t (same bisect pattern as PD7Day).
        # 0-filled + sdo_mask_arr=False where SDO unavailable (pre-2025-03).
        if sdo_grouped_by_rt is not None and sdo_run_times_sorted:
            bisect_sdo = bisect.bisect_right(sdo_run_times_sorted, run_t) - 1
            if bisect_sdo >= 0:
                sdo_run_t = sdo_run_times_sorted[bisect_sdo]
                try:
                    sdo_run = sdo_grouped_by_rt.get_group(sdo_run_t)
                    sdo_sub = (sdo_run.set_index("interval_dt")
                               .reindex(dec_intervals)
                               [["scheduled_demand", "net_interchange"]])
                    valid_sdo = ~sdo_sub["scheduled_demand"].isna()
                    dec_arr[:, sd_demand_idx] = sdo_sub["scheduled_demand"].fillna(0.0).values
                    dec_arr[:, sd_ni_idx]     = sdo_sub["net_interchange"].fillna(0.0).values
                    sdo_mask_arr[:] = valid_sdo.values
                except KeyError:
                    pass   # no SDO run ≤ run_t (shouldn't happen after 2025-03)

        covar_mask = mask_arr.copy()
        dec_arr[:, n_cont + 7] = covar_mask.astype(np.float32)  # predispatch_active (1=present)

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

        enc_list.append(enc_full)
        dec_list.append(dec_arr)
        target_list.append(targ_arr)
        mask_list.append(mask_arr)
        covar_mask_list.append(covar_mask)
        sdo_covar_mask_list.append(sdo_mask_arr)
        run_time_list.append(run_t)

    print(f"  Valid samples:  {len(enc_list):,}")
    print(f"  Skipped — encoder gaps:   {stats['skipped_encoder']}")
    print(f"  Skipped — <{min_valid_steps} valid steps: {stats['skipped_min_valid']}")

    X_enc         = np.stack(enc_list)          # [N, 96,  N_ENC]
    X_dec         = np.stack(dec_list)          # [N, 144, N_DEC]
    y_raw         = np.stack(target_list)       # [N, 144]
    y_mask        = np.stack(mask_list)         # [N, 144] bool
    y_covar_mask  = np.stack(covar_mask_list)   # [N, 144] bool
    sdo_covar_mask = np.stack(sdo_covar_mask_list)  # [N, 144] bool

    # Print coverage stats
    cov_by_step = y_mask.mean(axis=0)
    print(f"  Mask coverage by horizon:")
    print(f"    Steps  1–32  (16h):  {cov_by_step[:32].mean():.1%}")
    print(f"    Steps  1–56  (28h):  {cov_by_step[:56].mean():.1%}")
    print(f"    Steps 57–144 (72h):  {cov_by_step[56:].mean():.1%}")
    print(f"  SDO coverage: {sdo_covar_mask.mean():.1%} of all steps "
          f"({sdo_covar_mask[:, :56].mean():.1%} PREDISPATCH, "
          f"{sdo_covar_mask[:, 56:].mean():.1%} PD7Day)")
    pd7_rrp_col = DEC_CONTINUOUS.index("pd7_rrp")
    pd7_cov = (X_dec[:, :, pd7_rrp_col] != 0.0).mean()
    pd7_avail_frac = X_dec[:, 0, DEC_FEATURES.index("pd7_available")].mean()
    print(f"  PD7Day coverage: {pd7_avail_frac:.1%} of samples have PD7Day; "
          f"{pd7_cov:.1%} of all decoder steps non-zero")

    return (X_enc,
            X_dec,
            y_raw,
            y_mask,
            y_covar_mask,
            sdo_covar_mask,
            np.array(run_time_list, dtype="datetime64[ns]"))


# ─── Train/val split ─────────────────────────────────────────────────────────

def split_by_time(run_times: np.ndarray, val_days: int = VAL_DAYS, train_gap_hours: int = 72):
    rts = pd.DatetimeIndex(run_times)
    cutoff_val = rts.max() - pd.Timedelta(days=val_days)
    cutoff_train = cutoff_val - pd.Timedelta(hours=train_gap_hours)
    return rts < cutoff_train, rts >= cutoff_val


# ─── Normalisation ───────────────────────────────────────────────────────────

def fit_scalers(X_enc_train, X_dec_train, y_train, y_mask_train, y_covar_mask_train,
                sdo_covar_mask_train=None, target_scaling="quantile"):
    """
    Fit QuantileTransformer per continuous feature on train split only.
    If target_scaling='log', rrp features use log-scaling instead of QuantileTransformer.
    Time encoding and horizon_norm features are NOT normalised (already bounded).
    """
    scalers = {}

    def fit_one(name, values):
        # Use log-scaling for rrp and targets if requested
        is_rrp = any(x in name for x in ["rrp", "target_rrp", "pd_rrp", "pd7_rrp"])
        if target_scaling == "log" and is_rrp:
            scalers[name] = "log"
            v = values
            print(f"    {name}: [log-scaling] p50={np.nanpercentile(v, 50):.2f}, "
                  f"p95={np.nanpercentile(v, 95):.2f}, max={np.nanmax(v):.2f}")
            return

        qt = QuantileTransformer(n_quantiles=2000, output_distribution="normal",
                                 random_state=42)
        qt.fit(values.reshape(-1, 1))
        scalers[name] = qt
        v = values
        print(f"    {name}: p50={np.nanpercentile(v, 50):.2f}, "
              f"p95={np.nanpercentile(v, 95):.2f}, max={np.nanmax(v):.2f}")

    # rrp_5m_missing flag (index ENC_5M_MISSING_IDX): 1 = no 5m data, 0 = data present
    has_5m_train = (X_enc_train[:, :, ENC_5M_MISSING_IDX] == 0)  # [N, 96] bool

    for j, feat in enumerate(ENC_CONTINUOUS):
        if feat in ENC_5M_FEATURES_SET:
            valid_vals = X_enc_train[:, :, j][has_5m_train]
            if len(valid_vals) == 0:
                print(f"    WARNING: no 5m data in train split for {feat} — scaler fitted on zeros")
                valid_vals = np.array([0.0])
            fit_one(feat, valid_vals)
        else:
            fit_one(feat, X_enc_train[:, :, j])
    pd7_avail_idx_dec = DEC_FEATURES.index("pd7_available")
    pd7_available_train = X_dec_train[:, 0, pd7_avail_idx_dec].astype(bool)  # [N] constant per sample

    for j, feat in enumerate(DEC_CONTINUOUS):
        if feat in ADJ_REGION_FEATURES:
            # VIC1/NSW1 only valid in steps 0-55 (PREDISPATCH horizon).
            pd_mask = y_covar_mask_train[:, :56]
            valid_covars = X_dec_train[:, :56, j][pd_mask]
        elif feat in SDO_FEATURES:
            # SDO available from ~2025-03-22; 0-filled before that.
            if sdo_covar_mask_train is not None and sdo_covar_mask_train.any():
                valid_covars = X_dec_train[:, :, j][sdo_covar_mask_train]
            else:
                valid_covars = X_dec_train[:, :, j].reshape(-1)
                print(f"    WARNING: no SDO data in train split for {feat}")
        elif feat in PD7_FEATURES:
            # pd7_rrp: fit only on samples where PD7Day was available (post-2026-02).
            # Broadcast pd7_available [N] → [N, 144] mask, then intersect with covar mask.
            pd7_mask = pd7_available_train[:, np.newaxis] & y_covar_mask_train
            valid_covars = X_dec_train[:, :, j][pd7_mask]
            if len(valid_covars) == 0:
                print(f"    WARNING: no PD7Day data in train split for {feat} — scaler fitted on zeros")
                valid_covars = np.array([0.0])
        else:
            valid_covars = X_dec_train[:, :, j][y_covar_mask_train]
        fit_one(feat, valid_covars)

    # Target: fit on VALID train steps only to avoid fitting on zero-padded values
    valid_targets = y_train[y_mask_train]
    fit_one("target_rrp", valid_targets)

    return scalers


def apply_scalers(X_enc, X_dec, y_raw, y_mask, y_covar_mask, scalers,
                  sdo_covar_mask=None):
    """Apply fitted scalers. Non-continuous features pass through unchanged."""
    X_enc_n = X_enc.copy()
    X_dec_n = X_dec.copy()
    y_norm  = np.zeros_like(y_raw)

    def transform(val, s):
        if s == "log":
            return np.sign(val) * np.log1p(np.abs(val) / LOG_SCALE_FACTOR)
        return s.transform(val.reshape(-1, 1)).reshape(-1)

    has_5m = (X_enc_n[:, :, ENC_5M_MISSING_IDX] == 0)  # [N, 96] bool

    for j, feat in enumerate(ENC_CONTINUOUS):
        if feat in ENC_5M_FEATURES_SET:
            # Transform only non-missing steps; leave 0-filled missing steps as 0
            view = X_enc_n[:, :, j]
            view[has_5m] = transform(view[has_5m], scalers[feat])
        else:
            X_enc_n[:, :, j] = transform(X_enc_n[:, :, j], scalers[feat]).reshape(X_enc_n[:, :, j].shape)

    pd7_avail_idx_dec = DEC_FEATURES.index("pd7_available")
    pd7_available = X_dec_n[:, 0, pd7_avail_idx_dec].astype(bool)  # [N]

    for j, feat in enumerate(DEC_CONTINUOUS):
        if feat in ADJ_REGION_FEATURES:
            view = X_dec_n[:, :56, j]
            pd_mask = y_covar_mask[:, :56]
            view[pd_mask] = transform(view[pd_mask], scalers[feat])
        elif feat in SDO_FEATURES:
            view = X_dec_n[:, :, j]
            if sdo_covar_mask is not None and sdo_covar_mask.any():
                view[sdo_covar_mask] = transform(view[sdo_covar_mask], scalers[feat])
        elif feat in PD7_FEATURES:
            view = X_dec_n[:, :, j]
            pd7_mask = pd7_available[:, np.newaxis] & y_covar_mask
            if pd7_mask.any():
                view[pd7_mask] = transform(view[pd7_mask], scalers[feat])
        else:
            view = X_dec_n[:, :, j]
            view[y_covar_mask] = transform(view[y_covar_mask], scalers[feat])

    # Target: use transform helper
    y_norm[y_mask] = transform(y_raw[y_mask], scalers["target_rrp"])

    return X_enc_n, X_dec_n, y_norm


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build run-aligned TFT training dataset (144-step / 72h)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Process first 200 runs; print shapes and exit")
    parser.add_argument("--no-normalise", action="store_true",
                        help="Skip normalisation (saves raw arrays for inspection)")
    parser.add_argument("--target-scaling", choices=["quantile", "log"], default="log",
                        help="Scaling method for rrp/target (default: log)")
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

    # Adjacent-region PREDISPATCH (VIC1/NSW1) — optional decoder features
    vic1_pd = nsw1_pd = None
    for region, varname in (("vic1", "vic1_pd"), ("nsw1", "nsw1_pd")):
        path = PARQUET_DIR / f"aemo_predispatch_{region}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            for col in ["interval_dt", "run_time"]:
                df[col] = pd.to_datetime(df[col], utc=True)
            if varname == "vic1_pd":
                vic1_pd = df
            else:
                nsw1_pd = df
            print(f"  {region.upper()} PREDISPATCH: {len(df):,} rows, "
                  f"{df.run_time.nunique():,} unique run_times")
        else:
            print(f"  WARNING: {path.name} not found — {region.upper()} features will be 0-filled")

    # SevenDayOutlook (sd_demand + sd_net_interchange, all 144 decoder steps)
    sdo_path = PARQUET_DIR / "aemo_sevendayoutlook_sa1.parquet"
    if sdo_path.exists():
        sdo = pd.read_parquet(sdo_path)
        for col in ["interval_dt", "run_time"]:
            sdo[col] = pd.to_datetime(sdo[col], utc=True)
        print(f"  SevenDayOutlook: {len(sdo):,} rows, {sdo.run_time.nunique():,} unique run_times "
              f"({sdo.run_time.min().date()} → {sdo.run_time.max().date()})")
    else:
        sdo = None
        print(f"  WARNING: aemo_sevendayoutlook_sa1.parquet not found — sd_demand/sd_net_interchange 0-filled")

    # OOF debiased PREDISPATCH RRP (substitutes raw pd_rrp at steps 0-55)
    oof_path = PARQUET_DIR / "debiased_pd_rrp_oof.parquet"
    if oof_path.exists():
        oof_df = pd.read_parquet(oof_path)
        for col in ["interval_dt", "run_time"]:
            oof_df[col] = pd.to_datetime(oof_df[col], utc=True)
        oof_grouped = oof_df.groupby("run_time", sort=False)
        print(f"  OOF debiased PD RRP: {len(oof_df):,} rows, "
              f"{oof_df.run_time.nunique():,} unique run_times")
        del oof_df  # free memory — we only need the groupby
    else:
        oof_grouped = None
        print(f"  WARNING: debiased_pd_rrp_oof.parquet not found — raw pd_rrp used")

    for df in [predispatch, pd7day]:
        for col in ["interval_dt", "run_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)
    actuals["time"] = pd.to_datetime(actuals["time"], utc=True)

    # 5m volatility aggregates (optional; 0-filled + flagged in build_samples if absent)
    fpath_5m = PARQUET_DIR / "actuals_sa1_5m_agg.parquet"
    if fpath_5m.exists():
        df_5m = pd.read_parquet(fpath_5m)
        if df_5m.index.tz is None:
            df_5m.index = df_5m.index.tz_localize("UTC")
        else:
            df_5m.index = df_5m.index.tz_convert("UTC")
        actuals = (actuals.set_index("time")
                   .join(df_5m[ENC_5M_CONTINUOUS], how="left")
                   .reset_index())
        n_5m = df_5m.notna().all(axis=1).sum()
        print(f"  5m volatility agg: {n_5m:,} rows merged "
              f"({df_5m.index.min().date()} → {df_5m.index.max().date()})")
    else:
        print(f"  WARNING: {fpath_5m.name} not found — run export_parquet.py first")
        print(f"    rrp_5m features will be 0-filled with rrp_5m_missing=1")
        for col in ENC_5M_CONTINUOUS:
            actuals[col] = float("nan")

    print(f"  SA1 PREDISPATCH: {len(predispatch):,} rows, "
          f"{predispatch.run_time.nunique():,} unique run_times")
    print(f"  PD7Day:          {len(pd7day):,} rows, "
          f"{pd7day.run_time.nunique():,} unique run_times "
          f"({pd7day.run_time.min().date()} → {pd7day.run_time.max().date()})")
    print(f"  Actuals:         {len(actuals):,} rows, "
          f"{actuals.time.min()} → {actuals.time.max()}")

    # Forward-fill minor actuals gaps (base features only — 5m NaNs are intentional)
    gap_counts = actuals[ENC_CONT_BASE].isna().sum()
    if gap_counts.sum() > 0:
        print(f"  Actuals NaN counts (forward-filling): "
              f"{gap_counts[gap_counts > 0].to_dict()}")
    for col in ENC_CONT_BASE:
        if col in actuals.columns:
            actuals[col] = actuals[col].ffill().bfill()
    n_5m_rows = actuals[ENC_5M_CONTINUOUS[0]].notna().sum()
    print(f"  5m feature coverage: {n_5m_rows:,}/{len(actuals):,} rows "
          f"({n_5m_rows / len(actuals):.1%})")

    # ── Build samples
    print("\nBuilding run-aligned samples...")
    X_enc, X_dec, y_raw, y_mask, y_covar_mask, sdo_covar_mask, run_times = build_samples(
        actuals, predispatch, pd7day,
        vic1_pd=vic1_pd, nsw1_pd=nsw1_pd,
        sdo=sdo, oof_grouped=oof_grouped,
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

    # ── Stratified eval hold-out (test split)
    # If build_stratified_eval.py has been run, those run_times are excluded
    # from both train and val and saved as a separate test split.
    test_mask = np.zeros(len(run_times), dtype=bool)
    strat_path = PARQUET_DIR / "stratified_eval_run_times.npy"
    if strat_path.exists():
        strat_rt = np.load(strat_path)                     # datetime64[ns]
        strat_int64 = set(strat_rt.view(np.int64).tolist())
        rt_int64 = run_times.view(np.int64)
        for i, rt in enumerate(rt_int64):
            if int(rt) in strat_int64:
                test_mask[i]  = True
                train_mask[i] = False
                val_mask[i]   = False
        n_test = test_mask.sum()
        n_missing = len(strat_rt) - n_test
        print(f"\nStratified eval hold-out: {n_test:,} samples excluded from train/val")
        if n_missing:
            print(f"  ({n_missing} stratified run_times not found in current dataset — "
                  f"may be outside the build window)")
    else:
        print(f"\nNo stratified eval file found ({strat_path.name}) — skipping hold-out.")
        print("  Run data/build_stratified_eval.py to create the benchmark set.")

    n_train, n_val = train_mask.sum(), val_mask.sum()
    print(f"\nTrain/val split (last {VAL_DAYS} days = val, with 72h gap):")
    print(f"  Train: {n_train:,}  Val: {n_val:,}")

    # ── Price weights (log-growth, fitted on train split p50)
    # weight = 1 + log1p(max(0, (raw_price - ref) / ref))
    # Grows quickly above baseload, slows at extreme spikes — no cap needed.
    # Ref is training-set p50 so weight=1.0 at the median and grows from there.
    # Invalid (masked) steps get weight=0 so they contribute nothing to the loss.
    valid_train_rrp = y_raw[train_mask][y_mask[train_mask]]
    price_weight_ref = float(np.percentile(valid_train_rrp, 50))
    y_weights = (1.0 + np.log1p(
        np.maximum(0.0, (y_raw - price_weight_ref) / price_weight_ref)
    )).astype(np.float32)
    y_weights[~y_mask] = 0.0   # zero out invalid steps explicitly
    valid_w = y_weights[y_mask]
    print(f"\nPrice weights (log-growth, ref={price_weight_ref:.1f} $/MWh):")
    print(f"  min={valid_w.min():.2f}  p50={np.percentile(valid_w, 50):.2f}  "
          f"p90={np.percentile(valid_w, 90):.2f}  p99={np.percentile(valid_w, 99):.2f}  "
          f"max={valid_w.max():.2f}")

    # ── Normalisation
    if not args.no_normalise:
        print(f"\nFitting scalers on train split (target_scaling={args.target_scaling})...")
        scalers = fit_scalers(X_enc[train_mask], X_dec[train_mask],
                              y_raw[train_mask], y_mask[train_mask], y_covar_mask[train_mask],
                              sdo_covar_mask_train=sdo_covar_mask[train_mask],
                              target_scaling=args.target_scaling)
        print("\nApplying scalers...")
        X_enc_n, X_dec_n, y_norm = apply_scalers(
            X_enc, X_dec, y_raw, y_mask, y_covar_mask, scalers,
            sdo_covar_mask=sdo_covar_mask)
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
    np.save(PARQUET_DIR / "y_weights.npy",    y_weights)
    np.save(PARQUET_DIR / "y_covar_mask.npy",    y_covar_mask)
    np.save(PARQUET_DIR / "sdo_covar_mask.npy",  sdo_covar_mask)
    np.save(PARQUET_DIR / "run_times.npy",        run_times)
    train_idx = np.where(train_mask)[0]
    val_idx   = np.where(val_mask)[0]
    test_idx  = np.where(test_mask)[0]
    np.savez(PARQUET_DIR / "split_indices.npz", train=train_idx, val=val_idx, test=test_idx)

    # ── Save metadata
    meta = {
        "input_length": INPUT_LENGTH,
        "output_length": out_len,
        "min_valid_steps": args.min_valid_steps,
        "enc_features": ENC_FEATURES,
        "dec_features": DEC_FEATURES,
        "enc_continuous": ENC_CONTINUOUS,
        "enc_cont_base": ENC_CONT_BASE,
        "enc_5m_continuous": ENC_5M_CONTINUOUS,
        "dec_continuous": DEC_CONTINUOUS,
        "n_enc_features": len(ENC_FEATURES),
        "n_dec_features": len(DEC_FEATURES),
        "n_samples": int(len(run_times)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test_stratified": int(test_mask.sum()),
        "val_days": VAL_DAYS,
        "normalised": not args.no_normalise,
        "target_scaling": args.target_scaling,
        "log_scale_factor": LOG_SCALE_FACTOR,
        "price_weight_ref": price_weight_ref,
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
    n_test = test_mask.sum()
    test_s = f", {n_test:,} stratified eval" if n_test else ""
    print(f"  {len(run_times):,} samples ({n_train:,} train, {n_val:,} val{test_s})")
    print(f"  Mask coverage: 1–32h {meta['mask_coverage']['steps_1_32']:.1%}, "
          f"1–28h {meta['mask_coverage']['steps_1_56']:.1%}, "
          f"28–72h {meta['mask_coverage']['steps_57_144']:.1%}")
    print(f"  Valid RRP stats ($/MWh): mean={valid_rrp.mean():.1f}, "
          f"p50={np.percentile(valid_rrp, 50):.1f}, max={valid_rrp.max():.1f}")


if __name__ == "__main__":
    main()
