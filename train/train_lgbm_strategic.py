#!/usr/bin/env python3
"""
Train LightGBM strategic price model: 30-min resolution, 0–72h horizon.

Architecture: 3 LightGBM quantile regressors (q5, q50, q95) trained in
"long format" — one row per (run_time, step_idx).

  step 0  = run_time + 30min  (first 30-min interval)
  step 55 = run_time + 28h    (last PREDISPATCH-covered step)
  step 56 = run_time + 28.5h  (first step beyond PREDISPATCH)
  step 143= run_time + 72h    (last step)

Base features:
  step_idx              [int]   — horizon step 0..143
  has_pd_covariate      [0/1]   — 1 for steps 0..55 where PREDISPATCH exists
  pd_rrp_debiased       [float] — OOF-debiased PREDISPATCH RRP (NaN for steps 56+)
  pd_demand             [float] — PREDISPATCH total_demand (NaN for steps 56+)
  pd_net_interchange    [float] — PREDISPATCH net_interchange (NaN for steps 56+)
  actual_rrp_lag1/2/4/8 [float] — actual RRP at run_time - 1/2/4/8 × 30min
  actual_rrp_max_6h     [float] — max actual RRP in 6h before run_time
  actual_rrp_max_24h    [float] — max actual RRP in 24h before run_time
  hour_sin/cos          [float] — cyclic hour-of-day of target interval (AEST)
  dow_sin/cos           [float] — cyclic day-of-week of target interval
  month_sin/cos         [float] — cyclic month of target interval
  rt_hour_sin/cos       [float] — cyclic hour-of-day of run_time (current context)
  rt_dow_sin/cos        [float] — cyclic day-of-week of run_time

Optional STPASA tail features, enabled with --stpasa-tail-features:
  stpasa_uigf / wind / solar availability fields joined by latest STPASA
  run_time <= model run_time for the same target interval.

Target: actual 30-min RRP at the target interval (NaN rows dropped).

LightGBM handles NaN natively — pd_rrp_debiased NaN for steps 56+ is a natural
split boundary distinguishing PREDISPATCH vs beyond-PREDISPATCH horizons.

Val split: time-ordered, last 60 days of run_times.
No sample weighting (quantile loss naturally handles heavy tails).

Outputs (models/lgbm_strategic/):
  lgbm_q5.pkl / lgbm_q50.pkl / lgbm_q95.pkl
  training_meta.json

Usage:
  python train/train_lgbm_strategic.py
  python train/train_lgbm_strategic.py --dry-run   # 5k run_times, fast check
  python train/train_lgbm_strategic.py --stpasa-tail-features
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
MODEL_DIR   = ROOT / "models" / "lgbm_strategic"
DEFAULT_STPASA = PARQUET_DIR / "aemo_stpasa_regionsolution_sa1.parquet"

WINDOW_STEPS          = 144     # 144 × 30min = 72h
PD_STEPS              = 56      # steps 0..55 covered by PREDISPATCH
VAL_DAYS              = 60
TRAIN_GAP_H           = 1
QUANTILES             = [0.05, 0.50, 0.95]
BRISBANE_TZ           = "Australia/Brisbane"
SPIKE_ROUTE_THRESHOLD = 0.65    # matches retro_lgbm_strategic_inference.py

BASE_FEATURE_NAMES = (
    ["step_idx", "has_pd_covariate"]
    + ["pd_rrp_debiased", "pd_demand", "pd_net_interchange"]
    + [f"actual_rrp_lag{k}" for k in [1, 2, 4, 8]]
    + ["actual_rrp_max_6h", "actual_rrp_max_24h"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    + ["rt_hour_sin", "rt_hour_cos", "rt_dow_sin", "rt_dow_cos"]
)

STPASA_FEATURE_NAMES = [
    "stpasa_uigf",
    "stpasa_total_intermittent_generation",
    "stpasa_ss_wind_uigf",
    "stpasa_ss_solar_uigf",
    "stpasa_ss_wind_capacity",
    "stpasa_ss_solar_capacity",
    "stpasa_wind_avail_frac",
    "stpasa_solar_avail_frac",
    "stpasa_source_horizon_hours",
]

FEATURE_NAMES = BASE_FEATURE_NAMES

LGBM_BASE_PARAMS = {
    "objective":         "quantile",
    "n_estimators":      2000,
    "learning_rate":     0.05,
    "num_leaves":        127,
    "min_child_samples": 100,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_jobs":            -1,
    "random_state":      42,
    "verbose":           -1,
}
EARLY_STOPPING_ROUNDS = 50


# ── Data loading ─────────────────────────────────────────────────────────────

def load_predispatch() -> pd.DataFrame:
    print("Loading PREDISPATCH...", flush=True)
    df = pd.read_parquet(PARQUET_DIR / "aemo_predispatch_sa1.parquet",
                         columns=["run_time", "interval_dt", "rrp",
                                  "total_demand", "net_interchange"])
    df["run_time"]    = pd.to_datetime(df["run_time"],    utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    # Compute step_idx: step 0 = run_time + 30min
    df["step_idx"] = (
        (df["interval_dt"] - df["run_time"]).dt.total_seconds() / 1800
    ).round().astype(int) - 1
    df = df[(df["step_idx"] >= 0) & (df["step_idx"] < PD_STEPS)].copy()
    print(f"  {df['run_time'].nunique():,} run_times, {len(df):,} step rows", flush=True)
    return df


def load_oof() -> pd.DataFrame:
    print("Loading OOF debiased PREDISPATCH...", flush=True)
    df = pd.read_parquet(PARQUET_DIR / "debiased_pd_rrp_oof.parquet")
    df["run_time"]    = pd.to_datetime(df["run_time"],    utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    return df[["run_time", "interval_dt", "oof_debiased_rrp"]]


def load_spike_routing() -> set:
    """
    Return set of run_times where prob_spike > threshold (bypass debiaser).
    Returns empty set if predictions parquet not found.
    """
    clf_path = PARQUET_DIR / "spike_clf_predictions.parquet"
    if not clf_path.exists():
        print("  WARNING: spike_clf_predictions.parquet not found — OOF debiasing applied to all windows",
              flush=True)
        return set()
    print("Loading spike classifier predictions...", flush=True)
    df = pd.read_parquet(clf_path)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    bypass = df[df["prob_spike"] > SPIKE_ROUTE_THRESHOLD]["run_time"]
    print(f"  {len(bypass):,} of {len(df):,} run_times routed to raw bypass "
          f"({len(bypass)/len(df):.1%})", flush=True)
    return set(bypass)


def load_actuals() -> pd.DataFrame:
    print("Loading 30-min actuals...", flush=True)
    df = pd.read_parquet(PARQUET_DIR / "actuals_sa1.parquet",
                         columns=["time", "rrp"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=["rrp"]).sort_values("time").reset_index(drop=True)


def load_stpasa(path: Path = DEFAULT_STPASA) -> pd.DataFrame:
    print(f"Loading STPASA REGIONSOLUTION from {path}...", flush=True)
    df = pd.read_parquet(
        path,
        columns=[
            "interval_dt",
            "run_time",
            "uigf",
            "total_intermittent_generation",
            "ss_wind_uigf",
            "ss_solar_uigf",
            "ss_wind_capacity",
            "ss_solar_capacity",
        ],
    )
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    print(f"  {df['run_time'].nunique():,} run_times, {len(df):,} rows", flush=True)
    return df.sort_values(["interval_dt", "run_time"]).reset_index(drop=True)


# ── Feature construction ─────────────────────────────────────────────────────

def _time_enc(ts: pd.DatetimeIndex, prefix: str = "") -> pd.DataFrame:
    """Cyclic sin/cos for hour, day-of-week, month (AEST, no DST)."""
    t = ts.tz_convert(BRISBANE_TZ)
    return pd.DataFrame({
        f"{prefix}hour_sin":  np.sin(2 * np.pi * t.hour / 24),
        f"{prefix}hour_cos":  np.cos(2 * np.pi * t.hour / 24),
        f"{prefix}dow_sin":   np.sin(2 * np.pi * t.dayofweek / 7),
        f"{prefix}dow_cos":   np.cos(2 * np.pi * t.dayofweek / 7),
        f"{prefix}month_sin": np.sin(2 * np.pi * (t.month - 1) / 12),
        f"{prefix}month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
    }, index=ts)


def build_dataset(pd_df: pd.DataFrame,
                  oof_df: pd.DataFrame,
                  actuals_df: pd.DataFrame,
                  spike_bypass_rts: set,
                  stpasa_df: pd.DataFrame | None = None,
                  run_times_subset: np.ndarray | None = None) -> pd.DataFrame:
    """
    Build long-format training table: one row per (run_time, step_idx).
    Returns DataFrame with FEATURE_NAMES columns + 'target' + 'run_time' + 'interval_dt'.
    """
    # --- Merge OOF debiased RRP into PREDISPATCH rows ---
    pd_df = pd_df.merge(
        oof_df.rename(columns={"oof_debiased_rrp": "pd_rrp_debiased"}),
        on=["run_time", "interval_dt"], how="left"
    )
    pd_df["has_pd_covariate"] = 1
    pd_df = pd_df.rename(columns={
        "rrp":             "pd_rrp_raw",
        "total_demand":    "pd_demand",
        "net_interchange": "pd_net_interchange",
    })
    # Spike routing: for bypass run_times, use raw PREDISPATCH instead of OOF.
    # This ensures training/inference consistency — model sees raw prices during
    # spikes at both train time and inference time (same routing threshold=0.65).
    if spike_bypass_rts:
        bypass_mask = pd_df["run_time"].isin(spike_bypass_rts)
        pd_df.loc[bypass_mask, "pd_rrp_debiased"] = pd_df.loc[bypass_mask, "pd_rrp_raw"].astype("float32")
        n_bypass_rts = pd_df.loc[bypass_mask, "run_time"].nunique()
        print(f"  Spike bypass applied to {n_bypass_rts:,} run_times "
              f"({bypass_mask.sum():,} rows)", flush=True)

    # Fall back to raw PREDISPATCH rrp where OOF is missing (non-spike windows)
    pd_df["pd_rrp_debiased"] = pd_df["pd_rrp_debiased"].fillna(pd_df["pd_rrp_raw"])

    all_run_times = np.sort(pd_df["run_time"].unique())
    if run_times_subset is not None:
        all_run_times = np.intersect1d(all_run_times, run_times_subset)
        pd_df = pd_df[pd_df["run_time"].isin(all_run_times)]

    print(f"  Building long format for {len(all_run_times):,} run_times...", flush=True)

    # --- Generate steps 56..143 rows (no PREDISPATCH covariate) ---
    step_arr = np.arange(PD_STEPS, WINDOW_STEPS)
    rt_rep   = np.repeat(all_run_times, len(step_arr))
    st_rep   = np.tile(step_arr, len(all_run_times))
    beyond_df = pd.DataFrame({
        "run_time":          rt_rep,
        "step_idx":          st_rep.astype(np.int16),
        "has_pd_covariate":  np.zeros(len(rt_rep), dtype=np.int8),
        "pd_rrp_debiased":   np.nan,
        "pd_demand":         np.nan,
        "pd_net_interchange":np.nan,
    })
    beyond_df["run_time"] = pd.to_datetime(beyond_df["run_time"], utc=True)

    # --- Stack PREDISPATCH rows (steps 0-55) + beyond rows (56-143) ---
    pd_rows = pd_df[["run_time", "step_idx", "has_pd_covariate",
                     "pd_rrp_debiased", "pd_demand", "pd_net_interchange"]].copy()
    pd_rows["step_idx"] = pd_rows["step_idx"].astype(np.int16)
    pd_rows["has_pd_covariate"] = pd_rows["has_pd_covariate"].astype(np.int8)

    df = pd.concat([pd_rows, beyond_df], ignore_index=True)

    # interval_dt = run_time + (step_idx + 1) * 30min
    df["interval_dt"] = df["run_time"] + pd.to_timedelta(
        (df["step_idx"].astype(np.int32) + 1) * 30, unit="m"
    )

    if stpasa_df is not None:
        df = attach_stpasa_features(df, stpasa_df)

    # --- Add target: actual RRP at interval_dt ---
    actuals_map = actuals_df.set_index("time")["rrp"]
    df["target"] = df["interval_dt"].map(actuals_map)

    # --- Lag features (precomputed per run_time then joined) ---
    rt_idx = pd.DatetimeIndex(all_run_times)
    lags = pd.DataFrame(index=all_run_times)
    lags.index = pd.DatetimeIndex(lags.index)
    for k in [1, 2, 4, 8]:
        lag_times = rt_idx - pd.Timedelta(minutes=30 * k)
        lags[f"actual_rrp_lag{k}"] = lag_times.map(actuals_map)

    # Rolling max: compute on actuals time series, look up at run_time
    actuals_ts = actuals_df.set_index("time")["rrp"].sort_index()
    actuals_ts = actuals_ts.asfreq("30min")  # fill gaps with NaN for rolling
    roll_6h  = actuals_ts.rolling(12, min_periods=1).max()
    roll_24h = actuals_ts.rolling(48, min_periods=1).max()
    lags["actual_rrp_max_6h"]  = rt_idx.map(roll_6h.reindex(rt_idx, method="ffill"))
    lags["actual_rrp_max_24h"] = rt_idx.map(roll_24h.reindex(rt_idx, method="ffill"))
    lags.index.name = "run_time"
    lags = lags.reset_index()

    df = df.merge(lags, on="run_time", how="left")

    # --- Time features at target interval ---
    iv_idx = pd.DatetimeIndex(df["interval_dt"])
    tf = _time_enc(iv_idx, prefix="")
    df[["hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos"]] = tf.values

    # --- Run_time context features ---
    rt_idx_full = pd.DatetimeIndex(df["run_time"])
    rtf = _time_enc(rt_idx_full, prefix="rt_")
    df[["rt_hour_sin", "rt_hour_cos",
        "rt_dow_sin",  "rt_dow_cos"]] = rtf[
            ["rt_hour_sin", "rt_hour_cos", "rt_dow_sin", "rt_dow_cos"]
        ].values

    # --- Drop rows with no target, sort ---
    n_before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"  {n_before:,} rows → {len(df):,} after dropping NaN targets", flush=True)

    return df


def attach_stpasa_features(df: pd.DataFrame, stpasa_df: pd.DataFrame) -> pd.DataFrame:
    """Attach latest STPASA row available at model run_time for each target interval."""

    out = df.copy()
    value_cols = [
        "uigf",
        "total_intermittent_generation",
        "ss_wind_uigf",
        "ss_solar_uigf",
        "ss_wind_capacity",
        "ss_solar_capacity",
    ]
    for col in value_cols:
        out[f"stpasa_{col}"] = np.nan
    out["stpasa_run_time"] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

    source_by_target = {
        target: grp.sort_values("run_time")
        for target, grp in stpasa_df.groupby("interval_dt", sort=False)
    }

    for target, idx in out.groupby("interval_dt", sort=False).groups.items():
        source = source_by_target.get(target)
        if source is None or source.empty:
            continue
        source_run_times = source["run_time"].to_numpy(dtype="datetime64[ns]")
        left_run_times = out.loc[idx, "run_time"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(source_run_times, left_run_times, side="right") - 1
        valid = positions >= 0
        if not valid.any():
            continue

        valid_idx = np.asarray(idx)[valid]
        source_rows = source.iloc[positions[valid]]
        for col in value_cols:
            out.loc[valid_idx, f"stpasa_{col}"] = pd.to_numeric(
                source_rows[col], errors="coerce"
            ).to_numpy()
        out.loc[valid_idx, "stpasa_run_time"] = pd.to_datetime(
            source_rows["run_time"], utc=True
        ).array

    out["stpasa_wind_avail_frac"] = (
        out["stpasa_ss_wind_uigf"] / out["stpasa_ss_wind_capacity"]
    ).replace([np.inf, -np.inf], np.nan)
    out["stpasa_solar_avail_frac"] = (
        out["stpasa_ss_solar_uigf"] / out["stpasa_ss_solar_capacity"]
    ).replace([np.inf, -np.inf], np.nan)
    out["stpasa_source_horizon_hours"] = (
        out["interval_dt"] - out["stpasa_run_time"]
    ).dt.total_seconds() / 3600.0
    coverage = out.loc[out["step_idx"] >= PD_STEPS, "stpasa_uigf"].notna().mean()
    print(f"  STPASA tail coverage: {coverage:.1%} of steps {PD_STEPS + 1}-{WINDOW_STEPS}", flush=True)
    return out


# ── Train/val split ───────────────────────────────────────────────────────────

def make_split(run_times: np.ndarray, val_days: int, gap_h: float
               ) -> tuple[np.ndarray, np.ndarray]:
    rts = pd.DatetimeIndex(np.sort(run_times))
    cutoff = rts.max() - pd.Timedelta(days=val_days)
    gap    = pd.Timedelta(hours=gap_h)
    train_mask = rts <= (cutoff - gap)
    val_mask   = rts > cutoff
    return np.where(train_mask)[0], np.where(val_mask)[0]


# ── LightGBM training ─────────────────────────────────────────────────────────

def train_quantile(q: float,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val:   np.ndarray, y_val:   np.ndarray,
                   feat_names: list[str]) -> lgb.LGBMRegressor:
    params = {**LGBM_BASE_PARAMS, "alpha": q}
    model = lgb.LGBMRegressor(**params)
    X_tr_df = pd.DataFrame(X_train, columns=feat_names)
    X_vl_df = pd.DataFrame(X_val,   columns=feat_names)
    model.fit(
        X_tr_df, y_train,
        eval_set=[(X_vl_df, y_val)],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
    print(f"  q{q:.0%}: best_iteration={model.best_iteration_}", flush=True)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Use first 5k run_times only (fast sanity check)")
    parser.add_argument("--stpasa-tail-features", action="store_true",
                        help="Add experimental STPASA renewable availability tail features")
    parser.add_argument("--stpasa-path", type=Path, default=DEFAULT_STPASA,
                        help=f"STPASA parquet path (default: {DEFAULT_STPASA})")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = list(BASE_FEATURE_NAMES)

    # Load source data
    pd_df          = load_predispatch()
    oof_df         = load_oof()
    actuals_df     = load_actuals()
    spike_bypass_rts = load_spike_routing()
    stpasa_df = None
    if args.stpasa_tail_features:
        if not args.stpasa_path.exists():
            raise FileNotFoundError(
                f"STPASA parquet not found: {args.stpasa_path}. "
                "Run ingest/backfill_stpasa_regionsolution.py first."
            )
        stpasa_df = load_stpasa(args.stpasa_path)
        feature_names += STPASA_FEATURE_NAMES

    # Optionally restrict to subset for dry-run
    run_times_subset = None
    if args.dry_run:
        all_rts = np.sort(pd_df["run_time"].unique())
        run_times_subset = all_rts[:5000]
        print(f"DRY RUN: using first {len(run_times_subset):,} run_times", flush=True)

    print("\nBuilding long-format dataset...", flush=True)
    df = build_dataset(
        pd_df,
        oof_df,
        actuals_df,
        spike_bypass_rts,
        stpasa_df=stpasa_df,
        run_times_subset=run_times_subset,
    )

    # Train/val split by run_time
    all_run_times = df["run_time"].unique()
    train_rt_idx, val_rt_idx = make_split(all_run_times, VAL_DAYS, TRAIN_GAP_H)
    train_rts = all_run_times[train_rt_idx]
    val_rts   = all_run_times[val_rt_idx]

    train_mask = df["run_time"].isin(train_rts)
    val_mask   = df["run_time"].isin(val_rts)

    X = df[feature_names].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"\nSplit: train={len(X_train):,}  val={len(X_val):,}")
    print(f"  Val run_times: {pd.DatetimeIndex(val_rts).min().date()} → "
          f"{pd.DatetimeIndex(val_rts).max().date()}")

    # Train per quantile
    models = {}
    for q in QUANTILES:
        print(f"\nTraining q{q:.0%}...", flush=True)
        model = train_quantile(q, X_train, y_train, X_val, y_val, feature_names)
        models[q] = model

        out_path = MODEL_DIR / f"lgbm_q{int(q*100):02d}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({"model": model, "features": feature_names,
                         "quantile": q}, f)
        print(f"  Saved {out_path}", flush=True)

    # Quick val MAE on q50
    q50 = models[0.50]
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    y_pred_val = q50.predict(X_val_df)
    mae_val = float(np.mean(np.abs(y_pred_val - y_val)))
    print(f"\nVal MAE (q50): ${mae_val:.2f}/MWh")

    # Stratum-level val MAE
    for stratum_label, mask_fn in [
        ("spike",  lambda d: d["target"] >= 300),
        ("low",    lambda d: d["target"] <= -50),
        ("normal", lambda d: (d["target"] > -50) & (d["target"] < 300)),
    ]:
        sub = df[val_mask]
        sub_mask = mask_fn(sub)
        if sub_mask.sum() > 0:
            pred = q50.predict(pd.DataFrame(X_val[sub_mask.values], columns=feature_names))
            mae  = float(np.mean(np.abs(pred - y_val[sub_mask.values])))
            print(f"  {stratum_label:6s} MAE: ${mae:.2f}/MWh  (n={sub_mask.sum():,})")

    # Feature importances (top 10)
    imps = pd.Series(q50.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop feature importances (q50):")
    print(imps.head(10).to_string())

    # Save metadata
    meta = {
        "features": feature_names,
        "n_features": len(feature_names),
        "stpasa_tail_features": args.stpasa_tail_features,
        "stpasa_path": str(args.stpasa_path) if args.stpasa_tail_features else None,
        "window_steps": WINDOW_STEPS,
        "pd_steps": PD_STEPS,
        "quantiles": QUANTILES,
        "val_days": VAL_DAYS,
        "train_samples": int(len(X_train)),
        "val_samples":   int(len(X_val)),
        "val_mae_q50":   round(mae_val, 4),
        "val_run_time_start": str(pd.DatetimeIndex(val_rts).min()),
        "val_run_time_end":   str(pd.DatetimeIndex(val_rts).max()),
        "dry_run": args.dry_run,
        "spike_routing": len(spike_bypass_rts) > 0,
        "spike_route_threshold": SPIKE_ROUTE_THRESHOLD,
        "n_spike_bypass_rts": len(spike_bypass_rts),
        "lgbm_params": LGBM_BASE_PARAMS,
        "top_features": imps.head(10).to_dict(),
    }
    with open(MODEL_DIR / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved training metadata → {MODEL_DIR / 'training_meta.json'}")


if __name__ == "__main__":
    main()
