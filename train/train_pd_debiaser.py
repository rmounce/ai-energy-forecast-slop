#!/usr/bin/env python3
"""
Train PREDISPATCH debiaser: LightGBM on (PD forecast → actual settlement) pairs.

Corrects the systematic bias in AEMO PREDISPATCH price forecasts before they are
fed as decoder covariates to the TFT (Run 011+).

OOF constraint (data leakage prevention)
-----------------------------------------
The historical `debiased_pd_rrp` series is generated via Out-of-Fold (OOF) K-Fold
cross-validation. For each time fold: train on the remaining K−1 folds, predict the
held-out fold. Concatenate predictions → the TFT trains on this imperfect OOF series,
never on a series produced by a debiaser that has already seen those actuals.

Without OOF, the TFT would see an artificially perfect debiased feature (the debiaser
having already fitted the same targets) and fail at inference when the debiaser is
realistically imperfect.

Usage
-----
  python train/train_pd_debiaser.py              # full run
  python train/train_pd_debiaser.py --dry-run    # use first 20k rows, check shapes
  python train/train_pd_debiaser.py --k-folds 5 --n-estimators 2000

Inputs (data/parquet/):
  aemo_predispatch_sa1.parquet  — interval_dt, run_time, rrp, total_demand, net_interchange
  actuals_sa1.parquet           — time, rrp (actual settlement price)

Outputs:
  data/parquet/debiased_pd_rrp_oof.parquet  — (interval_dt, run_time, debiased_rrp)
  models/pd_debiaser/lgbm_final.pkl         — model trained on all data (for inference)
  models/pd_debiaser/metrics.json           — per-fold + overall + regime breakdown
  models/pd_debiaser/residuals_by_horizon.png
  models/pd_debiaser/residuals_by_regime.png
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR    = ROOT / "data" / "parquet"
MODELS_DIR     = ROOT / "models" / "pd_debiaser"
SPIKE_CLF_FILE = PARQUET_DIR / "spike_clf_predictions.parquet"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

K_FOLDS       = 5
N_ESTIMATORS  = 1000
EARLY_STOP    = 50

FEATURES = [
    "pd_rrp",
    "pd_demand",
    "pd_net_interchange",
    "horizon_steps",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    "prob_spike",  # spike classifier P(genuine spike) per run_time — enables smooth correction
]
TARGET = "actual_rrp"

# Regime thresholds ($/MWh) — for validation breakdown
REGIME_OVERSUPPLY = -50.0
REGIME_SPIKE      = 150.0


# ─── Feature engineering ─────────────────────────────────────────────────────

def time_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclic sin/cos encodings in Brisbane/AEST time (no DST)."""
    t = timestamps.tz_convert("Australia/Brisbane")
    return pd.DataFrame({
        "hour_sin":  np.sin(2 * np.pi * t.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * t.hour / 24),
        "dow_sin":   np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos":   np.cos(2 * np.pi * t.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * (t.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
    }, index=timestamps)


def build_features(pd_df: pd.DataFrame, actuals: pd.DataFrame,
                   dry_run: bool = False) -> pd.DataFrame:
    """
    Join PREDISPATCH forecasts with actual settlement prices and compute features.

    Returns a DataFrame with columns FEATURES + [TARGET, 'interval_dt', 'run_time'].
    Rows where no actual price exists for the target interval are dropped.
    """
    print("  Joining PREDISPATCH forecasts with actuals...")
    # Actuals indexed by time (UTC)
    act = actuals.set_index("time")[["rrp"]].rename(columns={"rrp": TARGET})

    # Rename PREDISPATCH columns up front (avoids column-name conflicts with actuals)
    pd_df = pd_df.rename(columns={
        "rrp":             "pd_rrp",
        "total_demand":    "pd_demand",
        "net_interchange": "pd_net_interchange",
    })

    # Merge on interval_dt == actuals.time (inner join: only intervals with known actuals)
    df = pd_df.join(act, on="interval_dt", how="inner")

    if dry_run:
        # Truncate early so feature engineering stays fast
        n_runs = df["run_time"].nunique()
        early_run_times = sorted(df["run_time"].unique())[:max(1, n_runs // 10)]
        df = df[df["run_time"].isin(early_run_times)].copy()
        print(f"  [dry-run] Truncated to first 10% of run_times: {len(df):,} rows")
    else:
        print(f"  Joined: {len(df):,} (interval_dt, run_time) pairs "
              f"({df.interval_dt.nunique():,} unique target intervals)")

    # Horizon in 30-min steps (≥ 0; 0 = same-period nowcast)
    df["horizon_steps"] = (
        (df["interval_dt"] - df["run_time"]).dt.total_seconds() / 1800
    ).astype(np.float32)

    # Time features on unique interval_dt values only (efficient: ~57K not ~3M rows)
    unique_intervals = pd.DatetimeIndex(df["interval_dt"].unique())
    tf = time_features(unique_intervals)
    tf.index.name = "interval_dt"
    df = df.merge(tf.reset_index(), on="interval_dt", how="left")

    # Spike classifier probability per run_time
    if SPIKE_CLF_FILE.exists():
        clf_df = pd.read_parquet(SPIKE_CLF_FILE)
        clf_df["run_time"] = pd.to_datetime(clf_df["run_time"], utc=True)
        spike_prob = clf_df.set_index("run_time")["prob_spike"]
        df["prob_spike"] = df["run_time"].map(spike_prob).fillna(0.0).astype(np.float32)
        n_matched = df["run_time"].isin(spike_prob.index).sum()
        print(f"  Joined spike classifier: {clf_df.run_time.nunique():,} run_times matched "
              f"({n_matched:,}/{len(df):,} rows, mean prob={df['prob_spike'].mean():.3f})")
    else:
        print("  WARNING: spike_clf_predictions.parquet not found — prob_spike=0.0 for all rows")
        df["prob_spike"] = np.float32(0.0)

    # Drop rows with NaN in any feature or target
    required = FEATURES + [TARGET]
    before = len(df)
    df = df.dropna(subset=required)
    if len(df) < before:
        print(f"  Dropped {before - len(df):,} rows with NaN in features/target")

    return df.reset_index(drop=True)


# ─── Time-based K-Fold split ─────────────────────────────────────────────────

def make_time_folds(df: pd.DataFrame, k: int) -> list[np.ndarray]:
    """
    Split df into k equal-width folds by interval_dt (chronological).

    Returns list of k boolean arrays (True = this row is in the fold).
    """
    min_t = df["interval_dt"].min()
    max_t = df["interval_dt"].max()
    delta = (max_t - min_t) / k

    folds = []
    for i in range(k):
        fold_start = min_t + i * delta
        fold_end   = min_t + (i + 1) * delta
        if i == k - 1:
            mask = df["interval_dt"] >= fold_start
        else:
            mask = (df["interval_dt"] >= fold_start) & (df["interval_dt"] < fold_end)
        folds.append(mask.values)
        n = mask.sum()
        print(f"    Fold {i+1}: {fold_start.date()} → {fold_end.date()}, {n:,} rows")
    return folds


# ─── LightGBM training ───────────────────────────────────────────────────────

def make_lgbm_params(n_estimators: int) -> dict:
    return {
        "objective":        "regression_l1",   # MAE — robust to price spikes
        "num_leaves":       127,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 50,
        "n_estimators":     n_estimators,
        "verbose":          -1,
        "n_jobs":           -1,
    }


def train_lgbm(X_train, y_train, X_val, y_val, params: dict):
    """Train one LightGBM model with early stopping."""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            __import__("lightgbm").early_stopping(EARLY_STOP, verbose=False),
            __import__("lightgbm").log_evaluation(period=0),  # suppress per-round logs
        ],
    )
    best = model.best_iteration_
    return model, best


def oof_predict(df: pd.DataFrame, folds: list[np.ndarray],
                params: dict) -> np.ndarray:
    """
    Run OOF K-Fold training. Returns array of OOF predictions (same length as df).

    For each fold k: train on all other folds, predict fold k.
    Validation set for early stopping: fold k itself (correct — this is the holdout).
    """
    oof_preds = np.full(len(df), np.nan)
    X = df[FEATURES].astype(np.float32)  # DataFrame with named columns
    y = df[TARGET].values.astype(np.float32)

    for k, test_mask in enumerate(folds):
        train_mask = ~test_mask
        print(f"  Fold {k+1}/{len(folds)}: "
              f"train={train_mask.sum():,}, test={test_mask.sum():,}")

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        model, best_iter = train_lgbm(X_train, y_train, X_test, y_test, params)
        preds = model.predict(X_test)
        oof_preds[test_mask] = preds

        mae_raw = np.mean(np.abs(y_test - X_test["pd_rrp"].values))
        mae_deb = np.mean(np.abs(y_test - preds))
        print(f"    Best iter: {best_iter:,} | "
              f"MAE raw={mae_raw:.2f} → debiased={mae_deb:.2f} $/MWh")

    return oof_preds


# ─── Metrics ─────────────────────────────────────────────────────────────────

def regime_label(price: float) -> str:
    if price < REGIME_OVERSUPPLY:
        return "oversupply"
    if price > REGIME_SPIKE:
        return "spike"
    return "baseload"


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute bias/MAE/RMSE before and after debiasing, broken down by regime and horizon."""
    actual = df[TARGET].values
    raw    = df["pd_rrp"].values
    deb    = df["oof_debiased_rrp"].values

    def metrics_slice(a, r, d):
        err_r = a - r
        err_d = a - d
        return {
            "raw":      {"bias": float(np.mean(err_r)), "mae": float(np.mean(np.abs(err_r))),
                         "rmse": float(np.sqrt(np.mean(err_r ** 2)))},
            "debiased": {"bias": float(np.mean(err_d)), "mae": float(np.mean(np.abs(err_d))),
                         "rmse": float(np.sqrt(np.mean(err_d ** 2)))},
        }

    result = {
        "overall": {**metrics_slice(actual, raw, deb), "n": int(len(df))}
    }

    # By horizon bin
    horizon_bins = [(1, 6), (7, 16), (17, 32), (33, 56), (57, 99)]
    for lo, hi in horizon_bins:
        idx = (df["horizon_steps"] >= lo) & (df["horizon_steps"] <= hi)
        if idx.sum() == 0:
            continue
        m = idx.values
        result[f"horizon_{lo}_{hi}"] = {**metrics_slice(actual[m], raw[m], deb[m]),
                                         "n": int(idx.sum())}

    # By regime (based on actual price)
    for regime in ("baseload", "spike", "oversupply"):
        idx = df[TARGET].apply(regime_label) == regime
        if idx.sum() == 0:
            continue
        m = idx.values
        result[f"regime_{regime}"] = {**metrics_slice(actual[m], raw[m], deb[m]),
                                       "n": int(idx.sum())}

    return result


def print_metrics(m: dict):
    print(f"\n  {'Category':<28} {'N':>8} {'Raw bias':>10} {'Deb bias':>10} "
          f"{'Raw MAE':>10} {'Deb MAE':>10}")
    print("  " + "-" * 78)
    for key, v in m.items():
        n  = v["n"]
        rb = v["raw"]["bias"]
        db = v["debiased"]["bias"]
        rm = v["raw"]["mae"]
        dm = v["debiased"]["mae"]
        print(f"  {key:<28} {n:>8,} {rb:>10.2f} {db:>10.2f} {rm:>10.2f} {dm:>10.2f}")


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_residuals_by_horizon(df: pd.DataFrame, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    horizon_bins = [(1, 6), (7, 16), (17, 32), (33, 56)]
    labels, raw_maes, deb_maes, raw_bias, deb_bias = [], [], [], [], []

    for lo, hi in horizon_bins:
        mask = (df["horizon_steps"] >= lo) & (df["horizon_steps"] <= hi)
        if mask.sum() == 0:
            continue
        actual = df.loc[mask, TARGET].values
        raw    = df.loc[mask, "pd_rrp"].values
        deb    = df.loc[mask, "oof_debiased_rrp"].values
        labels.append(f"h={lo}–{hi}\n({hi*0.5:.0f}h)")
        raw_maes.append(np.mean(np.abs(actual - raw)))
        deb_maes.append(np.mean(np.abs(actual - deb)))
        raw_bias.append(np.mean(actual - raw))
        deb_bias.append(np.mean(actual - deb))

    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - w/2, raw_maes, w, label="Raw PD",    color="#e74c3c", alpha=0.8)
    ax1.bar(x + w/2, deb_maes, w, label="Debiased",  color="#2ecc71", alpha=0.8)
    ax1.set_title("MAE by horizon bin")
    ax1.set_ylabel("MAE ($/MWh)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.legend()

    ax2.bar(x - w/2, raw_bias, w, label="Raw PD",   color="#e74c3c", alpha=0.8)
    ax2.bar(x + w/2, deb_bias, w, label="Debiased", color="#2ecc71", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Mean bias by horizon bin  (positive = PD underestimates actual)")
    ax2.set_ylabel("Bias ($/MWh)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_residuals_by_regime(df: pd.DataFrame, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    regimes = ["baseload", "spike", "oversupply"]
    labels, raw_maes, deb_maes, raw_bias, deb_bias = [], [], [], [], []

    for regime in regimes:
        mask = df[TARGET].apply(regime_label) == regime
        if mask.sum() == 0:
            continue
        actual = df.loc[mask, TARGET].values
        raw    = df.loc[mask, "pd_rrp"].values
        deb    = df.loc[mask, "oof_debiased_rrp"].values
        labels.append(f"{regime}\n(n={mask.sum():,})")
        raw_maes.append(np.mean(np.abs(actual - raw)))
        deb_maes.append(np.mean(np.abs(actual - deb)))
        raw_bias.append(np.mean(actual - raw))
        deb_bias.append(np.mean(actual - deb))

    x = np.arange(len(labels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.bar(x - w/2, raw_maes, w, label="Raw PD",   color="#e74c3c", alpha=0.8)
    ax1.bar(x + w/2, deb_maes, w, label="Debiased", color="#2ecc71", alpha=0.8)
    ax1.set_title("MAE by price regime")
    ax1.set_ylabel("MAE ($/MWh)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.legend()

    ax2.bar(x - w/2, raw_bias, w, label="Raw PD",   color="#e74c3c", alpha=0.8)
    ax2.bar(x + w/2, deb_bias, w, label="Debiased", color="#2ecc71", alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_title("Mean bias by price regime")
    ax2.set_ylabel("Bias ($/MWh)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train PREDISPATCH debiaser (LightGBM, OOF K-Fold)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use first 20,000 rows — check shapes and metrics only")
    parser.add_argument("--k-folds",      type=int, default=K_FOLDS,
                        help=f"Number of time folds for OOF (default: {K_FOLDS})")
    parser.add_argument("--n-estimators", type=int, default=N_ESTIMATORS,
                        help=f"LightGBM n_estimators (default: {N_ESTIMATORS})")
    parser.add_argument("--no-plots",     action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    print("=== PREDISPATCH Debiaser Training ===")
    if args.dry_run:
        print("  [dry-run mode — first 20k rows only]")

    # ── Load data
    print("\nLoading parquet files...")
    for name in ("aemo_predispatch_sa1.parquet", "actuals_sa1.parquet"):
        path = PARQUET_DIR / name
        if not path.exists():
            print(f"ERROR: {path} not found. Run data/export_parquet.py first.", file=sys.stderr)
            sys.exit(1)

    pd_df   = pd.read_parquet(PARQUET_DIR / "aemo_predispatch_sa1.parquet")
    actuals = pd.read_parquet(PARQUET_DIR / "actuals_sa1.parquet")

    for col in ["interval_dt", "run_time"]:
        pd_df[col] = pd.to_datetime(pd_df[col], utc=True)
    actuals["time"] = pd.to_datetime(actuals["time"], utc=True)

    print(f"  PREDISPATCH: {len(pd_df):,} rows, "
          f"{pd_df.run_time.nunique():,} run_times, "
          f"{pd_df.interval_dt.min().date()} → {pd_df.interval_dt.max().date()}")
    print(f"  Actuals:     {len(actuals):,} rows, "
          f"{actuals.time.min().date()} → {actuals.time.max().date()}")

    # ── Build feature matrix
    print("\nBuilding feature matrix...")
    df = build_features(pd_df, actuals, dry_run=args.dry_run)

    print(f"\nFeature matrix: {len(df):,} rows, {len(FEATURES)} features + target")
    print(f"  interval_dt: {df.interval_dt.min().date()} → {df.interval_dt.max().date()}")
    print(f"  horizon_steps: min={df.horizon_steps.min():.0f}, "
          f"median={df.horizon_steps.median():.0f}, max={df.horizon_steps.max():.0f}")
    print(f"  pd_rrp:   mean={df.pd_rrp.mean():.1f}, p95={df.pd_rrp.quantile(0.95):.1f}")
    print(f"  actual_rrp: mean={df.actual_rrp.mean():.1f}, p95={df.actual_rrp.quantile(0.95):.1f}")

    # ── OOF training
    print(f"\nOOF training ({args.k_folds}-fold, split by interval_dt)...")
    folds  = make_time_folds(df, args.k_folds)
    params = make_lgbm_params(args.n_estimators)
    oof    = oof_predict(df, folds, params)

    df["oof_debiased_rrp"] = oof.astype(np.float32)

    # ── Metrics
    print("\nValidation metrics (OOF):")
    metrics = compute_metrics(df)
    print_metrics(metrics)

    # ── Final model (train on all data)
    print("\nTraining final model on all data...")
    from lightgbm import LGBMRegressor
    final_model = LGBMRegressor(**params)
    X_all = df[FEATURES].astype(np.float32)
    y_all = df[TARGET].values.astype(np.float32)
    final_model.fit(X_all, y_all)
    print(f"  Final model: {final_model.n_estimators_} trees")

    # ── Save outputs
    if not args.dry_run:
        print("\nSaving outputs...")

        # OOF debiased series — what the TFT decoder reads
        oof_out = df[["interval_dt", "run_time", "oof_debiased_rrp"]].copy()
        oof_out_path = PARQUET_DIR / "debiased_pd_rrp_oof.parquet"
        oof_out.to_parquet(oof_out_path, index=False)
        print(f"  OOF series:   {oof_out_path} ({len(oof_out):,} rows)")

        # Final model
        model_path = MODELS_DIR / "lgbm_final.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Final model:  {model_path}")

        # Metrics
        metrics_path = MODELS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics:      {metrics_path}")

        # Plots
        if not args.no_plots:
            print("  Generating plots...")
            plot_residuals_by_horizon(df, MODELS_DIR / "residuals_by_horizon.png")
            plot_residuals_by_regime( df, MODELS_DIR / "residuals_by_regime.png")
    else:
        print("\n[dry-run] Outputs not saved. Remove --dry-run to write files.")

    print("\nDone.")


if __name__ == "__main__":
    main()
