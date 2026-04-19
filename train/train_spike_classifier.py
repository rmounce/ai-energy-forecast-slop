#!/usr/bin/env python3
"""
Train upstream spike regime classifier for PREDISPATCH debiaser routing.

Predicts P(genuine spike window) at each PREDISPATCH run_time. Used to route
debiaser application:
  - prob_spike > threshold  →  bypass debiaser (raw PREDISPATCH passes through)
  - prob_spike <= threshold →  apply existing LightGBM debiaser

This resolves the debiaser spike guard problem: no scalar price threshold can
distinguish genuine spike prices from PREDISPATCH noise in the same magnitude
range. A classifier using time context + demand + recent actual lags can.

Inputs (data/parquet/):
  aemo_predispatch_sa1.parquet  — interval_dt, run_time, rrp, total_demand, net_interchange
  actuals_sa1.parquet           — time, rrp (30-min actual settlement price)

Outputs:
  models/spike_classifier/lgbm_spike_clf.pkl   — trained LightGBM classifier
  models/spike_classifier/metrics.json         — precision/recall at thresholds, val set stats
  data/parquet/spike_clf_predictions.parquet   — prob_spike for every run_time (for retro eval)

Usage:
  python train/train_spike_classifier.py
  python train/train_spike_classifier.py --dry-run   # fast subset check
  python train/train_spike_classifier.py --threshold 0.3  # override default routing threshold
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PARQUET_DIR = ROOT / "data" / "parquet"
MODELS_DIR = ROOT / "models" / "spike_classifier"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Label: any actual RRP >= this threshold in the window is a "spike"
SPIKE_RRP_THRESHOLD = 300.0

# Window: label looks forward this many hours from run_time
# Covers TFT decoder steps 0-55 (0-27.5h) with a small buffer
LABEL_WINDOW_HOURS = 28

# Default routing threshold — can be tuned after inspecting PR curves
DEFAULT_ROUTE_THRESHOLD = 0.35

# Time-based train/val split cutoff
VAL_CUTOFF = "2025-07-01"

FEATURES = [
    "pd_rrp_h0",
    "pd_rrp_max",
    "pd_rrp_p90",
    "pd_demand_max",
    "pd_net_interchange_h0",
    "actual_rrp_lag1",
    "actual_rrp_lag2",
    "actual_rrp_lag4",
    "actual_rrp_lag8",
    "actual_rrp_max_6h",
    "actual_rrp_max_24h",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]


def time_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    t = timestamps.tz_convert("Australia/Brisbane")
    return pd.DataFrame({
        "hour_sin":  np.sin(2 * np.pi * t.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * t.hour / 24),
        "dow_sin":   np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos":   np.cos(2 * np.pi * t.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * (t.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
    }, index=timestamps)


def build_dataset(pd_df: pd.DataFrame, actuals: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    """
    Build a per-run_time classification dataset.

    Features: PREDISPATCH summary stats + recent actual RRP lags + time.
    Label: any actual RRP >= SPIKE_RRP_THRESHOLD in the next LABEL_WINDOW_HOURS.
    """
    print("  Building PREDISPATCH per-run_time features...")

    if dry_run:
        run_times = sorted(pd_df["run_time"].unique())[:500]
        pd_df = pd_df[pd_df["run_time"].isin(run_times)].copy()
        print(f"  [dry-run] Using first 500 run_times")

    # Per-run_time aggregates
    grp = pd_df.groupby("run_time")

    # h0: interval_dt == run_time
    h0_mask = pd_df["interval_dt"] == pd_df["run_time"]
    h0 = pd_df[h0_mask].groupby("run_time").agg(
        pd_rrp_h0=("rrp", "first"),
        pd_net_interchange_h0=("net_interchange", "first"),
    )

    # Max / p90 across all steps
    agg = grp.agg(
        pd_rrp_max=("rrp", "max"),
        pd_demand_max=("total_demand", "max"),
    )
    agg["pd_rrp_p90"] = grp["rrp"].quantile(0.9)

    feat = agg.join(h0, how="left")
    feat.index = pd.DatetimeIndex(feat.index)

    print(f"  {len(feat):,} run_times with PREDISPATCH features")

    # Actuals: indexed by time (UTC)
    act = actuals.set_index("time")["rrp"].sort_index()
    act.index = pd.DatetimeIndex(act.index).tz_convert("UTC")

    # Lag features: recent actual RRP at fixed offsets before run_time
    lag_offsets = {
        "actual_rrp_lag1": pd.Timedelta(minutes=30),
        "actual_rrp_lag2": pd.Timedelta(minutes=60),
        "actual_rrp_lag4": pd.Timedelta(minutes=120),
        "actual_rrp_lag8": pd.Timedelta(minutes=240),
    }
    for col, offset in lag_offsets.items():
        feat[col] = feat.index.map(
            lambda rt, off=offset: act.get(rt - off, np.nan)
        )

    # Rolling max of actual RRP over last 6h and 24h
    print("  Computing rolling max actual RRP lags...")
    # Use reindex to align actuals with run_times
    run_times_utc = feat.index.tz_convert("UTC")
    rolling_act = act.reindex(run_times_utc, method="ffill")

    # Compute max over last N intervals by iterating with searchsorted
    rrp_vals = act.values
    rrp_times = act.index

    def rolling_max(rt, n_periods):
        end_pos = rrp_times.searchsorted(rt, side="right")
        start_pos = max(0, end_pos - n_periods)
        window = rrp_vals[start_pos:end_pos]
        return float(np.nanmax(window)) if len(window) > 0 else np.nan

    # 6h = 12 × 30-min intervals; 24h = 48 intervals
    feat["actual_rrp_max_6h"] = [rolling_max(rt, 12) for rt in feat.index]
    feat["actual_rrp_max_24h"] = [rolling_max(rt, 48) for rt in feat.index]

    # Time features
    tf = time_features(feat.index)
    for col in tf.columns:
        feat[col] = tf[col].values

    # Label: any actual RRP >= threshold in next LABEL_WINDOW_HOURS
    print("  Computing spike labels...")
    label_window = pd.Timedelta(hours=LABEL_WINDOW_HOURS)
    labels = []
    for rt in feat.index:
        window = act[(act.index >= rt) & (act.index < rt + label_window)]
        labels.append(int(window.max() >= SPIKE_RRP_THRESHOLD) if len(window) > 0 else np.nan)
    feat["spike_label"] = labels

    # Drop rows with NaN label or features
    before = len(feat)
    feat = feat.dropna(subset=["spike_label"] + FEATURES)
    print(f"  Dropped {before - len(feat):,} rows with missing label/features; {len(feat):,} remain")
    feat["spike_label"] = feat["spike_label"].astype(int)

    spike_rate = feat["spike_label"].mean()
    print(f"  Spike rate: {spike_rate:.1%} ({feat['spike_label'].sum():,} spike / {len(feat):,} total)")

    return feat.reset_index().rename(columns={"index": "run_time"}) if "run_time" not in feat.columns else feat.reset_index()


def train_classifier(X_train, y_train, X_val, y_val, n_estimators: int = 500):
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: uv pip install lightgbm")
        sys.exit(1)

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"  class_weight: {pos_weight:.1f}× (to balance spike minority)")

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return clf


def evaluate_thresholds(probs, y_true, thresholds=None):
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    auc = roc_auc_score(y_true, probs)
    ap = average_precision_score(y_true, probs)
    prec, rec, pr_thresholds = precision_recall_curve(y_true, probs)

    print(f"\n  Val set: ROC-AUC={auc:.3f}, AP={ap:.3f}")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Route spike%':>13}")
    results = {}
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        route_pct = preds.mean()
        print(f"  {t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>8.3f} {route_pct:>13.1%}")
        results[str(t)] = {"precision": p, "recall": r, "f1": f1, "route_pct": route_pct}

    return {"roc_auc": auc, "avg_precision": ap, "thresholds": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=DEFAULT_ROUTE_THRESHOLD,
                        help="Routing threshold to highlight in output (does not affect training)")
    args = parser.parse_args()

    print("Loading PREDISPATCH data...")
    pd_df = pd.read_parquet(PARQUET_DIR / "aemo_predispatch_sa1.parquet")
    print(f"  {len(pd_df):,} rows, {pd_df['run_time'].nunique():,} run_times")

    print("Loading actuals...")
    actuals = pd.read_parquet(PARQUET_DIR / "actuals_sa1.parquet")
    print(f"  {len(actuals):,} 30-min actuals ({actuals['time'].min()} → {actuals['time'].max()})")

    print("Building feature dataset...")
    feat = build_dataset(pd_df, actuals, dry_run=args.dry_run)

    # Time-based train/val split
    run_times = pd.DatetimeIndex(feat["run_time"])
    if args.dry_run:
        # Use last 20% as val in dry-run (cutoff won't apply to this small window)
        n = len(feat)
        split_idx = int(n * 0.8)
        train_mask = np.zeros(n, dtype=bool)
        train_mask[:split_idx] = True
        val_mask = ~train_mask
        print(f"\n[dry-run] Train: {train_mask.sum():,} | Val: {val_mask.sum():,} (80/20 split)")
    else:
        val_cutoff = pd.Timestamp(VAL_CUTOFF, tz="UTC")
        train_mask = np.array(run_times < val_cutoff)
        val_mask = ~train_mask
        print(f"\nTrain: {train_mask.sum():,} | Val: {val_mask.sum():,} (cutoff: {VAL_CUTOFF})")

    X = feat[FEATURES].values.astype(np.float32)
    y = feat["spike_label"].values

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train spike rate: {y_train.mean():.1%} | Val spike rate: {y_val.mean():.1%}")

    print(f"\nTraining LightGBM classifier (n_estimators={args.n_estimators})...")
    clf = train_classifier(X_train, y_train, X_val, y_val, n_estimators=args.n_estimators)

    probs_val = clf.predict_proba(X_val)[:, 1]
    metrics = evaluate_thresholds(probs_val, y_val)
    metrics["default_threshold"] = args.threshold
    metrics["val_cutoff"] = VAL_CUTOFF
    metrics["spike_rrp_threshold"] = SPIKE_RRP_THRESHOLD
    metrics["label_window_hours"] = LABEL_WINDOW_HOURS

    if not args.dry_run:
        # Save model
        model_path = MODELS_DIR / "lgbm_spike_clf.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": clf, "features": FEATURES,
                         "default_threshold": args.threshold}, f)
        print(f"\nSaved model → {model_path}")

        # Save metrics
        metrics_path = MODELS_DIR / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics → {metrics_path}")

        # Save predictions for all run_times (for use in retro eval)
        print("\nGenerating predictions for all run_times...")
        probs_all = clf.predict_proba(X)[:, 1]
        pred_df = pd.DataFrame({
            "run_time": feat["run_time"].values,
            "prob_spike": probs_all,
            "spike_label": y,
        })
        pred_path = PARQUET_DIR / "spike_clf_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        print(f"Saved predictions → {pred_path} ({len(pred_df):,} rows)")

        # Feature importance
        print("\nTop feature importances:")
        fi = sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])
        for name, imp in fi[:10]:
            print(f"  {name:30s} {imp:6.0f}")
    else:
        print("\n[dry-run] Skipping model save.")


if __name__ == "__main__":
    main()
