#!/usr/bin/env python3
"""Train a PD7Day debiaser on materialised PD7Day -> actual RRP pairs.

This is intentionally separate from the PREDISPATCH debiaser. PD7Day has much less
history and a different failure mode: high/cap-like values often behave more like a
categorical risk flag than a literal price forecast. The model is therefore trained
and evaluated side-by-side against raw PD7Day and the current hard-cap baseline.

Outputs:

- ``models/pd7day_debiaser/lgbm_final.pkl``
- ``models/pd7day_debiaser/metrics.json``
- ``data/parquet/debiased_pd7day_oof.parquet``
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
MODELS_DIR = ROOT / "models" / "pd7day_debiaser"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "pd7_rrp",
    "pd7_clip_300",
    "pd7_log_abs",
    "horizon_hours",
    "horizon_days",
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
    "run_hour_sin",
    "run_hour_cos",
    "is_ge_150",
    "is_ge_300",
    "is_ge_980",
    "is_ge_1000",
    "is_ge_20300",
    "cap_run_length",
    "cap_run_pos",
]
TARGET = "actual_rrp"


def _cyclic(series: pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * series.astype(float).to_numpy() / float(period)
    return np.sin(angle), np.cos(angle)


def _add_cap_run_features(df: pd.DataFrame, cap_threshold: float = 300.0) -> pd.DataFrame:
    df = df.sort_values(["run_time", "interval_dt"]).copy()
    df["cap_flag"] = df["pd7_rrp"] >= cap_threshold
    df["cap_run_length"] = 0
    df["cap_run_pos"] = 0

    for _, idx in df.groupby("run_time", sort=False).groups.items():
        run_idx = np.asarray(idx)
        flags = df.loc[run_idx, "cap_flag"].to_numpy(dtype=bool)
        if len(flags) == 0:
            continue
        starts = np.r_[True, flags[1:] != flags[:-1]]
        group_ids = np.cumsum(starts) - 1
        for group_id in np.unique(group_ids[flags]):
            pos = np.where(group_ids == group_id)[0]
            pos = pos[flags[pos]]
            if len(pos) == 0:
                continue
            abs_idx = run_idx[pos]
            df.loc[abs_idx, "cap_run_length"] = int(len(abs_idx))
            df.loc[abs_idx, "cap_run_pos"] = np.arange(1, len(abs_idx) + 1)
    return df


def build_features(pd7day: pd.DataFrame, actuals: pd.DataFrame, dry_run: bool = False) -> pd.DataFrame:
    pd7 = pd7day.rename(columns={"rrp": "pd7_rrp"}).copy()
    pd7["interval_dt"] = pd.to_datetime(pd7["interval_dt"], utc=True)
    pd7["run_time"] = pd.to_datetime(pd7["run_time"], utc=True)
    actuals = actuals[["time", "rrp"]].rename(columns={"rrp": TARGET}).copy()
    actuals["interval_dt"] = pd.to_datetime(actuals["time"], utc=True)

    df = pd7.merge(actuals[["interval_dt", TARGET]], on="interval_dt", how="inner")
    df = df.dropna(subset=["pd7_rrp", TARGET]).copy()
    df["horizon_hours"] = (df["interval_dt"] - df["run_time"]).dt.total_seconds() / 3600.0
    df = df[df["horizon_hours"] >= 0.0].copy()

    if dry_run:
        keep_runs = sorted(df["run_time"].unique())[: max(10, df["run_time"].nunique() // 5)]
        df = df[df["run_time"].isin(keep_runs)].copy()

    df = _add_cap_run_features(df)
    df["horizon_days"] = df["horizon_hours"] / 24.0
    df["pd7_clip_300"] = df["pd7_rrp"].clip(upper=300.0)
    df["pd7_log_abs"] = np.sign(df["pd7_rrp"]) * np.log1p(np.abs(df["pd7_rrp"]))

    hour_sin, hour_cos = _cyclic(df["interval_dt"].dt.hour, 24.0)
    df["target_hour_sin"] = hour_sin
    df["target_hour_cos"] = hour_cos
    dow_sin, dow_cos = _cyclic(df["interval_dt"].dt.dayofweek, 7.0)
    df["target_dow_sin"] = dow_sin
    df["target_dow_cos"] = dow_cos
    month_sin, month_cos = _cyclic(df["interval_dt"].dt.month - 1, 12.0)
    df["target_month_sin"] = month_sin
    df["target_month_cos"] = month_cos
    run_hour_sin, run_hour_cos = _cyclic(df["run_time"].dt.hour, 24.0)
    df["run_hour_sin"] = run_hour_sin
    df["run_hour_cos"] = run_hour_cos

    for threshold in (150, 300, 980, 1000, 20300):
        df[f"is_ge_{threshold}"] = (df["pd7_rrp"] >= float(threshold)).astype(np.float32)

    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def make_time_folds(df: pd.DataFrame, k: int) -> list[np.ndarray]:
    unique_times = pd.Series(sorted(df["interval_dt"].unique()))
    chunks = np.array_split(unique_times.to_numpy(), k)
    folds = []
    for i, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        mask = df["interval_dt"].isin(chunk).to_numpy()
        folds.append(mask)
        print(
            f"    Fold {i + 1}: {pd.Timestamp(chunk[0]).date()} -> "
            f"{pd.Timestamp(chunk[-1]).date()}, {mask.sum():,} rows"
        )
    return folds


def make_lgbm_params(n_estimators: int, n_jobs: int) -> dict:
    return {
        "objective": "regression_l1",
        "num_leaves": 31,
        "learning_rate": 0.04,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "n_estimators": n_estimators,
        "n_jobs": n_jobs,
        "verbose": -1,
    }


def train_model(X_train, y_train, X_val, y_val, params: dict):
    import lightgbm as lgb
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def oof_predict(df: pd.DataFrame, folds: list[np.ndarray], params: dict) -> np.ndarray:
    X = df[FEATURES].astype(np.float32)
    y = df[TARGET].astype(np.float32).to_numpy()
    out = np.full(len(df), np.nan, dtype=np.float32)
    for i, test_mask in enumerate(folds):
        train_mask = ~test_mask
        print(f"  Fold {i + 1}/{len(folds)}: train={train_mask.sum():,}, test={test_mask.sum():,}")
        model = train_model(X[train_mask], y[train_mask], X[test_mask], y[test_mask], params)
        out[test_mask] = model.predict(X[test_mask]).astype(np.float32)
        raw_mae = np.mean(np.abs(y[test_mask] - X.loc[test_mask, "pd7_rrp"].to_numpy()))
        cap_mae = np.mean(np.abs(y[test_mask] - X.loc[test_mask, "pd7_clip_300"].to_numpy()))
        model_mae = np.mean(np.abs(y[test_mask] - out[test_mask]))
        print(f"    MAE raw={raw_mae:.2f}, cap300={cap_mae:.2f}, model={model_mae:.2f}")
    return out


def _metrics_for(actual: np.ndarray, pred: np.ndarray) -> dict:
    err = actual - pred
    return {
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    actual = df[TARGET].to_numpy(dtype=np.float64)
    result = {
        "overall": {
            "n": int(len(df)),
            "raw": _metrics_for(actual, df["pd7_rrp"].to_numpy(dtype=np.float64)),
            "cap300": _metrics_for(actual, df["pd7_clip_300"].to_numpy(dtype=np.float64)),
            "debiased": _metrics_for(actual, df["oof_debiased_rrp"].to_numpy(dtype=np.float64)),
        }
    }
    for name, mask in {
        "forecast_ge_150": df["pd7_rrp"] >= 150.0,
        "forecast_ge_300": df["pd7_rrp"] >= 300.0,
        "forecast_ge_980": df["pd7_rrp"] >= 980.0,
        "actual_ge_150": df[TARGET] >= 150.0,
        "actual_ge_300": df[TARGET] >= 300.0,
    }.items():
        if int(mask.sum()) == 0:
            continue
        a = df.loc[mask, TARGET].to_numpy(dtype=np.float64)
        result[name] = {
            "n": int(mask.sum()),
            "raw": _metrics_for(a, df.loc[mask, "pd7_rrp"].to_numpy(dtype=np.float64)),
            "cap300": _metrics_for(a, df.loc[mask, "pd7_clip_300"].to_numpy(dtype=np.float64)),
            "debiased": _metrics_for(a, df.loc[mask, "oof_debiased_rrp"].to_numpy(dtype=np.float64)),
        }
    return result


def print_metrics(metrics: dict) -> None:
    print(f"\n  {'Slice':<18} {'N':>8} {'raw MAE':>10} {'cap300':>10} {'model':>10} "
          f"{'raw bias':>10} {'model bias':>11}")
    print("  " + "-" * 82)
    for key, val in metrics.items():
        print(
            f"  {key:<18} {val['n']:>8,} "
            f"{val['raw']['mae']:>10.2f} {val['cap300']['mae']:>10.2f} "
            f"{val['debiased']['mae']:>10.2f} {val['raw']['bias']:>10.2f} "
            f"{val['debiased']['bias']:>11.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--k-folds", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    print("=== PD7Day Debiaser Training ===")
    pd7_path = PARQUET_DIR / "aemo_pd7day_sa1.parquet"
    actuals_path = PARQUET_DIR / "actuals_sa1.parquet"
    if not pd7_path.exists() or not actuals_path.exists():
        print("ERROR: required parquet inputs missing. Run data/export_parquet.py first.", file=sys.stderr)
        sys.exit(1)

    pd7day = pd.read_parquet(pd7_path)
    actuals = pd.read_parquet(actuals_path)
    df = build_features(pd7day, actuals, dry_run=args.dry_run)
    print(
        f"Feature matrix: {len(df):,} rows, {df['run_time'].nunique():,} runs, "
        f"{df['interval_dt'].min()} -> {df['interval_dt'].max()}"
    )
    print(
        f"PD7Day range: mean={df['pd7_rrp'].mean():.1f}, p90={df['pd7_rrp'].quantile(.9):.1f}, "
        f"p99={df['pd7_rrp'].quantile(.99):.1f}; actual mean={df[TARGET].mean():.1f}"
    )

    params = make_lgbm_params(args.n_estimators, args.n_jobs)
    folds = make_time_folds(df, args.k_folds)
    df["oof_debiased_rrp"] = oof_predict(df, folds, params)
    metrics = compute_metrics(df)
    print_metrics(metrics)

    print("\nTraining final model on all materialised rows...")
    from lightgbm import LGBMRegressor

    final_model = LGBMRegressor(**params)
    X_all = df[FEATURES].astype(np.float32)
    y_all = df[TARGET].astype(np.float32)
    final_model.fit(X_all, y_all)

    if args.dry_run:
        print("\n[dry-run] Outputs not saved.")
        return

    print("\nSaving outputs...")
    oof_path = PARQUET_DIR / "debiased_pd7day_oof.parquet"
    df[["interval_dt", "run_time", "oof_debiased_rrp"]].to_parquet(oof_path, index=False)
    print(f"  OOF series: {oof_path}")

    model_path = MODELS_DIR / "lgbm_final.pkl"
    with model_path.open("wb") as f:
        pickle.dump(final_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Final model: {model_path}")

    payload = {
        "features": FEATURES,
        "target": TARGET,
        "rows": int(len(df)),
        "runs": int(df["run_time"].nunique()),
        "interval_start": str(df["interval_dt"].min()),
        "interval_end": str(df["interval_dt"].max()),
        "metrics": metrics,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
