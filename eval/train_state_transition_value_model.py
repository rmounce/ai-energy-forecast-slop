#!/usr/bin/env python3
"""
Train a small diagnostic model on state-transition labels.

This is the first modeling scaffold for the inventory-discipline branch. It
uses only production-side / current-time features from
build_state_transition_label_dataset.py and predicts oracle-vs-target path
labels over 30-60 minute horizons.

The output is intentionally diagnostic: metrics, feature importance, and
optional model bundles. A good result here means "there is learnable signal";
it is not yet a production control integration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_TARGETS = [
    "oracle_minus_target_step_pnl",
    "oracle_minus_target_soc_delta_kwh",
    "oracle_minus_target_throughput_kwh",
    "oracle_minus_target_import_kwh",
    "oracle_minus_target_export_kwh",
    "oracle_minus_target_curtail_kwh",
]
DEFAULT_FEATURES = [
    "horizon_steps",
    "soc_prev_kwh",
    "actual_general_price_mwh",
    "actual_feed_in_price_mwh",
    "actual_net_load_kw",
    "actual_load_kw",
    "actual_pv_kw",
    "forecast_feed_in_step0_mwh",
    "forecast_feed_in_mean_next_1h_mwh",
    "forecast_feed_in_mean_next_4h_mwh",
    "forecast_buy_mean_next_1h_mwh",
    "forecast_buy_mean_next_4h_mwh",
    "forecast_sell_mean_next_1h_mwh",
    "forecast_sell_mean_next_4h_mwh",
    "strategic_soc_target_kwh",
    "terminal_energy_value_per_kwh",
]
LGBM_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find labels file: {path_arg}")


def _parse_csv_list(value: str | None, default: Iterable[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _load_labels(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("Labels file must include a time column")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(["time", "horizon_steps"], kind="stable").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    minutes = out["time"].dt.hour * 60 + out["time"].dt.minute
    day_frac = minutes / (24.0 * 60.0)
    dow_frac = out["time"].dt.dayofweek / 7.0
    out["time_sin"] = np.sin(2.0 * np.pi * day_frac)
    out["time_cos"] = np.cos(2.0 * np.pi * day_frac)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow_frac)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow_frac)
    return out


def build_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = add_time_features(df)
    features = list(feature_cols)
    for col in ["time_sin", "time_cos", "dow_sin", "dow_cos"]:
        if col not in features:
            features.append(col)
    missing = [col for col in features if col not in out.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = out[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X, features


def time_ordered_split(df: pd.DataFrame, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique_times = np.array(sorted(df["time"].drop_duplicates()))
    if len(unique_times) < 3:
        raise ValueError("Need at least 3 unique timestamps for a time-ordered split")
    split_idx = int(np.floor(len(unique_times) * float(train_fraction)))
    split_idx = min(max(split_idx, 1), len(unique_times) - 1)
    cutoff = unique_times[split_idx]
    train_mask = df["time"].to_numpy() < cutoff
    val_mask = ~train_mask
    return np.where(train_mask)[0], np.where(val_mask)[0]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    baseline_mae = float(mean_absolute_error(y_true, baseline_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    baseline_rmse = float(mean_squared_error(y_true, baseline_pred) ** 0.5)
    if len(np.unique(y_true)) > 1:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float("nan")
    nonzero = np.abs(y_true) > 1e-9
    if nonzero.any():
        sign_accuracy = float(np.mean(np.sign(y_true[nonzero]) == np.sign(y_pred[nonzero])))
        baseline_sign_accuracy = float(np.mean(np.sign(y_true[nonzero]) == np.sign(baseline_pred[nonzero])))
    else:
        sign_accuracy = float("nan")
        baseline_sign_accuracy = float("nan")
    return {
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": baseline_mae - mae,
        "mae_improvement_pct": (baseline_mae - mae) / baseline_mae if baseline_mae > 0 else float("nan"),
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "r2": r2,
        "sign_accuracy": sign_accuracy,
        "baseline_sign_accuracy": baseline_sign_accuracy,
    }


def train_one_target(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    target: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
    train_valid = np.intersect1d(train_idx, np.where(valid)[0], assume_unique=False)
    val_valid = np.intersect1d(val_idx, np.where(valid)[0], assume_unique=False)
    if len(train_valid) < 20 or len(val_valid) < 10:
        raise ValueError(f"Not enough valid rows for target {target}: train={len(train_valid)} val={len(val_valid)}")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X.iloc[train_valid],
        y.iloc[train_valid],
        eval_set=[(X.iloc[val_valid], y.iloc[val_valid])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )

    baseline_value = float(y.iloc[train_valid].median())
    rows: list[dict[str, float | int | str]] = []
    pred_rows: list[pd.DataFrame] = []
    for split_name, idx in [("train", train_valid), ("validation", val_valid)]:
        pred = model.predict(X.iloc[idx])
        baseline = np.full(len(idx), baseline_value, dtype=np.float64)
        split_df = df.iloc[idx].copy()
        split_df["target"] = target
        split_df["split"] = split_name
        split_df["y_true"] = y.iloc[idx].to_numpy(dtype=np.float64)
        split_df["y_pred"] = pred
        split_df["baseline_pred"] = baseline
        pred_rows.append(
            split_df[
                [
                    "time",
                    "horizon_steps",
                    "target",
                    "split",
                    "y_true",
                    "y_pred",
                    "baseline_pred",
                    "actual_feed_in_price_mwh",
                    "actual_net_load_kw",
                    "soc_prev_kwh",
                ]
            ]
        )
        for horizon, group in split_df.groupby("horizon_steps", sort=True):
            metrics = _metrics(
                group["y_true"].to_numpy(dtype=np.float64),
                group["y_pred"].to_numpy(dtype=np.float64),
                group["baseline_pred"].to_numpy(dtype=np.float64),
            )
            rows.append(
                {
                    "target": target,
                    "split": split_name,
                    "horizon_steps": int(horizon),
                    "rows": int(len(group)),
                    "best_iteration": int(model.best_iteration_ or model.n_estimators),
                    **metrics,
                }
            )
        metrics = _metrics(
            split_df["y_true"].to_numpy(dtype=np.float64),
            split_df["y_pred"].to_numpy(dtype=np.float64),
            split_df["baseline_pred"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "target": target,
                "split": split_name,
                "horizon_steps": -1,
                "rows": int(len(split_df)),
                "best_iteration": int(model.best_iteration_ or model.n_estimators),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)
    predictions_df = pd.concat(pred_rows, ignore_index=True)
    return model, metrics_df, predictions_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="State-transition label CSV/parquet path or filename under eval/results")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS), help="Comma-separated label targets")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES), help="Comma-separated feature columns")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit after sorting")
    parser.add_argument("--model-dir", default=None, help="Optional directory to save joblib model bundles")
    args = parser.parse_args()

    labels_path = _resolve_path(args.labels)
    df = _load_labels(labels_path)
    if args.max_rows is not None:
        df = df.head(max(0, int(args.max_rows))).copy()
    targets = _parse_csv_list(args.targets, DEFAULT_TARGETS)
    features_requested = _parse_csv_list(args.features, DEFAULT_FEATURES)
    missing_targets = [col for col in targets if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    X, feature_cols = build_feature_frame(df, features_requested)
    train_idx, val_idx = time_ordered_split(df, args.train_fraction)

    models: dict[str, lgb.LGBMRegressor] = {}
    metrics_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []

    for target in targets:
        model, metrics_df, predictions_df = train_one_target(
            df=df,
            X=X,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        models[target] = model
        metrics_parts.append(metrics_df)
        pred_parts.append(predictions_df)
        importance_parts.append(
            pd.DataFrame(
                {
                    "target": target,
                    "feature": feature_cols,
                    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                    "importance_split": model.booster_.feature_importance(importance_type="split"),
                }
            ).sort_values(["importance_gain", "importance_split"], ascending=False)
        )

    metrics = pd.concat(metrics_parts, ignore_index=True)
    predictions = pd.concat(pred_parts, ignore_index=True)
    importances = pd.concat(importance_parts, ignore_index=True)

    metrics_path = RESULTS_DIR / f"{args.output_prefix}_state_value_model_metrics.csv"
    pred_path = RESULTS_DIR / f"{args.output_prefix}_state_value_model_predictions.parquet"
    imp_path = RESULTS_DIR / f"{args.output_prefix}_state_value_model_feature_importance.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(pred_path, index=False)
    importances.to_csv(imp_path, index=False)

    manifest = {
        "labels": str(labels_path),
        "output_prefix": args.output_prefix,
        "targets": targets,
        "features": feature_cols,
        "rows": int(len(df)),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "train_fraction": float(args.train_fraction),
        "metrics": str(metrics_path),
        "predictions": str(pred_path),
        "feature_importance": str(imp_path),
    }

    if args.model_dir:
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        model_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "models": models,
            "targets": targets,
            "features": feature_cols,
            "lgbm_params": LGBM_PARAMS,
            "labels": str(labels_path),
        }
        model_path = model_dir / f"{args.output_prefix}_state_value_models.joblib"
        joblib.dump(bundle, model_path)
        manifest["model_bundle"] = str(model_path)

    manifest_path = RESULTS_DIR / f"{args.output_prefix}_state_value_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] wrote {metrics_path}")
    print(f"[done] wrote {pred_path}")
    print(f"[done] wrote {imp_path}")
    print(f"[done] wrote {manifest_path}")
    val_overall = metrics[(metrics["split"] == "validation") & (metrics["horizon_steps"] == -1)]
    if not val_overall.empty:
        print("\nValidation overall:")
        print(
            val_overall[
                [
                    "target",
                    "rows",
                    "mae",
                    "baseline_mae",
                    "mae_improvement_pct",
                    "r2",
                    "sign_accuracy",
                    "baseline_sign_accuracy",
                ]
            ].to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
