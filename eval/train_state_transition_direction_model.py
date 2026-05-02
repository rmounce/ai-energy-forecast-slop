#!/usr/bin/env python3
"""
Train small diagnostic classifiers on state-transition direction labels.

The state-transition regression labels are zero-inflated: many prefixes match the oracle closely,
while the decision-relevant cases live in directional tails. This companion asks a simpler
question before another controller hook: can current-time / forecast-shape features identify
when the oracle wants a materially different short-horizon path?

Outputs are diagnostic metrics and feature importances. This does not create a production
dispatch artifact.
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.train_state_transition_value_model import (  # noqa: E402
    DEFAULT_FEATURES,
    RESULTS_DIR,
    _load_label_files,
    _load_table,
    _parse_csv_list,
    _resolve_path,
    build_feature_frame,
    build_tier1_vector_feature_frame,
    time_ordered_split,
)


LGBM_CLASSIFIER_PARAMS = {
    "objective": "binary",
    "n_estimators": 500,
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


def add_direction_labels(
    df: pd.DataFrame,
    *,
    pnl_deadband: float,
    kwh_deadband: float,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["oracle_minus_target_grid_exchange_kwh"] = (
        out["oracle_minus_target_import_kwh"] + out["oracle_minus_target_export_kwh"]
    )
    specs = {
        "pnl_gain": ("oracle_minus_target_step_pnl", "gt", pnl_deadband),
        "pnl_loss": ("oracle_minus_target_step_pnl", "lt", -pnl_deadband),
        "soc_up": ("oracle_minus_target_soc_delta_kwh", "gt", kwh_deadband),
        "soc_down": ("oracle_minus_target_soc_delta_kwh", "lt", -kwh_deadband),
        "throughput_up": ("oracle_minus_target_throughput_kwh", "gt", kwh_deadband),
        "throughput_down": ("oracle_minus_target_throughput_kwh", "lt", -kwh_deadband),
        "import_up": ("oracle_minus_target_import_kwh", "gt", kwh_deadband),
        "import_down": ("oracle_minus_target_import_kwh", "lt", -kwh_deadband),
        "export_up": ("oracle_minus_target_export_kwh", "gt", kwh_deadband),
        "export_down": ("oracle_minus_target_export_kwh", "lt", -kwh_deadband),
        "curtail_up": ("oracle_minus_target_curtail_kwh", "gt", kwh_deadband),
        "curtail_down": ("oracle_minus_target_curtail_kwh", "lt", -kwh_deadband),
        "grid_exchange_up": ("oracle_minus_target_grid_exchange_kwh", "gt", kwh_deadband),
        "grid_exchange_down": ("oracle_minus_target_grid_exchange_kwh", "lt", -kwh_deadband),
    }
    labels: list[str] = []
    for label, (col, op, threshold) in specs.items():
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        if op == "gt":
            out[label] = (values > threshold).astype("float")
        else:
            out[label] = (values < threshold).astype("float")
        out.loc[values.isna(), label] = np.nan
        labels.append(label)
    return out, labels


def _parse_optional_labels(value: str | None, default: Iterable[str]) -> list[str]:
    parsed = _parse_csv_list(value, [])
    return parsed if parsed else list(default)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = y_score >= threshold
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _metric_row(
    *,
    label: str,
    split: str,
    horizon_steps: int,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    best_iteration: int,
) -> dict[str, float | int | str]:
    y_pred = y_score >= threshold
    positive_rate = float(np.mean(y_true))
    baseline_accuracy = float(max(positive_rate, 1.0 - positive_rate))
    ap = (
        float(average_precision_score(y_true, y_score))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    return {
        "label": label,
        "split": split,
        "horizon_steps": int(horizon_steps),
        "rows": int(len(y_true)),
        "positive_rate": positive_rate,
        "threshold": float(threshold),
        "best_iteration": int(best_iteration),
        "roc_auc": _safe_auc(y_true, y_score),
        "average_precision": ap,
        "ap_lift_over_rate": ap / positive_rate if positive_rate > 0 and np.isfinite(ap) else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "baseline_accuracy": baseline_accuracy,
        "accuracy_lift": float(accuracy_score(y_true, y_pred) - baseline_accuracy),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def train_one_label(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    label: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[lgb.LGBMClassifier, pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df[label], errors="coerce")
    valid = y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
    train_valid = np.intersect1d(train_idx, np.where(valid)[0], assume_unique=False)
    val_valid = np.intersect1d(val_idx, np.where(valid)[0], assume_unique=False)
    if len(train_valid) < 30 or len(val_valid) < 10:
        raise ValueError(f"Not enough valid rows for label {label}: train={len(train_valid)} val={len(val_valid)}")
    if y.iloc[train_valid].nunique() < 2:
        raise ValueError(f"Training split has only one class for label {label}")

    model = lgb.LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
    model.fit(
        X.iloc[train_valid],
        y.iloc[train_valid].astype(int),
        eval_set=[(X.iloc[val_valid], y.iloc[val_valid].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(period=0)],
    )
    train_score = model.predict_proba(X.iloc[train_valid])[:, 1]
    threshold, _ = _best_f1_threshold(y.iloc[train_valid].to_numpy(dtype=int), train_score)

    rows: list[dict[str, float | int | str]] = []
    pred_parts: list[pd.DataFrame] = []
    best_iteration = int(model.best_iteration_ or model.n_estimators)
    for split_name, idx in [("train", train_valid), ("validation", val_valid)]:
        y_true = y.iloc[idx].to_numpy(dtype=int)
        y_score = model.predict_proba(X.iloc[idx])[:, 1]
        split_df = df.iloc[idx].copy()
        split_df["label"] = label
        split_df["split"] = split_name
        split_df["y_true"] = y_true
        split_df["y_score"] = y_score
        split_df["threshold"] = threshold
        pred_parts.append(
            split_df[
                [
                    "time",
                    "horizon_steps",
                    "label_file",
                    "label",
                    "split",
                    "y_true",
                    "y_score",
                    "threshold",
                    "actual_feed_in_price_mwh",
                    "actual_net_load_kw",
                    "soc_prev_kwh",
                ]
            ]
        )
        for horizon, group in split_df.groupby("horizon_steps", sort=True):
            rows.append(
                _metric_row(
                    label=label,
                    split=split_name,
                    horizon_steps=int(horizon),
                    y_true=group["y_true"].to_numpy(dtype=int),
                    y_score=group["y_score"].to_numpy(dtype=float),
                    threshold=threshold,
                    best_iteration=best_iteration,
                )
            )
        rows.append(
            _metric_row(
                label=label,
                split=split_name,
                horizon_steps=-1,
                y_true=y_true,
                y_score=y_score,
                threshold=threshold,
                best_iteration=best_iteration,
            )
        )

    return model, pd.DataFrame(rows), pd.concat(pred_parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Comma-separated state-transition label files")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--direction-labels", default=None, help="Comma-separated derived direction labels")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES), help="Comma-separated feature columns")
    parser.add_argument("--pnl-deadband", type=float, default=0.01)
    parser.add_argument("--kwh-deadband", type=float, default=0.10)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--vector-rows", default=None, help="Optional h0-h11 vector rows CSV/parquet")
    parser.add_argument("--vector-source", default="model_a_hybrid")
    parser.add_argument("--model-dir", default=None, help="Optional directory to save the diagnostic model bundle")
    args = parser.parse_args()

    label_args = _parse_csv_list(args.labels, [])
    if not label_args:
        raise ValueError("--labels must include at least one label file")
    df, label_paths = _load_label_files(label_args)
    if args.max_rows is not None:
        df = df.head(max(0, int(args.max_rows))).copy()
    df, default_labels = add_direction_labels(
        df,
        pnl_deadband=float(args.pnl_deadband),
        kwh_deadband=float(args.kwh_deadband),
    )
    direction_labels = _parse_optional_labels(args.direction_labels, default_labels)
    missing_labels = [label for label in direction_labels if label not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing direction labels: {missing_labels}")

    features_requested = _parse_csv_list(args.features, DEFAULT_FEATURES)
    vector_path: Path | None = None
    vector_feature_cols: list[str] = []
    if args.vector_rows:
        vector_path = _resolve_path(args.vector_rows)
        vector_features, vector_feature_cols = build_tier1_vector_feature_frame(
            _load_table(vector_path),
            source=args.vector_source,
        )
        before_rows = len(df)
        df = df.merge(vector_features, on="time", how="left", validate="many_to_one")
        matched = int(df[vector_feature_cols].notna().any(axis=1).sum())
        if matched == 0:
            raise ValueError(
                f"No label rows matched vector features from {vector_path.name} for source {args.vector_source}"
            )
        print(
            f"[info] joined {len(vector_feature_cols)} vector features from {vector_path.name}: "
            f"{matched}/{before_rows} label rows matched"
        )
        for col in vector_feature_cols:
            if col not in features_requested:
                features_requested.append(col)

    X, feature_cols = build_feature_frame(df, features_requested)
    train_idx, val_idx = time_ordered_split(df, args.train_fraction)

    metrics_parts: list[pd.DataFrame] = []
    pred_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    models: dict[str, lgb.LGBMClassifier] = {}
    thresholds: dict[str, float] = {}
    skipped: list[dict[str, str]] = []

    for label in direction_labels:
        try:
            model, metrics_df, predictions_df = train_one_label(
                df=df,
                X=X,
                label=label,
                train_idx=train_idx,
                val_idx=val_idx,
            )
        except ValueError as exc:
            skipped.append({"label": label, "reason": str(exc)})
            continue
        models[label] = model
        threshold_rows = metrics_df[
            (metrics_df["split"] == "train")
            & (metrics_df["horizon_steps"] == -1)
            & (metrics_df["label"] == label)
        ]
        thresholds[label] = (
            float(threshold_rows["threshold"].iloc[0])
            if not threshold_rows.empty
            else float(predictions_df["threshold"].iloc[0])
        )
        metrics_parts.append(metrics_df)
        pred_parts.append(predictions_df)
        importance_parts.append(
            pd.DataFrame(
                {
                    "label": label,
                    "feature": feature_cols,
                    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                    "importance_split": model.booster_.feature_importance(importance_type="split"),
                }
            ).sort_values(["importance_gain", "importance_split"], ascending=False)
        )

    if not metrics_parts:
        raise ValueError(f"No direction labels could be trained; skipped={skipped}")

    metrics = pd.concat(metrics_parts, ignore_index=True)
    predictions = pd.concat(pred_parts, ignore_index=True)
    importances = pd.concat(importance_parts, ignore_index=True)

    metrics_path = RESULTS_DIR / f"{args.output_prefix}_direction_model_metrics.csv"
    pred_path = RESULTS_DIR / f"{args.output_prefix}_direction_model_predictions.parquet"
    imp_path = RESULTS_DIR / f"{args.output_prefix}_direction_model_feature_importance.csv"
    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(pred_path, index=False)
    importances.to_csv(imp_path, index=False)

    manifest = {
        "labels": [str(path) for path in label_paths],
        "output_prefix": args.output_prefix,
        "direction_labels": direction_labels,
        "trained_labels": sorted(metrics["label"].unique().tolist()),
        "skipped": skipped,
        "features": feature_cols,
        "vector_rows": str(vector_path) if vector_path is not None else None,
        "vector_source": args.vector_source if vector_path is not None else None,
        "vector_feature_count": len(vector_feature_cols),
        "pnl_deadband": float(args.pnl_deadband),
        "kwh_deadband": float(args.kwh_deadband),
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
            "thresholds": thresholds,
            "direction_labels": direction_labels,
            "trained_labels": sorted(models.keys()),
            "features": feature_cols,
            "lgbm_params": LGBM_CLASSIFIER_PARAMS,
            "pnl_deadband": float(args.pnl_deadband),
            "kwh_deadband": float(args.kwh_deadband),
            "labels": [str(path) for path in label_paths],
            "vector_rows": str(vector_path) if vector_path is not None else None,
            "vector_source": args.vector_source if vector_path is not None else None,
            "vector_feature_count": len(vector_feature_cols),
        }
        model_path = model_dir / f"{args.output_prefix}_direction_models.joblib"
        joblib.dump(bundle, model_path)
        manifest["model_bundle"] = str(model_path)

    manifest_path = RESULTS_DIR / f"{args.output_prefix}_direction_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] wrote {metrics_path}")
    print(f"[done] wrote {pred_path}")
    print(f"[done] wrote {imp_path}")
    print(f"[done] wrote {manifest_path}")
    if skipped:
        print("\nSkipped labels:")
        print(pd.DataFrame(skipped).to_string(index=False))
    val_overall = metrics[(metrics["split"] == "validation") & (metrics["horizon_steps"] == -1)]
    if not val_overall.empty:
        print("\nValidation overall:")
        print(
            val_overall[
                [
                    "label",
                    "rows",
                    "positive_rate",
                    "roc_auc",
                    "average_precision",
                    "ap_lift_over_rate",
                    "f1",
                    "precision",
                    "recall",
                    "accuracy",
                    "baseline_accuracy",
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
