#!/usr/bin/env python3
"""
Train state-transition direction classifiers on one window and evaluate on another.

This is a diagnostic for regime transfer. It asks whether labels learned on one corrected
rolling window rank the same oracle-vs-target path events on another window, without relying on
the within-window time split used by train_state_transition_direction_model.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.train_state_transition_direction_model import (  # noqa: E402
    LGBM_CLASSIFIER_PARAMS,
    _best_f1_threshold,
    _metric_row,
    add_direction_labels,
)
from eval.train_state_transition_value_model import (  # noqa: E402
    DEFAULT_FEATURES,
    RESULTS_DIR,
    _load_label_files,
    _parse_csv_list,
    build_feature_frame,
)


def _load_direction_frame(
    label_args: list[str],
    *,
    pnl_deadband: float,
    kwh_deadband: float,
) -> tuple[pd.DataFrame, list[Path]]:
    df, paths = _load_label_files(label_args)
    df, _ = add_direction_labels(df, pnl_deadband=pnl_deadband, kwh_deadband=kwh_deadband)
    return df, paths


def _valid_indices(df: pd.DataFrame, X: pd.DataFrame, label: str) -> np.ndarray:
    y = pd.to_numeric(df[label], errors="coerce")
    valid = y.notna().to_numpy() & X.notna().all(axis=1).to_numpy()
    return np.where(valid)[0]


def _score_split(
    *,
    df: pd.DataFrame,
    idx: np.ndarray,
    label: str,
    split: str,
    y_score: np.ndarray,
    threshold: float,
    best_iteration: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_df = df.iloc[idx].copy()
    split_df["label"] = label
    split_df["split"] = split
    split_df["y_true"] = pd.to_numeric(split_df[label], errors="coerce").astype(int)
    split_df["y_score"] = y_score
    split_df["threshold"] = threshold

    metrics_rows: list[dict[str, float | int | str]] = []
    for horizon, group in split_df.groupby("horizon_steps", sort=True):
        metrics_rows.append(
            _metric_row(
                label=label,
                split=split,
                horizon_steps=int(horizon),
                y_true=group["y_true"].to_numpy(dtype=int),
                y_score=group["y_score"].to_numpy(dtype=float),
                threshold=threshold,
                best_iteration=best_iteration,
            )
        )
    metrics_rows.append(
        _metric_row(
            label=label,
            split=split,
            horizon_steps=-1,
            y_true=split_df["y_true"].to_numpy(dtype=int),
            y_score=split_df["y_score"].to_numpy(dtype=float),
            threshold=threshold,
            best_iteration=best_iteration,
        )
    )

    pred_cols = [
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
    return pd.DataFrame(metrics_rows), split_df[pred_cols]


def _label_distribution(df: pd.DataFrame, labels: list[str], *, split: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for label in labels:
        if label not in df.columns:
            continue
        valid = pd.to_numeric(df[label], errors="coerce").dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "split": split,
                "label": label,
                "rows": int(len(valid)),
                "positive_rate": float(valid.mean()),
                "mean_actual_feed_in_price_mwh": float(pd.to_numeric(df["actual_feed_in_price_mwh"], errors="coerce").mean()),
                "mean_actual_net_load_kw": float(pd.to_numeric(df["actual_net_load_kw"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-labels", required=True, help="Comma-separated label files for training")
    parser.add_argument("--test-labels", required=True, help="Comma-separated label files for cross-window test")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--direction-labels", default="pnl_gain,throughput_down,grid_exchange_down,soc_down")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--pnl-deadband", type=float, default=0.01)
    parser.add_argument("--kwh-deadband", type=float, default=0.10)
    args = parser.parse_args()

    train_args = _parse_csv_list(args.train_labels, [])
    test_args = _parse_csv_list(args.test_labels, [])
    labels = _parse_csv_list(args.direction_labels, [])
    if not train_args or not test_args:
        raise ValueError("--train-labels and --test-labels must be non-empty")
    if not labels:
        raise ValueError("--direction-labels must be non-empty")

    train_df, train_paths = _load_direction_frame(
        train_args,
        pnl_deadband=float(args.pnl_deadband),
        kwh_deadband=float(args.kwh_deadband),
    )
    test_df, test_paths = _load_direction_frame(
        test_args,
        pnl_deadband=float(args.pnl_deadband),
        kwh_deadband=float(args.kwh_deadband),
    )

    features_requested = _parse_csv_list(args.features, DEFAULT_FEATURES)
    X_train, feature_cols = build_feature_frame(train_df, features_requested)
    X_test, _ = build_feature_frame(test_df, features_requested)
    X_test = X_test[feature_cols]

    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []

    for label in labels:
        if label not in train_df.columns or label not in test_df.columns:
            skipped.append({"label": label, "reason": "missing label"})
            continue
        train_idx = _valid_indices(train_df, X_train, label)
        test_idx = _valid_indices(test_df, X_test, label)
        y_train = pd.to_numeric(train_df.iloc[train_idx][label], errors="coerce").astype(int)
        y_test = pd.to_numeric(test_df.iloc[test_idx][label], errors="coerce").astype(int)
        if len(train_idx) < 30 or len(test_idx) < 10:
            skipped.append({"label": label, "reason": f"not enough rows train={len(train_idx)} test={len(test_idx)}"})
            continue
        if y_train.nunique() < 2:
            skipped.append({"label": label, "reason": "training split has one class"})
            continue
        if y_test.nunique() < 2:
            skipped.append({"label": label, "reason": "test split has one class"})
            continue

        model = lgb.LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
        model.fit(X_train.iloc[train_idx], y_train)
        train_score = model.predict_proba(X_train.iloc[train_idx])[:, 1]
        threshold, _ = _best_f1_threshold(y_train.to_numpy(dtype=int), train_score)
        test_score = model.predict_proba(X_test.iloc[test_idx])[:, 1]
        best_iteration = int(model.best_iteration_ or model.n_estimators)

        train_metrics, train_pred = _score_split(
            df=train_df,
            idx=train_idx,
            label=label,
            split="train",
            y_score=train_score,
            threshold=threshold,
            best_iteration=best_iteration,
        )
        test_metrics, test_pred = _score_split(
            df=test_df,
            idx=test_idx,
            label=label,
            split="cross_window",
            y_score=test_score,
            threshold=threshold,
            best_iteration=best_iteration,
        )
        metric_parts.extend([train_metrics, test_metrics])
        prediction_parts.extend([train_pred, test_pred])
        importance_parts.append(
            pd.DataFrame(
                {
                    "label": label,
                    "feature": feature_cols,
                    "importance_gain": model.booster_.feature_importance(importance_type="gain"),
                    "importance_split": model.booster_.feature_importance(importance_type="split"),
                }
            ).sort_values(["label", "importance_gain", "importance_split"], ascending=[True, False, False])
        )

    if not metric_parts:
        raise ValueError(f"No labels trained; skipped={skipped}")

    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    importances = pd.concat(importance_parts, ignore_index=True)
    distributions = pd.concat(
        [
            _label_distribution(train_df, labels, split="train"),
            _label_distribution(test_df, labels, split="cross_window"),
        ],
        ignore_index=True,
    )

    metrics_path = RESULTS_DIR / f"{args.output_prefix}_cross_window_direction_metrics.csv"
    pred_path = RESULTS_DIR / f"{args.output_prefix}_cross_window_direction_predictions.parquet"
    imp_path = RESULTS_DIR / f"{args.output_prefix}_cross_window_direction_feature_importance.csv"
    dist_path = RESULTS_DIR / f"{args.output_prefix}_cross_window_direction_label_distribution.csv"
    manifest_path = RESULTS_DIR / f"{args.output_prefix}_cross_window_direction_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_parquet(pred_path, index=False)
    importances.to_csv(imp_path, index=False)
    distributions.to_csv(dist_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "train_labels": [str(path) for path in train_paths],
                "test_labels": [str(path) for path in test_paths],
                "direction_labels": labels,
                "trained_labels": sorted(metrics["label"].unique().tolist()),
                "skipped": skipped,
                "features": feature_cols,
                "pnl_deadband": float(args.pnl_deadband),
                "kwh_deadband": float(args.kwh_deadband),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "metrics": str(metrics_path),
                "predictions": str(pred_path),
                "feature_importance": str(imp_path),
                "label_distribution": str(dist_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"[done] wrote {metrics_path}")
    print(f"[done] wrote {pred_path}")
    print(f"[done] wrote {imp_path}")
    print(f"[done] wrote {dist_path}")
    print(f"[done] wrote {manifest_path}")
    if skipped:
        print("\nSkipped labels:")
        print(pd.DataFrame(skipped).to_string(index=False))
    overall = metrics[(metrics["split"] == "cross_window") & (metrics["horizon_steps"] == -1)]
    if not overall.empty:
        print("\nCross-window overall:")
        print(
            overall[
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
