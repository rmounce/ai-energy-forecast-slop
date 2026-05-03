#!/usr/bin/env python3
"""
Analyze calibration and threshold policy quality for direction-model scores.

Direction classifiers have shown useful ranking signal, but production progress needs calibrated
confidence bands rather than raw F1 thresholds. This script fits a simple isotonic calibration on
the training split contained in a prediction parquet, evaluates another split, and writes:

- calibrated prediction rows
- overall raw/calibrated ranking and calibration metrics
- reliability bins
- threshold policy sweeps
- top-score band precision/recall summaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

REQUIRED_COLUMNS = {"label", "split", "horizon_steps", "y_true", "y_score"}


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find prediction file: {path_arg}")


def _load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")
    out = df.copy()
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_score"] = pd.to_numeric(out["y_score"], errors="coerce")
    out["horizon_steps"] = pd.to_numeric(out["horizon_steps"], errors="raise").astype(int)
    return out.dropna(subset=["label", "split", "horizon_steps", "y_true", "y_score"]).reset_index(drop=True)


def _parse_csv_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or not value.strip():
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return ece


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def fit_isotonic(train_true: np.ndarray, train_score: np.ndarray) -> IsotonicRegression | None:
    if len(train_true) < 20 or len(np.unique(train_true)) < 2 or len(np.unique(train_score)) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(train_score, train_true)
    return model


def _calibrate_scores(
    train_score: np.ndarray,
    eval_score: np.ndarray,
    train_true: np.ndarray,
) -> tuple[np.ndarray, str]:
    model = fit_isotonic(train_true, train_score)
    if model is None:
        return np.clip(eval_score, 0.0, 1.0), "identity"
    return model.transform(eval_score), "isotonic"


def _metric_row(
    *,
    label: str,
    horizon_steps: int,
    train_rows: int,
    eval_rows: int,
    calibration_method: str,
    y_true: np.ndarray,
    raw_score: np.ndarray,
    calibrated_score: np.ndarray,
    n_bins: int,
) -> dict[str, float | int | str]:
    positive_rate = float(y_true.mean())
    raw_ap = _safe_ap(y_true, raw_score)
    cal_ap = _safe_ap(y_true, calibrated_score)
    return {
        "label": label,
        "horizon_steps": int(horizon_steps),
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "positive_rate": positive_rate,
        "calibration_method": calibration_method,
        "raw_roc_auc": _safe_auc(y_true, raw_score),
        "calibrated_roc_auc": _safe_auc(y_true, calibrated_score),
        "raw_average_precision": raw_ap,
        "calibrated_average_precision": cal_ap,
        "raw_ap_lift": raw_ap / positive_rate if positive_rate > 0 and np.isfinite(raw_ap) else float("nan"),
        "calibrated_ap_lift": cal_ap / positive_rate if positive_rate > 0 and np.isfinite(cal_ap) else float("nan"),
        "raw_brier": float(brier_score_loss(y_true, np.clip(raw_score, 0.0, 1.0))),
        "calibrated_brier": float(brier_score_loss(y_true, np.clip(calibrated_score, 0.0, 1.0))),
        "raw_ece": expected_calibration_error(y_true, np.clip(raw_score, 0.0, 1.0), n_bins=n_bins),
        "calibrated_ece": expected_calibration_error(y_true, calibrated_score, n_bins=n_bins),
    }


def reliability_bins(
    df: pd.DataFrame,
    *,
    score_col: str,
    label: str,
    horizon_steps: int,
    n_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    scores = df[score_col].to_numpy(dtype=float)
    y_true = df["y_true"].to_numpy(dtype=int)
    for bin_idx, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == 1.0:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        if not mask.any():
            continue
        rows.append(
            {
                "label": label,
                "horizon_steps": int(horizon_steps),
                "score_column": score_col,
                "bin": int(bin_idx),
                "score_min": float(low),
                "score_max": float(high),
                "rows": int(mask.sum()),
                "activation_rate": float(mask.mean()),
                "mean_score": float(scores[mask].mean()),
                "positive_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    df: pd.DataFrame,
    *,
    score_col: str,
    label: str,
    horizon_steps: int,
    thresholds: list[float],
) -> pd.DataFrame:
    y_true = df["y_true"].to_numpy(dtype=int)
    scores = df[score_col].to_numpy(dtype=float)
    base_rate = float(y_true.mean())
    rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        pred = scores >= threshold
        rows.append(
            {
                "label": label,
                "horizon_steps": int(horizon_steps),
                "score_column": score_col,
                "threshold": float(threshold),
                "rows": int(len(df)),
                "activated_rows": int(pred.sum()),
                "activation_rate": float(pred.mean()),
                "base_rate": base_rate,
                "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "precision_lift": (
                    float(precision_score(y_true, pred, zero_division=0)) / base_rate
                    if base_rate > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def top_fraction_summary(
    df: pd.DataFrame,
    *,
    score_col: str,
    label: str,
    horizon_steps: int,
    fractions: list[float],
) -> pd.DataFrame:
    ordered = df.sort_values(score_col, ascending=False, kind="stable")
    y_true = ordered["y_true"].to_numpy(dtype=int)
    base_rate = float(y_true.mean())
    rows: list[dict[str, float | int | str]] = []
    for frac in fractions:
        n = max(1, int(np.ceil(len(ordered) * frac)))
        selected = ordered.head(n)
        selected_true = selected["y_true"].to_numpy(dtype=int)
        precision = float(selected_true.mean())
        recall = float(selected_true.sum() / y_true.sum()) if y_true.sum() > 0 else float("nan")
        rows.append(
            {
                "label": label,
                "horizon_steps": int(horizon_steps),
                "score_column": score_col,
                "top_fraction": float(frac),
                "rows": int(len(ordered)),
                "selected_rows": int(n),
                "base_rate": base_rate,
                "precision": precision,
                "recall": recall,
                "precision_lift": precision / base_rate if base_rate > 0 else float("nan"),
                "min_selected_score": float(selected[score_col].min()),
            }
        )
    return pd.DataFrame(rows)


def _iter_groups(df: pd.DataFrame, labels: list[str]) -> list[tuple[str, int, pd.DataFrame]]:
    groups: list[tuple[str, int, pd.DataFrame]] = []
    for label in labels:
        label_df = df[df["label"] == label].copy()
        if label_df.empty:
            continue
        for horizon, group in label_df.groupby("horizon_steps", sort=True):
            groups.append((label, int(horizon), group.copy()))
        groups.append((label, -1, label_df.copy()))
    return groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Prediction parquet/CSV from direction model")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--eval-split", default="cross_window")
    parser.add_argument("--labels", default="", help="Comma-separated labels; default uses all labels")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--top-fractions", default="0.01,0.05,0.10,0.20")
    args = parser.parse_args()

    pred_path = _resolve_path(args.predictions)
    df = _load_predictions(pred_path)
    labels = _parse_csv_list(args.labels, sorted(df["label"].dropna().unique().tolist()))
    thresholds = [float(v) for v in _parse_csv_list(args.thresholds, [])]
    top_fractions = [float(v) for v in _parse_csv_list(args.top_fractions, [])]

    metric_rows: list[dict[str, float | int | str]] = []
    calibrated_parts: list[pd.DataFrame] = []
    reliability_parts: list[pd.DataFrame] = []
    threshold_parts: list[pd.DataFrame] = []
    top_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str | int]] = []

    for label, horizon, group in _iter_groups(df, labels):
        if horizon >= 0:
            train = group[(group["split"] == args.calibration_split) & (group["horizon_steps"] == horizon)].copy()
            eval_df = group[(group["split"] == args.eval_split) & (group["horizon_steps"] == horizon)].copy()
        else:
            train = group[group["split"] == args.calibration_split].copy()
            eval_df = group[group["split"] == args.eval_split].copy()
        if train.empty or eval_df.empty:
            skipped.append({"label": label, "horizon_steps": horizon, "reason": "missing split rows"})
            continue
        y_train = train["y_true"].to_numpy(dtype=int)
        y_eval = eval_df["y_true"].to_numpy(dtype=int)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_eval)) < 2:
            skipped.append({"label": label, "horizon_steps": horizon, "reason": "one-class split"})
            continue

        calibrated, method = _calibrate_scores(
            train["y_score"].to_numpy(dtype=float),
            eval_df["y_score"].to_numpy(dtype=float),
            y_train,
        )
        eval_scored = eval_df.copy()
        eval_scored["raw_score"] = eval_scored["y_score"].astype(float)
        eval_scored["calibrated_score"] = calibrated
        eval_scored["calibration_method"] = method
        eval_scored["calibration_horizon_steps"] = horizon
        calibrated_parts.append(eval_scored)

        metric_rows.append(
            _metric_row(
                label=label,
                horizon_steps=horizon,
                train_rows=len(train),
                eval_rows=len(eval_scored),
                calibration_method=method,
                y_true=y_eval,
                raw_score=eval_scored["raw_score"].to_numpy(dtype=float),
                calibrated_score=eval_scored["calibrated_score"].to_numpy(dtype=float),
                n_bins=int(args.bins),
            )
        )
        for score_col in ["raw_score", "calibrated_score"]:
            reliability_parts.append(
                reliability_bins(
                    eval_scored,
                    score_col=score_col,
                    label=label,
                    horizon_steps=horizon,
                    n_bins=int(args.bins),
                )
            )
            threshold_parts.append(
                threshold_sweep(
                    eval_scored,
                    score_col=score_col,
                    label=label,
                    horizon_steps=horizon,
                    thresholds=thresholds,
                )
            )
            top_parts.append(
                top_fraction_summary(
                    eval_scored,
                    score_col=score_col,
                    label=label,
                    horizon_steps=horizon,
                    fractions=top_fractions,
                )
            )

    if not metric_rows:
        raise ValueError(f"No calibration groups evaluated; skipped={skipped}")

    metrics = pd.DataFrame(metric_rows)
    calibrated_predictions = pd.concat(calibrated_parts, ignore_index=True)
    reliability = pd.concat(reliability_parts, ignore_index=True)
    thresholds_df = pd.concat(threshold_parts, ignore_index=True)
    top_df = pd.concat(top_parts, ignore_index=True)

    metrics_path = RESULTS_DIR / f"{args.output_prefix}_direction_calibration_metrics.csv"
    pred_out = RESULTS_DIR / f"{args.output_prefix}_direction_calibrated_predictions.parquet"
    reliability_path = RESULTS_DIR / f"{args.output_prefix}_direction_reliability_bins.csv"
    threshold_path = RESULTS_DIR / f"{args.output_prefix}_direction_threshold_sweep.csv"
    top_path = RESULTS_DIR / f"{args.output_prefix}_direction_top_fraction_summary.csv"
    manifest_path = RESULTS_DIR / f"{args.output_prefix}_direction_calibration_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    calibrated_predictions.to_parquet(pred_out, index=False)
    reliability.to_csv(reliability_path, index=False)
    thresholds_df.to_csv(threshold_path, index=False)
    top_df.to_csv(top_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "predictions": str(pred_path),
                "output_prefix": args.output_prefix,
                "calibration_split": args.calibration_split,
                "eval_split": args.eval_split,
                "labels": labels,
                "bins": int(args.bins),
                "thresholds": thresholds,
                "top_fractions": top_fractions,
                "metrics": str(metrics_path),
                "calibrated_predictions": str(pred_out),
                "reliability_bins": str(reliability_path),
                "threshold_sweep": str(threshold_path),
                "top_fraction_summary": str(top_path),
                "skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"[done] wrote {metrics_path}")
    print(f"[done] wrote {pred_out}")
    print(f"[done] wrote {reliability_path}")
    print(f"[done] wrote {threshold_path}")
    print(f"[done] wrote {top_path}")
    print(f"[done] wrote {manifest_path}")
    print("\nOverall metrics:")
    overall = metrics[metrics["horizon_steps"] == -1]
    print(
        overall[
            [
                "label",
                "eval_rows",
                "positive_rate",
                "raw_roc_auc",
                "calibrated_roc_auc",
                "raw_average_precision",
                "calibrated_average_precision",
                "raw_ece",
                "calibrated_ece",
                "raw_brier",
                "calibrated_brier",
            ]
        ].to_string(index=False)
    )
    if skipped:
        print("\nSkipped groups:")
        print(pd.DataFrame(skipped).to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
