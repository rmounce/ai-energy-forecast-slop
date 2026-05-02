#!/usr/bin/env python3
"""
Score rolling raw rows with a saved state-transition direction model bundle.

This creates the external event score file consumed by rolling_mpc_eval.py's eval-only
grid-exchange reduction gate. The scorer is deliberately separate from the controller so model
quality can be inspected and falsified before any production interface exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.train_state_transition_value_model import (  # noqa: E402
    RESULTS_DIR,
    _load_table,
    _parse_csv_list,
    _resolve_path,
    add_time_features,
)


def load_raw_rows(path: Path, *, source: str, horizon_steps: list[int]) -> pd.DataFrame:
    raw = _load_table(path)
    if "time" not in raw.columns:
        raise ValueError("Raw file must include a time column")
    if "source" not in raw.columns:
        raise ValueError("Raw file must include a source column")
    raw["time"] = pd.to_datetime(raw["time"], utc=True)
    df = raw[raw["source"] == source].copy().sort_values("time", kind="stable").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found for source {source} in {path.name}")
    rows = []
    for horizon in horizon_steps:
        part = df.copy()
        part["horizon_steps"] = int(horizon)
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["label_file"] = path.name
    return out


def build_inference_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    enriched = add_time_features(df)
    for col in feature_cols:
        if col not in enriched.columns:
            enriched[col] = np.nan
    X = enriched[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return X


def score_bundle(
    df: pd.DataFrame,
    *,
    bundle: dict,
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = list(bundle["features"])
    X = build_inference_features(df, feature_cols)
    rows = []
    skipped = []
    for label in labels:
        model = bundle.get("models", {}).get(label)
        if model is None:
            skipped.append({"label": label, "reason": "label not present in model bundle"})
            continue
        scores = model.predict_proba(X)[:, 1]
        threshold = float(bundle.get("thresholds", {}).get(label, 0.5))
        part = df[["time", "source", "horizon_steps", "label_file"]].copy()
        for optional in [
            "actual_general_price_mwh",
            "actual_feed_in_price_mwh",
            "actual_net_load_kw",
            "soc_prev_kwh",
            "strategic_soc_target_kwh",
        ]:
            if optional in df.columns:
                part[optional] = df[optional]
        part["label"] = label
        part["split"] = "scored"
        part["y_score"] = scores
        part["threshold"] = threshold
        part["y_pred"] = scores >= threshold
        rows.append(part)
    if not rows:
        raise ValueError(f"No labels could be scored; skipped={skipped}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(skipped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", required=True, help="Saved direction model bundle joblib")
    parser.add_argument("--raw", required=True, help="Rolling raw parquet/CSV path or filename under eval/results")
    parser.add_argument("--source", default="model_a_hybrid", help="Raw source to score")
    parser.add_argument("--labels", default="", help="Comma-separated labels; default uses bundle trained labels")
    parser.add_argument("--horizon-steps", default="12", help="Comma-separated horizons to score")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    bundle_path = _resolve_path(args.model_bundle)
    raw_path = _resolve_path(args.raw)
    bundle = joblib.load(bundle_path)
    labels = _parse_csv_list(args.labels, bundle.get("trained_labels", []))
    if not labels:
        raise ValueError("No labels requested and bundle has no trained_labels")
    horizons = [int(part) for part in _parse_csv_list(args.horizon_steps, ["12"])]

    df = load_raw_rows(raw_path, source=args.source, horizon_steps=horizons)
    scored, skipped = score_bundle(df, bundle=bundle, labels=labels)

    out_path = RESULTS_DIR / f"{args.output_prefix}_direction_scores.parquet"
    csv_path = RESULTS_DIR / f"{args.output_prefix}_direction_scores.csv"
    manifest_path = RESULTS_DIR / f"{args.output_prefix}_direction_scores_manifest.json"
    scored.to_parquet(out_path, index=False)
    scored.to_csv(csv_path, index=False)
    manifest = {
        "model_bundle": str(bundle_path),
        "raw": str(raw_path),
        "source": args.source,
        "labels": labels,
        "horizon_steps": horizons,
        "rows": int(len(scored)),
        "score_parquet": str(out_path),
        "score_csv": str(csv_path),
        "skipped": skipped.to_dict(orient="records") if not skipped.empty else [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[done] wrote {out_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {manifest_path}")
    summary = (
        scored.groupby(["label", "horizon_steps"], dropna=False)
        .agg(rows=("y_score", "size"), mean_score=("y_score", "mean"), pred_rate=("y_pred", "mean"))
        .reset_index()
    )
    print("\nScore summary:")
    print(summary.to_string(index=False))
    if not skipped.empty:
        print("\nSkipped labels:")
        print(skipped.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
