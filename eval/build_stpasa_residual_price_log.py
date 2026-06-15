#!/usr/bin/env python3
"""Build an APF-backed forecast log with STPASA residual tail correction.

This is an eval artifact generator. It keeps the incumbent `amber_apf_lgbm`
forecast curve as the base, trains a residual model on earlier APF log rows, and
only replaces validation-window 28.5-72h tail predictions with:

    corrected_mwh = logged_prediction_mwh - predicted_residual_mwh

The output is shaped like a price forecast log so `eval/rolling_mpc_eval.py` can
consume it via `AI_ENERGY_PRICE_FORECAST_LOG=...` without touching production.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-energy-forecast-slop-mplconfig")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.ablate_stpasa_tail_features import (  # noqa: E402
    BASELINE_FEATURES,
    RESULTS_DIR,
    STPASA_FEATURES,
    chronological_split,
    prepare_dataset,
    train_residual_model,
)
from eval.analyze_lgbm_residual_drivers import (  # noqa: E402
    DEFAULT_ACTUALS,
    DEFAULT_LOG,
    DEFAULT_SDO,
    DEFAULT_STPASA,
)
from eval.price_source_contracts import format_source_banner, require_apf_backed  # noqa: E402


TIMESTAMP_COLS = ["forecast_creation_time", "forecast_target_time"]


def _timestamp_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TIMESTAMP_COLS:
        out[col] = pd.to_datetime(out[col], utc=True, format="mixed")
    return out


def build_tail_adjustments(
    df: pd.DataFrame,
    *,
    val_fraction: float,
) -> tuple[pd.DataFrame, pd.Timestamp, int, int]:
    """Train on earlier rows and return validation tail corrections."""

    train_mask, val_mask, cutoff = chronological_split(
        df,
        time_col="forecast_creation_time",
        val_fraction=val_fraction,
    )
    features = BASELINE_FEATURES + STPASA_FEATURES
    model, _, val_pred = train_residual_model(
        df,
        features=features,
        train_mask=train_mask,
        val_mask=val_mask,
    )
    del model

    scored = df.loc[val_mask, TIMESTAMP_COLS + ["pred_mwh"]].copy()
    scored["predicted_residual_mwh"] = val_pred
    scored["corrected_prediction"] = (
        scored["pred_mwh"] - scored["predicted_residual_mwh"]
    ) / 1000.0
    scored = scored[TIMESTAMP_COLS + ["corrected_prediction"]]
    return scored, cutoff, int(train_mask.sum()), int(val_mask.sum())


def apply_tail_adjustments(
    base_log: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Replace matching base predictions with corrected validation predictions."""

    base = _timestamp_key_frame(base_log)
    adj = _timestamp_key_frame(adjustments)
    merged = base.merge(adj, on=TIMESTAMP_COLS, how="left")
    replace_mask = merged["corrected_prediction"].notna()
    merged.loc[replace_mask, "prediction"] = merged.loc[
        replace_mask, "corrected_prediction"
    ].to_numpy(dtype=np.float64)
    out = merged.drop(columns=["corrected_prediction"])
    return out, int(replace_mask.sum())


def load_base_log_window(
    path: Path,
    *,
    model_filter: str,
    creation_start: pd.Timestamp,
    creation_end: pd.Timestamp,
    max_horizon_hours: float,
) -> pd.DataFrame:
    """Load incumbent APF log rows for the candidate output window."""

    df = pd.read_csv(
        path,
        usecols=["forecast_creation_time", "forecast_target_time", "model_name", "prediction"],
        dtype_backend="pyarrow",
    )
    df = df[df["model_name"].astype(str) == model_filter].copy()
    df = _timestamp_key_frame(df)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["prediction"])
    df["horizon_hours"] = (
        df["forecast_target_time"] - df["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    df = df[
        (df["forecast_creation_time"] >= creation_start)
        & (df["forecast_creation_time"] < creation_end)
        & (df["horizon_hours"] >= 0.0)
        & (df["horizon_hours"] <= max_horizon_hours)
    ].copy()
    df = df.drop(columns=["horizon_hours"])
    return df.sort_values(TIMESTAMP_COLS).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--actuals-30m", type=Path, default=DEFAULT_ACTUALS)
    parser.add_argument("--sevendayoutlook", type=Path, default=DEFAULT_SDO)
    parser.add_argument("--stpasa", type=Path, default=DEFAULT_STPASA)
    parser.add_argument("--model-filter", default="price")
    parser.add_argument("--since", default="2026-04-01T00:00:00Z")
    parser.add_argument("--until", default="2026-05-13T00:00:00Z")
    parser.add_argument("--tail-start-hours", type=float, default=28.5)
    parser.add_argument("--max-horizon-hours", type=float, default=72.0)
    parser.add_argument("--min-stpasa-coverage", type=float, default=0.95)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=RESULTS_DIR / "amber_apf_lgbm_stpasa_residual_price_forecast_log.csv",
    )
    parser.add_argument(
        "--output-baseline-file",
        type=Path,
        default=None,
        help=(
            "Optional path for the matching unmodified APF baseline window. "
            "Useful for dispatch A/B runs so baseline and candidate parse the same small window."
        ),
    )
    args = parser.parse_args()

    require_apf_backed("amber_apf_lgbm")
    print(format_source_banner("amber_apf_lgbm", prefix="Building STPASA residual log from"))
    print(
        f"  log filter: model_name == {args.model_filter!r}; "
        "output remains derived from incumbent APF-backed predictions"
    )

    if not args.stpasa.exists():
        raise FileNotFoundError(
            f"STPASA parquet not found: {args.stpasa}. "
            "Run ingest/backfill_stpasa_regionsolution.py first."
        )

    df = prepare_dataset(args)
    adjustments, cutoff, train_rows, val_rows = build_tail_adjustments(
        df,
        val_fraction=args.val_fraction,
    )
    if adjustments.empty:
        raise ValueError("no validation adjustments were produced")

    creation_start = adjustments["forecast_creation_time"].min() - pd.Timedelta(hours=2)
    creation_end = adjustments["forecast_creation_time"].max() + pd.Timedelta(hours=2)
    base_log = load_base_log_window(
        args.log_file,
        model_filter=args.model_filter,
        creation_start=creation_start,
        creation_end=creation_end,
        max_horizon_hours=args.max_horizon_hours,
    )
    if base_log.empty:
        raise ValueError("base APF log window is empty")

    candidate, replaced = apply_tail_adjustments(base_log, adjustments)
    if replaced != len(adjustments):
        raise ValueError(
            f"Only replaced {replaced:,}/{len(adjustments):,} adjusted tail rows; "
            "base log timestamps did not line up"
        )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(args.output_file, index=False)
    if args.output_baseline_file is not None:
        args.output_baseline_file.parent.mkdir(parents=True, exist_ok=True)
        base_log.to_csv(args.output_baseline_file, index=False)

    print(
        f"Split cutoff={cutoff}; train tail rows={train_rows:,}; "
        f"validation tail rows={val_rows:,}"
    )
    print(
        f"Output creation window={candidate['forecast_creation_time'].min()} -> "
        f"{candidate['forecast_creation_time'].max()}; rows={len(candidate):,}; "
        f"tail rows corrected={replaced:,}"
    )
    print(f"Wrote {args.output_file}")
    if args.output_baseline_file is not None:
        print(f"Wrote baseline window {args.output_baseline_file}")


if __name__ == "__main__":
    main()
