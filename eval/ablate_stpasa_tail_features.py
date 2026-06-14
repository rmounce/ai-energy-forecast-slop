#!/usr/bin/env python3
"""Ablate STPASA renewable availability features for 28.5-72h price tail.

This is an offline validation step, not a production model trainer. It measures
whether STPASA fields add incremental signal when correcting the existing price
forecast tail residuals.

Target:
  residual_mwh = prediction_mwh - actual_rrp_mwh

Prediction use:
  corrected_price_mwh = prediction_mwh - predicted_residual_mwh

Feature sets:
  baseline         existing forecast context, time features, SDO where available
  baseline_stpasa  baseline + STPASA UIGF / wind / solar availability fields
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.analyze_lgbm_residual_drivers import (
    DEFAULT_ACTUALS,
    DEFAULT_LOG,
    DEFAULT_SDO,
    DEFAULT_STPASA,
    attach_actuals,
    horizon_bucket,
    latest_asof_by_target,
    load_price_log,
    load_source,
    validate_join_coverage,
    validate_source_horizon,
)
from eval.price_source_contracts import format_source_banner, require_apf_backed

RESULTS_DIR = REPO_ROOT / "eval" / "results"
LOCAL_TZ = "Australia/Brisbane"

BASELINE_FEATURES = [
    "pred_mwh",
    "horizon_hours",
    "target_hour_sin",
    "target_hour_cos",
    "target_dow_sin",
    "target_dow_cos",
    "target_month_sin",
    "target_month_cos",
    "creation_hour_sin",
    "creation_hour_cos",
    "creation_dow_sin",
    "creation_dow_cos",
    "power_pv",
    "temperature_adelaide",
    "humidity_adelaide",
    "wind_speed_adelaide",
    "sdo_scheduled_demand",
    "sdo_net_interchange",
]

STPASA_FEATURES = [
    "stpasa_uigf",
    "stpasa_total_intermittent_generation",
    "stpasa_ss_wind_uigf",
    "stpasa_ss_solar_uigf",
    "stpasa_ss_wind_capacity",
    "stpasa_ss_solar_capacity",
    "stpasa_wind_avail_frac",
    "stpasa_solar_avail_frac",
    "stpasa_net_load_proxy",
    "stpasa_source_horizon_hours",
]

LGBM_PARAMS = {
    "objective": "regression_l1",
    "n_estimators": 700,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for source_col, prefix in [
        ("forecast_target_time", "target"),
        ("forecast_creation_time", "creation"),
    ]:
        local = out[source_col].dt.tz_convert(LOCAL_TZ)
        out[f"{prefix}_hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
        out[f"{prefix}_hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
        out[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * local.dt.dayofweek / 7)
        out[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * local.dt.dayofweek / 7)
        if prefix == "target":
            out[f"{prefix}_month_sin"] = np.sin(2 * np.pi * (local.dt.month - 1) / 12)
            out[f"{prefix}_month_cos"] = np.cos(2 * np.pi * (local.dt.month - 1) / 12)
    return out


def add_stpasa_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stpasa_source_horizon_hours"] = (
        out["forecast_target_time"] - out["stpasa_run_time"]
    ).dt.total_seconds() / 3600.0
    out["stpasa_wind_avail_frac"] = (
        out["stpasa_ss_wind_uigf"] / out["stpasa_ss_wind_capacity"]
    ).replace([np.inf, -np.inf], np.nan)
    out["stpasa_solar_avail_frac"] = (
        out["stpasa_ss_solar_uigf"] / out["stpasa_ss_solar_capacity"]
    ).replace([np.inf, -np.inf], np.nan)
    out["stpasa_net_load_proxy"] = out["sdo_scheduled_demand"] - out["stpasa_uigf"]
    return out


def chronological_split(
    df: pd.DataFrame,
    *,
    time_col: str,
    val_fraction: float,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """Return train/validation masks using a chronological cutoff."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    ordered_times = pd.Series(df[time_col].dropna().sort_values().unique())
    if len(ordered_times) < 2:
        raise ValueError("not enough timestamps for chronological split")
    cutoff_idx = int(np.floor(len(ordered_times) * (1.0 - val_fraction)))
    cutoff_idx = min(max(cutoff_idx, 1), len(ordered_times) - 1)
    cutoff = pd.Timestamp(ordered_times.iloc[cutoff_idx])
    train_mask = df[time_col] < cutoff
    val_mask = df[time_col] >= cutoff
    return train_mask, val_mask, cutoff


def metric_rows(df: pd.DataFrame, *, model_name: str, split_name: str) -> list[dict]:
    rows = []
    for bucket in ["all", *sorted(df["horizon_bucket"].dropna().unique())]:
        sub = df if bucket == "all" else df[df["horizon_bucket"] == bucket]
        if sub.empty:
            continue
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "horizon_bucket": bucket,
                "n": int(len(sub)),
                "original_mae": float((sub["pred_mwh"] - sub["actual_rrp"]).abs().mean()),
                "corrected_mae": float((sub["corrected_mwh"] - sub["actual_rrp"]).abs().mean()),
                "mae_delta": float(
                    (sub["corrected_mwh"] - sub["actual_rrp"]).abs().mean()
                    - (sub["pred_mwh"] - sub["actual_rrp"]).abs().mean()
                ),
                "original_bias": float((sub["pred_mwh"] - sub["actual_rrp"]).mean()),
                "corrected_bias": float((sub["corrected_mwh"] - sub["actual_rrp"]).mean()),
            }
        )
    return rows


def train_residual_model(
    df: pd.DataFrame,
    *,
    features: list[str],
    train_mask: pd.Series,
    val_mask: pd.Series,
) -> tuple[lgb.LGBMRegressor, np.ndarray, np.ndarray]:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    X_train = df.loc[train_mask, features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    y_train = pd.to_numeric(df.loc[train_mask, "residual_mwh"], errors="coerce").astype(np.float32)
    X_val = df.loc[val_mask, features].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    y_val = pd.to_numeric(df.loc[val_mask, "residual_mwh"], errors="coerce").astype(np.float32)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model, model.predict(X_train), model.predict(X_val)


def prepare_dataset(args: argparse.Namespace) -> pd.DataFrame:
    require_apf_backed("amber_apf_lgbm")
    df = load_price_log(args.log_file, model_filter=args.model_filter)
    df = df[df["forecast_creation_time"] >= pd.Timestamp(args.since)]
    if args.until:
        df = df[df["forecast_creation_time"] < pd.Timestamp(args.until)]
    df["horizon_hours"] = (
        df["forecast_target_time"] - df["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    df = df[
        (df["horizon_hours"] >= args.tail_start_hours)
        & (df["horizon_hours"] <= args.max_horizon_hours)
    ].copy()
    if args.max_rows and len(df) > args.max_rows:
        df = df.sort_values("forecast_creation_time").tail(args.max_rows).copy()

    df = attach_actuals(df, args.actuals_30m)
    df = df.dropna(subset=["actual_rrp"]).copy()
    df["pred_mwh"] = df["prediction"] * 1000.0
    df["residual_mwh"] = df["pred_mwh"] - df["actual_rrp"]

    if args.sevendayoutlook.exists():
        sdo_source = load_source(
            args.sevendayoutlook,
            ["interval_dt", "run_time", "scheduled_demand", "net_interchange"],
        )
        df = latest_asof_by_target(
            df,
            sdo_source,
            left_target="forecast_target_time",
            left_time="forecast_creation_time",
            right_target="interval_dt",
            right_time="run_time",
            value_cols=["scheduled_demand", "net_interchange"],
            prefix="sdo",
        )
    else:
        df["sdo_scheduled_demand"] = np.nan
        df["sdo_net_interchange"] = np.nan

    stpasa_source = load_source(
        args.stpasa,
        [
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
    min_h, max_h = validate_source_horizon(
        stpasa_source,
        source_name="STPASA",
        min_horizon_hours=args.max_horizon_hours,
    )
    print(f"STPASA source horizon: {min_h:.1f}h -> {max_h:.1f}h")
    df = latest_asof_by_target(
        df,
        stpasa_source,
        left_target="forecast_target_time",
        left_time="forecast_creation_time",
        right_target="interval_dt",
        right_time="run_time",
        value_cols=[
            "uigf",
            "total_intermittent_generation",
            "ss_wind_uigf",
            "ss_solar_uigf",
            "ss_wind_capacity",
            "ss_solar_capacity",
        ],
        prefix="stpasa",
    )
    coverage = validate_join_coverage(
        df,
        prefix="stpasa",
        value_col="uigf",
        min_horizon_hours=args.tail_start_hours,
        max_horizon_hours=args.max_horizon_hours,
        min_coverage=args.min_stpasa_coverage,
    )
    print(f"STPASA tail join coverage: {coverage:.1%}")

    df = add_cyclic_time_features(df)
    df = add_stpasa_derived_features(df)
    df["horizon_bucket"] = horizon_bucket(df["horizon_hours"])
    return df.sort_values(["forecast_creation_time", "forecast_target_time"]).reset_index(drop=True)


def run_ablation(df: pd.DataFrame, *, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask, val_mask, cutoff = chronological_split(
        df,
        time_col="forecast_creation_time",
        val_fraction=val_fraction,
    )
    print(
        f"Split cutoff={cutoff}; train rows={int(train_mask.sum()):,}; "
        f"val rows={int(val_mask.sum()):,}"
    )

    feature_sets = {
        "baseline": BASELINE_FEATURES,
        "baseline_stpasa": BASELINE_FEATURES + STPASA_FEATURES,
    }
    metric_out = []
    importance_out = []

    for name, features in feature_sets.items():
        print(f"Training {name} residual corrector ({len(features)} features)...")
        model, train_pred, val_pred = train_residual_model(
            df,
            features=features,
            train_mask=train_mask,
            val_mask=val_mask,
        )

        for split_name, mask, residual_pred in [
            ("train", train_mask, train_pred),
            ("val", val_mask, val_pred),
        ]:
            scored = df.loc[mask].copy()
            scored["predicted_residual_mwh"] = residual_pred
            scored["corrected_mwh"] = scored["pred_mwh"] - scored["predicted_residual_mwh"]
            metric_out.extend(metric_rows(scored, model_name=name, split_name=split_name))

        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        importance_out.extend(
            {"model": name, "feature": feature, "importance": int(importance)}
            for feature, importance in importances.items()
        )

    return pd.DataFrame(metric_out), pd.DataFrame(importance_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--actuals-30m", type=Path, default=DEFAULT_ACTUALS)
    parser.add_argument("--sevendayoutlook", type=Path, default=DEFAULT_SDO)
    parser.add_argument("--stpasa", type=Path, default=DEFAULT_STPASA)
    parser.add_argument("--model-filter", default="price")
    parser.add_argument("--since", default="2026-04-01T00:00:00Z")
    parser.add_argument("--until", default=None)
    parser.add_argument("--tail-start-hours", type=float, default=28.5)
    parser.add_argument("--max-horizon-hours", type=float, default=72.0)
    parser.add_argument("--min-stpasa-coverage", type=float, default=0.95)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=RESULTS_DIR / "stpasa_tail_ablation",
    )
    args = parser.parse_args()

    if not args.stpasa.exists():
        raise FileNotFoundError(
            f"STPASA parquet not found: {args.stpasa}. "
            "Run ingest/backfill_stpasa_regionsolution.py first."
        )

    print("Preparing tail residual dataset...")
    print(format_source_banner("amber_apf_lgbm", prefix="Evaluating residuals for"))
    print(
        f"  log filter: model_name == {args.model_filter!r}; "
        "target residual = logged prediction - actual RRP"
    )
    df = prepare_dataset(args)
    print(
        f"Dataset rows={len(df):,}; creation window="
        f"{df['forecast_creation_time'].min()} -> {df['forecast_creation_time'].max()}"
    )

    metrics, importances = run_ablation(df, val_fraction=args.val_fraction)

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = prefix.with_name(prefix.name + "_metrics.csv")
    importance_path = prefix.with_name(prefix.name + "_feature_importance.csv")
    metrics.to_csv(metrics_path, index=False, float_format="%.6f")
    importances.to_csv(importance_path, index=False)

    print("\nValidation metrics:")
    print(
        metrics[(metrics["split"] == "val") & (metrics["horizon_bucket"] == "all")]
        .sort_values("model")
        .to_string(index=False)
    )
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {importance_path}")


if __name__ == "__main__":
    main()
