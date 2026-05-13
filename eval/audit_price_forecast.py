#!/usr/bin/env python3
"""Audit price-forecast MAE/bias vs realised RRP, bucketed by Adelaide local
time-of-day and forecast horizon.

Generic companion to `audit_pd_direct_debiaser.py` — that one is specific to
PD-direct because it compares debiased vs raw vs realised. This one runs the
simpler forecast-vs-realised comparison and is intended for end-to-end models
(LGBM, TFT) that don't have a "raw upstream" to compare against.

Used to test whether the strategic LP over-confidence pattern seen in Run B v3
(`amber_apf_lgbm` and `model_a_hybrid` source contracts) is driven by bias in
the strategic price curve (Amber + LGBM tail for amber_apf_lgbm, TFT for
model_a_hybrid).

Convention notes:
  - Forecast logs store `prediction` in $/kWh and `forecast_target_time` as
    interval-END (matching AEMO).
  - `actuals_sa1.parquet.time` is interval-START. Lookup key is
    `target_time - 30min`.
  - Actuals in the price log (column `actual`) are 30-min realised RRP in
    $/MWh, written by `forecast.py backfill-actuals`. When that column is
    present and non-null, we use it directly. Otherwise we join against the
    actuals parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
ACTUALS_30M = REPO_ROOT / "data" / "parquet" / "actuals_sa1.parquet"
LOCAL_TZ = "Australia/Adelaide"

ADELAIDE_BUCKETS = ("overnight", "morning", "solar", "evening", "late")
HORIZON_BUCKETS = (
    ("0-1h", 0.0, 1.0),
    ("1-4h", 1.0, 4.0),
    ("4-12h", 4.0, 12.0),
    ("12-24h", 12.0, 24.0),
    ("24h+", 24.0, 1e9),
)


def _adelaide_bucket(target_utc: pd.Series) -> pd.Series:
    local_hour = target_utc.dt.tz_convert(LOCAL_TZ).dt.hour
    return pd.cut(
        local_hour,
        bins=[-1, 5, 11, 16, 20, 24],
        labels=list(ADELAIDE_BUCKETS),
        right=True,
    ).astype(str)


def _horizon_bucket(h: float) -> str:
    for label, lo, hi in HORIZON_BUCKETS:
        if lo <= h < hi:
            return label
    return "n/a"


def _load_log(path: Path, model_filter: str | None) -> pd.DataFrame:
    print(f"Reading {path}...")
    df = pd.read_csv(
        path,
        usecols=[
            "forecast_creation_time", "forecast_target_time", "model_name",
            "prediction", "actual",
        ],
        dtype_backend="pyarrow",
    )
    if model_filter:
        df = df[df["model_name"] == model_filter].copy()
    df["forecast_creation_time"] = pd.to_datetime(
        df["forecast_creation_time"], utc=True, format="mixed"
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="mixed"
    )
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["prediction"])
    return df


def _attach_actuals(df: pd.DataFrame, actuals_parquet: Path) -> pd.DataFrame:
    a = pd.read_parquet(actuals_parquet)
    a["time"] = pd.to_datetime(a["time"], utc=True)
    a = a.set_index("time")["rrp"].astype(np.float64)
    lookup = df["forecast_target_time"] - pd.Timedelta(minutes=30)
    actual_join = a.reindex(lookup).values
    df = df.copy()
    df["actual_rrp_join"] = actual_join
    return df


def _summary(label: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    if frame.empty:
        return {"bucket": label, "n": 0}
    return {
        "bucket": label,
        "n": int(len(frame)),
        "actual_mean": float(frame["actual_rrp"].mean()),
        "pred_mean": float(frame["pred_mwh"].mean()),
        "mae": float((frame["pred_mwh"] - frame["actual_rrp"]).abs().mean()),
        "bias": float((frame["pred_mwh"] - frame["actual_rrp"]).mean()),
        "p90_overshoot": float(
            (frame["pred_mwh"] - frame["actual_rrp"]).quantile(0.90)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument(
        "--model-filter", default=None,
        help="If set, keep only rows with model_name == this string.",
    )
    parser.add_argument(
        "--label", default=None,
        help="Display label (defaults to log file stem).",
    )
    parser.add_argument(
        "--actuals-30m", type=Path, default=ACTUALS_30M,
    )
    parser.add_argument(
        "--since", default="2026-04-01T00:00:00Z",
        help="Only score forecasts created at or after this timestamp.",
    )
    parser.add_argument(
        "--max-horizon-hours", type=float, default=72.0,
    )
    args = parser.parse_args()

    label = args.label or args.log_file.stem
    df = _load_log(args.log_file, args.model_filter)
    print(f"  rows after model filter: {len(df):,}")

    since = pd.Timestamp(args.since)
    df = df[df["forecast_creation_time"] >= since]
    print(f"  rows since {since}: {len(df):,}")

    df["horizon_hours"] = (
        df["forecast_target_time"] - df["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    df = df[(df["horizon_hours"] >= 0.0) & (df["horizon_hours"] <= args.max_horizon_hours)]
    print(f"  rows within horizon: {len(df):,}")

    df["pred_mwh"] = df["prediction"].astype(np.float64) * 1000.0

    df = _attach_actuals(df, args.actuals_30m)
    # Log `actual` is in $/kWh (mirrors `prediction`, see log_forecast_data in
    # forecast.py — actual is backfilled into the same per-row schema as the
    # prediction column was originally written). Convert to $/MWh to match
    # the parquet-join fallback. Use the log value when present, else fall
    # back to the parquet join.
    actual_log_mwh = df["actual"].astype(np.float64) * 1000.0
    df["actual_rrp"] = actual_log_mwh.where(
        actual_log_mwh.notna(), df["actual_rrp_join"].astype(np.float64)
    )
    df = df.dropna(subset=["actual_rrp"])
    print(f"  rows with actuals: {len(df):,}")

    if df.empty:
        print("No rows to score. Stopping.")
        return

    print(
        f"  target window: {df['forecast_target_time'].min()} → "
        f"{df['forecast_target_time'].max()}"
    )

    df["adelaide_bucket"] = _adelaide_bucket(df["forecast_target_time"])
    df["horizon_bucket"] = df["horizon_hours"].apply(_horizon_bucket)

    rows = [_summary("OVERALL", df)]
    for b in ADELAIDE_BUCKETS:
        rows.append(_summary(f"  {b}", df[df["adelaide_bucket"] == b]))
    for hb, _, _ in HORIZON_BUCKETS:
        rows.append(_summary(f"  h={hb}", df[df["horizon_bucket"] == hb]))

    out = pd.DataFrame(rows).set_index("bucket")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:8.2f}")
    print(f"\n[{label}] forecast vs realised RRP ($/MWh):")
    print(out.to_string())


if __name__ == "__main__":
    main()
