#!/usr/bin/env python3
"""Audit PD-direct debiaser: compare debiased vs raw PREDISPATCH vs realised RRP.

For each forecast in `pd_direct_forecast_log.csv` whose target interval has a
realised actual in `actuals_sa1.parquet`, join:

  - prediction (debiased PD-direct, from the log)
  - raw_pd_rrp  (raw PREDISPATCH RRP at the matching run_time/interval, from
                 `aemo_predispatch_sa1.parquet`)
  - actual_rrp  (realised 30-min RRP, from `actuals_sa1.parquet`)

Report MAE, mean signed bias, and sample count overall and bucketed by:
  - Adelaide local time-of-day (overnight/morning/solar/evening/late)
  - Horizon bucket (hours from creation_time to target_time)

The point of the audit is to answer: is the PD-direct debiaser correctly
calibrated, or is it systematically pulling raw PREDISPATCH too far below
realised? `debiaser_delta = prediction - raw_pd_rrp` shows how much the
debiaser is moving from PREDISPATCH; if the debiased prediction is closer
to actual than raw is, the debiaser is doing its job.

Convention notes:
  - `pd_direct_forecast_log.csv` stores `prediction` in $/kWh and
    `forecast_target_time` as interval-END (AEMO convention, before the
    publish-time -30 min shift). To match `actuals_sa1.parquet` whose
    `time` is interval-START, we look up actuals at `target_time - 30min`.
  - Raw PREDISPATCH parquet has `interval_dt` as interval-END (matches the
    log convention). Run alignment: pick the latest `run_time` ≤
    `forecast_creation_time` and the row with `interval_dt == target_time`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = REPO_ROOT / "pd_direct_forecast_log.csv"
ACTUALS_30M = REPO_ROOT / "data" / "parquet" / "actuals_sa1.parquet"
PD_PARQUET = REPO_ROOT / "data" / "parquet" / "aemo_predispatch_sa1.parquet"
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


def _join_predispatch(df: pd.DataFrame, pd_parquet: Path) -> pd.DataFrame:
    pdf = pd.read_parquet(pd_parquet)
    pdf["run_time"] = pd.to_datetime(pdf["run_time"], utc=True)
    pdf["interval_dt"] = pd.to_datetime(pdf["interval_dt"], utc=True)
    pdf = pdf.sort_values(["interval_dt", "run_time"])

    # For each (creation_time, target_time), find the latest run_time <= creation_time
    # with interval_dt == target_time.
    runs_by_interval = pdf.groupby("interval_dt")
    raw_rrp = np.full(len(df), np.nan, dtype=np.float64)
    matched_run_time = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for i, (creation, target) in enumerate(
        zip(df["forecast_creation_time"], df["forecast_target_time"], strict=True)
    ):
        if target not in runs_by_interval.groups:
            continue
        grp = runs_by_interval.get_group(target)
        eligible = grp[grp["run_time"] <= creation]
        if eligible.empty:
            continue
        row = eligible.iloc[-1]
        raw_rrp[i] = float(row["rrp"])
        matched_run_time.iloc[i] = row["run_time"]
    df = df.copy()
    df["raw_pd_rrp"] = raw_rrp
    df["pd_run_time"] = matched_run_time
    return df


def _join_actuals(df: pd.DataFrame, actuals_parquet: Path) -> pd.DataFrame:
    a = pd.read_parquet(actuals_parquet)
    a["time"] = pd.to_datetime(a["time"], utc=True)
    a = a.set_index("time")["rrp"].astype(np.float64)
    # actuals.time is interval-START; forecast_target_time is interval-END.
    # Lookup key = target_time - 30min.
    lookup = df["forecast_target_time"] - pd.Timedelta(minutes=30)
    actual_rrp = a.reindex(lookup).values
    df = df.copy()
    df["actual_rrp"] = actual_rrp
    return df


def _summary(label: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    if frame.empty:
        return {"bucket": label, "n": 0}
    return {
        "bucket": label,
        "n": int(len(frame)),
        "actual_mean": float(frame["actual_rrp"].mean()),
        "raw_pd_mean": float(frame["raw_pd_rrp"].mean()),
        "pd_direct_mean": float(frame["pred_mwh"].mean()),
        "raw_mae": float((frame["raw_pd_rrp"] - frame["actual_rrp"]).abs().mean()),
        "pd_direct_mae": float((frame["pred_mwh"] - frame["actual_rrp"]).abs().mean()),
        "raw_bias": float((frame["raw_pd_rrp"] - frame["actual_rrp"]).mean()),
        "pd_direct_bias": float((frame["pred_mwh"] - frame["actual_rrp"]).mean()),
        "debiaser_delta_mean": float((frame["pred_mwh"] - frame["raw_pd_rrp"]).mean()),
        "debiaser_helped_pct": float(
            (
                (frame["pred_mwh"] - frame["actual_rrp"]).abs()
                < (frame["raw_pd_rrp"] - frame["actual_rrp"]).abs()
            ).mean()
            * 100.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, default=LOG_FILE)
    parser.add_argument("--actuals-30m", type=Path, default=ACTUALS_30M)
    parser.add_argument("--pd-parquet", type=Path, default=PD_PARQUET)
    parser.add_argument(
        "--max-horizon-hours", type=float, default=72.0,
        help="Skip forecasts where target_time - creation_time exceeds this.",
    )
    args = parser.parse_args()

    print(f"Reading {args.log_file}...")
    df = pd.read_csv(args.log_file)
    df["forecast_creation_time"] = pd.to_datetime(
        df["forecast_creation_time"], utc=True, format="mixed"
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="mixed"
    )
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["prediction"])
    df["pred_mwh"] = df["prediction"] * 1000.0  # log is $/kWh; align to PREDISPATCH $/MWh
    df["horizon_hours"] = (
        df["forecast_target_time"] - df["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    df = df[df["horizon_hours"] >= 0.0]
    df = df[df["horizon_hours"] <= args.max_horizon_hours]
    print(f"  raw rows: {len(df):,}")

    print(f"Joining raw PREDISPATCH from {args.pd_parquet}...")
    df = _join_predispatch(df, args.pd_parquet)
    print(f"  rows with raw_pd_rrp matched: {df['raw_pd_rrp'].notna().sum():,}")

    print(f"Joining actuals from {args.actuals_30m}...")
    df = _join_actuals(df, args.actuals_30m)
    print(f"  rows with actual_rrp matched: {df['actual_rrp'].notna().sum():,}")

    df = df.dropna(subset=["raw_pd_rrp", "actual_rrp"])
    print(f"  rows usable (both backfilled): {len(df):,}")

    if df.empty:
        print("No usable rows after joins. Stopping.")
        return

    print(f"\nWindow: {df['forecast_target_time'].min()} → {df['forecast_target_time'].max()}")
    print(f"Unique forecast targets: {df['forecast_target_time'].nunique():,}")
    print(f"Unique creation times:   {df['forecast_creation_time'].nunique():,}")

    df["adelaide_bucket"] = _adelaide_bucket(df["forecast_target_time"])
    df["horizon_bucket"] = df["horizon_hours"].apply(_horizon_bucket)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:8.2f}")

    rows = [_summary("OVERALL", df)]
    for b in ADELAIDE_BUCKETS:
        rows.append(_summary(f"  {b}", df[df["adelaide_bucket"] == b]))
    for hb, _, _ in HORIZON_BUCKETS:
        rows.append(_summary(f"  h={hb}", df[df["horizon_bucket"] == hb]))

    out = pd.DataFrame(rows).set_index("bucket")
    print("\nDebiaser audit (all values in $/MWh):")
    print(out.to_string())

    if out.loc["OVERALL", "pd_direct_mae"] > out.loc["OVERALL", "raw_mae"]:
        verdict = (
            "Debiaser HURTS overall. PD-direct MAE > raw PREDISPATCH MAE — "
            "the correction is moving forecasts further from realised."
        )
    else:
        verdict = (
            "Debiaser HELPS overall. PD-direct MAE < raw PREDISPATCH MAE — "
            "the correction is moving forecasts closer to realised."
        )
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
