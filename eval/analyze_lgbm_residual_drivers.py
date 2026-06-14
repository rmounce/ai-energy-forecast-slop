#!/usr/bin/env python3
"""Decompose LGBM price forecast residuals by operational drivers.

This extends the basic forecast-vs-actual audit in ``audit_price_forecast.py``.
It keeps row-level residuals, joins AEMO PREDISPATCH and SevenDayOutlook data
as issued at the forecast creation time, then writes compact CSV tables that
show where the Amber APF + LGBM tail is biased.

Conventions:
  - price forecast log ``prediction`` and ``actual`` are $/kWh.
  - AEMO parquet RRP values are $/MWh.
  - ``forecast_target_time`` is interval-END; ``actuals_sa1.time`` is
    interval-START, so actual lookup is ``target_time - 30min``.
  - PREDISPATCH / SevenDayOutlook joins choose latest ``run_time`` <=
    ``forecast_creation_time`` for the same target interval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"

DEFAULT_LOG = REPO_ROOT / "price_forecast_log.csv"
DEFAULT_ACTUALS = REPO_ROOT / "data" / "parquet" / "actuals_sa1.parquet"
DEFAULT_PREDISPATCH = REPO_ROOT / "data" / "parquet" / "aemo_predispatch_sa1.parquet"
DEFAULT_SDO = REPO_ROOT / "data" / "parquet" / "aemo_sevendayoutlook_sa1.parquet"

LOCAL_TZ = "Australia/Adelaide"

ADELAIDE_BUCKETS = ("overnight", "morning", "solar", "evening", "late")
HORIZON_BUCKETS = (
    ("0-1h", 0.0, 1.0),
    ("1-4h", 1.0, 4.0),
    ("4-12h", 4.0, 12.0),
    ("12-24h", 12.0, 24.0),
    ("24h+", 24.0, 1e9),
)
PRICE_REGIME_BINS = [-np.inf, 0.0, 50.0, 150.0, 300.0, np.inf]
PRICE_REGIME_LABELS = ["negative", "low_0_50", "mid_50_150", "high_150_300", "spike_300p"]


def adelaide_bucket(target_utc: pd.Series) -> pd.Series:
    local_hour = target_utc.dt.tz_convert(LOCAL_TZ).dt.hour
    return pd.cut(
        local_hour,
        bins=[-1, 5, 11, 16, 20, 24],
        labels=list(ADELAIDE_BUCKETS),
        right=True,
    ).astype(str)


def horizon_bucket(hours: pd.Series) -> pd.Series:
    out = pd.Series("n/a", index=hours.index, dtype="object")
    for label, lo, hi in HORIZON_BUCKETS:
        out.loc[(hours >= lo) & (hours < hi)] = label
    return out


def signed_tercile_bucket(series: pd.Series, *, deadband: float = 0.0) -> pd.Series:
    """Three-way bucket for signed forecast errors with useful labels."""

    out = pd.Series("missing", index=series.index, dtype="object")
    valid = series.dropna()
    if valid.empty:
        return out
    neg = series < -deadband
    pos = series > deadband
    mid = ~(neg | pos) & series.notna()
    out.loc[neg] = "forecast_low"
    out.loc[mid] = "near_zero"
    out.loc[pos] = "forecast_high"
    return out


def quantile_bucket(series: pd.Series, labels: tuple[str, str, str] = ("low", "mid", "high")) -> pd.Series:
    out = pd.Series("missing", index=series.index, dtype="object")
    valid = series.dropna()
    if len(valid) < len(labels):
        return out
    try:
        bucketed = pd.qcut(valid.rank(method="first"), q=len(labels), labels=list(labels))
    except ValueError:
        return out
    out.loc[valid.index] = bucketed.astype(str)
    return out


def latest_asof_by_target(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_target: str,
    left_time: str,
    right_target: str,
    right_time: str,
    value_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Join latest source row per target where source run_time <= left time.

    ``pandas.merge_asof`` is awkward for this shape because the as-of key needs
    to be monotonic globally while target intervals partition the join. A small
    grouped search keeps the semantics explicit and is fast enough for forecast
    logs of this size.
    """

    out = left.copy()
    for col in value_cols:
        out[f"{prefix}_{col}"] = np.nan
    out[f"{prefix}_run_time"] = pd.Series(
        pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
    )

    if right.empty or left.empty:
        return out

    source_by_target = {
        target: grp.sort_values(right_time)
        for target, grp in right.groupby(right_target, sort=False)
    }

    for target, idx in out.groupby(left_target, sort=False).groups.items():
        source = source_by_target.get(target)
        if source is None or source.empty:
            continue

        run_times = pd.to_datetime(source[right_time], utc=True).to_numpy(dtype="datetime64[ns]")
        left_times = pd.to_datetime(out.loc[idx, left_time], utc=True).to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(run_times, left_times, side="right") - 1
        valid = positions >= 0
        if not valid.any():
            continue

        valid_idx = np.asarray(idx)[valid]
        source_rows = source.iloc[positions[valid]]
        for col in value_cols:
            out.loc[valid_idx, f"{prefix}_{col}"] = pd.to_numeric(
                source_rows[col], errors="coerce"
            ).to_numpy()
        out.loc[valid_idx, f"{prefix}_run_time"] = pd.to_datetime(
            source_rows[right_time], utc=True
        ).array

    return out


def load_price_log(path: Path, *, model_filter: str | None) -> pd.DataFrame:
    cols = [
        "forecast_creation_time",
        "forecast_target_time",
        "model_name",
        "prediction",
        "actual",
        "power_pv",
        "power_pv_actual",
        "temperature_adelaide",
        "temperature_adelaide_actual",
        "humidity_adelaide",
        "humidity_adelaide_actual",
        "wind_speed_adelaide",
        "wind_speed_adelaide_actual",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols, dtype_backend="pyarrow")
    if model_filter:
        df = df[df["model_name"] == model_filter].copy()
    df["forecast_creation_time"] = pd.to_datetime(
        df["forecast_creation_time"], utc=True, format="mixed"
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="mixed"
    )
    for col in cols:
        if col not in {"forecast_creation_time", "forecast_target_time", "model_name"} and col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["prediction"])


def attach_actuals(df: pd.DataFrame, actuals_path: Path) -> pd.DataFrame:
    actuals = pd.read_parquet(actuals_path)
    actuals["time"] = pd.to_datetime(actuals["time"], utc=True)
    actuals = actuals.set_index("time").sort_index()

    lookup = df["forecast_target_time"] - pd.Timedelta(minutes=30)
    actual_join = actuals.reindex(lookup)

    out = df.copy()
    out["actual_rrp_join"] = actual_join["rrp"].to_numpy(dtype=np.float64)
    for col in ("total_demand", "net_interchange"):
        out[f"actual_{col}"] = actual_join[col].to_numpy(dtype=np.float64)

    actual_log_mwh = pd.to_numeric(out["actual"], errors="coerce") * 1000.0
    out["actual_rrp"] = actual_log_mwh.where(actual_log_mwh.notna(), out["actual_rrp_join"])
    return out


def load_source(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=cols)
    for col in ("interval_dt", "run_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df.sort_values(["interval_dt", "run_time"])


def add_driver_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pred_mwh"] = out["prediction"] * 1000.0
    out["residual_mwh"] = out["pred_mwh"] - out["actual_rrp"]
    out["abs_error_mwh"] = out["residual_mwh"].abs()
    out["horizon_hours"] = (
        out["forecast_target_time"] - out["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    out["adelaide_bucket"] = adelaide_bucket(out["forecast_target_time"])
    out["horizon_bucket"] = horizon_bucket(out["horizon_hours"])
    out["actual_price_regime"] = pd.cut(
        out["actual_rrp"],
        bins=PRICE_REGIME_BINS,
        labels=PRICE_REGIME_LABELS,
        right=False,
    ).astype(str)

    out["pd_demand_error"] = out["pd_total_demand"] - out["actual_total_demand"]
    out["pd_net_interchange_error"] = out["pd_net_interchange"] - out["actual_net_interchange"]
    out["sdo_demand_error"] = out["sdo_scheduled_demand"] - out["actual_total_demand"]
    out["sdo_net_interchange_error"] = out["sdo_net_interchange"] - out["actual_net_interchange"]
    out["pv_error"] = out["power_pv"] - out["power_pv_actual"]
    out["temp_error"] = out["temperature_adelaide"] - out["temperature_adelaide_actual"]
    out["humidity_error"] = out["humidity_adelaide"] - out["humidity_adelaide_actual"]
    out["local_wind_error"] = out["wind_speed_adelaide"] - out["wind_speed_adelaide_actual"]

    out["actual_demand_bucket"] = quantile_bucket(out["actual_total_demand"])
    out["actual_net_interchange_bucket"] = quantile_bucket(
        out["actual_net_interchange"], labels=("net_export_high", "mid", "net_import_high")
    )
    out["pd_demand_error_bucket"] = signed_tercile_bucket(out["pd_demand_error"], deadband=50.0)
    out["pd_net_interchange_error_bucket"] = signed_tercile_bucket(
        out["pd_net_interchange_error"], deadband=50.0
    )
    out["sdo_demand_error_bucket"] = signed_tercile_bucket(out["sdo_demand_error"], deadband=50.0)
    out["sdo_net_interchange_error_bucket"] = signed_tercile_bucket(
        out["sdo_net_interchange_error"], deadband=50.0
    )
    out["pv_error_bucket"] = signed_tercile_bucket(out["pv_error"], deadband=250.0)
    out["actual_pv_bucket"] = quantile_bucket(out["power_pv_actual"])
    out["local_wind_actual_bucket"] = quantile_bucket(out["wind_speed_adelaide_actual"])
    out["local_wind_error_bucket"] = signed_tercile_bucket(out["local_wind_error"], deadband=2.0)
    return out


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_cols, dropna=False, observed=False)
    out = grouped.agg(
        n=("residual_mwh", "size"),
        actual_mean=("actual_rrp", "mean"),
        pred_mean=("pred_mwh", "mean"),
        mae=("abs_error_mwh", "mean"),
        bias=("residual_mwh", "mean"),
        p10_residual=("residual_mwh", lambda s: s.quantile(0.10)),
        p90_residual=("residual_mwh", lambda s: s.quantile(0.90)),
        actual_demand_mean=("actual_total_demand", "mean"),
        actual_net_interchange_mean=("actual_net_interchange", "mean"),
        pd_demand_error_mean=("pd_demand_error", "mean"),
        pd_net_interchange_error_mean=("pd_net_interchange_error", "mean"),
        sdo_demand_error_mean=("sdo_demand_error", "mean"),
        sdo_net_interchange_error_mean=("sdo_net_interchange_error", "mean"),
        pv_error_mean=("pv_error", "mean"),
        local_wind_error_mean=("local_wind_error", "mean"),
    )
    return out.reset_index().sort_values(group_cols)


def driver_rank(frame: pd.DataFrame, bucket_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in bucket_cols:
        table = summarize(frame[frame[col] != "missing"], [col])
        if table.empty:
            continue
        rows.append(
            {
                "driver": col,
                "n": int(table["n"].sum()),
                "bucket_count": int(len(table)),
                "bias_min": float(table["bias"].min()),
                "bias_max": float(table["bias"].max()),
                "bias_spread": float(table["bias"].max() - table["bias"].min()),
                "mae_min": float(table["mae"].min()),
                "mae_max": float(table["mae"].max()),
                "mae_spread": float(table["mae"].max() - table["mae"].min()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["bias_spread", "mae_spread"], ascending=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--actuals-30m", type=Path, default=DEFAULT_ACTUALS)
    parser.add_argument("--predispatch", type=Path, default=DEFAULT_PREDISPATCH)
    parser.add_argument("--sevendayoutlook", type=Path, default=DEFAULT_SDO)
    parser.add_argument("--model-filter", default="price")
    parser.add_argument("--since", default="2026-04-01T00:00:00Z")
    parser.add_argument("--until", default=None)
    parser.add_argument("--max-horizon-hours", type=float, default=72.0)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=RESULTS_DIR / "lgbm_residual_drivers",
        help="Prefix for CSV output files.",
    )
    parser.add_argument(
        "--write-row-sample",
        action="store_true",
        help="Also write a row-level parquet with joined drivers.",
    )
    args = parser.parse_args()

    print(f"Reading {args.log_file}...")
    df = load_price_log(args.log_file, model_filter=args.model_filter)
    print(f"  rows after model filter: {len(df):,}")

    since = pd.Timestamp(args.since)
    df = df[df["forecast_creation_time"] >= since]
    if args.until:
        until = pd.Timestamp(args.until)
        df = df[df["forecast_creation_time"] < until]
    df["horizon_hours"] = (
        df["forecast_target_time"] - df["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    df = df[(df["horizon_hours"] >= 0.0) & (df["horizon_hours"] <= args.max_horizon_hours)]
    print(f"  rows in time/horizon window: {len(df):,}")

    print(f"Joining actuals from {args.actuals_30m}...")
    df = attach_actuals(df, args.actuals_30m)
    df = df.dropna(subset=["actual_rrp", "actual_total_demand", "actual_net_interchange"])
    print(f"  rows with actuals: {len(df):,}")

    print(f"Joining PREDISPATCH from {args.predispatch}...")
    pd_source = load_source(
        args.predispatch,
        ["interval_dt", "run_time", "total_demand", "net_interchange"],
    )
    df = latest_asof_by_target(
        df,
        pd_source,
        left_target="forecast_target_time",
        left_time="forecast_creation_time",
        right_target="interval_dt",
        right_time="run_time",
        value_cols=["total_demand", "net_interchange"],
        prefix="pd",
    )
    print(f"  rows with PREDISPATCH demand: {df['pd_total_demand'].notna().sum():,}")

    print(f"Joining SevenDayOutlook from {args.sevendayoutlook}...")
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
    print(f"  rows with SDO demand: {df['sdo_scheduled_demand'].notna().sum():,}")

    df = add_driver_columns(df)
    prefix = args.output_prefix

    print("Writing summary tables...")
    write_table(summarize(df, ["adelaide_bucket"]), prefix.with_name(prefix.name + "_by_tod.csv"))
    write_table(summarize(df, ["horizon_bucket"]), prefix.with_name(prefix.name + "_by_horizon.csv"))
    write_table(
        summarize(df, ["actual_price_regime"]),
        prefix.with_name(prefix.name + "_by_price_regime.csv"),
    )
    write_table(
        summarize(df, ["horizon_bucket", "adelaide_bucket"]),
        prefix.with_name(prefix.name + "_by_horizon_tod.csv"),
    )

    bucket_cols = [
        "actual_demand_bucket",
        "actual_net_interchange_bucket",
        "pd_demand_error_bucket",
        "pd_net_interchange_error_bucket",
        "sdo_demand_error_bucket",
        "sdo_net_interchange_error_bucket",
        "actual_pv_bucket",
        "pv_error_bucket",
        "local_wind_actual_bucket",
        "local_wind_error_bucket",
    ]
    for col in bucket_cols:
        write_table(summarize(df[df[col] != "missing"], [col]), prefix.with_name(prefix.name + f"_by_{col}.csv"))
    write_table(driver_rank(df, bucket_cols), prefix.with_name(prefix.name + "_driver_rank.csv"))

    overall = summarize(df, ["model_name"])
    write_table(overall, prefix.with_name(prefix.name + "_overall.csv"))

    if args.write_row_sample:
        row_path = prefix.with_name(prefix.name + "_rows.parquet")
        keep_cols = [
            "forecast_creation_time",
            "forecast_target_time",
            "horizon_hours",
            "pred_mwh",
            "actual_rrp",
            "residual_mwh",
            "actual_total_demand",
            "actual_net_interchange",
            "pd_total_demand",
            "pd_net_interchange",
            "sdo_scheduled_demand",
            "sdo_net_interchange",
            "power_pv",
            "power_pv_actual",
            "wind_speed_adelaide",
            "wind_speed_adelaide_actual",
        ]
        df[keep_cols].to_parquet(row_path, index=False)
        print(f"  wrote {row_path}")

    print(f"\nWindow: {df['forecast_target_time'].min()} -> {df['forecast_target_time'].max()}")
    print(f"Rows scored: {len(df):,}")
    if not overall.empty:
        print("\nOverall:")
        pd.set_option("display.width", 180)
        pd.set_option("display.float_format", lambda x: f"{x:8.2f}")
        print(overall.to_string(index=False))
    print(f"\nWrote CSV files with prefix: {prefix}")


if __name__ == "__main__":
    main()
