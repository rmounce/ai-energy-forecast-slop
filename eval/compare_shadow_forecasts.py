#!/usr/bin/env python3
"""Phase α-prime shadow-and-compare: read live forecast logs, score against actuals.

This is the analysis half of "publish forecasts to HA in parallel, then assess how the
shadow would have behaved in comparison" (per the 2026-05-08 user direction). It does
*not* run any dispatch simulation; it only scores forecast quality. A future extension
can feed the same log entries into eval/rolling_mpc_eval.py for what-if dispatch.

Inputs (CSV logs continuously appended by forecast.py):

  - tft_price_forecast_log.csv     (existing TFT shadow — months of data)
  - pd_direct_forecast_log.csv     (Phase α-prime Step 4 — accumulating since 2026-05-08)
  - price_forecast_log.csv         (legacy APF/LightGBM incumbent — production)
  - p5min_forecast_log.csv         (Tier 1 tactical, 0–60 min horizon)

Each row contains a (forecast_creation_time, forecast_target_time, prediction) tuple.
We join these with realised prices from actuals_sa1.parquet to compute per-horizon
forecast quality metrics, stratified by price regime.

Usage:

  python eval/compare_shadow_forecasts.py
  python eval/compare_shadow_forecasts.py --since 2026-05-08T00:00:00Z
  python eval/compare_shadow_forecasts.py --sources tft,pd_direct
  python eval/compare_shadow_forecasts.py --output eval/results/shadow_compare_$(date +%Y%m%d).csv

Output: a tidy CSV with one row per (source, horizon_bucket, regime, metric).

Notes on the current state (2026-05-08):

  - The TFT and incumbent logs span months; meaningful comparison is possible today.
  - The PD-direct log started 2026-05-08; meaningful comparison needs ≥1 week of
    accumulated data, ideally including a price-volatile period. May is low-volatility
    for SA1, so material differentiation may take weeks.
  - This script is a scaffold. Expand the metrics block as the analysis matures. The
    dispatch what-if integration is a follow-up: feed log entries as a synthetic source
    into eval/rolling_mpc_eval.py (or a smaller variant of it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ACTUALS_PARQUET = ROOT / "data" / "parquet" / "actuals_sa1.parquet"

LOG_FILES = {
    "tft":           ROOT / "tft_price_forecast_log.csv",
    "pd_direct":     ROOT / "pd_direct_forecast_log.csv",
    "incumbent":     ROOT / "price_forecast_log.csv",   # APF/LightGBM
    "tier1":         ROOT / "p5min_forecast_log.csv",
}

# The price forecast logs are not unit-consistent: TFT writes raw RRP in $/MWh, while
# the others write tariff-applied $/kWh (incumbent and pd_direct) or wholesale $/kWh
# (tier1). Multiplier converts each source's `prediction` column to $/MWh for joining
# with actuals_sa1.parquet.rrp.
#
# Caveat: `incumbent` (price_forecast_log.csv) is post-tariff general_price, so its
# magnitude is inflated by network tariff + GST relative to the wholesale RRP. For
# spike/peak comparison this is mostly directional; for absolute MAE numbers it is
# biased high. A future cleanup could log raw wholesale RRP separately.
SOURCE_TO_MWH_MULT = {
    "tft":         1.0,
    "pd_direct":   1000.0,
    "incumbent":   1000.0,
    "tier1":       1000.0,
}

# Horizon buckets (minutes from forecast_creation_time → forecast_target_time).
HORIZON_BUCKETS = [
    (0,    60,   "0-60min"),
    (60,   240,  "1-4h"),
    (240,  720,  "4-12h"),
    (720,  1680, "12-28h"),
    (1680, 4320, "28-72h"),
]

# Regime buckets ($/MWh on realised RRP).
REGIME_BUCKETS = [
    (-1e9, 0,    "negative"),
    (0,    60,   "low"),
    (60,   150,  "normal"),
    (150,  300,  "elevated"),
    (300,  1e9,  "spike"),
]


def _bucketise(value: float, buckets: list[tuple]) -> str:
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return "unbucketed"


def load_forecast_log(source: str, path: Path, since: pd.Timestamp | None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(
        path,
        usecols=["forecast_creation_time", "forecast_target_time", "prediction"],
    )
    df["forecast_creation_time"] = pd.to_datetime(
        df["forecast_creation_time"], utc=True, format="ISO8601"
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="ISO8601"
    )
    if since is not None:
        df = df[df["forecast_creation_time"] >= since]
    df["source"] = source
    df["prediction_mwh"] = df["prediction"] * SOURCE_TO_MWH_MULT.get(source, 1000.0)
    df["horizon_min"] = (
        (df["forecast_target_time"] - df["forecast_creation_time"]).dt.total_seconds() / 60.0
    )
    df["horizon_bucket"] = df["horizon_min"].apply(
        lambda m: _bucketise(m, HORIZON_BUCKETS)
    )
    return df.drop(columns=["prediction"])


def load_actuals() -> pd.Series:
    df = pd.read_parquet(ACTUALS_PARQUET)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    s = df.set_index("time")["rrp"].sort_index()
    return s[~s.index.duplicated(keep="last")]


def join_actuals(forecasts: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
    """Inner-join each forecast row with the realised RRP at its target time."""
    if forecasts.empty:
        return forecasts.assign(actual_mwh=[], regime=[])
    actuals_aligned = actuals.reindex(forecasts["forecast_target_time"]).values
    out = forecasts.copy()
    out["actual_mwh"] = actuals_aligned
    out = out.dropna(subset=["actual_mwh"])
    out["regime"] = out["actual_mwh"].apply(lambda v: _bucketise(v, REGIME_BUCKETS))
    return out


def compute_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per (source, horizon_bucket, regime) with summary metrics."""
    if joined.empty:
        return pd.DataFrame()
    joined = joined.copy()
    joined["error"] = joined["prediction_mwh"] - joined["actual_mwh"]
    joined["abs_error"] = joined["error"].abs()
    grouped = joined.groupby(["source", "horizon_bucket", "regime"], observed=True)
    summary = grouped.agg(
        n=("error", "size"),
        mean_actual=("actual_mwh", "mean"),
        mean_prediction=("prediction_mwh", "mean"),
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
        rmse=("error", lambda s: float(np.sqrt(np.mean(s**2)))),
    ).reset_index()
    return summary


def add_overall_rows(summary: pd.DataFrame, joined: pd.DataFrame) -> pd.DataFrame:
    """Append per-source overall and per-source-per-horizon overall rows."""
    if joined.empty:
        return summary
    joined = joined.copy()
    joined["error"] = joined["prediction_mwh"] - joined["actual_mwh"]
    joined["abs_error"] = joined["error"].abs()

    extra_rows = []
    for keys, agg in [
        (["source"], "ALL"),
        (["source", "horizon_bucket"], "ALL_REGIMES"),
    ]:
        g = joined.groupby(keys, observed=True).agg(
            n=("error", "size"),
            mean_actual=("actual_mwh", "mean"),
            mean_prediction=("prediction_mwh", "mean"),
            bias=("error", "mean"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda s: float(np.sqrt(np.mean(s**2)))),
        ).reset_index()
        if "horizon_bucket" not in g.columns:
            g["horizon_bucket"] = agg
        if "regime" not in g.columns:
            g["regime"] = agg
        extra_rows.append(g)

    full = pd.concat([summary] + extra_rows, ignore_index=True)
    return full[["source", "horizon_bucket", "regime",
                 "n", "mean_actual", "mean_prediction", "bias", "mae", "rmse"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", default="tft,pd_direct,incumbent,tier1",
        help="Comma-separated source labels to include. Available: "
             + ",".join(LOG_FILES.keys()),
    )
    parser.add_argument(
        "--since", default="",
        help="Only include forecasts created on/after this UTC timestamp "
             "(e.g. 2026-05-08T00:00:00Z). Default: include everything.",
    )
    parser.add_argument(
        "--output", default="",
        help="Output CSV path. Default: print to stdout.",
    )
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in LOG_FILES]
    if unknown:
        parser.error(f"Unknown sources: {unknown}. Available: {list(LOG_FILES)}")

    since = pd.Timestamp(args.since, tz="UTC") if args.since else None

    print(f"Loading actuals from {ACTUALS_PARQUET}...", file=sys.stderr)
    actuals = load_actuals()
    print(f"  {len(actuals):,} rows, range {actuals.index.min()} → {actuals.index.max()}",
          file=sys.stderr)

    all_joined = []
    for src in sources:
        path = LOG_FILES[src]
        if not path.exists():
            print(f"  [{src}] log not found at {path}; skipping", file=sys.stderr)
            continue
        df = load_forecast_log(src, path, since)
        if df.empty:
            print(f"  [{src}] no rows after filter; skipping", file=sys.stderr)
            continue
        joined = join_actuals(df, actuals)
        print(f"  [{src}] {len(df):,} forecasts → {len(joined):,} matched with actuals "
              f"(coverage: {len(joined)/max(len(df),1):.1%})", file=sys.stderr)
        all_joined.append(joined)

    if not all_joined:
        print("No forecast data available. Run forecast.py predict-all to populate logs.",
              file=sys.stderr)
        sys.exit(0)

    joined = pd.concat(all_joined, ignore_index=True)
    summary = compute_metrics(joined)
    summary = add_overall_rows(summary, joined)
    summary = summary.sort_values(
        ["source", "horizon_bucket", "regime"]
    ).reset_index(drop=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"Wrote {len(summary)} rows to {out_path}", file=sys.stderr)
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
