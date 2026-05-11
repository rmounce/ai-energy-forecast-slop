#!/usr/bin/env python3
"""Timestamp alignment diagnostic — per implementer's 2026-05-11 audit recommendation.

The audit doc (`docs/timestamp_convention_audit_2026-05-11.md`) showed PREDISPATCH
covariate rows align better against actuals_sa1.parquet at `interval_dt - 30 min`
than at `interval_dt`, suggesting an internal interval-end vs interval-start
mismatch. The audit explicitly recommended NOT silently shifting all internal
data, but **first running a dedicated diagnostic** that:

  - covers PREDISPATCH, PD7Day, SDO, P5MIN, and actuals;
  - reports same/-30/+30 minute MAE;
  - stratifies by source, horizon, regime, and run period.

This script is that diagnostic. It compares each AEMO-derived forecast row (rrp
column where present) against `actuals_sa1.parquet.rrp` at three timestamp
shifts and produces a tidy CSV with one row per (source, horizon_bucket,
regime, run_period, shift_min).

The output's role is to inform a controlled rebuild/retrain branch, not to
trigger a silent change. See the audit doc's "Do Not Silently Patch Yet"
section.

Usage:
  python eval/timestamp_alignment_diagnostic.py
  python eval/timestamp_alignment_diagnostic.py --since 2025-06-01 --output ts_align.csv
  python eval/timestamp_alignment_diagnostic.py --sources predispatch,pd7day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"

SOURCE_FILES = {
    "predispatch": (PARQUET_DIR / "aemo_predispatch_sa1.parquet", "rrp"),
    "pd7day":      (PARQUET_DIR / "aemo_pd7day_sa1.parquet",     "rrp"),
    "p5min":       (PARQUET_DIR / "aemo_p5min_sa1.parquet",      "rrp"),
}
ACTUALS_FILE = PARQUET_DIR / "actuals_sa1.parquet"

# Horizon buckets (minutes between run_time and interval_dt). Edges chosen to
# expose the PREDISPATCH coverage curve and the PD7Day-only zone.
HORIZON_BUCKETS = [
    (0,     30,    "h<=30m"),
    (30,    360,   "h=30m-6h"),
    (360,   840,   "h=6-14h"),
    (840,   1680,  "h=14-28h"),
    (1680,  4320,  "h=28-72h"),
    (4320,  10080, "h=72h-7d"),
]

# Regime buckets ($/MWh on the actual realised price). Chosen to match the
# pricing regimes the dispatch eval and structural critique have used.
REGIME_BUCKETS = [
    (-1e9, 0,    "negative"),
    (0,    60,   "low"),
    (60,   150,  "normal"),
    (150,  300,  "elevated"),
    (300,  1e9,  "spike"),
]

# Run-period buckets (year-quarter strings derived from run_time). Lets us see
# whether AEMO's timestamp convention may have shifted at some point.
def _run_period(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{((ts.month - 1) // 3) + 1}"


def _bucket(value: float, buckets):
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return "unbucketed"


def load_actuals() -> pd.Series:
    df = pd.read_parquet(ACTUALS_FILE, columns=["time", "rrp"]).dropna()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    s = df.set_index("time")["rrp"].sort_index()
    return s[~s.index.duplicated(keep="last")]


def load_source(label: str, path: Path, value_col: str,
                since: pd.Timestamp | None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cols = ["interval_dt", "run_time", value_col]
    df = pd.read_parquet(path, columns=cols).dropna()
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    df["run_time"]    = pd.to_datetime(df["run_time"], utc=True)
    if since is not None:
        df = df[df["run_time"] >= since]
    df = df.rename(columns={value_col: "forecast_mwh"})
    df["source"] = label
    df["horizon_min"] = (
        (df["interval_dt"] - df["run_time"]).dt.total_seconds() / 60.0
    )
    # Drop pathological negative or absurd horizons (data hygiene).
    df = df[(df["horizon_min"] >= 0) & (df["horizon_min"] <= 10080)]
    df["horizon_bucket"] = df["horizon_min"].apply(
        lambda m: _bucket(m, HORIZON_BUCKETS)
    )
    df["run_period"] = df["run_time"].apply(_run_period)
    return df


def join_at_shift(forecasts: pd.DataFrame, actuals: pd.Series,
                  shift_min: int) -> pd.DataFrame:
    """Return forecasts with `actual_mwh` joined at `interval_dt + shift_min`.

    shift_min = 0   : raw interval_dt matched against actuals
    shift_min = -30 : interval_dt - 30 min (AEMO interval-end → control interval-start)
    shift_min = +30 : interval_dt + 30 min (opposite direction sanity check)
    """
    if forecasts.empty:
        return forecasts
    target = forecasts["interval_dt"] + pd.Timedelta(minutes=shift_min)
    actual_values = actuals.reindex(target).values
    out = forecasts.copy()
    out["actual_mwh"] = actual_values
    out["shift_min"]  = shift_min
    out = out.dropna(subset=["actual_mwh"])
    return out


def compute_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    joined = joined.copy()
    joined["error"]     = joined["forecast_mwh"] - joined["actual_mwh"]
    joined["abs_error"] = joined["error"].abs()
    joined["regime"]    = joined["actual_mwh"].apply(
        lambda v: _bucket(v, REGIME_BUCKETS)
    )

    rows = []
    keys = ["source", "horizon_bucket", "regime", "run_period", "shift_min"]
    for key_vals, sub in joined.groupby(keys, observed=True):
        rows.append({
            **dict(zip(keys, key_vals)),
            "n":           int(len(sub)),
            "mae_mwh":     float(sub["abs_error"].mean()),
            "median_mwh":  float(sub["abs_error"].median()),
            "bias_mwh":    float(sub["error"].mean()),
        })
    return pd.DataFrame(rows)


def winner_per_stratum(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (source, horizon_bucket, regime, run_period), identify the
    shift_min with the lowest MAE."""
    if summary.empty:
        return summary
    keys = ["source", "horizon_bucket", "regime", "run_period"]
    idx = summary.groupby(keys, observed=True)["mae_mwh"].idxmin()
    winners = summary.loc[idx, keys + ["shift_min", "mae_mwh", "n"]].rename(
        columns={"shift_min": "winning_shift_min", "mae_mwh": "best_mae_mwh"}
    )
    return winners.sort_values(keys).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", default="predispatch,pd7day,p5min",
        help="Comma-separated source labels. Available: "
             + ",".join(SOURCE_FILES),
    )
    parser.add_argument(
        "--since", default="",
        help="Only include forecast rows with run_time >= this UTC date "
             "(e.g. 2025-06-01). Default: include everything available.",
    )
    parser.add_argument(
        "--output", default="",
        help="Path to write the per-stratum summary CSV. Default: stdout view.",
    )
    parser.add_argument(
        "--winners-output", default="",
        help="Path to write the winner-shift-per-stratum CSV.",
    )
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown = [s for s in sources if s not in SOURCE_FILES]
    if unknown:
        parser.error(f"Unknown sources: {unknown}")

    since = pd.Timestamp(args.since, tz="UTC") if args.since else None

    print(f"Loading actuals from {ACTUALS_FILE}…", file=sys.stderr)
    actuals = load_actuals()
    print(f"  {len(actuals):,} rows, range {actuals.index.min()} → "
          f"{actuals.index.max()}", file=sys.stderr)

    all_summaries = []
    for src in sources:
        path, val_col = SOURCE_FILES[src]
        print(f"\nLoading {src} from {path.name}…", file=sys.stderr)
        forecasts = load_source(src, path, val_col, since)
        if forecasts.empty:
            print(f"  [{src}] no rows after filter", file=sys.stderr)
            continue
        print(f"  {len(forecasts):,} forecast rows", file=sys.stderr)
        joined_parts = []
        for shift_min in (-30, 0, 30):
            j = join_at_shift(forecasts, actuals, shift_min)
            print(f"  [{src}] shift={shift_min:+d}min  matched={len(j):,}",
                  file=sys.stderr)
            joined_parts.append(j)
        joined = pd.concat(joined_parts, ignore_index=True)
        summary = compute_summary(joined)
        all_summaries.append(summary)

    if not all_summaries:
        print("No data to summarise.", file=sys.stderr)
        return 0

    summary = pd.concat(all_summaries, ignore_index=True).sort_values(
        ["source", "horizon_bucket", "regime", "run_period", "shift_min"]
    ).reset_index(drop=True)

    winners = winner_per_stratum(summary)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Wrote per-stratum summary ({len(summary)} rows) → {args.output}",
              file=sys.stderr)
    if args.winners_output:
        Path(args.winners_output).parent.mkdir(parents=True, exist_ok=True)
        winners.to_csv(args.winners_output, index=False)
        print(f"Wrote winners summary ({len(winners)} rows) → "
              f"{args.winners_output}", file=sys.stderr)

    if not (args.output or args.winners_output):
        # Print a compact view: shift winner share per (source, horizon_bucket).
        share = winners.groupby(
            ["source", "horizon_bucket", "winning_shift_min"], observed=True
        )["n"].sum().reset_index()
        total = share.groupby(["source", "horizon_bucket"], observed=True)[
            "n"
        ].transform("sum")
        share["share_pct"] = (share["n"] / total * 100.0).round(1)
        pivot = share.pivot_table(
            index=["source", "horizon_bucket"],
            columns="winning_shift_min", values="share_pct",
            fill_value=0.0,
        ).round(1)
        pivot.columns = [f"shift={c:+d}min" for c in pivot.columns]
        print()
        print("Winning shift share (% of sample-weighted strata) per source × horizon:")
        print(pivot.to_string())

        # Plus a global MAE-by-shift snapshot per source.
        print()
        global_mae = summary.groupby(["source", "shift_min"], observed=True).apply(
            lambda g: float(np.average(g["mae_mwh"], weights=g["n"]))
        ).unstack()
        global_mae.columns = [f"shift={c:+d}min" for c in global_mae.columns]
        print("Sample-weighted MAE ($/MWh) per source × shift:")
        print(global_mae.round(2).to_string())


if __name__ == "__main__":
    sys.exit(main() or 0)
