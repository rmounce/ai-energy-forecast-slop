#!/usr/bin/env python3
"""
Backfill AEMO PREDISPATCH SA1 data via NEMSEER.

Downloads PREDISPATCH PRICE + REGIONSUM for SA1 from NEMweb historical archives
and merges into data/parquet/aemo_predispatch_sa1.parquet.

NEMSEER uses NEM time (AEST, UTC+10, no DST). Output is converted to UTC to
match the existing parquet schema.
Raw download cache: data/nemseer_cache/ (~1 GB for 2 years; safe to delete after run)

Usage:
    python ingest/backfill_predispatch_nemseer.py
    python ingest/backfill_predispatch_nemseer.py --start 2024-04 --end 2025-03
    python ingest/backfill_predispatch_nemseer.py --dry-run

Default:
    --start 2024-04   (2 years back from current dataset)
    --end   2025-02   (month before existing parquet begins 2025-03-22)
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

# ── Patch nemseer for pandas 3.0 compatibility (Arrow-backed string columns)
import nemseer.data_handlers as _dh
from nemseer.data import DATETIME_COLS, DATETIME_FORMAT as _DTFMT

def _patched_parse_datetime_cols(df):
    dt_cols_present = DATETIME_COLS.intersection(set(df.columns.tolist()))
    for col in dt_cols_present:
        df[col] = pd.to_datetime(df[col].astype(str), format=_DTFMT + ":%S")
    return df

_dh._parse_datetime_cols = _patched_parse_datetime_cols
# ────────────────────────────────────────────────────────────────────────────

import nemseer  # noqa: E402 (must import after patch)

ROOT       = Path(__file__).resolve().parent.parent
PARQUET    = ROOT / "data" / "parquet" / "aemo_predispatch_sa1.parquet"
CACHE_DIR  = ROOT / "data" / "nemseer_cache"

NEM_TZ_OFFSET = timedelta(hours=10)   # AEST = UTC+10, no DST in NEM time


def nem_to_utc(dt_series: pd.Series) -> pd.Series:
    """Convert naive NEM-time (AEST UTC+10) datetime series to UTC-aware."""
    return (dt_series - NEM_TZ_OFFSET).dt.tz_localize("UTC")


def fetch_month(year: int, month: int) -> pd.DataFrame:
    """Fetch one calendar month of PREDISPATCH SA1 data via NEMSEER.

    Returns DataFrame with columns: interval_dt, run_time, rrp, total_demand,
    net_interchange — UTC-aware, INTERVENTION=0, SA1 only.
    """
    # NEMSEER date strings are in NEM time (AEST)
    month_start = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)

    # run window: full calendar month in NEM time
    run_end_dt = next_month - timedelta(minutes=30)
    run_start  = month_start.strftime("%Y/%m/%d %H:%M")
    run_end    = run_end_dt.strftime("%Y/%m/%d %H:%M")

    # forecasted window: fc_end = last run + 28h (PREDISPATCH max horizon)
    fc_start = run_start
    fc_end   = (run_end_dt + timedelta(hours=28)).strftime("%Y/%m/%d %H:%M")

    data = nemseer.compile_data(
        run_start=run_start,
        run_end=run_end,
        forecasted_start=fc_start,
        forecasted_end=fc_end,
        forecast_type="PREDISPATCH",
        tables=["PRICE", "REGIONSUM"],
        raw_cache=str(CACHE_DIR),
    )

    price_df  = data["PRICE"]
    region_df = data["REGIONSUM"]

    # Filter: SA1 region, INTERVENTION=0 (standard non-intervention forecast)
    price_df  = price_df[(price_df["REGIONID"] == "SA1") & (price_df["INTERVENTION"] == 0)]
    region_df = region_df[(region_df["REGIONID"] == "SA1") & (region_df["INTERVENTION"] == 0)]

    if price_df.empty or region_df.empty:
        print(f"  WARNING: No data for {year}-{month:02d}")
        return pd.DataFrame(columns=["interval_dt", "run_time", "rrp",
                                     "total_demand", "net_interchange"])

    # Select and rename columns
    price_df  = price_df[["PREDISPATCH_RUN_DATETIME", "DATETIME", "RRP"]].copy()
    region_df = region_df[["PREDISPATCH_RUN_DATETIME", "DATETIME",
                            "TOTALDEMAND", "NETINTERCHANGE"]].copy()

    price_df.rename(columns={
        "PREDISPATCH_RUN_DATETIME": "run_time",
        "DATETIME": "interval_dt",
        "RRP": "rrp",
    }, inplace=True)
    region_df.rename(columns={
        "PREDISPATCH_RUN_DATETIME": "run_time",
        "DATETIME": "interval_dt",
        "TOTALDEMAND": "total_demand",
        "NETINTERCHANGE": "net_interchange",
    }, inplace=True)

    # Merge on (run_time, interval_dt)
    merged = price_df.merge(region_df, on=["run_time", "interval_dt"], how="inner")

    # Convert NEM time → UTC
    merged["run_time"]    = nem_to_utc(merged["run_time"])
    merged["interval_dt"] = nem_to_utc(merged["interval_dt"])

    return merged[["interval_dt", "run_time", "rrp", "total_demand", "net_interchange"]]


def iter_months(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def main():
    parser = argparse.ArgumentParser(description="Backfill PREDISPATCH SA1 via NEMSEER")
    parser.add_argument("--start", default="2024-04",
                        help="First month to fetch (YYYY-MM, default 2024-04)")
    parser.add_argument("--end", default="2025-02",
                        help="Last month to fetch (YYYY-MM, default 2025-02)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch first month only and show stats; do not write parquet")
    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    end_year,   end_month   = map(int, args.end.split("-"))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== NEMSEER PREDISPATCH Backfill ===")
    print(f"  Range: {args.start} → {args.end}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Output: {PARQUET}\n")

    if args.dry_run:
        print("[dry-run] Fetching first month only...\n")
        end_year, end_month = start_year, start_month

    months = list(iter_months(start_year, start_month, end_year, end_month))
    print(f"Fetching {len(months)} month(s)...\n")

    frames = []
    for i, (year, month) in enumerate(months, 1):
        print(f"[{i}/{len(months)}] {year}-{month:02d} ...", end=" ", flush=True)
        df = fetch_month(year, month)
        n_runs = df["run_time"].nunique() if not df.empty else 0
        print(f"{len(df):,} rows  ({n_runs} runs)")
        if not df.empty:
            frames.append(df)

    if not frames:
        print("No data fetched. Exiting.")
        sys.exit(1)

    backfill = pd.concat(frames, ignore_index=True)
    print(f"\nBackfill total: {len(backfill):,} rows  "
          f"({backfill['run_time'].nunique()} runs)")
    print(f"  run_time range: {backfill['run_time'].min()} → {backfill['run_time'].max()}")

    if args.dry_run:
        print("\n[dry-run] Not writing parquet. Sample:")
        print(backfill.head(5).to_string())
        return

    # ── Merge with existing parquet
    if PARQUET.exists():
        existing = pd.read_parquet(PARQUET)
        print(f"\nExisting parquet: {len(existing):,} rows  "
              f"({existing['run_time'].nunique()} runs)")
        combined = pd.concat([backfill, existing], ignore_index=True)
    else:
        print("\nNo existing parquet — writing fresh.")
        combined = backfill

    # Deduplicate on (run_time, interval_dt), keep last (existing data preferred)
    before = len(combined)
    combined.drop_duplicates(subset=["run_time", "interval_dt"], keep="last", inplace=True)
    dupes = before - len(combined)
    if dupes:
        print(f"  Removed {dupes:,} duplicate rows (overlap with existing)")

    combined.sort_values(["run_time", "interval_dt"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\nFinal merged parquet: {len(combined):,} rows  "
          f"({combined['run_time'].nunique()} runs)")
    print(f"  run_time range: {combined['run_time'].min()} → {combined['run_time'].max()}")

    combined.to_parquet(PARQUET, index=False)
    print(f"\nSaved: {PARQUET}")
    print("\nNext steps:")
    print("  1. python data/build_training_dataset.py")
    print("  2. python train/train_tft_price.py --epochs 100")
    print("  3. python train/evaluate_tft.py")


if __name__ == "__main__":
    main()
