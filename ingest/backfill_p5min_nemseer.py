#!/usr/bin/env python3
"""
Backfill AEMO P5MIN (5-minute predispatch) forecast data for SA1 into a parquet file.

P5MIN runs every 5 minutes and produces 12 × 5-min interval forecasts (~60 min ahead).
This data is needed to train the Tier 1 tactical LightGBM model.

Archive format changed in August 2024:
  Before 2024-08:  DATA/PUBLIC_DVD_P5MIN_REGIONSOLUTION_YYYYMM010000.zip  (NEMSEER handles)
  2024-08+:        DATA/PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#FILE01#YYYYMM010000.zip  (direct)

Output: data/parquet/aemo_p5min_sa1.parquet
  columns: interval_dt (UTC), run_time (UTC), rrp, total_demand, net_interchange

Idempotent: re-running merges with existing parquet and deduplicates.

Raw download cache: data/nemseer_cache/p5min/  (~2-4 GB for 2 years; safe to delete)

Usage:
    python ingest/backfill_p5min_nemseer.py
    python ingest/backfill_p5min_nemseer.py --start 2024-04 --end 2026-03
    python ingest/backfill_p5min_nemseer.py --dry-run         # first month only, no write
    python ingest/backfill_p5min_nemseer.py --region SA1 VIC1

Default:
    --start 2024-04   (earliest month in training dataset)
    --end   2026-03   (last complete month before live collection from 2026-04-12)
"""

import argparse
import csv
import io
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# ── Patch nemseer for pandas 3.0 compatibility (Arrow-backed string columns)
import nemseer.data_handlers as _dh
from nemseer.data import DATETIME_COLS, DATETIME_FORMAT as _DTFMT

def _patched_parse_datetime_cols(df):
    dt_cols_present = DATETIME_COLS.intersection(set(df.columns.tolist()))
    for col in dt_cols_present:
        df[col] = pd.to_datetime(df[col].astype(str), format=_DTFMT + ":%S")
    return df

_dh._parse_datetime_cols = _patched_parse_datetime_cols
# ─────────────────────────────────────────────────────────────────────────────

import nemseer  # noqa: E402 (must import after patch)

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "nemseer_cache" / "p5min"

NEM_TZ_OFFSET     = timedelta(hours=10)  # AEST = UTC+10, no DST
DIRECT_FROM       = (2024, 8)            # ARCHIVE format starts here

NEMWEB_ARCHIVE_URL = (
    "http://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader"
    "/DATA/PUBLIC_ARCHIVE%23P5MIN_REGIONSOLUTION%23FILE01%23{year}{month:02d}010000.zip"
)

DEFAULT_REGIONS = {"SA1"}


# ── helpers ───────────────────────────────────────────────────────────────────

def nem_to_utc(dt_series: pd.Series) -> pd.Series:
    return (dt_series - NEM_TZ_OFFSET).dt.tz_localize("UTC")


def _download_zip(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return r.content


def _parse_aemo_csv(fileobj) -> pd.DataFrame:
    """Parse AEMO I/D/C row format into a DataFrame."""
    cols = None
    rows = []
    reader = csv.reader(io.TextIOWrapper(fileobj, encoding="utf-8", errors="replace"))
    for row in reader:
        if not row:
            continue
        if row[0] == "I" and len(row) > 4:
            cols = row[4:]
        elif row[0] == "D" and len(row) > 4:
            rows.append(row[4:])
    if not cols or not rows:
        return pd.DataFrame()
    ncols = len(cols)
    rows = [r[:ncols] if len(r) >= ncols else r + [""] * (ncols - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=cols)


def iter_months(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _normalise(df: pd.DataFrame, regions: set) -> pd.DataFrame:
    """Filter to target regions, INTERVENTION=0, and normalise column names/types."""
    if df.empty:
        return df

    # Normalise column names to upper (NEMSEER may vary)
    df.columns = [c.upper() for c in df.columns]

    # Handle NETINTERCHANGE vs NET_INTERCHANGE
    if "NETINTERCHANGE" in df.columns and "NET_INTERCHANGE" not in df.columns:
        df.rename(columns={"NETINTERCHANGE": "NET_INTERCHANGE"}, inplace=True)

    for col in ("INTERVENTION",):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter region and intervention
    if "REGIONID" in df.columns:
        df = df[df["REGIONID"].str.strip().isin(regions)].copy()
    if "INTERVENTION" in df.columns:
        df = df[df["INTERVENTION"] == 0].copy()

    if df.empty:
        return df

    # Parse datetimes
    for col in ("RUN_DATETIME", "INTERVAL_DATETIME"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), format="%Y/%m/%d %H:%M:%S",
                                     errors="coerce")

    for col in ("RRP", "TOTALDEMAND", "NET_INTERCHANGE"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["RUN_DATETIME", "INTERVAL_DATETIME", "RRP"]).copy()

    out = df[["RUN_DATETIME", "INTERVAL_DATETIME", "RRP"]].copy()
    for col, rename in [("TOTALDEMAND", "total_demand"), ("NET_INTERCHANGE", "net_interchange")]:
        if col in df.columns:
            out[rename] = df[col].values
        else:
            out[rename] = float("nan")

    out.rename(columns={
        "RUN_DATETIME": "run_time",
        "INTERVAL_DATETIME": "interval_dt",
        "RRP": "rrp",
    }, inplace=True)

    out["run_time"]    = nem_to_utc(out["run_time"])
    out["interval_dt"] = nem_to_utc(out["interval_dt"])

    return out[["interval_dt", "run_time", "rrp", "total_demand", "net_interchange"]]


# ── fetch methods ─────────────────────────────────────────────────────────────

def fetch_month_nemseer(year: int, month: int, regions: set) -> pd.DataFrame:
    """Fetch one month via NEMSEER (pre-2024-08 DVD format)."""
    month_start = datetime(year, month, 1)
    next_month  = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    run_end_dt  = next_month - timedelta(minutes=5)

    run_start = month_start.strftime("%Y/%m/%d %H:%M")
    run_end   = run_end_dt.strftime("%Y/%m/%d %H:%M")
    fc_end    = (run_end_dt + timedelta(minutes=55)).strftime("%Y/%m/%d %H:%M")

    data = nemseer.compile_data(
        run_start=run_start, run_end=run_end,
        forecasted_start=run_start, forecasted_end=fc_end,
        forecast_type="P5MIN",
        tables=["REGIONSOLUTION"],
        raw_cache=str(CACHE_DIR),
    )

    df = data.get("REGIONSOLUTION", pd.DataFrame())
    if df.empty:
        print(f"  WARNING: empty REGIONSOLUTION for {year}-{month:02d}")
        return pd.DataFrame()

    # NEMSEER may have already parsed datetimes — force string representation for
    # _normalise to handle uniformly
    for col in ("RUN_DATETIME", "INTERVAL_DATETIME"):
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y/%m/%d %H:%M:%S")

    return _normalise(df, regions)


def fetch_month_direct(year: int, month: int, regions: set) -> pd.DataFrame:
    """Fetch one month via direct NEMWeb ARCHIVE download (2024-08+)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url  = NEMWEB_ARCHIVE_URL.format(year=year, month=month)
    path = CACHE_DIR / f"p5min_archive_{year}{month:02d}.zip"

    try:
        raw = _download_zip(url, path)
    except requests.HTTPError as e:
        print(f"  HTTP {e.response.status_code} for {year}-{month:02d} — skipping")
        return pd.DataFrame()

    z = zipfile.ZipFile(io.BytesIO(raw))
    csv_name = z.namelist()[0]
    with z.open(csv_name) as f:
        df = _parse_aemo_csv(f)

    if df.empty:
        print(f"  WARNING: empty CSV for {year}-{month:02d}")
        return pd.DataFrame()

    return _normalise(df, regions)


def fetch_month(year: int, month: int, regions: set) -> pd.DataFrame:
    if (year, month) >= DIRECT_FROM:
        return fetch_month_direct(year, month, regions)
    return fetch_month_nemseer(year, month, regions)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill AEMO P5MIN SA1 forecast data to parquet"
    )
    parser.add_argument("--start", default="2024-04",
                        help="First month (YYYY-MM, default 2024-04)")
    parser.add_argument("--end",   default="2026-03",
                        help="Last month (YYYY-MM, default 2026-03)")
    parser.add_argument("--region", nargs="+", default=sorted(DEFAULT_REGIONS),
                        metavar="REGION",
                        help=f"Regions to fetch (default: {' '.join(sorted(DEFAULT_REGIONS))})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch first month only; do not write parquet")
    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    end_year,   end_month   = map(int, args.end.split("-"))
    regions = set(args.region)

    out_file = ROOT / "data" / "parquet" / "aemo_p5min_sa1.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== P5MIN Backfill ===")
    print(f"  Range:   {args.start} → {args.end}")
    print(f"  Regions: {sorted(regions)}")
    print(f"  Cache:   {CACHE_DIR}")
    print(f"  Output:  {out_file}")
    print(f"  Format:  NEMSEER (< 2024-08), direct ARCHIVE (>= 2024-08)")
    if args.dry_run:
        print("  [dry-run] first month only, no write\n")
        end_year, end_month = start_year, start_month

    months = list(iter_months(start_year, start_month, end_year, end_month))
    print(f"\nFetching {len(months)} month(s)...\n")

    frames = []
    for i, (year, month) in enumerate(months, 1):
        source = "direct" if (year, month) >= DIRECT_FROM else "NEMSEER"
        print(f"[{i}/{len(months)}] {year}-{month:02d} ({source}) ...", end=" ", flush=True)
        df = fetch_month(year, month, regions)
        if df.empty:
            print("— no data")
            continue
        n_runs = df["run_time"].nunique()
        print(f"{len(df):,} rows  ({n_runs:,} runs)")
        frames.append(df)
        if args.dry_run:
            print("\nSample:")
            print(df.head(5).to_string(index=False))
            break

    if not frames:
        print("No data fetched.")
        sys.exit(1)

    backfill = pd.concat(frames, ignore_index=True)
    print(f"\nBackfill total: {len(backfill):,} rows  "
          f"({backfill['run_time'].nunique():,} runs)")
    print(f"  run_time range:  {backfill['run_time'].min()} → {backfill['run_time'].max()}")
    print(f"  interval range:  {backfill['interval_dt'].min()} → {backfill['interval_dt'].max()}")

    if args.dry_run:
        print("\n[dry-run] Not writing parquet.")
        return

    # ── Merge with existing parquet
    if out_file.exists():
        existing = pd.read_parquet(out_file)
        print(f"\nExisting parquet: {len(existing):,} rows  "
              f"({existing['run_time'].nunique():,} runs)")
        combined = pd.concat([backfill, existing], ignore_index=True)
    else:
        print("\nNo existing parquet — writing fresh.")
        combined = backfill

    before = len(combined)
    combined.drop_duplicates(subset=["run_time", "interval_dt"], keep="last", inplace=True)
    dupes = before - len(combined)
    if dupes:
        print(f"  Removed {dupes:,} duplicate rows")

    combined.sort_values(["run_time", "interval_dt"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    print(f"\nFinal parquet: {len(combined):,} rows  "
          f"({combined['run_time'].nunique():,} runs)")
    print(f"  run_time range:  {combined['run_time'].min()} → {combined['run_time'].max()}")

    combined.to_parquet(out_file, index=False)
    print(f"\nSaved: {out_file}")
    print("\nNext steps:")
    print("  Tier 1 training data ready — build features and train tactical LightGBM")
    print("  python train/train_lgbm_tactical.py")


if __name__ == "__main__":
    main()
