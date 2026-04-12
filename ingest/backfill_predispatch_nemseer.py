#!/usr/bin/env python3
"""
Backfill AEMO PREDISPATCH SA1 data via NEMSEER (+ direct NEMWeb for 2024-08+).

AEMO restructured their NEMWeb archive in August 2024:
  Before: DATA/PUBLIC_DVD_PREDISPATCH{TABLE}_{YYYYMM}010000.zip  (NEMSEER handles)
  After:  PREDISP_ALL_DATA/PUBLIC_ARCHIVE#PREDISPATCH{TABLE}#ALL#FILE01#...zip

This script uses NEMSEER for months up to July 2024 and direct downloads
for August 2024 onwards (identical CSV format, different URL).

NEMSEER/AEMO times are in NEM time (AEST, UTC+10, no DST). Output is
converted to UTC to match the existing parquet schema.

Raw download cache: data/nemseer_cache/ (~1 GB for 2 years; safe to delete after run)

Usage:
    python ingest/backfill_predispatch_nemseer.py
    python ingest/backfill_predispatch_nemseer.py --start 2024-04 --end 2025-02
    python ingest/backfill_predispatch_nemseer.py --dry-run

Default:
    --start 2024-04   (2 years back from current dataset)
    --end   2025-02   (month before existing parquet begins 2025-03-22)
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
# ────────────────────────────────────────────────────────────────────────────

import nemseer  # noqa: E402 (must import after patch)

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "nemseer_cache"

NEM_TZ_OFFSET = timedelta(hours=10)   # AEST = UTC+10, no DST in NEM time

# AEMO switched archive format starting August 2024
DIRECT_DOWNLOAD_FROM = (2024, 8)

NEMWEB_BASE = (
    "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader"
    "/PREDISP_ALL_DATA"
    "/PUBLIC_ARCHIVE%23PREDISPATCH{table}%23ALL%23FILE01%23{year}{month:02d}010000.zip"
)


def nem_to_utc(dt_series: pd.Series) -> pd.Series:
    """Convert naive NEM-time (AEST UTC+10) datetime series to UTC-aware."""
    return (dt_series - NEM_TZ_OFFSET).dt.tz_localize("UTC")


def _parse_predispatch_seqno(df: pd.DataFrame) -> pd.DataFrame:
    """Compute PREDISPATCH_RUN_DATETIME from PREDISPATCHSEQNO.

    Mirrors nemseer's data_handlers logic:
      date  = PREDISPATCHSEQNO[:8] as YYYYMMDD
      period = PREDISPATCHSEQNO[8:10]
      run_time (NEM) = date + (period-1)*30min + 4h30min
    """
    seq = df["PREDISPATCHSEQNO"].astype(str).str.zfill(10)
    parsed = seq.str.extract(r"^([0-9]{8})([0-9]{2})$")
    year_month_day = pd.to_datetime(parsed[0], format="%Y%m%d")
    hour_min = ((parsed[1].astype(int) - 1) * pd.Timedelta(minutes=30)).add(
        pd.Timedelta(hours=4, minutes=30)
    )
    df = df.copy()
    df["PREDISPATCH_RUN_DATETIME"] = year_month_day + hour_min
    return df


def _parse_aemo_csv(fileobj, subtable: str) -> pd.DataFrame:
    """Parse AEMO standard CSV format (I/D/C rows) for a given subtable.

    Handles quoted fields. Returns raw DataFrame before type conversions.
    """
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
    # Pad/trim rows to match column count
    ncols = len(cols)
    rows = [r[:ncols] if len(r) >= ncols else r + [""] * (ncols - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=cols)


def _download_zip(url: str, cache_path: Path) -> bytes:
    """Download ZIP from URL, caching to disk."""
    if cache_path.exists():
        return cache_path.read_bytes()
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return r.content


def fetch_month_direct(year: int, month: int, region_id: str = "SA1") -> pd.DataFrame:
    """Fetch PREDISPATCH for one month via direct NEMWeb download."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    for table, subtable in [("PRICE", "REGION_PRICES"), ("REGIONSUM", "REGION_SOLUTION")]:
        url = NEMWEB_BASE.format(year=year, month=month, table=table)
        cache_file = CACHE_DIR / f"direct_{year}{month:02d}_{table}.zip"
        raw = _download_zip(url, cache_file)
        z = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = _parse_aemo_csv(f, subtable)
        if df.empty:
            print(f"  WARNING: Empty {table} for {year}-{month:02d}")
            return pd.DataFrame(columns=["interval_dt", "run_time", "rrp",
                                         "total_demand", "net_interchange"])
        # Type conversions
        for col in ("PREDISPATCHSEQNO", "PERIODID", "INTERVENTION"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["RRP"] = pd.to_numeric(df.get("RRP", df.get("rrp", 0)), errors="coerce") if table == "PRICE" else None
        df["TOTALDEMAND"]    = pd.to_numeric(df.get("TOTALDEMAND", 0),    errors="coerce") if table == "REGIONSUM" else None
        df["NETINTERCHANGE"] = pd.to_numeric(df.get("NETINTERCHANGE", 0), errors="coerce") if table == "REGIONSUM" else None
        df["DATETIME"] = pd.to_datetime(df["DATETIME"].astype(str), format="%Y/%m/%d %H:%M:%S", errors="coerce")
        df = _parse_predispatch_seqno(df)
        frames[table] = df

    price_df  = frames["PRICE"]
    region_df = frames["REGIONSUM"]

    # Filter Region, INTERVENTION=0
    price_df  = price_df[(price_df["REGIONID"].str.strip() == region_id) & (price_df["INTERVENTION"] == 0)]
    region_df = region_df[(region_df["REGIONID"].str.strip() == region_id) & (region_df["INTERVENTION"] == 0)]

    if price_df.empty or region_df.empty:
        print(f"  WARNING: No {region_id} data for {year}-{month:02d}")
        return pd.DataFrame(columns=["interval_dt", "run_time", "rrp",
                                     "total_demand", "net_interchange"])

    # Filter Region, INTERVENTION=0
    price_df  = price_df[(price_df["REGIONID"].str.strip() == region_id) & (price_df["INTERVENTION"] == 0)]
    region_df = region_df[(region_df["REGIONID"].str.strip() == region_id) & (region_df["INTERVENTION"] == 0)]

    if price_df.empty or region_df.empty:
        print(f"  WARNING: No {region_id}/INTERVENTION=0 data for {year}-{month:02d}")
        return pd.DataFrame(columns=["interval_dt", "run_time", "rrp",
                                     "total_demand", "net_interchange"])

    price_df  = price_df[["PREDISPATCH_RUN_DATETIME", "DATETIME", "RRP"]].copy()
    region_df = region_df[["PREDISPATCH_RUN_DATETIME", "DATETIME",
                            "TOTALDEMAND", "NETINTERCHANGE"]].copy()

    price_df.rename(columns={"PREDISPATCH_RUN_DATETIME": "run_time",
                              "DATETIME": "interval_dt", "RRP": "rrp"}, inplace=True)
    region_df.rename(columns={"PREDISPATCH_RUN_DATETIME": "run_time",
                               "DATETIME": "interval_dt", "TOTALDEMAND": "total_demand",
                               "NETINTERCHANGE": "net_interchange"}, inplace=True)

    merged = price_df.merge(region_df, on=["run_time", "interval_dt"], how="inner")
    merged["run_time"]    = nem_to_utc(merged["run_time"])
    merged["interval_dt"] = nem_to_utc(merged["interval_dt"])
    return merged[["interval_dt", "run_time", "rrp", "total_demand", "net_interchange"]]


def fetch_month_nemseer(year: int, month: int, region_id: str = "SA1") -> pd.DataFrame:
    """Fetch one calendar month of PREDISPATCH data via NEMSEER."""
    month_start = datetime(year, month, 1)
    next_month  = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    run_end_dt  = next_month - timedelta(minutes=30)

    run_start = month_start.strftime("%Y/%m/%d %H:%M")
    run_end   = run_end_dt.strftime("%Y/%m/%d %H:%M")
    fc_end    = (run_end_dt + timedelta(hours=28)).strftime("%Y/%m/%d %H:%M")

    try:
        data = nemseer.compile_data(
            run_start=run_start, run_end=run_end,
            forecasted_start=run_start, forecasted_end=fc_end,
            forecast_type="PREDISPATCH",
            tables=["PRICE", "REGIONSUM"],
            raw_cache=str(CACHE_DIR),
        )
    except ValueError as e:
        if "Table(s) not available" in str(e):
            print(f"  Attempting with _D suffixes for {year}-{month:02d}...")
            data = nemseer.compile_data(
                run_start=run_start, run_end=run_end,
                forecasted_start=run_start, forecasted_end=fc_end,
                forecast_type="PREDISPATCH",
                tables=["PRICE_D", "REGIONSUM_D"],
                raw_cache=str(CACHE_DIR),
            )
            # Normalize keys back to PRICE/REGIONSUM
            data["PRICE"] = data["PRICE_D"]
            data["REGIONSUM"] = data["REGIONSUM_D"]
        else:
            raise e

    price_df  = data["PRICE"]
    region_df = data["REGIONSUM"]

    price_df  = price_df[(price_df["REGIONID"] == region_id) & (price_df["INTERVENTION"] == 0)]
    region_df = region_df[(region_df["REGIONID"] == region_id) & (region_df["INTERVENTION"] == 0)]

    if price_df.empty or region_df.empty:
        print(f"  WARNING: No data for {year}-{month:02d}")
        return pd.DataFrame(columns=["interval_dt", "run_time", "rrp",
                                     "total_demand", "net_interchange"])

    price_df  = price_df[["PREDISPATCH_RUN_DATETIME", "DATETIME", "RRP"]].copy()
    region_df = region_df[["PREDISPATCH_RUN_DATETIME", "DATETIME",
                            "TOTALDEMAND", "NETINTERCHANGE"]].copy()

    for df, renames in [
        (price_df,  {"PREDISPATCH_RUN_DATETIME": "run_time", "DATETIME": "interval_dt", "RRP": "rrp"}),
        (region_df, {"PREDISPATCH_RUN_DATETIME": "run_time", "DATETIME": "interval_dt",
                     "TOTALDEMAND": "total_demand", "NETINTERCHANGE": "net_interchange"}),
    ]:
        df.rename(columns=renames, inplace=True)

    merged = price_df.merge(region_df, on=["run_time", "interval_dt"], how="inner")
    merged["run_time"]    = nem_to_utc(merged["run_time"])
    merged["interval_dt"] = nem_to_utc(merged["interval_dt"])
    return merged[["interval_dt", "run_time", "rrp", "total_demand", "net_interchange"]]


def fetch_month(year: int, month: int, region_id: str = "SA1") -> pd.DataFrame:
    """Dispatch to NEMSEER or direct download based on archive availability."""
    if (year, month) >= DIRECT_DOWNLOAD_FROM:
        return fetch_month_direct(year, month, region_id=region_id)
    return fetch_month_nemseer(year, month, region_id=region_id)


def iter_months(start_year, start_month, end_year, end_month):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def main():
    parser = argparse.ArgumentParser(description="Backfill PREDISPATCH SA1 via NEMSEER + NEMWeb")
    parser.add_argument("--start", default="2024-04",
                        help="First month to fetch (YYYY-MM, default 2024-04)")
    parser.add_argument("--end", default="2025-02",
                        help="Last month to fetch (YYYY-MM, default 2025-02)")
    parser.add_argument("--region", default="SA1",
                        help="AEMO Region (default SA1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch first month only; do not write parquet")
    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    end_year,   end_month   = map(int, args.end.split("-"))
    region_id = args.region.upper()
    out_file = ROOT / "data" / "parquet" / f"aemo_predispatch_{region_id.lower()}.parquet"

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== NEMSEER PREDISPATCH {region_id} Backfill ===")
    print(f"  Range: {args.start} → {args.end}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Output: {out_file}\n")
    print(f"  Using NEMSEER for months before {DIRECT_DOWNLOAD_FROM[0]}-{DIRECT_DOWNLOAD_FROM[1]:02d}, "
          f"direct NEMWeb download for {DIRECT_DOWNLOAD_FROM[0]}-{DIRECT_DOWNLOAD_FROM[1]:02d}+\n")

    if args.dry_run:
        print("[dry-run] Fetching first month only...\n")
        end_year, end_month = start_year, start_month

    months = list(iter_months(start_year, start_month, end_year, end_month))
    print(f"Fetching {len(months)} month(s)...\n")

    frames = []
    for i, (year, month) in enumerate(months, 1):
        source = "direct" if (year, month) >= DIRECT_DOWNLOAD_FROM else "NEMSEER"
        print(f"[{i}/{len(months)}] {year}-{month:02d} ({source}) ...", end=" ", flush=True)
        df = fetch_month(year, month, region_id=region_id)
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
    if out_file.exists():
        existing = pd.read_parquet(out_file)
        print(f"\nExisting parquet: {len(existing):,} rows  "
              f"({existing['run_time'].nunique()} runs)")
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

    print(f"\nFinal merged parquet: {len(combined):,} rows  "
          f"({combined['run_time'].nunique()} runs)")
    print(f"  run_time range: {combined['run_time'].min()} → {combined['run_time'].max()}")

    combined.to_parquet(out_file, index=False)
    print(f"\nSaved: {out_file}")
    print("\nNext steps:")
    print("  1. python data/build_training_dataset.py")
    print("  2. python train/train_tft_price.py --epochs 100")
    print("  3. python train/evaluate_tft.py")


if __name__ == "__main__":
    main()
