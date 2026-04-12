#!/usr/bin/env python3
"""
Backfill AEMO 5-minute DISPATCH actual prices for SA1 into InfluxDB.

NEMSEER does not support the DISPATCH (actual) data type — uses direct NEMWeb
downloads for all months via two URL patterns:

  Before Aug 2024:  DATA/PUBLIC_DVD_DISPATCH{TABLE}_{YYYYMM}010000.zip
  Aug 2024+:        DATA/PUBLIC_ARCHIVE#DISPATCH{TABLE}#FILE01#{YYYYMM}010000.zip

Downloads DISPATCHPRICE (RRP) and DISPATCHREGIONSUM (demand, interchange),
filters SA1 INTERVENTION=0, converts SETTLEMENTDATE from NEM time (AEST UTC+10)
to UTC, and writes into InfluxDB measurement rp_5m.aemo_dispatch_sa1_5m:
  fields:  price ($/MWh), total_demand (MW), net_interchange (MW)
  no tags

Idempotent: InfluxDB overwrites duplicate timestamps (last-write-wins).
Re-running a month is safe. Existing months can be skipped with --skip-existing.

Raw download cache: data/nemseer_cache/dispatch5m/  (~500 MB for 12 months)

Usage:
    python ingest/backfill_dispatch_5m_nemseer.py
    python ingest/backfill_dispatch_5m_nemseer.py --start 2024-04 --end 2025-03
    python ingest/backfill_dispatch_5m_nemseer.py --dry-run
    python ingest/backfill_dispatch_5m_nemseer.py --skip-existing

Default:
    --start 2024-04   (first month of training dataset)
    --end   2025-03   (month before existing 5m live data begins 2025-03-31)
"""

import argparse
import csv
import io
import json
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from influxdb import InfluxDBClient

ROOT      = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "nemseer_cache" / "dispatch5m"

NEM_TZ_OFFSET  = timedelta(hours=10)   # AEST = UTC+10, no DST in NEM time
MEASUREMENT    = "aemo_dispatch_sa1_5m"
RETENTION_POLICY = "rp_5m"
WRITE_BATCH    = 5_000                  # InfluxDB points per write call

# URL format changed in August 2024
ARCHIVE_FORMAT_FROM = (2024, 8)

NEMWEB_DVD_BASE = (
    "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader"
    "/DATA"
    "/PUBLIC_DVD_DISPATCH{table}_{year}{month:02d}010000.zip"
)

NEMWEB_ARCHIVE_BASE = (
    "https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader"
    "/DATA"
    "/PUBLIC_ARCHIVE%23DISPATCH{table}%23FILE01%23{year}{month:02d}010000.zip"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_config(path="config.json"):
    with open(path) as f:
        return json.load(f)


def influx_client(cfg):
    ic = cfg["influxdb"]
    return InfluxDBClient(
        host=ic["host"], port=ic.get("port", 8086),
        username=ic["username"], password=ic["password"],
        database=ic["database"],
    )


def nem_to_utc(dt_series: pd.Series) -> pd.Series:
    """Convert naive NEM-time (AEST UTC+10) to UTC-aware."""
    return (dt_series - NEM_TZ_OFFSET).dt.tz_localize("UTC")


def _download_zip(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    r.raise_for_status()
    cache_path.write_bytes(r.content)
    return r.content


def _parse_aemo_csv(fileobj) -> pd.DataFrame:
    """Parse AEMO I/D/C row format. Columns from 'I' header row, data from 'D' rows."""
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


# ── fetch ─────────────────────────────────────────────────────────────────────

def url_for(year: int, month: int, table: str) -> str:
    if (year, month) >= ARCHIVE_FORMAT_FROM:
        return NEMWEB_ARCHIVE_BASE.format(year=year, month=month, table=table)
    return NEMWEB_DVD_BASE.format(year=year, month=month, table=table)


def fetch_month(year: int, month: int) -> pd.DataFrame:
    """Download DISPATCHPRICE + DISPATCHREGIONSUM for one month, return merged SA1 df."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    label = "archive" if (year, month) >= ARCHIVE_FORMAT_FROM else "dvd"

    frames = {}
    for table in ("PRICE", "REGIONSUM"):
        url  = url_for(year, month, table)
        path = CACHE_DIR / f"dispatch_{year}{month:02d}_{table}_{label}.zip"
        try:
            raw = _download_zip(url, path)
        except requests.HTTPError as e:
            print(f"    HTTP {e.response.status_code} for {table} — skipping month")
            return pd.DataFrame()

        z = zipfile.ZipFile(io.BytesIO(raw))
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = _parse_aemo_csv(f)
        if df.empty:
            print(f"    WARNING: empty CSV for {table} {year}-{month:02d}")
            return pd.DataFrame()
        frames[table] = df

    price_df  = frames["PRICE"]
    region_df = frames["REGIONSUM"]

    # Numeric coercions
    for df in (price_df, region_df):
        if "INTERVENTION" in df.columns:
            df["INTERVENTION"] = pd.to_numeric(df["INTERVENTION"], errors="coerce")

    # Filter SA1, INTERVENTION=0
    price_df  = price_df[(price_df["REGIONID"].str.strip() == "SA1")
                         & (price_df["INTERVENTION"] == 0)].copy()
    region_df = region_df[(region_df["REGIONID"].str.strip() == "SA1")
                          & (region_df["INTERVENTION"] == 0)].copy()

    if price_df.empty or region_df.empty:
        print(f"    WARNING: no SA1/INTERVENTION=0 rows for {year}-{month:02d}")
        return pd.DataFrame()

    # Parse SETTLEMENTDATE (NEM time → UTC)
    for df in (price_df, region_df):
        df["SETTLEMENTDATE"] = pd.to_datetime(
            df["SETTLEMENTDATE"].astype(str), format="%Y/%m/%d %H:%M:%S", errors="coerce"
        )

    price_df["RRP"]          = pd.to_numeric(price_df["RRP"], errors="coerce")
    region_df["TOTALDEMAND"] = pd.to_numeric(region_df["TOTALDEMAND"], errors="coerce")
    region_df["NETINTERCHANGE"] = pd.to_numeric(region_df["NETINTERCHANGE"], errors="coerce")

    price_df  = price_df[["SETTLEMENTDATE", "RRP"]].copy()
    region_df = region_df[["SETTLEMENTDATE", "TOTALDEMAND", "NETINTERCHANGE"]].copy()

    # Deduplicate: keep last RUNNO per settlement interval (most updated value)
    price_df.drop_duplicates(subset=["SETTLEMENTDATE"], keep="last", inplace=True)
    region_df.drop_duplicates(subset=["SETTLEMENTDATE"], keep="last", inplace=True)

    merged = price_df.merge(region_df, on="SETTLEMENTDATE", how="inner")
    merged.dropna(subset=["RRP", "TOTALDEMAND", "NETINTERCHANGE"], inplace=True)

    merged["time"] = nem_to_utc(merged["SETTLEMENTDATE"])
    merged.rename(columns={"RRP": "price", "TOTALDEMAND": "total_demand",
                            "NETINTERCHANGE": "net_interchange"}, inplace=True)
    return merged[["time", "price", "total_demand", "net_interchange"]]


# ── InfluxDB ──────────────────────────────────────────────────────────────────

def existing_month_range(client) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Return (earliest, latest) UTC timestamps already in InfluxDB, or (None, None)."""
    try:
        res = client.query(
            f'SELECT FIRST(price), LAST(price) FROM "{RETENTION_POLICY}"."{MEASUREMENT}"'
        )
        pts = list(res.get_points())
        if not pts:
            return None, None
        # query doesn't expose time directly for FIRST/LAST; use separate queries
        first_res = client.query(
            f'SELECT price FROM "{RETENTION_POLICY}"."{MEASUREMENT}" '
            f'ORDER BY time ASC LIMIT 1'
        )
        last_res  = client.query(
            f'SELECT price FROM "{RETENTION_POLICY}"."{MEASUREMENT}" '
            f'ORDER BY time DESC LIMIT 1'
        )
        first_pts = list(first_res.get_points())
        last_pts  = list(last_res.get_points())
        t0 = pd.Timestamp(first_pts[0]["time"]) if first_pts else None
        t1 = pd.Timestamp(last_pts[0]["time"])  if last_pts  else None
        return t0, t1
    except Exception:
        return None, None


def month_already_in_influx(client, year: int, month: int) -> bool:
    """Return True if this month appears fully covered in InfluxDB."""
    month_start = datetime(year, month, 1, tzinfo=pd.Timestamp("now", tz="UTC").tzinfo)
    if month == 12:
        month_end = datetime(year + 1, 1, 1)
    else:
        month_end = datetime(year, month + 1, 1)
    # Check count of 5-min rows in the NEM-time-shifted window
    # (SETTLEMENTDATE 00:05 NEM = prior day 14:05 UTC)
    query = (
        f'SELECT COUNT(price) FROM "{RETENTION_POLICY}"."{MEASUREMENT}" '
        f'WHERE time >= \'{month_start.strftime("%Y-%m-%dT%H:%M:%SZ")}\' '
        f'AND time < \'{month_end.strftime("%Y-%m-%dT%H:%M:%SZ")}\''
    )
    try:
        res = list(client.query(query).get_points())
        count = res[0]["count"] if res else 0
        # A full month has at minimum ~8,000 5-min intervals; threshold conservatively
        return count >= 7_000
    except Exception:
        return False


def write_to_influx(client, df: pd.DataFrame, dry_run: bool = False) -> int:
    """Write DataFrame rows as InfluxDB points. Returns count written."""
    points = []
    for row in df.itertuples(index=False):
        points.append({
            "measurement": MEASUREMENT,
            "time": row.time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fields": {
                "price":           float(row.price),
                "total_demand":    float(row.total_demand),
                "net_interchange": float(row.net_interchange),
            },
        })

    if dry_run:
        return len(points)

    for i in range(0, len(points), WRITE_BATCH):
        client.write_points(
            points[i:i + WRITE_BATCH],
            retention_policy=RETENTION_POLICY,
            time_precision="s",
        )
    return len(points)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Backfill AEMO 5-min dispatch actual prices into InfluxDB"
    )
    parser.add_argument("--start", default="2024-04",
                        help="First month to fetch (YYYY-MM, default 2024-04)")
    parser.add_argument("--end", default="2025-03",
                        help="Last month to fetch (YYYY-MM, default 2025-03)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch first month, parse, but do not write to InfluxDB")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip months that appear fully covered in InfluxDB already")
    args = parser.parse_args()

    start_year, start_month = map(int, args.start.split("-"))
    end_year,   end_month   = map(int, args.end.split("-"))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    client = influx_client(cfg)

    print("=== DISPATCH 5-Min Backfill ===")
    print(f"  Range:  {args.start} → {args.end}")
    print(f"  Target: {RETENTION_POLICY}.{MEASUREMENT}")
    print(f"  Cache:  {CACHE_DIR}")
    if args.dry_run:
        print("  [dry-run] will not write to InfluxDB\n")
    if args.skip_existing:
        print("  [skip-existing] will skip months with ≥7,000 points already in InfluxDB\n")

    if args.dry_run:
        end_year, end_month = start_year, start_month

    months = list(iter_months(start_year, start_month, end_year, end_month))
    print(f"Fetching {len(months)} month(s)...\n")

    total_written = 0
    for i, (year, month) in enumerate(months, 1):
        fmt = "archive" if (year, month) >= ARCHIVE_FORMAT_FROM else "dvd"
        print(f"[{i}/{len(months)}] {year}-{month:02d} ({fmt})", end=" ", flush=True)

        if args.skip_existing and not args.dry_run:
            if month_already_in_influx(client, year, month):
                print("— SKIP (already in InfluxDB)")
                continue

        df = fetch_month(year, month)
        if df.empty:
            print("— no data")
            continue

        print(f"→ {len(df):,} intervals", end=" ", flush=True)

        n = write_to_influx(client, df, dry_run=args.dry_run)
        total_written += n
        tag = "[dry-run]" if args.dry_run else "written"
        print(f"({n:,} pts {tag})")

        if args.dry_run:
            print("\nSample rows:")
            print(df.head(5).to_string(index=False))
            break

    print(f"\n{'─'*50}")
    print(f"Total points {'(dry-run) ' if args.dry_run else ''}written: {total_written:,}")
    if not args.dry_run:
        print("\nNext steps:")
        print("  1. python data/export_parquet.py --actuals-5m")
        print("  2. python data/build_training_dataset.py")
        print("  3. python train/train_tft_price.py")
        print("  4. python train/evaluate_tft.py --eval-set stratified")


if __name__ == "__main__":
    main()
