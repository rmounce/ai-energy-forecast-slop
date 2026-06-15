#!/usr/bin/env python3
"""Backfill AEMO STPASA REGIONSOLUTION data into parquet.

STPASA is useful for the price-model tail because it starts after the
PREDISPATCH horizon and extends well beyond 72h. This script reads monthly
NEMWeb archive ZIPs or current hourly PUBLIC_STPASA files and writes a compact
region-level parquet.

Output schema:
  interval_dt, run_time, demand10, demand50, demand90,
  aggregate_capacity_available, aggregate_pasa_availability,
  total_intermittent_generation, demand_and_nonschedgen, uigf,
  semischeduled_capacity, lor_semischeduled_capacity,
  ss_solar_uigf, ss_wind_uigf, ss_solar_capacity, ss_wind_capacity,
  ss_solar_cleared, ss_wind_cleared

AEMO timestamps are NEM time (AEST, UTC+10, no DST). Output timestamps are UTC
to match the rest of the parquet layer.
"""

from __future__ import annotations

import argparse
import csv
from html.parser import HTMLParser
import io
import re
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "nemseer_cache" / "stpasa_regionsolution"
DEFAULT_OUT = ROOT / "data" / "parquet" / "aemo_stpasa_regionsolution_sa1.parquet"

NEM_TZ_OFFSET = timedelta(hours=10)
CURRENT_STPASA_DIR = "https://www.nemweb.com.au/Reports/CURRENT/Short_Term_PASA_Reports/"
CURRENT_ZIP_RE = re.compile(r"PUBLIC_STPASA_(\d{10})\d{2}_\d+\.zip$", re.IGNORECASE)

NEMWEB_ARCHIVE_BASE = (
    "https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM"
    "/{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader"
    "/DATA"
    "/PUBLIC_ARCHIVE%2523STPASA_REGIONSOLUTION%2523FILE01%2523{year}{month:02d}010000.zip"
)

RAW_TO_OUTPUT = {
    "DEMAND10": "demand10",
    "DEMAND50": "demand50",
    "DEMAND90": "demand90",
    "AGGREGATECAPACITYAVAILABLE": "aggregate_capacity_available",
    "AGGREGATEPASAAVAILABILITY": "aggregate_pasa_availability",
    "TOTALINTERMITTENTGENERATION": "total_intermittent_generation",
    "DEMAND_AND_NONSCHEDGEN": "demand_and_nonschedgen",
    "UIGF": "uigf",
    "SEMISCHEDULEDCAPACITY": "semischeduled_capacity",
    "LOR_SEMISCHEDULEDCAPACITY": "lor_semischeduled_capacity",
    "SS_SOLAR_UIGF": "ss_solar_uigf",
    "SS_WIND_UIGF": "ss_wind_uigf",
    "SS_SOLAR_CAPACITY": "ss_solar_capacity",
    "SS_WIND_CAPACITY": "ss_wind_capacity",
    "SS_SOLAR_CLEARED": "ss_solar_cleared",
    "SS_WIND_CLEARED": "ss_wind_cleared",
}

OUTPUT_COLUMNS = ["interval_dt", "run_time", *RAW_TO_OUTPUT.values()]


def nem_to_utc(dt_series: pd.Series) -> pd.Series:
    """Convert naive NEM-time (AEST UTC+10) datetime series to UTC-aware."""

    return (dt_series - NEM_TZ_OFFSET).dt.tz_localize("UTC")


def archive_url(year: int, month: int) -> str:
    return NEMWEB_ARCHIVE_BASE.format(year=year, month=month)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def _parse_aemo_csv(fileobj, *, table_name: str | None = None) -> pd.DataFrame:
    """Parse AEMO I/D/C row CSV format.

    Current STPASA ZIPs contain several tables in one CSV. ``table_name`` keeps
    parsing scoped to REGIONSOLUTION instead of accidentally mixing rows from
    CASESOLUTION, INTERCONNECTORSOLN, or CONSTRAINTSOLUTION.
    """

    cols_by_table: dict[str, list[str]] = {}
    rows = []
    reader = csv.reader(io.TextIOWrapper(fileobj, encoding="utf-8", errors="replace"))
    for row in reader:
        if len(row) <= 4:
            continue
        table = row[2] if len(row) > 2 else ""
        if table_name is not None and table != table_name:
            continue
        if row[0] == "I":
            cols_by_table[table] = row[4:]
        elif row[0] == "D":
            rows.append(row[4:])
    if table_name is not None:
        cols = cols_by_table.get(table_name)
    elif len(cols_by_table) == 1:
        cols = next(iter(cols_by_table.values()))
    else:
        cols = next(reversed(cols_by_table.values()), None)
    if not cols or not rows:
        return pd.DataFrame()
    ncols = len(cols)
    rows = [r[:ncols] if len(r) >= ncols else r + [""] * (ncols - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=cols)


def _download_zip(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    response = None
    for attempt in range(5):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
        if response.status_code not in {403, 429, 500, 502, 503, 504}:
            break
        time.sleep(2 * (attempt + 1))
    assert response is not None
    response.raise_for_status()
    cache_path.write_bytes(response.content)
    return response.content


def _current_month_bounds(start: tuple[int, int], end: tuple[int, int]) -> tuple[str, str]:
    start_year, start_month = start
    end_year, end_month = end
    next_year, next_month = end_year, end_month + 1
    if next_month > 12:
        next_year, next_month = next_year + 1, 1
    return f"{start_year:04d}{start_month:02d}0100", f"{next_year:04d}{next_month:02d}0100"


def list_current_stpasa_files(
    *,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[str, str]]:
    """List current hourly PUBLIC_STPASA ZIPs for an inclusive month range."""

    response = requests.get(CURRENT_STPASA_DIR, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    response.raise_for_status()
    parser = _HrefParser()
    parser.feed(response.text)

    start_key, end_key = _current_month_bounds(start, end)
    out = []
    for href in parser.hrefs:
        filename = href.rstrip("/").split("/")[-1]
        match = CURRENT_ZIP_RE.match(filename)
        if not match:
            continue
        run_key = match.group(1)
        if start_key <= run_key < end_key:
            url = href if href.startswith("http") else f"https://www.nemweb.com.au{href}"
            out.append((run_key, url))
    return sorted(out)


def normalise_regionsolution(raw: pd.DataFrame, region_id: str = "SA1") -> pd.DataFrame:
    """Filter and normalise raw STPASA REGIONSOLUTION rows."""

    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    required = {"RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"STPASA REGIONSOLUTION missing required columns: {sorted(missing)}")

    df = raw[raw["REGIONID"].astype(str).str.strip() == region_id].copy()
    if "INTERVENTION" in df.columns:
        intervention = pd.to_numeric(df["INTERVENTION"], errors="coerce")
        df = df[intervention.fillna(0).eq(0)].copy()
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["run_time"] = nem_to_utc(
        pd.to_datetime(df["RUN_DATETIME"].astype(str), format="%Y/%m/%d %H:%M:%S", errors="coerce")
    )
    df["interval_dt"] = nem_to_utc(
        pd.to_datetime(df["INTERVAL_DATETIME"].astype(str), format="%Y/%m/%d %H:%M:%S", errors="coerce")
    )

    out = df[["interval_dt", "run_time"]].copy()
    for raw_col, out_col in RAW_TO_OUTPUT.items():
        if raw_col in df.columns:
            out[out_col] = pd.to_numeric(df[raw_col], errors="coerce")
        else:
            out[out_col] = pd.NA

    out = out.dropna(subset=["interval_dt", "run_time"])
    out = out[OUTPUT_COLUMNS].sort_values(["run_time", "interval_dt"]).reset_index(drop=True)
    return out


def fetch_month(year: int, month: int, region_id: str = "SA1") -> pd.DataFrame:
    """Download and parse one monthly STPASA REGIONSOLUTION archive."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = archive_url(year, month)
    cache_path = CACHE_DIR / f"stpasa_regionsolution_{year}{month:02d}.zip"
    raw_zip = _download_zip(url, cache_path)

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV member found in {url}")
        with zf.open(csv_names[0]) as f:
            raw = _parse_aemo_csv(f, table_name="REGIONSOLUTION")
    return normalise_regionsolution(raw, region_id=region_id)


def fetch_current_file(run_key: str, url: str, region_id: str = "SA1") -> pd.DataFrame:
    """Download and parse one current hourly PUBLIC_STPASA archive."""

    current_cache = CACHE_DIR / "current"
    current_cache.mkdir(parents=True, exist_ok=True)
    cache_path = current_cache / f"stpasa_{run_key}.zip"
    raw_zip = _download_zip(url, cache_path)

    with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
        csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV member found in {url}")
        with zf.open(csv_names[0]) as f:
            raw = _parse_aemo_csv(f, table_name="REGIONSOLUTION")
    return normalise_regionsolution(raw, region_id=region_id)


def iter_months(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def horizon_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["run_time", "min_horizon_hours", "max_horizon_hours", "rows"])
    tmp = df[["run_time", "interval_dt"]].copy()
    tmp["horizon_hours"] = (tmp["interval_dt"] - tmp["run_time"]).dt.total_seconds() / 3600.0
    return (
        tmp.groupby("run_time", observed=False)
        .agg(
            min_horizon_hours=("horizon_hours", "min"),
            max_horizon_hours=("horizon_hours", "max"),
            rows=("horizon_hours", "size"),
        )
        .reset_index()
    )


def validate_horizon(df: pd.DataFrame, *, min_horizon_hours: float) -> None:
    """Fail if any STPASA run does not reach the required horizon."""

    summary = horizon_summary(df)
    if summary.empty:
        raise ValueError("STPASA data is empty; cannot validate horizon")
    short = summary[summary["max_horizon_hours"] < min_horizon_hours]
    if not short.empty:
        min_max_horizon = float(short["max_horizon_hours"].min())
        raise ValueError(
            f"{len(short):,}/{len(summary):,} STPASA runs stop before "
            f"{min_horizon_hours:.1f}h; shortest max horizon is {min_max_horizon:.1f}h"
        )


def month_arg(value: str) -> tuple[int, int]:
    try:
        year, month = map(int, value.split("-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("month must be YYYY-MM") from exc
    if month < 1 or month > 12:
        raise argparse.ArgumentTypeError("month must be 01..12")
    return year, month


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill AEMO STPASA REGIONSOLUTION to parquet")
    parser.add_argument(
        "--source",
        choices=["archive", "current"],
        default="archive",
        help=(
            "archive reads monthly MMSDM archive ZIPs; current reads hourly "
            "PUBLIC_STPASA files from NEMWeb CURRENT/Short_Term_PASA_Reports"
        ),
    )
    parser.add_argument("--start", default="2026-04", help="First archive month, YYYY-MM")
    parser.add_argument("--end", default="2026-04", help="Last archive month, YYYY-MM")
    parser.add_argument("--region", default="SA1", help="AEMO region, default SA1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Fetch first month only and do not write parquet")
    parser.add_argument(
        "--min-horizon-hours",
        type=float,
        default=72.0,
        help="Fail if the fetched data does not reach this horizon at least once.",
    )
    args = parser.parse_args()

    start_year, start_month = month_arg(args.start)
    end_year, end_month = month_arg(args.end)
    if (end_year, end_month) < (start_year, start_month):
        raise SystemExit("--end must be >= --start")

    if args.dry_run:
        end_year, end_month = start_year, start_month

    months = list(iter_months(start_year, start_month, end_year, end_month))
    region_id = args.region.upper()

    print("=== AEMO STPASA REGIONSOLUTION Backfill ===")
    print(f"  Source: {args.source}")
    print(f"  Range: {args.start} -> {args.end}")
    print(f"  Region: {region_id}")
    print(f"  Cache:  {CACHE_DIR}")
    print(f"  Output: {args.output}")

    frames = []
    if args.source == "archive":
        for i, (year, month) in enumerate(months, 1):
            print(f"[{i}/{len(months)}] {year}-{month:02d} ...", end=" ", flush=True)
            try:
                month_df = fetch_month(year, month, region_id=region_id)
            except requests.HTTPError as exc:
                print(f"HTTP {exc.response.status_code}; skipped")
                continue
            n_runs = month_df["run_time"].nunique() if not month_df.empty else 0
            print(f"{len(month_df):,} rows ({n_runs:,} runs)")
            if not month_df.empty:
                frames.append(month_df)
    else:
        current_files = list_current_stpasa_files(
            start=(start_year, start_month),
            end=(end_year, end_month),
        )
        if args.dry_run:
            current_files = current_files[:1]
        print(f"  Current files: {len(current_files):,}")
        for i, (run_key, url) in enumerate(current_files, 1):
            if i == 1 or i == len(current_files) or i % 100 == 0:
                print(f"[{i}/{len(current_files)}] {run_key} ...", end=" ", flush=True)
            try:
                run_df = fetch_current_file(run_key, url, region_id=region_id)
            except requests.HTTPError as exc:
                if i == 1 or i == len(current_files) or i % 100 == 0:
                    print(f"HTTP {exc.response.status_code}; skipped")
                continue
            n_runs = run_df["run_time"].nunique() if not run_df.empty else 0
            if i == 1 or i == len(current_files) or i % 100 == 0:
                print(f"{len(run_df):,} rows ({n_runs:,} runs)")
            if not run_df.empty:
                frames.append(run_df)

    if not frames:
        print("No STPASA rows fetched.")
        raise SystemExit(1)

    backfill = pd.concat(frames, ignore_index=True)
    before = len(backfill)
    backfill = backfill.drop_duplicates(["run_time", "interval_dt"], keep="last")
    if before != len(backfill):
        print(f"  Removed {before - len(backfill):,} duplicate fetched rows")

    validate_horizon(backfill, min_horizon_hours=args.min_horizon_hours)
    horizons = horizon_summary(backfill)
    print(
        "  Horizon range: "
        f"{horizons['min_horizon_hours'].min():.1f}h -> {horizons['max_horizon_hours'].max():.1f}h"
    )

    if args.dry_run:
        print("\n[dry-run] Not writing parquet. Sample:")
        print(backfill.head(5).to_string(index=False))
        return

    out_file = args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        existing = pd.read_parquet(out_file)
        print(f"Existing parquet: {len(existing):,} rows ({existing['run_time'].nunique():,} runs)")
        combined = pd.concat([existing, backfill], ignore_index=True)
    else:
        print("No existing parquet; writing fresh.")
        combined = backfill

    before = len(combined)
    combined = combined.drop_duplicates(["run_time", "interval_dt"], keep="last")
    if before != len(combined):
        print(f"  Removed {before - len(combined):,} duplicate merged rows")
    combined = combined.sort_values(["run_time", "interval_dt"]).reset_index(drop=True)
    validate_horizon(combined, min_horizon_hours=args.min_horizon_hours)

    combined.to_parquet(out_file, index=False, compression="snappy")
    print(f"Saved: {out_file}")
    print(f"Rows: {len(combined):,}; runs: {combined['run_time'].nunique():,}")
    print(f"run_time range: {combined['run_time'].min()} -> {combined['run_time'].max()}")
    print("Next: run eval/analyze_lgbm_residual_drivers.py with this parquet present.")


if __name__ == "__main__":
    main()
