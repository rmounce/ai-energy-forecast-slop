#!/usr/bin/env python3
"""
Ingest AEMO PD7Day price forecasts into InfluxDB.

Modes:
  --backfill DIR   Process all PD7Day ZIP files in DIR (e.g. scratch/pd7day_backfill/)
  --fetch          Download the latest PD7Day ZIP from NEMweb and ingest it
  --mmsdm ZIP      Backfill from an AEMO MMSDM monthly PD7DAY_PRICESOLUTION archive
                   (cumulative — one zip contains 2 years of forecast runs)

InfluxDB schema:
  measurement: aemo_pd7day_forecast
  time:   interval_datetime (UTC)
  tags:   region (SA1, VIC1, etc.)
          run_time (ISO UTC string)
  fields: rrp ($/MWh)

Idempotent: run_times already present in InfluxDB are skipped.

NEMweb CURRENT zips tag run_time with the file publication time (e.g.
2026-02-09T21:09:53Z). MMSDM archives tag run_time with the AEMO-published
RUN_DATETIME field (top-of-half-hour, e.g. 2024-04-23T21:30:00Z). For the
overlapping window both records coexist with the same `interval_datetime` —
the data is equivalent, only the run_time tag differs.
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import pytz
from influxdb import InfluxDBClient

BASE_URL = "https://www.nemweb.com.au/REPORTS/CURRENT/PD7Day/"
FILE_PATTERN = re.compile(r"PUBLIC_PD7DAY_(\d{14})_\d+\.(ZIP|zip)$")
# PD7Day timestamps are AEST (UTC+10, no DST — Brisbane time)
NEM_TZ = pytz.timezone("Australia/Brisbane")
MEASUREMENT = "aemo_pd7day_forecast"
# Use same retention policy as other 30-min ML input data
RETENTION_POLICY = "rp_30m"
DEFAULT_REGIONS = ["SA1", "VIC1", "NSW1"]

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config_utils import load_config


def influx_client(cfg: dict) -> InfluxDBClient:
    ic = cfg["influxdb"]
    return InfluxDBClient(
        host=ic["host"],
        port=ic.get("port", 8086),
        username=ic["username"],
        password=ic["password"],
        database=ic["database"],
    )


def run_time_from_filename(name: str) -> datetime | None:
    """Extract run_time as UTC datetime from filename timestamp (AEST → UTC)."""
    m = FILE_PATTERN.search(name)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return NEM_TZ.localize(ts).astimezone(timezone.utc)


def fetch_existing_run_times(client: InfluxDBClient) -> set:
    """Return set of run_time tag values already in InfluxDB."""
    try:
        result = client.query(
            f'SHOW TAG VALUES FROM "{MEASUREMENT}" WITH KEY = "run_time"',
            database=client._database,
        )
        return {v["value"] for v in result.get_points()}
    except Exception:
        return set()


def parse_zip(zip_path: Path, run_time_utc: datetime, regions: list[str]) -> list[dict]:
    """Parse PRICESOLUTION rows from a PD7Day ZIP, return InfluxDB points."""
    points = []
    run_time_iso = run_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
        if not csv_names:
            raise ValueError(f"No CSV inside {zip_path.name}")
        with zf.open(sorted(csv_names)[0]) as raw:
            reader = csv.reader(line.decode("utf-8", errors="ignore") for line in raw)
            for row in reader:
                if not row or row[0] != "D":
                    continue
                if len(row) < 9:
                    continue
                if row[1] != "PD7DAY" or row[2] != "PRICESOLUTION":
                    continue
                region = row[7]
                if regions and region not in regions:
                    continue
                interval_str = row[6]  # INTERVAL_DATETIME: "YYYY/MM/DD HH:MM:SS"
                rrp = float(row[8])    # $/MWh — do NOT divide by 1000
                interval_naive = datetime.strptime(interval_str, "%Y/%m/%d %H:%M:%S")
                interval_utc = NEM_TZ.localize(interval_naive).astimezone(timezone.utc)
                points.append({
                    "measurement": MEASUREMENT,
                    "time": interval_utc,
                    "tags": {
                        "region": region,
                        "run_time": run_time_iso,
                    },
                    "fields": {
                        "rrp": rrp,
                    },
                })

    return points


def iter_mmsdm_points(zip_path: Path, regions: list[str], existing_run_times: set):
    """Yield InfluxDB points from an MMSDM PD7DAY_PRICESOLUTION zip.

    One zip contains all forecast runs from the inception of PD7Day through the
    archive month. Each D row has its own RUN_DATETIME (col 4) and
    INTERVAL_DATETIME (col 6), so unlike the NEMweb CURRENT format the run_time
    is parsed per-row rather than from the filename.

    Skips D rows whose run_time is already present in InfluxDB (idempotency).
    """
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
        if not csv_names:
            raise ValueError(f"No CSV inside {zip_path.name}")
        with zf.open(sorted(csv_names)[0]) as raw:
            reader = csv.reader(line.decode("utf-8", errors="ignore") for line in raw)
            for row in reader:
                if not row or row[0] != "D":
                    continue
                if len(row) < 9:
                    continue
                if row[1] != "PD7DAY" or row[2] != "PRICESOLUTION":
                    continue
                region = row[7]
                if regions and region not in regions:
                    continue
                run_naive = datetime.strptime(row[4], "%Y/%m/%d %H:%M:%S")
                run_time_utc = NEM_TZ.localize(run_naive).astimezone(timezone.utc)
                run_time_iso = run_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
                if run_time_iso in existing_run_times:
                    continue
                interval_naive = datetime.strptime(row[6], "%Y/%m/%d %H:%M:%S")
                interval_utc = NEM_TZ.localize(interval_naive).astimezone(timezone.utc)
                try:
                    rrp = float(row[8])
                except ValueError:
                    continue
                yield {
                    "measurement": MEASUREMENT,
                    "time": interval_utc,
                    "tags": {
                        "region": region,
                        "run_time": run_time_iso,
                    },
                    "fields": {
                        "rrp": rrp,
                    },
                }


def ingest_mmsdm(
    zip_path: Path,
    client: InfluxDBClient,
    existing_run_times: set,
    regions: list[str],
    dry_run: bool = False,
    batch_size: int = 10000,
) -> tuple[int, int]:
    """Stream-parse an MMSDM zip and bulk-write to InfluxDB.

    Returns (points_written, new_run_times).
    """
    batch: list[dict] = []
    points_written = 0
    new_run_times: set = set()
    last_report = 0

    for pt in iter_mmsdm_points(zip_path, regions, existing_run_times):
        new_run_times.add(pt["tags"]["run_time"])
        batch.append(pt)
        if len(batch) >= batch_size:
            if not dry_run:
                client.write_points(
                    batch,
                    time_precision="s",
                    retention_policy=RETENTION_POLICY,
                    batch_size=batch_size,
                )
            points_written += len(batch)
            batch = []
            if points_written - last_report >= 100000:
                print(f"  ... {points_written:>10,} points written, "
                      f"{len(new_run_times):>5,} run_times so far", flush=True)
                last_report = points_written

    if batch:
        if not dry_run:
            client.write_points(
                batch,
                time_precision="s",
                retention_policy=RETENTION_POLICY,
                batch_size=batch_size,
            )
        points_written += len(batch)

    existing_run_times.update(new_run_times)
    return points_written, len(new_run_times)


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def fetch_latest_zip(dest_dir: Path) -> Path:
    """Download the newest PD7Day ZIP from NEMweb into dest_dir."""
    with urllib.request.urlopen(BASE_URL) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    parser = LinkExtractor()
    parser.feed(html)
    files = []
    for href in parser.links:
        name = href.split("/")[-1]
        if FILE_PATTERN.search(name):
            files.append((name, urljoin(BASE_URL, href)))
    if not files:
        raise FileNotFoundError("No PD7Day ZIP files found on NEMweb")
    name, url = sorted(files, key=lambda x: x[0])[-1]
    dest = dest_dir / name
    if not dest.exists():
        print(f"  Downloading {name}...")
        with urllib.request.urlopen(url) as resp, dest.open("wb") as f:
            f.write(resp.read())
    return dest


def ingest_zip(
    zip_path: Path,
    client: InfluxDBClient,
    existing_run_times: set,
    regions: list[str],
    dry_run: bool = False,
) -> tuple[int, str]:
    """Parse and write one ZIP. Returns (points_written, status_str)."""
    run_time_utc = run_time_from_filename(zip_path.name)
    if run_time_utc is None:
        return 0, "skipped (filename not recognised)"
    run_time_iso = run_time_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    if run_time_iso in existing_run_times:
        return 0, "already present"

    points = parse_zip(zip_path, run_time_utc, regions)
    if not points:
        return 0, "no matching rows"

    if not dry_run:
        client.write_points(
            points,
            time_precision="s",
            retention_policy=RETENTION_POLICY,
            batch_size=5000,
        )
        existing_run_times.add(run_time_iso)

    return len(points), "written" if not dry_run else "dry-run"


def main():
    parser = argparse.ArgumentParser(description="Ingest AEMO PD7Day forecasts into InfluxDB")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill", metavar="DIR", help="Process all ZIPs in DIR")
    mode.add_argument("--fetch", action="store_true", help="Download + ingest latest from NEMweb")
    mode.add_argument("--mmsdm", metavar="ZIP",
                      help="Ingest one AEMO MMSDM PD7DAY_PRICESOLUTION monthly zip "
                           "(cumulative — contains full history)")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=DEFAULT_REGIONS,
        metavar="REGION",
        help=f"Regions to ingest (default: {' '.join(DEFAULT_REGIONS)})",
    )
    parser.add_argument("--all-regions", action="store_true", help="Ingest all 5 NEM regions")
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write to InfluxDB")
    args = parser.parse_args()

    regions = [] if args.all_regions else args.regions

    cfg = load_config(args.config)
    client = influx_client(cfg)

    print(f"Connected to InfluxDB: {cfg['influxdb']['host']}:{cfg['influxdb'].get('port', 8086)}")
    print(f"Measurement: {MEASUREMENT}, retention policy: {RETENTION_POLICY}")
    print(f"Regions: {regions if regions else 'ALL'}")

    existing = fetch_existing_run_times(client)
    print(f"Found {len(existing)} existing run_times in InfluxDB")

    if args.backfill:
        backfill_dir = Path(args.backfill)
        zips = sorted(backfill_dir.glob("PUBLIC_PD7DAY_*.zip")) + sorted(
            backfill_dir.glob("PUBLIC_PD7DAY_*.ZIP")
        )
        zips = sorted(set(zips), key=lambda p: p.name)
        print(f"\nBackfill: {len(zips)} ZIP files in {backfill_dir}")
        total_written = 0
        for i, zip_path in enumerate(zips, 1):
            n, status = ingest_zip(zip_path, client, existing, regions, dry_run=args.dry_run)
            total_written += n
            if status != "already present":
                print(f"  [{i:3d}/{len(zips)}] {zip_path.name}: {n} points — {status}")
        print(f"\nDone. Total points written: {total_written}")

    elif args.fetch:
        fetch_dir = Path("scratch/pd7day_latest")
        fetch_dir.mkdir(parents=True, exist_ok=True)
        zip_path = fetch_latest_zip(fetch_dir)
        n, status = ingest_zip(zip_path, client, existing, regions, dry_run=args.dry_run)
        print(f"{zip_path.name}: {n} points — {status}")
        zip_path.unlink(missing_ok=True)

    elif args.mmsdm:
        zip_path = Path(args.mmsdm)
        if not zip_path.exists():
            print(f"ERROR: {zip_path} not found")
            sys.exit(1)
        print(f"\nMMSDM ingest: {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")
        n, n_runs = ingest_mmsdm(zip_path, client, existing, regions,
                                  dry_run=args.dry_run)
        print(f"\nDone. {n:,} points written across {n_runs:,} new run_times "
              f"({'dry-run' if args.dry_run else 'committed'})")


if __name__ == "__main__":
    main()
