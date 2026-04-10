#!/usr/bin/env python3
"""
Ingest AEMO PD7Day price forecasts into InfluxDB.

Modes:
  --backfill DIR   Process all PD7Day ZIP files in DIR (e.g. scratch/pd7day_backfill/)
  --fetch          Download the latest PD7Day ZIP from NEMweb and ingest it

InfluxDB schema:
  measurement: aemo_pd7day_forecast
  time:   interval_datetime (UTC)
  tags:   region (SA1, VIC1, etc.)
          run_time (ISO UTC string, from filename)
  fields: rrp ($/MWh)

Idempotent: run_times already present in InfluxDB are skipped.
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


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path) as f:
        return json.load(f)


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


if __name__ == "__main__":
    main()
