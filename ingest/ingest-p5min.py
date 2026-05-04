#!/usr/bin/env python3
"""
Ingest AEMO P5MIN (5-minute predispatch) forecasts into InfluxDB.

P5MIN runs every 5 minutes and provides 12 × 5-min interval forecasts (~60 min
ahead). This is the near-term signal for the 0–1h forecast window, and will
eventually replace Amber APF's near-term debiased signal (Step 5).

NEMweb: https://nemweb.com.au/Reports/Current/P5_MIN_PREDISPATCH/
Files:  PUBLIC_P5MIN_YYYYMMDDHHMM_YYYYMMDDHHMMSS.zip (first ts = run_time AEST)
Table:  P5MIN_REGIONSOLUTION (columns parsed from "I" header row — robust to
        schema version changes)

InfluxDB schema:
  measurement:      aemo_p5min_forecast
  retention_policy: rp_5m
  time:    interval_datetime (UTC)
  tags:    region (SA1, VIC1, etc.)
           run_time (ISO UTC string)
  fields:  rrp ($/MWh), total_demand (MW), net_interchange (MW, if present)

⚠️  Check that rp_5m has a long enough retention duration to accumulate bias-
    correction training history (target: 12+ months). If rp_5m was created
    for intermediate CQ data it may have a short TTL — create a dedicated
    rp_p5min policy or extend rp_5m in InfluxDB if needed.

Idempotent: run_times already present in InfluxDB are skipped.

Usage:
    python ingest/ingest-p5min.py --fetch                     # latest file
    python ingest/ingest-p5min.py --fetch --dry-run           # no write
    python ingest/ingest-p5min.py --fetch --regions SA1 VIC1  # subset
"""

import argparse
import csv
import io
import json
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

CURRENT_URL  = "https://nemweb.com.au/Reports/CURRENT/P5_Reports/"
TABLE_PACKAGE = "P5MIN"
TABLE_NAME    = "REGIONSOLUTION"    # row[1]=P5MIN, row[2]=REGIONSOLUTION in "I"/"D" rows
MEASUREMENT  = "aemo_p5min_forecast"
RETENTION_POLICY = "rp_5m"

# PUBLIC_P5MIN_YYYYMMDDHHMM_YYYYMMDDHHMMSS.zip
CURRENT_FILE_RE = re.compile(
    r"PUBLIC_P5MIN_(\d{12})_\d{14}\.zip$", re.IGNORECASE
)

NEM_TZ = pytz.timezone("Australia/Brisbane")
DEFAULT_REGIONS = {"SA1", "VIC1", "NSW1"}

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config_utils import load_config


def influx_client(cfg):
    ic = cfg["influxdb"]
    return InfluxDBClient(
        host=ic["host"], port=ic.get("port", 8086),
        username=ic["username"], password=ic["password"],
        database=ic["database"],
    )


def run_time_from_filename(name):
    """Extract run_time (UTC) from PUBLIC_P5MIN_YYYYMMDDHHMM_... filename."""
    m = CURRENT_FILE_RE.search(name)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y%m%d%H%M")
    return NEM_TZ.localize(ts).astimezone(timezone.utc)


def to_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_existing_run_times(client):
    try:
        result = client.query(
            f'SHOW TAG VALUES FROM "{MEASUREMENT}" WITH KEY = "run_time"',
            database=client._database,
        )
        return {v["value"] for v in result.get_points()}
    except Exception:
        return set()


class LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href", "")
            if href:
                self.links.append(href)


def list_current_files(url):
    with urllib.request.urlopen(url) as r:
        html = r.read().decode("utf-8", errors="ignore")
    parser = LinkExtractor()
    parser.feed(html)
    files = []
    for href in parser.links:
        name = href.split("/")[-1]
        if CURRENT_FILE_RE.search(name):
            files.append((name, urljoin(url, href)))
    return sorted(files)


def parse_p5min_csv(csv_bytes, regions, run_time_utc):
    """Parse P5MIN_REGIONSOLUTION rows. Column positions read from 'I' header.

    Returns list of InfluxDB points, or [] if table not found.
    """
    run_time_iso = to_iso(run_time_utc)
    points = []

    # Build column index map from the 'I' header row for this table
    col_map = None   # dict: column_name (upper) → index within D row values
    table_version = None

    reader = csv.reader(
        line.decode("utf-8", errors="ignore")
        for line in io.BytesIO(csv_bytes)
    )

    for row in reader:
        if not row:
            continue

        rec_type = row[0].upper()

        # "I" row: I, <package>, <table>, <version>, col1, col2, ...
        if rec_type == "I" and len(row) >= 5:
            if row[1].upper() == TABLE_PACKAGE and row[2].upper() == TABLE_NAME:
                table_version = row[3]
                # Columns start at index 4; map name → position within the
                # full row (so "D" rows can be indexed identically)
                col_map = {col.upper(): idx for idx, col in enumerate(row)}
            continue

        # "D" rows: only process if we've seen the matching "I" header
        if rec_type != "D":
            continue
        if col_map is None:
            continue
        if len(row) < 5 or row[1].upper() != TABLE_PACKAGE or row[2].upper() != TABLE_NAME:
            continue

        # Extract fields by column name
        try:
            # Skip intervention runs (INTERVENTION=1); keep only standard dispatch
            if "INTERVENTION" in col_map and row[col_map["INTERVENTION"]].strip() != "0":
                continue

            region = row[col_map["REGIONID"]]
            if regions and region not in regions:
                continue

            interval_str = row[col_map["INTERVAL_DATETIME"]]
            rrp_str      = row[col_map["RRP"]]

            interval_naive = datetime.strptime(interval_str, "%Y/%m/%d %H:%M:%S")
            interval_utc   = NEM_TZ.localize(interval_naive).astimezone(timezone.utc)
            rrp = float(rrp_str)

        except (KeyError, ValueError, IndexError):
            continue   # malformed row — skip

        fields = {"rrp": rrp}

        for field, col in (("total_demand", "TOTALDEMAND"),
                           ("net_interchange", "NET_INTERCHANGE")):
            if col in col_map:
                raw = row[col_map[col]]
                if raw:
                    try:
                        fields[field] = float(raw)
                    except ValueError:
                        pass

        points.append({
            "measurement": MEASUREMENT,
            "time": interval_utc,
            "tags": {"region": region, "run_time": run_time_iso},
            "fields": fields,
        })

    return points, col_map is not None


def ingest_zip(zip_bytes, name, existing, regions, client, dry_run):
    """Parse one P5MIN ZIP and write to InfluxDB. Returns (n_written, status)."""
    run_time_utc = run_time_from_filename(name)
    if run_time_utc is None:
        return 0, "unrecognised filename"
    run_time_iso = to_iso(run_time_utc)
    if run_time_iso in existing:
        return 0, "already present"

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.upper().endswith(".CSV")]
        if not csv_names:
            return 0, "no CSV inside"
        csv_bytes = zf.read(sorted(csv_names)[0])

    points, table_found = parse_p5min_csv(csv_bytes, regions, run_time_utc)
    if not table_found:
        return 0, f"table {TABLE_PACKAGE}/{TABLE_NAME} not found in CSV"
    if not points:
        return 0, "no matching region rows"

    if not dry_run:
        client.write_points(points, time_precision="s",
                            retention_policy=RETENTION_POLICY, batch_size=5000)
        existing.add(run_time_iso)

    return len(points), "written" if not dry_run else "dry-run"


def main():
    parser = argparse.ArgumentParser(
        description="Ingest AEMO P5MIN forecasts into InfluxDB"
    )
    parser.add_argument("--fetch", action="store_true", required=True,
                        help="Download + ingest the latest P5MIN file from CURRENT")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--regions", nargs="+", default=sorted(DEFAULT_REGIONS),
                        metavar="REGION",
                        help=f"Regions to ingest (default: {' '.join(sorted(DEFAULT_REGIONS))})")
    parser.add_argument("--all-regions", action="store_true",
                        help="Ingest all NEM regions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    regions = set() if args.all_regions else set(args.regions)

    cfg = load_config(args.config)
    client = influx_client(cfg)

    print(f"Connected to InfluxDB: {cfg['influxdb']['host']}:{cfg['influxdb'].get('port', 8086)}")
    print(f"Measurement: {MEASUREMENT}, rp: {RETENTION_POLICY}")
    print(f"Regions: {sorted(regions) if regions else 'ALL'}")

    existing = fetch_existing_run_times(client)
    print(f"Found {len(existing)} existing run_times in InfluxDB")

    current_files = list_current_files(CURRENT_URL)
    if not current_files:
        print("No P5MIN files found in CURRENT")
        return

    # Try newest first; stop at the first file not already imported
    for name, url in reversed(current_files):
        run_time_utc = run_time_from_filename(name)
        if run_time_utc and to_iso(run_time_utc) in existing:
            continue
        print(f"Fetching {name}...")
        with urllib.request.urlopen(url) as r:
            zip_bytes = r.read()
        n, status = ingest_zip(zip_bytes, name, existing, regions, client, args.dry_run)
        print(f"  {n} points — {status}")
        break
    else:
        print("Latest P5MIN file already present in InfluxDB")


if __name__ == "__main__":
    main()
