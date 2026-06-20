#!/usr/bin/env python3
"""
Ingest AEMO SEVENDAYOUTLOOK_FULL demand forecasts into InfluxDB.

Provides SCHEDULED_DEMAND, SCHEDULED_CAPACITY, NET_INTERCHANGE, SCHEDULED_RESERVE
for all NEM regions at 30-min resolution, 7-day horizon, updated every 30 min.
Use as demand/interchange future covariates for 28h+ TFT decoder inputs.
Note: these are AEMO-produced (not market-strategic), so no debiasing is needed.

Modes:
  --backfill-archive   Stream all weekly archive ZIPs from NEMweb (March 2025+)
  --fetch              Download + ingest the latest file from CURRENT

InfluxDB schema:
  measurement: aemo_sevendayoutlook
  time:   interval_datetime (UTC)
  tags:   region (SA1, VIC1, etc.)
          run_time (ISO UTC string, from filename)
  fields: scheduled_demand (MW), scheduled_capacity (MW),
          net_interchange (MW), scheduled_reserve (MW)

Idempotent: run_times already present in InfluxDB are skipped.
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

# Allow importing aemo_session from project root when invoked via systemd
# (WorkingDirectory is the project root, which is already on sys.path via the
# venv activation, but add it explicitly for direct invocations too)
sys.path.insert(0, str(Path(__file__).parent.parent))
from aemo_session import make_aemo_session

CURRENT_URL = "https://nemweb.com.au/Reports/CURRENT/SEVENDAYOUTLOOK_FULL/"
ARCHIVE_URL = "https://nemweb.com.au/Reports/ARCHIVE/SEVENDAYOUTLOOK_FULL/"
# Current individual files: PUBLIC_SEVENDAYOUTLOOK_FULL_YYYYMMDDHHMMSS_<number>.zip
CURRENT_FILE_RE = re.compile(
    r"PUBLIC_SEVENDAYOUTLOOK_FULL_(\d{14})_\d+\.zip$", re.IGNORECASE
)
# Archive weekly containers: PUBLIC_SEVENDAYOUTLOOK_FULL_YYYYMMDD.zip
ARCHIVE_FILE_RE = re.compile(r"PUBLIC_SEVENDAYOUTLOOK_FULL_\d{8}\.zip$", re.IGNORECASE)

NEM_TZ = pytz.timezone("Australia/Brisbane")
MEASUREMENT = "aemo_sevendayoutlook"
RETENTION_POLICY = "rp_30m"
DEFAULT_REGIONS = {"SA1", "VIC1", "NSW1"}


from config_utils import load_config


def influx_client(cfg):
    ic = cfg["influxdb"]
    return InfluxDBClient(
        host=ic["host"], port=ic.get("port", 8086),
        username=ic["username"], password=ic["password"],
        database=ic["database"],
    )


def run_time_from_filename(name):
    """YYYYMMDDHHMMSS (AEST) → UTC datetime, from individual file name."""
    m = CURRENT_FILE_RE.search(name)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
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


def list_files(url, pattern, session=None):
    if session is not None:
        resp = session.get(url, timeout=30)
        html = resp.text
    else:
        with urllib.request.urlopen(url) as r:
            html = r.read().decode("utf-8", errors="ignore")
    parser = LinkExtractor()
    parser.feed(html)
    files = []
    for href in parser.links:
        name = href.split("/")[-1]
        if pattern.search(name):
            files.append((name, urljoin(url, href)))
    return sorted(files)


def parse_sdoutlook_csv(csv_bytes, regions, run_time_utc):
    """Parse SEVENDAYOUTLOOK/PEAK rows from CSV bytes. Returns InfluxDB points."""
    run_time_iso = to_iso(run_time_utc)
    points = []
    reader = csv.reader(
        line.decode("utf-8", errors="ignore")
        for line in io.BytesIO(csv_bytes)
    )
    for row in reader:
        if not row or row[0] != "D" or len(row) < 11:
            continue
        if row[1] != "SEVENDAYOUTLOOK" or row[2] not in ("PEAK", "REGIONDATA"):
            continue
        region = row[4]
        if regions and region not in regions:
            continue
        interval_str = row[10]   # INTERVAL_DATETIME (AEST)
        sched_demand = float(row[6]) if row[6] else None
        sched_capacity = float(row[7]) if row[7] else None
        net_interchange = float(row[8]) if row[8] else None
        sched_reserve = float(row[9]) if row[9] else None

        interval_naive = datetime.strptime(interval_str, "%Y/%m/%d %H:%M:%S")
        interval_utc = NEM_TZ.localize(interval_naive).astimezone(timezone.utc)

        fields = {}
        if sched_demand is not None:
            fields["scheduled_demand"] = sched_demand
        if sched_capacity is not None:
            fields["scheduled_capacity"] = sched_capacity
        if net_interchange is not None:
            fields["net_interchange"] = net_interchange
        if sched_reserve is not None:
            fields["scheduled_reserve"] = sched_reserve
        if not fields:
            continue

        points.append({
            "measurement": MEASUREMENT,
            "time": interval_utc,
            "tags": {"region": region, "run_time": run_time_iso},
            "fields": fields,
        })
    return points


def ingest_individual_zip(zip_bytes, name, existing, regions, client, dry_run):
    """Parse one individual SEVENDAYOUTLOOK ZIP (bytes). Returns (n_written, status)."""
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
        csv_bytes_content = zf.read(sorted(csv_names)[0])

    points = parse_sdoutlook_csv(csv_bytes_content, regions, run_time_utc)
    if not points:
        return 0, "no matching rows"

    if not dry_run:
        client.write_points(points, time_precision="s",
                            retention_policy=RETENTION_POLICY, batch_size=5000)
        existing.add(run_time_iso)

    return len(points), "written" if not dry_run else "dry-run"


def process_weekly_archive_zip(url, name, existing, regions, client, dry_run):
    """Download a weekly archive ZIP and process all inner SEVENDAYOUTLOOK ZIPs."""
    print(f"  Downloading {name}...")
    with urllib.request.urlopen(url) as r:
        weekly_bytes = r.read()
    print(f"    {len(weekly_bytes)//1024//1024}MB — processing inner files...")

    total_written = 0
    skipped = 0
    with zipfile.ZipFile(io.BytesIO(weekly_bytes)) as outer:
        inner_names = sorted(
            n for n in outer.namelist()
            if CURRENT_FILE_RE.search(n.split("/")[-1])
        )
        for inner_name in inner_names:
            base = inner_name.split("/")[-1]
            # Inner ZIPs contain ZIPs (doubly nested)
            inner_zip_bytes = outer.read(inner_name)
            if base.upper().endswith(".ZIP"):
                n, status = ingest_individual_zip(
                    inner_zip_bytes, base, existing, regions, client, dry_run
                )
            else:
                # Might be a direct CSV in some archive versions
                run_time_utc = run_time_from_filename(base)
                if run_time_utc is None:
                    continue
                run_time_iso = to_iso(run_time_utc)
                if run_time_iso in existing:
                    n, status = 0, "already present"
                else:
                    points = parse_sdoutlook_csv(inner_zip_bytes, regions, run_time_utc)
                    if points and not dry_run:
                        client.write_points(points, time_precision="s",
                                            retention_policy=RETENTION_POLICY, batch_size=5000)
                        existing.add(run_time_iso)
                    n = len(points)
                    status = "written" if (points and not dry_run) else ("dry-run" if points else "no matching rows")

            total_written += n
            if status == "already present":
                skipped += 1

    return total_written, skipped, len(inner_names)


def main():
    parser = argparse.ArgumentParser(
        description="Ingest AEMO SEVENDAYOUTLOOK_FULL demand forecasts into InfluxDB"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill-archive", action="store_true",
                      help="Stream all weekly archive ZIPs from NEMweb")
    mode.add_argument("--fetch", action="store_true",
                      help="Download + ingest the latest from CURRENT")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--regions", nargs="+", default=sorted(DEFAULT_REGIONS),
                        metavar="REGION")
    parser.add_argument("--all-regions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    regions = set() if args.all_regions else set(args.regions)

    cfg = load_config(args.config)
    client = influx_client(cfg)

    print(f"Connected to InfluxDB: {cfg['influxdb']['host']}:{cfg['influxdb'].get('port', 8086)}")
    print(f"Measurement: {MEASUREMENT}, retention policy: {RETENTION_POLICY}")
    print(f"Regions: {sorted(regions) if regions else 'ALL'}")

    existing = fetch_existing_run_times(client)
    print(f"Found {len(existing)} existing run_times in InfluxDB")

    if args.backfill_archive:
        weekly_files = list_files(ARCHIVE_URL, ARCHIVE_FILE_RE)
        print(f"\nArchive backfill: {len(weekly_files)} weekly ZIPs")
        grand_total = 0
        for i, (name, url) in enumerate(weekly_files, 1):
            print(f"\n[{i}/{len(weekly_files)}] {name}")
            total, skipped, inner_count = process_weekly_archive_zip(
                url, name, existing, regions, client, args.dry_run
            )
            grand_total += total
            action = "dry-run" if args.dry_run else "written"
            print(f"    {inner_count} inner files: {total} points {action}, {skipped} already present")
        print(f"\nDone. Grand total points written: {grand_total}")

    elif args.fetch:
        session = make_aemo_session()
        current_files = list_files(CURRENT_URL, CURRENT_FILE_RE, session=session)
        if not current_files:
            print("No SEVENDAYOUTLOOK files found in CURRENT")
            return
        for name, url in reversed(current_files):
            run_time_utc = run_time_from_filename(name)
            if run_time_utc and to_iso(run_time_utc) in existing:
                continue
            print(f"Fetching {name}...")
            zip_bytes = session.get(url, timeout=60).content
            n, status = ingest_individual_zip(zip_bytes, name, existing, regions, client, args.dry_run)
            print(f"{name}: {n} points — {status}")
            break
        else:
            print("Latest file already present in InfluxDB")


if __name__ == "__main__":
    main()
