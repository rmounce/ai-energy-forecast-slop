#!/usr/bin/env python3
"""
Ingest AEMO P30 PREDISPATCH forecasts into InfluxDB.

Modes:
  --backfill-archive   Stream all weekly archive ZIPs from NEMweb (March 2025+)
  --fetch              Download + ingest the latest file from CURRENT

InfluxDB schema:
  measurement: aemo_predispatch_forecast
  time:   interval_datetime (UTC)
  tags:   region (SA1, VIC1, etc.)
          run_time (ISO UTC string, from filename first timestamp)
  fields: rrp ($/MWh), total_demand (MW), net_interchange (MW)

Idempotent: run_times already present in InfluxDB are skipped.
"""

import argparse
import csv
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin

import pytz
from influxdb import InfluxDBClient

CURRENT_URL = "https://nemweb.com.au/Reports/CURRENT/Predispatch_Reports/"
ARCHIVE_URL = "https://nemweb.com.au/Reports/ARCHIVE/Predispatch_Reports/"
# Current individual files: PUBLIC_PREDISPATCH_YYYYMMDDHHMM_YYYYMMDDHHMMSS_LEGACY.zip
CURRENT_FILE_RE = re.compile(r"PUBLIC_PREDISPATCH_(\d{12})_\d{14}_LEGACY\.zip$", re.IGNORECASE)
# Archive weekly containers: PUBLIC_PREDISPATCH_YYYYMMDD_YYYYMMDD.zip
ARCHIVE_FILE_RE = re.compile(r"PUBLIC_PREDISPATCH_\d{8}_\d{8}\.zip$", re.IGNORECASE)

NEM_TZ = pytz.timezone("Australia/Brisbane")
MEASUREMENT = "aemo_predispatch_forecast"
RETENTION_POLICY = "rp_30m"
DEFAULT_REGIONS = {"SA1", "VIC1", "NSW1"}


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


def run_time_from_current_filename(name):
    """YYYYMMDDHHMM (AEST) → UTC datetime."""
    m = CURRENT_FILE_RE.search(name)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y%m%d%H%M")
    return NEM_TZ.localize(ts).astimezone(timezone.utc)


def run_time_from_inner_filename(name):
    """Same format as current files."""
    return run_time_from_current_filename(name)


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


def list_files(url, pattern):
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


def parse_pdregion_csv(csv_bytes, regions, run_time_utc):
    """Parse PDREGION rows from CSV bytes. Returns list of InfluxDB points."""
    run_time_iso = to_iso(run_time_utc)
    points = []
    reader = csv.reader(
        line.decode("utf-8", errors="ignore")
        for line in io.BytesIO(csv_bytes)
    )
    for row in reader:
        if not row or row[0] != "D" or len(row) < 14:
            continue
        if row[1] != "PDREGION":
            continue
        region = row[6]
        if regions and region not in regions:
            continue
        interval_str = row[7]   # PERIODID = interval_datetime AEST
        rrp = float(row[8])
        total_demand = float(row[10]) if row[10] else None
        net_interchange = float(row[13]) if row[13] else None

        interval_naive = datetime.strptime(interval_str, "%Y/%m/%d %H:%M:%S")
        interval_utc = NEM_TZ.localize(interval_naive).astimezone(timezone.utc)

        pt = {
            "measurement": MEASUREMENT,
            "time": interval_utc,
            "tags": {"region": region, "run_time": run_time_iso},
            "fields": {"rrp": rrp},
        }
        if total_demand is not None:
            pt["fields"]["total_demand"] = total_demand
        if net_interchange is not None:
            pt["fields"]["net_interchange"] = net_interchange
        points.append(pt)
    return points


def ingest_zip_bytes(zip_bytes, name, existing, regions, client, dry_run):
    """Parse one individual PREDISPATCH ZIP (bytes). Returns (n_written, status)."""
    run_time_utc = run_time_from_current_filename(name)
    if run_time_utc is None:
        run_time_utc = run_time_from_inner_filename(name)
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

    points = parse_pdregion_csv(csv_bytes, regions, run_time_utc)
    if not points:
        return 0, "no matching rows"

    if not dry_run:
        client.write_points(points, time_precision="s",
                            retention_policy=RETENTION_POLICY, batch_size=5000)
        existing.add(run_time_iso)

    return len(points), "written" if not dry_run else "dry-run"


def process_weekly_archive_zip(url, name, existing, regions, client, dry_run):
    """Download a weekly archive ZIP and process all inner PREDISPATCH ZIPs."""
    print(f"  Downloading {name} ({url})...")
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
            inner_bytes = outer.read(inner_name)
            n, status = ingest_zip_bytes(inner_bytes, base, existing, regions, client, dry_run)
            total_written += n
            if status == "already present":
                skipped += 1
            elif status != "no matching rows":
                pass  # progress shown at week level

    return total_written, skipped, len(inner_names)


def main():
    parser = argparse.ArgumentParser(description="Ingest AEMO PREDISPATCH forecasts into InfluxDB")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--backfill-archive", action="store_true",
                      help="Stream all weekly archive ZIPs from NEMweb")
    mode.add_argument("--fetch", action="store_true",
                      help="Download + ingest the latest from CURRENT")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--regions", nargs="+", default=sorted(DEFAULT_REGIONS),
                        metavar="REGION", help=f"Regions to ingest (default: {' '.join(sorted(DEFAULT_REGIONS))})")
    parser.add_argument("--all-regions", action="store_true", help="Ingest all 5 NEM regions")
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
        current_files = list_files(CURRENT_URL, CURRENT_FILE_RE)
        if not current_files:
            print("No PREDISPATCH files found in CURRENT")
            return
        # Try from newest until we find one not already imported
        for name, url in reversed(current_files):
            run_time_utc = run_time_from_current_filename(name)
            if run_time_utc and to_iso(run_time_utc) in existing:
                continue
            print(f"Fetching {name}...")
            with urllib.request.urlopen(url) as r:
                zip_bytes = r.read()
            n, status = ingest_zip_bytes(zip_bytes, name, existing, regions, client, args.dry_run)
            print(f"{name}: {n} points — {status}")
            break
        else:
            print("Latest file already present in InfluxDB")


if __name__ == "__main__":
    main()
