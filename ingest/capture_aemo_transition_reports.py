#!/usr/bin/env python3
"""Archive as-issued AEMO reports and the production 5MIN JSON response.

Raw files are written byte-for-byte below ``data/aemo_transition/raw``. A JSON
sidecar records capture time, source URL, report run time, response headers, and
SHA-256. The canary stores observed CSV schemas and fails when a known schema
changes or required regional/interconnector data disappears.
"""

from __future__ import annotations

import csv
import email.utils
import hashlib
import io
import json
import logging
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from healthchecks import record_job_status  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "aemo_transition"
DEFAULT_CAPTURE_FROM = "2026-09-22T00:00:00+00:00"
NEM_TZ = timezone(timedelta(hours=10))
REQUIRED_REGIONS = {"SA1", "VIC1", "NSW1"}
FLOW_COLUMNS = {"MWFLOW", "METEREDMWFLOW", "TOTALCLEARED"}
TARGET_COLUMNS = {"INTERVAL_DATETIME", "DATETIME", "DISPATCHINTERVAL", "SETTLEMENTDATE", "PERIODID"}
NET_INTERCHANGE_COLUMNS = {"NETINTERCHANGE", "NET_INTERCHANGE"}
MAX_STITCH_GAP_HOURS = 30
MAX_STITCH_JUMP_MW = 3000.0
MAX_HORIZON_CHANGE_HOURS = 0.5


@dataclass(frozen=True)
class ReportSource:
    key: str
    url: str
    filename_re: re.Pattern[str]
    max_age_minutes: int
    region_tables: tuple[str, ...]
    interconnector_tables: tuple[str, ...] = ()
    requires_net_interchange: bool = True
    cadence_minutes: int = 0


def _pattern(value: str) -> re.Pattern[str]:
    return re.compile(value, re.IGNORECASE)


SOURCES = (
    ReportSource("dispatchis", "https://nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/",
                 _pattern(r"PUBLIC_DISPATCHIS_(\d{12})_\d+\.zip$"), 20,
                 ("REGIONSUM", "DISPATCHREGIONSUM"), ("INTERCONNECTORRES", "DISPATCHINTERCONNECTORRES"),
                 cadence_minutes=5),
    ReportSource("dispatch_legacy", "https://nemweb.com.au/Reports/CURRENT/Dispatch_Reports/",
                 _pattern(r"PUBLIC_DISPATCH_(\d{12})_\d{14}_LEGACY\.zip$"), 25,
                 ("DREGION",), ("DINT",), cadence_minutes=5),
    ReportSource("p5min", "https://nemweb.com.au/Reports/CURRENT/P5_Reports/",
                 _pattern(r"PUBLIC_P5MIN_(\d{12})_\d{14}\.zip$"), 20,
                 ("REGIONSOLUTION",), ("INTERCONNECTORSOLN", "P5MIN_INTERCONNECTORSOLN"), cadence_minutes=5),
    ReportSource("predispatchis", "https://nemweb.com.au/Reports/CURRENT/PredispatchIS_Reports/",
                 _pattern(r"PUBLIC_PREDISPATCHIS_(\d{12})_\d{14}\.zip$"), 90,
                 ("REGION_SOLUTION", "PREDISPATCHREGIONSUM"),
                 ("INTERCONNECTOR_SOLN", "PREDISPATCHINTERCONNECTORRES"), cadence_minutes=30),
    ReportSource("predispatch_legacy", "https://nemweb.com.au/Reports/CURRENT/Predispatch_Reports/",
                 _pattern(r"PUBLIC_PREDISPATCH_(\d{12})_\d{14}_LEGACY\.zip$"), 90,
                 ("PDREGION",), cadence_minutes=30),
    ReportSource("stpasa", "https://nemweb.com.au/Reports/CURRENT/Short_Term_PASA_Reports/",
                 _pattern(r"PUBLIC_STPASA_(\d{12})_\d+\.zip$"), 180,
                 ("REGIONSOLUTION",), requires_net_interchange=False, cadence_minutes=60),
    ReportSource("sevendayoutlook", "https://nemweb.com.au/Reports/CURRENT/SEVENDAYOUTLOOK_FULL/",
                 _pattern(r"PUBLIC_SEVENDAYOUTLOOK_FULL_(\d{14})_\d+\.zip$"), 90,
                 ("PEAK", "REGIONDATA"), cadence_minutes=30),
)

API_URL = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
API_PAYLOAD = {"timeScale": ["30MIN"]}


class HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def _utc_iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_time_from_name(source: ReportSource, filename: str) -> str | None:
    match = source.filename_re.search(filename)
    if not match:
        return None
    raw = match.group(1)
    fmt = "%Y%m%d%H%M%S" if len(raw) == 14 else "%Y%m%d%H%M"
    return _utc_iso(datetime.strptime(raw, fmt).replace(tzinfo=NEM_TZ))


def _cadence_diagnostics(source: ReportSource, filenames: Iterable[str]) -> dict | None:
    if source.cadence_minutes <= 0:
        return None
    run_times = sorted({
        datetime.fromisoformat(run_time.replace("Z", "+00:00"))
        for filename in filenames
        if (run_time := _run_time_from_name(source, filename)) is not None
    })
    if len(run_times) < 2:
        return {
            "expected_minutes": source.cadence_minutes,
            "sample_count": len(run_times),
            "median_gap_minutes": None,
            "latest_gap_minutes": None,
        }
    sample = run_times[-13:]
    gaps = [
        (right - left).total_seconds() / 60
        for left, right in zip(sample, sample[1:])
    ]
    return {
        "expected_minutes": source.cadence_minutes,
        "sample_count": min(len(run_times), 13),
        "median_gap_minutes": round(statistics.median(gaps), 1),
        "latest_gap_minutes": round(gaps[-1], 1),
    }


def _record_horizon(source_key: str, horizon_hours: float | None, state: dict,
                    issues: list[str]) -> None:
    if horizon_hours is None:
        return
    previous = state.setdefault("horizons_hours", {}).get(source_key)
    if previous is not None and abs(horizon_hours - previous) > MAX_HORIZON_CHANGE_HOURS:
        issues.append(
            f"{source_key}: forecast horizon changed from {previous:.2f} to {horizon_hours:.2f} hours"
        )
    state["horizons_hours"][source_key] = round(horizon_hours, 2)


def _max_report_horizon_hours(run_time_utc: str, tables: list[dict]) -> float | None:
    targets = [
        datetime.fromisoformat(table["target_max"].replace("Z", "+00:00"))
        for table in tables if table.get("target_max")
    ]
    if not targets:
        return None
    run_time = datetime.fromisoformat(run_time_utc.replace("Z", "+00:00"))
    return round((max(targets) - run_time).total_seconds() / 3600, 2)


def _request(url: str, *, data: bytes | None = None, content_type: str | None = None) -> tuple[bytes, dict[str, str]]:
    headers = {"User-Agent": "Mozilla/5.0"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read(), {k.lower(): v for k, v in response.headers.items()}
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after 3 attempts: {url}: {last_error}")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_bytes(content)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _save_capture(root: Path, category: str, filename: str, content: bytes,
                  source_url: str, headers: dict[str, str], run_time_utc: str | None,
                  request_payload: dict | None = None) -> tuple[Path, bool]:
    data_path = root / "raw" / category / filename
    meta_path = data_path.with_name(data_path.name + ".meta.json")
    if data_path.is_file() and meta_path.is_file():
        return data_path, False
    meta = {
        "source_url": source_url,
        "filename": filename,
        "captured_utc": _utc_iso(),
        "run_time_utc": run_time_utc,
        "source_last_modified": headers.get("last-modified"),
        "source_last_modified_utc": _modified_utc(headers.get("last-modified")),
        "http_date": headers.get("date"),
        "etag": headers.get("etag"),
        "content_type": headers.get("content-type"),
        "content_length_header": headers.get("content-length"),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "request_payload": request_payload,
    }
    _atomic_write(data_path, content)
    _atomic_write(meta_path, (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode())
    return data_path, True


def _modified_utc(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return _utc_iso(parsed)
    except (TypeError, ValueError, OverflowError):
        return None


def _table_name(value: str) -> str:
    value = value.strip().strip('"').upper()
    if value.startswith("P5MIN_"):
        return value.removeprefix("P5MIN_")
    return value


def inspect_aemo_zip(content: bytes) -> dict:
    """Inspect I/D records without changing or rewriting the retained ZIP."""
    tables: dict[str, dict] = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("report ZIP has no CSV member")
        for csv_name in sorted(csv_names):
            with archive.open(csv_name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                reader = csv.reader(text)
                headers: dict[tuple[str, str], tuple[str, list[str]]] = {}
                for row in reader:
                    if len(row) < 3:
                        continue
                    record_type = row[0].strip().upper()
                    package = row[1].strip().strip('"').upper()
                    table = _table_name(row[2])
                    key = (package, table)
                    if record_type == "I" and len(row) >= 5:
                        version = row[3].strip().strip('"')
                        columns = [col.strip().strip('"').upper() for col in row[4:]]
                        headers[key] = (version, columns)
                        item = tables.setdefault(f"{package}/{table}", {
                            "package": package, "table": table, "version": version,
                            "columns": columns, "row_count": 0, "regions": set(),
                            "interconnector_ids": set(), "target_min": None,
                            "target_max": None, "exact_duplicate_rows": 0,
                            "duplicate_keys": 0, "field_width_mismatches": 0,
                            "_row_signatures": set(), "_primary_keys": set(),
                            "_regional_net_interchange": [],
                        })
                        item["version"] = version
                        item["columns"] = columns
                    elif record_type == "D":
                        header = headers.get(key)
                        values = row[4:]
                        column_names = header[1] if header else []
                        table_key = f"{package}/{table}"
                        item = tables.setdefault(table_key, {
                            "package": package, "table": table,
                            "version": header[0] if header else None,
                            "columns": column_names, "row_count": 0,
                            "regions": set(), "interconnector_ids": set(),
                            "target_min": None, "target_max": None,
                            "exact_duplicate_rows": 0, "duplicate_keys": 0,
                            "field_width_mismatches": 0,
                            "_row_signatures": set(), "_primary_keys": set(),
                            "_regional_net_interchange": [],
                        })
                        item["row_count"] += 1
                        signature = tuple(row)
                        if signature in item["_row_signatures"]:
                            item["exact_duplicate_rows"] += 1
                        item["_row_signatures"].add(signature)
                        mapped = dict(zip(column_names, values))
                        if header and len(values) != len(column_names):
                            item["field_width_mismatches"] += 1
                        region = mapped.get("REGIONID", "").strip().upper()
                        if not region and not column_names:
                            region = next((cell.strip().upper() for cell in row if cell.strip().upper() in REQUIRED_REGIONS), "")
                        if region in REQUIRED_REGIONS:
                            item["regions"].add(region)
                        if "INTERCONNECTORID" in mapped:
                            interconnector_id = mapped["INTERCONNECTORID"].strip().upper()
                            if interconnector_id:
                                item["interconnector_ids"].add(interconnector_id)
                        for col in TARGET_COLUMNS:
                            value = mapped.get(col)
                            if value:
                                parsed = _parse_nem_time(value)
                                if parsed:
                                    iso = _utc_iso(parsed)
                                    item["target_min"] = min(item["target_min"], iso) if item["target_min"] else iso
                                    item["target_max"] = max(item["target_max"], iso) if item["target_max"] else iso
                                    interchange_col = next((name for name in NET_INTERCHANGE_COLUMNS if mapped.get(name)), None)
                                    if region in REQUIRED_REGIONS and interchange_col:
                                        try:
                                            item["_regional_net_interchange"].append({
                                                "target_utc": _utc_iso(parsed - timedelta(minutes=30)),
                                                "region": region,
                                                "net_interchange_mw": float(mapped[interchange_col]),
                                            })
                                        except ValueError:
                                            pass
                                    break
                        primary_key_fields = _primary_key_fields(column_names)
                        if primary_key_fields:
                            key_values = tuple(mapped.get(col, "") for col in primary_key_fields)
                            if key_values in item["_primary_keys"]:
                                item["duplicate_keys"] += 1
                            item["_primary_keys"].add(key_values)
        for item in tables.values():
            item["regions"] = sorted(item["regions"])
            item["interconnector_ids"] = sorted(item["interconnector_ids"])
            item["regional_net_interchange"] = item.pop("_regional_net_interchange")
            item.pop("_row_signatures", None)
            item.pop("_primary_keys", None)
    return {"tables": list(tables.values())}


def _parse_nem_time(value: str) -> datetime | None:
    raw = value.strip().strip('"')
    if "T" in raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=NEM_TZ) if parsed.tzinfo is None else parsed
        except ValueError:
            pass
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=NEM_TZ)
        except ValueError:
            pass
    return None


def _primary_key_fields(columns: Iterable[str]) -> tuple[str, ...]:
    available = _canonical_cols(columns)
    candidates = (
        ("DISPATCHINTERVAL", "INTERCONNECTORID", "INTERVENTION", "RUNNO"),
        ("RUN_DATETIME", "INTERVAL_DATETIME", "INTERCONNECTORID", "RUNTYPE", "STUDYREGIONID"),
        ("INTERVAL_DATETIME", "INTERCONNECTORID", "RUN_DATETIME"),
        ("DATETIME", "INTERCONNECTORID", "PREDISPATCHSEQNO"),
        ("PREDISPATCHSEQNO", "RUNNO", "INTERCONNECTORID", "PERIODID", "INTERVENTION"),
        ("PREDISPATCHSEQNO", "RUNNO", "REGIONID", "PERIODID", "INTERVENTION"),
        ("INTERVAL_DATETIME", "REGIONID", "RUN_DATETIME"),
        ("DATETIME", "REGIONID", "PREDISPATCHSEQNO"),
        ("INTERVAL_DATETIME", "REGIONID"),
        ("DATETIME", "REGIONID"),
    )
    for candidate in candidates:
        if set(candidate) <= available:
            return candidate
    return ()


def _canonical_cols(columns: Iterable[str]) -> set[str]:
    return {column.strip().strip('"').upper() for column in columns}


def validate_report(source: ReportSource, inspection: dict) -> list[str]:
    tables = inspection["tables"]
    issues: list[str] = []
    found = {}
    for table in tables:
        found[_table_name(table["table"])] = table
        if table["package"]:
            found.setdefault(_table_name(table["package"]), table)
    region_table = next((found[name] for name in source.region_tables if name in found), None)
    if region_table is None:
        issues.append(f"missing regional table; expected one of {source.region_tables}")
    else:
        cols = _canonical_cols(region_table["columns"])
        regions = set(region_table["regions"])
        if region_table["columns"] and "REGIONID" not in cols:
            issues.append(f"{region_table['table']} missing REGIONID column")
        if region_table["columns"] and not (TARGET_COLUMNS & cols):
            issues.append(f"{region_table['table']} missing interval timestamp column")
        if source.requires_net_interchange and region_table["columns"] and not (NET_INTERCHANGE_COLUMNS & cols):
            issues.append(f"{region_table['table']} missing net-interchange column")
        missing_regions = REQUIRED_REGIONS - regions
        if missing_regions:
            issues.append(f"{region_table['table']} missing region rows: {sorted(missing_regions)}")
        if region_table["duplicate_keys"]:
            issues.append(f"{region_table['table']} has {region_table['duplicate_keys']} duplicate keys")
        if region_table["field_width_mismatches"]:
            issues.append(f"{region_table['table']} has {region_table['field_width_mismatches']} row-width mismatches")

    interconnector_tables = []
    for expected in source.interconnector_tables:
        table = found.get(_table_name(expected))
        if table is not None and table not in interconnector_tables:
            interconnector_tables.append(table)
    if source.interconnector_tables and not interconnector_tables:
        issues.append(f"missing interconnector table; expected one of {source.interconnector_tables}")
    for table in interconnector_tables:
        cols = _canonical_cols(table["columns"])
        if "INTERCONNECTORID" not in cols:
            issues.append(f"{table['table']} missing INTERCONNECTORID column")
        if not (FLOW_COLUMNS & cols):
            issues.append(f"{table['table']} missing interconnector flow column")
        if table["duplicate_keys"]:
            issues.append(f"{table['table']} has {table['duplicate_keys']} duplicate keys")
        if table["field_width_mismatches"]:
            issues.append(f"{table['table']} has {table['field_width_mismatches']} row-width mismatches")
    return issues


def _list_source_files(source: ReportSource) -> list[tuple[str, str]]:
    body, _ = _request(source.url)
    parser = HrefParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    found = {}
    for href in parser.hrefs:
        filename = href.rstrip("/").rsplit("/", 1)[-1]
        if source.filename_re.search(filename):
            found[filename] = urljoin(source.url, href)
    return sorted(found.items(), key=lambda item: _run_time_from_name(source, item[0]) or "")


def _capture_json_api(root: Path) -> tuple[dict, bytes]:
    body = json.dumps(API_PAYLOAD, separators=(",", ":")).encode()
    response, headers = _request(API_URL, data=body, content_type="application/json")
    captured = datetime.now(timezone.utc)
    digest = hashlib.sha256(response).hexdigest()
    name = f"{captured.strftime('%Y%m%dT%H%M%SZ')}-{digest[:12]}.json"
    path, saved = _save_capture(root, "visualisations_5min", name, response, API_URL,
                                headers, None, API_PAYLOAD)
    data = json.loads(response)
    records = data.get("5MIN")
    if not isinstance(records, list) or not records:
        raise ValueError("5MIN API response has no rows")
    required_columns = {"SETTLEMENTDATE", "REGIONID", "TOTALDEMAND", "NETINTERCHANGE"}
    missing = required_columns - set(records[0])
    if missing:
        raise ValueError(f"5MIN API response missing fields: {sorted(missing)}")
    regions = {str(row.get("REGIONID", "")).upper() for row in records}
    missing_regions = REQUIRED_REGIONS - regions
    if missing_regions:
        raise ValueError(f"5MIN API response missing regions: {sorted(missing_regions)}")
    targets = [
        parsed - timedelta(minutes=30)
        for row in records
        if row.get("SETTLEMENTDATE")
        for parsed in [_parse_nem_time(str(row["SETTLEMENTDATE"]))]
        if parsed
    ]
    valid_targets = [dt for dt in targets if dt]
    if not valid_targets:
        raise ValueError("5MIN API response has no parseable target timestamps")
    keys = [(str(row.get("REGIONID", "")).upper(), str(row.get("SETTLEMENTDATE", ""))) for row in records]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        raise ValueError(f"5MIN API response has {duplicates} duplicate region/interval keys")
    summary = {
        "source": "visualisations_5min", "filename": path.name, "saved": saved,
        "rows": len(records), "regions": sorted(regions),
        "target_min": _utc_iso(min(valid_targets)), "target_max": _utc_iso(max(valid_targets)),
        "max_horizon_hours": round((max(valid_targets) - captured).total_seconds() / 3600, 2),
        "captured_utc": _utc_iso(captured), "sha256": digest,
        "interval_count_by_region": {
            region: sum(1 for row in records if str(row.get("REGIONID", "")).upper() == region)
            for region in sorted(regions)
        },
        "regional_net_interchange": [
            {
                "target_utc": _utc_iso(_parse_nem_time(str(row["SETTLEMENTDATE"])) - timedelta(minutes=30)),
                "region": str(row["REGIONID"]).upper(),
                "net_interchange_mw": float(row["NETINTERCHANGE"]),
            }
            for row in records
            if str(row.get("REGIONID", "")).upper() in REQUIRED_REGIONS
            and row.get("SETTLEMENTDATE") and row.get("NETINTERCHANGE") not in (None, "")
            and _parse_nem_time(str(row["SETTLEMENTDATE"])) is not None
        ],
    }
    return summary, response


def stitch_diagnostics(api_samples: list[dict], outlook_samples: list[dict]) -> list[dict]:
    """Measure the API-to-Seven-Day boundary for each regional interchange series."""
    results = []
    for region in sorted(REQUIRED_REGIONS):
        api = [row for row in api_samples if row["region"] == region]
        outlook = [row for row in outlook_samples if row["region"] == region]
        if not api or not outlook:
            results.append({"region": region, "status": "insufficient source rows"})
            continue
        api_last = max(api, key=lambda row: row["target_utc"])
        after = [row for row in outlook if row["target_utc"] > api_last["target_utc"]]
        if not after:
            results.append({"region": region, "status": "no Seven-Day target after short-term horizon"})
            continue
        outlook_first = min(after, key=lambda row: row["target_utc"])
        gap = (datetime.fromisoformat(outlook_first["target_utc"].replace("Z", "+00:00"))
               - datetime.fromisoformat(api_last["target_utc"].replace("Z", "+00:00"))).total_seconds() / 3600
        results.append({
            "region": region,
            "api_last_target_utc": api_last["target_utc"],
            "outlook_boundary_target_utc": outlook_first["target_utc"],
            "gap_hours": round(gap, 2),
            "net_interchange_jump_mw": round(
                float(outlook_first["net_interchange_mw"]) - float(api_last["net_interchange_mw"]), 1
            ),
        })
    return results


def _load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {"schemas": {}, "schema_change_history": [], "first_nsw1_sa1_utc": None}


def _write_json(path: Path, value: dict) -> None:
    _atomic_write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def capture(root: Path = DEFAULT_ROOT) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "canary_state.json"
    state = _load_state(state_path)
    schema_changes: list[str] = []
    issues: list[str] = []
    results: list[dict] = []
    api_samples: list[dict] = []
    outlook_samples: list[dict] = []
    now = datetime.now(timezone.utc)
    capture_from = datetime.fromisoformat(os.environ.get(
        "AEMO_TRANSITION_CAPTURE_FROM_UTC", DEFAULT_CAPTURE_FROM
    ).replace("Z", "+00:00"))

    try:
        api_summary, _ = _capture_json_api(root)
        results.append(api_summary)
        api_samples = api_summary["regional_net_interchange"]
        _record_horizon("visualisations_5min", api_summary["max_horizon_hours"], state, issues)
    except Exception as exc:
        issues.append(f"visualisations_5min: {exc}")

    for source in SOURCES:
        try:
            files = [
                (filename, url) for filename, url in _list_source_files(source)
                if datetime.fromisoformat(_run_time_from_name(source, filename).replace("Z", "+00:00")) >= capture_from
            ]
            if not files:
                raise ValueError(f"no matching ZIP files in CURRENT listing from {capture_from.isoformat()}")
            cadence = _cadence_diagnostics(source, [filename for filename, _url in files])
            if cadence and cadence["median_gap_minutes"] is not None:
                expected = cadence["expected_minutes"]
                tolerance = max(2.0, expected * 0.2)
                if abs(cadence["median_gap_minutes"] - expected) > tolerance:
                    issues.append(
                        f"{source.key}: report cadence median is {cadence['median_gap_minutes']:.1f} minutes; "
                        f"expected {expected}"
                    )
                if cadence["latest_gap_minutes"] > expected * 1.5:
                    issues.append(
                        f"{source.key}: latest report gap is {cadence['latest_gap_minutes']:.1f} minutes; "
                        f"expected {expected}"
                    )
            latest_name, _latest_url = files[-1]
            run_time = _run_time_from_name(source, latest_name)
            if run_time is None:
                raise ValueError(f"cannot parse run time from {latest_name}")

            captured_count = 0
            latest_modified_utc = None
            latest_tables = None
            max_horizon_hours = None
            latest_content = None
            for filename, url in files:
                destination = root / "raw" / source.key / filename
                meta = destination.with_name(destination.name + ".meta.json")
                if destination.is_file() and meta.is_file():
                    if filename != latest_name:
                        continue
                    latest_content = destination.read_bytes()
                    try:
                        latest_modified_utc = json.loads(meta.read_text()).get("source_last_modified_utc")
                    except (OSError, json.JSONDecodeError):
                        latest_modified_utc = None
                else:
                    try:
                        content, headers = _request(url)
                    except Exception as exc:
                        issues.append(f"{source.key}/{filename}: report download failed: {exc}")
                        if destination.is_file() and filename == latest_name:
                            latest_content = destination.read_bytes()
                        continue
                    _path, saved = _save_capture(
                        root, source.key, filename, content, url, headers,
                        _run_time_from_name(source, filename),
                    )
                    captured_count += int(saved)
                    if filename == latest_name:
                        latest_content = content
                        latest_modified_utc = _modified_utc(headers.get("last-modified"))
                    if filename != latest_name:
                        continue
            if latest_content is None:
                raise ValueError(f"could not retain latest report {latest_name}")
            try:
                inspection = inspect_aemo_zip(latest_content)
                latest_tables = inspection["tables"]
                max_horizon_hours = _max_report_horizon_hours(run_time, latest_tables)
                _record_horizon(source.key, max_horizon_hours, state, issues)
                report_issues = validate_report(source, inspection)
                issues.extend(f"{source.key}/{latest_name}: {issue}" for issue in report_issues)
                if source.key == "sevendayoutlook":
                    for table in latest_tables:
                        if _table_name(table["table"]) in source.region_tables:
                            outlook_samples = table["regional_net_interchange"]
                            break
                for table in latest_tables:
                    schema_key = f"{source.key}/{table['package']}/{table['table']}"
                    signature = {"version": table["version"], "columns": table["columns"]}
                    previous = state["schemas"].get(schema_key)
                    if previous is not None and previous != signature:
                        schema_changes.append(
                            f"{schema_key}: {previous.get('version')} -> {signature.get('version')}"
                        )
                        state.setdefault("schema_change_history", []).append({
                            "detected_utc": _utc_iso(), "schema_key": schema_key,
                            "previous": previous, "observed": signature,
                        })
                    state["schemas"][schema_key] = signature
                    if "NSW1-SA1" in table["interconnector_ids"] and not state.get("first_nsw1_sa1_utc"):
                        state["first_nsw1_sa1_utc"] = run_time
            except Exception as exc:
                issues.append(f"{source.key}/{latest_name}: report parse failed: {exc}")
            publication_time = latest_modified_utc or run_time
            age = (now - datetime.fromisoformat(publication_time.replace("Z", "+00:00"))).total_seconds() / 60
            future_tolerance = 40 if "predispatch" in source.key else 10
            if age < -future_tolerance or age > source.max_age_minutes:
                issues.append(f"{source.key}: latest publication/run {publication_time} is {age:.0f} minutes from now")
            results.append({
                "source": source.key, "latest_filename": latest_name,
                "latest_run_time_utc": run_time, "listing_files": len(files),
                "latest_publication_time_utc": latest_modified_utc,
                "new_files_saved": captured_count, "latest_saved_path": str(root / "raw" / source.key / latest_name),
                "cadence": cadence, "max_horizon_hours": max_horizon_hours,
                "latest_tables": latest_tables,
            })
        except Exception as exc:
            issues.append(f"{source.key}: {exc}")

    issues.extend(f"schema changed: {change}" for change in schema_changes)
    stitch = stitch_diagnostics(api_samples, outlook_samples) if api_samples and outlook_samples else []
    for item in stitch:
        if item.get("status"):
            issues.append(f"{item['region']} API/Seven-Day stitch: {item['status']}")
            continue
        if item["gap_hours"] > MAX_STITCH_GAP_HOURS:
            issues.append(f"{item['region']} API/Seven-Day stitch gap is {item['gap_hours']:.2f} hours")
        if abs(item["net_interchange_jump_mw"]) > MAX_STITCH_JUMP_MW:
            issues.append(
                f"{item['region']} API/Seven-Day interchange jump is "
                f"{item['net_interchange_jump_mw']:.0f} MW"
            )
    report = {
        "captured_utc": _utc_iso(now),
        "raw_root": str(root / "raw"),
        "flow_direction_convention": "positive from INTERCONNECTOR.FROMREGION",
        "first_nsw1_sa1_utc": state.get("first_nsw1_sa1_utc"),
        "results": results,
        "api_to_sevendayoutlook_stitch": stitch,
        "schema_changes": schema_changes,
        "issues": issues,
        "status": "failed" if issues else "ok",
    }
    _write_json(state_path, state)
    _write_json(root / "canary_latest.json", report)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = Path(os.environ.get("AEMO_TRANSITION_ROOT", DEFAULT_ROOT))
    try:
        report = capture(root)
    except Exception:
        logging.exception("AEMO transition report capture failed")
        try:
            record_job_status("aemo-pec-mi-transition", 1)
        except Exception as exc:
            logging.error("Could not record capture failure status: %s", exc)
        return 1
    for result in report["results"]:
        if result["source"] == "visualisations_5min":
            logging.info("%s API rows=%d regions=%s target=%s..%s saved=%s",
                         result["source"], result["rows"], result["regions"],
                         result["target_min"], result["target_max"], result["saved"])
        else:
            tables = result.get("latest_tables") or []
            ids = sorted({identifier for table in tables for identifier in table["interconnector_ids"]})
            logging.info("%s latest=%s run=%s captured=%d/%d interconnectors=%s",
                         result["source"], result["latest_filename"], result["latest_run_time_utc"],
                         result["new_files_saved"], result["listing_files"], ids)
    for issue in report["issues"]:
        logging.error("CANARY: %s", issue)
    failed = bool(report["issues"])
    try:
        record_job_status("aemo-pec-mi-transition", 1 if failed else 0)
    except Exception as exc:
        logging.error("Could not record capture status: %s", exc)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
