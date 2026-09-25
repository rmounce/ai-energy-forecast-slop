import io
import json
import zipfile
from datetime import datetime, timedelta, timezone

from ingest import capture_aemo_transition_reports as capture


def _zip_report(*, version="4", extra_column=False, duplicate=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        rows = [
            ["I", "P5MIN", "REGIONSOLUTION", "7", "RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID", "NET_INTERCHANGE"],
        ]
        if extra_column:
            rows[0].append("SCHEDULED_DEMAND")
        run = "2026/09/23 12:00:00"
        target = "2026/09/23 12:30:00"
        for region in ("SA1", "VIC1", "NSW1"):
            row = ["D", "P5MIN", "REGIONSOLUTION", "7", run, target, region, "100.0"]
            if extra_column:
                row.append("200.0")
            rows.append(row)
            if duplicate and region == "SA1":
                rows.append(row)
        rows.extend([
            ["I", "P5MIN", "INTERCONNECTORSOLN", version, "RUN_DATETIME", "INTERVAL_DATETIME", "INTERCONNECTORID", "MWFLOW"],
            ["D", "P5MIN", "INTERCONNECTORSOLN", version, run, target, "VIC1-SA1", "25.0"],
        ])
        csv_data = "\n".join(",".join(row) for row in rows) + "\n"
        archive.writestr("report.csv", csv_data)
    return output.getvalue()


def _source():
    return capture.ReportSource(
        "test", "https://example.invalid/current/",
        capture._pattern(r"PUBLIC_TEST_(\d{12})\.zip$"), 20,
        ("REGIONSOLUTION",), ("INTERCONNECTORSOLN", "P5MIN_INTERCONNECTORSOLN"),
    )


def test_inspect_and_validate_aemo_report_tables():
    inspection = capture.inspect_aemo_zip(_zip_report())
    issues = capture.validate_report(_source(), inspection)

    assert issues == []
    region = next(table for table in inspection["tables"] if table["table"] == "REGIONSOLUTION")
    interconnector = next(table for table in inspection["tables"] if table["table"] == "INTERCONNECTORSOLN")
    assert region["regions"] == ["NSW1", "SA1", "VIC1"]
    assert region["target_min"] == "2026-09-23T02:30:00Z"
    assert interconnector["interconnector_ids"] == ["VIC1-SA1"]


def test_report_canary_finds_missing_rows_fields_and_duplicate_keys():
    inspection = capture.inspect_aemo_zip(_zip_report(duplicate=True))
    region = next(table for table in inspection["tables"] if table["table"] == "REGIONSOLUTION")
    region["regions"].remove("NSW1")
    region["columns"].remove("NET_INTERCHANGE")

    issues = capture.validate_report(_source(), inspection)

    assert any("missing region rows" in issue for issue in issues)
    assert any("missing net-interchange column" in issue for issue in issues)
    assert any("duplicate keys" in issue for issue in issues)


def test_nem_iso_timestamps_and_legacy_package_region_table():
    source = capture.ReportSource("legacy", "https://example.invalid", capture._pattern(".*"), 20,
                                  ("PDREGION",), requires_net_interchange=True)
    inspection = {"tables": [{
        "package": "PDREGION", "table": "", "columns": ["PERIODID", "REGIONID", "NETINTERCHANGE"],
        "regions": ["NSW1", "SA1", "VIC1"], "row_count": 3,
        "duplicate_keys": 0, "field_width_mismatches": 0,
        "interconnector_ids": [], "target_min": "2026-09-23T03:00:00Z",
        "target_max": "2026-09-23T03:00:00Z", "exact_duplicate_rows": 0,
        "regional_net_interchange": [],
    }]}

    assert capture._utc_iso(capture._parse_nem_time("2026-09-23T13:05:00")) == "2026-09-23T03:05:00Z"
    assert capture.validate_report(source, inspection) == []


def test_stpasa_duplicate_key_uses_run_type_and_study_region():
    columns = ["RUN_DATETIME", "INTERVAL_DATETIME", "INTERCONNECTORID", "RUNTYPE", "STUDYREGIONID"]
    assert capture._primary_key_fields(columns) == tuple(columns)


def test_cadence_and_horizon_diagnostics_detect_source_changes():
    source = capture.ReportSource(
        "test", "https://example.invalid", capture._pattern(r"PUBLIC_TEST_(\d{12})\.zip$"),
        20, ("REGIONSOLUTION",), cadence_minutes=5,
    )
    cadence = capture._cadence_diagnostics(source, [
        "PUBLIC_TEST_202609231200.zip",
        "PUBLIC_TEST_202609231205.zip",
        "PUBLIC_TEST_202609231210.zip",
    ])
    assert cadence["median_gap_minutes"] == 5
    assert cadence["latest_gap_minutes"] == 5

    state = {}
    issues = []
    capture._record_horizon("test", 10.25, state, issues)
    capture._record_horizon("test", 9.5, state, issues)
    assert issues == ["test: forecast horizon changed from 10.25 to 9.50 hours"]


def test_horizon_shrinking_with_fixed_forecast_end_is_not_a_failure():
    state = {}
    issues = []
    capture._record_horizon("stpasa", 163, state, issues,
                            run_time_utc="2026-09-24T23:00:00Z")
    capture._record_horizon("stpasa", 162, state, issues,
                            run_time_utc="2026-09-25T00:00:00Z")
    assert issues == []
    assert state["horizon_end_utc"]["stpasa"] == "2026-10-01T18:00:00Z"

    capture._record_horizon("stpasa", 160, state, issues,
                            run_time_utc="2026-09-25T01:00:00Z")
    assert issues == ["stpasa: forecast horizon changed from 162.00 to 160.00 hours"]


def test_short_api_coverage_requires_three_consecutive_captures():
    state = {}
    diagnostics = []
    issues = []
    for _ in range(2):
        capture._record_api_coverage(-0.46, state, diagnostics, issues)
    assert issues == []
    assert state["api_short_coverage_runs"] == 2
    capture._record_api_coverage(-0.46, state, diagnostics, issues)
    assert issues == ["visualisations_5min: only -0.46 hours of future data (3 captures)"]
    capture._record_api_coverage(17, state, diagnostics, issues)
    assert state["api_short_coverage_runs"] == 0


def test_api_transport_outage_alerts_after_30_minutes_and_resets_on_success():
    started = datetime(2026, 9, 26, 0, tzinfo=timezone.utc)
    state = {}
    error = capture.RequestFailure("upstream timed out")
    diagnostics = []
    issues = []
    capture._record_api_transport_failure(error, state, started, diagnostics, issues)
    capture._record_api_transport_failure(
        error, state, started + timedelta(minutes=29), diagnostics, issues,
    )
    assert len(diagnostics) == 2
    assert issues == []
    capture._record_api_transport_failure(
        error, state, started + timedelta(minutes=30), diagnostics, issues,
    )
    assert len(issues) == 1
    assert "30 minutes of failed captures" in issues[0]

    state["api_transport_failure_started_utc"] = None
    diagnostics.clear()
    issues.clear()
    capture._record_api_transport_failure(
        error, state, started + timedelta(minutes=35), diagnostics, issues,
    )
    assert len(diagnostics) == 1
    assert issues == []


def test_current_dispatchis_and_predispatchis_table_names_are_recognized():
    for source, region_table, interconnector_table, time_column, interchange_column in (
        (_source(), "REGIONSOLUTION", "INTERCONNECTORSOLN", "INTERVAL_DATETIME", "NET_INTERCHANGE"),
         (capture.ReportSource("dispatchis", "", capture._pattern(".*"), 20,
                              ("REGIONSUM",), ("INTERCONNECTORRES", "DISPATCHINTERCONNECTORRES")),
         "REGIONSUM", "INTERCONNECTORRES", "DISPATCHINTERVAL", "NETINTERCHANGE"),
        (capture.ReportSource("predispatchis", "", capture._pattern(".*"), 90,
                              ("REGION_SOLUTION",), ("INTERCONNECTOR_SOLN", "PREDISPATCHINTERCONNECTORRES")),
         "REGION_SOLUTION", "INTERCONNECTOR_SOLN", "PERIODID", "NETINTERCHANGE"),
    ):
        inspection = {"tables": [
            {"package": source.key.upper(), "table": region_table,
             "columns": [time_column, "REGIONID", interchange_column],
             "regions": ["NSW1", "SA1", "VIC1"], "row_count": 3,
             "duplicate_keys": 0, "field_width_mismatches": 0},
            {"package": source.key.upper(), "table": interconnector_table,
             "columns": ["INTERCONNECTORID", "MWFLOW"], "regions": [],
             "interconnector_ids": [], "row_count": 1,
             "duplicate_keys": 0, "field_width_mismatches": 0},
        ]}
        assert capture.validate_report(source, inspection) == []


def test_stitch_check_compares_first_non_overlapping_interval():
    api = [{"region": "VIC1", "target_utc": "2026-09-24T17:30:00Z", "net_interchange_mw": 1272.99}]
    outlook = [
        {"region": "VIC1", "target_utc": "2026-09-24T17:30:00Z", "net_interchange_mw": 0.0},
        {"region": "VIC1", "target_utc": "2026-09-24T18:00:00Z", "net_interchange_mw": -1280.99},
    ]

    result = next(row for row in capture.stitch_diagnostics(api, outlook) if row["region"] == "VIC1")

    assert result["outlook_boundary_target_utc"] == "2026-09-24T18:00:00Z"
    assert result["gap_hours"] == 0.5
    assert result["net_interchange_jump_mw"] == -2554.0


def test_save_capture_preserves_bytes_and_records_separate_times(tmp_path):
    content = b"exact as-issued response bytes\x00"
    path, saved = capture._save_capture(
        tmp_path, "fixture", "report.zip", content,
        "https://example.invalid/report.zip",
        {"last-modified": "Wed, 23 Sep 2026 02:00:00 GMT", "content-type": "application/zip"},
        "2026-09-23T02:05:00Z",
    )
    _, saved_again = capture._save_capture(
        tmp_path, "fixture", "report.zip", b"changed", "https://example.invalid/report.zip", {}, None,
    )
    metadata = json.loads(path.with_name(path.name + ".meta.json").read_text())

    assert saved is True
    assert saved_again is False
    assert path.read_bytes() == content
    assert metadata["run_time_utc"] == "2026-09-23T02:05:00Z"
    assert metadata["source_last_modified_utc"] == "2026-09-23T02:00:00Z"
    assert metadata["sha256"] == capture.hashlib.sha256(content).hexdigest()


def test_capture_reports_new_schema_once_and_retains_change_history(tmp_path, monkeypatch):
    source = _source()
    base = datetime.now(timezone.utc).astimezone(capture.NEM_TZ).replace(second=0, microsecond=0)
    current = {"name": base.strftime("PUBLIC_TEST_%Y%m%d%H%M.zip"), "content": _zip_report()}
    monkeypatch.setattr(capture, "SOURCES", (source,))
    monkeypatch.setattr(capture, "_capture_json_api", lambda root: ({"source": "api", "regional_net_interchange": []}, b"{}"))
    monkeypatch.setattr(capture, "_list_source_files", lambda selected: [(current["name"], "https://example.invalid/file")])
    monkeypatch.setattr(
        capture, "_request",
        lambda url, **kwargs: (current["content"], {"last-modified": datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")}),
    )

    first = capture.capture(tmp_path)
    current["name"] = "PUBLIC_TEST_" + (base + timedelta(minutes=5)).strftime("%Y%m%d%H%M") + ".zip"
    current["content"] = _zip_report(version="5", extra_column=True)
    second = capture.capture(tmp_path)
    third = capture.capture(tmp_path)
    state = json.loads((tmp_path / "canary_state.json").read_text())

    assert first["schema_changes"] == []
    assert any("schema changed" in issue for issue in second["issues"])
    assert third["schema_changes"] == []
    assert state["schema_change_history"]
