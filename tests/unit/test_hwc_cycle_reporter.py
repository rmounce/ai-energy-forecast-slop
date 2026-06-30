import math

import pandas as pd

import hwc_cop_analysis as hca
import hwc_cycle_reporter as cr

TZ = "Australia/Adelaide"


# ── counter selection (hwc_cop_analysis) ────────────────────────────────────


def _energy_series(values, start="2026-06-18T04:00:00Z", freq="60s"):
    idx = pd.date_range(start, periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def test_counter_cycle_kwh_returns_delta_over_window():
    s = _energy_series([5.0, 5.4, 6.0, 6.72])
    cs, ce = s.index[0], s.index[-1]
    assert cr_close(hca.counter_cycle_kwh(s, cs, ce), 1.72)


def test_counter_cycle_kwh_none_when_missing_or_too_few_samples():
    assert hca.counter_cycle_kwh(None, pd.Timestamp("2026-06-18T04:00Z"),
                                 pd.Timestamp("2026-06-18T06:00Z")) is None
    assert hca.counter_cycle_kwh(pd.Series(dtype=float),
                                 pd.Timestamp("2026-06-18T04:00Z"),
                                 pd.Timestamp("2026-06-18T06:00Z")) is None
    s = _energy_series([5.0])
    assert hca.counter_cycle_kwh(s, s.index[0], s.index[0]) is None


def test_counter_cycle_kwh_none_on_reset_or_implausible_delta():
    reset = _energy_series([6.0, 6.2, 0.1, 0.3])  # mid-cycle meter reset
    assert hca.counter_cycle_kwh(reset, reset.index[0], reset.index[-1]) is None
    huge = _energy_series([0.0, 99.0])             # rollover/glitch
    assert hca.counter_cycle_kwh(huge, huge.index[0], huge.index[-1]) is None


# ── store-row projection (the live reporter feeds build_payload from SQLite) ─────────────────


def _store_row(start_ts=1.75e9, **over):
    base = dict(
        start_ts=start_ts, start_local="2026-06-30 12:00", end_ts=start_ts + 3600,
        dur_min=60, tank_start=45.0, tank_end=60.0, ambient=14.0, wet_bulb=11.0,
        elec_kwh=1.5, elec_source="counter", therm_kwh=3.1, cop=2.07,
        element_on=0, defrost_on=0, clean=1, status="complete",
    )
    base.update(over)
    return base


def test_record_from_store_row_projects_and_restores_bools():
    rec = cr.record_from_store_row(_store_row(clean=1, element_on=0, cop=2.07))
    assert rec["start"] == "2026-06-30 12:00"   # start_local becomes 'start'
    assert rec["cop"] == 2.07
    assert rec["clean"] is True                 # 0/1 -> bool
    assert rec["element_on"] is False
    assert set(rec) == set(cr.PUBLISH_COLS)


def test_record_from_store_row_nan_and_none_become_none():
    rec = cr.record_from_store_row(_store_row(cop=float("nan"), elec_kwh=None))
    assert rec["cop"] is None
    assert rec["elec_kwh"] is None


def test_records_from_store_rows_sorts_oldest_first():
    rows = [_store_row(start_ts=200, start_local="2026-06-30 12:00"),
            _store_row(start_ts=100, start_local="2026-06-29 12:00")]
    recs = cr.records_from_store_rows(rows)
    assert [r["start"] for r in recs] == ["2026-06-29 12:00", "2026-06-30 12:00"]


# ── live (in-progress) row from the daemon's edge snapshot + cache ───────────────────────────


def test_live_record_reports_counter_delta_and_elapsed():
    rc = {"start_ts": 1.75e9, "tank_start": 45.0, "energy_start": 100.0}
    view = cr.live_record(rc, tank_now=52.0, energy_now=100.5,
                          now_ts=1.75e9 + 1800, tz_name=TZ)
    assert view["status"] == "running"
    assert view["dur_min"] == 30
    assert cr_close(view["elec_kwh"], 0.5)
    assert cr_close(view["dt_c"], 7.0)
    assert view["elec_source"] == "counter"


def test_live_record_handles_counter_reset_and_missing():
    rc = {"start_ts": 1.75e9, "tank_start": 45.0, "energy_start": 100.0}
    # energy_now below energy_start (reset) -> no elec
    reset = cr.live_record(rc, tank_now=46.0, energy_now=99.0, now_ts=1.75e9 + 60, tz_name=TZ)
    assert reset["elec_kwh"] is None and reset["elec_source"] is None
    # missing tank_now -> no dt_c
    missing = cr.live_record(rc, tank_now=None, energy_now=100.2, now_ts=1.75e9 + 60, tz_name=TZ)
    assert missing["dt_c"] is None and cr_close(missing["elec_kwh"], 0.2)


def test_live_record_none_without_open_cycle():
    assert cr.live_record(None, tank_now=50.0, energy_now=1.0, now_ts=1.75e9, tz_name=TZ) is None
    assert cr.live_record({}, tank_now=50.0, energy_now=1.0, now_ts=1.75e9, tz_name=TZ) is None


def test_build_payload_state_is_last_cop_and_counters():
    cycles = [
        {"start": "2026-06-26 12:00", "cop": 2.2, "clean": True},
        {"start": "2026-06-27 12:00", "cop": 2.5, "clean": False},
    ]
    state, attrs = cr.build_payload(cycles, live=None, today_local="2026-06-27")
    assert state == 2.5
    assert attrs["cycles"][0]["start"] == "2026-06-27 12:00"  # most recent first
    assert attrs["cycles_today"] == 1
    assert attrs["last_clean_cop"] == 2.2  # the most recent clean cycle
    assert attrs["current"] is None

    empty_state, empty_attrs = cr.build_payload([], live={"status": "running"}, today_local="x")
    assert empty_state == "unknown"
    assert empty_attrs["current"] == {"status": "running"}


def test_build_payload_headline_skips_null_cop_rows():
    # A run can land with a null COP (e.g. tank-probe gap at the cycle start); the headline should
    # fall back to the most recent computable COP rather than going "unknown".
    cycles = [
        {"start": "2026-06-29 12:51", "cop": 2.53, "clean": True},
        {"start": "2026-06-30 12:28", "cop": None, "clean": False},
    ]
    state, _ = cr.build_payload(cycles, live=None, today_local="2026-06-30")
    assert state == 2.53
    # all-null → unknown
    allnull, _ = cr.build_payload([{"start": "x", "cop": None}], live=None, today_local="x")
    assert allnull == "unknown"


def cr_close(a, b, tol=1e-6):
    return a is not None and math.isclose(a, b, abs_tol=tol)

