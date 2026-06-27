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


# ── live row state machine ──────────────────────────────────────────────────


def test_advance_live_opens_running_on_compressor_start():
    cur = cr.advance_live(None, raw_on=True, now_ts=1000.0, tank_c=47.0, energy_kwh=5.0)
    assert cur["status"] == "running"
    assert cur["start_ts"] == 1000.0
    assert cur["tank_start"] == 47.0 and cur["energy_start"] == 5.0


def test_advance_live_updates_while_running_then_cools_down():
    cur = cr.advance_live(None, raw_on=True, now_ts=1000.0, tank_c=47.0, energy_kwh=5.0)
    cur = cr.advance_live(cur, raw_on=True, now_ts=1600.0, tank_c=52.0, energy_kwh=5.5)
    assert cur["status"] == "running"
    assert cur["tank_now"] == 52.0 and cur["energy_now"] == 5.5
    assert cur["tank_start"] == 47.0  # start snapshot preserved

    cur = cr.advance_live(cur, raw_on=False, now_ts=2200.0, tank_c=60.0, energy_kwh=6.7)
    assert cur["status"] == "cooldown"
    assert cur["ended_ts"] == 2200.0 and cur["energy_now"] == 6.7


def test_advance_live_cooldown_persists_until_cleared_and_reopens_on_next_start():
    cur = {"status": "cooldown", "start_ts": 1000.0, "ended_ts": 2200.0,
           "tank_start": 47.0, "energy_start": 5.0, "tank_now": 60.0, "energy_now": 6.7}
    same = cr.advance_live(cur, raw_on=False, now_ts=2300.0, tank_c=59.0, energy_kwh=6.7)
    assert same["status"] == "cooldown"  # unchanged while off
    nxt = cr.advance_live(cur, raw_on=True, now_ts=9000.0, tank_c=50.0, energy_kwh=7.0)
    assert nxt["status"] == "running" and nxt["start_ts"] == 9000.0


def test_advance_live_unknown_compressor_read_left_to_caller():
    # The daemon skips advance_live when the compressor read fails (raw_on=None), so a flaky
    # read never spuriously closes a cycle. This asserts the contract the daemon relies on.
    cur = cr.advance_live(None, raw_on=True, now_ts=1000.0, tank_c=47.0, energy_kwh=5.0)
    # passing raw_on=False would cool it down; daemon avoids that by not calling on None.
    assert cur["status"] == "running"


# ── live view rendering ─────────────────────────────────────────────────────


def test_live_view_reports_counter_delta_and_elapsed():
    cur = {"status": "running", "start_ts": 1000.0,
           "tank_start": 47.0, "energy_start": 5.0, "tank_now": 52.0, "energy_now": 5.5}
    view = cr.live_view(cur, now_ts=1000.0 + 1800, tz_name=TZ)
    assert view["dur_min"] == 30
    assert view["dt_c"] == 5.0
    assert cr_close(view["elec_kwh"], 0.5)
    assert view["elec_source"] == "counter"
    assert view["status"] == "running"


def test_live_view_handles_counter_reset_and_missing():
    cur = {"status": "running", "start_ts": 1000.0,
           "tank_start": 47.0, "energy_start": 6.0, "tank_now": 52.0, "energy_now": 0.2}
    view = cr.live_view(cur, now_ts=1100.0, tz_name=TZ)
    assert view["elec_kwh"] is None and view["elec_source"] is None
    assert cr.live_view(None, now_ts=1100.0, tz_name=TZ) is None


# ── records / merge / capture / payload ─────────────────────────────────────


def _analysis_df():
    return pd.DataFrame([
        {"start": pd.Timestamp("2026-06-18T04:07:00Z"), "dur_min": 136,
         "tank_start": 47.1, "tank_end": 60.0, "ambient": 14.5, "wet_bulb": 9.8,
         "elec_kwh": 1.72, "elec_source": "counter", "therm_kwh": 3.66, "cop": 2.13,
         "element_on": False, "defrost_on": False, "clean": True},
        {"start": pd.Timestamp("2026-06-17T03:06:00Z"), "dur_min": 59,
         "tank_start": 53.9, "tank_end": 60.0, "ambient": 19.3, "wet_bulb": 16.0,
         "elec_kwh": 0.82, "elec_source": "power_integration", "therm_kwh": 1.70,
         "cop": float("nan"), "element_on": False, "defrost_on": False, "clean": True},
    ])


def test_records_from_analysis_normalises_start_and_nan():
    records = cr.records_from_analysis(_analysis_df())
    assert records[0]["start"] == "2026-06-18 13:37"   # UTC 04:07 → Adelaide
    assert records[0]["elec_source"] == "counter"
    assert records[1]["cop"] is None                    # NaN → None
    assert cr.records_from_analysis(pd.DataFrame()) == []


def test_merge_records_dedupes_by_start_and_trims():
    existing = [{"start": "2026-06-16 12:13", "cop": 2.27}]
    new = [
        {"start": "2026-06-17 12:36", "cop": 2.09},
        {"start": "2026-06-16 12:13", "cop": 9.99},  # replaces existing
    ]
    merged = cr.merge_records(existing, new, maxlen=2)
    assert [r["start"] for r in merged] == ["2026-06-16 12:13", "2026-06-17 12:36"]
    assert merged[0]["cop"] == 9.99
    assert len(cr.merge_records(existing, new, maxlen=1)) == 1  # trims to most recent


def test_backfill_captured_matches_within_tolerance():
    start_ts = cr._local_str_to_ts("2026-06-18 13:37", TZ)
    cur = {"status": "cooldown", "start_ts": start_ts}
    # analyse start 2 min off the polled start still matches
    assert cr.backfill_captured(cur, [{"start": "2026-06-18 13:39"}], TZ) is True
    assert cr.backfill_captured(cur, [{"start": "2026-06-18 14:30"}], TZ) is False
    running = {"status": "running", "start_ts": start_ts}
    assert cr.backfill_captured(running, [{"start": "2026-06-18 13:37"}], TZ) is False


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


def cr_close(a, b, tol=1e-6):
    return a is not None and math.isclose(a, b, abs_tol=tol)
