import pandas as pd
import pytest

import hwc_cycle_store as store


@pytest.fixture
def conn(tmp_path):
    c = store.connect(tmp_path / "hwc_cycles.sqlite")
    yield c
    c.close()


def _cycle(start_ts, **over):
    base = dict(
        start_ts=start_ts, start_local="2026-06-30 12:00", end_ts=start_ts + 3600,
        dur_min=60, tank_start=45.0, tank_end=60.0, ambient=14.0, wet_bulb=11.0,
        elec_kwh=1.5, elec_source="counter", therm_kwh=3.1, cop=2.07,
        hp_mean_w=900, hp_p95_w=1050, probe_lag_min=2.0,
        probe_rise_10_min=3.0, probe_rise_50_min=20.0, probe_rise_90_min=50.0,
        element_on=False, defrost_on=False, four_way_on=False,
        clean=True, status="complete", updated_at=start_ts + 3600,
    )
    base.update(over)
    return base


def test_init_creates_both_tables(conn):
    names = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"hwc_cycles", "hwc_cycle_samples"} <= names


def test_upsert_then_read_round_trips_and_coerces_bools(conn):
    store.upsert_cycle(conn, _cycle(1000.0, clean=True, element_on=False))
    rows = store.recent_cycles(conn, 5)
    assert len(rows) == 1
    row = rows[0]
    assert row["start_ts"] == 1000.0
    assert row["cop"] == 2.07
    # bools stored as 0/1 ints
    assert row["clean"] == 1
    assert row["element_on"] == 0


def test_upsert_replaces_same_start_ts(conn):
    store.upsert_cycle(conn, _cycle(2000.0, status="running", cop=None, tank_end=None))
    store.upsert_cycle(conn, _cycle(2000.0, status="complete", cop=2.5))
    rows = store.recent_cycles(conn, 5)
    assert len(rows) == 1
    assert rows[0]["status"] == "complete"
    assert rows[0]["cop"] == 2.5


def test_nan_and_missing_columns_store_as_null(conn):
    store.upsert_cycle(conn, {"start_ts": 3000.0, "status": "running", "cop": float("nan")})
    row = store.get_cycle(conn, 3000.0)
    assert row["cop"] is None
    assert row["tank_end"] is None  # absent key → NULL


def test_upsert_requires_start_ts(conn):
    with pytest.raises(ValueError):
        store.upsert_cycle(conn, {"status": "running"})


def test_recent_cycles_newest_first_and_limited(conn):
    for i in range(5):
        store.upsert_cycle(conn, _cycle(1000.0 + i * 100))
    rows = store.recent_cycles(conn, 3)
    assert [r["start_ts"] for r in rows] == [1400.0, 1300.0, 1200.0]


def test_recent_cycles_can_exclude_running(conn):
    store.upsert_cycle(conn, _cycle(1000.0, status="complete"))
    store.upsert_cycle(conn, _cycle(2000.0, status="running"))
    assert len(store.recent_cycles(conn, 5)) == 2
    finalised = store.recent_cycles(conn, 5, include_running=False)
    assert [r["start_ts"] for r in finalised] == [1000.0]


def test_get_running_returns_latest_running(conn):
    store.upsert_cycle(conn, _cycle(1000.0, status="complete"))
    store.upsert_cycle(conn, _cycle(2000.0, status="running"))
    store.upsert_cycle(conn, _cycle(3000.0, status="running"))
    assert store.get_running(conn)["start_ts"] == 3000.0


def test_get_running_none_when_all_complete(conn):
    store.upsert_cycle(conn, _cycle(1000.0, status="complete"))
    assert store.get_running(conn) is None


def test_append_and_load_trace_round_trips(conn):
    cs = 1000.0
    samples = [
        {"ts": cs + 30 * i, "tank": 45.0 + i, "power_w": 900.0, "energy_kwh": 10.0 + 0.01 * i,
         "ambient": 14.0, "element": False, "defrost": False, "exhaust": 20.0 + i,
         "coil": None, "return_air": 18.0, "inlet": 15.0}
        for i in range(4)
    ]
    n = store.append_samples(conn, cs, samples)
    assert n == 4
    trace = store.load_trace(conn, cs)
    assert len(trace) == 4
    assert isinstance(trace.index, pd.DatetimeIndex)
    assert str(trace.index.tz) == "UTC"
    assert list(trace.columns) == [
        "tank", "power_w", "energy_kwh", "ambient", "humidity",
        "element", "defrost", "four_way", "exhaust", "coil", "return_air", "inlet", "fan",
    ]
    assert trace["tank"].iloc[0] == 45.0
    assert trace["element"].iloc[0] == 0
    assert pd.isna(trace["coil"].iloc[0])  # stored NULL → null-like on load


def test_append_samples_idempotent_on_ts(conn):
    cs = 1000.0
    store.append_samples(conn, cs, [{"ts": cs, "tank": 45.0}])
    store.append_samples(conn, cs, [{"ts": cs, "tank": 46.0}])  # same ts → replace
    trace = store.load_trace(conn, cs)
    assert len(trace) == 1
    assert trace["tank"].iloc[0] == 46.0


def test_append_samples_accepts_dataframe(conn):
    cs = 1000.0
    df = pd.DataFrame({"ts": [cs, cs + 30], "tank": [45.0, 46.0], "power_w": [900.0, 905.0]})
    assert store.append_samples(conn, cs, df) == 2
    assert len(store.load_trace(conn, cs)) == 2


def test_load_trace_empty_for_unknown_cycle(conn):
    trace = store.load_trace(conn, 9999.0)
    assert trace.empty
    assert "tank" in trace.columns
    assert isinstance(trace.index, pd.DatetimeIndex)


def test_read_only_connection_cannot_write(tmp_path):
    path = tmp_path / "hwc_cycles.sqlite"
    w = store.connect(path)
    store.upsert_cycle(w, _cycle(1000.0))
    w.close()
    ro = store.connect(path, read_only=True)
    assert len(store.recent_cycles(ro, 5)) == 1
    with pytest.raises(Exception):
        store.upsert_cycle(ro, _cycle(2000.0))
    ro.close()
