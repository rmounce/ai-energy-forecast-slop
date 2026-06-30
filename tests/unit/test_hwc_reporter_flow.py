"""Integration test for the store-based daemon reporter: compressor edges → SQLite cycle.

Drives the async reporter methods directly (via asyncio.run) on a __new__'d daemon with a temp
store, the HA publish monkeypatched out — exercising open → sample → close → cycle_metrics finalise
and the edge-snapshot elec.
"""

import asyncio
from datetime import datetime, timezone

import pytest

import hwc_cycle_store as store
import hwc_planner
import services.hwc_daemon as hd

ENTITIES = {
    "compressor": "binary_sensor.aquatech_compressor",
    "tank": "sensor.aquatech_current_temperature_local",
    "power": "sensor.power_2",
    "energy": "sensor.energy_2",
    "ambient": "sensor.aquatech_temperature",
    "element": "binary_sensor.aquatech_element",
    "defrost": "binary_sensor.aquatech_defrost",
    "four_way": "binary_sensor.aquatech_four_way_valve",
    "exhaust": "sensor.aquatech_exhaust_temperature",
    "coil": "sensor.aquatech_coil_temperature",
    "return_air": "sensor.aquatech_return_air_temperature",
}


def _reporter_daemon(tmp_path):
    cfg = {
        "timezone": "Australia/Adelaide",
        "home_assistant": {"url": "http://x", "token": "t"},
        "hwc": {
            "daemon": {"state_file": str(tmp_path / "state.json")},
            "reporting": {
                "enabled": True,
                "cycles_entity": "sensor.hwc_cycles",
                "db_path": str(tmp_path / "hwc.sqlite"),
                "sample_seconds": 30,
                "history_len": 20,
                "min_cycle_seconds": 300,
                "humidity_entity": "weather.x",
                "entities": ENTITIES,
            },
        },
    }
    d = hd.HwcDaemon.__new__(hd.HwcDaemon)
    d.config = cfg
    d.last_reached_target_at = None
    d._report_enabled = True
    d._report_db_path = cfg["hwc"]["reporting"]["db_path"]
    d.report_entities = dict(ENTITIES)
    d.report_entity_role = {eid: role for role, eid in ENTITIES.items()}
    d.humidity_entity = "weather.x"
    d.report_cache = {}
    d.reporter_cycle = None
    d._reporter_prev_on = False  # as if startup seeded the compressor 'off'
    return d


def _evt(ts_epoch):
    return datetime.fromtimestamp(ts_epoch, tz=timezone.utc)


def test_open_sample_close_writes_finalised_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(hwc_planner, "_ha_set_state", lambda *a, **k: None)
    d = _reporter_daemon(tmp_path)
    t0 = 1_750_000_000.0

    async def flow():
        # warm the cache, then compressor on -> open
        d.report_cache.update(
            tank=45.0, energy=100.0, power=950.0, ambient=14.0, humidity=70.0,
            element=False, defrost=False, four_way=False,
            exhaust=20.0, coil=5.0, return_air=18.0,
        )
        await d._reporter_observe(ENTITIES["compressor"], {"state": "on"}, _evt(t0))
        assert d.reporter_cycle is not None and d.reporter_cycle["energy_start"] == 100.0

        # midway sample
        d.report_cache.update(tank=51.0, energy=100.5, power=1000.0)
        await d._reporter_sample_tick()

        # compressor off after 1 h -> close + finalise (tank/energy from the edge snapshot)
        d.report_cache.update(tank=56.0, energy=101.0, power=2.0)
        await d._reporter_observe(ENTITIES["compressor"], {"state": "off"}, _evt(t0 + 3600))
        assert d.reporter_cycle is None

    asyncio.run(flow())

    conn = store.connect(d._report_db_path, read_only=True)
    rows = store.recent_cycles(conn, 10)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "complete"
    assert abs(row["elec_kwh"] - 1.0) < 1e-6          # 101.0 - 100.0, edge counter
    assert row["elec_source"] == "counter"
    assert row["tank_start"] == 45.0 and row["tank_end"] == 56.0   # edges, not +90s anchor
    assert row["cop"] is not None
    trace = store.load_trace(conn, row["start_ts"])
    assert len(trace) >= 3                            # open + sample_tick + close samples
    conn.close()


def test_subminimum_run_is_discarded(tmp_path, monkeypatch):
    monkeypatch.setattr(hwc_planner, "_ha_set_state", lambda *a, **k: None)
    d = _reporter_daemon(tmp_path)
    t0 = 1_750_000_000.0

    async def flow():
        d.report_cache.update(tank=45.0, energy=100.0)
        await d._reporter_observe(ENTITIES["compressor"], {"state": "on"}, _evt(t0))
        # off after 60 s (< min_cycle_seconds 300) -> discarded
        await d._reporter_observe(ENTITIES["compressor"], {"state": "off"}, _evt(t0 + 60))

    asyncio.run(flow())
    conn = store.connect(d._report_db_path, read_only=True)
    assert store.recent_cycles(conn, 10) == []
    assert store.get_running(conn) is None
    conn.close()


def test_humidity_cached_from_weather_attribute(tmp_path, monkeypatch):
    monkeypatch.setattr(hwc_planner, "_ha_set_state", lambda *a, **k: None)
    d = _reporter_daemon(tmp_path)
    asyncio.run(d._reporter_observe("weather.x", {"state": "cloudy", "attributes": {"humidity": 82}},
                                    _evt(1_750_000_000.0)))
    assert d.report_cache["humidity"] == 82.0
