"""Unit tests for the pure (non-InfluxDB) helpers of the throwaway seed script."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import hwc_cycle_store as store

_SEED_PATH = Path(__file__).resolve().parents[2] / "scripts" / "seed_hwc_store.py"
_spec = importlib.util.spec_from_file_location("seed_hwc_store", _SEED_PATH)
seed = importlib.util.module_from_spec(_spec)
sys.modules["seed_hwc_store"] = seed
_spec.loader.exec_module(seed)


def _grid(minutes_on=60, gap=20):
    """A grid: `gap` min off, `minutes_on` on, `gap` min off — one detectable cycle."""
    n = (gap + minutes_on + gap) * 2 + 1
    idx = pd.date_range("2026-06-20T01:00:00Z", periods=n, freq="30s", tz="UTC")
    on = np.zeros(n, dtype=bool)
    on[gap * 2: (gap + minutes_on) * 2] = True
    df = pd.DataFrame(index=idx)
    df["on"] = on
    for col in seed.TRACE_COLS:
        df[col] = 1.0
    df["tank"] = np.linspace(45.0, 60.0, n)
    df["element"] = 0
    df["defrost"] = 0
    df["four_way"] = 0
    return df


def test_iter_cycles_detects_on_run_and_slices_trace_columns():
    cycles = list(seed.iter_cycles(_grid(minutes_on=60), min_minutes=5))
    assert len(cycles) == 1
    cs, ce, trace = cycles[0]
    assert len(trace) == 120  # 60 min of 30 s samples (span 59.5 min)
    assert list(trace.columns) == seed.TRACE_COLS  # 'on' dropped from the slice


def test_iter_cycles_skips_runs_under_min_minutes():
    assert list(seed.iter_cycles(_grid(minutes_on=3), min_minutes=5)) == []


def test_trace_samples_carry_epoch_ts_and_sensor_columns():
    _, _, trace = next(seed.iter_cycles(_grid(), min_minutes=5))
    rows = seed._trace_samples(trace)
    assert len(rows) == len(trace)
    first = rows[0]
    assert "ts" in first and isinstance(first["ts"], float)
    assert set(seed.TRACE_COLS) <= set(first)
    # ts is epoch seconds of the trace index
    assert abs(first["ts"] - trace.index[0].timestamp()) < 1e-6


def test_seed_csv_backfills_only_pre_window_rows(tmp_path):
    csv = tmp_path / "cycles.csv"
    pd.DataFrame(
        [
            {"start": "2026-05-28 10:00", "dur_min": 60, "tank_start": 40, "tank_end": 60,
             "cop": 2.1, "clean": True},
            {"start": "2026-06-25 10:00", "dur_min": 55, "tank_start": 45, "tank_end": 60,
             "cop": 2.3, "clean": True},
        ]
    ).to_csv(csv, index=False)
    conn = store.connect(tmp_path / "store.sqlite")
    # window opens 2026-06-01: only the 05-28 row is "before" and should seed
    before_ts = pd.Timestamp("2026-06-01T00:00:00", tz="Australia/Adelaide").timestamp()
    n = seed.seed_csv(conn, csv, before_ts, "Australia/Adelaide")
    assert n == 1
    rows = store.recent_cycles(conn, 10)
    assert len(rows) == 1
    assert rows[0]["start_local"] == "2026-05-28 10:00"
    assert rows[0]["cop"] == 2.1
    assert rows[0]["status"] == "complete"
    assert rows[0]["elec_source"] is None  # CSV format has no elec_source
    conn.close()


def test_seed_csv_missing_file_is_noop(tmp_path):
    conn = store.connect(tmp_path / "store.sqlite")
    assert seed.seed_csv(conn, tmp_path / "absent.csv", 0.0, "Australia/Adelaide") == 0
    conn.close()
