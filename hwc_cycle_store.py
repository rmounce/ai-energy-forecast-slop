#!/usr/bin/env python3
"""Local SQLite system-of-record for HWC (heat-pump hot water) per-cycle data.

InfluxDB cannot hold HWC history: the default retention policy (``rp_raw``) keeps only 30 days and
no Aquatech/HWC entity is downsampled into the longer policies (those continuous queries cover only
the forecasting signals). So per-cycle telemetry — needed both for the live "recent runs" card and
for long-term COP characterisation — lives here instead, in-repo runtime data the daemon owns.
See ``docs/hwc_local_store.md``.

Two tables:

  ``hwc_cycles``          one summary row per cycle (the 20-row card ring is a ``LIMIT N`` of this;
                          the in-progress run is the ``status='running'`` row).
  ``hwc_cycle_samples``   a compact 30 s trace per cycle (full sensor set, nullable columns) so a
                          changed ``analyse`` methodology can recompute from raw long after
                          ``rp_raw`` would have aged the data out.

The daemon is the sole writer (WAL mode); analysis scripts open read-only. Helpers here are pure
I/O — the thermal/COP math lives in ``hwc_cop_analysis.cycle_metrics``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

# One summary row per cycle. ``start_ts`` (epoch UTC seconds, compressor-on edge) is the identity:
# the daemon upserts the same row from 'running' to 'complete'.
CYCLES_DDL = """
CREATE TABLE IF NOT EXISTS hwc_cycles (
    start_ts          REAL PRIMARY KEY,
    start_local       TEXT,
    end_ts            REAL,
    dur_min           INTEGER,
    tank_start        REAL,
    tank_end          REAL,
    ambient           REAL,
    wet_bulb          REAL,
    elec_kwh          REAL,
    elec_source       TEXT,
    therm_kwh         REAL,
    cop               REAL,
    hp_mean_w         REAL,
    hp_p95_w          REAL,
    probe_lag_min     REAL,
    probe_rise_10_min REAL,
    probe_rise_50_min REAL,
    probe_rise_90_min REAL,
    exhaust_start     REAL,
    exhaust_max       REAL,
    exhaust_end       REAL,
    coil_mean         REAL,
    return_air_mean   REAL,
    inlet_mean        REAL,
    element_on        INTEGER,
    defrost_on        INTEGER,
    four_way_on       INTEGER,
    clean             INTEGER,
    status            TEXT,
    updated_at        REAL
)
"""

# Per-cycle 30 s trace. Nullable sensor columns so a temporarily-absent probe never blocks a row;
# ``ts`` is on the same epoch-UTC grid as the cycle's ``start_ts``.
SAMPLES_DDL = """
CREATE TABLE IF NOT EXISTS hwc_cycle_samples (
    cycle_start_ts REAL NOT NULL,
    ts             REAL NOT NULL,
    tank           REAL,
    power_w        REAL,
    energy_kwh     REAL,
    ambient        REAL,
    humidity       REAL,
    element        INTEGER,
    defrost        INTEGER,
    four_way       INTEGER,
    exhaust        REAL,
    coil           REAL,
    return_air     REAL,
    inlet          REAL,
    PRIMARY KEY (cycle_start_ts, ts)
)
"""

# Columns the store persists for a summary row, in declaration order. Writers pass a dict keyed by
# these names; missing keys are stored NULL so a caller need not supply every diagnostic.
CYCLE_COLS = [
    "start_ts", "start_local", "end_ts", "dur_min",
    "tank_start", "tank_end", "ambient", "wet_bulb",
    "elec_kwh", "elec_source", "therm_kwh", "cop",
    "hp_mean_w", "hp_p95_w",
    "probe_lag_min", "probe_rise_10_min", "probe_rise_50_min", "probe_rise_90_min",
    "exhaust_start", "exhaust_max", "exhaust_end",
    "coil_mean", "return_air_mean", "inlet_mean",
    "element_on", "defrost_on", "four_way_on",
    "clean", "status", "updated_at",
]

SAMPLE_COLS = [
    "cycle_start_ts", "ts", "tank", "power_w", "energy_kwh", "ambient", "humidity",
    "element", "defrost", "four_way", "exhaust", "coil", "return_air", "inlet",
]

# Columns stored as 0/1 integers; a writer may pass python bools.
_BOOL_COLS = {
    "element_on", "defrost_on", "four_way_on", "clean",
    "element", "defrost", "four_way",
}


def connect(path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the store. Writers get WAL + a created schema; readers open the file URI read-only.

    A read-only open never creates or migrates the file, so an analysis script can't accidentally
    write to the daemon's database; it raises if the file is missing.
    """
    path = Path(path)
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(CYCLES_DDL)
    conn.execute(SAMPLES_DDL)
    conn.commit()


def _coerce(col: str, value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if col in _BOOL_COLS:
        return int(bool(value))
    if hasattr(value, "item"):  # numpy scalar → python scalar
        value = value.item()
    return value


def upsert_cycle(conn: sqlite3.Connection, cycle: dict) -> None:
    """Insert or replace one summary row, keyed by ``start_ts``.

    The daemon writes the row first with ``status='running'`` (start fields only) and again with
    ``status='complete'`` once the off-edge is seen; both are the same ``start_ts`` so the second
    overwrites the first. Unknown keys in ``cycle`` are ignored; absent columns store NULL.
    """
    if cycle.get("start_ts") is None:
        raise ValueError("cycle row requires start_ts")
    values = [_coerce(col, cycle.get(col)) for col in CYCLE_COLS]
    placeholders = ", ".join("?" for _ in CYCLE_COLS)
    conn.execute(
        f"INSERT OR REPLACE INTO hwc_cycles ({', '.join(CYCLE_COLS)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()


def append_samples(conn: sqlite3.Connection, cycle_start_ts: float, samples) -> int:
    """Append trace rows for a cycle (idempotent on ``(cycle_start_ts, ts)``).

    ``samples`` is an iterable of dicts (or a DataFrame) carrying any subset of the sensor columns
    plus ``ts``; ``cycle_start_ts`` is injected so a caller need not repeat it per row. Returns the
    number of rows written.
    """
    if isinstance(samples, pd.DataFrame):
        samples = samples.to_dict("records")
    rows = []
    for s in samples:
        row = dict(s)
        row["cycle_start_ts"] = cycle_start_ts
        if row.get("ts") is None:
            raise ValueError("sample row requires ts")
        rows.append([_coerce(col, row.get(col)) for col in SAMPLE_COLS])
    if not rows:
        return 0
    placeholders = ", ".join("?" for _ in SAMPLE_COLS)
    conn.executemany(
        f"INSERT OR REPLACE INTO hwc_cycle_samples ({', '.join(SAMPLE_COLS)}) "
        f"VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    return len(rows)


def recent_cycles(conn: sqlite3.Connection, n: int, *, include_running: bool = True) -> list[dict]:
    """Up to ``n`` most recent cycles, newest first (the card ring).

    With ``include_running`` false, only finalised ('complete') rows are returned — used where an
    in-progress row would be noise.
    """
    where = "" if include_running else "WHERE status = 'complete'"
    cur = conn.execute(
        f"SELECT * FROM hwc_cycles {where} ORDER BY start_ts DESC LIMIT ?", (n,)
    )
    return [dict(row) for row in cur.fetchall()]


def get_running(conn: sqlite3.Connection) -> dict | None:
    """The current in-progress ('running') cycle row, or None.

    There should be at most one; if a crash left several, the most recent wins.
    """
    cur = conn.execute(
        "SELECT * FROM hwc_cycles WHERE status = 'running' ORDER BY start_ts DESC LIMIT 1"
    )
    row = cur.fetchone()
    return dict(row) if row else None


def get_cycle(conn: sqlite3.Connection, start_ts: float) -> dict | None:
    cur = conn.execute("SELECT * FROM hwc_cycles WHERE start_ts = ?", (start_ts,))
    row = cur.fetchone()
    return dict(row) if row else None


def delete_cycle(conn: sqlite3.Connection, start_ts: float) -> None:
    """Remove a cycle and its trace samples — used to discard a too-short run (e.g. a defrost
    flicker) that opened a 'running' row but never reached the minimum duration."""
    conn.execute("DELETE FROM hwc_cycle_samples WHERE cycle_start_ts = ?", (start_ts,))
    conn.execute("DELETE FROM hwc_cycles WHERE start_ts = ?", (start_ts,))
    conn.commit()


def load_trace(conn: sqlite3.Connection, cycle_start_ts: float) -> pd.DataFrame:
    """The 30 s trace for one cycle as a DataFrame indexed by a UTC ``DatetimeIndex``.

    Empty (with the sensor columns) when the cycle has no samples — e.g. a CSV-seeded historical
    cycle whose raw trace predates the store. ``cycle_metrics`` consumes this frame.
    """
    cur = conn.execute(
        "SELECT * FROM hwc_cycle_samples WHERE cycle_start_ts = ? ORDER BY ts", (cycle_start_ts,)
    )
    rows = [dict(r) for r in cur.fetchall()]
    cols = [c for c in SAMPLE_COLS if c != "cycle_start_ts"]
    if not rows:
        return pd.DataFrame(columns=cols).set_index(pd.DatetimeIndex([], tz="UTC", name="ts"))
    df = pd.DataFrame(rows)[["ts"] + [c for c in cols if c != "ts"]]
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    df.index.name = "ts"
    return df.drop(columns=["ts"])
