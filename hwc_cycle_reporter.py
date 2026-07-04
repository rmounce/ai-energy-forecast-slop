#!/usr/bin/env python3
"""Pure helpers for the daemon's HWC per-cycle reporting (``sensor.hwc_cycles``).

The daemon records every cycle live to the SQLite store (``hwc_cycle_store``): a 30 s trace plus
precise compressor-edge snapshots, finalised through ``hwc_cop_analysis.cycle_metrics``. This
module holds the side-effect-free projection from store rows to the published HA payload, so it is
unit-testable without a database or network:

  - ``record_from_store_row`` projects a completed ``hwc_cycles`` row to a compact published record.
  - ``live_record`` renders the in-progress ('running') cycle from the daemon's edge snapshot plus
    the latest cached tank/energy readings.
  - ``build_payload`` assembles the ``sensor.hwc_cycles`` state + attributes.

Design intent (docs/hwc/cycle_reporting.md, docs/hwc/local_store.md): publish-only telemetry, fully
firewalled from the planner/executor — a failure here must never perturb actuation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Columns surfaced to HA per completed cycle (keep the attribute payload small).
PUBLISH_COLS = [
    "start", "dur_min", "tank_start", "tank_end", "ambient", "wet_bulb",
    "elec_kwh", "elec_source", "therm_kwh", "cop", "element_on", "defrost_on", "fan_high_on",
    "clean",
]
_BOOL_COLS = {"element_on", "defrost_on", "fan_high_on", "clean"}


def _local_str(now_ts: float, tz_name: str) -> str:
    """Local 'YYYY-MM-DD HH:MM' — matches hwc_cop_analysis.format_cycles_for_output."""
    return datetime.fromtimestamp(now_ts, tz=ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M")


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar → python scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def record_from_store_row(row: dict) -> dict:
    """Project a completed ``hwc_cycles`` row to a JSON-safe published record.

    ``start`` is the stored local string (``start_local``); 0/1 flag columns are restored to bools.
    """
    out = {"start": row.get("start_local")}
    for col in PUBLISH_COLS[1:]:
        value = _clean_value(row.get(col))
        if col in _BOOL_COLS:
            out[col] = bool(value) if value is not None else None
        else:
            out[col] = value
    return out


def records_from_store_rows(rows: list[dict]) -> list[dict]:
    """Project store rows (any order) to published records, oldest-first for ``build_payload``."""
    records = [record_from_store_row(r) for r in rows]
    records.sort(key=lambda r: r.get("start") or "")
    return records


def live_record(
    reporter_cycle: dict | None,
    *,
    tank_now: float | None,
    energy_now: float | None,
    fan_high: bool | None = None,
    now_ts: float,
    tz_name: str,
) -> dict | None:
    """Render the in-progress row from the daemon's edge snapshot + latest cached readings.

    ``reporter_cycle`` is the daemon's in-memory open cycle ``{start_ts, tank_start, energy_start}``
    (persisted across restarts). Elec is ``energy_now − energy_start`` from the dedicated counter;
    ΔT is ``tank_now − tank_start``. ``fan_high`` is the current cached fan-speed reading (True/False),
    not yet classified over the whole cycle the way ``fan_high_on`` is for a completed row. Returns
    None when no cycle is open.
    """
    if not reporter_cycle:
        return None
    start_ts = reporter_cycle.get("start_ts")
    if start_ts is None:
        return None
    es, ts = reporter_cycle.get("energy_start"), reporter_cycle.get("tank_start")
    elec = (
        round(energy_now - es, 3)
        if (es is not None and energy_now is not None and energy_now >= es)
        else None
    )
    dt_c = round(tank_now - ts, 1) if (ts is not None and tank_now is not None) else None
    return {
        "start": _local_str(start_ts, tz_name),
        "status": "running",
        "dur_min": round((now_ts - start_ts) / 60),
        "tank_start": ts,
        "tank_now": tank_now,
        "dt_c": dt_c,
        "elec_kwh": elec,
        "elec_source": "counter" if elec is not None else None,
        "fan_high": fan_high,
    }


def build_payload(
    cycles: list[dict], live: dict | None, *, today_local: str
) -> tuple[object, dict]:
    """Return (state_scalar, attributes) for ``sensor.hwc_cycles``.

    ``cycles`` is oldest-first published records. State is the most recent *computable* COP (a run
    can land with a null COP — e.g. a tank-probe gap — and 'unknown' would then hide an otherwise
    healthy history), 'unknown' until the first cycle closes.
    """
    last_cop = next((c.get("cop") for c in reversed(cycles) if c.get("cop") is not None), None)
    clean_cops = [c.get("cop") for c in cycles if c.get("clean") and c.get("cop") is not None]
    attributes = {
        "cycles": list(reversed(cycles)),  # most recent first for the card
        "current": live,
        "cycle_count": len(cycles),
        "cycles_today": sum(1 for c in cycles if str(c.get("start", "")).startswith(today_local)),
        "last_clean_cop": clean_cops[-1] if clean_cops else None,
        "unit_of_measurement": "COP",
    }
    return (last_cop if last_cop is not None else "unknown"), attributes
