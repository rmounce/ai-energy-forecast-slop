#!/usr/bin/env python3
"""Pure helpers for the daemon's HWC per-cycle reporting (``sensor.hwc_cycles``).

The HWC daemon owns the I/O (reading the compressor/tank/energy entities, calling
``hwc_cop_analysis.analyse`` over the recent window, publishing to Home Assistant). This module
holds the side-effect-free state machine so it is unit-testable without a network:

  - ``advance_live`` tracks the in-progress cycle (running → cooldown) from polled compressor
    state, snapshotting the cumulative energy meter at the start so the live row can show
    ``energy_now − energy_start`` kWh.
  - ``records_from_analysis`` / ``merge_records`` turn the reused ``analyse`` output into the
    published ring buffer (deduped by local start time).
  - ``backfill_captured`` clears the live row once the just-finished cycle appears in the ring
    (analyse only sees a cycle after its post-window exists in InfluxDB).
  - ``build_payload`` renders the ``sensor.hwc_cycles`` state + attributes.

Design intent (docs/hwc_cycle_reporting.md): publish-only telemetry, fully firewalled from the
planner/executor — a failure here must never perturb actuation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import math

import pandas as pd

# Columns surfaced to HA (keep the attribute payload small).
PUBLISH_COLS = [
    "start", "dur_min", "tank_start", "tank_end", "ambient", "wet_bulb",
    "elec_kwh", "elec_source", "therm_kwh", "cop", "element_on", "defrost_on", "clean",
]


def _now_iso(now_ts: float) -> str:
    return datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat()


def _local_str(now_ts: float, tz_name: str) -> str:
    """Local 'YYYY-MM-DD HH:MM' — matches hwc_cop_analysis.format_cycles_for_output."""
    return datetime.fromtimestamp(now_ts, tz=ZoneInfo(tz_name)).strftime("%Y-%m-%d %H:%M")


def _local_str_to_ts(local_str: str, tz_name: str) -> float | None:
    try:
        dt = datetime.strptime(local_str, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=ZoneInfo(tz_name)).timestamp()


def advance_live(
    current: dict | None,
    *,
    raw_on: bool,
    now_ts: float,
    tank_c: float | None,
    energy_kwh: float | None,
) -> dict | None:
    """Update the in-progress cycle record from a polled compressor reading.

    State machine: ``None``/``cooldown`` + on → ``running`` (snapshot start temp/energy);
    ``running`` + off → ``cooldown`` (snapshot end temp/energy). A ``cooldown`` row is kept for
    display until ``backfill_captured`` clears it. Coarse 60 s polling is fine for a stats table;
    the precise boundaries come from the reused ``analyse`` window, not from these edges.
    """
    if raw_on:
        if current is None or current.get("status") == "cooldown":
            return {
                "status": "running",
                "start_ts": now_ts,
                "tank_start": tank_c,
                "energy_start": energy_kwh,
                "tank_now": tank_c,
                "energy_now": energy_kwh,
            }
        cur = dict(current)
        if tank_c is not None:
            cur["tank_now"] = tank_c
        if energy_kwh is not None:
            cur["energy_now"] = energy_kwh
        return cur
    if current is not None and current.get("status") == "running":
        cur = dict(current)
        cur["status"] = "cooldown"
        cur["ended_ts"] = now_ts
        if tank_c is not None:
            cur["tank_now"] = tank_c
        if energy_kwh is not None:
            cur["energy_now"] = energy_kwh
        return cur
    return current


def live_view(current: dict | None, now_ts: float, tz_name: str) -> dict | None:
    """Render the publishable in-progress row, or None when no cycle is active."""
    if not current:
        return None
    start_ts = current.get("start_ts")
    if start_ts is None:
        return None
    end_ts = current.get("ended_ts") or now_ts
    es, en = current.get("energy_start"), current.get("energy_now")
    elec = round(en - es, 3) if (es is not None and en is not None and en >= es) else None
    ts, tn = current.get("tank_start"), current.get("tank_now")
    dt_c = round(tn - ts, 1) if (ts is not None and tn is not None) else None
    return {
        "start": _local_str(start_ts, tz_name),
        "status": current.get("status"),
        "dur_min": round((end_ts - start_ts) / 60),
        "tank_start": ts,
        "tank_now": tn,
        "dt_c": dt_c,
        "elec_kwh": elec,
        "elec_source": "counter" if elec is not None else None,
    }


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


def records_from_analysis(df: pd.DataFrame) -> list[dict]:
    """Project an ``hwc_cop_analysis.analyse`` frame to JSON-safe published records.

    Accepts either a raw frame (datetime ``start``) or one already run through
    ``format_cycles_for_output`` (string ``start``); normalises ``start`` to the local
    'YYYY-MM-DD HH:MM' string either way so records dedupe consistently.
    """
    if df is None or df.empty:
        return []
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out["start"]):
        from hwc_cop_analysis import format_cycles_for_output

        out = format_cycles_for_output(out)
    cols = [c for c in PUBLISH_COLS if c in out.columns]
    records = []
    for _, row in out[cols].iterrows():
        records.append({c: _clean_value(row[c]) for c in cols})
    return records


def _start_dt(start: str) -> datetime | None:
    try:
        return datetime.strptime(start, "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None


def merge_records(
    existing: list[dict], new: list[dict], maxlen: int, tol_min: float = 3.0
) -> list[dict]:
    """Merge by local start time (new wins), sort ascending, keep the most recent ``maxlen``.

    Dedup is **tolerant** (``tol_min``), not exact-string: the same physical cycle can be analysed
    by two passes over different windows (per-cycle finalise vs the incremental cold-start
    re-scan), and ``analyse`` self-detects the compressor-on edge inside whichever window it is
    given, so the detected start can wobble by a resample bin (~1 min). Exact-string keying would
    leak twin rows one minute apart. ``tol_min`` (3 min) absorbs that wobble while staying well
    under the analyser's 5-min minimum cycle length, so two genuinely-distinct runs (starts always
    ≥5 min apart) can never be collapsed. This also self-heals any twins already in the ring.

    The later-processed record wins a near-match, so ``new`` overrides ``existing`` and, within the
    ring, the later twin survives — both are near-identical, so either is fine.
    """
    merged: list[dict] = []
    for rec in list(existing) + list(new):
        start = rec.get("start")
        if not start:
            continue
        dt = _start_dt(start)
        slot = None
        if dt is not None:
            for i, kept in enumerate(merged):
                kdt = _start_dt(kept.get("start", ""))
                if kdt is not None and abs((dt - kdt).total_seconds()) <= tol_min * 60:
                    slot = i
                    break
        if slot is None:
            merged.append(rec)
        else:
            merged[slot] = rec
    merged.sort(key=lambda r: r["start"])
    return merged[-maxlen:] if maxlen and len(merged) > maxlen else merged


def cold_start_since_ts(
    cycles: list[dict],
    now_ts: float,
    *,
    maxlen: int,
    seed_hours: float,
    incremental_margin_hours: float,
    tz_name: str,
) -> float:
    """Epoch ``since`` for the one-shot cold-start backfill.

    Steady-state capture is event-driven (per-cycle finalise), so the only blind scan is this
    single backfill when the process starts:

      - ring not yet full → a **deep** ``seed_hours`` window to populate the table;
      - ring already full → an **incremental** catch-up since the newest row it already holds (less
        a margin), which only covers cycles that completed while the daemon was down. Usually a few
        hours; bounded by how stale the ring is.
    """
    if len(cycles) < maxlen:
        return now_ts - seed_hours * 3600
    newest = max((_local_str_to_ts(c.get("start", ""), tz_name) or 0.0) for c in cycles)
    if newest <= 0:
        return now_ts - seed_hours * 3600
    return newest - incremental_margin_hours * 3600


def cooldown_settled(current: dict | None, now_ts: float, settle_seconds: float) -> bool:
    """True when a cooled-down cycle is old enough that its post-window exists in InfluxDB.

    ``analyse`` needs ~10 min of post-cycle samples for the baseline/standing-loss context, so a
    finished cycle can only be reconstructed once it has settled for ``settle_seconds``.
    """
    return bool(
        current
        and current.get("status") == "cooldown"
        and current.get("ended_ts") is not None
        and now_ts - current["ended_ts"] >= settle_seconds
    )


def cooldown_expired(current: dict | None, now_ts: float, giveup_seconds: float) -> bool:
    """True when a cooled-down cycle has gone uncaptured long enough to give up on (e.g. a run
    shorter than the analyser's ``min_minutes`` floor, which will never produce a row)."""
    return bool(
        current
        and current.get("status") == "cooldown"
        and current.get("ended_ts") is not None
        and now_ts - current["ended_ts"] >= giveup_seconds
    )


def backfill_captured(
    current: dict | None, cycles: list[dict], tz_name: str, tol_min: float = 6.0
) -> bool:
    """True once the cooled-down live cycle appears in the ring (so the live row can clear).

    Matches on start time within ``tol_min`` — the analyse window boundary and the polled
    on-edge differ by up to the poll interval plus the ~50 s compressor-sensor lag.
    """
    if not current or current.get("status") != "cooldown":
        return False
    start_ts = current.get("start_ts")
    if start_ts is None:
        return True
    for rec in cycles:
        rec_ts = _local_str_to_ts(rec.get("start", ""), tz_name)
        if rec_ts is not None and abs(rec_ts - start_ts) <= tol_min * 60:
            return True
    return False


def build_payload(
    cycles: list[dict], live: dict | None, *, today_local: str
) -> tuple[object, dict]:
    """Return (state_scalar, attributes) for ``sensor.hwc_cycles``.

    State is the last completed cycle's COP (the headline at-a-glance number), 'unknown' until the
    first cycle closes; its ``clean``/``element_on`` flags travel in the row to explain a low value.
    """
    # Headline = most recent *computable* COP, not literally cycles[-1]: a run can land with a
    # null COP (e.g. a tank-probe gap at the cycle start → NaN thermal), and "unknown" then hides
    # an otherwise healthy history. Skip such rows for the at-a-glance number.
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
