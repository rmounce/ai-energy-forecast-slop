#!/usr/bin/env python3
"""One-time seed of the HWC SQLite store from InfluxDB + the CSV anchor history.

THROWAWAY migration tool (docs/hwc_local_store.md). After it has run once the daemon records
forward and InfluxDB is never read for HWC again; ``hwc_cop_analysis`` is repointed to the store.
This script is the *only* place the InfluxDB HWC extraction survives, and it deliberately leans on
``hwc_cop_analysis``'s current low-level fetch helpers — so run it **before** those helpers are
removed from the live tool.

What it does:

  1. ``--csv``  backfill ``hwc_cycles`` from ``data/hwc_cop_cycles.csv`` for cycles older than the
     InfluxDB window (the install-period runs ``rp_raw`` has already aged out). Summary only — no
     trace is recoverable for those.
  2. ``--influx`` for the last ``--influx-days`` (≤30, inside ``rp_raw`` retention): rebuild the
     30 s grid exactly as ``analyse`` does, slice a per-cycle trace into ``hwc_cycle_samples``, and
     write the summary via the shared ``cycle_metrics`` — so the trace table isn't born empty and
     every seeded summary comes from the same brain the daemon will use.
  3. ``--parity`` compare those ``cycle_metrics`` summaries against the legacy InfluxDB-direct
     ``analyse`` output, cycle-by-cycle. This is the no-divergence proof: target max|ΔCOP| ≈ 0.

Live InfluxDB access needs the docker socket (run outside the sandbox) and credentials from
config; ``nice -n 19`` it.

Both phases run by default; use ``--no-influx`` / ``--no-csv`` to skip one.

Usage:
  python scripts/seed_hwc_store.py --parity
  python scripts/seed_hwc_store.py --influx-days 28 --db data/hwc_cycles.sqlite
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import hwc_cop_analysis as hca  # noqa: E402
import hwc_cycle_store as store  # noqa: E402

DEFAULT_DB = "data/hwc_cycles.sqlite"
DEFAULT_CSV = "data/hwc_cop_cycles.csv"
# Trace columns, in the order hwc_cycle_store expects (minus the injected cycle_start_ts/ts).
TRACE_COLS = [
    "tank", "power_w", "energy_kwh", "ambient", "humidity",
    "element", "defrost", "four_way", "exhaust", "coil", "return_air", "inlet",
]
# Summary columns the CSV anchor history can supply (older format: no elec_source; baseline_w /
# power_source are dropped in the store).
CSV_SUMMARY_COLS = [
    "dur_min", "tank_start", "tank_end", "ambient", "wet_bulb", "hp_mean_w", "hp_p95_w",
    "elec_kwh", "therm_kwh", "cop", "clean", "probe_lag_min", "probe_rise_10_min",
    "probe_rise_50_min", "probe_rise_90_min", "exhaust_start", "exhaust_max", "exhaust_end",
    "coil_mean", "return_air_mean", "inlet_mean", "element_on", "defrost_on", "four_way_on",
]


def _build_grid(c, since, until) -> pd.DataFrame:
    """The 30 s grid analyse builds, plus the extra columns the trace needs (power/energy/ambient/
    humidity gridded too). Cycle detection over this grid yields the same boundaries as analyse, so
    a trace sliced here is apples-to-apples with the legacy summary."""
    comp = hca._series(c, "binary_sensor__running", "aquatech_compressor", since=since, until=until)
    power = hca._series(c, "sensor__power", hca.HWC_POWER_ENTITY, since=since, until=until)
    energy = hca._series(c, "sensor__energy", hca.HWC_ENERGY_ENTITY, since=since, until=until)
    tank = hca._series_anchored(c, "sensor__temperature", "heat_pump_temperature",
                                since=since, until=until)
    amb = hca._series_anchored(c, "sensor__temperature", "aquatech_temperature",
                               since=since, until=until)
    hum = hca._series_anchored(c, "humidity_adelaide", since=since, until=until,
                               field="mean_value", rp="rp_30m")
    exhaust = hca._series(c, "sensor__temperature", "aquatech_exhaust_temperature",
                          since=since, until=until)
    coil = hca._series(c, "sensor__temperature", "aquatech_coil_temperature",
                       since=since, until=until)
    return_air = hca._series(c, "sensor__temperature", "aquatech_return_air_temperature",
                             since=since, until=until)
    inlet = hca._series(c, "sensor__temperature", "aquatech_inlet_temperature",
                        since=since, until=until)
    element = hca._series(c, "binary_sensor__running", "aquatech_element", since=since, until=until)
    defrost = hca._series(c, "binary_sensor__running", "aquatech_defrost", since=since, until=until)
    four_way = hca._series(c, "binary_sensor__running", "aquatech_four_way_valve",
                           since=since, until=until)
    if comp.empty or power.empty:
        raise SystemExit("No compressor/power data in the requested InfluxDB window")

    idx_min = min(comp.index.min(), power.index.min())
    idx_max = max(comp.index.max(), power.index.max())
    idx = pd.date_range(idx_min, idx_max, freq="30s", tz="UTC")
    return pd.DataFrame(
        {
            "on": hca._state_to_idx(comp, idx),
            "tank": hca._interp_to_idx(tank, idx),
            "power_w": hca._interp_to_idx(power, idx),
            "energy_kwh": hca._interp_to_idx(energy, idx),
            "ambient": hca._interp_to_idx(amb, idx),
            "humidity": hca._interp_to_idx(hum, idx),
            "element": hca._state_to_idx(element, idx),
            "defrost": hca._state_to_idx(defrost, idx),
            "four_way": hca._state_to_idx(four_way, idx),
            "exhaust": hca._interp_to_idx(exhaust, idx),
            "coil": hca._interp_to_idx(coil, idx),
            "return_air": hca._interp_to_idx(return_air, idx),
            "inlet": hca._interp_to_idx(inlet, idx),
        },
        index=idx,
    )


def iter_cycles(grid: pd.DataFrame, min_minutes: int):
    """Yield (cs, ce, trace) for each compressor-on run at least ``min_minutes`` long — the same
    on-mask groupby analyse uses, so boundaries match."""
    on = grid["on"]
    grp = (on != on.shift()).cumsum()
    for _, g in on.groupby(grp):
        if not g.iloc[0] or len(g) < min_minutes * 2:  # 30 s steps
            continue
        cs, ce = g.index[0], g.index[-1]
        yield cs, ce, grid.loc[cs:ce, TRACE_COLS]


def _trace_samples(trace: pd.DataFrame) -> list[dict]:
    rows = []
    for ts, r in trace.iterrows():
        row = {col: r.get(col) for col in TRACE_COLS}
        row["ts"] = ts.timestamp()
        rows.append(row)
    return rows


def seed_influx(conn, since, until, min_minutes) -> tuple[int, list[dict]]:
    """Write summaries + traces for the recent InfluxDB window; return (n, cycle_metrics rows)."""
    c = hca._client()
    grid = _build_grid(c, since, until)
    now = time.time()
    metrics_rows = []
    for cs, ce, trace in iter_cycles(grid, min_minutes):
        m = hca.cycle_metrics(trace)
        if m is None:
            continue
        store.upsert_cycle(conn, {**m, "status": "complete", "updated_at": now})
        store.append_samples(conn, m["start_ts"], _trace_samples(trace))
        metrics_rows.append(m)
    return len(metrics_rows), metrics_rows


def seed_csv(conn, csv_path: Path, before_ts: float, tz_name: str) -> int:
    """Backfill summary-only rows from the CSV anchor history for cycles before ``before_ts``."""
    if not csv_path.exists():
        print(f"  (no CSV at {csv_path}; skipping)")
        return 0
    df = pd.read_csv(csv_path)
    tz = ZoneInfo(tz_name)
    now = time.time()
    n = 0
    for _, row in df.iterrows():
        start_local = str(row["start"])
        try:
            dt = datetime.strptime(start_local, "%Y-%m-%d %H:%M").replace(tzinfo=tz)
        except (TypeError, ValueError):
            continue
        start_ts = dt.timestamp()
        if start_ts >= before_ts:  # in/after the InfluxDB window → take the traced version instead
            continue
        summary = {"start_ts": start_ts, "start_local": start_local,
                   "end_ts": None, "elec_source": None, "status": "complete", "updated_at": now}
        for col in CSV_SUMMARY_COLS:
            if col in df.columns:
                summary[col] = row[col]
        store.upsert_cycle(conn, summary)
        n += 1
    return n


def report_parity(conn, since, until, min_minutes) -> None:
    """Compare the seeded store's cycle_metrics summaries against legacy InfluxDB-direct analyse.

    Reads the trace-based summaries straight from the store (so it can run standalone over a short
    window for a quick check), and re-derives the legacy numbers over the same window.
    """
    legacy = hca.analyse(since=since, until=until, min_minutes=min_minutes)
    if legacy.empty:
        print("  parity: legacy analyse returned no cycles")
        return
    legacy = legacy.assign(_ts=legacy["start"].map(lambda s: pd.Timestamp(s).timestamp()))
    lo_ts, hi_ts = pd.Timestamp(since).timestamp(), pd.Timestamp(until).timestamp()
    seeded = [c for c in store.recent_cycles(conn, 100000)
              if lo_ts <= c["start_ts"] <= hi_ts and c.get("end_ts") is not None]
    deltas, matched, unmatched = [], 0, 0
    for m in seeded:
        near = legacy[(legacy["_ts"] - m["start_ts"]).abs() <= 90]
        if near.empty:
            unmatched += 1
            continue
        matched += 1
        lo = near.iloc[0]
        if pd.notna(lo["cop"]) and m["cop"] is not None:
            deltas.append((m["start_local"], float(lo["cop"]), float(m["cop"]),
                           abs(float(lo["cop"]) - float(m["cop"]))))
    print(f"  parity: matched {matched}/{len(seeded)} (unmatched {unmatched})")
    if deltas:
        worst = max(deltas, key=lambda d: d[3])
        print(f"  parity: max |ΔCOP| = {worst[3]:.3f} at {worst[0]} "
              f"(legacy {worst[1]:.2f} vs cycle_metrics {worst[2]:.2f})")
        for local, lcop, ncop, d in sorted(deltas, key=lambda d: -d[3])[:5]:
            if d > 0.02:
                print(f"    {local}: legacy {lcop:.2f} vs metrics {ncop:.2f}  Δ{d:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--csv", dest="csv_path", default=DEFAULT_CSV)
    ap.add_argument("--influx-days", type=int, default=29,
                    help="Recent window to read from InfluxDB (≤30; rp_raw retains 30 days)")
    ap.add_argument("--min-minutes", type=int, default=5)
    ap.add_argument("--no-influx", action="store_true", help="Skip the InfluxDB window")
    ap.add_argument("--no-csv", action="store_true", help="Skip the CSV backfill")
    ap.add_argument("--parity", action="store_true",
                    help="Compare cycle_metrics vs legacy analyse over the InfluxDB window")
    ap.add_argument("--tz", default=hca.LOCAL_TZ)
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    conn = store.connect(db_path)

    since = datetime.now(timezone.utc) - timedelta(days=args.influx_days)
    until = datetime.now(timezone.utc)

    if not args.no_influx:
        print(f"Seeding InfluxDB window: since {since.isoformat()} (≤{args.influx_days}d)")
        n, _ = seed_influx(conn, since, until, args.min_minutes)
        print(f"  wrote {n} cycles (summary + 30 s trace)")

    if not args.no_csv:
        print(f"Backfilling CSV anchors older than the InfluxDB window from {args.csv_path}")
        n = seed_csv(conn, Path(args.csv_path), since.timestamp(), args.tz)
        print(f"  wrote {n} summary-only cycles")

    if args.parity:
        print("Parity check (seeded cycle_metrics vs legacy analyse):")
        report_parity(conn, since, until, args.min_minutes)

    total = len(store.recent_cycles(conn, 100000))
    print(f"Store now holds {total} cycles at {db_path}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
