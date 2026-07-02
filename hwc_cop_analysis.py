#!/usr/bin/env python3
"""Measure heat-pump hot water (HWC) per-cycle COP from InfluxDB.

The preferred electrical input is the dedicated heat-pump circuit's cumulative
energy meter (Athom channel 2, ``energy_2``), differenced over the cycle — no
baseline subtraction or sampling error. It falls back to integrating the channel-2
power (and, for older history, baseline-subtracted ``sensor.remaining_power_load``)
when the counter is missing or has reset. The chosen source is reported per cycle as
``elec_source``. This sweeps recent compressor cycles and reports, per cycle:

  start/end tank temp, ambient, duration, electrical-in (baseline-subtracted),
  thermal-out (single-probe ΔT + standing loss), apparent COP, and a cleanliness
  flag (so contaminated windows aren't over-trusted).

Caveats (see docs/hwc/thermal_characterisation.md):
  - Thermal-out uses the single tank probe; the tank stratifies, so this is
    approximate. The hard COP ceiling (elec vs 45→target sensible capacity) is
    more robust than the point estimate.
  - COP is specific to the current fan-speed setting and to the target temp.

Usage:
  python hwc_cop_analysis.py
  python hwc_cop_analysis.py --days 3 --csv data/hwc_cop_cycles.csv
  python hwc_cop_analysis.py --since 2026-06-03 --merge-existing
  python hwc_cop_analysis.py --summary-md docs/hwc/calibration_cycles.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

import hwc_cycle_store as store
from config_utils import load_config

# Durable HWC system-of-record (docs/hwc/local_store.md); ``analyse`` reads it, the daemon writes it.
DEFAULT_DB_PATH = "data/hwc_cycles.sqlite"

TANK_LITRES = 225
C_WATER = 4.186  # kJ/kg·K
STANDING_LOSS_KW = 0.12
TANK_SETTLE_SECONDS = 60  # tank probe keeps rising a few s past compressor-off; final temp settles here
HP_POWER_MAX_W = 1100  # plausible upper bound for this unit; above → contamination
BASELINE_DRIFT_MAX_W = 80  # pre/post off-state baseline mismatch tolerated on the integration path
COP_CLEAN_MIN = 0.8        # below → broken/standby-dominated estimate
COP_CLEAN_MAX = 3.3        # above → contamination (elec too low) / stratification-inflated thermal
LOCAL_TZ = "Australia/Adelaide"
DEFAULT_SINCE = "2026-05-28"  # Aquatech install date; earlier HA history is unrelated.
HWC_POWER_ENTITY = (
    "athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2"
)
HWC_ENERGY_ENTITY = (
    "athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_energy_2"
)
RESIDUAL_POWER_ENTITY = "remaining_power_load"
# Reject a cumulative-counter cycle delta above this (kWh): a plausible single reheat is
# ~2 kWh, so anything beyond this is a meter rollover/glitch → fall back to power integration.
COUNTER_MAX_CYCLE_KWH = 6.0


def _client():
    ic = load_config()["influxdb"]
    return InfluxDBClient(
        host=ic["host"], port=ic["port"], username=ic["username"],
        password=ic["password"], database=ic["database"],
    )


def _format_influx_time(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TZ)
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _series_query(
    meas,
    eid=None,
    days=None,
    since=DEFAULT_SINCE,
    until=None,
    field="value",
    rp="",
):
    src = f'"{rp}"."{meas}"' if rp else f'"{meas}"'
    where = []
    if eid:
        where.append(f"entity_id='{eid}'")
    if since:
        where.append(f"time >= '{_format_influx_time(since)}'")
    if until:
        where.append(f"time <= '{_format_influx_time(until)}'")
    if days is not None:
        where.append(f"time> now()-{days}d")
    if not where:
        raise ValueError("Refusing to query unbounded time range")
    return f"SELECT \"{field}\" FROM {src} WHERE {' AND '.join(where)}"


def _series(
    c,
    meas,
    eid=None,
    days=None,
    since=DEFAULT_SINCE,
    until=None,
    field="value",
    rp="",
):
    q = _series_query(
        meas, eid=eid, days=days, since=since, until=until, field=field, rp=rp
    )
    pts = list(c.query(q).get_points())
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(p["time"]): float(p[field])
                   for p in pts if p.get(field) is not None}).sort_index()
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
    return s[~s.index.duplicated(keep="last")]


def _anchor_query(meas, eid=None, since=None, field="value", rp=""):
    src = f'"{rp}"."{meas}"' if rp else f'"{meas}"'
    where = [f"entity_id='{eid}'"] if eid else []
    where.append(f"time < '{_format_influx_time(since)}'")
    return f"SELECT \"{field}\" FROM {src} WHERE {' AND '.join(where)} ORDER BY time DESC LIMIT 1"


def _anchor_before(c, meas, eid=None, since=None, field="value", rp=""):
    """The single most-recent sample strictly before ``since`` (a 1-point Series), or empty.

    A sparse, on-change series (tank/ambient/humidity) may have no sample inside a narrow analyse
    window when that window opens in a reporting gap. Seeding the series with its last-known prior
    value lets a boundary read resolve to the real reading instead of a leading-NaN — a cheap point
    query (``ORDER BY time DESC LIMIT 1``) rather than widening the scan.
    """
    if since is None:
        return pd.Series(dtype=float)
    pts = list(c.query(_anchor_query(meas, eid=eid, since=since, field=field, rp=rp)).get_points())
    s = pd.Series({pd.to_datetime(p["time"]): float(p[field])
                   for p in pts if p.get(field) is not None})
    if s.empty:
        return s
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
    return s


def _series_anchored(c, meas, eid=None, days=None, since=DEFAULT_SINCE, until=None,
                     field="value", rp=""):
    """``_series`` for a sparse input, plus a left-anchor so boundary reads never hit a leading gap.

    See ``_anchor_before``: the extra point is the last value before ``since``, prepended so an
    ``asof`` (or interpolation) at the window's start resolves to a real reading.
    """
    s = _series(c, meas, eid=eid, days=days, since=since, until=until, field=field, rp=rp)
    anchor = _anchor_before(c, meas, eid=eid, since=since, field=field, rp=rp)
    if anchor.empty:
        return s
    combined = pd.concat([anchor, s])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


def _interp_to_idx(s, idx):
    if s.empty:
        return pd.Series(index=idx, dtype=float)
    return s.reindex(idx.union(s.index)).interpolate("time").reindex(idx)


def _state_to_idx(s, idx):
    if s.empty:
        return pd.Series(False, index=idx)
    return s.reindex(idx, method="ffill").fillna(0) > 0.5


def _series_has_window(s: pd.Series, start, end) -> bool:
    if s.empty:
        return False
    return not s[(s.index >= start) & (s.index <= end)].dropna().empty


def _round_or_nan(value, ndigits=1):
    return round(value, ndigits) if pd.notna(value) else np.nan


def cycle_is_clean(b_pre, b_post, hp_p95_w, cop, elec_source) -> bool:
    """Whether a cycle is trustworthy enough to use as a calibration anchor.

    The baseline-drift term (``|b_pre - b_post|``) is a contamination proxy that *only* matters on
    the ``power_integration`` path, where elec is the baseline-subtracted power integral. For
    ``counter`` cycles elec comes straight from differencing the dedicated ``energy_2`` meter, so
    the off-state baseline never feeds the COP — and ``b_pre`` is unreliable anyway, since the
    compressor-on edge (from the laggy ``aquatech_compressor`` sensor) lags the real power ramp and
    the pre-window often catches spin-up. So the drift gate is applied only when it's relevant.

    The peak-power and COP-band terms always apply: they catch element-assist / mis-attribution and
    stratification-inflated (or broken) thermal estimates regardless of the elec source.
    """
    baseline_ok = elec_source != "power_integration" or abs(b_pre - b_post) < BASELINE_DRIFT_MAX_W
    return bool(
        baseline_ok
        and hp_p95_w < HP_POWER_MAX_W
        and pd.notna(cop)
        and COP_CLEAN_MIN < cop < COP_CLEAN_MAX
    )


def _first_rise_minutes(series: pd.Series, start, start_temp: float, end_temp: float, fraction: float):
    lift = end_temp - start_temp
    if lift <= 0:
        return np.nan
    threshold = start_temp + lift * fraction
    reached = series[series >= threshold]
    if reached.empty:
        return np.nan
    return (reached.index[0] - start).total_seconds() / 60


def counter_cycle_kwh(energy_cumulative: pd.Series, cs, ce):
    """kWh drawn over [cs, ce] from a cumulative (monotonic) energy meter; None if unusable.

    Returns None — so the caller falls back to power integration — when the counter is missing,
    has fewer than two in-window samples, or the delta is negative (a mid-cycle meter reset) or
    implausibly large (rollover/glitch above ``COUNTER_MAX_CYCLE_KWH``). The counter measures the
    whole heat-pump circuit, so it correctly includes any resistive-element assist.
    """
    if energy_cumulative is None or energy_cumulative.empty:
        return None
    seg = energy_cumulative[
        (energy_cumulative.index >= cs) & (energy_cumulative.index <= ce)
    ].dropna()
    if len(seg) < 2:
        return None
    delta = float(seg.iloc[-1] - seg.iloc[0])
    if delta < 0 or delta > COUNTER_MAX_CYCLE_KWH:
        return None
    return delta


def stull_wet_bulb(t, rh):
    if pd.isna(t) or pd.isna(rh):
        return np.nan
    rh = max(1.0, min(100.0, rh))
    return (t * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + math.atan(t + rh)
            - math.atan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035)


def _counter_kwh_from(edges, energy_series, cs, ce):
    """Cycle elec from the cumulative counter — precise edge snapshots if the daemon supplied them,
    else differenced over [cs, ce] from the trace. None (→ power-integration fallback) when missing,
    negative (mid-cycle reset) or implausibly large (rollover, see ``COUNTER_MAX_CYCLE_KWH``)."""
    if edges and edges.get("energy_start") is not None and edges.get("energy_end") is not None:
        delta = float(edges["energy_end"]) - float(edges["energy_start"])
        if delta < 0 or delta > COUNTER_MAX_CYCLE_KWH:
            return None
        return delta
    return counter_cycle_kwh(energy_series, cs, ce)


def _trace_col(trace, name):
    """A float Series for a (possibly absent) trace column, aligned to the trace index.

    Absent → all-NaN (never an empty Series), so a full-length boolean cycle mask always applies
    cleanly and a missing sensor degrades to NaN rather than raising.
    """
    if name not in trace.columns:
        return pd.Series(np.nan, index=trace.index, dtype=float)
    return pd.to_numeric(trace[name], errors="coerce")


def cycle_metrics(trace, *, edges=None, sample_seconds=30, standing_loss_kw=STANDING_LOSS_KW,
                  tank_settle_seconds=TANK_SETTLE_SECONDS, tz=LOCAL_TZ):
    """Canonical per-cycle summary from a raw 30 s trace — the one shared brain (docs/hwc/local_store.md).

    Both feeders call this so they can never diverge: the daemon accumulates a live trace + precise
    compressor-edge snapshots and calls it on the off-edge; ``hwc_cop_analysis`` loads a stored trace
    and calls the *same* function. The InfluxDB-era ``analyse`` loop is retired (its dual power-source
    selection and ±10 min baseline machinery survive, frozen, only in the one-time seed script).

    ``trace`` is a DataFrame indexed by a UTC ``DatetimeIndex`` (30 s grid) with any subset of the
    sensor columns (tank, power_w, energy_kwh, ambient, humidity, element, defrost, four_way,
    exhaust, coil, return_air, inlet); absent columns degrade to NaN/False, never an error.

    ``edges`` (optional) carries the daemon's precise snapshots — ``cs``/``ce`` (UTC Timestamps),
    ``tank_start``/``tank_end``, ``energy_start``/``energy_end`` — which override the trace-derived
    boundaries. Offline, boundaries come from the trace itself.

    Post-meter simplification (see doc): there is **no baseline subtraction** — the dedicated meter's
    standby is single-digit watts, so ``hp_mean/p95`` are raw cycle power, the ``power_integration``
    fallback integrates ``power_w`` directly, and the baseline-drift term in ``cycle_is_clean`` is
    inert (``b_pre=b_post=0``).

    Returns the rich row dict (a superset of the stored ``CYCLE_COLS``), or ``None`` when the trace
    is empty / has zero duration / lacks a usable boundary tank temperature.
    """
    if trace is None or trace.empty:
        return None
    idx = trace.index

    if edges and edges.get("cs") is not None and edges.get("ce") is not None:
        cs, ce = pd.Timestamp(edges["cs"]), pd.Timestamp(edges["ce"])
    else:
        cs, ce = idx[0], idx[-1]
    dur_h = (ce - cs).total_seconds() / 3600
    if dur_h <= 0:
        return None

    cyc_mask = (idx >= cs) & (idx <= ce)
    if not cyc_mask.any():
        return None

    # Power stats and the integration fallback — raw cycle power, no baseline (post-meter).
    P = _trace_col(trace, "power_w")[cyc_mask].dropna().clip(lower=0)
    hp_mean = P.mean() if not P.empty else np.nan
    hp_p95 = P.quantile(0.95) if not P.empty else np.nan
    integrated_kwh = P.sum() * (sample_seconds / 3600) / 1000 if not P.empty else np.nan

    counter_kwh = _counter_kwh_from(edges, _trace_col(trace, "energy_kwh"), cs, ce)
    if counter_kwh is not None:
        elec, elec_source = counter_kwh, "counter"
    else:
        elec, elec_source = integrated_kwh, "power_integration"

    tank = _trace_col(trace, "tank")
    if edges and edges.get("tank_start") is not None and edges.get("tank_end") is not None:
        t_start, t_end = float(edges["tank_start"]), float(edges["tank_end"])
    else:
        t0 = tank[tank.index <= cs + pd.Timedelta("90s")].dropna()
        # The probe keeps climbing a few seconds past compressor-off; look a short settle window
        # beyond ``ce`` for the final temperature (power/energy stay bounded by ``ce`` above).
        t1 = tank[tank.index <= ce + pd.Timedelta(seconds=tank_settle_seconds)].dropna()
        t_start = t0.iloc[-1] if not t0.empty else np.nan
        t_end = t1.iloc[-1] if not t1.empty else np.nan
    if pd.isna(t_start) or pd.isna(t_end):
        return None

    therm = TANK_LITRES * C_WATER * (t_end - t_start) / 3600 + standing_loss_kw * dur_h
    cop = therm / elec if (elec is not None and pd.notna(elec) and elec > 0) else np.nan

    amb = _trace_col(trace, "ambient")[cyc_mask].dropna()
    a = amb.mean() if not amb.empty else np.nan
    humv = _trace_col(trace, "humidity")[cyc_mask].dropna()
    h = humv.mean() if not humv.empty else np.nan

    tank_cycle = tank[cyc_mask].dropna()
    probe_rise = tank_cycle[tank_cycle >= t_start + 0.5]
    probe_lag_min = (
        (probe_rise.index[0] - cs).total_seconds() / 60 if not probe_rise.empty else np.nan
    )

    def _stat(name):
        return _trace_col(trace, name)[cyc_mask].dropna()

    def _any_on(name):
        s = _trace_col(trace, name)[cyc_mask].dropna()
        return bool((s > 0.5).any()) if not s.empty else False

    x, c_coil, r_ra, i_in = _stat("exhaust"), _stat("coil"), _stat("return_air"), _stat("inlet")
    clean = cycle_is_clean(0.0, 0.0, hp_p95, cop, elec_source)

    return dict(
        start=cs, start_ts=cs.timestamp(),
        start_local=cs.tz_convert(tz).strftime("%Y-%m-%d %H:%M"),
        end_ts=ce.timestamp(), dur_min=round(dur_h * 60),
        tank_start=round(float(t_start), 1), tank_end=round(float(t_end), 1),
        ambient=round(a, 1) if pd.notna(a) else np.nan,
        wet_bulb=_round_or_nan(stull_wet_bulb(a, h), 1),
        elec_kwh=round(float(elec), 2) if (elec is not None and pd.notna(elec)) else np.nan,
        elec_source=elec_source,
        therm_kwh=round(therm, 2),
        cop=round(cop, 2) if pd.notna(cop) else np.nan,
        hp_mean_w=round(hp_mean) if pd.notna(hp_mean) else np.nan,
        hp_p95_w=round(hp_p95) if pd.notna(hp_p95) else np.nan,
        probe_lag_min=_round_or_nan(probe_lag_min, 1),
        probe_rise_10_min=_round_or_nan(_first_rise_minutes(tank_cycle, cs, t_start, t_end, 0.10), 1),
        probe_rise_50_min=_round_or_nan(_first_rise_minutes(tank_cycle, cs, t_start, t_end, 0.50), 1),
        probe_rise_90_min=_round_or_nan(_first_rise_minutes(tank_cycle, cs, t_start, t_end, 0.90), 1),
        exhaust_start=_round_or_nan(x.iloc[0] if not x.empty else np.nan, 1),
        exhaust_max=_round_or_nan(x.max() if not x.empty else np.nan, 1),
        exhaust_end=_round_or_nan(x.iloc[-1] if not x.empty else np.nan, 1),
        coil_mean=_round_or_nan(c_coil.mean() if not c_coil.empty else np.nan, 1),
        return_air_mean=_round_or_nan(r_ra.mean() if not r_ra.empty else np.nan, 1),
        inlet_mean=_round_or_nan(i_in.mean() if not i_in.empty else np.nan, 1),
        element_on=_any_on("element"),
        defrost_on=_any_on("defrost"),
        four_way_on=_any_on("four_way"),
        clean=clean,
    )


_SUMMARY_FLOAT_COLS = [
    "start_ts", "start_local", "end_ts", "dur_min", "tank_start", "tank_end", "ambient",
    "wet_bulb", "elec_kwh", "elec_source", "therm_kwh", "cop", "hp_mean_w", "hp_p95_w",
    "probe_lag_min", "probe_rise_10_min", "probe_rise_50_min", "probe_rise_90_min",
    "exhaust_start", "exhaust_max", "exhaust_end", "coil_mean", "return_air_mean", "inlet_mean",
]
_SUMMARY_BOOL_COLS = ["element_on", "defrost_on", "four_way_on", "clean"]


def _to_utc(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(LOCAL_TZ)
    return ts.tz_convert("UTC")


def _stored_edges(c: dict):
    """Bound a recompute's power/energy window by the stored compressor on/off timestamps.

    The daemon appends trailing post-off samples during the tank settle window, so the trace's last
    index sits past compressor-off; pinning ``ce`` to the stored ``end_ts`` keeps ``[cs, ce]`` on the
    real cycle. tank_start/tank_end and the counter elec are deliberately left out so they still
    derive from the trace (the settle window then picks up the post-off peak). ``None`` (fully
    trace-derived boundaries) when either timestamp is missing.
    """
    cs, ce = c.get("start_ts"), c.get("end_ts")
    if cs is None or ce is None:
        return None
    return {"cs": pd.Timestamp(cs, unit="s", tz="UTC"), "ce": pd.Timestamp(ce, unit="s", tz="UTC")}


def _summary_row(c: dict) -> dict:
    """A stored summary row → an ``analyse`` output row: ``start`` as a UTC Timestamp, bools
    restored from 0/1. Used as-is for trace-less cycles (the CSV-seeded install-period anchors)."""
    row = {k: c.get(k) for k in _SUMMARY_FLOAT_COLS}
    row["start"] = pd.Timestamp(c["start_ts"], unit="s", tz="UTC")
    for col in _SUMMARY_BOOL_COLS:
        row[col] = bool(c.get(col))
    return row


def analyse(db_path=DEFAULT_DB_PATH, since=None, until=None, min_minutes=None, recompute=True):
    """Per-cycle COP table from the local SQLite store (docs/hwc/local_store.md).

    InfluxDB is no longer read here — the daemon records cycles live to the store. Each completed
    cycle is recomputed from its stored 30 s trace via ``cycle_metrics`` (so a methodology change
    reprocesses history on the next run); the stored summary is used as-is for a trace-less cycle
    (the CSV-seeded anchors, or any future summary-only row). ``recompute=False`` skips the trace
    pass and returns the stored summaries directly.

    ``since``/``until`` are optional local-time bounds on the cycle start; ``min_minutes`` filters
    short runs (the store is already ≥5 min from seeding). The frozen InfluxDB extraction lives in
    ``scripts/seed_hwc_store.py`` (``legacy_analyse``).
    """
    conn = store.connect(db_path, read_only=True)
    try:
        stored = store.recent_cycles(conn, 10_000_000, include_running=False)
        rows = []
        for c in stored:
            metrics = None
            if recompute:
                trace = store.load_trace(conn, c["start_ts"])
                if not trace.empty:
                    metrics = cycle_metrics(trace, edges=_stored_edges(c))
            rows.append(metrics if metrics is not None else _summary_row(c))
    finally:
        conn.close()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("start").reset_index(drop=True)
    if since is not None:
        df = df[df["start"] >= _to_utc(since)]
    if until is not None:
        df = df[df["start"] <= _to_utc(until)]
    if min_minutes is not None:
        df = df[df["dur_min"].fillna(0) >= min_minutes]
    return df.reset_index(drop=True)


def format_cycles_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not out.empty and pd.api.types.is_datetime64_any_dtype(out["start"]):
        out["start"] = out["start"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M")
    return out


def merge_cycle_tables(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge extracted cycle tables by local start time, replacing duplicate rows."""
    frames = [df for df in (existing, new) if not df.empty]
    if not frames:
        return new.copy()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["start"], keep="last")
    order = pd.to_datetime(merged["start"], errors="coerce")
    merged = (
        merged.assign(_sort_start=order)
        .sort_values(["_sort_start", "start"], kind="mergesort")
        .drop(columns=["_sort_start"])
        .reset_index(drop=True)
    )
    return merged


def write_summary_markdown(
    df: pd.DataFrame,
    path: str,
    since: str | None,
    days: int | None,
    until: str | None = None,
    merged: bool = False,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = df[df["clean"]] if not df.empty else df
    source_window = []
    if since:
        source_window.append(f"since `{since}`")
    if until:
        source_window.append(f"until `{until}`")
    if days is not None:
        source_window.append(f"last `{days}` days")
    if merged:
        extracted_source = " and ".join(source_window) if source_window else "custom bounded query"
        source = f"existing CSV merged with extracted window ({extracted_source})"
    else:
        source = " and ".join(source_window) if source_window else "custom bounded query"

    lines = [
        "# HWC Calibration Cycles",
        "",
        "Curated cycle-level calibration output from `hwc_cop_analysis.py`.",
        "",
        f"- Source window: {source}",
        f"- Rows: {len(df)} total, {len(clean)} clean",
        "- Method: compressor-on windows from HA/InfluxDB; electrical input prefers the Athom channel-2 cumulative meter (`sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_energy_2`, differenced over the cycle — see `elec_source`), falling back to baseline-subtracted power integration (raw channel-2 power, then `sensor.remaining_power_load` for older history) on a counter reset/gap; thermal output from tank probe delta plus standing loss.",
        "- Caveat: tank stratification means single-probe thermal output is approximate; use clean flags and cycle context before fitting model parameters.",
        "",
    ]
    if df.empty:
        lines.append("No qualifying cycles found.")
    else:
        display = format_cycles_for_output(df)
        cols = [
            "start", "dur_min", "tank_start", "tank_end", "ambient", "wet_bulb",
            "baseline_w", "hp_mean_w", "hp_p95_w", "power_source", "elec_kwh", "elec_source",
            "therm_kwh",
            "cop", "probe_lag_min", "probe_rise_10_min", "probe_rise_50_min",
            "probe_rise_90_min", "exhaust_start", "exhaust_max", "exhaust_end",
            "element_on", "defrost_on", "four_way_on", "clean",
        ]
        cols = [col for col in cols if col in display.columns]
        table = display[cols].fillna("").astype(str)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in table.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB_PATH,
                    help="HWC SQLite store to read (docs/hwc/local_store.md)")
    ap.add_argument("--since", default=None,
                    help="Earliest local date/time to include")
    ap.add_argument("--until", default=None,
                    help="Optional latest local date/time to include")
    ap.add_argument("--min-minutes", type=int, default=None,
                    help="Drop cycles shorter than this many minutes")
    ap.add_argument("--no-recompute", action="store_true",
                    help="Use stored summaries as-is instead of recomputing from traces")
    ap.add_argument("--merge-existing", action="store_true",
                    help="Merge extracted rows into --csv by local cycle start time")
    ap.add_argument("--csv", default="data/hwc_cop_cycles.csv")
    ap.add_argument("--summary-md", default=None,
                    help="Optional curated Markdown table to write, e.g. docs/hwc/calibration_cycles.md")
    args = ap.parse_args()
    df = analyse(db_path=args.db, since=args.since, until=args.until,
                 min_minutes=args.min_minutes, recompute=not args.no_recompute)
    output_df = format_cycles_for_output(df)
    if args.merge_existing:
        csv_path = Path(args.csv)
        existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        output_df = merge_cycle_tables(existing, output_df)

    if output_df.empty:
        print("No qualifying cycles found.")
    else:
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(output_df.to_string(index=False))
        clean = output_df[output_df["clean"]]
        if not clean.empty:
            print(
                f"\nclean cycles: {len(clean)}/{len(output_df)}  |  "
                f"mean COP (clean) = {clean['cop'].mean():.2f}"
            )
        output_df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if args.summary_md:
        write_summary_markdown(
            output_df, args.summary_md, since=args.since, days=None,
            until=args.until, merged=args.merge_existing,
        )
        print(f"wrote {args.summary_md}")
