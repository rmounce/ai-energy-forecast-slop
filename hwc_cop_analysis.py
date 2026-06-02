#!/usr/bin/env python3
"""Measure heat-pump hot water (HWC) per-cycle COP from InfluxDB.

The unit doesn't meter its own power, but `sensor.remaining_power_load` (all
un-individually-metered house load) is a usable proxy on clean windows: subtract
the pre/post baseline to isolate the heat pump. This sweeps recent compressor
cycles and reports, per cycle:

  start/end tank temp, ambient, duration, electrical-in (baseline-subtracted),
  thermal-out (single-probe ΔT + standing loss), apparent COP, and a cleanliness
  flag (so contaminated windows aren't over-trusted).

Caveats (see docs/hwc_thermal_characterisation.md):
  - Thermal-out uses the single tank probe; the tank stratifies, so this is
    approximate. The hard COP ceiling (elec vs 45→target sensible capacity) is
    more robust than the point estimate.
  - COP is specific to the current fan-speed setting and to the target temp.

Usage:
  python hwc_cop_analysis.py
  python hwc_cop_analysis.py --days 3 --csv data/hwc_cop_cycles.csv
  python hwc_cop_analysis.py --summary-md docs/hwc_calibration_cycles.md
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

from config_utils import load_config

TANK_LITRES = 225
C_WATER = 4.186  # kJ/kg·K
STANDING_LOSS_KW = 0.12
HP_POWER_MAX_W = 1100  # plausible upper bound for this unit; above → contamination
LOCAL_TZ = "Australia/Adelaide"
DEFAULT_SINCE = "2026-05-28"  # Aquatech install date; earlier HA history is unrelated.


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


def _series_query(meas, eid=None, days=None, since=DEFAULT_SINCE, field="value", rp=""):
    src = f'"{rp}"."{meas}"' if rp else f'"{meas}"'
    where = []
    if eid:
        where.append(f"entity_id='{eid}'")
    if since:
        where.append(f"time >= '{_format_influx_time(since)}'")
    if days is not None:
        where.append(f"time> now()-{days}d")
    if not where:
        raise ValueError("Refusing to query unbounded time range")
    return f"SELECT \"{field}\" FROM {src} WHERE {' AND '.join(where)}"


def _series(c, meas, eid=None, days=None, since=DEFAULT_SINCE, field="value", rp=""):
    q = _series_query(meas, eid=eid, days=days, since=since, field=field, rp=rp)
    pts = list(c.query(q).get_points())
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series({pd.to_datetime(p["time"]): float(p[field])
                   for p in pts if p.get(field) is not None}).sort_index()
    s.index = s.index.tz_localize("UTC") if s.index.tz is None else s.index.tz_convert("UTC")
    return s[~s.index.duplicated(keep="last")]


def stull_wet_bulb(t, rh):
    if pd.isna(t) or pd.isna(rh):
        return np.nan
    rh = max(1.0, min(100.0, rh))
    return (t * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + math.atan(t + rh)
            - math.atan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035)


def analyse(days=None, since=DEFAULT_SINCE, min_minutes=20):
    c = _client()
    comp = _series(c, "binary_sensor__running", "aquatech_compressor", days=days, since=since)
    pw = _series(c, "sensor__power", "remaining_power_load", days=days, since=since)
    tank = _series(c, "sensor__temperature", "heat_pump_temperature", days=days, since=since)
    amb = _series(c, "sensor__temperature", "aquatech_temperature", days=days, since=since)
    hum = _series(c, "humidity_adelaide", days=days, since=since, field="mean_value", rp="rp_30m")
    if comp.empty or pw.empty:
        raise SystemExit("Missing compressor or remaining_power_load data")

    idx = pd.date_range(pw.index.min(), pw.index.max(), freq="30s", tz="UTC")
    P = pw.reindex(idx.union(pw.index)).interpolate("time").reindex(idx)
    on = comp.reindex(idx, method="ffill").fillna(0) > 0.5
    T = tank.reindex(idx.union(tank.index)).interpolate("time").reindex(idx)

    rows = []
    grp = (on != on.shift()).cumsum()
    for _, g in pd.Series(on, index=idx).groupby(grp):
        if not g.iloc[0] or len(g) < min_minutes * 2:  # 30s steps
            continue
        cs, ce = g.index[0], g.index[-1]
        dur_h = (ce - cs).total_seconds() / 3600
        pre = P[(idx >= cs - pd.Timedelta("10min")) & (idx < cs) & (~on)]
        post = P[(idx > ce) & (idx <= ce + pd.Timedelta("10min")) & (~on)]
        if pre.empty or post.empty:
            continue
        b_pre, b_post = pre.median(), post.median()
        baseline = np.mean([b_pre, b_post])
        cyc = P[(idx >= cs) & (idx <= ce)]
        hp = (cyc - baseline).clip(lower=0)
        elec = hp.sum() * (30 / 3600) / 1000  # kWh
        t0 = T[T.index <= cs + pd.Timedelta("90s")]
        t1 = T[T.index <= ce]
        if t0.empty or t1.empty:
            continue
        t_start, t_end = t0.iloc[-1], t1.iloc[-1]
        therm = TANK_LITRES * C_WATER * (t_end - t_start) / 3600 + STANDING_LOSS_KW * dur_h
        cop = therm / elec if elec > 0 else np.nan
        a = amb[(amb.index >= cs) & (amb.index <= ce)].mean()
        h = hum[(hum.index >= cs) & (hum.index <= ce)].mean() if not hum.empty else np.nan
        # Clean = stable, matching pre/post baselines, plausible HP power, and a
        # physically plausible apparent COP (contamination shows up as elec too low
        # → COP above the ~3 sensible-capacity ceiling for a to-60 °C reheat).
        clean = (abs(b_pre - b_post) < 80 and hp.quantile(0.95) < HP_POWER_MAX_W
                 and pd.notna(cop) and 0.8 < cop < 3.3)
        rows.append(dict(
            start=cs, dur_min=round(dur_h * 60), tank_start=round(t_start, 1),
            tank_end=round(t_end, 1), ambient=round(a, 1) if pd.notna(a) else np.nan,
            wet_bulb=round(stull_wet_bulb(a, h), 1), baseline_w=round(baseline),
            elec_kwh=round(elec, 2), therm_kwh=round(therm, 2),
            cop=round(cop, 2) if pd.notna(cop) else np.nan, clean=clean,
        ))
    return pd.DataFrame(rows)


def write_summary_markdown(df: pd.DataFrame, path: str, since: str | None, days: int | None) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = df[df["clean"]] if not df.empty else df
    source_window = []
    if since:
        source_window.append(f"since `{since}`")
    if days is not None:
        source_window.append(f"last `{days}` days")
    source = " and ".join(source_window) if source_window else "custom bounded query"

    lines = [
        "# HWC Calibration Cycles",
        "",
        "Curated cycle-level calibration output from `hwc_cop_analysis.py`.",
        "",
        f"- Source window: {source}",
        f"- Rows: {len(df)} total, {len(clean)} clean",
        "- Method: compressor-on windows from HA/InfluxDB; electrical input from baseline-subtracted `sensor.remaining_power_load`; thermal output from tank probe delta plus standing loss.",
        "- Caveat: tank stratification means single-probe thermal output is approximate; use clean flags and cycle context before fitting model parameters.",
        "",
    ]
    if df.empty:
        lines.append("No qualifying cycles found.")
    else:
        display = df.copy()
        if pd.api.types.is_datetime64_any_dtype(display["start"]):
            display["start"] = display["start"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M")
        cols = [
            "start", "dur_min", "tank_start", "tank_end", "ambient", "wet_bulb",
            "baseline_w", "elec_kwh", "therm_kwh", "cop", "clean",
        ]
        table = display[cols].astype(str)
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in table.iterrows():
            lines.append("| " + " | ".join(row[col] for col in cols) + " |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None,
                    help="Optional rolling lookback, still bounded by --since when set")
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help="Earliest local date/time to query; default is Aquatech install date")
    ap.add_argument("--csv", default="data/hwc_cop_cycles.csv")
    ap.add_argument("--summary-md", default=None,
                    help="Optional curated Markdown table to write, e.g. docs/hwc_calibration_cycles.md")
    args = ap.parse_args()
    df = analyse(days=args.days, since=args.since)
    if df.empty:
        print("No qualifying cycles found.")
    else:
        df["start"] = df["start"].dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d %H:%M")
        with pd.option_context("display.width", 160, "display.max_columns", None):
            print(df.to_string(index=False))
        clean = df[df["clean"]]
        if not clean.empty:
            print(f"\nclean cycles: {len(clean)}/{len(df)}  |  mean COP (clean) = {clean['cop'].mean():.2f}")
        df.to_csv(args.csv, index=False)
        print(f"wrote {args.csv}")
    if args.summary_md:
        write_summary_markdown(df, args.summary_md, since=args.since, days=args.days)
        print(f"wrote {args.summary_md}")
