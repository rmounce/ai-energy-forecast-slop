#!/usr/bin/env python3
"""Measure heat-pump hot water (HWC) per-cycle COP from InfluxDB.

The preferred electrical input is the dedicated heat-pump circuit meter (raw
Athom channel 2). Older history can still use `sensor.remaining_power_load` as a
residual proxy: subtract the pre/post baseline to isolate the heat pump on clean
windows. This sweeps recent compressor cycles and reports, per cycle:

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
  python hwc_cop_analysis.py --since 2026-06-03 --merge-existing
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
HWC_POWER_ENTITY = (
    "athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2"
)
RESIDUAL_POWER_ENTITY = "remaining_power_load"


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


def _first_rise_minutes(series: pd.Series, start, start_temp: float, end_temp: float, fraction: float):
    lift = end_temp - start_temp
    if lift <= 0:
        return np.nan
    threshold = start_temp + lift * fraction
    reached = series[series >= threshold]
    if reached.empty:
        return np.nan
    return (reached.index[0] - start).total_seconds() / 60


def stull_wet_bulb(t, rh):
    if pd.isna(t) or pd.isna(rh):
        return np.nan
    rh = max(1.0, min(100.0, rh))
    return (t * math.atan(0.151977 * (rh + 8.313659) ** 0.5) + math.atan(t + rh)
            - math.atan(rh - 1.676331) + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh)
            - 4.686035)


def analyse(days=None, since=DEFAULT_SINCE, until=None, min_minutes=20):
    c = _client()
    comp = _series(
        c, "binary_sensor__running", "aquatech_compressor",
        days=days, since=since, until=until,
    )
    hwc_pw = _series(
        c, "sensor__power", HWC_POWER_ENTITY,
        days=days, since=since, until=until,
    )
    residual_pw = _series(
        c, "sensor__power", RESIDUAL_POWER_ENTITY,
        days=days, since=since, until=until,
    )
    tank = _series(
        c, "sensor__temperature", "heat_pump_temperature",
        days=days, since=since, until=until,
    )
    amb = _series(
        c, "sensor__temperature", "aquatech_temperature",
        days=days, since=since, until=until,
    )
    hum = _series(
        c, "humidity_adelaide", days=days, since=since, until=until,
        field="mean_value", rp="rp_30m",
    )
    exhaust = _series(
        c, "sensor__temperature", "aquatech_exhaust_temperature",
        days=days, since=since, until=until,
    )
    coil = _series(
        c, "sensor__temperature", "aquatech_coil_temperature",
        days=days, since=since, until=until,
    )
    return_air = _series(
        c, "sensor__temperature", "aquatech_return_air_temperature",
        days=days, since=since, until=until,
    )
    inlet = _series(
        c, "sensor__temperature", "aquatech_inlet_temperature",
        days=days, since=since, until=until,
    )
    element = _series(
        c, "binary_sensor__running", "aquatech_element",
        days=days, since=since, until=until,
    )
    defrost = _series(
        c, "binary_sensor__running", "aquatech_defrost",
        days=days, since=since, until=until,
    )
    four_way = _series(
        c, "binary_sensor__running", "aquatech_four_way_valve",
        days=days, since=since, until=until,
    )
    power_series = [s for s in (hwc_pw, residual_pw) if not s.empty]
    if comp.empty or not power_series:
        raise SystemExit(
            "Missing compressor or HWC power data "
            f"({HWC_POWER_ENTITY} or {RESIDUAL_POWER_ENTITY})"
        )

    idx_min = min([comp.index.min()] + [s.index.min() for s in power_series])
    idx_max = max([comp.index.max()] + [s.index.max() for s in power_series])
    idx = pd.date_range(idx_min, idx_max, freq="30s", tz="UTC")
    on = _state_to_idx(comp, idx)
    T = _interp_to_idx(tank, idx)
    X = _interp_to_idx(exhaust, idx)
    C = _interp_to_idx(coil, idx)
    R = _interp_to_idx(return_air, idx)
    I = _interp_to_idx(inlet, idx)
    element_on = _state_to_idx(element, idx)
    defrost_on = _state_to_idx(defrost, idx)
    four_way_on = _state_to_idx(four_way, idx)

    rows = []
    grp = (on != on.shift()).cumsum()
    for _, g in pd.Series(on, index=idx).groupby(grp):
        if not g.iloc[0] or len(g) < min_minutes * 2:  # 30s steps
            continue
        cs, ce = g.index[0], g.index[-1]
        dur_h = (ce - cs).total_seconds() / 3600
        window_start = cs - pd.Timedelta("10min")
        window_end = ce + pd.Timedelta("10min")
        if _series_has_window(hwc_pw, window_start, window_end):
            P = _interp_to_idx(hwc_pw, idx)
            power_source = HWC_POWER_ENTITY
        elif _series_has_window(residual_pw, window_start, window_end):
            P = _interp_to_idx(residual_pw, idx)
            power_source = RESIDUAL_POWER_ENTITY
        else:
            continue
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
        cyc_mask = (idx >= cs) & (idx <= ce)
        probe_rise = T[cyc_mask & (T >= t_start + 0.5)]
        probe_lag_min = (
            (probe_rise.index[0] - cs).total_seconds() / 60
            if not probe_rise.empty else np.nan
        )
        tank_cycle = T[cyc_mask].dropna()
        probe_rise_10_min = _first_rise_minutes(tank_cycle, cs, t_start, t_end, 0.10)
        probe_rise_50_min = _first_rise_minutes(tank_cycle, cs, t_start, t_end, 0.50)
        probe_rise_90_min = _first_rise_minutes(tank_cycle, cs, t_start, t_end, 0.90)
        x_cycle = X[cyc_mask].dropna()
        c_cycle = C[cyc_mask].dropna()
        r_cycle = R[cyc_mask].dropna()
        i_cycle = I[cyc_mask].dropna()
        # Clean = stable, matching pre/post baselines, plausible HP power, and a
        # physically plausible apparent COP (contamination shows up as elec too low
        # → COP above the ~3 sensible-capacity ceiling for a to-60 °C reheat).
        clean = (abs(b_pre - b_post) < 80 and hp.quantile(0.95) < HP_POWER_MAX_W
                 and pd.notna(cop) and 0.8 < cop < 3.3)
        rows.append(dict(
            start=cs, dur_min=round(dur_h * 60), tank_start=round(t_start, 1),
            tank_end=round(t_end, 1), ambient=round(a, 1) if pd.notna(a) else np.nan,
            wet_bulb=round(stull_wet_bulb(a, h), 1), baseline_w=round(baseline),
            hp_mean_w=round(hp.mean()), hp_p95_w=round(hp.quantile(0.95)),
            power_source=power_source,
            elec_kwh=round(elec, 2), therm_kwh=round(therm, 2),
            cop=round(cop, 2) if pd.notna(cop) else np.nan, clean=clean,
            probe_lag_min=_round_or_nan(probe_lag_min, 1),
            probe_rise_10_min=_round_or_nan(probe_rise_10_min, 1),
            probe_rise_50_min=_round_or_nan(probe_rise_50_min, 1),
            probe_rise_90_min=_round_or_nan(probe_rise_90_min, 1),
            exhaust_start=_round_or_nan(x_cycle.iloc[0] if not x_cycle.empty else np.nan, 1),
            exhaust_max=_round_or_nan(x_cycle.max() if not x_cycle.empty else np.nan, 1),
            exhaust_end=_round_or_nan(x_cycle.iloc[-1] if not x_cycle.empty else np.nan, 1),
            coil_mean=_round_or_nan(c_cycle.mean() if not c_cycle.empty else np.nan, 1),
            return_air_mean=_round_or_nan(r_cycle.mean() if not r_cycle.empty else np.nan, 1),
            inlet_mean=_round_or_nan(i_cycle.mean() if not i_cycle.empty else np.nan, 1),
            element_on=bool(element_on[cyc_mask].any()),
            defrost_on=bool(defrost_on[cyc_mask].any()),
            four_way_on=bool(four_way_on[cyc_mask].any()),
        ))
    return pd.DataFrame(rows)


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
        "- Method: compressor-on windows from HA/InfluxDB; electrical input prefers raw Athom channel 2 (`sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2`), with baseline-subtracted `sensor.remaining_power_load` fallback for older history; thermal output from tank probe delta plus standing loss.",
        "- Caveat: tank stratification means single-probe thermal output is approximate; use clean flags and cycle context before fitting model parameters.",
        "",
    ]
    if df.empty:
        lines.append("No qualifying cycles found.")
    else:
        display = format_cycles_for_output(df)
        cols = [
            "start", "dur_min", "tank_start", "tank_end", "ambient", "wet_bulb",
            "baseline_w", "hp_mean_w", "hp_p95_w", "power_source", "elec_kwh", "therm_kwh",
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
    ap.add_argument("--days", type=int, default=None,
                    help="Optional rolling lookback, still bounded by --since when set")
    ap.add_argument("--since", default=DEFAULT_SINCE,
                    help="Earliest local date/time to query; default is Aquatech install date")
    ap.add_argument("--until", default=None,
                    help="Optional latest local date/time to query")
    ap.add_argument("--merge-existing", action="store_true",
                    help="Merge extracted rows into --csv by local cycle start time")
    ap.add_argument("--csv", default="data/hwc_cop_cycles.csv")
    ap.add_argument("--summary-md", default=None,
                    help="Optional curated Markdown table to write, e.g. docs/hwc_calibration_cycles.md")
    args = ap.parse_args()
    df = analyse(days=args.days, since=args.since, until=args.until)
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
            output_df, args.summary_md, since=args.since, days=args.days,
            until=args.until, merged=args.merge_existing,
        )
        print(f"wrote {args.summary_md}")
