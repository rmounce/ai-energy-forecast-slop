#!/usr/bin/env python3
"""Extract fine-resolution HWC tank traces from InfluxDB for SoC-model calibration.

Feeds the stratified two-state ``(V_hot, T_hot)`` tank model (design: the heat-rate/COP and the
probe map are calibrated from real cycles rather than a hand-set FULL/TOP-UP latch — see
``docs/hwc_thermal_characterisation.md`` and ``docs/hwc_short_cycle_review_2026-06-26.md``).

Two modes:
  --list     survey recent compressor-on reheats + the deepest probe draw, to pick windows.
  --extract  pull every signal over one window onto a common grid and write a tidy CSV.

A clean cold-reheat calibrates the probe map ``g`` and ``COP``; the big-draw event calibrates the
discharge (plug-flow) dynamics. Reuses the InfluxDB client + series helpers from
``hwc_cop_analysis`` (single source of the Aquatech entity map).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hwc_cop_analysis import (
    HWC_POWER_ENTITY,
    LOCAL_TZ,
    _client,
    _interp_to_idx,
    _series,
    _state_to_idx,
)

# entity_id under measurement "sensor__temperature"
TEMP_SIGNALS = {
    "probe_ctrl": "aquatech_current_temperature_local",  # the controller/obligation probe
    "probe_hp": "heat_pump_temperature",                 # probe used by the COP analyser
    "exhaust": "aquatech_exhaust_temperature",           # condensing temp → COP driver
    "coil": "aquatech_coil_temperature",                 # evaporator
    "inlet": "aquatech_inlet_temperature",               # water-in (T_mains); NOT logging as of
    #                                                      2026-06-26 — no water-side sensor found,
    #                                                      so T_mains stays a model parameter
    "ambient": "aquatech_temperature",
}
# entity_id under measurement "binary_sensor__running"
STATE_SIGNALS = {
    "compressor": "aquatech_compressor",
    "defrost": "aquatech_defrost",
    "element": "aquatech_element",
}


def extract(since, until, freq="30s") -> pd.DataFrame:
    """Pull all signals over [since, until] onto a common UTC grid."""
    c = _client()
    temps = {
        k: _series(c, "sensor__temperature", e, since=since, until=until)
        for k, e in TEMP_SIGNALS.items()
    }
    power = _series(c, "sensor__power", HWC_POWER_ENTITY, since=since, until=until)
    states = {
        k: _series(c, "binary_sensor__running", e, since=since, until=until)
        for k, e in STATE_SIGNALS.items()
    }
    present = [s for s in (*temps.values(), power, *states.values()) if not s.empty]
    if not present:
        raise SystemExit("No data in the requested window")
    lo = min(s.index.min() for s in present)
    hi = max(s.index.max() for s in present)
    idx = pd.date_range(lo, hi, freq=freq, tz="UTC")

    df = pd.DataFrame(index=idx)
    for k, s in temps.items():
        df[k] = _interp_to_idx(s, idx)
    df["power_w"] = _interp_to_idx(power, idx)
    for k, s in states.items():
        df[f"{k}_on"] = _state_to_idx(s, idx).astype(int)
    df.index = df.index.tz_convert(LOCAL_TZ)
    df.index.name = "time_local"
    return df


def list_events(days=8, min_reheat_min=20) -> None:
    """Print recent compressor-on reheats and the deepest probe draw, to pick windows."""
    c = _client()
    comp = _series(c, "binary_sensor__running", "aquatech_compressor", days=days)
    probe = _series(c, "sensor__temperature", TEMP_SIGNALS["probe_ctrl"], days=days)
    if probe.empty:
        probe = _series(c, "sensor__temperature", TEMP_SIGNALS["probe_hp"], days=days)
    if comp.empty or probe.empty:
        raise SystemExit("Missing compressor or probe data")

    idx = pd.date_range(comp.index.min(), comp.index.max(), freq="30s", tz="UTC")
    on = _state_to_idx(comp, idx)
    p = _interp_to_idx(probe, idx)

    print(f"Compressor-on reheats (>= {min_reheat_min} min), last {days} d:")
    grp = (on != on.shift()).cumsum()
    for _, g in pd.Series(on, index=idx).groupby(grp):
        if not g.iloc[0] or len(g) < min_reheat_min * 2:  # 30 s steps
            continue
        s, e = g.index[0], g.index[-1]
        sl, el = s.tz_convert(LOCAL_TZ), e.tz_convert(LOCAL_TZ)
        dur = (e - s).total_seconds() / 60
        print(
            f"  {sl:%Y-%m-%d %H:%M} -> {el:%H:%M}  ({dur:4.0f} min)  "
            f"probe {p.get(s, float('nan')):.1f} -> {p.get(e, float('nan')):.1f} C"
        )

    pmin_t = p.idxmin()
    print(
        f"\nDeepest probe (likely big draw): {p.min():.1f} C at "
        f"{pmin_t.tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pl = sub.add_parser("list", help="survey reheats + deepest draw")
    pl.add_argument("--days", type=int, default=8)
    pl.add_argument("--min-reheat-min", type=int, default=20)

    pe = sub.add_parser("extract", help="pull one window to CSV")
    pe.add_argument("--since", required=True, help="local start, e.g. '2026-06-26 04:00'")
    pe.add_argument("--until", required=True, help="local end")
    pe.add_argument("--freq", default="30s")
    pe.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.mode == "list":
        list_events(days=args.days, min_reheat_min=args.min_reheat_min)
    else:
        df = extract(args.since, args.until, freq=args.freq)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out}")
        cols = [c for c in ("probe_ctrl", "probe_hp", "exhaust", "inlet", "power_w") if c in df.columns]
        print(df[cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
