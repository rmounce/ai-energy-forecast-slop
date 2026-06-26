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


def _elec_kwh(run: pd.DataFrame) -> "pd.Series":
    dt_h = run.index.to_series().diff().dt.total_seconds() / 3600.0
    return (run.get("power_w", 0).fillna(0) / 1000.0) * dt_h


def iter_reheats(df: pd.DataFrame, min_min: int = 20):
    """Yield each contiguous compressor-on run (local-naive index) lasting >= min_min."""
    on = df.get("compressor_on", 0) > 0.5
    for _, g in df.groupby((on != on.shift()).cumsum()):
        if not bool(on.loc[g.index[0]]):
            continue
        if (g.index[-1] - g.index[0]).total_seconds() / 60 >= min_min:
            yield g


def segment_reheat(run: pd.DataFrame, blind_rise_c: float = 2.0) -> dict:
    """Split one compressor-on run into the probe-BLIND build phase and the RISE phase.

    "Blind" = on-start until the probe has risen ``blind_rise_c`` above its start value (robust to
    the probe's 1 C source quantisation). The blind phase's duration/energy is the part a
    probe-only heat-rate curve cannot predict — it is set by latent hot-volume, not the probe
    (see docs/hwc_thermal_characterisation.md Finding 4). Shared by ``batch`` here and
    ``hwc_soc_calibrate.py --mode phases`` so the definition can't diverge.
    """
    kwh = _elec_kwh(run)
    p0, p1 = run["probe_ctrl"].iloc[0], run["probe_ctrl"].iloc[-1]
    on_min = (run.index[-1] - run.index[0]).total_seconds() / 60
    risen = run["probe_ctrl"] >= p0 + blind_rise_c
    t_rise = run.index[risen][0] if risen.any() else run.index[-1]
    blind = run.loc[:t_rise]
    rise = run.loc[t_rise:]
    blind_min = (blind.index[-1] - blind.index[0]).total_seconds() / 60
    rise_min = (rise.index[-1] - rise.index[0]).total_seconds() / 60
    blind_kwh = float(_elec_kwh(blind).sum())
    on_kwh = float(kwh.sum())
    rise_rate = (rise["probe_ctrl"].iloc[-1] - rise["probe_ctrl"].iloc[0]) / (rise_min / 60) if rise_min else 0.0
    return dict(
        start=run.index[0], end=run.index[-1], p0=round(p0, 1), p1=round(p1, 1),
        on_min=round(on_min, 1), on_kwh=round(on_kwh, 3),
        blind_min=round(blind_min, 1), blind_kwh=round(blind_kwh, 3),
        blind_pct=round(100 * blind_kwh / on_kwh, 1) if on_kwh else 0.0,
        rise_min=round(rise_min, 1), rise_Cph=round(rise_rate, 2),
    )


def batch(since, until, min_min: int = 20, freq: str = "30s", out: Path | None = None):
    """Segment every reheat in a span into per-reheat features (the dynamics-fit dataset)."""
    df = extract(since, until, freq=freq)
    df.index = df.index.tz_localize(None)
    rows = [segment_reheat(g) for g in iter_reheats(df, min_min=min_min)]
    if not rows:
        raise SystemExit("No reheats in the requested span")
    tbl = pd.DataFrame(rows)
    show = tbl.assign(start=tbl["start"].dt.strftime("%m-%d %H:%M")).drop(columns=["end"])
    print(show.to_string(index=False))
    full = tbl[tbl["blind_pct"] < 100]  # drop aborted/short cycles where probe never rose 2 C
    print(f"\n{len(tbl)} reheats ({len(full)} full). blind_pct over full: "
          f"mean {full['blind_pct'].mean():.0f}%  range {full['blind_pct'].min():.0f}-{full['blind_pct'].max():.0f}%")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        tbl.to_csv(out, index=False)
        print(f"wrote {out}")
    return tbl


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

    pb = sub.add_parser("batch", help="segment every reheat in a span to per-reheat features")
    pb.add_argument("--since", required=True, help="local start (Athom power only from 2026-06-14)")
    pb.add_argument("--until", required=True, help="local end")
    pb.add_argument("--min-reheat-min", type=int, default=20)
    pb.add_argument("--freq", default="30s")
    pb.add_argument("--out", type=Path, help="optional CSV of the feature table")

    args = ap.parse_args()
    if args.mode == "list":
        list_events(days=args.days, min_reheat_min=args.min_reheat_min)
    elif args.mode == "batch":
        batch(args.since, args.until, min_min=args.min_reheat_min, freq=args.freq, out=args.out)
    else:
        df = extract(args.since, args.until, freq=args.freq)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out)
        print(f"Wrote {len(df)} rows x {len(df.columns)} cols to {args.out}")
        cols = [c for c in ("probe_ctrl", "probe_hp", "exhaust", "inlet", "power_w") if c in df.columns]
        print(df[cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
