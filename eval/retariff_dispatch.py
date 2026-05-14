#!/usr/bin/env python3
"""Re-tariff per-step dispatch decisions from existing rolling-MPC raw_rows
parquets under alternate retail-plan tariff structures.

Given:
  - charge_kw, discharge_kw, curtail_kw, actual_load_kw, actual_pv_kw,
    actual_price_mwh (realised wholesale RRP)
from a completed rolling-MPC eval run, recompute step PnL under each
candidate retail plan.

Plans modelled (Adelaide local time, half-hour granularity):
  - RESELE (the current/incumbent plan): tariff_profile.json + the Nov-Mar
    +$0.1225/kWh evening credit override. Wholesale flows through both
    sides with the network loss factor; GST applies asymmetrically.
  - Flow Power: import = wholesale × loss_factor + $0.20/kWh × GST.
    Export = flat $0.45/kWh at 17:30-19:30 Adelaide local (no wholesale
    upside), $0 other hours.
  - IO Energy: fully fixed both sides. Import $0.85 at 17:00-21:00,
    $0.08 at 10:00-16:00, $0.37 otherwise. Export $0.30 at 18:00-21:00,
    -$0.02 at 10:00-16:00, $0.04 otherwise.

**Caveat:** the dispatch decisions in the input raw_rows were taken by an
LP optimised against RESELE's tariff. Replaying them under Flow Power or
IO Energy gives a *lower bound* on those plans because their LP would
have charged/discharged differently. The replay is a fast directional
read, not a final answer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TZ = "Australia/Adelaide"
NETWORK_LOSS_FACTOR = 1.05
GST = 1.10
EFF_C = 0.95
EFF_D = 0.95
DEG_PER_KWH = 0.05
INTERVAL_H = 5 / 60.0  # 5-min step

RESELE_SUMMER_MONTHS = {11, 12, 1, 2, 3}


@dataclass
class TariffStep:
    """Per-row tariff result, all in $/kWh."""

    general_price: np.ndarray  # what you pay for grid import
    feed_in_price: np.ndarray  # what you get for grid export


def _local_hod_min(times_utc: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    local = times_utc.tz_convert(LOCAL_TZ)
    hod_min = local.hour * 60 + local.minute  # minutes from midnight, 0..1439
    month = local.month.to_numpy()
    return hod_min.to_numpy(), month


def resele_tariff(times_utc: pd.DatetimeIndex, wholesale_mwh: np.ndarray) -> TariffStep:
    """RESELE: 3-band general tariff + seasonal evening export credit."""
    hod_min, month = _local_hod_min(times_utc)
    wholesale = wholesale_mwh / 1000.0  # $/kWh

    # Import tariff bands (half-hour resolution from tariff_profile.json):
    #   10:00-15:30 → $0.0666 (12 slots starting at minute 600, ending 930+30)
    #   17:00-20:30 → $0.3580 (8 slots, minute 1020..1230+30)
    #   other       → $0.1331
    is_shoulder = (hod_min >= 600) & (hod_min < 960)     # 10:00-16:00
    is_peak = (hod_min >= 1020) & (hod_min < 1260)       # 17:00-21:00
    general_tariff = np.where(
        is_shoulder, 0.0666,
        np.where(is_peak, 0.3580, 0.1331)
    )

    # Feed-in tariff (winter baseline)
    #   -$0.01 at 10:00-15:30 (solar penalty)
    #   $0      elsewhere
    feed_in_tariff = np.where(is_shoulder, -0.0100, 0.0).astype(float)

    # Summer override: Nov-Mar 17:00-20:30 → +$0.1225
    is_summer = np.isin(month, list(RESELE_SUMMER_MONTHS))
    is_summer_credit_window = (hod_min >= 1020) & (hod_min < 1230)
    feed_in_tariff = np.where(is_summer & is_summer_credit_window, 0.1225, feed_in_tariff)

    general_ex_gst = wholesale * NETWORK_LOSS_FACTOR + general_tariff
    feed_in_ex_gst = wholesale * NETWORK_LOSS_FACTOR + feed_in_tariff

    general_price = np.where(general_ex_gst > 0, general_ex_gst * GST, general_ex_gst)
    feed_in_price = np.where(feed_in_ex_gst < 0, feed_in_ex_gst * GST, feed_in_ex_gst)
    return TariffStep(general_price=general_price, feed_in_price=feed_in_price)


def flowpower_tariff(times_utc: pd.DatetimeIndex, wholesale_mwh: np.ndarray) -> TariffStep:
    """Flow Power: wholesale-pass-through import + 20c markup;
    flat $0.45/kWh export at 17:30-19:30, $0 otherwise."""
    hod_min, _ = _local_hod_min(times_utc)
    wholesale = wholesale_mwh / 1000.0

    general_ex_gst = wholesale * NETWORK_LOSS_FACTOR + 0.20
    general_price = np.where(general_ex_gst > 0, general_ex_gst * GST, general_ex_gst)

    # Flat 45c only at 17:30-19:30. Treat as full feed-in price (no wholesale
    # component) — user described it as "flat", not wholesale-linked.
    is_fit_window = (hod_min >= 1050) & (hod_min < 1170)  # 17:30..19:30
    feed_in_price = np.where(is_fit_window, 0.45, 0.0)
    return TariffStep(general_price=general_price, feed_in_price=feed_in_price)


def ioenergy_tariff(times_utc: pd.DatetimeIndex, wholesale_mwh: np.ndarray) -> TariffStep:
    """IO Energy: fully fixed both sides, no wholesale exposure."""
    hod_min, _ = _local_hod_min(times_utc)

    # Import: $0.08 at 10:00-16:00, $0.85 at 17:00-21:00, $0.37 otherwise
    is_solar = (hod_min >= 600) & (hod_min < 960)        # 10:00-16:00
    is_peak = (hod_min >= 1020) & (hod_min < 1260)       # 17:00-21:00
    general_price = np.where(is_solar, 0.08,
                    np.where(is_peak, 0.85, 0.37)).astype(float)

    # Export: $0.30 at 18:00-21:00, -$0.02 at 10:00-16:00, $0.04 otherwise
    is_evening_credit = (hod_min >= 1080) & (hod_min < 1260)  # 18:00-21:00
    feed_in_price = np.where(is_solar, -0.02,
                    np.where(is_evening_credit, 0.30, 0.04)).astype(float)
    return TariffStep(general_price=general_price, feed_in_price=feed_in_price)


PLANS = {
    "resele":     resele_tariff,
    "flowpower":  flowpower_tariff,
    "ioenergy":   ioenergy_tariff,
}


def retariff(raw_rows: pd.DataFrame, plan_fn) -> pd.DataFrame:
    df = raw_rows.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    times = pd.DatetimeIndex(df["time"])
    wholesale = df["actual_price_mwh"].to_numpy(dtype=np.float64)

    t = plan_fn(times, wholesale)

    # Realised grid flow under the existing dispatch decisions.
    # Formula mirrors eval/rolling_mpc_eval.py:1771.
    load = df["actual_load_kw"].to_numpy(dtype=np.float64)
    pv = df["actual_pv_kw"].to_numpy(dtype=np.float64)
    c0 = df["charge_kw"].to_numpy(dtype=np.float64)
    d0 = df["discharge_kw"].to_numpy(dtype=np.float64)
    curtail = df.get("curtail_kw", pd.Series(0.0, index=df.index)).to_numpy(dtype=np.float64)

    grid_kw = load - pv + c0 - d0 * EFF_D + curtail
    grid_import = np.maximum(grid_kw, 0.0)
    grid_export = np.maximum(-grid_kw, 0.0)

    pnl = (
        grid_export * t.feed_in_price
        - grid_import * t.general_price
        - DEG_PER_KWH * (c0 * EFF_C + d0)
    ) * INTERVAL_H

    df["plan_general_price"] = t.general_price
    df["plan_feed_in_price"] = t.feed_in_price
    df["plan_grid_import_kw"] = grid_import
    df["plan_grid_export_kw"] = grid_export
    df["plan_step_pnl"] = pnl
    return df


def summarise(retariffed: pd.DataFrame, label: str) -> dict:
    """Roll up a re-tariffed dataframe into a one-row summary dict."""
    total_pnl = float(retariffed["plan_step_pnl"].sum())
    days = (retariffed["time"].max() - retariffed["time"].min()).total_seconds() / 86400.0
    days = max(days, 1e-9)
    total_import_kwh = float((retariffed["plan_grid_import_kw"] * INTERVAL_H).sum())
    total_export_kwh = float((retariffed["plan_grid_export_kw"] * INTERVAL_H).sum())
    return {
        "label": label,
        "n_steps": len(retariffed),
        "n_days": days,
        "total_pnl_$": total_pnl,
        "mean_pnl_per_day_$": total_pnl / days,
        "grid_import_kwh": total_import_kwh,
        "grid_export_kwh": total_export_kwh,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-rows", type=Path, required=True, action="append",
        help="Path to a rolling-MPC raw_rows parquet. May be repeated.",
    )
    parser.add_argument(
        "--source-filter", default="amber_apf_lgbm",
        help="Source contract to keep when raw_rows has multiple sources.",
    )
    parser.add_argument(
        "--plans", default="resele,flowpower,ioenergy",
    )
    args = parser.parse_args()

    plan_keys = [p.strip() for p in args.plans.split(",") if p.strip()]
    for p in plan_keys:
        if p not in PLANS:
            raise SystemExit(f"Unknown plan '{p}'; valid: {sorted(PLANS)}")

    summary_rows = []
    for path in args.raw_rows:
        df = pd.read_parquet(path)
        df = df[df["source"] == args.source_filter].copy()
        if df.empty:
            print(f"WARN: {path} has no rows for source='{args.source_filter}'")
            continue

        label_base = path.stem.replace("_raw", "")
        for plan_key in plan_keys:
            retariffed = retariff(df, PLANS[plan_key])
            summary_rows.append({
                "run": label_base, "plan": plan_key,
                **summarise(retariffed, f"{label_base}/{plan_key}"),
            })

    out = pd.DataFrame(summary_rows).set_index(["run", "plan"])
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:9.3f}")
    print(out.to_string())
    print()

    # If both summer and winter runs are present, give a naive annualised
    # extrapolation per plan.
    print("Naive annualised per-plan extrapolation (152 summer days + 213 winter days):")
    annual = {}
    for plan_key in plan_keys:
        per_day = {}
        for (run, plan), row in out.iterrows():
            if plan != plan_key:
                continue
            if "summer" in run:
                per_day["summer"] = row["mean_pnl_per_day_$"]
            elif any(tag in run for tag in ("B_v3", "winter")):
                per_day["winter"] = row["mean_pnl_per_day_$"]
        if "summer" in per_day and "winter" in per_day:
            annual_pnl = 152 * per_day["summer"] + 213 * per_day["winter"]
            print(
                f"  {plan_key:10s} summer=${per_day['summer']:6.2f}/day  "
                f"winter=${per_day['winter']:6.2f}/day  "
                f"→ annual ≈ ${annual_pnl:7.0f}"
            )
            annual[plan_key] = annual_pnl
        else:
            print(f"  {plan_key:10s} insufficient runs for annualisation")


if __name__ == "__main__":
    main()
