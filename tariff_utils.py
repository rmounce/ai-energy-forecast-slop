#!/usr/bin/env python3
"""
Shared tariff helpers for deterministic import/export price reconstruction.

These utilities intentionally implement the same current-tariff assumptions used
by live forecast publication and rolling MPC eval:
  - wholesale price plus network loss factor
  - separate general and feed-in tariff schedules
  - GST on the import leg only, and only when it is a net cost (> 0); the feed-in
    (export) leg is GST-free in both directions (Amber, from ~FY27)

This supports "current-tariff backtest" experiments without requiring a
separately persisted historical effective-rate dataset.
"""

from __future__ import annotations

import json
import statistics
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz


# Time-of-day buckets used to smooth the reconstructed tariff profile. Each interval
# is collapsed to its bucket median so the deterministic adders don't inherit the
# per-interval noise of a single day's reconstruction.
_PEAK_START, _PEAK_END = time(17, 0), time(20, 59)
_SOLAR_START, _SOLAR_END = time(10, 0), time(15, 59)


def tariff_bucket(t: time) -> str:
    """Peak / Solar-sponge / Off-peak bucket for a time-of-day."""
    if _PEAK_START <= t <= _PEAK_END:
        return "peak"
    if _SOLAR_START <= t <= _SOLAR_END:
        return "solar"
    return "off_peak"


def fit_shared_slope(
    wholesale,
    y,
    groups,
    *,
    min_points: int = 8,
) -> tuple[float, float, dict] | None:
    """Pooled OLS with one shared slope and a separate intercept per group.

    Fits ``y == slope * wholesale + intercept[group]`` across all observations, so a
    single loss factor (the slope) is estimated from every interval and leg at once
    while each group (e.g. import/off-peak, feed-in/solar) gets its own fixed adder
    (the intercept). This is the BLUE for the reconstruction and pools the per-interval
    rounding noise far better than taking per-interval reconstructions and medianing.

    Returns ``(slope, se_slope, {group: intercept})`` or ``None`` if the design is
    rank-deficient or there are too few points.
    """
    wholesale = np.asarray(wholesale, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = list(groups)
    n = len(y)
    if n < min_points or len(wholesale) != n or len(groups) != n:
        return None

    uniq = sorted(set(groups), key=lambda g: (str(type(g)), repr(g)))
    gidx = {g: i for i, g in enumerate(uniq)}
    design = np.zeros((n, len(uniq) + 1), dtype=np.float64)
    design[:, 0] = wholesale
    for row, g in enumerate(groups):
        design[row, 1 + gidx[g]] = 1.0

    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    dof = n - design.shape[1]
    if dof <= 0:
        return None

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    s2 = float(resid @ resid / dof)
    xtx_inv = np.linalg.inv(design.T @ design)
    se_slope = float(np.sqrt(max(s2 * xtx_inv[0, 0], 0.0)))
    intercepts = {g: float(beta[1 + gidx[g]]) for g in uniq}
    return float(beta[0]), se_slope, intercepts


def smooth_tariff_maps(profile: dict) -> dict:
    """Return a copy of `profile` with each tariff schedule collapsed to bucket medians.

    Only the `general_tariff` and `feed_in_tariff` maps are smoothed; scalar fields
    (e.g. `amber_api_scaling_factor`, `network_loss_factor`) are passed through
    unchanged. Each HH:MM:SS interval is replaced by the median of its Peak /
    Solar-sponge / Off-peak bucket, rounded to 6 dp.
    """
    smoothed = dict(profile)
    for key in ("general_tariff", "feed_in_tariff"):
        schedule = profile.get(key)
        if not schedule:
            continue
        buckets: dict[str, list[float]] = {"peak": [], "solar": [], "off_peak": []}
        for time_str, value in schedule.items():
            buckets[tariff_bucket(time.fromisoformat(time_str))].append(value)
        medians = {
            b: (round(statistics.median(vals), 6) if vals else 0.0)
            for b, vals in buckets.items()
        }
        smoothed[key] = {
            time_str: medians[tariff_bucket(time.fromisoformat(time_str))]
            for time_str in schedule
        }
    return smoothed


def load_tariff_profile(config: dict, root: Path) -> tuple[dict[str, float], dict[str, float], float]:
    tariff_path = root / config["paths"]["tariff_file"]
    try:
        with open(tariff_path) as f:
            tariffs = json.load(f)
    except FileNotFoundError:
        return {}, {}, 1.05
    return (
        tariffs.get("general_tariff", {}),
        tariffs.get("feed_in_tariff", {}),
        float(tariffs.get("network_loss_factor", 1.05)),
    )


def ensure_utc_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Return a copy with a timezone-aware UTC DatetimeIndex.

    Internal code should keep timestamps in UTC. Local time is a boundary concern
    used only for tariff lookup, display, and provider-specific API parsing.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    elif out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def export_value_to_amber_feed_in_price(export_value_per_kwh: float) -> float:
    """
    Convert canonical export value to Amber feed-in price convention.

    Canonical internal convention:
      positive export value = consumer earns money by exporting.

    Amber/Home Assistant feed-in convention:
      negative price = consumer earns money by exporting.
    """
    return -float(export_value_per_kwh)


def amber_feed_in_price_to_export_value(amber_feed_in_price_per_kwh: float) -> float:
    """Inverse of export_value_to_amber_feed_in_price()."""
    return -float(amber_feed_in_price_per_kwh)


def tariffed_price_frame_from_wholesale_mwh(
    wholesale_prices_mwh: pd.Series,
    *,
    timezone: str,
    general_tariff_map: dict[str, float],
    feed_in_tariff_map: dict[str, float],
    network_loss_factor: float,
    gst_rate: float,
) -> pd.DataFrame:
    """
    Convert a wholesale $/MWh series into effective import/export price series.

    Returns a DataFrame indexed like the input with:
      wholesale_price
      general_tariff
      feed_in_tariff
      general_price
      feed_in_price
      general_price_mwh
      feed_in_price_mwh
    where `*_price` is in $/kWh and `*_price_mwh` is in $/MWh.
    """
    wholesale_prices_mwh = ensure_utc_index(wholesale_prices_mwh)
    frame = pd.DataFrame(index=wholesale_prices_mwh.index.copy())
    frame["wholesale_price"] = wholesale_prices_mwh.astype(np.float64) / 1000.0

    local_tz = pytz.timezone(timezone)
    local_time = pd.Series(
        frame.index.tz_convert(local_tz).floor("30min").time.astype(str),
        index=frame.index,
        dtype="string",
    )
    frame["general_tariff"] = local_time.map(general_tariff_map).fillna(0.0).astype(np.float64)
    frame["feed_in_tariff"] = local_time.map(feed_in_tariff_map).fillna(0.0).astype(np.float64)

    general_price_ex_gst = frame["wholesale_price"] * network_loss_factor + frame["general_tariff"]
    feed_in_price_ex_gst = frame["wholesale_price"] * network_loss_factor + frame["feed_in_tariff"]

    # GST applies to the import leg only, and only when it is a net cost (> 0). The
    # feed-in (export) leg is GST-free in both directions — credits and export charges
    # alike (Amber, from ~FY27; see docs/tariff_gst_regime.md).
    frame["general_price"] = np.where(
        general_price_ex_gst > 0,
        general_price_ex_gst * gst_rate,
        general_price_ex_gst,
    )
    frame["feed_in_price"] = feed_in_price_ex_gst
    frame["general_price_mwh"] = frame["general_price"] * 1000.0
    frame["feed_in_price_mwh"] = frame["feed_in_price"] * 1000.0
    return frame
