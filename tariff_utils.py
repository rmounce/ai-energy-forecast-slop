#!/usr/bin/env python3
"""
Shared tariff helpers for deterministic import/export price reconstruction.

These utilities intentionally implement the same current-tariff assumptions used
by live forecast publication and rolling MPC eval:
  - wholesale price plus network loss factor
  - separate general and feed-in tariff schedules
  - conditional GST on positive import prices and negative feed-in prices

This supports "current-tariff backtest" experiments without requiring a
separately persisted historical effective-rate dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytz


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

    frame["general_price"] = np.where(
        general_price_ex_gst > 0,
        general_price_ex_gst * gst_rate,
        general_price_ex_gst,
    )
    frame["feed_in_price"] = np.where(
        feed_in_price_ex_gst < 0,
        feed_in_price_ex_gst * gst_rate,
        feed_in_price_ex_gst,
    )
    frame["general_price_mwh"] = frame["general_price"] * 1000.0
    frame["feed_in_price_mwh"] = frame["feed_in_price"] * 1000.0
    return frame
