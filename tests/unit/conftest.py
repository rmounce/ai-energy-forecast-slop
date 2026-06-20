"""Shared fixtures for unit tests. No InfluxDB or HA calls."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import forecast module once — config.yaml must exist at ROOT
import forecast as fc


# ── Minimal CONFIG fixture ────────────────────────────────────────────────────

MINIMAL_CONFIG = {
    "timezone": "Australia/Adelaide",
    "gst_rate": 1.1,
    "paths": {
        "tariff_file": str(ROOT / "tariff_profile.json"),
    },
}


@pytest.fixture()
def patched_config(monkeypatch):
    """Patch forecast.CONFIG to a known minimal dict."""
    monkeypatch.setattr(fc, "CONFIG", MINIMAL_CONFIG)
    return MINIMAL_CONFIG


@pytest.fixture()
def fixed_loss_factor(monkeypatch):
    """Pin network loss factor to 1.05 so tariff tests are deterministic."""
    monkeypatch.setattr(fc, "get_network_loss_factor", lambda: 1.05)
    return 1.05


# ── Sample price DataFrames ───────────────────────────────────────────────────

def _make_price_df(wholesale_values, freq="5min"):
    """Create a UTC DatetimeIndex DataFrame with 'wholesale_price' column ($/kWh)."""
    idx = pd.date_range("2025-06-01 00:00", periods=len(wholesale_values), freq=freq, tz="UTC")
    return pd.DataFrame({"wholesale_price": wholesale_values}, index=idx)


@pytest.fixture()
def tier1_dfs():
    """12-step 5-min DataFrames (positive prices, ~50 $/MWh = 0.05 $/kWh)."""
    vals = [0.05] * 12
    return {
        "p50": _make_price_df(vals, "5min"),
        "low": _make_price_df([v * 0.8 for v in vals], "5min"),
        "high": _make_price_df([v * 1.2 for v in vals], "5min"),
    }


@pytest.fixture()
def tier2_dfs():
    """144-step 30-min DataFrames starting 1h after Tier 1."""
    idx = pd.date_range("2025-06-01 01:00", periods=144, freq="30min", tz="UTC")
    vals = [0.08] * 144
    return {
        "p50": pd.DataFrame({"wholesale_price": vals}, index=idx),
        "low": pd.DataFrame({"wholesale_price": [v * 0.7 for v in vals]}, index=idx),
        "high": pd.DataFrame({"wholesale_price": [v * 1.3 for v in vals]}, index=idx),
    }
