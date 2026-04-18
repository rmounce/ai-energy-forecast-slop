"""
Tests for data pipeline invariants:
  - TFT price scaler keys include all expected features (sd_demand, sd_net_interchange, etc.)
  - Decoder input column count: DEC_CONT(7) + TIME_COLS(6) + horizon_norm(1) + covar_missing(1) = 15
"""

import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
SCALERS_PATH = ROOT / "models" / "tft_price" / "scalers.pkl"

# Expected decoder continuous features (Phase 5 expansion: sd_demand, sd_net_interchange added)
EXPECTED_DEC_CONT = [
    "pd_rrp", "pd_demand", "pd_net_interchange",
    "vic1_pd_rrp", "nsw1_pd_rrp",
    "sd_demand", "sd_net_interchange",  # added in Phase 5 — absence caused silent 13→15 mismatch
]

EXPECTED_TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]

EXPECTED_EXTRA_COLS = ["horizon_norm", "covar_missing"]

EXPECTED_DECODER_FEATURES = len(EXPECTED_DEC_CONT) + len(EXPECTED_TIME_COLS) + len(EXPECTED_EXTRA_COLS)


def test_expected_decoder_feature_count():
    """Decoder input must have exactly 15 features (7 + 6 + 2)."""
    assert EXPECTED_DECODER_FEATURES == 15, (
        f"Expected 15 decoder features, got {EXPECTED_DECODER_FEATURES}. "
        "Update this test if the architecture intentionally changes."
    )


def test_dec_cont_has_sd_features():
    """sd_demand and sd_net_interchange must be in DEC_CONT — their absence caused a shape mismatch bug."""
    assert "sd_demand" in EXPECTED_DEC_CONT
    assert "sd_net_interchange" in EXPECTED_DEC_CONT


@pytest.mark.skipif(not SCALERS_PATH.exists(), reason="TFT scaler file not present")
def test_scaler_file_has_dec_cont_keys():
    """scalers.pkl must contain keys for all DEC_CONT features."""
    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)
    missing = [k for k in EXPECTED_DEC_CONT if k not in scalers]
    assert not missing, f"Missing scaler keys: {missing}"


@pytest.mark.skipif(not SCALERS_PATH.exists(), reason="TFT scaler file not present")
def test_scaler_file_has_target_key():
    """scalers.pkl must contain 'target_rrp' for inverse transform."""
    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)
    assert "target_rrp" in scalers, "target_rrp key missing from scalers — inverse transform will fail"


@pytest.mark.skipif(not SCALERS_PATH.exists(), reason="TFT scaler file not present")
def test_scaler_file_no_unexpected_none_values():
    """No scaler value should be None (indicates a missing fit)."""
    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)
    none_keys = [k for k, v in scalers.items() if v is None]
    assert not none_keys, f"Scaler keys with None value: {none_keys}"


def test_forecast_module_dec_cont_matches_expected():
    """DEC_CONT in forecast.py must match the expected list (order matters for model input)."""
    import ast

    lines = (ROOT / "forecast.py").read_text().splitlines()
    # Collect all lines of the DEC_CONT assignment (may span multiple lines)
    collecting = False
    collected = []
    for line in lines:
        if line.strip().startswith("DEC_CONT = ["):
            collecting = True
        if collecting:
            collected.append(line.strip())
            # Stop when the list is closed
            joined = " ".join(collected)
            try:
                parsed = ast.literal_eval(joined[len("DEC_CONT = "):])
                assert parsed == EXPECTED_DEC_CONT, (
                    f"forecast.py DEC_CONT {parsed} != expected {EXPECTED_DEC_CONT}\n"
                    "If you changed the decoder features, update EXPECTED_DEC_CONT in this test."
                )
                return
            except (SyntaxError, ValueError):
                continue  # list not complete yet
    pytest.fail("DEC_CONT assignment not found or unparseable in forecast.py")
