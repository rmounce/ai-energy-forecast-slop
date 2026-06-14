import numpy as np
import pandas as pd
import pytest

from eval.ablate_stpasa_tail_features import (
    add_stpasa_derived_features,
    chronological_split,
    metric_rows,
)


def test_chronological_split_uses_later_rows_for_validation():
    df = pd.DataFrame(
        {
            "forecast_creation_time": pd.to_datetime(
                [
                    "2026-04-01T00:00:00Z",
                    "2026-04-02T00:00:00Z",
                    "2026-04-03T00:00:00Z",
                    "2026-04-04T00:00:00Z",
                ],
                utc=True,
            )
        }
    )

    train_mask, val_mask, cutoff = chronological_split(
        df,
        time_col="forecast_creation_time",
        val_fraction=0.25,
    )

    assert cutoff == pd.Timestamp("2026-04-04T00:00:00Z")
    assert train_mask.tolist() == [True, True, True, False]
    assert val_mask.tolist() == [False, False, False, True]


def test_chronological_split_rejects_invalid_fraction():
    df = pd.DataFrame(
        {"forecast_creation_time": pd.to_datetime(["2026-04-01", "2026-04-02"], utc=True)}
    )

    with pytest.raises(ValueError, match="val_fraction"):
        chronological_split(df, time_col="forecast_creation_time", val_fraction=1.0)


def test_add_stpasa_derived_features_handles_zero_capacity_as_nan():
    df = pd.DataFrame(
        {
            "forecast_target_time": pd.to_datetime(["2026-04-03T00:00:00Z"], utc=True),
            "stpasa_run_time": pd.to_datetime(["2026-04-01T00:00:00Z"], utc=True),
            "stpasa_ss_wind_uigf": [250.0],
            "stpasa_ss_wind_capacity": [500.0],
            "stpasa_ss_solar_uigf": [100.0],
            "stpasa_ss_solar_capacity": [0.0],
            "stpasa_uigf": [350.0],
            "sdo_scheduled_demand": [1300.0],
        }
    )

    out = add_stpasa_derived_features(df)

    assert out.loc[0, "stpasa_source_horizon_hours"] == 48.0
    assert out.loc[0, "stpasa_wind_avail_frac"] == 0.5
    assert np.isnan(out.loc[0, "stpasa_solar_avail_frac"])
    assert out.loc[0, "stpasa_net_load_proxy"] == 950.0


def test_metric_rows_reports_mae_delta_for_corrected_forecast():
    df = pd.DataFrame(
        {
            "horizon_bucket": ["24h+", "24h+"],
            "pred_mwh": [100.0, 200.0],
            "corrected_mwh": [90.0, 180.0],
            "actual_rrp": [80.0, 190.0],
        }
    )

    rows = metric_rows(df, model_name="candidate", split_name="val")
    all_row = next(row for row in rows if row["horizon_bucket"] == "all")

    assert all_row["original_mae"] == 15.0
    assert all_row["corrected_mae"] == 10.0
    assert all_row["mae_delta"] == -5.0
