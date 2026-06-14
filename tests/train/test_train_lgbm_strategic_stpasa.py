import numpy as np
import pandas as pd

from train.train_lgbm_strategic import attach_stpasa_features


def test_attach_stpasa_features_uses_latest_source_run_before_model_run():
    rows = pd.DataFrame(
        {
            "run_time": pd.to_datetime(
                ["2026-04-01T01:00:00Z", "2026-04-01T02:00:00Z"],
                utc=True,
            ),
            "interval_dt": pd.to_datetime(
                ["2026-04-03T00:00:00Z", "2026-04-03T00:00:00Z"],
                utc=True,
            ),
            "step_idx": [93, 91],
        }
    )
    stpasa = pd.DataFrame(
        {
            "run_time": pd.to_datetime(
                [
                    "2026-04-01T00:30:00Z",
                    "2026-04-01T01:30:00Z",
                    "2026-04-01T02:30:00Z",
                ],
                utc=True,
            ),
            "interval_dt": pd.to_datetime(["2026-04-03T00:00:00Z"] * 3, utc=True),
            "uigf": [100.0, 200.0, 300.0],
            "total_intermittent_generation": [90.0, 190.0, 290.0],
            "ss_wind_uigf": [80.0, 160.0, 240.0],
            "ss_solar_uigf": [20.0, 40.0, 60.0],
            "ss_wind_capacity": [100.0, 200.0, 300.0],
            "ss_solar_capacity": [50.0, 100.0, 150.0],
        }
    )

    out = attach_stpasa_features(rows, stpasa)

    assert out["stpasa_uigf"].tolist() == [100.0, 200.0]
    assert out["stpasa_ss_wind_uigf"].tolist() == [80.0, 160.0]
    assert out["stpasa_wind_avail_frac"].tolist() == [0.8, 0.8]
    assert out["stpasa_solar_avail_frac"].tolist() == [0.4, 0.4]
    assert out["stpasa_source_horizon_hours"].tolist() == [47.5, 46.5]


def test_attach_stpasa_features_leaves_missing_target_as_nan():
    rows = pd.DataFrame(
        {
            "run_time": pd.to_datetime(["2026-04-01T01:00:00Z"], utc=True),
            "interval_dt": pd.to_datetime(["2026-04-03T00:00:00Z"], utc=True),
            "step_idx": [93],
        }
    )
    stpasa = pd.DataFrame(
        {
            "run_time": pd.to_datetime(["2026-04-01T00:30:00Z"], utc=True),
            "interval_dt": pd.to_datetime(["2026-04-04T00:00:00Z"], utc=True),
            "uigf": [100.0],
            "total_intermittent_generation": [90.0],
            "ss_wind_uigf": [80.0],
            "ss_solar_uigf": [20.0],
            "ss_wind_capacity": [100.0],
            "ss_solar_capacity": [50.0],
        }
    )

    out = attach_stpasa_features(rows, stpasa)

    assert np.isnan(out.loc[0, "stpasa_uigf"])
    assert np.isnan(out.loc[0, "stpasa_source_horizon_hours"])
