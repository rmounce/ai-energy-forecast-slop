import numpy as np
import pandas as pd

from eval.retro_lgbm_strategic_inference import (
    BASE_FEATURE_NAMES,
    STPASA_FEATURE_NAMES,
    build_window_features,
)


def test_build_window_features_supports_stpasa_contract():
    run_time = pd.Timestamp("2026-04-01T00:00:00Z")
    pd_rows = pd.DataFrame(
        {
            "run_time": [run_time],
            "step_idx": [0],
            "interval_dt": [run_time + pd.Timedelta(minutes=30)],
            "rrp": [50.0],
            "total_demand": [1300.0],
            "net_interchange": [-200.0],
        }
    ).set_index(["run_time", "step_idx"])
    empty_oof = pd.Series(
        dtype=float,
        index=pd.MultiIndex.from_tuples([], names=["run_time", "interval_dt"]),
    )
    actual_index = pd.date_range(
        run_time - pd.Timedelta(hours=24),
        run_time,
        freq="30min",
        tz="UTC",
    )
    actuals = pd.Series(np.linspace(10.0, 20.0, len(actual_index)), index=actual_index)
    roll_6h = actuals.rolling(12, min_periods=1).max()
    roll_24h = actuals.rolling(48, min_periods=1).max()

    tail_target = run_time + pd.Timedelta(hours=72)
    stpasa_by_target = {
        tail_target: pd.DataFrame(
            {
                "run_time": pd.to_datetime(["2026-03-31T23:00:00Z"], utc=True),
                "uigf": [500.0],
                "total_intermittent_generation": [480.0],
                "ss_wind_uigf": [400.0],
                "ss_solar_uigf": [100.0],
                "ss_wind_capacity": [800.0],
                "ss_solar_capacity": [200.0],
            }
        )
    }
    feature_names = list(BASE_FEATURE_NAMES) + STPASA_FEATURE_NAMES

    out = build_window_features(
        run_time,
        pd_rows,
        empty_oof,
        actuals,
        roll_6h,
        roll_24h,
        feature_names=feature_names,
        stpasa_by_target=stpasa_by_target,
    )

    assert out.shape == (144, len(feature_names))
    assert out[0, feature_names.index("pd_rrp_debiased")] == 50.0
    assert np.isnan(out[0, feature_names.index("stpasa_uigf")])
    assert out[143, feature_names.index("stpasa_uigf")] == 500.0
    assert out[143, feature_names.index("stpasa_wind_avail_frac")] == 0.5
    assert out[143, feature_names.index("stpasa_source_horizon_hours")] == 73.0
