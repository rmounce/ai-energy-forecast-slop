import pandas as pd

from eval.analyze_lgbm_residual_drivers import (
    adelaide_bucket,
    horizon_bucket,
    latest_asof_by_target,
)


def test_latest_asof_by_target_uses_same_interval_and_latest_prior_run():
    left = pd.DataFrame(
        {
            "forecast_target_time": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:30:00Z",
                ],
                utc=True,
            ),
            "forecast_creation_time": pd.to_datetime(
                [
                    "2026-01-01T00:20:00Z",
                    "2026-01-01T00:50:00Z",
                    "2026-01-01T00:50:00Z",
                ],
                utc=True,
            ),
        }
    )
    right = pd.DataFrame(
        {
            "interval_dt": pd.to_datetime(
                [
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T01:30:00Z",
                ],
                utc=True,
            ),
            "run_time": pd.to_datetime(
                [
                    "2026-01-01T00:10:00Z",
                    "2026-01-01T00:40:00Z",
                    "2026-01-01T00:55:00Z",
                ],
                utc=True,
            ),
            "total_demand": [1000.0, 1100.0, 1200.0],
        }
    )

    out = latest_asof_by_target(
        left,
        right,
        left_target="forecast_target_time",
        left_time="forecast_creation_time",
        right_target="interval_dt",
        right_time="run_time",
        value_cols=["total_demand"],
        prefix="pd",
    )

    assert out["pd_total_demand"].iloc[:2].tolist() == [1000.0, 1100.0]
    assert pd.isna(out["pd_total_demand"].iloc[2])


def test_time_bucket_helpers_are_stable():
    targets = pd.Series(pd.to_datetime(["2026-06-14T09:00:00Z"], utc=True))
    assert adelaide_bucket(targets).iloc[0] == "evening"

    horizons = pd.Series([0.5, 3.0, 8.0, 18.0, 30.0])
    assert horizon_bucket(horizons).tolist() == ["0-1h", "1-4h", "4-12h", "12-24h", "24h+"]
