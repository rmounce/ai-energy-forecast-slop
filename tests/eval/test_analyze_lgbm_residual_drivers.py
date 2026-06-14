import pandas as pd

from eval.analyze_lgbm_residual_drivers import (
    adelaide_bucket,
    horizon_bucket,
    latest_asof_by_target,
    validate_join_coverage,
    validate_source_horizon,
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


def test_validate_source_horizon_rejects_short_tail_source():
    source = pd.DataFrame(
        {
            "run_time": pd.to_datetime(["2026-06-14T00:00:00Z"], utc=True),
            "interval_dt": pd.to_datetime(["2026-06-15T05:00:00Z"], utc=True),
        }
    )

    try:
        validate_source_horizon(source, source_name="PDPASA", min_horizon_hours=72.0)
    except ValueError as exc:
        assert "1/1 PDPASA runs stop before 72.0h" in str(exc)
        assert "shortest max horizon is 29.0h" in str(exc)
    else:
        raise AssertionError("expected short-horizon source to fail validation")


def test_validate_join_coverage_checks_tail_band_only():
    df = pd.DataFrame(
        {
            "horizon_hours": [12.0, 28.5, 40.0, 72.0],
            "stpasa_uigf": [pd.NA, 100.0, 200.0, pd.NA],
        }
    )

    coverage = validate_join_coverage(
        df,
        prefix="stpasa",
        value_col="uigf",
        min_horizon_hours=28.5,
        max_horizon_hours=72.0,
        min_coverage=0.5,
    )
    assert coverage == 2 / 3

    try:
        validate_join_coverage(
            df,
            prefix="stpasa",
            value_col="uigf",
            min_horizon_hours=28.5,
            max_horizon_hours=72.0,
            min_coverage=0.95,
        )
    except ValueError as exc:
        assert "coverage is 66.7%" in str(exc)
    else:
        raise AssertionError("expected insufficient STPASA coverage to fail validation")
