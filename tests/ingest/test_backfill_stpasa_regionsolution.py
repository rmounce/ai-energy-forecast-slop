import pandas as pd
import pytest

from ingest.backfill_stpasa_regionsolution import (
    OUTPUT_COLUMNS,
    horizon_summary,
    normalise_regionsolution,
    validate_horizon,
)


def test_normalise_regionsolution_filters_region_and_converts_nem_time_to_utc():
    raw = pd.DataFrame(
        {
            "RUN_DATETIME": ["2026/04/30 00:00:00", "2026/04/30 00:00:00"],
            "INTERVAL_DATETIME": ["2026/05/03 00:00:00", "2026/05/03 00:00:00"],
            "REGIONID": ["SA1", "VIC1"],
            "INTERVENTION": ["0", "0"],
            "UIGF": ["1200.5", "9999"],
            "SS_WIND_UIGF": ["800", "9999"],
            "SS_SOLAR_UIGF": ["400.5", "9999"],
        }
    )

    out = normalise_regionsolution(raw, region_id="SA1")

    assert out.columns.tolist() == OUTPUT_COLUMNS
    assert len(out) == 1
    assert out.loc[0, "run_time"] == pd.Timestamp("2026-04-29T14:00:00Z")
    assert out.loc[0, "interval_dt"] == pd.Timestamp("2026-05-02T14:00:00Z")
    assert out.loc[0, "uigf"] == 1200.5
    assert out.loc[0, "ss_wind_uigf"] == 800.0
    assert out.loc[0, "ss_solar_uigf"] == 400.5


def test_validate_horizon_requires_at_least_one_run_to_reach_threshold():
    df = pd.DataFrame(
        {
            "run_time": pd.to_datetime(
                ["2026-04-30T14:00:00Z", "2026-04-30T14:00:00Z"],
                utc=True,
            ),
            "interval_dt": pd.to_datetime(
                ["2026-05-01T18:30:00Z", "2026-05-03T14:00:00Z"],
                utc=True,
            ),
        }
    )

    validate_horizon(df, min_horizon_hours=72.0)
    summary = horizon_summary(df)
    assert summary["min_horizon_hours"].iloc[0] == 28.5
    assert summary["max_horizon_hours"].iloc[0] == 72.0

    with pytest.raises(ValueError, match="1/1 STPASA runs stop before 73.0h"):
        validate_horizon(df, min_horizon_hours=73.0)
