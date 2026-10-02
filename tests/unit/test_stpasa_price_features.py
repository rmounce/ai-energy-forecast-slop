import numpy as np
import pandas as pd

import forecast as fc


def test_filtered_archive_preserves_historical_asof_and_forecast_features(tmp_path):
    targets = pd.date_range('2026-10-03T15:30:00Z', periods=4, freq='30min')
    rows = []
    for target in [targets[0]-pd.Timedelta(days=10), *targets, targets[-1]+pd.Timedelta(days=10)]:
        for lag in (48, 1, -1):
            rows.append({'interval_dt': target, 'run_time': target-pd.Timedelta(hours=lag),
                         **{col: float(lag+50) for col in fc.STPASA_SOURCE_COLUMNS}})
    path = tmp_path/'archive.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    full = fc._load_stpasa_regionsolution(path)
    filtered = fc._load_stpasa_regionsolution(path, targets=targets.tz_convert('Australia/Adelaide'))
    assert len(filtered) == 12
    base = pd.DataFrame({'total_demand_sa1': 1000.}, index=targets)
    for asof in (targets, targets[0]-pd.Timedelta(hours=2)):
        pd.testing.assert_frame_equal(
            fc._attach_stpasa_features_for_targets(base, full, asof_times=asof),
            fc._attach_stpasa_features_for_targets(base, filtered, asof_times=asof))
    assert fc._load_stpasa_regionsolution(path, targets=targets[:0]).empty


def test_attach_stpasa_features_uses_latest_run_available_at_asof():
    target = pd.Timestamp("2026-05-04T00:00:00Z")
    base = pd.DataFrame(
        {"total_demand_sa1": [1200.0, 1300.0]},
        index=pd.DatetimeIndex([target, target + pd.Timedelta(minutes=30)]),
    )
    stpasa = pd.DataFrame(
        {
            "interval_dt": pd.to_datetime(
                [
                    "2026-05-04T00:00:00Z",
                    "2026-05-04T00:00:00Z",
                    "2026-05-04T00:30:00Z",
                ],
                utc=True,
            ),
            "run_time": pd.to_datetime(
                [
                    "2026-05-02T00:00:00Z",
                    "2026-05-03T00:00:00Z",
                    "2026-05-03T00:00:00Z",
                ],
                utc=True,
            ),
            "uigf": [100.0, 200.0, 300.0],
            "total_intermittent_generation": [90.0, 180.0, 270.0],
            "ss_wind_uigf": [80.0, 160.0, 210.0],
            "ss_solar_uigf": [20.0, 40.0, 90.0],
            "ss_wind_capacity": [100.0, 200.0, 300.0],
            "ss_solar_capacity": [50.0, 100.0, 300.0],
        }
    )

    out = fc._attach_stpasa_features_for_targets(
        base,
        stpasa,
        asof_times=pd.Timestamp("2026-05-03T12:00:00Z"),
    )

    assert out["stpasa_uigf"].tolist() == [200.0, 300.0]
    assert out["stpasa_wind_avail_frac"].tolist() == [0.8, 0.7]
    assert out["stpasa_solar_avail_frac"].tolist() == [0.4, 0.3]
    assert out["stpasa_net_load_proxy"].tolist() == [1000.0, 1000.0]
    assert out["stpasa_source_horizon_hours"].tolist() == [24.0, 24.5]


def test_attach_stpasa_features_leaves_missing_targets_nan():
    base = pd.DataFrame(
        {"total_demand_sa1": [1200.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-05-04T00:00:00Z")]),
    )
    stpasa = pd.DataFrame(
        {
            "interval_dt": pd.to_datetime(["2026-05-04T00:30:00Z"], utc=True),
            "run_time": pd.to_datetime(["2026-05-03T00:00:00Z"], utc=True),
            "uigf": [100.0],
            "total_intermittent_generation": [90.0],
            "ss_wind_uigf": [80.0],
            "ss_solar_uigf": [20.0],
            "ss_wind_capacity": [100.0],
            "ss_solar_capacity": [50.0],
        }
    )

    out = fc._attach_stpasa_features_for_targets(
        base,
        stpasa,
        asof_times=pd.Timestamp("2026-05-03T12:00:00Z"),
    )

    assert np.isnan(out.loc[base.index[0], "stpasa_uigf"])
    assert np.isnan(out.loc[base.index[0], "stpasa_net_load_proxy"])


def test_get_stpasa_forecast_features_uses_forecast_demand(monkeypatch):
    target = pd.Timestamp("2026-05-04T00:00:00Z")
    base = pd.DataFrame(
        {"total_demand_sa1": [1500.0]},
        index=pd.DatetimeIndex([target]),
    )
    stpasa = pd.DataFrame(
        {
            "interval_dt": pd.to_datetime(["2026-05-04T00:00:00Z"], utc=True),
            "run_time": pd.to_datetime(["2026-05-03T00:00:00Z"], utc=True),
            "uigf": [400.0],
            "total_intermittent_generation": [360.0],
            "ss_wind_uigf": [300.0],
            "ss_solar_uigf": [100.0],
            "ss_wind_capacity": [500.0],
            "ss_solar_capacity": [250.0],
        }
    )
    monkeypatch.setattr(fc, "_load_stpasa_regionsolution", lambda **kwargs: stpasa)

    out = fc._get_stpasa_forecast_features(
        base,
        pd.Timestamp("2026-05-03T12:00:00Z"),
    )

    assert out.loc[target, "stpasa_net_load_proxy"] == 1100.0
