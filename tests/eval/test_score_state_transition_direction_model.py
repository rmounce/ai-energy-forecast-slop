import numpy as np
import pandas as pd

from eval.score_state_transition_direction_model import build_inference_features, load_raw_rows


def test_load_raw_rows_duplicates_requested_horizons(tmp_path):
    path = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-09-01T00:00:00Z", "2025-09-01T00:05:00Z"], utc=True),
            "source": ["model_a_hybrid", "other"],
            "soc_prev_kwh": [20.0, 21.0],
        }
    ).to_parquet(path, index=False)

    out = load_raw_rows(path, source="model_a_hybrid", horizon_steps=[6, 12])

    assert out["horizon_steps"].tolist() == [6, 12]
    assert out["source"].tolist() == ["model_a_hybrid", "model_a_hybrid"]
    assert out["label_file"].tolist() == ["raw.parquet", "raw.parquet"]


def test_build_inference_features_adds_time_features_and_missing_columns():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-09-01T00:00:00Z"], utc=True),
            "horizon_steps": [12],
            "soc_prev_kwh": [20.0],
        }
    )

    X = build_inference_features(df, ["horizon_steps", "soc_prev_kwh", "actual_load_kw", "time_sin"])

    assert X["horizon_steps"].iloc[0] == 12
    assert X["soc_prev_kwh"].iloc[0] == 20.0
    assert np.isnan(X["actual_load_kw"].iloc[0])
    assert np.isfinite(X["time_sin"].iloc[0])
