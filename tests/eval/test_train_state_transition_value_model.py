import pandas as pd
import pytest

from eval.train_state_transition_value_model import build_tier1_vector_feature_frame


def test_build_tier1_vector_feature_frame_uses_forecast_curve_only():
    times = pd.to_datetime(["2025-09-01T00:00:00Z", "2025-09-01T00:05:00Z"], utc=True)
    rows = []
    for source in ["model_a_hybrid", "amber_tactical_hybrid_strategic"]:
        for time_idx, ts in enumerate(times):
            for horizon in range(12):
                rows.append(
                    {
                        "time": ts,
                        "source": source,
                        "horizon": horizon,
                        "forecast_general_mwh": 100.0 + time_idx + horizon,
                        "forecast_feed_in_mwh": 50.0 + time_idx - horizon,
                        "actual_general_mwh": -999.0,
                        "actual_feed_in_mwh": -999.0,
                    }
                )
    vector_df = pd.DataFrame(rows)

    features, feature_cols = build_tier1_vector_feature_frame(vector_df, source="model_a_hybrid")

    assert len(features) == 2
    assert "tier1_vector_general_h11_mwh" in feature_cols
    assert "tier1_vector_feed_in_h11_minus_h00_mwh" in feature_cols
    assert "tier1_vector_feed_in_range_1h_mwh" in feature_cols
    assert "actual_general_mwh" not in features.columns
    assert "actual_feed_in_mwh" not in features.columns
    first = features.sort_values("time").iloc[0]
    assert first["tier1_vector_general_h11_mwh"] == pytest.approx(111.0)
    assert first["tier1_vector_feed_in_h11_minus_h00_mwh"] == pytest.approx(-11.0)
    assert first["tier1_vector_feed_in_range_1h_mwh"] == pytest.approx(11.0)
    assert first["tier1_vector_general_argmax_1h_step"] == pytest.approx(11.0)
    assert first["tier1_vector_feed_in_argmin_1h_step"] == pytest.approx(11.0)


def test_build_tier1_vector_feature_frame_requires_selected_source():
    vector_df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2025-09-01T00:00:00Z"], utc=True),
            "source": ["other"],
            "horizon": [0],
            "forecast_general_mwh": [1.0],
            "forecast_feed_in_mwh": [1.0],
        }
    )

    with pytest.raises(ValueError, match="No vector rows found"):
        build_tier1_vector_feature_frame(vector_df, source="model_a_hybrid")
