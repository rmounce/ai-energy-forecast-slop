import pandas as pd

from eval.build_stpasa_residual_price_log import apply_tail_adjustments


def test_apply_tail_adjustments_replaces_matching_rows_only():
    base = pd.DataFrame(
        {
            "forecast_creation_time": [
                "2026-05-01T00:00:00Z",
                "2026-05-01T00:00:00Z",
                "2026-05-01T00:30:00Z",
            ],
            "forecast_target_time": [
                "2026-05-02T06:30:00Z",
                "2026-05-02T07:00:00Z",
                "2026-05-02T07:00:00Z",
            ],
            "model_name": ["price", "price", "price"],
            "prediction": [0.10, 0.20, 0.30],
        }
    )
    adjustments = pd.DataFrame(
        {
            "forecast_creation_time": pd.to_datetime(
                ["2026-05-01T00:00:00Z"], utc=True
            ),
            "forecast_target_time": pd.to_datetime(
                ["2026-05-02T07:00:00Z"], utc=True
            ),
            "corrected_prediction": [0.123],
        }
    )

    out, replaced = apply_tail_adjustments(base, adjustments)

    assert replaced == 1
    assert out["prediction"].tolist() == [0.10, 0.123, 0.30]
    assert "corrected_prediction" not in out.columns
