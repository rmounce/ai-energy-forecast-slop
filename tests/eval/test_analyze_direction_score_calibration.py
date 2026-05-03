import numpy as np
import pandas as pd
import pytest

from eval.analyze_direction_score_calibration import (
    expected_calibration_error,
    top_fraction_summary,
)


def test_expected_calibration_error_weights_bin_gaps():
    y_true = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    ece = expected_calibration_error(y_true, y_prob, n_bins=2)

    assert ece == pytest.approx(0.5 * abs(0.5 - 0.15) + 0.5 * abs(1.0 - 0.85))


def test_top_fraction_summary_reports_precision_lift():
    df = pd.DataFrame(
        {
            "y_true": [1, 0, 1, 0],
            "calibrated_score": [0.9, 0.8, 0.2, 0.1],
        }
    )

    out = top_fraction_summary(
        df,
        score_col="calibrated_score",
        label="grid_exchange_down",
        horizon_steps=-1,
        fractions=[0.5],
    )

    row = out.iloc[0]
    assert row["selected_rows"] == 2
    assert row["base_rate"] == pytest.approx(0.5)
    assert row["precision"] == pytest.approx(0.5)
    assert row["recall"] == pytest.approx(0.5)
    assert row["precision_lift"] == pytest.approx(1.0)
