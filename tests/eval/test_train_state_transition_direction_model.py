import pandas as pd
import pytest

from eval.train_state_transition_direction_model import add_direction_labels


def test_add_direction_labels_applies_deadbands_and_grid_exchange():
    df = pd.DataFrame(
        {
            "oracle_minus_target_step_pnl": [0.02, -0.02, 0.0],
            "oracle_minus_target_soc_delta_kwh": [0.2, -0.2, 0.0],
            "oracle_minus_target_throughput_kwh": [-0.2, 0.2, 0.0],
            "oracle_minus_target_import_kwh": [-0.2, 0.2, 0.0],
            "oracle_minus_target_export_kwh": [-0.3, 0.1, 0.0],
            "oracle_minus_target_curtail_kwh": [0.2, -0.2, 0.0],
        }
    )

    out, labels = add_direction_labels(df, pnl_deadband=0.01, kwh_deadband=0.1)

    assert "pnl_gain" in labels
    assert "grid_exchange_down" in labels
    assert out["pnl_gain"].tolist() == [1.0, 0.0, 0.0]
    assert out["pnl_loss"].tolist() == [0.0, 1.0, 0.0]
    assert out["soc_up"].tolist() == [1.0, 0.0, 0.0]
    assert out["soc_down"].tolist() == [0.0, 1.0, 0.0]
    assert out["throughput_down"].tolist() == [1.0, 0.0, 0.0]
    assert out["export_down"].tolist() == [1.0, 0.0, 0.0]
    assert out["curtail_up"].tolist() == [1.0, 0.0, 0.0]
    assert out["curtail_down"].tolist() == [0.0, 1.0, 0.0]
    assert out["oracle_minus_target_grid_exchange_kwh"].tolist() == pytest.approx([-0.5, 0.3, 0.0])
    assert out["grid_exchange_down"].tolist() == [1.0, 0.0, 0.0]
    assert out["grid_exchange_up"].tolist() == [0.0, 1.0, 0.0]
