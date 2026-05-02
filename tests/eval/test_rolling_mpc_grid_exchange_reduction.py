import pandas as pd

from eval.rolling_mpc_eval import (
    _load_grid_exchange_reduction_signal,
    _should_apply_grid_exchange_reduction,
)


def test_load_grid_exchange_reduction_signal_filters_label_horizon_split_and_score(tmp_path):
    path = tmp_path / "signals.csv"
    pd.DataFrame(
        {
            "time": [
                "2025-09-01T00:00:00Z",
                "2025-09-01T00:00:00Z",
                "2025-09-01T00:05:00Z",
                "2025-09-01T00:10:00Z",
            ],
            "label": [
                "grid_exchange_down",
                "grid_exchange_down",
                "grid_exchange_down",
                "soc_down",
            ],
            "split": ["validation", "validation", "train", "validation"],
            "horizon_steps": [12, 6, 12, 12],
            "y_score": [0.91, 0.99, 0.95, 0.99],
        }
    ).to_csv(path, index=False)

    signal = _load_grid_exchange_reduction_signal(
        str(path),
        label="grid_exchange_down",
        score_col="y_score",
        min_score=0.9,
        horizon_steps=12,
        split="validation",
    )

    assert signal == {pd.Timestamp("2025-09-01T00:00:00Z"): 0.91}


def test_should_apply_grid_exchange_reduction_requires_source_score_and_cost():
    scores = {pd.Timestamp("2025-09-01T00:00:00Z"): 0.91}

    assert _should_apply_grid_exchange_reduction(
        source="model_a_hybrid_grid_exchange_gate",
        ts=pd.Timestamp("2025-09-01T00:00:00Z"),
        allowed_sources={"model_a_hybrid_grid_exchange_gate"},
        cycle_cost_adder_per_kwh=0.05,
        scores_by_time=scores,
    )
    assert not _should_apply_grid_exchange_reduction(
        source="model_a_hybrid",
        ts=pd.Timestamp("2025-09-01T00:00:00Z"),
        allowed_sources={"model_a_hybrid_grid_exchange_gate"},
        cycle_cost_adder_per_kwh=0.05,
        scores_by_time=scores,
    )
    assert not _should_apply_grid_exchange_reduction(
        source="model_a_hybrid_grid_exchange_gate",
        ts=pd.Timestamp("2025-09-01T00:05:00Z"),
        allowed_sources={"model_a_hybrid_grid_exchange_gate"},
        cycle_cost_adder_per_kwh=0.05,
        scores_by_time=scores,
    )
