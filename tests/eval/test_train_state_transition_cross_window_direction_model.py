import pandas as pd

from eval.train_state_transition_cross_window_direction_model import _label_distribution


def test_label_distribution_reports_positive_rate_and_context_means():
    df = pd.DataFrame(
        {
            "pnl_gain": [1.0, 0.0, 1.0],
            "actual_feed_in_price_mwh": [10.0, 20.0, 30.0],
            "actual_net_load_kw": [-1.0, 0.0, 1.0],
        }
    )

    out = _label_distribution(df, ["pnl_gain"], split="cross_window")

    assert out.to_dict(orient="records") == [
        {
            "split": "cross_window",
            "label": "pnl_gain",
            "rows": 3,
            "positive_rate": 2.0 / 3.0,
            "mean_actual_feed_in_price_mwh": 20.0,
            "mean_actual_net_load_kw": 0.0,
        }
    ]
