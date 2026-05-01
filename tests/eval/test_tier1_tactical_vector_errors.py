from __future__ import annotations

import pandas as pd

from eval.analyze_tier1_tactical_vector_errors import (
    build_pairwise_curve,
    summarize_pairwise_curve,
    summarize_pairwise_horizon,
)


def test_pairwise_curve_summary_reports_act_now_match_by_bucket():
    rows = [
        {
            "time": pd.Timestamp("2025-09-01T00:00:00Z"),
            "source": "amber_tactical_hybrid_strategic",
            "dispatch_bucket": "fit_lt_300_negload",
            "utc_date": "2025-09-01",
            "actual_feed_in_price_mwh": 100.0,
            "actual_general_price_mwh": 200.0,
            "actual_net_load_kw": -1.0,
            "step_pnl": 0.2,
            "charge_kwh": 0.0,
            "discharge_kwh": 0.1,
            "realized_import_kwh": 0.0,
            "realized_export_kwh": 0.2,
            "curtail_kwh": 0.0,
            "soc_kwh": 19.9,
            "forecast_feed_in_mean_1h_mwh": 150.0,
            "actual_feed_in_mean_1h_mwh": 140.0,
            "forecast_feed_in_argmax_1h_step": 2,
            "actual_feed_in_argmax_1h_step": 3,
            "forecast_feed_in_act_now": False,
            "actual_feed_in_act_now": False,
        },
        {
            "time": pd.Timestamp("2025-09-01T00:00:00Z"),
            "source": "model_a_hybrid",
            "dispatch_bucket": "fit_lt_300_negload",
            "utc_date": "2025-09-01",
            "actual_feed_in_price_mwh": 100.0,
            "actual_general_price_mwh": 200.0,
            "actual_net_load_kw": -1.0,
            "step_pnl": 0.0,
            "charge_kwh": 0.1,
            "discharge_kwh": 0.0,
            "realized_import_kwh": 0.1,
            "realized_export_kwh": 0.0,
            "curtail_kwh": 0.0,
            "soc_kwh": 20.1,
            "forecast_feed_in_mean_1h_mwh": 120.0,
            "actual_feed_in_mean_1h_mwh": 140.0,
            "forecast_feed_in_argmax_1h_step": 1,
            "actual_feed_in_argmax_1h_step": 3,
            "forecast_feed_in_act_now": True,
            "actual_feed_in_act_now": False,
        },
    ]
    pairwise = build_pairwise_curve(
        pd.DataFrame(rows),
        source_a="amber_tactical_hybrid_strategic",
        source_b="model_a_hybrid",
    )
    summary = summarize_pairwise_curve(pairwise)
    bucket = summary[summary["bucket"] == "fit_lt_300_negload"].iloc[0]
    assert bucket["mean_a_minus_b_step_pnl"] == 0.2
    assert bucket["mean_a_minus_b_realized_export_kwh"] == 0.2
    assert bucket["p_a_feed_in_act_now_match"] == 1.0
    assert bucket["p_b_feed_in_act_now_match"] == 0.0


def test_pairwise_horizon_summary_keeps_horizon_dimension():
    rows = []
    for horizon in [0, 1]:
        rows.append(
                {
                    "time": pd.Timestamp("2025-09-01T00:00:00Z"),
                    "horizon": horizon,
                    "dispatch_bucket": "fit_lt_300_negload",
                    "utc_date": "2025-09-01",
                    "actual_feed_in_price_mwh": 100.0,
                    "a_forecast_feed_in_mwh": 100.0 + horizon,
                "b_forecast_feed_in_mwh": 90.0 + horizon,
                "a_minus_b_forecast_feed_in_mwh": 10.0,
                "a_feed_in_error_mwh": 5.0,
                "b_feed_in_error_mwh": -5.0,
                "a_minus_b_feed_in_error_mwh": 10.0,
                "a_forecast_general_mwh": 200.0,
                "b_forecast_general_mwh": 180.0,
                "a_minus_b_forecast_general_mwh": 20.0,
                "a_general_error_mwh": 8.0,
                "b_general_error_mwh": -2.0,
                "a_minus_b_general_error_mwh": 10.0,
            }
        )
    summary = summarize_pairwise_horizon(pd.DataFrame(rows))
    assert list(summary[summary["bucket"] == "fit_lt_300_negload"]["horizon"]) == [0, 1]
