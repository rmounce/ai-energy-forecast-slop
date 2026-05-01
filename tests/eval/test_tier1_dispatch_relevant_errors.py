from __future__ import annotations

import numpy as np
import pandas as pd

from eval.analyze_tier1_dispatch_relevant_errors import (
    add_actual_horizon_metrics,
    add_buckets,
    add_forecast_error_metrics,
    analyze,
    bucket_label,
)


def _synthetic_raw() -> pd.DataFrame:
    times = pd.date_range("2025-09-01 00:00:00+00:00", periods=4, freq="5min")
    actual_feed_in = [100.0, 200.0, 400.0, 50.0]
    actual_general = [150.0, 250.0, 500.0, 100.0]
    rows = []
    for source in ["amber_tactical_hybrid_strategic", "model_a_hybrid"]:
        for i, ts in enumerate(times):
            is_amber = source == "amber_tactical_hybrid_strategic"
            rows.append(
                {
                    "time": ts,
                    "source": source,
                    "actual_price_mwh": actual_feed_in[i],
                    "actual_general_price_mwh": actual_general[i],
                    "actual_feed_in_price_mwh": actual_feed_in[i],
                    "actual_net_load_kw": -1.0 if i != 2 else 1.0,
                    "actual_load_kw": 1.0,
                    "actual_pv_kw": 2.0 if i != 2 else 0.0,
                    "forecast_step0_mwh": actual_feed_in[i],
                    "forecast_mean_next_1h_mwh": 190.0 if is_amber else 150.0,
                    "forecast_mean_next_4h_mwh": 190.0 if is_amber else 150.0,
                    "forecast_mean_next_14h_mwh": 190.0 if is_amber else 150.0,
                    "forecast_general_step0_mwh": actual_general[i],
                    "forecast_buy_mean_next_1h_mwh": 250.0 if is_amber else 125.0,
                    "forecast_buy_mean_next_4h_mwh": 250.0 if is_amber else 125.0,
                    "forecast_buy_mean_next_14h_mwh": 250.0 if is_amber else 125.0,
                    "forecast_feed_in_step0_mwh": actual_feed_in[i],
                    "forecast_feed_in_mean_next_1h_mwh": 200.0 if is_amber else 120.0,
                    "forecast_feed_in_mean_next_4h_mwh": 200.0 if is_amber else 120.0,
                    "forecast_feed_in_mean_next_14h_mwh": 200.0 if is_amber else 120.0,
                    "charge_kw": 0.0 if is_amber else 1.0,
                    "discharge_kw": 1.0 if is_amber else 0.0,
                    "grid_import_kw": 0.0 if is_amber else 1.0,
                    "grid_export_kw": 1.0 if is_amber else 0.0,
                    "realized_grid_import_kw": 0.0 if is_amber else 1.0,
                    "realized_grid_export_kw": 1.0 if is_amber else 0.0,
                    "curtail_kw": 0.0,
                    "soc_prev_kwh": 20.0,
                    "soc_kwh": 19.9 if is_amber else 20.1,
                    "step_pnl": 0.1 if is_amber else -0.1,
                }
            )
    return pd.DataFrame(rows)


def test_bucket_label_key_regimes():
    assert (
        bucket_label(pd.Series({"actual_feed_in_price_mwh": 100.0, "actual_net_load_kw": -0.1}))
        == "fit_lt_300_negload"
    )
    assert (
        bucket_label(pd.Series({"actual_feed_in_price_mwh": 100.0, "actual_net_load_kw": 0.0}))
        == "fit_lt_300_nonnegload"
    )
    assert (
        bucket_label(pd.Series({"actual_feed_in_price_mwh": 300.0, "actual_net_load_kw": -0.1}))
        == "fit_gte_300_negload"
    )


def test_forward_actual_horizon_metrics_use_future_rows_once_per_timestamp():
    enriched = add_actual_horizon_metrics(_synthetic_raw())
    first = enriched[enriched["time"] == pd.Timestamp("2025-09-01 00:00:00+00:00")].iloc[0]
    assert first["actual_feed_in_mean_next_1h_mwh"] == np.mean([100.0, 200.0, 400.0, 50.0])
    assert first["actual_feed_in_max_next_1h_mwh"] == 400.0
    assert first["actual_feed_in_argmax_next_1h_steps"] == 2.0


def test_analyze_produces_source_and_pairwise_bucket_summaries():
    raw = _synthetic_raw()
    enriched, by_source, by_pair, events = analyze(
        raw,
        source_a="amber_tactical_hybrid_strategic",
        source_b="model_a_hybrid",
        top_k=2,
    )
    assert "dispatch_bucket" in enriched.columns
    assert "feed_in_1h_future_minus_now_error_mwh" in enriched.columns
    assert set(by_source["source"]) == {"amber_tactical_hybrid_strategic", "model_a_hybrid"}
    assert "fit_lt_300_negload" in set(by_pair["bucket"])
    all_row = by_pair[by_pair["bucket"] == "all"].iloc[0]
    assert all_row["mean_a_minus_b_step_pnl"] > 0.0
    assert len(events) == 2


def test_act_now_match_flags_future_value_misranking():
    raw = _synthetic_raw()
    enriched = add_forecast_error_metrics(add_buckets(add_actual_horizon_metrics(raw)))
    hybrid_first = enriched[
        (enriched["source"] == "model_a_hybrid")
        & (enriched["time"] == pd.Timestamp("2025-09-01 00:00:00+00:00"))
    ].iloc[0]
    assert bool(hybrid_first["feed_in_1h_act_now_forecast"]) is False
    assert bool(hybrid_first["feed_in_1h_act_now_actual"]) is False
    assert bool(hybrid_first["feed_in_1h_act_now_match"]) is True
