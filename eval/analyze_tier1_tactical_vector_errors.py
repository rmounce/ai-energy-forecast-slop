#!/usr/bin/env python3
"""
Reconstruct and diagnose full h0-h11 Tier 1 tactical forecast vectors.

The rolling raw parquet files only persist step-0 and horizon-summary forecast columns.
This companion rebuilds the first-hour tactical vectors for Amber APF and Tier 1 from the
same local inputs used by rolling_mpc_eval.py, then compares each 5-minute horizon against
realized tariffed import/feed-in prices.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from analyze_tier1_dispatch_relevant_errors import bucket_label  # noqa: E402
from rolling_mpc_eval import (  # noqa: E402
    TACTICAL_STEPS,
    ForecastProviders,
    _tariffed_price_frame_from_wholesale_mwh,
)


RESULTS_DIR = ROOT / "eval" / "results"
DT_HOURS = 5.0 / 60.0


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find path: {path_arg}")


def _provider_args(tactical_model_dir: str) -> SimpleNamespace:
    return SimpleNamespace(
        tactical_model_dir=str(_resolve_path(tactical_model_dir)),
        tft_checkpoint="",
        tft_scalers="",
        tier1_quantile_blend=0.0,
        tier2_quantile_blend=0.0,
        tier2_upper_quantile=0.90,
        split_curve_tactical_sources=set(),
        buy_curve_quantile_blend=0.0,
        sell_curve_quantile_blend=0.0,
    )


def _kw_col(df: pd.DataFrame, realized_col: str, planned_col: str) -> str:
    if realized_col in df.columns:
        return realized_col
    return planned_col


def _source_context(raw_df: pd.DataFrame, source: str) -> pd.DataFrame:
    source_df = raw_df[raw_df["source"] == source].copy()
    if source_df.empty:
        raise ValueError(f"Source not found in raw parquet: {source}")
    import_col = _kw_col(source_df, "realized_grid_import_kw", "grid_import_kw")
    export_col = _kw_col(source_df, "realized_grid_export_kw", "grid_export_kw")
    source_df["realized_import_kwh"] = source_df[import_col] * DT_HOURS
    source_df["realized_export_kwh"] = source_df[export_col] * DT_HOURS
    source_df["charge_kwh"] = source_df["charge_kw"] * DT_HOURS
    source_df["discharge_kwh"] = source_df["discharge_kw"] * DT_HOURS
    source_df["curtail_kwh"] = source_df["curtail_kw"] * DT_HOURS if "curtail_kw" in source_df else 0.0
    source_df["dispatch_bucket"] = source_df.apply(bucket_label, axis=1)
    source_df["utc_date"] = pd.to_datetime(source_df["time"], utc=True).dt.strftime("%Y-%m-%d")
    return source_df.sort_values("time", kind="stable").reset_index(drop=True)


def _actual_lookup(raw_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "time",
        "actual_price_mwh",
        "actual_general_price_mwh",
        "actual_feed_in_price_mwh",
        "actual_net_load_kw",
    ]
    optional = [c for c in ("actual_load_kw", "actual_pv_kw") if c in raw_df.columns]
    actual = (
        raw_df[cols + optional]
        .drop_duplicates(subset=["time"])
        .sort_values("time", kind="stable")
        .set_index("time")
    )
    return actual


def _curve_for_source(provider: ForecastProviders, tactical_source: str, ts: pd.Timestamp) -> np.ndarray | None:
    idx = pd.date_range(start=ts, periods=TACTICAL_STEPS, freq="5min", tz="UTC")
    current_actual = provider.current_actual_price(ts)
    if not np.isfinite(current_actual):
        return None
    if tactical_source == "amber_apf_lgbm":
        amber = provider.amber_expanded(ts)
        if amber is None:
            return None
        curve = amber.reindex(idx).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
    elif tactical_source == "model_a_hybrid":
        tier1 = provider.tier1_quantile(ts, 0.50)
        if tier1 is None:
            return None
        curve = tier1.reindex(idx).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
    else:
        raise ValueError(f"Unsupported tactical source for vector diagnostic: {tactical_source}")
    if len(curve) != TACTICAL_STEPS:
        return None
    curve[0] = current_actual
    if np.isfinite(curve).all():
        return curve
    repaired = pd.Series(curve).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
    repaired[0] = current_actual
    return repaired


def _tariffed_arrays(curve: np.ndarray, start_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    idx = pd.date_range(start=start_time, periods=len(curve), freq="5min", tz="UTC")
    tariffed = _tariffed_price_frame_from_wholesale_mwh(pd.Series(curve, index=idx))
    return (
        tariffed["general_price_mwh"].to_numpy(dtype=np.float64, copy=True),
        tariffed["feed_in_price_mwh"].to_numpy(dtype=np.float64, copy=True),
    )


def build_vector_rows(
    raw_df: pd.DataFrame,
    provider: ForecastProviders,
    *,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = raw_df.copy()
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    context = _source_context(raw_df, source)
    actual = _actual_lookup(raw_df)
    tactical_source = str(context["tactical_source"].dropna().iloc[0]) if "tactical_source" in context else source

    horizon_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for row in context.itertuples(index=False):
        ts = getattr(row, "time")
        curve = _curve_for_source(provider, tactical_source, ts)
        if curve is None:
            continue
        forecast_general, forecast_feed_in = _tariffed_arrays(curve, ts)
        future_idx = pd.date_range(start=ts, periods=TACTICAL_STEPS, freq="5min", tz="UTC")
        actual_future = actual.reindex(future_idx)
        actual_wholesale = actual_future["actual_price_mwh"].to_numpy(dtype=np.float64, copy=True)
        actual_general = actual_future["actual_general_price_mwh"].to_numpy(dtype=np.float64, copy=True)
        actual_feed_in = actual_future["actual_feed_in_price_mwh"].to_numpy(dtype=np.float64, copy=True)

        forecast_feed_in_mean = float(np.nanmean(forecast_feed_in))
        actual_feed_in_mean = float(np.nanmean(actual_feed_in))
        forecast_general_mean = float(np.nanmean(forecast_general))
        actual_general_mean = float(np.nanmean(actual_general))
        forecast_feed_in_max = float(np.nanmax(forecast_feed_in))
        actual_feed_in_max = float(np.nanmax(actual_feed_in))
        forecast_general_max = float(np.nanmax(forecast_general))
        actual_general_max = float(np.nanmax(actual_general))
        forecast_feed_in_argmax = int(np.nanargmax(forecast_feed_in))
        actual_feed_in_argmax = int(np.nanargmax(actual_feed_in))
        forecast_general_argmax = int(np.nanargmax(forecast_general))
        actual_general_argmax = int(np.nanargmax(actual_general))

        curve_rows.append(
            {
                "time": ts,
                "source": source,
                "tactical_source": tactical_source,
                "dispatch_bucket": getattr(row, "dispatch_bucket"),
                "utc_date": getattr(row, "utc_date"),
                "actual_feed_in_price_mwh": float(getattr(row, "actual_feed_in_price_mwh")),
                "actual_general_price_mwh": float(getattr(row, "actual_general_price_mwh")),
                "actual_net_load_kw": float(getattr(row, "actual_net_load_kw")),
                "step_pnl": float(getattr(row, "step_pnl")),
                "charge_kwh": float(getattr(row, "charge_kwh")),
                "discharge_kwh": float(getattr(row, "discharge_kwh")),
                "realized_import_kwh": float(getattr(row, "realized_import_kwh")),
                "realized_export_kwh": float(getattr(row, "realized_export_kwh")),
                "curtail_kwh": float(getattr(row, "curtail_kwh")),
                "soc_kwh": float(getattr(row, "soc_kwh")),
                "forecast_feed_in_mean_1h_mwh": forecast_feed_in_mean,
                "actual_feed_in_mean_1h_mwh": actual_feed_in_mean,
                "forecast_feed_in_max_1h_mwh": forecast_feed_in_max,
                "actual_feed_in_max_1h_mwh": actual_feed_in_max,
                "forecast_feed_in_argmax_1h_step": forecast_feed_in_argmax,
                "actual_feed_in_argmax_1h_step": actual_feed_in_argmax,
                "forecast_feed_in_future_minus_now_mwh": forecast_feed_in_mean - forecast_feed_in[0],
                "actual_feed_in_future_minus_now_mwh": actual_feed_in_mean - actual_feed_in[0],
                "forecast_feed_in_act_now": bool(forecast_feed_in[0] >= forecast_feed_in_mean),
                "actual_feed_in_act_now": bool(actual_feed_in[0] >= actual_feed_in_mean),
                "forecast_general_mean_1h_mwh": forecast_general_mean,
                "actual_general_mean_1h_mwh": actual_general_mean,
                "forecast_general_max_1h_mwh": forecast_general_max,
                "actual_general_max_1h_mwh": actual_general_max,
                "forecast_general_argmax_1h_step": forecast_general_argmax,
                "actual_general_argmax_1h_step": actual_general_argmax,
                "forecast_general_future_minus_now_mwh": forecast_general_mean - forecast_general[0],
                "actual_general_future_minus_now_mwh": actual_general_mean - actual_general[0],
                "forecast_general_act_now": bool(forecast_general[0] >= forecast_general_mean),
                "actual_general_act_now": bool(actual_general[0] >= actual_general_mean),
            }
        )
        for horizon in range(TACTICAL_STEPS):
            horizon_rows.append(
                {
                    "time": ts,
                    "source": source,
                    "tactical_source": tactical_source,
                    "dispatch_bucket": getattr(row, "dispatch_bucket"),
                    "utc_date": getattr(row, "utc_date"),
                    "actual_feed_in_price_mwh": float(getattr(row, "actual_feed_in_price_mwh")),
                    "actual_general_price_mwh": float(getattr(row, "actual_general_price_mwh")),
                    "actual_net_load_kw": float(getattr(row, "actual_net_load_kw")),
                    "horizon": horizon,
                    "target_time": future_idx[horizon],
                    "forecast_wholesale_mwh": float(curve[horizon]),
                    "actual_wholesale_mwh": float(actual_wholesale[horizon]),
                    "forecast_general_mwh": float(forecast_general[horizon]),
                    "actual_general_mwh": float(actual_general[horizon]),
                    "forecast_feed_in_mwh": float(forecast_feed_in[horizon]),
                    "actual_feed_in_mwh": float(actual_feed_in[horizon]),
                    "wholesale_error_mwh": float(curve[horizon] - actual_wholesale[horizon]),
                    "general_error_mwh": float(forecast_general[horizon] - actual_general[horizon]),
                    "feed_in_error_mwh": float(forecast_feed_in[horizon] - actual_feed_in[horizon]),
                }
            )

    return pd.DataFrame(horizon_rows), pd.DataFrame(curve_rows)


def _bucket_masks(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("all", pd.Series(True, index=df.index)),
        ("fit_lt_300_negload", df["dispatch_bucket"] == "fit_lt_300_negload"),
        ("fit_lt_300_nonnegload", df["dispatch_bucket"] == "fit_lt_300_nonnegload"),
        ("fit_gte_300", df["actual_feed_in_price_mwh"] >= 300.0),
        ("date_2025_09_01", df["utc_date"] == "2025-09-01"),
        ("not_2025_09_01", df["utc_date"] != "2025-09-01"),
    ]


def summarize_horizon(horizon_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, src in horizon_df.groupby("source", sort=True):
        for bucket, mask in _bucket_masks(src):
            sub_bucket = src[mask]
            if sub_bucket.empty:
                continue
            for horizon, sub in sub_bucket.groupby("horizon", sort=True):
                rows.append(
                    {
                        "source": source,
                        "bucket": bucket,
                        "horizon": int(horizon),
                        "n_rows": int(len(sub)),
                        "mean_feed_in_error_mwh": float(sub["feed_in_error_mwh"].mean()),
                        "mae_feed_in_error_mwh": float(sub["feed_in_error_mwh"].abs().mean()),
                        "mean_general_error_mwh": float(sub["general_error_mwh"].mean()),
                        "mae_general_error_mwh": float(sub["general_error_mwh"].abs().mean()),
                        "mean_wholesale_error_mwh": float(sub["wholesale_error_mwh"].mean()),
                        "mae_wholesale_error_mwh": float(sub["wholesale_error_mwh"].abs().mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, src in curve_df.groupby("source", sort=True):
        for bucket, mask in _bucket_masks(src):
            sub = src[mask]
            if sub.empty:
                continue
            rows.append(
                {
                    "source": source,
                    "bucket": bucket,
                    "n_rows": int(len(sub)),
                    "mean_step_pnl": float(sub["step_pnl"].mean()),
                    "mean_actual_feed_in_price_mwh": float(sub["actual_feed_in_price_mwh"].mean()),
                    "mean_actual_net_load_kw": float(sub["actual_net_load_kw"].mean()),
                    "mean_feed_in_mean_error_mwh": float(
                        (sub["forecast_feed_in_mean_1h_mwh"] - sub["actual_feed_in_mean_1h_mwh"]).mean()
                    ),
                    "mae_feed_in_mean_error_mwh": float(
                        (sub["forecast_feed_in_mean_1h_mwh"] - sub["actual_feed_in_mean_1h_mwh"]).abs().mean()
                    ),
                    "mean_feed_in_max_error_mwh": float(
                        (sub["forecast_feed_in_max_1h_mwh"] - sub["actual_feed_in_max_1h_mwh"]).mean()
                    ),
                    "mean_feed_in_argmax_error_steps": float(
                        (sub["forecast_feed_in_argmax_1h_step"] - sub["actual_feed_in_argmax_1h_step"]).mean()
                    ),
                    "rate_feed_in_act_now_match": float(
                        (sub["forecast_feed_in_act_now"] == sub["actual_feed_in_act_now"]).mean()
                    ),
                    "mean_general_mean_error_mwh": float(
                        (sub["forecast_general_mean_1h_mwh"] - sub["actual_general_mean_1h_mwh"]).mean()
                    ),
                    "mae_general_mean_error_mwh": float(
                        (sub["forecast_general_mean_1h_mwh"] - sub["actual_general_mean_1h_mwh"]).abs().mean()
                    ),
                    "mean_general_argmax_error_steps": float(
                        (sub["forecast_general_argmax_1h_step"] - sub["actual_general_argmax_1h_step"]).mean()
                    ),
                    "rate_general_act_now_match": float(
                        (sub["forecast_general_act_now"] == sub["actual_general_act_now"]).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_curve(curve_df: pd.DataFrame, *, source_a: str, source_b: str) -> pd.DataFrame:
    a = curve_df[curve_df["source"] == source_a].copy()
    b = curve_df[curve_df["source"] == source_b].copy()
    if a.empty or b.empty:
        raise ValueError(f"Pairwise sources missing: {source_a}, {source_b}")
    context_cols = [
        "time",
        "dispatch_bucket",
        "utc_date",
        "actual_feed_in_price_mwh",
        "actual_general_price_mwh",
        "actual_net_load_kw",
    ]
    metric_cols = [c for c in a.columns if c not in context_cols and c not in {"source", "tactical_source"}]
    a = a[["time"] + metric_cols].rename(columns={c: f"a_{c}" for c in metric_cols})
    b = b[["time"] + metric_cols].rename(columns={c: f"b_{c}" for c in metric_cols})
    merged = curve_df[curve_df["source"] == source_a][context_cols].merge(a, on="time").merge(b, on="time")
    for col in metric_cols:
        a_col = f"a_{col}"
        b_col = f"b_{col}"
        if pd.api.types.is_bool_dtype(merged[a_col]) or pd.api.types.is_bool_dtype(merged[b_col]):
            continue
        merged[f"a_minus_b_{col}"] = merged[a_col] - merged[b_col]
    return merged


def build_pairwise_horizon(horizon_df: pd.DataFrame, *, source_a: str, source_b: str) -> pd.DataFrame:
    a = horizon_df[horizon_df["source"] == source_a].copy()
    b = horizon_df[horizon_df["source"] == source_b].copy()
    key = ["time", "horizon"]
    context_cols = [
        "time",
        "horizon",
        "dispatch_bucket",
        "utc_date",
        "actual_feed_in_price_mwh",
        "actual_general_price_mwh",
        "actual_net_load_kw",
    ]
    context = a[[c for c in context_cols if c in a.columns]].copy()
    cols = ["forecast_feed_in_mwh", "feed_in_error_mwh", "forecast_general_mwh", "general_error_mwh"]
    a = a[key + cols].rename(columns={c: f"a_{c}" for c in cols})
    b = b[key + cols].rename(columns={c: f"b_{c}" for c in cols})
    merged = context.merge(a, on=key).merge(b, on=key)
    for col in cols:
        merged[f"a_minus_b_{col}"] = merged[f"a_{col}"] - merged[f"b_{col}"]
    return merged


def summarize_pairwise_curve(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, mask in _bucket_masks(pairwise):
        sub = pairwise[mask]
        if sub.empty:
            continue
        rows.append(
            {
                "bucket": bucket,
                "n_rows": int(len(sub)),
                "mean_a_minus_b_step_pnl": float(sub["a_minus_b_step_pnl"].mean()),
                "mean_a_minus_b_charge_kwh": float(sub["a_minus_b_charge_kwh"].mean()),
                "mean_a_minus_b_discharge_kwh": float(sub["a_minus_b_discharge_kwh"].mean()),
                "mean_a_minus_b_realized_import_kwh": float(sub["a_minus_b_realized_import_kwh"].mean()),
                "mean_a_minus_b_realized_export_kwh": float(sub["a_minus_b_realized_export_kwh"].mean()),
                "mean_a_minus_b_feed_in_mean_error_mwh": float(
                    (
                        (sub["a_forecast_feed_in_mean_1h_mwh"] - sub["a_actual_feed_in_mean_1h_mwh"])
                        - (sub["b_forecast_feed_in_mean_1h_mwh"] - sub["b_actual_feed_in_mean_1h_mwh"])
                    ).mean()
                ),
                "mean_a_minus_b_feed_in_argmax_error_steps": float(
                    (
                        (sub["a_forecast_feed_in_argmax_1h_step"] - sub["a_actual_feed_in_argmax_1h_step"])
                        - (sub["b_forecast_feed_in_argmax_1h_step"] - sub["b_actual_feed_in_argmax_1h_step"])
                    ).mean()
                ),
                "p_a_feed_in_act_now_match": float(
                    (sub["a_forecast_feed_in_act_now"] == sub["a_actual_feed_in_act_now"]).mean()
                ),
                "p_b_feed_in_act_now_match": float(
                    (sub["b_forecast_feed_in_act_now"] == sub["b_actual_feed_in_act_now"]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairwise_horizon(pairwise_horizon: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, mask in _bucket_masks(pairwise_horizon):
        sub_bucket = pairwise_horizon[mask]
        if sub_bucket.empty:
            continue
        for horizon, sub in sub_bucket.groupby("horizon", sort=True):
            rows.append(
                {
                    "bucket": bucket,
                    "horizon": int(horizon),
                    "n_rows": int(len(sub)),
                    "mean_a_minus_b_forecast_feed_in_mwh": float(sub["a_minus_b_forecast_feed_in_mwh"].mean()),
                    "mean_a_minus_b_feed_in_error_mwh": float(sub["a_minus_b_feed_in_error_mwh"].mean()),
                    "mean_a_minus_b_forecast_general_mwh": float(sub["a_minus_b_forecast_general_mwh"].mean()),
                    "mean_a_minus_b_general_error_mwh": float(sub["a_minus_b_general_error_mwh"].mean()),
                }
            )
    return pd.DataFrame(rows)


def build_event_rows(pairwise: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    cols = [
        "time",
        "dispatch_bucket",
        "actual_feed_in_price_mwh",
        "actual_general_price_mwh",
        "actual_net_load_kw",
        "a_minus_b_step_pnl",
        "a_minus_b_charge_kwh",
        "a_minus_b_discharge_kwh",
        "a_minus_b_realized_import_kwh",
        "a_minus_b_realized_export_kwh",
        "a_forecast_feed_in_mean_1h_mwh",
        "b_forecast_feed_in_mean_1h_mwh",
        "a_actual_feed_in_mean_1h_mwh",
        "a_forecast_feed_in_argmax_1h_step",
        "b_forecast_feed_in_argmax_1h_step",
        "a_actual_feed_in_argmax_1h_step",
        "a_forecast_feed_in_act_now",
        "b_forecast_feed_in_act_now",
        "a_actual_feed_in_act_now",
    ]
    return pairwise.sort_values("a_minus_b_step_pnl", ascending=False, kind="stable")[
        [c for c in cols if c in pairwise.columns]
    ].head(top_k)


def analyze(
    raw_df: pd.DataFrame,
    *,
    source_a: str,
    source_b: str,
    tactical_model_dir: str,
    top_k: int = 50,
    max_times: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if max_times is not None and max_times > 0:
        raw_df = raw_df.copy()
        raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
        keep_times = (
            raw_df["time"]
            .drop_duplicates()
            .sort_values(kind="stable")
            .head(int(max_times))
        )
        raw_df = raw_df[raw_df["time"].isin(set(keep_times))].copy()
    provider = ForecastProviders(_provider_args(tactical_model_dir))
    a_horizon, a_curve = build_vector_rows(raw_df, provider, source=source_a)
    b_horizon, b_curve = build_vector_rows(raw_df, provider, source=source_b)
    horizon = pd.concat([a_horizon, b_horizon], ignore_index=True)
    curve = pd.concat([a_curve, b_curve], ignore_index=True)
    source_horizon_summary = summarize_horizon(horizon)
    source_curve_summary = summarize_curve(curve)
    pairwise_curve = build_pairwise_curve(curve, source_a=source_a, source_b=source_b)
    pairwise_horizon = build_pairwise_horizon(horizon, source_a=source_a, source_b=source_b)
    pairwise_curve_summary = summarize_pairwise_curve(pairwise_curve)
    pairwise_horizon_summary = summarize_pairwise_horizon(pairwise_horizon)
    events = build_event_rows(pairwise_curve, top_k=top_k)
    return (
        horizon,
        source_horizon_summary,
        source_curve_summary,
        pairwise_curve_summary,
        pairwise_horizon_summary,
        events,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Rolling raw parquet path or filename under eval/results")
    parser.add_argument("--source-a", default="amber_tactical_hybrid_strategic")
    parser.add_argument("--source-b", default="model_a_hybrid")
    parser.add_argument("--tactical-model-dir", default="models/lgbm_tactical")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-times", type=int, default=0, help="Optional first-N timestamps for quick smoke tests")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    raw_df = pd.read_parquet(_resolve_path(args.raw))
    (
        horizon,
        source_horizon_summary,
        source_curve_summary,
        pairwise_curve_summary,
        pairwise_horizon_summary,
        events,
    ) = analyze(
        raw_df,
        source_a=args.source_a,
        source_b=args.source_b,
        tactical_model_dir=args.tactical_model_dir,
        top_k=max(0, int(args.top_k)),
        max_times=int(args.max_times) if int(args.max_times) > 0 else None,
    )

    prefix = args.output_prefix
    horizon_path = RESULTS_DIR / f"{prefix}_tier1_vector_rows.parquet"
    source_horizon_path = RESULTS_DIR / f"{prefix}_tier1_vector_by_source_bucket_horizon.csv"
    source_curve_path = RESULTS_DIR / f"{prefix}_tier1_vector_by_source_bucket.csv"
    pairwise_curve_path = RESULTS_DIR / f"{prefix}_tier1_vector_pairwise_by_bucket.csv"
    pairwise_horizon_path = RESULTS_DIR / f"{prefix}_tier1_vector_pairwise_by_bucket_horizon.csv"
    events_path = RESULTS_DIR / f"{prefix}_tier1_vector_events.csv"

    horizon.to_parquet(horizon_path, index=False)
    source_horizon_summary.to_csv(source_horizon_path, index=False)
    source_curve_summary.to_csv(source_curve_path, index=False)
    pairwise_curve_summary.to_csv(pairwise_curve_path, index=False)
    pairwise_horizon_summary.to_csv(pairwise_horizon_path, index=False)
    events.to_csv(events_path, index=False)

    print(f"[done] wrote {horizon_path}")
    print(f"[done] wrote {source_horizon_path}")
    print(f"[done] wrote {source_curve_path}")
    print(f"[done] wrote {pairwise_curve_path}")
    print(f"[done] wrote {pairwise_horizon_path}")
    print(f"[done] wrote {events_path}")
    print(pairwise_curve_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
