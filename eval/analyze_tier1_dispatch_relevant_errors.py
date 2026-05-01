#!/usr/bin/env python3
"""
Diagnose Tier 1 tactical forecast errors in dispatch-relevant tariffed regimes.

This intentionally works from rolling MPC raw parquet outputs. Those files persist
step-0 and horizon-summary forecast columns, not the full h0-h11 forecast vector, so
the diagnostic focuses on h0 plus 30/60 minute and longer shape proxies that are
available without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
DT_HOURS = 5.0 / 60.0
HORIZON_SPECS = {
    "1h": 12,
    "4h": 48,
    "14h": 168,
}


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find path: {path_arg}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _safe_mae(series: pd.Series) -> float:
    return float(series.abs().mean()) if len(series) else float("nan")


def _safe_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else float("nan")


def _forward_mean(series: pd.Series, steps: int) -> pd.Series:
    values = series.to_numpy(dtype=np.float64, copy=False)
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(len(values)):
        window = values[i : i + steps]
        if len(window):
            out[i] = float(np.nanmean(window))
    return pd.Series(out, index=series.index)


def _forward_max_and_argmax(series: pd.Series, steps: int) -> tuple[pd.Series, pd.Series]:
    values = series.to_numpy(dtype=np.float64, copy=False)
    max_out = np.full(len(values), np.nan, dtype=np.float64)
    argmax_out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(len(values)):
        window = values[i : i + steps]
        finite = np.isfinite(window)
        if finite.any():
            finite_window = np.where(finite, window, -np.inf)
            argmax = int(np.argmax(finite_window))
            max_out[i] = float(finite_window[argmax])
            argmax_out[i] = float(argmax)
    return pd.Series(max_out, index=series.index), pd.Series(argmax_out, index=series.index)


def add_actual_horizon_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Attach realized forward price means/maxima by timestamp.

    The raw rolling output has one row per source per timestamp. Realized prices are
    source-invariant, so this builds one actual table and merges it back.
    """
    df = raw_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    required_actual_cols = [
        "time",
        "actual_price_mwh",
        "actual_general_price_mwh",
        "actual_feed_in_price_mwh",
        "actual_net_load_kw",
    ]
    optional_actual_cols = [
        "actual_load_kw",
        "actual_pv_kw",
    ]
    missing = [c for c in required_actual_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")
    actual_cols = required_actual_cols + [c for c in optional_actual_cols if c in df.columns]

    actual = (
        df[actual_cols]
        .drop_duplicates(subset=["time"])
        .sort_values("time", kind="stable")
        .reset_index(drop=True)
    )
    price_specs = {
        "wholesale": "actual_price_mwh",
        "general": "actual_general_price_mwh",
        "feed_in": "actual_feed_in_price_mwh",
    }
    for label, col in price_specs.items():
        for suffix, steps in HORIZON_SPECS.items():
            actual[f"actual_{label}_mean_next_{suffix}_mwh"] = _forward_mean(actual[col], steps)
            max_series, argmax_series = _forward_max_and_argmax(actual[col], steps)
            actual[f"actual_{label}_max_next_{suffix}_mwh"] = max_series
            actual[f"actual_{label}_argmax_next_{suffix}_steps"] = argmax_series

    return df.merge(actual.drop(columns=actual_cols[1:]), on="time", how="left", validate="many_to_one")


def bucket_label(row: pd.Series) -> str:
    fit = float(row["actual_feed_in_price_mwh"])
    net_load = float(row["actual_net_load_kw"])
    if fit >= 300.0 and net_load < 0.0:
        return "fit_gte_300_negload"
    if fit >= 300.0:
        return "fit_gte_300_nonnegload"
    if net_load < 0.0:
        return "fit_lt_300_negload"
    return "fit_lt_300_nonnegload"


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dispatch_bucket"] = out.apply(bucket_label, axis=1)
    out["utc_date"] = pd.to_datetime(out["time"], utc=True).dt.strftime("%Y-%m-%d")
    out["window_bucket"] = np.where(out["utc_date"] == "2025-09-01", "date_2025_09_01", "not_2025_09_01")
    return out


def add_forecast_error_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("wholesale_step0", "forecast_step0_mwh", "actual_price_mwh"),
        ("general_step0", "forecast_general_step0_mwh", "actual_general_price_mwh"),
        ("feed_in_step0", "forecast_feed_in_step0_mwh", "actual_feed_in_price_mwh"),
        ("wholesale_1h_mean", "forecast_mean_next_1h_mwh", "actual_wholesale_mean_next_1h_mwh"),
        ("wholesale_4h_mean", "forecast_mean_next_4h_mwh", "actual_wholesale_mean_next_4h_mwh"),
        ("wholesale_14h_mean", "forecast_mean_next_14h_mwh", "actual_wholesale_mean_next_14h_mwh"),
        ("general_1h_mean", "forecast_buy_mean_next_1h_mwh", "actual_general_mean_next_1h_mwh"),
        ("general_4h_mean", "forecast_buy_mean_next_4h_mwh", "actual_general_mean_next_4h_mwh"),
        ("general_14h_mean", "forecast_buy_mean_next_14h_mwh", "actual_general_mean_next_14h_mwh"),
        ("feed_in_1h_mean", "forecast_feed_in_mean_next_1h_mwh", "actual_feed_in_mean_next_1h_mwh"),
        ("feed_in_4h_mean", "forecast_feed_in_mean_next_4h_mwh", "actual_feed_in_mean_next_4h_mwh"),
        ("feed_in_14h_mean", "forecast_feed_in_mean_next_14h_mwh", "actual_feed_in_mean_next_14h_mwh"),
    ]
    for label, forecast_col, actual_col in pairs:
        if forecast_col in out.columns and actual_col in out.columns:
            out[f"{label}_error_mwh"] = out[forecast_col] - out[actual_col]
            out[f"{label}_abs_error_mwh"] = out[f"{label}_error_mwh"].abs()

    for price_label, step0_col, one_h_col, four_h_col in [
        (
            "feed_in",
            "forecast_feed_in_step0_mwh",
            "forecast_feed_in_mean_next_1h_mwh",
            "forecast_feed_in_mean_next_4h_mwh",
        ),
        (
            "general",
            "forecast_general_step0_mwh",
            "forecast_buy_mean_next_1h_mwh",
            "forecast_buy_mean_next_4h_mwh",
        ),
        ("wholesale", "forecast_step0_mwh", "forecast_mean_next_1h_mwh", "forecast_mean_next_4h_mwh"),
    ]:
        for suffix, mean_col in [("1h", one_h_col), ("4h", four_h_col)]:
            actual_step0 = {
                "feed_in": "actual_feed_in_price_mwh",
                "general": "actual_general_price_mwh",
                "wholesale": "actual_price_mwh",
            }[price_label]
            actual_mean = f"actual_{price_label}_mean_next_{suffix}_mwh"
            if step0_col not in out.columns or mean_col not in out.columns or actual_mean not in out.columns:
                continue
            forecast_spread = out[mean_col] - out[step0_col]
            actual_spread = out[actual_mean] - out[actual_step0]
            out[f"{price_label}_{suffix}_future_minus_now_forecast_mwh"] = forecast_spread
            out[f"{price_label}_{suffix}_future_minus_now_actual_mwh"] = actual_spread
            out[f"{price_label}_{suffix}_future_minus_now_error_mwh"] = forecast_spread - actual_spread
            out[f"{price_label}_{suffix}_act_now_forecast"] = out[step0_col] >= out[mean_col]
            out[f"{price_label}_{suffix}_act_now_actual"] = out[actual_step0] >= out[actual_mean]
            out[f"{price_label}_{suffix}_act_now_match"] = (
                out[f"{price_label}_{suffix}_act_now_forecast"] == out[f"{price_label}_{suffix}_act_now_actual"]
            )

    import_col = "realized_grid_import_kw" if "realized_grid_import_kw" in out.columns else "grid_import_kw"
    export_col = "realized_grid_export_kw" if "realized_grid_export_kw" in out.columns else "grid_export_kw"
    out["realized_import_kwh"] = out[import_col] * DT_HOURS
    out["realized_export_kwh"] = out[export_col] * DT_HOURS
    out["charge_kwh"] = out["charge_kw"] * DT_HOURS
    out["discharge_kwh"] = out["discharge_kw"] * DT_HOURS
    if "curtail_kw" in out.columns:
        out["curtail_kwh"] = out["curtail_kw"] * DT_HOURS
    else:
        out["curtail_kwh"] = 0.0
    return out


def _bucket_masks(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("all", pd.Series(True, index=df.index)),
        ("fit_lt_300_negload", df["dispatch_bucket"] == "fit_lt_300_negload"),
        ("fit_lt_300_nonnegload", df["dispatch_bucket"] == "fit_lt_300_nonnegload"),
        ("fit_gte_300", df["actual_feed_in_price_mwh"] >= 300.0),
        ("fit_gte_300_negload", df["dispatch_bucket"] == "fit_gte_300_negload"),
        ("fit_gte_300_nonnegload", df["dispatch_bucket"] == "fit_gte_300_nonnegload"),
        ("date_2025_09_01", df["window_bucket"] == "date_2025_09_01"),
        ("not_2025_09_01", df["window_bucket"] == "not_2025_09_01"),
    ]


def summarize_by_source_bucket(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "feed_in_step0_error_mwh",
        "feed_in_1h_mean_error_mwh",
        "feed_in_4h_mean_error_mwh",
        "general_step0_error_mwh",
        "general_1h_mean_error_mwh",
        "general_4h_mean_error_mwh",
        "wholesale_step0_error_mwh",
        "wholesale_1h_mean_error_mwh",
        "wholesale_4h_mean_error_mwh",
        "feed_in_1h_future_minus_now_error_mwh",
        "feed_in_4h_future_minus_now_error_mwh",
        "general_1h_future_minus_now_error_mwh",
        "general_4h_future_minus_now_error_mwh",
    ]
    rows: list[dict[str, Any]] = []
    for source, src in df.groupby("source", sort=True):
        for bucket, mask in _bucket_masks(src):
            sub = src[mask]
            if sub.empty:
                continue
            row: dict[str, Any] = {
                "source": source,
                "bucket": bucket,
                "n_rows": int(len(sub)),
                "mean_step_pnl": _safe_mean(sub["step_pnl"]) if "step_pnl" in sub else float("nan"),
                "mean_actual_feed_in_price_mwh": _safe_mean(sub["actual_feed_in_price_mwh"]),
                "mean_actual_net_load_kw": _safe_mean(sub["actual_net_load_kw"]),
                "mean_charge_kwh": _safe_mean(sub["charge_kwh"]),
                "mean_discharge_kwh": _safe_mean(sub["discharge_kwh"]),
                "mean_export_kwh": _safe_mean(sub["realized_export_kwh"]),
                "mean_import_kwh": _safe_mean(sub["realized_import_kwh"]),
                "mean_curtail_kwh": _safe_mean(sub["curtail_kwh"]),
            }
            for col in metric_cols:
                if col in sub.columns:
                    row[f"mean_{col}"] = _safe_mean(sub[col])
                    row[f"mae_{col}"] = _safe_mae(sub[col])
            for col in [
                "feed_in_1h_act_now_match",
                "feed_in_4h_act_now_match",
                "general_1h_act_now_match",
                "general_4h_act_now_match",
            ]:
                if col in sub.columns:
                    row[f"rate_{col}"] = _safe_rate(sub[col])
            rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_rows(df: pd.DataFrame, *, source_a: str, source_b: str) -> pd.DataFrame:
    a = df[df["source"] == source_a].copy()
    b = df[df["source"] == source_b].copy()
    if a.empty:
        raise ValueError(f"Source A not found: {source_a}")
    if b.empty:
        raise ValueError(f"Source B not found: {source_b}")
    keep_cols = [
        "time",
        "step_pnl",
        "charge_kwh",
        "discharge_kwh",
        "realized_import_kwh",
        "realized_export_kwh",
        "curtail_kwh",
        "soc_prev_kwh",
        "soc_kwh",
        "forecast_feed_in_step0_mwh",
        "forecast_feed_in_mean_next_1h_mwh",
        "forecast_feed_in_mean_next_4h_mwh",
        "forecast_general_step0_mwh",
        "forecast_buy_mean_next_1h_mwh",
        "forecast_buy_mean_next_4h_mwh",
        "feed_in_1h_future_minus_now_error_mwh",
        "feed_in_4h_future_minus_now_error_mwh",
        "general_1h_future_minus_now_error_mwh",
        "general_4h_future_minus_now_error_mwh",
        "feed_in_1h_act_now_match",
        "feed_in_4h_act_now_match",
        "general_1h_act_now_match",
        "general_4h_act_now_match",
    ]
    context_cols = [
        "time",
        "dispatch_bucket",
        "window_bucket",
        "utc_date",
        "actual_feed_in_price_mwh",
        "actual_general_price_mwh",
        "actual_net_load_kw",
        "actual_load_kw",
        "actual_pv_kw",
        "actual_feed_in_mean_next_1h_mwh",
        "actual_feed_in_mean_next_4h_mwh",
        "actual_general_mean_next_1h_mwh",
        "actual_general_mean_next_4h_mwh",
        "actual_feed_in_max_next_1h_mwh",
        "actual_feed_in_argmax_next_1h_steps",
        "actual_feed_in_max_next_4h_mwh",
        "actual_feed_in_argmax_next_4h_steps",
    ]
    a = a[[c for c in keep_cols if c in a.columns]].rename(columns={c: f"a_{c}" for c in keep_cols if c != "time"})
    b = b[[c for c in keep_cols if c in b.columns]].rename(columns={c: f"b_{c}" for c in keep_cols if c != "time"})
    context = df[df["source"] == source_a][[c for c in context_cols if c in df.columns]].copy()
    merged = context.merge(a, on="time", how="inner", validate="one_to_one").merge(
        b,
        on="time",
        how="inner",
        validate="one_to_one",
    )
    for metric in [
        "step_pnl",
        "charge_kwh",
        "discharge_kwh",
        "realized_import_kwh",
        "realized_export_kwh",
        "curtail_kwh",
        "soc_kwh",
        "forecast_feed_in_mean_next_1h_mwh",
        "forecast_feed_in_mean_next_4h_mwh",
        "forecast_buy_mean_next_1h_mwh",
        "forecast_buy_mean_next_4h_mwh",
        "feed_in_1h_future_minus_now_error_mwh",
        "feed_in_4h_future_minus_now_error_mwh",
        "general_1h_future_minus_now_error_mwh",
        "general_4h_future_minus_now_error_mwh",
    ]:
        a_col = f"a_{metric}"
        b_col = f"b_{metric}"
        if a_col in merged.columns and b_col in merged.columns:
            merged[f"a_minus_b_{metric}"] = merged[a_col] - merged[b_col]
    return merged


def join_optional_labels(
    pairwise: pd.DataFrame,
    *,
    prefix_attribution: pd.DataFrame | None = None,
    state_labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = pairwise.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    if prefix_attribution is not None:
        prefix = prefix_attribution.copy()
        prefix["time"] = pd.to_datetime(prefix["time"], utc=True)
        prefix_cols = [
            "time",
            "prefix_steps_requested",
            "a_minus_b_forced_prefix_regret",
            "a_minus_b_prefix_charge_kwh",
            "a_minus_b_prefix_discharge_kwh",
            "a_minus_b_prefix_import_kwh",
            "a_minus_b_prefix_export_kwh",
            "a_minus_b_prefix_curtail_kwh",
            "a_minus_b_prefix_step_pnl",
            "a_minus_b_prefix_soc_delta_kwh",
        ]
        out = out.merge(prefix[[c for c in prefix_cols if c in prefix.columns]], on="time", how="left")
    if state_labels is not None:
        labels = state_labels.copy()
        labels["time"] = pd.to_datetime(labels["time"], utc=True)
        label_cols = [
            "time",
            "horizon_steps",
            "oracle_minus_target_throughput_kwh",
            "oracle_minus_target_soc_delta_kwh",
            "oracle_minus_target_step_pnl",
            "comparator_minus_target_throughput_kwh",
            "comparator_minus_target_soc_delta_kwh",
            "comparator_minus_target_step_pnl",
        ]
        labels = labels[[c for c in label_cols if c in labels.columns]]
        labels = labels.sort_values(["time", "horizon_steps"], kind="stable").drop_duplicates(
            subset=["time", "horizon_steps"],
            keep="first",
        )
        wide = labels.pivot(index="time", columns="horizon_steps")
        wide.columns = [f"state_h{int(h)}_{name}" for name, h in wide.columns]
        wide = wide.reset_index()
        out = out.merge(wide, on="time", how="left")
    return out


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "a_minus_b_step_pnl",
        "a_minus_b_charge_kwh",
        "a_minus_b_discharge_kwh",
        "a_minus_b_realized_import_kwh",
        "a_minus_b_realized_export_kwh",
        "a_minus_b_curtail_kwh",
        "a_minus_b_soc_kwh",
        "a_minus_b_forecast_feed_in_mean_next_1h_mwh",
        "a_minus_b_forecast_feed_in_mean_next_4h_mwh",
        "a_minus_b_feed_in_1h_future_minus_now_error_mwh",
        "a_minus_b_feed_in_4h_future_minus_now_error_mwh",
        "a_minus_b_general_1h_future_minus_now_error_mwh",
        "a_minus_b_general_4h_future_minus_now_error_mwh",
        "a_minus_b_forced_prefix_regret",
        "a_minus_b_prefix_charge_kwh",
        "a_minus_b_prefix_discharge_kwh",
        "a_minus_b_prefix_import_kwh",
        "a_minus_b_prefix_export_kwh",
        "a_minus_b_prefix_curtail_kwh",
        "a_minus_b_prefix_step_pnl",
        "a_minus_b_prefix_soc_delta_kwh",
    ]
    for bucket, mask in _bucket_masks(pairwise):
        sub = pairwise[mask]
        if sub.empty:
            continue
        row: dict[str, Any] = {
            "bucket": bucket,
            "n_rows": int(len(sub)),
            "mean_actual_feed_in_price_mwh": _safe_mean(sub["actual_feed_in_price_mwh"]),
            "mean_actual_net_load_kw": _safe_mean(sub["actual_net_load_kw"]),
            "p_a_step_pnl_gt_b": _safe_rate(sub["a_minus_b_step_pnl"] > 1e-9),
            "p_b_step_pnl_gt_a": _safe_rate(sub["a_minus_b_step_pnl"] < -1e-9),
        }
        if "a_minus_b_forced_prefix_regret" in sub.columns:
            row["p_a_lower_prefix_regret"] = _safe_rate(sub["a_minus_b_forced_prefix_regret"] < -1e-9)
            row["p_b_lower_prefix_regret"] = _safe_rate(sub["a_minus_b_forced_prefix_regret"] > 1e-9)
        for col in metric_cols:
            if col in sub.columns:
                row[f"mean_{col}"] = _safe_mean(sub[col])
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_rows(pairwise: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    sort_cols = ["a_minus_b_step_pnl"]
    ascending = [False]
    if "a_minus_b_forced_prefix_regret" in pairwise.columns:
        sort_cols.append("a_minus_b_forced_prefix_regret")
        ascending.append(True)
    cols = [
        "time",
        "dispatch_bucket",
        "window_bucket",
        "actual_feed_in_price_mwh",
        "actual_general_price_mwh",
        "actual_net_load_kw",
        "actual_pv_kw",
        "a_minus_b_step_pnl",
        "a_minus_b_charge_kwh",
        "a_minus_b_discharge_kwh",
        "a_minus_b_realized_import_kwh",
        "a_minus_b_realized_export_kwh",
        "a_minus_b_curtail_kwh",
        "a_minus_b_feed_in_1h_future_minus_now_error_mwh",
        "a_minus_b_feed_in_4h_future_minus_now_error_mwh",
        "a_minus_b_general_1h_future_minus_now_error_mwh",
        "a_minus_b_general_4h_future_minus_now_error_mwh",
        "a_minus_b_forced_prefix_regret",
        "a_minus_b_prefix_step_pnl",
        "a_minus_b_prefix_soc_delta_kwh",
    ]
    available = [c for c in cols if c in pairwise.columns]
    return pairwise.sort_values(sort_cols, ascending=ascending, kind="stable")[available].head(top_k)


def build_contract_audit(raw_df: pd.DataFrame, model_dirs: list[str]) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "diagnostic_scope": "rolling_raw_step0_and_horizon_summary_forecast_shape",
        "model_target": "Tier 1 tactical LightGBM predicts raw wholesale RRP by 5-minute horizon.",
        "dispatch_gate": "netload_tariffed uses separate effective import/feed-in curves plus load/PV site economics.",
        "known_mismatch": (
            "A raw wholesale-price target can improve generic price MAE while still misranking tariffed "
            "buy/sell opportunity and short-horizon inventory value."
        ),
        "raw_columns_present": sorted(str(c) for c in raw_df.columns),
        "sources": sorted(str(s) for s in raw_df["source"].dropna().unique()) if "source" in raw_df else [],
        "model_dirs": {},
    }
    for model_dir in model_dirs:
        path = _resolve_path(model_dir)
        meta_path = path / "training_meta.json"
        if not meta_path.exists():
            audit["model_dirs"][str(path)] = {"training_meta_found": False}
            continue
        meta = json.loads(meta_path.read_text())
        audit["model_dirs"][str(path)] = {
            "training_meta_found": True,
            "feature_count": len(meta.get("feature_names", [])),
            "feature_names": meta.get("feature_names", []),
            "val_mae": meta.get("val_mae"),
            "baseline_val_mae": meta.get("baseline_val_mae"),
            "stratified_q50_mae": meta.get("stratified", {}).get("q50", {}).get("mae"),
            "stratified_q50_bias": meta.get("stratified", {}).get("q50", {}).get("bias"),
            "stratified_q95_coverage": meta.get("stratified", {}).get("q95", {}).get("coverage"),
        }
    return audit


def analyze(
    raw_df: pd.DataFrame,
    *,
    source_a: str,
    source_b: str,
    top_k: int = 50,
    prefix_attribution: pd.DataFrame | None = None,
    state_labels: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched = add_actual_horizon_metrics(raw_df)
    enriched = add_buckets(enriched)
    enriched = add_forecast_error_metrics(enriched)
    by_source_bucket = summarize_by_source_bucket(enriched)
    pairwise = build_pairwise_rows(enriched, source_a=source_a, source_b=source_b)
    pairwise = join_optional_labels(pairwise, prefix_attribution=prefix_attribution, state_labels=state_labels)
    by_pair_bucket = summarize_pairwise(pairwise)
    events = build_event_rows(pairwise, top_k=top_k)
    return enriched, by_source_bucket, by_pair_bucket, events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Rolling raw parquet path or filename under eval/results")
    parser.add_argument("--source-a", default="amber_tactical_hybrid_strategic")
    parser.add_argument("--source-b", default="model_a_hybrid")
    parser.add_argument("--prefix-attribution", default=None, help="Optional forced-prefix path-attribution CSV/parquet")
    parser.add_argument("--state-labels", default=None, help="Optional state-transition label CSV/parquet")
    parser.add_argument("--model-dir", action="append", default=[], help="Model directory to include in contract audit")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    raw_path = _resolve_path(args.raw)
    raw_df = pd.read_parquet(raw_path)
    prefix_df = _read_table(_resolve_path(args.prefix_attribution)) if args.prefix_attribution else None
    state_df = _read_table(_resolve_path(args.state_labels)) if args.state_labels else None

    enriched, by_source_bucket, by_pair_bucket, events = analyze(
        raw_df,
        source_a=args.source_a,
        source_b=args.source_b,
        top_k=max(0, int(args.top_k)),
        prefix_attribution=prefix_df,
        state_labels=state_df,
    )
    audit = build_contract_audit(raw_df, args.model_dir)

    prefix = args.output_prefix
    by_source_csv = RESULTS_DIR / f"{prefix}_tier1_dispatch_error_by_source_bucket.csv"
    by_pair_csv = RESULTS_DIR / f"{prefix}_tier1_dispatch_error_pairwise_by_bucket.csv"
    rows_parquet = RESULTS_DIR / f"{prefix}_tier1_dispatch_error_rows.parquet"
    events_csv = RESULTS_DIR / f"{prefix}_tier1_dispatch_error_events.csv"
    audit_json = RESULTS_DIR / f"{prefix}_tier1_dispatch_contract_audit.json"

    by_source_bucket.to_csv(by_source_csv, index=False)
    by_pair_bucket.to_csv(by_pair_csv, index=False)
    enriched.to_parquet(rows_parquet, index=False)
    events.to_csv(events_csv, index=False)
    audit_json.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"[done] wrote {by_source_csv}")
    print(f"[done] wrote {by_pair_csv}")
    print(f"[done] wrote {rows_parquet}")
    print(f"[done] wrote {events_csv}")
    print(f"[done] wrote {audit_json}")
    print(by_pair_bucket.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
