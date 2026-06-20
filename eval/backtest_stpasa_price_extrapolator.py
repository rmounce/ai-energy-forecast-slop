#!/usr/bin/env python3
"""Causal backtest for STPASA covariates in the APF-backed price extrapolator.

This evaluates the incumbent production shape:

1. Train the p50 ``price`` Darts/LightGBM model only on data before
   ``--train-until``.
2. For held-out logged ``price`` run slots, seed the near horizon from the
   logged APF-backed curve.
3. Predict the remaining tail with the STPASA-enabled extrapolator.
4. Score against realised SA1 RRP and compare with the incumbent log.

The script is deliberately artifact-heavy because it is intended for long
overnight runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytz
import pyarrow.parquet as pq
from influxdb import InfluxDBClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import forecast as fc
from config_utils import load_config
from eval.analyze_lgbm_residual_drivers import (
    DEFAULT_ACTUALS,
    DEFAULT_LOG,
    DEFAULT_PREDISPATCH,
    DEFAULT_STPASA,
    HORIZON_BUCKETS,
    adelaide_bucket,
    attach_actuals,
    horizon_bucket,
    latest_asof_by_target,
    load_price_log,
    load_source,
)
from eval.price_source_contracts import require_apf_backed

RESULTS_DIR = REPO_ROOT / "eval" / "results"
LOCAL_TZ = "Australia/Adelaide"
REGIONS = ("SA1", "VIC1", "NSW1")


def _as_utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def add_stpasa_to_config(config: dict, stpasa_path: Path) -> dict:
    config = json.loads(json.dumps(config))
    feature_cols = config["models"]["price"]["feature_cols"]
    for col in fc.STPASA_FEATURE_COLUMNS:
        if col not in feature_cols:
            feature_cols.append(col)
    config.setdefault("paths", {})["stpasa_regionsolution_file"] = str(stpasa_path)
    return config


def train_price_model(config: dict, *, train_until: pd.Timestamp):
    from darts import TimeSeries
    from darts.models import LightGBMModel

    fc.CONFIG = config
    model_config = config["models"]["price"]
    target_col = model_config["target_column"]
    feature_cols = model_config["feature_cols"]
    training_days = model_config.get("training_history_days", config["training_history_days"])
    start_time = train_until - pd.Timedelta(days=training_days)

    logging.info("Training cutoff: %s", train_until)
    logging.info("Training history start: %s", start_time)
    historical_df = load_historical_training_frame(config, start_time=start_time, end_time=train_until)
    if historical_df.empty:
        raise SystemExit("No historical data available for training")

    model_data = historical_df[[target_col, *feature_cols]].copy().ffill().dropna()
    logging.info("Training rows after feature availability: %s", len(model_data))
    if model_data.empty:
        raise SystemExit("No training rows after feature availability")

    min_val = model_data[target_col].min()
    shift_value = abs(min_val) + 1 if min_val <= 0 else 0
    model_data[target_col] = np.log(model_data[target_col] + shift_value)
    target_series = TimeSeries.from_series(model_data[target_col], freq="30min")
    covariates_ts = TimeSeries.from_dataframe(model_data, value_cols=feature_cols, freq="30min")

    sample_weight = None
    weighting_config = model_config.get("recency_weighting")
    if weighting_config and weighting_config.get("enabled"):
        half_life_days = weighting_config.get("half_life_days", 90)
        time_delta_days = (model_data.index.max() - model_data.index).total_seconds() / (24 * 3600)
        weights = pd.Series(np.power(2, -time_delta_days / half_life_days), index=model_data.index)
        sample_weight = TimeSeries.from_series(weights, freq="30min")

    params = model_config["lgbm_params"].copy()
    q50 = model_config.get("quantile_models", {}).get("price")
    if q50:
        params.update({k: v for k, v in q50.items() if k in {"objective", "alpha"}})

    model = LightGBMModel(
        lags=model_config.get("target_lags"),
        lags_future_covariates=model_config.get("future_covariate_lags"),
        output_chunk_length=model_config["forecast_horizon"],
        add_encoders=model_config.get("add_encoders"),
        **params,
    )
    model.fit(series=target_series, future_covariates=covariates_ts, sample_weight=sample_weight)
    return model, {"shift_value": shift_value}, historical_df


def load_historical_training_frame(
    config: dict,
    *,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    fc.CONFIG = config
    client = InfluxDBClient(**config["influxdb"])
    try:
        return fc.get_historical_data(client, start_time.to_pydatetime(), end_time.to_pydatetime())
    finally:
        client.close()


def load_predispatch_by_region(path_template: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    required = {"interval_dt", "run_time", "total_demand", "net_interchange"}
    for region in REGIONS:
        path = Path(str(path_template).replace("sa1", region.lower()))
        if not path.exists():
            logging.warning("Missing PREDISPATCH parquet for %s: %s", region, path)
            continue
        available = set(pq.read_schema(path).names)
        missing = required - available
        if missing:
            logging.warning(
                "Skipping PREDISPATCH parquet for %s because it lacks %s: %s",
                region,
                sorted(missing),
                path,
            )
            continue
        source = load_source(path, ["interval_dt", "run_time", "total_demand", "net_interchange"])
        out[region] = source
    return out


def load_sdo_from_influx(config: dict, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    query = (
        'SELECT "scheduled_demand", "net_interchange" '
        'FROM "rp_30m"."aemo_sevendayoutlook" '
        f"WHERE time >= '{start_str}' AND time <= '{end_str}'"
        ' GROUP BY "region", "run_time"'
    )
    client = InfluxDBClient(**config["influxdb"])
    try:
        result = client.query(query)
    finally:
        client.close()
    rows = []
    for series in result.raw.get("series", []):
        tags = series.get("tags", {})
        columns = series.get("columns", [])
        for values in series.get("values", []):
            row = dict(zip(columns, values))
            row["region"] = tags.get("region")
            row["run_time"] = tags.get("run_time")
            rows.append(row)
    if not rows:
        logging.warning("No SevenDayOutlook rows returned from InfluxDB")
        return pd.DataFrame(columns=["interval_dt", "run_time", "region", "scheduled_demand", "net_interchange"])
    df = pd.DataFrame(rows)
    df["interval_dt"] = pd.to_datetime(df["time"], utc=True)
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df = df[df["region"].isin(REGIONS)].copy()
    return df[["interval_dt", "run_time", "region", "scheduled_demand", "net_interchange"]].sort_values(
        ["interval_dt", "run_time", "region"]
    )


def load_eval_rows(args: argparse.Namespace) -> pd.DataFrame:
    require_apf_backed("amber_apf_lgbm")
    df = load_price_log(args.log_file, model_filter="price")
    df = df[
        (df["forecast_creation_time"] >= args.eval_start)
        & (df["forecast_creation_time"] < args.eval_until)
    ].copy()
    df["horizon_hours"] = (
        df["forecast_target_time"] - df["forecast_creation_time"]
    ).dt.total_seconds() / 3600.0
    df = df[(df["horizon_hours"] >= 0.0) & (df["horizon_hours"] <= args.max_horizon_hours)].copy()
    df = attach_actuals(df, args.actuals_30m)
    df = df.dropna(subset=["actual_rrp"]).copy()

    run_times = pd.Series(df["forecast_creation_time"].sort_values().unique())
    if args.run_stride > 1:
        run_times = run_times.iloc[:: args.run_stride]
    if args.max_runs:
        run_times = run_times.iloc[: args.max_runs]
    keep = set(pd.to_datetime(run_times, utc=True))
    df = df[df["forecast_creation_time"].isin(keep)].copy()
    logging.info("Held-out run slots selected: %s", df["forecast_creation_time"].nunique())
    logging.info("Held-out rows selected: %s", len(df))
    return df


def build_future_covariates(
    run_df: pd.DataFrame,
    *,
    creation_time: pd.Timestamp,
    model_config: dict,
    pred_by_region: dict[str, pd.DataFrame],
    sdo_df: pd.DataFrame,
    stpasa_df: pd.DataFrame,
) -> pd.DataFrame:
    left = run_df.sort_values("forecast_target_time").drop_duplicates("forecast_target_time").copy()
    out = left.set_index("forecast_target_time")[
        ["power_pv", "temperature_adelaide", "humidity_adelaide", "wind_speed_adelaide"]
    ].copy()

    join_left = left[["forecast_creation_time", "forecast_target_time"]].copy()
    for region in REGIONS:
        suffix = region.lower()
        pd_source = pred_by_region.get(region, pd.DataFrame())
        if not pd_source.empty:
            joined = latest_asof_by_target(
                join_left,
                pd_source,
                left_target="forecast_target_time",
                left_time="forecast_creation_time",
                right_target="interval_dt",
                right_time="run_time",
                value_cols=["total_demand", "net_interchange"],
                prefix="pd",
            )
            out[f"total_demand_{suffix}"] = joined["pd_total_demand"].to_numpy(dtype=float)
            out[f"net_interchange_{suffix}"] = joined["pd_net_interchange"].to_numpy(dtype=float)
        else:
            out[f"total_demand_{suffix}"] = np.nan
            out[f"net_interchange_{suffix}"] = np.nan

        region_sdo = sdo_df[sdo_df["region"] == region]
        if not region_sdo.empty:
            joined = latest_asof_by_target(
                join_left,
                region_sdo,
                left_target="forecast_target_time",
                left_time="forecast_creation_time",
                right_target="interval_dt",
                right_time="run_time",
                value_cols=["scheduled_demand", "net_interchange"],
                prefix="sdo",
            )
            out[f"total_demand_{suffix}"] = out[f"total_demand_{suffix}"].combine_first(
                pd.Series(joined["sdo_scheduled_demand"].to_numpy(dtype=float), index=out.index)
            )
            out[f"net_interchange_{suffix}"] = out[f"net_interchange_{suffix}"].combine_first(
                pd.Series(joined["sdo_net_interchange"].to_numpy(dtype=float), index=out.index)
            )

    stpasa = fc._attach_stpasa_features_for_targets(out, stpasa_df, asof_times=creation_time)
    out = out.join(stpasa)
    for col in model_config["feature_cols"]:
        if col not in out.columns:
            out[col] = np.nan
    return out[model_config["feature_cols"]].astype(float).ffill().bfill()


def prepend_historical_covariate_context(
    covariates: pd.DataFrame,
    historical_df: pd.DataFrame,
    model_config: dict,
) -> pd.DataFrame:
    """Prepend covariate rows required by negative future-covariate lags."""

    lags = model_config.get("future_covariate_lags") or [0]
    min_lag = min(lags)
    if min_lag >= 0 or covariates.empty:
        return covariates

    freq = pd.Timedelta(minutes=30)
    context_start = covariates.index.min() + min_lag * freq
    context_end = covariates.index.min() - freq
    feature_cols = model_config["feature_cols"]
    context = historical_df.reindex(
        pd.date_range(context_start, context_end, freq="30min", tz="UTC")
    )
    for col in feature_cols:
        if col not in context.columns:
            context[col] = np.nan
    context = context[feature_cols].astype(float).ffill().bfill()
    return pd.concat([context, covariates[feature_cols]], axis=0)


def metric_rows(scored: pd.DataFrame) -> list[dict]:
    rows = []
    for label, sub in [
        ("all", scored),
        *[(name, scored[scored["horizon_bucket"] == name]) for name, _, _ in HORIZON_BUCKETS],
        *[(name, scored[scored["adelaide_bucket"] == name]) for name in sorted(scored["adelaide_bucket"].dropna().unique())],
    ]:
        if sub.empty:
            continue
        inc_err = sub["incumbent_mwh"] - sub["actual_rrp"]
        cand_err = sub["candidate_mwh"] - sub["actual_rrp"]
        rows.append(
            {
                "bucket": label,
                "n": int(len(sub)),
                "incumbent_mae": float(inc_err.abs().mean()),
                "candidate_mae": float(cand_err.abs().mean()),
                "mae_delta": float(cand_err.abs().mean() - inc_err.abs().mean()),
                "incumbent_bias": float(inc_err.mean()),
                "candidate_bias": float(cand_err.mean()),
                "bias_delta": float(cand_err.mean() - inc_err.mean()),
            }
        )
    return rows


def write_progress(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--actuals-30m", type=Path, default=DEFAULT_ACTUALS)
    parser.add_argument("--predispatch-template", type=Path, default=DEFAULT_PREDISPATCH)
    parser.add_argument("--stpasa", type=Path, default=DEFAULT_STPASA)
    parser.add_argument("--train-until", required=True, type=_as_utc)
    parser.add_argument("--eval-start", required=True, type=_as_utc)
    parser.add_argument("--eval-until", required=True, type=_as_utc)
    parser.add_argument("--apf-seed-hours", type=float, default=28.5)
    parser.add_argument("--max-horizon-hours", type=float, default=72.0)
    parser.add_argument("--run-stride", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument(
        "--reuse-trained-model",
        action="store_true",
        help="Load existing output-prefix model/params and only rebuild historical context.",
    )
    parser.add_argument("--output-prefix", default="stpasa_price_extrapolator_backtest")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{args.output_prefix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    config = add_stpasa_to_config(load_config(args.config), args.stpasa)
    fc.CONFIG = config
    manifest_path = RESULTS_DIR / f"{args.output_prefix}_manifest.json"
    progress_path = RESULTS_DIR / f"{args.output_prefix}.progress.json"
    predictions_path = RESULTS_DIR / f"{args.output_prefix}_predictions.parquet"
    metrics_path = RESULTS_DIR / f"{args.output_prefix}_metrics.csv"
    model_path = RESULTS_DIR / f"{args.output_prefix}_model.pkl"
    params_path = RESULTS_DIR / f"{args.output_prefix}_params.json"

    manifest_path.write_text(
        json.dumps(
            {
                "train_until": str(args.train_until),
                "eval_start": str(args.eval_start),
                "eval_until": str(args.eval_until),
                "apf_seed_hours": args.apf_seed_hours,
                "max_horizon_hours": args.max_horizon_hours,
                "run_stride": args.run_stride,
                "max_runs": args.max_runs,
            },
            indent=2,
        )
    )

    if args.reuse_trained_model and model_path.exists() and params_path.exists():
        write_progress(progress_path, {"stage": "loading_existing_model"})
        logging.info("Reusing trained model from %s", model_path)
        model = joblib.load(model_path)
        params = json.loads(params_path.read_text())
        model_config = config["models"]["price"]
        training_days = model_config.get("training_history_days", config["training_history_days"])
        start_time = args.train_until - pd.Timedelta(days=training_days)
        historical_df = load_historical_training_frame(
            config,
            start_time=start_time,
            end_time=args.eval_until,
        )
    else:
        write_progress(progress_path, {"stage": "training"})
        model, params, historical_df = train_price_model(config, train_until=args.train_until)
        joblib.dump(model, model_path)
        params_path.write_text(json.dumps(params, indent=2))
        model_config = config["models"]["price"]
        training_days = model_config.get("training_history_days", config["training_history_days"])
        start_time = args.train_until - pd.Timedelta(days=training_days)
        historical_df = load_historical_training_frame(
            config,
            start_time=start_time,
            end_time=args.eval_until,
        )
    write_progress(
        progress_path,
        {
            "stage": "loading_eval_data",
            "model": str(model_path),
            "params": str(params_path),
        },
    )

    eval_rows = load_eval_rows(args)
    pred_by_region = load_predispatch_by_region(args.predispatch_template)
    sdo_df = load_sdo_from_influx(
        config,
        start=args.eval_start - pd.Timedelta(days=2),
        end=args.eval_until + pd.Timedelta(hours=args.max_horizon_hours),
    )
    stpasa_df = fc._load_stpasa_regionsolution(args.stpasa)
    model_config = config["models"]["price"]

    parts: list[pd.DataFrame] = []
    run_times = list(eval_rows["forecast_creation_time"].drop_duplicates().sort_values())
    write_progress(progress_path, {"stage": "predicting", "completed_runs": 0, "total_runs": len(run_times)})

    for i, creation_time in enumerate(run_times, start=1):
        run_df = eval_rows[eval_rows["forecast_creation_time"] == creation_time].copy()
        covariates = build_future_covariates(
            run_df,
            creation_time=creation_time,
            model_config=model_config,
            pred_by_region=pred_by_region,
            sdo_df=sdo_df,
            stpasa_df=stpasa_df,
        )
        hist = historical_df[historical_df.index < creation_time].copy()
        covariates = prepend_historical_covariate_context(covariates, hist, model_config)
        required_nonnull = covariates[model_config["feature_cols"]].notna().all(axis=1)
        if not required_nonnull.all():
            logging.warning(
                "Run %s has %s/%s target rows with complete covariates before fill",
                creation_time,
                int(required_nonnull.sum()),
                len(required_nonnull),
            )

        amber_seed = run_df[run_df["horizon_hours"] <= args.apf_seed_hours].copy()
        amber_seed = amber_seed.set_index("forecast_target_time")[["prediction"]].rename(
            columns={"prediction": model_config["target_column"]}
        )
        pred = fc._predict_with_dynamic_handoff(model, params, hist, covariates, model_config, amber_seed)
        pred = pred.rename(columns={model_config["target_column"]: "candidate_prediction"})

        scored = run_df.merge(
            pred[["candidate_prediction"]],
            left_on="forecast_target_time",
            right_index=True,
            how="left",
        )
        scored["incumbent_mwh"] = scored["prediction"] * 1000.0
        scored["candidate_mwh"] = scored["candidate_prediction"] * 1000.0
        scored = scored.dropna(subset=["candidate_mwh", "actual_rrp"])
        parts.append(
            scored[
                [
                    "forecast_creation_time",
                    "forecast_target_time",
                    "horizon_hours",
                    "incumbent_mwh",
                    "candidate_mwh",
                    "actual_rrp",
                ]
            ]
        )

        if i % 10 == 0 or i == len(run_times):
            out = pd.concat(parts, ignore_index=True)
            out.to_parquet(predictions_path, index=False)
            write_progress(
                progress_path,
                {
                    "stage": "predicting",
                    "completed_runs": i,
                    "total_runs": len(run_times),
                    "rows_written": len(out),
                    "last_run": str(creation_time),
                },
            )
            logging.info("Completed %s/%s run slots", i, len(run_times))

    all_scored = pd.concat(parts, ignore_index=True)
    all_scored["adelaide_bucket"] = adelaide_bucket(all_scored["forecast_target_time"])
    all_scored["horizon_bucket"] = horizon_bucket(all_scored["horizon_hours"])
    all_scored.to_parquet(predictions_path, index=False)
    all_scored.to_csv(RESULTS_DIR / f"{args.output_prefix}_predictions.csv", index=False)
    pd.DataFrame(metric_rows(all_scored)).to_csv(metrics_path, index=False)
    write_progress(
        progress_path,
        {
            "stage": "complete",
            "completed_runs": len(run_times),
            "total_runs": len(run_times),
            "rows_written": len(all_scored),
            "metrics": str(metrics_path),
            "predictions": str(predictions_path),
        },
    )
    logging.info("Backtest complete: %s", metrics_path)


if __name__ == "__main__":
    main()
