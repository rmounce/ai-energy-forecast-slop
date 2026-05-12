#!/usr/bin/env python3
"""Compare live LGBM and TFT load forecast accuracy against realised load.

This complements ``analyze_live_load_shadow_gap.py``. The gap script answers
"is TFT above or below LGBM?". This script answers the more operational question:
"when both forecasts and actual load are available on the same run/target grid,
which forecast was closer, and which one under-forecast more often?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
HORIZON_BUCKETS = [
    (0, 48, "0-24h"),
    (48, 96, "24-48h"),
    (96, 144, "48-72h"),
]


def _read_rows(
    path: Path,
    model_name: str,
    *,
    cutoff: pd.Timestamp | None,
    chunksize: int,
    output_prefix: str | None = None,
) -> pd.DataFrame:
    output_prefix = output_prefix or model_name
    cols = [
        "model_name",
        "forecast_creation_time",
        "forecast_target_time",
        "prediction",
        "actual",
    ]
    if not path.exists():
        return pd.DataFrame(
            columns=["run_slot", "target", f"{output_prefix}_prediction", f"{output_prefix}_actual"]
        )

    parts = []
    for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["model_name"].eq(model_name)].copy()
        if chunk.empty:
            continue

        chunk["creation"] = pd.to_datetime(
            chunk["forecast_creation_time"], utc=True, errors="coerce"
        )
        chunk["target"] = pd.to_datetime(
            chunk["forecast_target_time"], utc=True, errors="coerce"
        )
        chunk["prediction"] = pd.to_numeric(chunk["prediction"], errors="coerce")
        chunk["actual"] = pd.to_numeric(chunk["actual"], errors="coerce")
        chunk = chunk.dropna(subset=["creation", "target", "prediction", "actual"])

        if cutoff is not None:
            chunk = chunk[chunk["creation"] >= cutoff]
        if chunk.empty:
            continue

        chunk["run_slot"] = chunk["creation"].dt.floor("5min")
        parts.append(chunk[["run_slot", "target", "prediction", "actual"]])

    if not parts:
        return pd.DataFrame(
            columns=["run_slot", "target", f"{output_prefix}_prediction", f"{output_prefix}_actual"]
        )

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["run_slot", "target"]).drop_duplicates(
        ["run_slot", "target"], keep="last"
    )
    return out.rename(
        columns={
            "prediction": f"{output_prefix}_prediction",
            "actual": f"{output_prefix}_actual",
        }
    )


def _time_bucket(target_utc: pd.Series) -> pd.Series:
    local_hour = target_utc.dt.tz_convert("Australia/Adelaide").dt.hour
    return pd.cut(
        local_hour,
        bins=[-1, 5, 11, 16, 20, 24],
        labels=["overnight", "morning", "solar", "evening", "late"],
        right=True,
    ).astype(str)


def _metric_summary(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"n": 0}

    lgbm_err = frame["load_prediction"] - frame["actual_load_w"]
    tft_err = frame["tft_load_prediction"] - frame["actual_load_w"]
    abs_delta = tft_err.abs() - lgbm_err.abs()

    return {
        "n": int(len(frame)),
        "lgbm_mae_w": float(lgbm_err.abs().mean()),
        "tft_mae_w": float(tft_err.abs().mean()),
        "tft_minus_lgbm_mae_w": float(abs_delta.mean()),
        "lgbm_bias_w": float(lgbm_err.mean()),
        "tft_bias_w": float(tft_err.mean()),
        "lgbm_underforecast_fraction": float((lgbm_err < 0).mean()),
        "tft_underforecast_fraction": float((tft_err < 0).mean()),
        "tft_closer_fraction": float((tft_err.abs() < lgbm_err.abs()).mean()),
        "lgbm_closer_fraction": float((lgbm_err.abs() < tft_err.abs()).mean()),
        "equal_fraction": float((lgbm_err.abs() == tft_err.abs()).mean()),
        "mean_actual_load_w": float(frame["actual_load_w"].mean()),
        "mean_lgbm_prediction_w": float(frame["load_prediction"].mean()),
        "mean_tft_prediction_w": float(frame["tft_load_prediction"].mean()),
    }


def analyze(args: argparse.Namespace) -> tuple[dict, pd.DataFrame]:
    cutoff = None
    if args.days is not None:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.days)

    lgbm = _read_rows(
        args.load_log,
        args.lgbm_model_name,
        cutoff=cutoff,
        chunksize=args.chunksize,
        output_prefix="load",
    )
    tft = _read_rows(
        args.tft_load_log,
        "tft_load",
        cutoff=cutoff,
        chunksize=args.chunksize,
    )

    merged = lgbm.merge(tft, on=["run_slot", "target"], how="inner")
    if merged.empty:
        result = {
            "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "days": args.days,
            "load_log": str(args.load_log),
            "tft_load_log": str(args.tft_load_log),
            "lgbm_model_name": args.lgbm_model_name,
            "error": "no matched rows with realised actuals",
        }
        return result, merged

    merged["actual_load_w"] = merged["load_actual"]
    merged["actual_delta_w"] = (merged["load_actual"] - merged["tft_load_actual"]).abs()
    merged["horizon_step"] = (
        (merged["target"] - merged["run_slot"]).dt.total_seconds() / 1800
    ).round().astype(int)
    merged = merged[(merged["horizon_step"] >= 0) & (merged["horizon_step"] < 144)].copy()
    merged["target_time_bucket"] = _time_bucket(merged["target"])

    result = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "days": args.days,
        "load_log": str(args.load_log),
        "tft_load_log": str(args.tft_load_log),
        "lgbm_model_name": args.lgbm_model_name,
        "matched_rows": int(len(merged)),
        "matched_runs": int(merged["run_slot"].nunique()),
        "run_start_utc": merged["run_slot"].min().isoformat() if not merged.empty else None,
        "run_end_utc": merged["run_slot"].max().isoformat() if not merged.empty else None,
        "actual_disagreement_max_w": float(merged["actual_delta_w"].max()) if not merged.empty else None,
        "overall": _metric_summary(merged),
        "by_horizon_bucket": {},
        "by_target_time_bucket_adelaide": {},
    }

    for lo, hi, label in HORIZON_BUCKETS:
        bucket = merged[(merged["horizon_step"] >= lo) & (merged["horizon_step"] < hi)]
        result["by_horizon_bucket"][label] = _metric_summary(bucket)

    for label, bucket in merged.groupby("target_time_bucket", observed=True):
        result["by_target_time_bucket_adelaide"][str(label)] = _metric_summary(bucket)

    return result, merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument(
        "--lgbm-model-name",
        default="load",
        help=(
            "Model name to read from load_forecast_log.csv. Existing history usually "
            "has 'load'; future logs also include 'load_p65' and 'load_p75'."
        ),
    )
    parser.add_argument("--load-log", type=Path, default=ROOT / "load_forecast_log.csv")
    parser.add_argument("--tft-load-log", type=Path, default=ROOT / "tft_load_forecast_log.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "live_load_accuracy_latest.json")
    parser.add_argument("--matched-output", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()

    result, matched = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    if args.matched_output is not None and not matched.empty:
        args.matched_output.parent.mkdir(parents=True, exist_ok=True)
        matched.to_csv(args.matched_output, index=False)

    if "error" in result:
        print(result["error"])
        return

    overall = result["overall"]
    print(
        f"Matched {result['matched_rows']:,} rows across {result['matched_runs']:,} runs "
        f"({result['run_start_utc']} -> {result['run_end_utc']})"
    )
    print(
        "Overall: "
        f"LGBM MAE {overall['lgbm_mae_w']:.1f} W, "
        f"TFT MAE {overall['tft_mae_w']:.1f} W, "
        f"TFT-LGBM MAE {overall['tft_minus_lgbm_mae_w']:+.1f} W"
    )
    print(
        "Bias/underforecast: "
        f"LGBM {overall['lgbm_bias_w']:+.1f} W "
        f"({overall['lgbm_underforecast_fraction']:.1%} under), "
        f"TFT {overall['tft_bias_w']:+.1f} W "
        f"({overall['tft_underforecast_fraction']:.1%} under)"
    )
    for label, row in result["by_horizon_bucket"].items():
        if row["n"] == 0:
            continue
        print(
            f"{label}: LGBM {row['lgbm_mae_w']:.1f} W, "
            f"TFT {row['tft_mae_w']:.1f} W, "
            f"TFT-LGBM {row['tft_minus_lgbm_mae_w']:+.1f} W"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
