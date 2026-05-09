#!/usr/bin/env python3
"""Compare live LightGBM and TFT load shadow forecasts.

This is a forecast-shape diagnostic, not an accuracy verdict. The live forecast
logs do not backfill realised outcomes for every target interval, but they do
show whether the TFT load shadow is systematically above or below the current
production LightGBM load forecast on the same run/target grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "eval" / "results"
BUCKETS = [(0, 48, "0-24h"), (48, 96, "24-48h"), (96, 144, "48-72h")]


def _read_forecast_rows(path: Path, model_name: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    cols = ["model_name", "forecast_creation_time", "forecast_target_time", "prediction"]
    parts = []
    if not path.exists():
        return pd.DataFrame(columns=["run_slot", "target", model_name])

    for chunk in pd.read_csv(path, usecols=cols, chunksize=500_000, low_memory=False):
        chunk = chunk[chunk["model_name"].eq(model_name)].copy()
        if chunk.empty:
            continue
        chunk["creation"] = pd.to_datetime(
            chunk["forecast_creation_time"], utc=True, errors="coerce"
        )
        chunk = chunk[chunk["creation"] >= cutoff]
        if chunk.empty:
            continue
        chunk["target"] = pd.to_datetime(
            chunk["forecast_target_time"], utc=True, errors="coerce"
        )
        chunk["run_slot"] = chunk["creation"].dt.floor("5min")
        parts.append(chunk[["run_slot", "target", "prediction"]])

    if not parts:
        return pd.DataFrame(columns=["run_slot", "target", model_name])
    return pd.concat(parts, ignore_index=True).rename(columns={"prediction": model_name})


def _summary(frame: pd.DataFrame) -> dict:
    out = {
        "n": int(len(frame)),
        "lgbm_mean_w": float(frame["load"].mean()),
        "tft_mean_w": float(frame["tft_load"].mean()),
        "delta_mean_w": float(frame["delta_w"].mean()),
        "delta_p10_w": float(frame["delta_w"].quantile(0.10)),
        "delta_p90_w": float(frame["delta_w"].quantile(0.90)),
        "fraction_tft_below_lgbm": float((frame["delta_w"] < 0).mean()),
    }
    return out


def analyze(days: int, load_log: Path, tft_load_log: Path, fallback_tft_log: Path) -> dict:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    lgbm = _read_forecast_rows(load_log, "load", cutoff)

    tft_source = tft_load_log if tft_load_log.exists() else fallback_tft_log
    tft = _read_forecast_rows(tft_source, "tft_load", cutoff)

    merged = lgbm.merge(tft, on=["run_slot", "target"], how="inner")
    if merged.empty:
        return {
            "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "days": days,
            "load_log": str(load_log),
            "tft_load_log": str(tft_source),
            "error": "no overlapping load/tft_load rows",
        }

    merged["horizon_step"] = (
        (merged["target"] - merged["run_slot"]).dt.total_seconds() / 1800
    ).round().astype(int)
    merged = merged[(merged["horizon_step"] >= 0) & (merged["horizon_step"] < 144)].copy()
    if merged.empty:
        return {
            "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "days": days,
            "load_log": str(load_log),
            "tft_load_log": str(tft_source),
            "error": "no overlapping rows inside the 0-143 step load forecast horizon",
        }
    merged["delta_w"] = merged["tft_load"] - merged["load"]

    result = {
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "days": days,
        "load_log": str(load_log),
        "tft_load_log": str(tft_source),
        "matched_rows": int(len(merged)),
        "matched_runs": int(merged["run_slot"].nunique()),
        "run_start_utc": merged["run_slot"].min().isoformat(),
        "run_end_utc": merged["run_slot"].max().isoformat(),
        "overall": _summary(merged),
        "by_horizon_bucket": {},
    }

    for lo, hi, label in BUCKETS:
        bucket = merged[(merged["horizon_step"] >= lo) & (merged["horizon_step"] < hi)]
        if not bucket.empty:
            result["by_horizon_bucket"][label] = _summary(bucket)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--load-log", type=Path, default=ROOT / "load_forecast_log.csv")
    parser.add_argument("--tft-load-log", type=Path, default=ROOT / "tft_load_forecast_log.csv")
    parser.add_argument(
        "--fallback-tft-log", type=Path, default=ROOT / "tft_price_forecast_log.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "live_load_shadow_gap_latest.json",
    )
    args = parser.parse_args()

    result = analyze(args.days, args.load_log, args.tft_load_log, args.fallback_tft_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

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
        f"LGBM {overall['lgbm_mean_w']:.1f} W, "
        f"TFT {overall['tft_mean_w']:.1f} W, "
        f"delta {overall['delta_mean_w']:+.1f} W, "
        f"TFT below LGBM {overall['fraction_tft_below_lgbm']:.1%}"
    )
    for label, row in result["by_horizon_bucket"].items():
        print(
            f"{label}: delta {row['delta_mean_w']:+.1f} W "
            f"(p10 {row['delta_p10_w']:+.1f}, p90 {row['delta_p90_w']:+.1f})"
        )


if __name__ == "__main__":
    main()
