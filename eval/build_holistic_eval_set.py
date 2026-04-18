#!/usr/bin/env python3
"""
Phase 6 — Build holistic eval index.

Builds a stratified sample of 72h evaluation windows for holistic dispatch
simulation.  Windows are restricted to the period where legacy LightGBM
forecast logs are available (July 2025 onwards).

Stratification (by actual SA1 dispatch price within each 72h window):
  spike   — ≥1 interval with RRP ≥ $300/MWh
  low     — ≥1 interval with RRP ≤ $0/MWh  (no spike in same window)
  normal  — all other windows

Output: eval/results/holistic_eval_index.parquet
  columns: start_time (UTC, tz-aware), stratum

Usage:
    source .venv/bin/activate
    python eval/build_holistic_eval_set.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "eval" / "results"
FORECAST_LOG = ROOT / "price_forecast_log.csv"

# ── Eval configuration ────────────────────────────────────────────────────────
WINDOW_STEPS = 144        # 72h at 30-min resolution
STRIDE_H     = 6          # hours between candidate window starts
TARGET_PER_STRATUM = 300
SPIKE_THRESH = 300.0      # $/MWh — dispatch price event
LOW_THRESH   = -50.0      # $/MWh — genuine curtailment (not just cheap solar midday)
# Note: SA prices are often $0-$50/MWh at midday solar peaks, so using $0 as low
# threshold left almost no "normal" windows. -$50 captures genuine negative-price events.

# Eval window: start where LightGBM log begins (July 2025), leave gap at end
EVAL_START = "2025-07-21T00:00:00Z"  # first full day after first forecast log entry
EVAL_END   = "2026-04-01T00:00:00Z"  # leave most recent 2-3 weeks for out-of-sample guard


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


def query_actual_prices(client: InfluxDBClient, start: str, end: str) -> pd.Series:
    """Query 30-min SA1 dispatch prices ($/MWh) from InfluxDB."""
    q = (
        f'SELECT mean("price") AS price'
        f' FROM "rp_30m"."aemo_dispatch_sa1_30m"'
        f' WHERE time >= \'{start}\' AND time < \'{end}\''
        f' GROUP BY time(30m) fill(none)'
    )
    result = client.query(q)
    rows = list(result.get_points())
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")["price"].sort_index()


def load_forecast_log_run_times() -> set:
    """
    Load the set of 30-min window start times that have LightGBM forecasts.
    Returns set of UTC Timestamps (rounded to 30-min boundary).
    """
    print("Loading price_forecast_log.csv creation times...")
    df = pd.read_csv(
        FORECAST_LOG,
        usecols=["forecast_creation_time", "forecast_target_time"],
        dtype_backend="pyarrow",
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="mixed"
    )
    # Minimum target time per run = window start
    min_targets = df.groupby(df["forecast_creation_time"])["forecast_target_time"].min()
    run_starts = set(min_targets.dt.floor("30min").dt.tz_convert("UTC"))
    print(f"  Found {len(run_starts):,} unique LightGBM forecast runs")
    return run_starts


def classify_window(prices: np.ndarray) -> str:
    """Classify a 72h window by its price regime."""
    if np.any(prices >= SPIKE_THRESH):
        return "spike"
    if np.any(prices <= LOW_THRESH):
        return "low"
    return "normal"


def main():
    config = load_config()
    client = InfluxDBClient(**config["influxdb"])

    print(f"Querying SA1 actual prices {EVAL_START} → {EVAL_END}...")
    try:
        prices = query_actual_prices(client, EVAL_START, EVAL_END)
    finally:
        client.close()

    print(f"  Got {len(prices):,} 30-min price points")
    if prices.empty:
        print("ERROR: No price data returned from InfluxDB.")
        sys.exit(1)

    # ── Load LightGBM run starts ──────────────────────────────────────────────
    lgbm_starts = load_forecast_log_run_times()

    # ── Build candidate windows (stride = STRIDE_H) ───────────────────────────
    stride_steps = STRIDE_H * 2  # steps per stride (30-min)
    all_times = prices.index
    price_arr = prices.values

    records = []
    step = 0
    while step + WINDOW_STEPS <= len(all_times):
        start_ts = all_times[step]
        window_prices = price_arr[step: step + WINDOW_STEPS]

        # Only keep windows where all steps have actual prices (no gaps)
        if len(window_prices) < WINDOW_STEPS:
            step += stride_steps
            continue

        # Only keep windows where a LightGBM forecast exists
        start_floor = start_ts.floor("30min")
        if start_floor not in lgbm_starts:
            step += stride_steps
            continue

        stratum = classify_window(window_prices)
        records.append({"start_time": start_ts, "stratum": stratum})
        step += stride_steps

    df_all = pd.DataFrame(records)
    if df_all.empty:
        print("ERROR: No valid windows found.")
        sys.exit(1)

    counts = df_all["stratum"].value_counts()
    print(f"\nCandidate windows: {len(df_all):,}")
    for s in ["spike", "low", "normal"]:
        print(f"  {s:8s}: {counts.get(s, 0):,}")

    # ── Stratified sample ─────────────────────────────────────────────────────
    sampled = []
    for stratum in ["spike", "low", "normal"]:
        subset = df_all[df_all["stratum"] == stratum]
        n_take = min(len(subset), TARGET_PER_STRATUM)
        if n_take == 0:
            print(f"  WARNING: no '{stratum}' windows available")
            continue
        sampled.append(subset.sample(n=n_take, random_state=42))
        print(f"  Sampled {n_take:,} {stratum} windows (available: {len(subset):,})")

    df_index = pd.concat(sampled).sort_values("start_time").reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "holistic_eval_index.parquet"
    df_index.to_parquet(out, index=False)
    print(f"\nSaved {len(df_index):,} windows → {out.relative_to(ROOT)}")
    print(df_index["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
