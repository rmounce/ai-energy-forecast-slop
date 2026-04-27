#!/usr/bin/env python3
"""
Retrospective Tier 1 tactical LGBM inference for holistic eval hybrid source.

For each eval window in holistic_eval_index.parquet, runs the Tier 1 LGBM q50
model using historical P5MIN + actual price/demand/PV data.

Data sources:
  P5MIN forecasts  — data/parquet/aemo_p5min_sa1.parquet (no InfluxDB retention issue)
  5min actuals     — data/parquet/actuals_sa1_5m.parquet
  5min PV          — InfluxDB rp_5m.power_pv_5m (covers July 2025 onwards)

Output: eval/results/retro_tier1_forecasts.pkl
  dict with keys:
    'forecasts': {UTC Timestamp (window start) -> np.ndarray shape (2,)} in $/MWh
      [0] = mean of h0..h5 (0–30 min)
      [1] = mean of h6..h11 (30–60 min)
    'n_steps': 2
    'description': str

These 2 steps map to steps 0 and 1 of the 30-min 144-step holistic eval window.
The tier1_tier2_hybrid source in holistic_eval.py uses these for the first hour,
TFT q50 for steps 2–143 (1h–72h).

Usage:
    source .venv/bin/activate
    nice -n 19 python eval/retro_tier1_inference.py [--overwrite]
"""

import json
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytz
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tariff_utils import load_tariff_profile, tariffed_price_frame_from_wholesale_mwh

RESULTS_DIR     = ROOT / "eval" / "results"
PARQUET_DIR     = ROOT / "data" / "parquet"
INDEX_FILE      = RESULTS_DIR / "holistic_eval_index.parquet"
OUT_FILE        = RESULTS_DIR / "retro_tier1_forecasts.pkl"
MODEL_DIR       = ROOT / "models" / "lgbm_tactical"

P5MIN_PARQUET   = PARQUET_DIR / "aemo_p5min_sa1.parquet"
ACTUALS_PARQUET = PARQUET_DIR / "actuals_sa1_5m.parquet"

OUTPUT_STEPS = 12   # 12 × 5min = 60min
LOOKBACK_H   = 3    # hours of 5min history needed before each window start


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


CONFIG = load_config()
GENERAL_TARIFF_MAP, FEED_IN_TARIFF_MAP, NETWORK_LOSS_FACTOR = load_tariff_profile(CONFIG, ROOT)


def load_p5min_from_parquet() -> dict:
    """
    Load P5MIN runs from data/parquet/aemo_p5min_sa1.parquet.
    Returns {run_time Timestamp (UTC) → list[12] rrp values ($/MWh) sorted by interval_dt}.
    """
    print(f"Loading P5MIN from {P5MIN_PARQUET.relative_to(ROOT)}...")
    t0 = time.time()
    df = pd.read_parquet(P5MIN_PARQUET, columns=["run_time", "interval_dt", "rrp"])
    df = df.sort_values(["run_time", "interval_dt"])

    p5min_runs: dict = {}
    for rt, grp in df.groupby("run_time"):
        if len(grp) >= OUTPUT_STEPS:
            rt_utc = rt if rt.tzinfo is not None else rt.tz_localize("UTC")
            p5min_runs[rt_utc] = grp["rrp"].values[:OUTPUT_STEPS].tolist()

    print(f"  {len(p5min_runs):,} P5MIN runs in {time.time()-t0:.1f}s  "
          f"({df['run_time'].min().date()} – {df['run_time'].max().date()})")
    return p5min_runs


def load_actuals_from_parquet() -> pd.DataFrame:
    """
    Load 5min actual prices and demand from data/parquet/actuals_sa1_5m.parquet.
    Returns UTC-indexed DataFrame with columns [rrp, total_demand].
    """
    print(f"Loading 5min actuals from {ACTUALS_PARQUET.relative_to(ROOT)}...")
    t0 = time.time()
    df = pd.read_parquet(ACTUALS_PARQUET, columns=["rrp", "total_demand"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    print(f"  {len(df):,} rows  ({df.index.min().date()} – {df.index.max().date()})  "
          f"in {time.time()-t0:.1f}s")
    return df


def fetch_pv_from_influxdb(client, start_iso: str, end_iso: str) -> pd.Series:
    """
    Fetch 5min PV data from InfluxDB rp_5m.power_pv_5m.
    Returns UTC-indexed float Series. Empty series if no data (falls back to 0.0 in features).
    """
    print(f"Fetching 5min PV from InfluxDB {start_iso} → {end_iso}...")
    t0 = time.time()
    q = (
        f'SELECT mean("mean_value") AS pv'
        f' FROM "rp_5m"."power_pv_5m"'
        f' WHERE time >= \'{start_iso}\' AND time < \'{end_iso}\''
        f' GROUP BY time(5m) fill(none)'
    )
    rows = list(client.query(q).get_points())
    if not rows:
        print("  No PV data; residual_demand will fall back to total_demand")
        return pd.Series(dtype=float, name="pv")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    s = df.set_index("time")["pv"].sort_index()
    print(f"  {len(s):,} PV rows in {time.time()-t0:.1f}s")
    return s


def _tariff_feature_block(run_time: pd.Timestamp, p5min_rrp: list[float]) -> np.ndarray:
    intervals = pd.date_range(start=run_time, periods=OUTPUT_STEPS, freq="5min", tz="UTC")
    tariffed = tariffed_price_frame_from_wholesale_mwh(
        pd.Series(np.asarray(p5min_rrp, dtype=np.float64), index=intervals),
        timezone=CONFIG["timezone"],
        general_tariff_map=GENERAL_TARIFF_MAP,
        feed_in_tariff_map=FEED_IN_TARIFF_MAP,
        network_loss_factor=NETWORK_LOSS_FACTOR,
        gst_rate=CONFIG["gst_rate"],
    )
    import_curve = tariffed["general_price_mwh"].to_numpy(dtype=np.float32, copy=False)
    export_curve = tariffed["feed_in_price_mwh"].to_numpy(dtype=np.float32, copy=False)
    return np.array([
        import_curve[0],
        export_curve[0],
        import_curve.mean(),
        import_curve.max(),
        np.ptp(import_curve),
        export_curve.mean(),
        export_curve.max(),
        np.ptp(export_curve),
    ], dtype=np.float32)


def build_features(run_time: pd.Timestamp,
                   p5min_rrp: list,
                   prev_p5min_h0: float,
                   act_df: pd.DataFrame,
                   pv_series: pd.Series) -> np.ndarray:
    """
    Build 32-feature base vector at run_time.
    Feature order matches FEATURE_NAMES in data/build_tactical_dataset.py exactly.
    """
    def asof(series, ts, default=0.0):
        try:
            v = series.asof(ts) if len(series) > 0 else np.nan
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    t1 = run_time - pd.Timedelta(minutes=5)
    t2 = run_time - pd.Timedelta(minutes=10)
    t6 = run_time - pd.Timedelta(minutes=30)

    rrp_series = act_df["rrp"] if "rrp" in act_df.columns else pd.Series(dtype=float)
    td_series  = act_df["total_demand"] if "total_demand" in act_df.columns else pd.Series(dtype=float)

    actual_t1 = asof(rrp_series, t1)
    actual_t2 = asof(rrp_series, t2)
    actual_t6 = asof(rrp_series, t6)

    divergence_t1 = (actual_t1 - prev_p5min_h0) if not np.isnan(prev_p5min_h0) else 0.0

    rrp_to_t1 = rrp_series.loc[:t1] if len(rrp_series) > 0 else pd.Series(dtype=float)
    if len(rrp_to_t1) >= 6:
        rolling_1h_std = float(rrp_to_t1.tail(12).std())
        rolling_3h_max = float(rrp_to_t1.tail(36).max())
    else:
        rolling_1h_std = 0.0
        rolling_3h_max = float(max(p5min_rrp)) if p5min_rrp else 0.0

    td_t1 = asof(td_series, t1)
    pv_t1 = asof(pv_series, t1, default=0.0)
    residual_demand_t1 = td_t1 - pv_t1   # falls back to td_t1 when pv_t1=0.0

    brisbane_tz = pytz.timezone("Australia/Brisbane")
    rt_bne = run_time.astimezone(brisbane_tz)
    hour_frac = rt_bne.hour + rt_bne.minute / 60.0
    hour_sin  = float(np.sin(2 * np.pi * hour_frac / 24.0))
    hour_cos  = float(np.cos(2 * np.pi * hour_frac / 24.0))
    dow_sin   = float(np.sin(2 * np.pi * rt_bne.weekday() / 7.0))
    dow_cos   = float(np.cos(2 * np.pi * rt_bne.weekday() / 7.0))

    tariff_block = _tariff_feature_block(run_time, p5min_rrp)

    return np.array([
        *p5min_rrp,           # h0..h11  [12]
        *tariff_block,        # tariff-aware import/export summaries [8]
        divergence_t1,        # [1]
        actual_t1,            # [1]
        actual_t2,            # [1]
        actual_t6,            # [1]
        rolling_1h_std,       # [1]
        rolling_3h_max,       # [1]
        residual_demand_t1,   # [1]
        hour_sin,             # [1]
        hour_cos,             # [1]
        dow_sin,              # [1]
        dow_cos,              # [1]
        0.0,                  # is_imputed_p5min [1]
    ], dtype=np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.overwrite:
        print(f"Output exists: {OUT_FILE.relative_to(ROOT)}")
        print("Use --overwrite to regenerate.")
        sys.exit(0)

    if not (MODEL_DIR / "lgbm_q50.pkl").exists():
        print(f"ERROR: Tier 1 model not found at {MODEL_DIR}/lgbm_q50.pkl")
        sys.exit(1)

    print("Loading Tier 1 LGBM q50 model...")
    q50_model = joblib.load(MODEL_DIR / "lgbm_q50.pkl")

    df_index = pd.read_parquet(INDEX_FILE)
    df_index["start_time"] = pd.to_datetime(df_index["start_time"], utc=True)
    print(f"Eval index: {len(df_index)} windows  "
          f"({df_index['start_time'].min().date()} – {df_index['start_time'].max().date()})")

    p5min_runs = load_p5min_from_parquet()
    act_df     = load_actuals_from_parquet()

    # PV: InfluxDB for eval period (covers July 2025+)
    fetch_start = (df_index["start_time"].min() - pd.Timedelta(hours=LOOKBACK_H)
                   ).strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_end = (df_index["start_time"].max() + pd.Timedelta(minutes=5)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
    client = InfluxDBClient(**CONFIG["influxdb"])
    try:
        pv_series = fetch_pv_from_influxdb(client, fetch_start, fetch_end)
    finally:
        client.close()

    # Build sorted run_time array for O(log n) per-window lookup
    run_times_sorted = sorted(p5min_runs.keys())
    run_times_ns = np.array([t.value for t in run_times_sorted], dtype=np.int64)
    print(f"\nP5MIN coverage: {run_times_sorted[0]} – {run_times_sorted[-1]}")

    print("Running Tier 1 inference...")
    forecasts: dict = {}
    skipped_no_p5min = 0
    t0 = time.time()

    for row in df_index.itertuples():
        start_ts = row.start_time

        # Latest P5MIN run_time at or before start_ts
        ts_ns = start_ts.value
        idx_ge = int(np.searchsorted(run_times_ns, ts_ns, side="right"))
        if idx_ge == 0:
            skipped_no_p5min += 1
            continue
        run_time  = run_times_sorted[idx_ge - 1]
        p5min_rrp = p5min_runs[run_time]

        # Previous run (for aemo_divergence_t1 feature)
        prev_rt = run_time - pd.Timedelta(minutes=5)
        prev_p5min_h0 = (p5min_runs[prev_rt][0]
                         if prev_rt in p5min_runs else float("nan"))

        feats = build_features(run_time, p5min_rrp, prev_p5min_h0, act_df, pv_series)

        # Long format: [12, 33] — base features repeated × 12 + horizon index
        X_long = np.column_stack([
            np.tile(feats, (OUTPUT_STEPS, 1)),
            np.arange(OUTPUT_STEPS, dtype=np.float32).reshape(-1, 1),
        ])

        q50_raw = q50_model.predict(X_long).astype(np.float64)  # [12] $/MWh

        # Average to 2 × 30min steps
        forecasts[start_ts] = np.array([
            float(np.mean(q50_raw[:6])),   # h0..h5  → step 0 (0–30 min)
            float(np.mean(q50_raw[6:])),   # h6..h11 → step 1 (30–60 min)
        ])

    elapsed = time.time() - t0
    n = len(forecasts)
    total = len(df_index)
    print(f"Done: {n}/{total} windows in {elapsed:.1f}s ({n/max(elapsed, 0.001):.0f} wins/s)")
    if skipped_no_p5min:
        print(f"  Skipped {skipped_no_p5min} windows: no P5MIN run found before window start")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "wb") as f:
        pickle.dump({
            "forecasts":   forecasts,
            "n_steps":     2,
            "description": "Tier 1 LGBM q50 averaged to 2×30min steps ($/MWh). "
                           "step[0]=mean(h0..h5, 0-30min), step[1]=mean(h6..h11, 30-60min). "
                           "P5MIN from data/parquet/aemo_p5min_sa1.parquet.",
        }, f)
    print(f"Saved → {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
