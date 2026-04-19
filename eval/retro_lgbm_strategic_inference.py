#!/usr/bin/env python3
"""
Retrospective LightGBM strategic (30-min / 72-hour) inference for holistic eval.

For each eval window in holistic_eval_index.parquet, runs the LightGBM strategic
model (q5/q50/q95) using PREDISPATCH + OOF debiased RRP + 30-min actuals.

Spike routing: upstream LightGBM spike classifier (spike_clf_predictions.parquet,
threshold=0.65) determines whether to apply the OOF debiaser or pass raw PREDISPATCH
through unchanged — same routing logic as retro_tft_inference.py.

Key alignment:
  start_ts  = window start (first 30-min interval)
  run_time  = start_ts - 30min  (PREDISPATCH run that produced step 0 = start_ts)
  step 0    = start_ts = run_time + 30min
  step 143  = start_ts + 143 × 30min = run_time + 72h

Data sources (all parquet, no InfluxDB required):
  aemo_predispatch_sa1.parquet    — PREDISPATCH RRP/demand/net_interchange steps 0-55
  debiased_pd_rrp_oof.parquet     — OOF-debiased PREDISPATCH RRP for steps 0-55
  spike_clf_predictions.parquet   — prob_spike per run_time for routing
  actuals_sa1.parquet             — 30-min actual RRP for lag features

Output: eval/results/retro_lgbm_strategic_forecasts.pkl
  {UTC Timestamp (window start_ts) -> np.ndarray shape (144, 3)} in $/MWh
    column 0 = q5, column 1 = q50, column 2 = q95

Usage:
    source .venv/bin/activate
    nice -n 19 python eval/retro_lgbm_strategic_inference.py [--overwrite]
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR      = ROOT / "eval" / "results"
PARQUET_DIR      = ROOT / "data" / "parquet"
INDEX_FILE       = RESULTS_DIR / "holistic_eval_index.parquet"
OUT_FILE         = RESULTS_DIR / "retro_lgbm_strategic_forecasts.pkl"
MODEL_DIR        = ROOT / "models" / "lgbm_strategic"
SPIKE_CLF_FILE   = PARQUET_DIR / "spike_clf_predictions.parquet"

WINDOW_STEPS         = 144
PD_STEPS             = 56
BRISBANE_TZ          = "Australia/Brisbane"
SPIKE_ROUTE_THRESHOLD = 0.65  # prob_spike > this → bypass debiaser; matches TFT routing

FEATURE_NAMES = (
    ["step_idx", "has_pd_covariate"]
    + ["pd_rrp_debiased", "pd_demand", "pd_net_interchange"]
    + [f"actual_rrp_lag{k}" for k in [1, 2, 4, 8]]
    + ["actual_rrp_max_6h", "actual_rrp_max_24h"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    + ["rt_hour_sin", "rt_hour_cos", "rt_dow_sin", "rt_dow_cos"]
)


def _time_enc(ts: pd.DatetimeIndex, prefix: str = "") -> dict:
    t = ts.tz_convert(BRISBANE_TZ)
    return {
        f"{prefix}hour_sin":  np.sin(2 * np.pi * t.hour / 24),
        f"{prefix}hour_cos":  np.cos(2 * np.pi * t.hour / 24),
        f"{prefix}dow_sin":   np.sin(2 * np.pi * t.dayofweek / 7),
        f"{prefix}dow_cos":   np.cos(2 * np.pi * t.dayofweek / 7),
        f"{prefix}month_sin": np.sin(2 * np.pi * (t.month - 1) / 12),
        f"{prefix}month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
    }


def load_models() -> dict:
    models = {}
    for q, fname in [(0.05, "lgbm_q05.pkl"), (0.50, "lgbm_q50.pkl"), (0.95, "lgbm_q95.pkl")]:
        with open(MODEL_DIR / fname, "rb") as f:
            obj = pickle.load(f)
        models[q] = obj["model"]
    return models


def load_predispatch() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / "aemo_predispatch_sa1.parquet",
                         columns=["run_time", "interval_dt", "rrp",
                                  "total_demand", "net_interchange"])
    df["run_time"]    = pd.to_datetime(df["run_time"],    utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    df["step_idx"] = (
        (df["interval_dt"] - df["run_time"]).dt.total_seconds() / 1800
    ).round().astype(int) - 1
    df = df[(df["step_idx"] >= 0) & (df["step_idx"] < PD_STEPS)]
    return df.set_index(["run_time", "step_idx"])


def load_oof() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / "debiased_pd_rrp_oof.parquet")
    df["run_time"]    = pd.to_datetime(df["run_time"],    utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    return df.set_index(["run_time", "interval_dt"])["oof_debiased_rrp"]


def load_actuals() -> pd.Series:
    df = pd.read_parquet(PARQUET_DIR / "actuals_sa1.parquet", columns=["time", "rrp"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.dropna(subset=["rrp"]).set_index("time")["rrp"].sort_index()


def build_window_features(run_time: pd.Timestamp,
                          pd_by_run: pd.DataFrame,
                          oof_by_run: pd.Series,
                          actuals_ts: pd.Series,
                          roll_6h: pd.Series,
                          roll_24h: pd.Series,
                          spike_prob_by_run: dict | None = None) -> np.ndarray:
    """
    Build (WINDOW_STEPS, len(FEATURE_NAMES)) feature matrix for one run_time.
    Returns float32 array.

    If spike_prob_by_run is provided and prob_spike > SPIKE_ROUTE_THRESHOLD,
    raw PREDISPATCH rrp is used instead of OOF-debiased (bypass debiaser).
    """
    step_range = np.arange(WINDOW_STEPS)
    target_dts = pd.DatetimeIndex([
        run_time + pd.Timedelta(minutes=30 * (h + 1)) for h in step_range
    ], tz="UTC")

    # Spike routing: bypass debiaser for windows classified as spike
    bypass_debiaser = False
    if spike_prob_by_run is not None:
        prob = spike_prob_by_run.get(run_time, 0.0)
        bypass_debiaser = prob > SPIKE_ROUTE_THRESHOLD

    # PREDISPATCH covariates (steps 0..55)
    pd_rrp   = np.full(WINDOW_STEPS, np.nan, dtype=np.float32)
    pd_dem   = np.full(WINDOW_STEPS, np.nan, dtype=np.float32)
    pd_netix = np.full(WINDOW_STEPS, np.nan, dtype=np.float32)
    has_pd   = np.zeros(WINDOW_STEPS, dtype=np.float32)

    if run_time in pd_by_run.index.get_level_values(0):
        pd_rows = pd_by_run.loc[run_time]
        for step_idx, row in pd_rows.iterrows():
            if 0 <= step_idx < PD_STEPS:
                if bypass_debiaser:
                    pd_rrp[step_idx] = float(row["rrp"])
                else:
                    # OOF debiased — fall back to raw if missing
                    iv_dt = run_time + pd.Timedelta(minutes=30 * (step_idx + 1))
                    oof_key = (run_time, iv_dt)
                    if oof_key in oof_by_run.index:
                        pd_rrp[step_idx] = float(oof_by_run.loc[oof_key])
                    else:
                        pd_rrp[step_idx] = float(row["rrp"])
                pd_dem[step_idx]   = float(row["total_demand"])
                pd_netix[step_idx] = float(row["net_interchange"])
                has_pd[step_idx]   = 1.0

    # Lag features (same for all steps — from actual history at run_time)
    lags = {}
    for k in [1, 2, 4, 8]:
        lt = run_time - pd.Timedelta(minutes=30 * k)
        lags[f"actual_rrp_lag{k}"] = float(actuals_ts.get(lt, np.nan))
    lags["actual_rrp_max_6h"]  = float(
        roll_6h.reindex([run_time], method="ffill").iloc[0]
        if run_time >= roll_6h.index.min() else np.nan
    )
    lags["actual_rrp_max_24h"] = float(
        roll_24h.reindex([run_time], method="ffill").iloc[0]
        if run_time >= roll_24h.index.min() else np.nan
    )

    # Time encodings at target intervals
    tf = _time_enc(target_dts)
    rt_tf = _time_enc(pd.DatetimeIndex([run_time] * WINDOW_STEPS, tz="UTC"), prefix="rt_")

    mat = np.column_stack([
        step_range.astype(np.float32),                     # step_idx
        has_pd,                                             # has_pd_covariate
        pd_rrp,                                             # pd_rrp_debiased
        pd_dem,                                             # pd_demand
        pd_netix,                                           # pd_net_interchange
        np.full(WINDOW_STEPS, lags["actual_rrp_lag1"]),
        np.full(WINDOW_STEPS, lags["actual_rrp_lag2"]),
        np.full(WINDOW_STEPS, lags["actual_rrp_lag4"]),
        np.full(WINDOW_STEPS, lags["actual_rrp_lag8"]),
        np.full(WINDOW_STEPS, lags["actual_rrp_max_6h"]),
        np.full(WINDOW_STEPS, lags["actual_rrp_max_24h"]),
        tf["hour_sin"].astype(np.float32),
        tf["hour_cos"].astype(np.float32),
        tf["dow_sin"].astype(np.float32),
        tf["dow_cos"].astype(np.float32),
        tf["month_sin"].astype(np.float32),
        tf["month_cos"].astype(np.float32),
        rt_tf["rt_hour_sin"].astype(np.float32),
        rt_tf["rt_hour_cos"].astype(np.float32),
        rt_tf["rt_dow_sin"].astype(np.float32),
        rt_tf["rt_dow_cos"].astype(np.float32),
    ])
    return mat.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.overwrite:
        print(f"Output exists: {OUT_FILE}")
        print("Use --overwrite to regenerate.")
        sys.exit(0)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models...", flush=True)
    models = load_models()
    print(f"  Loaded q5/q50/q95 from {MODEL_DIR}", flush=True)

    print("Loading eval index...", flush=True)
    index = pd.read_parquet(INDEX_FILE)
    print(f"  {len(index):,} eval windows", flush=True)

    print("Loading PREDISPATCH...", flush=True)
    t0 = time.time()
    pd_by_run = load_predispatch()
    print(f"  {pd_by_run.index.get_level_values(0).nunique():,} run_times "
          f"in {time.time()-t0:.1f}s", flush=True)

    print("Loading OOF debiased...", flush=True)
    oof_by_run = load_oof()
    print(f"  {oof_by_run.index.get_level_values(0).nunique():,} run_times", flush=True)

    print("Loading 30-min actuals...", flush=True)
    actuals_ts = load_actuals()
    actuals_30m = actuals_ts.asfreq("30min")
    roll_6h  = actuals_30m.rolling(12,  min_periods=1).max()
    roll_24h = actuals_30m.rolling(48,  min_periods=1).max()
    print(f"  {len(actuals_ts):,} rows "
          f"({actuals_ts.index.min().date()} – {actuals_ts.index.max().date()})",
          flush=True)

    print("Loading spike classifier predictions...", flush=True)
    spike_prob_by_run = None
    if SPIKE_CLF_FILE.exists():
        clf_df = pd.read_parquet(SPIKE_CLF_FILE)
        clf_df["run_time"] = pd.to_datetime(clf_df["run_time"], utc=True)
        spike_prob_by_run = dict(zip(clf_df["run_time"], clf_df["prob_spike"]))
        n_bypass = (clf_df["prob_spike"] > SPIKE_ROUTE_THRESHOLD).sum()
        print(f"  {len(spike_prob_by_run):,} run_times, {n_bypass:,} routed to bypass "
              f"({n_bypass/len(clf_df):.1%}, threshold={SPIKE_ROUTE_THRESHOLD})", flush=True)
    else:
        print("  WARNING: spike_clf_predictions.parquet not found — debiaser applied to all windows",
              flush=True)

    # Run inference
    forecasts: dict = {}
    n_missing_pd = 0
    n_bypassed = 0
    t_start = time.time()

    for i, row in enumerate(index.itertuples()):
        start_ts = row.start_time
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        run_time = start_ts - pd.Timedelta(minutes=30)

        if spike_prob_by_run is not None:
            prob = spike_prob_by_run.get(run_time, 0.0)
            if prob > SPIKE_ROUTE_THRESHOLD:
                n_bypassed += 1

        X = build_window_features(run_time, pd_by_run, oof_by_run,
                                  actuals_ts, roll_6h, roll_24h,
                                  spike_prob_by_run=spike_prob_by_run)

        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

        preds = np.column_stack([
            models[0.05].predict(X_df),
            models[0.50].predict(X_df),
            models[0.95].predict(X_df),
        ]).astype(np.float32)   # (144, 3)

        forecasts[start_ts] = preds

        if X[:PD_STEPS, 1].sum() == 0:   # has_pd_covariate all zero
            n_missing_pd += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (len(index) - i - 1) / rate
            print(f"  {i+1:4d}/{len(index)} "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)  "
                  f"sample q50 step0={preds[0,1]:.1f} step143={preds[143,1]:.1f} $/MWh",
                  flush=True)

    total = time.time() - t_start
    print(f"\nDone: {len(forecasts):,} windows in {total:.1f}s "
          f"({total/len(forecasts)*1000:.0f}ms/window)")
    print(f"  Spike bypassed: {n_bypassed}/{len(forecasts)} windows "
          f"({n_bypassed/len(forecasts):.1%})")
    if n_missing_pd > 0:
        print(f"  WARNING: {n_missing_pd} windows had no PREDISPATCH data")

    out = {
        "forecasts":   forecasts,
        "n_steps":     WINDOW_STEPS,
        "n_quantiles": 3,
        "quantiles":   [0.05, 0.50, 0.95],
        "description": "LightGBM strategic q5/q50/q95, 30-min/72-hour, "
                        "PREDISPATCH steps 0-55 + time/lags steps 56-143. "
                        f"Spike routing threshold={SPIKE_ROUTE_THRESHOLD}.",
    }
    with open(OUT_FILE, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
