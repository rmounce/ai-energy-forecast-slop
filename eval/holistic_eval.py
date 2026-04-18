#!/usr/bin/env python3
"""
Phase 6 — Holistic dispatch simulation.

Runs rolling-MPC LP dispatch on a stratified eval set, comparing:
  oracle       — perfect foresight (actual prices as forecast)
  lgbm_legacy  — as-run legacy LightGBM predictions from price_forecast_log.csv
                 (Amber APF seeded for first 14-28h, LightGBM extrapolation thereafter)
  p5min_naive  — last observed P5MIN price held constant beyond 1h

Actual household load and PV are used as fixed inputs so that only price-forecast
quality drives differences between sources.

Outputs:
  eval/results/holistic_eval_results.csv — per-stratum $/day profit table
  eval/results/holistic_eval_raw.parquet — per-window results for inspection

Usage:
    source .venv/bin/activate
    python eval/holistic_eval.py [--price-only] [--max-windows N]

Flags:
  --price-only    Skip net_load_actuals (price arbitrage only, faster)
  --max-windows N Run at most N windows per stratum (default: all)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.dispatch_simulator import simulate_mpc

RESULTS_DIR = ROOT / "eval" / "results"
FORECAST_LOG = ROOT / "price_forecast_log.csv"
INDEX_FILE = RESULTS_DIR / "holistic_eval_index.parquet"

WINDOW_STEPS  = 144        # 72h at 30-min resolution
INTERVAL_H    = 30 / 60   # 30-min steps

SPIKE_THRESH  = 300.0   # $/MWh
LOW_THRESH    = -50.0   # $/MWh — genuine curtailment (matches build_holistic_eval_set.py)


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


# ── InfluxDB queries ──────────────────────────────────────────────────────────

def query_30m_series(client, measurement, field, start_iso, end_iso):
    """Query a 30-min field from InfluxDB, return float64 Series indexed by UTC time."""
    q = (
        f'SELECT mean("{field}") AS val'
        f' FROM "rp_30m"."{measurement}"'
        f' WHERE time >= \'{start_iso}\' AND time < \'{end_iso}\''
        f' GROUP BY time(30m) fill(none)'
    )
    result = client.query(q)
    rows = list(result.get_points())
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")["val"].sort_index()


def fetch_window_data(client, start_ts: pd.Timestamp) -> dict | None:
    """
    Fetch actual prices + load/PV for a 72h window starting at start_ts.
    Returns dict or None if insufficient data.
    """
    end_ts = start_ts + pd.Timedelta(hours=72)
    start_iso = start_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    prices = query_30m_series(client, "aemo_dispatch_sa1_30m", "price", start_iso, end_iso)
    load   = query_30m_series(client, "power_load_30m",  "mean_value", start_iso, end_iso)
    pv     = query_30m_series(client, "power_pv_30m",    "mean_value", start_iso, end_iso)

    # Require at least 80% of steps; gaps are forward-filled
    if len(prices) < WINDOW_STEPS * 0.80:
        return None

    # Align to uniform 30-min grid
    idx = pd.date_range(start_ts, periods=WINDOW_STEPS, freq="30min", tz="UTC")
    prices_aligned = prices.reindex(idx).ffill().bfill()
    if prices_aligned.isna().sum() > WINDOW_STEPS * 0.05:
        return None  # too many leading/trailing NaN even after fill
    prices_arr = prices_aligned.ffill().bfill().values.astype(np.float64)

    # Load/PV in Watts → convert to kW
    load_kw = load.reindex(idx).ffill().bfill().fillna(0.0).values / 1000.0
    pv_kw   = pv.reindex(idx).ffill().bfill().fillna(0.0).values / 1000.0
    net_load_kw = load_kw - pv_kw  # positive = drawing from grid

    return {
        "actual_prices_mwh": prices_arr,
        "net_load_kw": net_load_kw,
    }


# ── Forecast loading ──────────────────────────────────────────────────────────

def load_lgbm_forecasts() -> dict:
    """
    Load legacy LightGBM forecast log, indexed by window start time.
    Returns dict: Timestamp → np.ndarray of shape (144,) in $/MWh.
    """
    print("Loading LightGBM forecast log (may take a moment)...")
    df = pd.read_csv(
        FORECAST_LOG,
        usecols=["forecast_creation_time", "forecast_target_time", "prediction"],
        dtype_backend="pyarrow",
    )
    df["forecast_target_time"] = pd.to_datetime(df["forecast_target_time"], utc=True, format="mixed")

    # Group by creation time, find window start = min target time
    grouped = df.groupby("forecast_creation_time")
    forecasts = {}
    for creation_time, grp in grouped:
        grp = grp.sort_values("forecast_target_time")
        start_ts = grp["forecast_target_time"].iloc[0].floor("30min")
        if len(grp) < WINDOW_STEPS:
            continue
        # prediction is $/kWh → convert to $/MWh for LP
        preds_mwh = grp["prediction"].values[:WINDOW_STEPS].astype(np.float64) * 1000.0
        forecasts[start_ts] = preds_mwh

    print(f"  Loaded {len(forecasts):,} LightGBM runs")
    return forecasts


# ── Simulation ────────────────────────────────────────────────────────────────

def run_window(actual_prices_mwh, forecast_mwh, net_load_kw, price_only: bool) -> float:
    """Run simulate_mpc for one window, return total P&L ($)."""
    net_load = None if price_only else net_load_kw
    result = simulate_mpc(
        forecast_prices=forecast_mwh,
        actual_prices=actual_prices_mwh,
        net_load_actuals=net_load,
        interval_h=INTERVAL_H,
    )
    return result["total_pnl"]


def classify_window(prices_mwh: np.ndarray) -> str:
    if np.any(prices_mwh >= SPIKE_THRESH):
        return "spike"
    if np.any(prices_mwh <= LOW_THRESH):
        return "low"
    return "normal"


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-only", action="store_true",
                        help="Skip load/PV data (price arbitrage only)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Max windows per stratum (default: all)")
    args = parser.parse_args()

    if not INDEX_FILE.exists():
        print(f"ERROR: eval index not found at {INDEX_FILE}")
        print("Run: python eval/build_holistic_eval_set.py")
        sys.exit(1)

    config = load_config()
    df_index = pd.read_parquet(INDEX_FILE)
    df_index["start_time"] = pd.to_datetime(df_index["start_time"], utc=True)

    print(f"Eval index: {len(df_index):,} windows")
    print(df_index["stratum"].value_counts().to_string())

    lgbm_forecasts = load_lgbm_forecasts()

    client = InfluxDBClient(**config["influxdb"])
    rows = []
    strata = ["spike", "low", "normal"]
    t0 = time.time()

    try:
        for stratum in strata:
            subset = df_index[df_index["stratum"] == stratum]
            if args.max_windows:
                subset = subset.head(args.max_windows)

            print(f"\n--- {stratum.upper()} ({len(subset)} windows) ---")
            for i, row in enumerate(subset.itertuples()):
                start_ts = row.start_time

                # Fetch actual data
                data = fetch_window_data(client, start_ts)
                if data is None:
                    print(f"  [{i+1}/{len(subset)}] SKIP: incomplete InfluxDB data at {start_ts}")
                    continue

                actual_prices = data["actual_prices_mwh"]
                net_load_kw   = data["net_load_kw"]

                # Oracle: perfect foresight
                oracle_pnl = run_window(actual_prices, actual_prices, net_load_kw, args.price_only)

                # Legacy LightGBM
                lgbm_fcst = lgbm_forecasts.get(start_ts.floor("30min"))
                if lgbm_fcst is None:
                    print(f"  [{i+1}/{len(subset)}] SKIP: no LightGBM forecast for {start_ts}")
                    continue
                lgbm_pnl = run_window(actual_prices, lgbm_fcst, net_load_kw, args.price_only)

                # P5MIN naive: hold last actual price constant for all steps
                p5min_price = actual_prices[0]
                p5min_fcst = np.full(WINDOW_STEPS, p5min_price)
                p5min_pnl = run_window(actual_prices, p5min_fcst, net_load_kw, args.price_only)

                rows.append({
                    "start_time":  start_ts,
                    "stratum":     stratum,
                    "oracle_pnl":  oracle_pnl,
                    "lgbm_pnl":    lgbm_pnl,
                    "p5min_pnl":   p5min_pnl,
                })

                if (i + 1) % 20 == 0:
                    elapsed = time.time() - t0
                    print(f"  {i+1}/{len(subset)} windows  ({elapsed:.0f}s elapsed)")

    finally:
        client.close()

    if not rows:
        print("ERROR: no results collected.")
        sys.exit(1)

    df_raw = pd.DataFrame(rows)

    # Convert per-window P&L to $/day (window = 72h = 3 days)
    hours_per_window = WINDOW_STEPS * INTERVAL_H
    days_per_window  = hours_per_window / 24
    for col in ("oracle_pnl", "lgbm_pnl", "p5min_pnl"):
        df_raw[col + "_per_day"] = df_raw[col] / days_per_window

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Holistic Dispatch Simulation — Results")
    print(f"Mode: {'price-only' if args.price_only else 'price + load/PV'}")
    print("=" * 70)

    summary_rows = []
    all_strata = strata + ["all"]
    for s in all_strata:
        mask = df_raw["stratum"] == s if s != "all" else pd.Series(True, index=df_raw.index)
        sub = df_raw[mask]
        if sub.empty:
            continue
        for source, col in [("oracle", "oracle_pnl_per_day"),
                             ("lgbm_legacy", "lgbm_pnl_per_day"),
                             ("p5min_naive", "p5min_pnl_per_day")]:
            mean_val = sub[col].mean()
            vs_lgbm  = (mean_val / max(abs(sub["lgbm_pnl_per_day"].mean()), 1e-9) - 1) * 100
            summary_rows.append({
                "stratum": s, "source": source, "n": len(sub),
                "mean_per_day": round(mean_val, 4),
                "vs_lgbm_pct": round(vs_lgbm, 1) if source != "lgbm_legacy" else 0.0,
            })

    df_summary = pd.DataFrame(summary_rows)
    for s in all_strata:
        sub = df_summary[df_summary["stratum"] == s]
        if sub.empty:
            continue
        print(f"\n  [{s.upper():8s}]  n={sub['n'].iloc[0]}")
        print(f"  {'Source':<14}  {'Mean $/day':>12}  {'vs LightGBM':>12}")
        print(f"  {'-'*42}")
        for _, row in sub.iterrows():
            vs = f"{row['vs_lgbm_pct']:+.1f}%" if row["source"] != "lgbm_legacy" else "baseline"
            print(f"  {row['source']:<14}  {row['mean_per_day']:>12.4f}  {vs:>12}")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_DIR / "holistic_eval_results.csv"
    parquet_out = RESULTS_DIR / "holistic_eval_raw.parquet"
    df_summary.to_csv(csv_out, index=False)
    df_raw.to_parquet(parquet_out, index=False)
    print(f"\nSaved → {csv_out.relative_to(ROOT)}")
    print(f"Saved → {parquet_out.relative_to(ROOT)}")
    elapsed = time.time() - t0
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
