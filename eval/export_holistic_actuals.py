#!/usr/bin/env python3
"""
Freeze InfluxDB actuals used by holistic_eval.py to a parquet snapshot.

Queries price, load, and PV for the full eval period once, then saves
them to eval/results/holistic_eval_actuals.parquet. Subsequent holistic
eval runs load from this frozen snapshot instead of InfluxDB, making
results reproducible regardless of InfluxDB state.

Usage:
    python eval/export_holistic_actuals.py            # freeze current data
    python eval/export_holistic_actuals.py --refresh  # re-query and overwrite

The output parquet has one row per 30-min interval in the eval period:
    time (UTC timestamp), price_mwh, load_kw, pv_kw

See holistic_eval.py for the eval period definition (eval index start/end).
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
INDEX_FILE  = RESULTS_DIR / "holistic_eval_index.parquet"
OUT_FILE    = RESULTS_DIR / "holistic_eval_actuals.parquet"

sys.path.insert(0, str(ROOT))


def query_bulk(client, measurement, field, start_iso, end_iso) -> pd.Series:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="Re-query InfluxDB and overwrite existing snapshot")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.refresh:
        print(f"Snapshot already exists: {OUT_FILE}")
        print("Use --refresh to re-query InfluxDB.")
        df = pd.read_parquet(OUT_FILE)
        print(f"  {len(df):,} rows, {df['time'].min()} → {df['time'].max()}")
        return

    if not INDEX_FILE.exists():
        print(f"ERROR: eval index not found at {INDEX_FILE}")
        print("Run eval/build_holistic_eval_set.py first.")
        sys.exit(1)

    df_index = pd.read_parquet(INDEX_FILE)
    start_ts = pd.to_datetime(df_index["start_time"]).min().tz_localize("UTC") if df_index["start_time"].dt.tz is None else df_index["start_time"].min()
    # Eval windows are 72h; add buffer to cover the last window
    end_ts   = pd.to_datetime(df_index["start_time"]).max() + pd.Timedelta(hours=73)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    start_iso = start_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"Eval period: {start_iso} → {end_iso}")

    from config_utils import load_config
    config = load_config(ROOT / "config.yaml")

    from influxdb import InfluxDBClient
    client = InfluxDBClient(**config["influxdb"])

    try:
        t0 = time.time()
        print("Querying InfluxDB (price / load / PV)...")
        prices = query_bulk(client, "aemo_dispatch_sa1_30m", "price", start_iso, end_iso)
        load   = query_bulk(client, "power_load_30m", "mean_value", start_iso, end_iso)
        pv     = query_bulk(client, "power_pv_30m",   "mean_value", start_iso, end_iso)
        print(f"  price: {len(prices):,}  load: {len(load):,}  PV: {len(pv):,}  ({time.time()-t0:.1f}s)")
    finally:
        client.close()

    # Align on a common UTC 30-min grid (union of all available timestamps)
    all_times = prices.index.union(load.index).union(pv.index)
    df = pd.DataFrame({
        "time":      all_times,
        "price_mwh": prices.reindex(all_times).values,
        "load_kw":   (load / 1000.0).reindex(all_times).values,
        "pv_kw":     (pv  / 1000.0).reindex(all_times).values,
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    print(f"Saved → {OUT_FILE}")
    print(f"  {len(df):,} rows, price coverage: {df['price_mwh'].notna().sum():,} / {len(df):,}")


if __name__ == "__main__":
    main()
