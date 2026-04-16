#!/usr/bin/env python3
"""
Export household load + weather data from InfluxDB to Parquet for TFT load training.

Pulls power_load, power_dump_load, power_pv, temperature, humidity, wind_speed
from InfluxDB and writes a clean parquet file with dump load subtracted.

Output: data/parquet/load_actuals_tft.parquet
  Columns: time (UTC), power_load (W, net of dump load), power_pv (W),
           temp (°C), humidity (%), wind_speed (km/h)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Start from when load sensor data begins (earlier than actuals_sa1.parquet)
LOAD_START = "2021-01-01T00:00:00Z"
RP         = "rp_30m"


def load_config(path=ROOT / "config.json"):
    with open(path) as f:
        return json.load(f)


def influx_client(cfg):
    ic = cfg["influxdb"]
    return InfluxDBClient(
        host=ic["host"], port=ic.get("port", 8086),
        username=ic["username"], password=ic["password"],
        database=ic["database"],
    )


def query_batched(client, measurement, field, start_time, batch_months=3):
    """Fetch a measurement in time batches; returns DataFrame with 'time' + field."""
    t_start = pd.to_datetime(start_time, utc=True)

    r = client.query(f"SELECT LAST({field}) FROM {measurement}")
    last_pts = list(r.get_points())
    if not last_pts:
        print(f"  WARNING: {measurement} appears empty")
        return pd.DataFrame()
    t_end = pd.to_datetime(last_pts[0]["time"], utc=True) + pd.Timedelta(seconds=1)

    frames, t_lo = [], t_start
    while t_lo < t_end:
        t_hi = min(t_lo + pd.DateOffset(months=batch_months), t_end)
        lo_str = t_lo.strftime("%Y-%m-%dT%H:%M:%SZ")
        hi_str = t_hi.strftime("%Y-%m-%dT%H:%M:%SZ")
        q = f"SELECT {field} FROM {measurement} WHERE time >= '{lo_str}' AND time < '{hi_str}'"
        pts = list(client.query(q).get_points())
        if pts:
            frames.append(pd.DataFrame(pts))
        t_lo = t_hi

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def main():
    print("=" * 60)
    print("Export: load actuals for TFT training")
    print("=" * 60)

    cfg    = load_config()
    client = influx_client(cfg)

    sensors = {
        "power_load":      (f"{RP}.power_load_30m",      "mean_value"),
        "power_dump_load": (f"{RP}.power_dump_load_30m",  "mean_value"),
        "power_pv":        (f"{RP}.power_pv_30m",         "mean_value"),
        "temp":            (f"{RP}.temperature_adelaide", "mean_value"),
        "humidity":        (f"{RP}.humidity_adelaide",    "mean_value"),
        "wind_speed":      (f"{RP}.wind_speed_adelaide",  "mean_value"),
    }

    dfs = {}
    for col, (meas, field) in sensors.items():
        print(f"\nFetching {col}...")
        df = query_batched(client, meas, field, LOAD_START)
        if df.empty:
            print(f"  WARNING: no data for {col}")
            continue
        df = df.rename(columns={field: col}).set_index("time")
        df = df.groupby(level=0)[col].mean()   # deduplicate
        dfs[col] = df
        print(f"  {len(df):,} rows  [{df.index.min()} → {df.index.max()}]")

    client.close()

    if "power_load" not in dfs:
        print("ERROR: power_load data missing — cannot build dataset")
        sys.exit(1)

    # Align to 30-min grid from first load timestamp to last
    t_min = min(s.index.min() for s in dfs.values())
    t_max = max(s.index.max() for s in dfs.values())
    idx = pd.date_range(t_min, t_max, freq="30min", tz="UTC")
    df = pd.DataFrame(index=idx)

    for col, series in dfs.items():
        df[col] = series.reindex(idx)

    # Subtract dump load from load
    dump = df.pop("power_dump_load").fillna(0.0)
    df["power_load"] = df["power_load"].fillna(0.0) - dump

    # Clip negatives (dump load occasionally over-estimated)
    df["power_load"] = df["power_load"].clip(lower=0.0)

    # Forward-fill short weather gaps (BOM drops a reading occasionally)
    for col in ["temp", "humidity", "wind_speed"]:
        if col in df.columns:
            df[col] = df[col].ffill(limit=4)

    df.index.name = "time"
    df = df.reset_index()
    df["time"] = df["time"].astype("datetime64[ns, UTC]")

    n_total     = len(df)
    n_load_nan  = df["power_load"].isna().sum()
    n_pv_nan    = df["power_pv"].isna().sum()
    n_temp_nan  = df["temp"].isna().sum()
    print(f"\nDataset summary:")
    print(f"  Rows:           {n_total:,}")
    print(f"  Time range:     {df.time.min()} → {df.time.max()}")
    print(f"  power_load NaN: {n_load_nan:,} ({n_load_nan/n_total:.1%})")
    print(f"  power_pv NaN:   {n_pv_nan:,}   ({n_pv_nan/n_total:.1%})")
    print(f"  temp NaN:       {n_temp_nan:,}   ({n_temp_nan/n_total:.1%})")

    out = OUT_DIR / "load_actuals_tft.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"\nWritten: {out}  ({out.stat().st_size / 1024:.0f}KB)")


if __name__ == "__main__":
    main()
