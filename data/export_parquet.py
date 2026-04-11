#!/usr/bin/env python3
"""
Export AEMO and local sensor data from InfluxDB to Parquet files.

This is the ML cache layer: InfluxDB stays as the ingest/timeseries store;
Parquet is rebuilt from InfluxDB on demand. Re-run to refresh.

Outputs (data/parquet/):
  aemo_predispatch_sa1.parquet  — interval_dt, run_time, rrp, total_demand, net_interchange
  aemo_pd7day_sa1.parquet       — interval_dt, run_time, rrp
  aemo_sevendayoutlook_sa1.parquet — interval_dt, run_time, scheduled_demand, net_interchange
  actuals_sa1.parquet           — time, rrp, total_demand, net_interchange, power_load, power_pv,
                                   temp, humidity, wind_speed

Key design decision:
  Queries do NOT use GROUP BY run_time — that's what caused the 8.5h hang
  (17,814 unique tag values → InfluxDB loads everything into memory as series).
  Instead: SELECT * WHERE region='SA1' → run_time arrives as a plain column.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION = "SA1"
RP = "rp_30m"


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


def query_to_df(client, query, time_col="time"):
    """Execute an InfluxDB query and return a DataFrame.
    Uses a simple (non-chunked) query. For large measurements we use
    time-batched helpers instead (see query_batched).
    """
    print(f"  Query: {query[:120]}{'...' if len(query) > 120 else ''}")
    result = client.query(query)
    pts = list(result.get_points())
    if not pts:
        print("  WARNING: query returned 0 rows")
        return pd.DataFrame()
    df = pd.DataFrame(pts)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    print(f"  → {len(df):,} rows")
    return df


def query_batched(client, measurement, fields, where_extra="",
                  start_time=None, batch_months=2, time_col="time"):
    """
    Fetch a large measurement in time batches to avoid HTTP response size limits.
    Returns a concatenated DataFrame.

    where_extra: additional WHERE clauses (e.g. "region='SA1'"). Applied in every batch.
    start_time: ISO string override for start. If None, queries FIRST() from measurement.
    """
    field_str = ", ".join(fields)

    def build_where(*clauses):
        parts = [c for c in clauses if c]
        return "WHERE " + " AND ".join(parts) if parts else ""

    # Get time range
    if start_time:
        t_start = pd.to_datetime(start_time, utc=True)
    else:
        where_range = build_where(where_extra)
        r = client.query(f"SELECT FIRST({fields[0]}) FROM {measurement} {where_range}")
        first_pts = list(r.get_points())
        if not first_pts:
            print("  WARNING: measurement appears empty")
            return pd.DataFrame()
        t_start = pd.to_datetime(first_pts[0]["time"], utc=True)

    where_range = build_where(where_extra)
    r2 = client.query(f"SELECT LAST({fields[0]}) FROM {measurement} {where_range}")
    last_pts = list(r2.get_points())
    if not last_pts:
        print("  WARNING: measurement appears empty")
        return pd.DataFrame()

    t_end = pd.to_datetime(last_pts[0]["time"], utc=True) + pd.Timedelta(seconds=1)

    frames = []
    t_lo = t_start
    total = 0
    while t_lo < t_end:
        t_hi = t_lo + pd.DateOffset(months=batch_months)
        if t_hi > t_end:
            t_hi = t_end

        lo_str = t_lo.strftime("%Y-%m-%dT%H:%M:%SZ")
        hi_str = t_hi.strftime("%Y-%m-%dT%H:%M:%SZ")
        time_clause = f"time >= '{lo_str}' AND time < '{hi_str}'"
        where_clause = build_where(where_extra, time_clause)
        q = f"SELECT {field_str} FROM {measurement} {where_clause}"

        result = client.query(q)
        pts = list(result.get_points())
        if pts:
            chunk = pd.DataFrame(pts)
            total += len(chunk)
            frames.append(chunk)
            print(f"    {lo_str[:10]} → {hi_str[:10]}: {len(chunk):,} rows (cumulative: {total:,})")

        t_lo = t_hi

    if not frames:
        print("  WARNING: query returned 0 rows")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    print(f"  → {len(df):,} rows total")
    return df


def export_predispatch(client):
    print("\n[1/4] Exporting PREDISPATCH SA1...")
    # ~990K rows — use batched query to avoid HTTP response size limits
    df = query_batched(
        client,
        measurement=f"{RP}.aemo_predispatch_forecast",
        fields=["rrp", "total_demand", "net_interchange", "run_time"],
        where_extra=f"region='{REGION}'",
    )
    if df.empty:
        print("  SKIP: no data")
        return

    df = df.rename(columns={"time": "interval_dt"})
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df = df[["interval_dt", "run_time", "rrp", "total_demand", "net_interchange"]].copy()
    df = df.sort_values(["run_time", "interval_dt"]).reset_index(drop=True)

    out = OUT_DIR / "aemo_predispatch_sa1.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  Written: {out} ({size_mb:.1f}MB)")
    print(f"  run_time range: {df.run_time.min()} → {df.run_time.max()}")
    print(f"  Unique run_times: {df.run_time.nunique():,}")
    return df


def export_pd7day(client):
    print("\n[2/4] Exporting PD7Day SA1...")
    # ~64K rows — small enough for direct query
    df = query_to_df(
        client,
        f"SELECT rrp, run_time FROM {RP}.aemo_pd7day_forecast WHERE region='{REGION}'"
    )
    if df.empty:
        print("  SKIP: no data")
        return

    df = df.rename(columns={"time": "interval_dt"})
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df = df[["interval_dt", "run_time", "rrp"]].copy()
    df = df.sort_values(["run_time", "interval_dt"]).reset_index(drop=True)

    out = OUT_DIR / "aemo_pd7day_sa1.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    size_kb = out.stat().st_size / 1024
    print(f"  Written: {out} ({size_kb:.0f}KB)")
    print(f"  run_time range: {df.run_time.min()} → {df.run_time.max()}")
    print(f"  Unique run_times: {df.run_time.nunique():,}")
    return df


def export_sevendayoutlook(client):
    print("\n[3/4] Exporting SevenDayOutlook SA1...")
    # Batched: ~13 months of 30-min data with multiple runs per interval
    df = query_batched(
        client,
        measurement=f"{RP}.aemo_sevendayoutlook",
        fields=["scheduled_demand", "net_interchange", "run_time"],
        where_extra=f"region='{REGION}'",
    )
    if df.empty:
        print("  SKIP: no data")
        return

    df = df.rename(columns={"time": "interval_dt"})
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df = df[["interval_dt", "run_time", "scheduled_demand", "net_interchange"]].copy()
    df = df.sort_values(["run_time", "interval_dt"]).reset_index(drop=True)

    out = OUT_DIR / "aemo_sevendayoutlook_sa1.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  Written: {out} ({size_mb:.1f}MB)")
    return df


def export_actuals(client):
    """
    Join dispatch actuals + local sensors into one actuals DataFrame.
    All indexed by 30-min interval timestamp (UTC).
    """
    print("\n[4/4] Exporting actuals SA1...")

    # 1. AEMO dispatch actuals (price, demand, interchange)
    # Start 2 days before first PREDISPATCH run (needed for encoder context)
    ACTUALS_START = "2024-03-29T00:00:00Z"
    print("  Fetching dispatch actuals...")
    dispatch = query_batched(
        client,
        measurement=f"{RP}.aemo_dispatch_sa1_30m",
        fields=["price", "total_demand", "net_interchange"],
        start_time=ACTUALS_START,
    )
    if dispatch.empty:
        print("  ERROR: no dispatch data — cannot build actuals")
        return
    dispatch = dispatch.rename(columns={
        "time": "time",
        "price": "rrp",
        "total_demand": "total_demand",
        "net_interchange": "net_interchange",
    })
    dispatch = dispatch.set_index("time")

    # 2. Local sensors (optional — fill NaN if missing)
    sensors = {
        "power_load": f"{RP}.power_load_30m",
        "power_pv":   f"{RP}.power_pv_30m",
        "temp":       f"{RP}.temperature_adelaide",
        "humidity":   f"{RP}.humidity_adelaide",
        "wind_speed": f"{RP}.wind_speed_adelaide",
    }
    for col_name, measurement in sensors.items():
        print(f"  Fetching {col_name}...")
        df_s = query_batched(
            client, measurement=measurement,
            fields=["mean_value"],
            start_time=ACTUALS_START,
        )
        if df_s.empty:
            print(f"  WARNING: no data for {col_name}")
            dispatch[col_name] = float("nan")
        else:
            df_s = df_s.rename(columns={"mean_value": col_name})
            df_s = df_s.set_index("time")[col_name]
            # Deduplicate (HA can write multiple records per interval)
            df_s = df_s.groupby(level=0).mean()
            dispatch[col_name] = df_s

    dispatch = dispatch.reset_index()
    if "time" not in dispatch.columns and dispatch.columns[0] != "time":
        dispatch = dispatch.rename(columns={dispatch.columns[0]: "time"})
    dispatch = dispatch.sort_values("time").reset_index(drop=True)

    out = OUT_DIR / "actuals_sa1.parquet"
    dispatch.to_parquet(out, index=False, compression="snappy")
    print(f"  Written: {out} ({out.stat().st_size // 1024}KB)")
    print(f"  time range: {dispatch.time.min()} → {dispatch.time.max()}")
    return dispatch


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export InfluxDB data to Parquet")
    parser.add_argument("--actuals-only", action="store_true",
                        help="Re-export only actuals_sa1.parquet (preserves backfilled PREDISPATCH)")
    args = parser.parse_args()

    print("=== Parquet export from InfluxDB ===")
    print(f"Output directory: {OUT_DIR}")

    cfg = load_config()
    client = influx_client(cfg)
    print(f"Connected to InfluxDB: {cfg['influxdb']['host']}:{cfg['influxdb'].get('port', 8086)}")

    if args.actuals_only:
        print("(--actuals-only: skipping PREDISPATCH, PD7Day, SevenDayOutlook)")
        export_actuals(client)
    else:
        export_predispatch(client)
        export_pd7day(client)
        export_sevendayoutlook(client)
        export_actuals(client)

    print("\n=== Export complete ===")
    print("Files:")
    for f in sorted(OUT_DIR.glob("*.parquet")):
        mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name}: {mb:.1f}MB")


if __name__ == "__main__":
    main()
