
import sqlite3
from influxdb import InfluxDBClient
import datetime
import os
import pytz
import pandas as pd

# --- Configuration ---
SQLITE_DB_PATH = '/opt/dockerfiles/hass/config/home-assistant_v2.db'
INFLUXDB_HOST = 'localhost'
INFLUXDB_PORT = 8086
INFLUXDB_USERNAME = 'user'
INFLUXDB_PASSWORD = 'm2gxrEJMX3'
INFLUXDB_DATABASE = 'hass'

# Metadata ID for sensor.solcast_pv_forecast_power_now in STATISTICS_META
TARGET_METADATA_ID = 449
# IMPORTANT: InfluxDB entity_id tag does NOT have 'sensor.' prefix in raw data
TARGET_ENTITY_ID = 'solcast_pv_forecast_power_now'

def backfill_solcast_hres():
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"Error: SQLite database not found at '{SQLITE_DB_PATH}'")
        return

    # 1. Connect to HA DB
    conn = sqlite3.connect(SQLITE_DB_PATH)
    
    # Range: 2026-01-23 to present
    start_date = datetime.datetime(2026, 1, 23, tzinfo=datetime.timezone.utc)
    start_ts = start_date.timestamp()

    print(f"Fetching 5-minute short-term statistics since {start_date}")
    
    # Query 'statistics_short_term' (5-minute data)
    query = f"SELECT start_ts, mean FROM statistics_short_term WHERE metadata_id = {TARGET_METADATA_ID} AND start_ts >= {start_ts} ORDER BY start_ts ASC"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No 5-minute data found in 'statistics_short_term' table.")
        conn.close()
        return

    print(f"Found {len(df)} records.")
    
    # Process data
    df['time'] = pd.to_datetime(df['start_ts'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    # 2. Connect to InfluxDB
    client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD, database=INFLUXDB_DATABASE)
    
    # 3. Write new data to rp_5m.power_pv_5m
    print("Writing 5-minute data to InfluxDB (rp_5m)...")
    points_5m = []
    for timestamp, row in df.iterrows():
        if pd.isna(row['mean']): continue
        
        point = {
            "measurement": "power_pv_5m",
            "tags": {
                "entity_id": TARGET_ENTITY_ID,
                "source": "hres_backfill_script"
            },
            "time": timestamp.isoformat(),
            "fields": {
                "mean_value": float(row['mean'])
            }
        }
        points_5m.append(point)
        
        if len(points_5m) >= 1000:
            client.write_points(points_5m, retention_policy='rp_5m')
            points_5m = []
            
    if points_5m:
        client.write_points(points_5m, retention_policy='rp_5m')

    # 4. Upsample to 30m and write to rp_30m.power_pv_30m
    print("Upsampling to 30m and writing to InfluxDB (rp_30m)...")
    df_30m = df.resample('30min').mean()
    
    points_30m = []
    for timestamp, row in df_30m.iterrows():
        if pd.isna(row['mean']): continue
        
        point = {
            "measurement": "power_pv_30m",
            "tags": {
                "entity_id": TARGET_ENTITY_ID,
                "source": "hres_backfill_script"
            },
            "time": timestamp.isoformat(),
            "fields": {
                "mean_value": float(row['mean'])
            }
        }
        points_30m.append(point)
        
        if len(points_30m) >= 1000:
            client.write_points(points_30m, retention_policy='rp_30m')
            points_30m = []
            
    if points_30m:
        client.write_points(points_30m, retention_policy='rp_30m')
        
    print("Backfill of both 5m and 30m data complete.")
    client.close()
    conn.close()

if __name__ == "__main__":
    backfill_solcast_hres()
