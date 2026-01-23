
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
TARGET_ENTITY_ID = 'sensor.solcast_pv_forecast_power_now'

def replace_solcast_pv_30m():
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"Error: SQLite database not found at '{SQLITE_DB_PATH}'")
        return

    # 1. Connect to HA DB
    conn = sqlite3.connect(SQLITE_DB_PATH)
    
    # Range: 2025-06-19 to present
    start_date = datetime.datetime(2025, 6, 19, tzinfo=datetime.timezone.utc)
    start_ts = start_date.timestamp()

    print(f"Fetching hourly statistics since {start_date}")
    
    # Query 'statistics' (Hourly data)
    query = f"SELECT start_ts, mean FROM statistics WHERE metadata_id = {TARGET_METADATA_ID} AND start_ts >= {start_ts} ORDER BY start_ts ASC"
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("No hourly data found in 'statistics' table.")
        conn.close()
        return

    print(f"Found {len(df)} hourly records.")
    
    # Process data
    df['time'] = pd.to_datetime(df['start_ts'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    
    # Upsample to 30m
    # We use ffill() because an hourly mean value applies reasonably to both 30m slots.
    df_30m = df.resample('30min').ffill()
    
    print(f"Upsampled to {len(df_30m)} 30-minute records.")
    
    # 2. Connect to InfluxDB
    client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD, database=INFLUXDB_DATABASE)
    
    # 3. DELETE all existing data in the 30m measurement for the target range
    # In InfluxDB 1.x, DELETE is measurement-wide (or filtered by tags/time)
    # We will target the 30m measurement specifically.
    delete_query = f"DELETE FROM \"power_pv_30m\" WHERE time >= '{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
    print(f"Wiping existing 30m data: {delete_query}")
    client.query(delete_query, params={'rp': 'rp_30m'})
    
    # 4. Write new data
    points = []
    for timestamp, row in df_30m.iterrows():
        if pd.isna(row['mean']): continue
        
        point = {
            "measurement": "power_pv_30m",
            "tags": {
                "entity_id": TARGET_ENTITY_ID,
                "source": "replacement_script"
            },
            "time": timestamp.isoformat(),
            "fields": {
                "mean_value": float(row['mean'])
            }
        }
        points.append(point)
        
        if len(points) >= 1000:
            client.write_points(points, retention_policy='rp_30m')
            points = []
            
    if points:
        client.write_points(points, retention_policy='rp_30m')
        
    print("Replacement of 30m data complete.")
    client.close()
    conn.close()

if __name__ == "__main__":
    replace_solcast_pv_30m()
