
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

TARGET_METADATA_ID = 449
TARGET_ENTITY_ID = 'sensor.solcast_pv_forecast_power_now'

def patch_pv_gaps():
    client = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD, database=INFLUXDB_DATABASE)
    
    # 1. Define range for today (last 24h)
    now = datetime.datetime.now(pytz.UTC)
    start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = now.replace(minute=now.minute // 30 * 30, second=0, microsecond=0)
    
    print(f"Checking for gaps between {start_date} and {end_date}")
    
    # 2. Query InfluxDB for existing points
    res = client.query(f"SELECT mean_value FROM \"rp_30m\".\"power_pv_30m\" WHERE time >= '{start_date.isoformat()}' AND time <= '{end_date.isoformat()}'")
    existing_points = {item['time'] for item in res.get_points()}
    
    # 3. Identify gaps
    expected_times = pd.date_range(start=start_date, end=end_date, freq='30min', tz='UTC')
    gaps = []
    for t in expected_times:
        t_str = t.strftime('%Y-%m-%dT%H:%M:%SZ')
        if t_str not in existing_points:
            gaps.append(t)
            
    if not gaps:
        print("No gaps found for today.")
        client.close()
        return

    print(f"Found {len(gaps)} gaps: {gaps}")
    
    # 4. Fetch data from HA statistics_short_term
    conn = sqlite3.connect(SQLITE_DB_PATH)
    
    points_to_write = []
    for gap_start in gaps:
        gap_end = gap_start + datetime.timedelta(minutes=30)
        start_ts = gap_start.timestamp()
        end_ts = gap_end.timestamp()
        
        # Query 5-min intervals for this 30-min block
        query = f"SELECT mean FROM statistics_short_term WHERE metadata_id = {TARGET_METADATA_ID} AND start_ts >= {start_ts} AND start_ts < {end_ts}"
        df_ha = pd.read_sql_query(query, conn)
        
        if not df_ha.empty:
            mean_val = df_ha['mean'].mean()
            print(f"Found data for {gap_start}: {mean_val}")
            
            point = {
                "measurement": "power_pv_30m",
                "tags": {
                    "entity_id": TARGET_ENTITY_ID,
                    "source": "gap_filler_script"
                },
                "time": gap_start.isoformat(),
                "fields": {
                    "mean_value": float(mean_val)
                }
            }
            points_to_write.append(point)
        else:
            print(f"No data in HA for gap at {gap_start}")

    # 5. Write to InfluxDB
    if points_to_write:
        print(f"Writing {len(points_to_write)} points to InfluxDB...")
        client.write_points(points_to_write, retention_policy='rp_30m')
        print("Done.")
    else:
        print("No points retrieved from HA to fill gaps.")
        
    client.close()
    conn.close()

if __name__ == "__main__":
    patch_pv_gaps()
