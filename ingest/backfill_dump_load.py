"""
Backfill estimated dump load power into rp_5m and rp_30m based on historical
switch states for snf15/snf18/snf19 in rp_raw.

Each heater is assumed to draw 2000W when its switch is on. The switch value
in InfluxDB is 0 (off) or 1 (on); the mean over an interval reflects duty
cycle when the thermostat cuts in/out.

snf19 was the original second heater, replaced by snf18. Both are included so
that the historical contribution of whichever was active at any given time is
captured. Since the failed snf19 would have been off, it contributes 0W.

Run once after creating the CQs. The CQs will handle ongoing data.
"""

import datetime
import pytz
import pandas as pd
from influxdb import InfluxDBClient

INFLUXDB_HOST = 'localhost'
INFLUXDB_PORT = 8086
INFLUXDB_USERNAME = 'user'
INFLUXDB_PASSWORD = 'm2gxrEJMX3'
INFLUXDB_DATABASE = 'hass'

HEATER_SWITCHES = ['snf15_snf15', 'snf18_snf18', 'snf19_snf19']
WATTS_PER_HEATER = 2000
ENTITY_ID = 'estimated_dump_load_power'
BACKFILL_DAYS = 30


def query_switch_5m(client, entity_id, start, end):
    """Query 5-minute mean/min/max for a switch entity from rp_raw."""
    start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end.strftime('%Y-%m-%dT%H:%M:%SZ')
    q = (
        f'SELECT mean(value) AS mean_val, min(value) AS min_val, max(value) AS max_val '
        f'FROM "rp_raw"."switch" '
        f'WHERE entity_id = \'{entity_id}\' '
        f'AND time >= \'{start_str}\' AND time < \'{end_str}\' '
        f'GROUP BY time(5m) fill(0)'
    )
    result = client.query(q)
    points = list(result.get_points())
    if not points:
        return pd.DataFrame(columns=['time', 'mean_val', 'min_val', 'max_val'])
    df = pd.DataFrame(points)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.set_index('time')
    return df


def main():
    client = InfluxDBClient(
        host=INFLUXDB_HOST,
        port=INFLUXDB_PORT,
        username=INFLUXDB_USERNAME,
        password=INFLUXDB_PASSWORD,
        database=INFLUXDB_DATABASE,
    )

    now = datetime.datetime.now(pytz.UTC)
    start = (now - datetime.timedelta(days=BACKFILL_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    # Truncate end to the last complete 5m interval
    end = now.replace(second=0, microsecond=0)
    end = end.replace(minute=(end.minute // 5) * 5)

    print(f"Backfilling dump load from {start} to {end}")

    # Build a full 5-minute time index for the range
    index = pd.date_range(start=start, end=end, freq='5min', tz='UTC', inclusive='left')
    total = pd.DataFrame(
        {'mean_val': 0.0, 'min_val': 0.0, 'max_val': 0.0},
        index=index,
    )

    for switch in HEATER_SWITCHES:
        print(f"  Querying {switch}...")
        df = query_switch_5m(client, switch, start, end)
        if df.empty:
            print(f"    No data for {switch}, skipping")
            continue
        df = df.reindex(index, fill_value=0.0)
        total['mean_val'] += df['mean_val'].fillna(0.0) * WATTS_PER_HEATER
        total['min_val'] += df['min_val'].fillna(0.0) * WATTS_PER_HEATER
        total['max_val'] += df['max_val'].fillna(0.0) * WATTS_PER_HEATER

    # Drop intervals where all switches were off (no dump load activity)
    active = total[total['mean_val'] > 0]
    print(f"Found {len(active)} 5-minute intervals with dump load activity")

    # --- Write to rp_5m ---
    points_5m = []
    for ts, row in total.iterrows():
        points_5m.append({
            'measurement': 'power_dump_load_5m',
            'tags': {'entity_id': ENTITY_ID},
            'time': ts.isoformat(),
            'fields': {
                'mean_value': float(row['mean_val']),
                'min_value': float(row['min_val']),
                'max_value': float(row['max_val']),
            },
        })

    print(f"Writing {len(points_5m)} points to rp_5m.power_dump_load_5m...")
    client.write_points(points_5m, retention_policy='rp_5m', batch_size=500)

    # --- Aggregate to 30m and write to rp_30m ---
    total_30m = total.resample('30min').agg({
        'mean_val': 'mean',
        'min_val': 'min',
        'max_val': 'max',
    })

    points_30m = []
    for ts, row in total_30m.iterrows():
        points_30m.append({
            'measurement': 'power_dump_load_30m',
            'tags': {'entity_id': ENTITY_ID},
            'time': ts.isoformat(),
            'fields': {
                'mean_value': float(row['mean_val']),
                'min_value': float(row['min_val']),
                'max_value': float(row['max_val']),
            },
        })

    print(f"Writing {len(points_30m)} points to rp_30m.power_dump_load_30m...")
    client.write_points(points_30m, retention_policy='rp_30m', batch_size=500)

    print("Backfill complete.")
    client.close()


if __name__ == '__main__':
    main()
