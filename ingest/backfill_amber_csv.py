import pandas as pd
from influxdb import InfluxDBClient
import pytz
import logging
import datetime
import json
import os

# --- Configuration ---
# Load configuration from config.json to ensure credentials match
try:
    with open('../config.json', 'r') as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    # Fallback if running from a different directory (though typically run from root or ingest)
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)

INFLUXDB_HOST = CONFIG['influxdb']['host']
INFLUXDB_PORT = CONFIG['influxdb']['port']
INFLUXDB_USERNAME = CONFIG['influxdb']['username']
INFLUXDB_PASSWORD = CONFIG['influxdb']['password']
INFLUXDB_DATABASE = CONFIG['influxdb']['database']

# Resolve CSV path relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "../amber-usage.csv") 
TARGET_MEASUREMENT = "power_load_30m"
TARGET_RP = "rp_30m"
TIMEZONE = pytz.timezone(CONFIG['timezone']) # e.g. Australia/Adelaide

# Date range to backfill (inclusive/exclusive boundaries handled by logic)
START_DATE = pd.Timestamp("2025-11-07", tz=TIMEZONE)
END_DATE = pd.Timestamp("2026-01-20 17:00:00", tz=TIMEZONE)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def backfill_data(dry_run=True):
    logging.info(f"Starting backfill process. Dry run: {dry_run}")
    
    # 1. Connect to InfluxDB
    client = InfluxDBClient(
        host=INFLUXDB_HOST, port=INFLUXDB_PORT,
        username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD,
        database=INFLUXDB_DATABASE
    )
    logging.info(f"Connected to InfluxDB: {INFLUXDB_HOST}:{INFLUXDB_PORT}")

    # 2. Read CSV
    logging.info(f"Reading CSV file: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        logging.error("CSV file not found. Make sure you are running this from the 'ingest' directory (e.g. python ingest/backfill...) or adjust path.")
        return

    # 3. Parse Dates
    # 'Start Time' format in CSV: 2025-10-22 00:30:01
    # We assume these are LOCAL TIME as per Amber app export usually.
    df['datetime'] = pd.to_datetime(df['Start Time'])
    
    # Localize to Adelaide time (assuming naive timestamps in CSV are local)
    df['datetime'] = df['datetime'].dt.tz_localize(TIMEZONE, ambiguous='infer', nonexistent='shift_forward')
    
    # 4. Filter Date Range
    mask = (df['datetime'] >= START_DATE) & (df['datetime'] < END_DATE)
    df_filtered = df.loc[mask].copy()
    
    if df_filtered.empty:
        logging.warning("No data found in the specified date range.")
        return

    logging.info(f"Found {len(df_filtered)} 5-minute records between {START_DATE} and {END_DATE}.")

    # 5. Resample to 30-minute intervals
    # We index by datetime and resample.
    # Logic: 5-min intervals. Sum 'Usage' (kWh) -> 30-min Usage (kWh).
    # Then convert 30-min Usage to Mean Power (W).
    
    df_filtered.set_index('datetime', inplace=True)
    
    # Resample logic:
    # 'Usage' is kWh within the interval starting at index.
    # We want to align to 30min buckets (00:00, 00:30, etc).
    # label='left', closed='left' is standard for "timestamp represents start of interval".
    df_30m = df_filtered['Usage'].resample('30min', label='left', closed='left').sum().to_frame()
    
    # Filter out empty bins if any (though sum() usually gives 0)
    # Actually, verify we have data coverage. If mostly 0s unexpectedly, warn.
    if df_30m['Usage'].sum() == 0:
        logging.warning("Total resampled usage is 0. Check data.")
    
    # 6. Convert to Power (Watts)
    # Energy (kWh) in 30 mins -> Power (kW) = Energy * 2
    # Power (W) = Power (kW) * 1000 = Energy * 2000
    df_30m['mean_value'] = df_30m['Usage'] * 2000.0
    
    # Remove rows where we might have had NO source data (optional, but 0 might be valid usage)
    # To be safe, we only probably want to write if we had underlying records.
    # But resample().sum() creates records for gaps with 0.
    # A cleaner approach with count to check data presence:
    df_30m_count = df_filtered['Usage'].resample('30min').count()
    df_30m = df_30m[df_30m_count > 0] # Only keep intervals where we had at least some source data
    
    logging.info(f"Resampled to {len(df_30m)} 30-minute intervals.")

    # 7. Prepare InfluxDB Points
    points = []
    for timestamp, row in df_30m.iterrows():
        # Timestamp must be UTC for InfluxDB or passed as aware datetime
        # Pandas timestamps are aware here.
        # Format:
        point = {
            "measurement": TARGET_MEASUREMENT,
            "tags": {
                "source": "amber_csv_backfill",
                "source_metadata_id": "backfill" # Using a distinct ID to be safe/traceable
            },
            "time": timestamp.isoformat(),
            "fields": {
                "mean_value": float(row['mean_value'])
            }
        }
        points.append(point)

    if dry_run:
        logging.info("DRY RUN: Sample points to be written:")
        for p in points[:5]:
            logging.info(p)
        logging.info(f"Total points to write: {len(points)}")
        logging.info("Skipping actual write.")
    else:
        logging.info(f"Writing {len(points)} points to InfluxDB (Measurement: {TARGET_MEASUREMENT}, RP: {TARGET_RP})...")
        client.write_points(points, time_precision='s', retention_policy=TARGET_RP)
        logging.info("Write complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without writing to DB")
    parser.add_argument("--execute", action="store_true", help="Execute the write")
    args = parser.parse_args()

    # Default to dry run if no flag
    if args.execute:
        backfill_data(dry_run=False)
    else:
        # Default or explicit dry-run
        backfill_data(dry_run=True)
