import pandas as pd
from influxdb import InfluxDBClient
import pytz
import logging
import os

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
INFLUXDB_HOST = 'REDACTED'
INFLUXDB_PORT = 8086
INFLUXDB_USERNAME = 'user'
INFLUXDB_PASSWORD = 'REDACTED' # Please ensure this is the correct password
INFLUXDB_DATABASE = 'hass'
NEM_TIMEZONE = pytz.timezone('Australia/Brisbane')

# --- List of CSV files to process ---
# Place these files in the same directory as the script, or provide the full path.
CSV_FILES_TO_PROCESS = [
    "PRICE_AND_DEMAND_202507_SA1.csv",
    "PRICE_AND_DEMAND_202507_NSW1.csv",
    "PRICE_AND_DEMAND_202507_VIC1.csv"
]

def ingest_csv_data():
    """
    Reads, processes, and ingests AEMO data from local CSV files into InfluxDB.
    """
    client = None
    try:
        # --- Connect to InfluxDB ---
        client = InfluxDBClient(
            host=INFLUXDB_HOST, port=INFLUXDB_PORT,
            username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD,
            database=INFLUXDB_DATABASE
        )
        print(f"Connected to InfluxDB: {INFLUXDB_HOST}:{INFLUXDB_PORT}")

        # --- Process each CSV file ---
        for csv_file in CSV_FILES_TO_PROCESS:
            if not os.path.exists(csv_file):
                print(f"\n--- File not found: '{csv_file}'. Skipping. ---")
                continue

            print(f"\n--- Starting to process file: '{csv_file}' ---")
            
            # --- Read and Clean CSV Data ---
            df = pd.read_csv(csv_file)
            # Rename columns to match the desired schema.
            df.rename(columns={
                'REGION': 'region',
                'SETTLEMENTDATE': 'settlementdate',
                'RRP': 'price',
                'TOTALDEMAND': 'total_demand'
            }, inplace=True)

            # Convert settlementdate to datetime objects
            # The original script adjusted timestamps by -5min. We replicate that here for consistency.
            df['settlementdate'] = pd.to_datetime(df['settlementdate']) - pd.Timedelta('5min')

            # Extract region for this file (e.g., 'SA1')
            region = df['region'].iloc[0]
            print(f"  Detected region: {region}")

            # --- JOB 1: Ingest Recent 5-minute Data ---
            try:
                print("  Processing Job: 5-minute raw data")
                measurement_5m = f"aemo_dispatch_{region.lower()}_5m"
                rp_5m = "rp_5m"
                
                df_5m = df[['settlementdate', 'price', 'total_demand']].copy()
                df_5m['TIMESTAMP_UTC'] = (df_5m['settlementdate']
                                            .dt.tz_localize(NEM_TIMEZONE)
                                            .dt.tz_convert('UTC'))
                
                points_5m = []
                for _, row in df_5m.iterrows():
                    points_5m.append({
                        "measurement": measurement_5m,
                        "fields": {
                            "price": float(row['price']),
                            "total_demand": float(row['total_demand'])
                            # net_interchange is omitted as it's not in the CSV
                        },
                        "time": row['TIMESTAMP_UTC']
                    })
                
                if points_5m:
                    print(f"    Attempting to write {len(points_5m)} points to measurement '{measurement_5m}'...")
                    client.write_points(points_5m, time_precision='s', retention_policy=rp_5m, batch_size=20000)
                    print(f"    Successfully ingested {len(points_5m)} points for {region} (5-minute).")
                else:
                    print("    No 5-minute points generated. Nothing to write.")

            except Exception as e:
                print(f"    An error occurred during the 5-minute job for {csv_file}: {e}")

            # --- JOB 2: Ingest Downsampled 30-minute Averages ---
            try:
                print("  Processing Job: 30-minute downsampled averages")
                measurement_30m = f"aemo_dispatch_{region.lower()}_30m"
                rp_30m = "rp_30m"

                # Set index for resampling
                df_resample_source = df.set_index('settlementdate')

                # Perform downsampling
                df_30m = df_resample_source[['price', 'total_demand']].resample('30min').mean()
                df_30m.dropna(inplace=True) # Remove intervals with no data
                df_30m.reset_index(inplace=True)
                
                print(f"    {len(df)} raw records downsampled to {len(df_30m)} 30-minute records.")

                df_30m['TIMESTAMP_UTC'] = (df_30m['settlementdate']
                                            .dt.tz_localize(NEM_TIMEZONE)
                                            .dt.tz_convert('UTC'))
                
                points_30m = []
                for _, row in df_30m.iterrows():
                    points_30m.append({
                        "measurement": measurement_30m,
                        "fields": {
                            "price": float(row['price']),
                            "total_demand": float(row['total_demand'])
                        },
                        "time": row['TIMESTAMP_UTC']
                    })
                
                if points_30m:
                    print(f"    Attempting to write {len(points_30m)} points to measurement '{measurement_30m}'...")
                    client.write_points(points_30m, time_precision='s', retention_policy=rp_30m, batch_size=20000)
                    print(f"    Successfully ingested {len(points_30m)} points for {region} (30-minute).")
                else:
                    print("    No 30-minute points generated. Nothing to write.")

            except Exception as e:
                print(f"    An error occurred during the 30-minute job for {csv_file}: {e}")


    except Exception as e:
        print(f"A script-level error occurred: {e}")
    finally:
        if client:
            client.close()
            print("\nInfluxDB connection closed.")

if __name__ == "__main__":
    print("Starting AEMO data ingestion from local CSV files...")
    ingest_csv_data()
    print("Script finished.")
