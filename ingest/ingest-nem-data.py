import pandas as pd
from influxdb import InfluxDBClient
import datetime
from nemosis import dynamic_data_compiler
import pytz
import logging
import os

# --- Basic Configuration ---
logging.getLogger("nemosis").setLevel(logging.WARNING)
INFLUXDB_HOST = 'REDACTED'
INFLUXDB_PORT = 8086
INFLUXDB_USERNAME = 'user'
INFLUXDB_PASSWORD = 'REDACTED'
INFLUXDB_DATABASE = 'hass'
RAW_DATA_CACHE_PATH = "./nemosis_cache"
NEM_TIMEZONE = pytz.timezone('Australia/Brisbane')

# --- Simplified Data Configuration ---
AEMO_CONFIG = {
    "name": "AEMO Dispatch Data",
    "base_measurement": "aemo_dispatch",
    "regions": ['SA1', 'VIC1', 'NSW1'],
    "source_tables": {
        "price": "DISPATCHPRICE",
        "summary": "DISPATCHREGIONSUM"
    },
    # Use lowercase names, as we will rename columns immediately after merge
    "agg_cols": ["price", "total_demand", "net_interchange"]
}

# --- Ingestion Jobs Configuration ---
INGESTION_JOBS = [
    {
        "name": "Historical 30-minute Averages",
        "start_date": datetime.date(2022, 1, 1),
        "end_date": datetime.date.today() + datetime.timedelta(days=1),
        "retention_policy": "rp_30m",
        "measurement_suffix": "_30m",
        "downsample": True,
        "downsample_freq": "30min"
    },
    {
        "name": "Recent 5-minute Data",
        "start_date": datetime.date(2024, 7, 13),
        "end_date": datetime.date.today() + datetime.timedelta(days=1),
        "retention_policy": "rp_5m",
        "measurement_suffix": "_5m",
        "downsample": False
    }
]

def fetch_aemo_data(table_name, start_str, end_str, regions):
    """Fetches and pre-processes data from a single AEMO table."""
    print(f"    Fetching data for table '{table_name}'...")
    df = dynamic_data_compiler(
        start_time=start_str,
        end_time=end_str,
        table_name=table_name,
        raw_data_location=RAW_DATA_CACHE_PATH,
        filter_cols=["REGIONID"],
        filter_values=[regions]
    )
    if not df.empty:
        df['SETTLEMENTDATE'] = df['SETTLEMENTDATE'] - pd.Timedelta('5min')
    return df

def ingest_aemo_data():
    """
    Fetches, MERGES, processes, and ingests historical AEMO data into separate measurements by region.
    """
    client = None
    try:
        client = InfluxDBClient(
            host=INFLUXDB_HOST, port=INFLUXDB_PORT,
            username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD,
            database=INFLUXDB_DATABASE
        )
        print(f"Connected to InfluxDB: {INFLUXDB_HOST}:{INFLUXDB_PORT}")

        for job in INGESTION_JOBS:
            print(f"\n--- Starting Job: {job['name']} ---")
            print(f"  Range: {job['start_date']} to {job['end_date']-datetime.timedelta(days=1)}")

            if job['start_date'] >= job['end_date']:
                print("  Job skipped (start date is on or after end date).")
                continue
                
            try:
                start_str = job['start_date'].strftime("%Y/%m/%d %H:%M:%S")
                end_str = job['end_date'].strftime("%Y/%m/%d %H:%M:%S")

                df_price = fetch_aemo_data(AEMO_CONFIG['source_tables']['price'], start_str, end_str, AEMO_CONFIG['regions'])
                df_summary = fetch_aemo_data(AEMO_CONFIG['source_tables']['summary'], start_str, end_str, AEMO_CONFIG['regions'])
                
                if df_price.empty or df_summary.empty:
                    print("    One or more source tables returned no data. Cannot merge. Skipping.")
                    continue

                print("    Merging price and summary data...")
                df_price.rename(columns={'RRP': 'price'}, inplace=True)
                df_merged = pd.merge(
                    df_price[['SETTLEMENTDATE', 'REGIONID', 'price']],
                    df_summary[['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND', 'NETINTERCHANGE']],
                    on=['SETTLEMENTDATE', 'REGIONID']
                )

                # Rename columns to their final, lowercase form immediately after merging.
                df_merged.rename(columns={
                    'REGIONID': 'region',
                    'TOTALDEMAND': 'total_demand',
                    'NETINTERCHANGE': 'net_interchange'
                }, inplace=True)
                print(f"    Fetched, merged, and cleaned {len(df_merged)} raw records.")
                
                df_to_process = df_merged

                if job['downsample']:
                    print(f"    Downsampling to {job['downsample_freq']} averages...")
                    # This now works because the column names in df_merged match the names in 'agg_cols'.
                    df_resampled = (df_merged.set_index('SETTLEMENTDATE')
                                      .groupby('region')[AEMO_CONFIG['agg_cols']]
                                      .resample(job['downsample_freq'])
                                      .mean())
                    df_resampled.dropna(inplace=True)
                    df_to_process = df_resampled.reset_index()
                    print(f"    {len(df_to_process)} records remaining after downsampling.")

                df_to_process['TIMESTAMP_UTC'] = (df_to_process['SETTLEMENTDATE']
                                                      .dt.tz_localize(NEM_TIMEZONE)
                                                      .dt.tz_convert('UTC'))
                
                # Iterate through each region and write to a separate measurement
                for region in AEMO_CONFIG['regions']:
                    print(f"    Processing data for region: {region}")
                    df_region = df_to_process[df_to_process['region'] == region]
                    
                    if df_region.empty:
                        print(f"      No data for region {region}. Skipping.")
                        continue

                    points = []
                    # Construct measurement name dynamically per region, e.g., aemo_dispatch_sa1_30m
                    final_measurement_name = f"{AEMO_CONFIG['base_measurement']}_{region.lower()}{job['measurement_suffix']}"
                    
                    for _, row in df_region.iterrows():
                        point = {
                            "measurement": final_measurement_name,
                            "fields": {
                                "price": float(row['price']),
                                "total_demand": float(row['total_demand']),
                                "net_interchange": float(row['net_interchange'])
                            },
                            "time": row['TIMESTAMP_UTC']
                        }
                        points.append(point)

                    if not points:
                        print(f"      No points generated for region {region}. Nothing to write.")
                        continue

                    print(f"      Attempting to write {len(points)} points to measurement '{final_measurement_name}'...")
                    client.write_points(points, time_precision='s', retention_policy=job['retention_policy'], batch_size=20000)
                    print(f"      Successfully ingested {len(points)} points for {region}.")

            except Exception as e:
                print(f"    An error occurred during job '{job['name']}': {e}")
                import traceback
                traceback.print_exc()
                continue

    except Exception as e:
        print(f"An unexpected script-level error occurred: {e}")
    finally:
        if client:
            client.close()
            print("\nInfluxDB connection closed.")

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_CACHE_PATH):
        os.makedirs(RAW_DATA_CACHE_PATH)
    print("Starting unified historical AEMO data ingestion script...")
    ingest_aemo_data()
    print("Script finished.")
