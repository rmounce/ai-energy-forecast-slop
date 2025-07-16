import sqlite3
from influxdb import InfluxDBClient
import datetime
import os

# --- Configuration ---
# Path to your Home Assistant SQLite database file
# IMPORTANT: Update this path to your actual database file location.
SQLITE_DB_PATH = '/opt/dockerfiles/hass/config/home-assistant_v2.db'

# InfluxDB 1.x Configuration
# IMPORTANT: Update these with your InfluxDB 1.x details.
INFLUXDB_HOST = 'REDACTED'  # e.g., 'localhost' or the IP of your InfluxDB server
INFLUXDB_PORT = 8086         # Default InfluxDB HTTP port
INFLUXDB_USERNAME = 'user' # e.g., 'admin'
INFLUXDB_PASSWORD = 'REDACTED' # e.g., 'supersecret'
INFLUXDB_DATABASE = 'hass' # e.g., 'home_assistant_data'


# --- Target Measurements for Historical Data ---
MEASUREMENT_5M_AGGREGATED = "power_pv_5m" # This will receive historical 5m data
MEASUREMENT_30M_AGGREGATED = "power_pv_30m" # This will receive historical 1h (upsampled to 30m)

# Configuration for each historical sensor data set
# Added 'upsample_to_30m' flag to trigger special handling
SENSOR_CONFIGS = [
    # Home PV Production Data
    {"table": "statistics", "metadata_id": 56, "sign_invert": False, "conversion_factor": 1, "target_measurement": MEASUREMENT_30M_AGGREGATED, "upsample_to_30m": True, "note": "Original 1-hour data (PV production), upsample to 30min, fill gaps with zeros"},
    {"table": "statistics", "metadata_id": 269, "sign_invert": False, "conversion_factor": 1, "target_measurement": MEASUREMENT_30M_AGGREGATED, "upsample_to_30m": True, "note": "Original 1-hour data (PV production), upsample to 30min, fill gaps with zeros"},
    {"table": "statistics_short_term", "metadata_id": 579, "sign_invert": False, "conversion_factor": 1000, "target_measurement": MEASUREMENT_5M_AGGREGATED, "upsample_to_30m": False, "note": "Newer 5-minute data (PV production), kW to W"}
]

def ingest_historical_data():
    """
    Connects to the Home Assistant SQLite DB, queries historical data for specified IDs,
    applies conversions (including 1h to 30m upsampling and gap filling for 1h data),
    and ingests them into their respective InfluxDB 1.x measurements with specified retention policies.
    """
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"Error: SQLite database not found at '{SQLITE_DB_PATH}'. Please update SQLITE_DB_PATH.")
        return

    sqlite_conn = None
    client = None
    try:
        sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
        sqlite_cursor = sqlite_conn.cursor()
        print(f"Successfully connected to SQLite database: {SQLITE_DB_PATH}")

        client = InfluxDBClient(
            host=INFLUXDB_HOST, port=INFLUXDB_PORT,
            username=INFLUXDB_USERNAME, password=INFLUXDB_PASSWORD,
            database=INFLUXDB_DATABASE
        )
        print(f"Connected to InfluxDB at {INFLUXDB_HOST}:{INFLUXDB_PORT}, database: {INFLUXDB_DATABASE}")

        # Ensure the database exists
        databases = client.get_list_database()
        if {'name': INFLUXDB_DATABASE} not in databases:
            print(f"InfluxDB database '{INFLUXDB_DATABASE}' does not exist. Attempting to create it.")
            client.create_database(INFLUXDB_DATABASE)
            print(f"Database '{INFLUXDB_DATABASE}' created (if user had permissions).")
        else:
            print(f"InfluxDB database '{INFLUXDB_DATABASE}' already exists.")

        # Process each sensor configuration separately to apply correct RPs
        for config in SENSOR_CONFIGS:
            metadata_id = config["metadata_id"]
            sign_invert = config["sign_invert"]
            conversion_factor = config["conversion_factor"]
            target_measurement = config["target_measurement"]
            table = config["table"]
            upsample_to_30m = config.get("upsample_to_30m", False) # Default to False
            note = config["note"]

            # Determine the retention policy based on the target measurement
            if target_measurement == MEASUREMENT_5M_AGGREGATED:
                rp_to_use = "rp_5m"
            elif target_measurement == MEASUREMENT_30M_AGGREGATED:
                rp_to_use = "rp_30m"
            else:
                rp_to_use = "autogen" # Fallback if measurement doesn't match a defined RP

            print(f"\nProcessing data for metadata_id = {metadata_id} ({note}). Target Measurement: {target_measurement}, RP: {rp_to_use}")
            query = f"SELECT start_ts, mean, min, max FROM {table} WHERE metadata_id = {metadata_id} ORDER BY start_ts ASC"
            sqlite_cursor.execute(query)
            rows = sqlite_cursor.fetchall()

            if not rows:
                print(f"No historical data found for metadata_id = {metadata_id}.")
                continue

            print(f"Found {len(rows)} records for metadata_id = {metadata_id}.")
            
            points_for_current_config = []

            if upsample_to_30m: # Special handling for 1-hour data with gap filling
                existing_data_map = {row[0]: row for row in rows} # Map timestamp (float) to full row tuple

                # Find the min and max timestamps from the *actual* data
                min_ts_actual = min(existing_data_map.keys())
                max_ts_actual = max(existing_data_map.keys())

                # Iterate through expected 1-hour intervals within the data's historical range
                # Home Assistant 'statistics' table's start_ts are typically aligned to hour boundaries.
                current_time_iter = datetime.datetime.fromtimestamp(min_ts_actual, tz=datetime.timezone.utc)
                
                processed_hours_count = 0
                # Iterate until current_time_iter goes past the last actual timestamp's hour
                while current_time_iter.timestamp() <= max_ts_actual:
                    ts_float = current_time_iter.timestamp()
                    
                    mean_val, min_val, max_val = None, None, None
                    is_gap_filled = False
                    
                    if ts_float in existing_data_map:
                        _, mean_val, min_val, max_val = existing_data_map[ts_float]
                    else:
                        # Gap detected, fill with zeros for PV production or other 1h data
                        mean_val = 0.0
                        min_val = 0.0
                        max_val = 0.0
                        is_gap_filled = True

                    # Apply conversion factor
                    mean_proc = mean_val * conversion_factor if mean_val is not None else None
                    min_proc = min_val * conversion_factor if min_val is not None else None
                    max_proc = max_val * conversion_factor if max_val is not None else None

                    # Apply sign inversion if needed (for 0, this has no effect)
                    if sign_invert:
                        mean_proc = -mean_proc if mean_proc is not None else None
                        min_proc = -min_proc if min_proc is not None else None
                        max_proc = -max_proc if max_proc is not None else None

                    fields = {}
                    if mean_proc is not None:
                        fields["mean_value"] = mean_proc
                    if min_proc is not None:
                        fields["min_value"] = min_proc
                    if max_proc is not None:
                        fields["max_value"] = max_proc
                    
                    if fields: # Only create points if there are valid fields after processing
                        # Create two 30-minute points from this 1-hour interval (whether original or gap-filled)
                        # First 30-min point
                        point1 = {
                            "measurement": target_measurement,
                            "tags": {
                                "source_metadata_id": str(metadata_id),
                                "original_resolution": "1h",
                                "gap_filled": "true" if is_gap_filled else "false" # Tag to indicate if it's a filled point
                            },
                            "time": current_time_iter,
                            "fields": fields
                        }
                        points_for_current_config.append(point1)

                        # Second 30-min point (30 minutes after the first)
                        timestamp_30m_later = current_time_iter + datetime.timedelta(minutes=30)
                        point2 = {
                            "measurement": target_measurement,
                            "tags": {
                                "source_metadata_id": str(metadata_id),
                                "original_resolution": "1h",
                                "gap_filled": "true" if is_gap_filled else "false"
                            },
                            "time": timestamp_30m_later,
                            "fields": fields
                        }
                        points_for_current_config.append(point2)
                        
                        processed_hours_count += 1
                        if processed_hours_count % 500 == 0:
                            print(f"  Processed {processed_hours_count} 1-hour intervals (generating {processed_hours_count*2} points) for ID {metadata_id}...")

                    current_time_iter += datetime.timedelta(hours=1) # Move to the next 1-hour interval
                
                # Sort all generated points by time to ensure correct ingestion order
                points_for_current_config.sort(key=lambda x: x['time'])

            else: # Standard point creation for non-upsampled data (e.g., 5-minute data)
                for row_index, row in enumerate(rows):
                    start_ts, mean_val, min_val, max_val = row

                    # Apply conversion factor
                    mean_proc = mean_val * conversion_factor if mean_val is not None else None
                    min_proc = min_val * conversion_factor if min_val is not None else None
                    max_proc = max_val * conversion_factor if max_val is not None else None

                    # Apply sign inversion if needed
                    if sign_invert:
                        mean_proc = -mean_proc if mean_proc is not None else None
                        min_proc = -min_proc if min_proc is not None else None
                        max_proc = -max_proc if max_proc is not None else None

                    base_timestamp_utc = datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc)

                    # Prepare fields for the point
                    fields = {}
                    if mean_proc is not None:
                        fields["mean_value"] = mean_proc
                    if min_proc is not None:
                        fields["min_value"] = min_proc
                    if max_proc is not None:
                        fields["max_value"] = max_proc
                    
                    if not fields: # Skip if no valid fields after processing
                        continue

                    point = {
                        "measurement": target_measurement,
                        "tags": {
                            "source_metadata_id": str(metadata_id)
                        },
                        "time": base_timestamp_utc,
                        "fields": fields
                    }
                    points_for_current_config.append(point)
                    
                    if (row_index + 1) % 1000 == 0:
                        print(f"  Processed {row_index + 1} records for ID {metadata_id}...")

            if not points_for_current_config:
                print(f"No valid points to write for metadata_id = {metadata_id} after processing.")
                continue

            print(f"\nAttempting to write {len(points_for_current_config)} points for metadata_id {metadata_id} to InfluxDB with RP '{rp_to_use}'...")
            # Write points for the current configuration with the specific retention policy
            client.write_points(points_for_current_config, time_precision='s', retention_policy=rp_to_use)
            print(f"Successfully ingested {len(points_for_current_config)} points for metadata_id {metadata_id}.")

    except sqlite3.Error as e:
        print(f"SQLite database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if sqlite_conn:
            sqlite_conn.close()
            print("SQLite connection closed.")
        if client:
            client.close()
            print("InfluxDB connection closed.")

if __name__ == "__main__":
    print("Starting historical data ingestion script for InfluxDB 1.x with 1h to 30m upsampling and RP application (with gap filling)...")
    ingest_historical_data()
    print("Script finished.")
