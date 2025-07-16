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


# --- Target Measurements for Historical Weather Data ---
# Separate measurements for each weather metric
MEASUREMENT_TEMP = "temperature_adelaide"
MEASUREMENT_HUMIDITY = "humidity_adelaide"
MEASUREMENT_WIND_SPEED = "wind_speed_adelaide"


# Configuration for BOM Weather Sensor Data
# These are 1-hour source data, upsampled to 30min, with no gap filling.
SENSOR_CONFIGS = [
    # BOM Temperature
    {"table": "statistics", "metadata_id": 281, "sign_invert": False, "conversion_factor": 1, 
     "target_measurement": MEASUREMENT_TEMP, 
     "note": "BOM Temperature (1-hour), upsample to 30min, no gap fill"},
    
    # BOM Humidity
    {"table": "statistics", "metadata_id": 278, "sign_invert": False, "conversion_factor": 1, 
     "target_measurement": MEASUREMENT_HUMIDITY, 
     "note": "BOM Humidity (1-hour), upsample to 30min, no gap fill"},
    
    # BOM Wind Speed
    {"table": "statistics", "metadata_id": 277, "sign_invert": False, "conversion_factor": 1, 
     "target_measurement": MEASUREMENT_WIND_SPEED, 
     "note": "BOM Wind Speed (1-hour), upsample to 30min, no gap fill"}
]

def ingest_historical_weather_data():
    """
    Connects to the Home Assistant SQLite DB, queries historical BOM weather data,
    upsamples 1-hour data to 30-minute intervals (without gap filling),
    and ingests it into separate InfluxDB measurements per metric.
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
        databases = client.get_list_database() # Fixed typo here
        if {'name': INFLUXDB_DATABASE} not in databases:
            print(f"InfluxDB database '{INFLUXDB_DATABASE}' does not exist. Attempting to create it.")
            client.create_database(INFLUXDB_DATABASE)
            print(f"Database '{INFLUXDB_DATABASE}' created (if user had permissions).")
        else:
            print(f"InfluxDB database '{INFLUXDB_DATABASE}' already exists.")

        # Process each weather sensor configuration
        for config in SENSOR_CONFIGS:
            metadata_id = config["metadata_id"]
            sign_invert = config["sign_invert"] 
            conversion_factor = config["conversion_factor"]
            target_measurement = config["target_measurement"] # This is now specific (temp, humid, wind)
            table = config["table"]
            note = config["note"]

            rp_to_use = "rp_30m" # All 30-minute weather data goes to this RP

            print(f"\nProcessing data for metadata_id = {metadata_id} ({note}). Target Measurement: {target_measurement}, RP: {rp_to_use}")
            query = f"SELECT start_ts, mean, min, max FROM {table} WHERE metadata_id = {metadata_id} ORDER BY start_ts ASC"
            sqlite_cursor.execute(query)
            rows = sqlite_cursor.fetchall()

            if not rows:
                print(f"No historical data found for metadata_id = {metadata_id}.")
                continue

            print(f"Found {len(rows)} records for metadata_id = {metadata_id}.")
            
            points_for_current_config = []

            # Logic for 1-hour source, upsampled to 30min, NO gap filling for weather
            for row_index, row in enumerate(rows):
                start_ts, mean_val, min_val, max_val = row

                # Apply conversion factor
                mean_proc = mean_val * conversion_factor if mean_val is not None else None
                
                # Apply sign inversion if needed (unlikely for weather, but kept for consistency)
                if sign_invert:
                    mean_proc = -mean_proc if mean_proc is not None else None

                base_timestamp_utc = datetime.datetime.fromtimestamp(start_ts, tz=datetime.timezone.utc)

                fields = {}
                # The primary field name for these measurements will be 'mean_value'
                if mean_proc is not None:
                    fields["mean_value"] = float(mean_proc) # Ensure float type for InfluxDB

                if fields: # Only create points if valid fields exist
                    # Create two 30-minute points from this 1-hour point
                    point1 = {
                        "measurement": target_measurement,
                        "tags": {
                            "source_metadata_id": str(metadata_id),
                            "original_resolution": "1h",
                            "location": "Adelaide" # Explicitly tag location
                        },
                        "time": base_timestamp_utc,
                        "fields": fields
                    }
                    points_for_current_config.append(point1)

                    # Second 30-min point (30 minutes after the first)
                    timestamp_30m_later = base_timestamp_utc + datetime.timedelta(minutes=30)
                    point2 = {
                        "measurement": target_measurement,
                        "tags": {
                            "source_metadata_id": str(metadata_id),
                            "original_resolution": "1h",
                            "location": "Adelaide"
                        },
                        "time": timestamp_30m_later,
                        "fields": fields
                    }
                    points_for_current_config.append(point2)
                
                if (row_index + 1) % 1000 == 0:
                    print(f"  Processed {row_index + 1} records for ID {metadata_id}...")

            if not points_for_current_config:
                print(f"No valid points to write for metadata_id = {metadata_id} after processing.")
                continue

            # Sort all generated points by time to ensure correct ingestion order for the current config
            points_for_current_config.sort(key=lambda x: x['time'])

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
    print("Starting historical BOM weather data ingestion script from Home Assistant statistics (separate measurements)...")
    ingest_historical_weather_data()
    print("Script finished.")
