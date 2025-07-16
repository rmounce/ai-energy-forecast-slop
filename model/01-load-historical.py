import pandas as pd
from influxdb import InfluxDBClient
from datetime import datetime, timedelta
import pytz

# InfluxDB 1.x Configuration
INFLUXDB_HOST = 'REDACTED'
INFLUXDB_PORT = 8086
INFLUXDB_USERNAME = 'user'
INFLUXDB_PASSWORD = 'REDACTED'
INFLUXDB_DATABASE = 'hass'

# Create InfluxDB client
client = InfluxDBClient(
    host=INFLUXDB_HOST,
    port=INFLUXDB_PORT,
    username=INFLUXDB_USERNAME,
    password=INFLUXDB_PASSWORD,
    database=INFLUXDB_DATABASE
)

# Test connection
try:
    client.ping()
    print("Connected to InfluxDB successfully!")
except Exception as e:
    print(f"Failed to connect to InfluxDB: {e}")

end_time = datetime.now()

# Define time range (last year)
#start_time = end_time - timedelta(days=3*365)
# start date of comprehensive data (including weather)
#start_time = datetime.fromisoformat("2023-04-19 12:00:00+00:00")
# more than enough for now
start_time = end_time - timedelta(days=1*365)

# Format times for InfluxDB query (RFC3339 format)
start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

print(f"Querying data from {start_time_str} to {end_time_str}")

# Define all data sources with their queries
data_sources = {
    'power_load': {
        'query': f'SELECT mean_value FROM "rp_30m"."power_load_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'mean_value'
    },
    'power_pv': {
        'query': f'SELECT mean_value FROM "rp_30m"."power_pv_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'mean_value'
    },
    'temperature_adelaide': {
        'query': f'SELECT mean_value FROM "rp_30m"."temperature_adelaide" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'mean_value'
    },
    'humidity_adelaide': {
        'query': f'SELECT mean_value FROM "rp_30m"."humidity_adelaide" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'mean_value'
    },
    'wind_speed_adelaide': {
        'query': f'SELECT mean_value FROM "rp_30m"."wind_speed_adelaide" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'mean_value'
    },
    'aemo_price_sa1': {
        'query': f'SELECT price / 1000 FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'price'
    },
    'aemo_total_demand_sa1': {
        'query': f'SELECT total_demand FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'total_demand'
    },
    'aemo_net_interchange_sa1': {
        'query': f'SELECT net_interchange FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'net_interchange'
    },
    'aemo_price_nsw1': {
        'query': f'SELECT price / 1000 FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'price'
    },
    'aemo_total_demand_nsw1': {
        'query': f'SELECT total_demand FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'total_demand'
    },
    'aemo_net_interchange_nsw1': {
        'query': f'SELECT net_interchange FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'net_interchange'
    },
    'aemo_price_vic1': {
        'query': f'SELECT price / 1000 FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'price'
    },
    'aemo_total_demand_vic1': {
        'query': f'SELECT total_demand FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'total_demand'
    },
    'aemo_net_interchange_vic1': {
        'query': f'SELECT net_interchange FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_time_str}\' AND time <= \'{end_time_str}\'',
        'value_column': 'net_interchange'
    }
}

# Function to execute query and return DataFrame
def query_to_dataframe(client, query, value_column, name):
    try:
        print(f"Querying {name}...")
        result = client.query(query)
        
        if result:
            points = list(result.get_points())
            
            if points:
                df = pd.DataFrame(points)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Rename the value column to the data source name for clarity
                if value_column in df.columns:
                    df = df[[value_column]]  # Keep only the value column
                    df.columns = [name]
                    
                    print(f"  ✓ Retrieved {len(df)} records from {df.index.min()} to {df.index.max()}")
                    return df
                else:
                    print(f"  ✗ Column '{value_column}' not found in results")
                    return pd.DataFrame()
            else:
                print(f"  ✗ No data points found")
                return pd.DataFrame()
        else:
            print(f"  ✗ Query returned no results")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  ✗ Error querying {name}: {e}")
        return pd.DataFrame()

# Dictionary to store all DataFrames
dataframes = {}

# Execute all queries
print("Retrieving all data sources...")
print("=" * 50)

for name, config in data_sources.items():
    df = query_to_dataframe(client, config['query'], config['value_column'], name)
    if not df.empty:
        dataframes[name] = df

# Close the client connection
client.close()

# Display summary of retrieved data
print("\n" + "=" * 50)
print("SUMMARY OF RETRIEVED DATA:")
print("=" * 50)

for name, df in dataframes.items():
    print(f"{name:25} : {len(df):6} records | {df.index.min()} to {df.index.max()}")

# Create a combined DataFrame with all data sources
if dataframes:
    print(f"\nCombining all data sources...")
    
    # Start with the first dataframe
    combined_df = list(dataframes.values())[0].copy()
    
    # Join all other dataframes
    for df in list(dataframes.values())[1:]:
        combined_df = combined_df.join(df, how='outer')
    
    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    # Display basic info about the combined dataset
    print("\nCombined DataFrame Info:")
    print(combined_df.info())
    
    print("\nMissing data summary:")
    missing_data = combined_df.isnull().sum()
    missing_percentage = (missing_data / len(combined_df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing %': missing_percentage
    })
    print(missing_summary)
    
    print("\nFirst few rows of combined data:")
    print(combined_df.head())
    
    print("\nBasic statistics:")
    print(combined_df.describe())
    
    # Optional: Save to CSV
    # combined_df.to_csv('all_energy_data.csv')
    # print("\nData saved to 'all_energy_data.csv'")
    
else:
    print("No data was retrieved from any source.")

print("\n" + "=" * 50)
print("Data retrieval complete!")
print("Individual DataFrames are available in the 'dataframes' dictionary")
print("Combined DataFrame is available as 'combined_df'")
print("Ready for use with Darts and other ML libraries!")
