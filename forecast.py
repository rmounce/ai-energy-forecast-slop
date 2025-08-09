#!/usr/bin/env python3

#import pdb

# # --------------------------------------------------------------------------- #
# 1. IMPORTS
# --------------------------------------------------------------------------- #
import argparse
import json
import logging
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytz
import requests
from darts import TimeSeries
from darts.models import LightGBMModel
from influxdb import InfluxDBClient
from sklearn.multioutput import MultiOutputRegressor

import io
import re
import zipfile

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
# 2. LOGGING SETUP
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --------------------------------------------------------------------------- #
# 3. CENTRALIZED CONFIGURATION
# --------------------------------------------------------------------------- #
def load_config(config_path='config.json'):
    """Loads the configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file at {config_path}")
        raise

CONFIG = load_config()

# --------------------------------------------------------------------------- #
# 4. DATA FETCHING & PROCESSING FUNCTIONS
# --------------------------------------------------------------------------- #

def get_historical_data(client, start_time, end_time):
    # This function is REFINED based on your InfluxDB schema
    logging.info(f"Querying historical data from {start_time.date()} to {end_time.date()}")
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    # This dictionary is now structured to query the correct fields from your measurements.
    data_sources = {
        # Core measurements
        'power_load': f'SELECT mean("mean_value") FROM "rp_30m"."power_load_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'power_pv': f'SELECT mean("mean_value") FROM "rp_30m"."power_pv_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'temperature_adelaide': f'SELECT mean("mean_value") FROM "rp_30m"."temperature_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'humidity_adelaide': f'SELECT mean("mean_value") FROM "rp_30m"."humidity_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'wind_speed_adelaide': f'SELECT mean("mean_value") FROM "rp_30m"."wind_speed_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',

        # SA1 Data (from a single measurement)
        'aemo_price_sa1': f'SELECT mean("price") / 1000 FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'total_demand_sa1': f'SELECT mean("total_demand") FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'net_interchange_sa1': f'SELECT mean("net_interchange") FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',

        # VIC1 Data (VERIFY aemo_dispatch_vic1_30m measurement name)
        'total_demand_vic1': f'SELECT mean("total_demand") FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'net_interchange_vic1': f'SELECT mean("net_interchange") FROM "rp_30m"."aemo_dispatch_vic1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        
        # NSW1 Data (VERIFY aemo_dispatch_nsw1_30m measurement name)
        'total_demand_nsw1': f'SELECT mean("total_demand") FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'net_interchange_nsw1': f'SELECT mean("net_interchange") FROM "rp_30m"."aemo_dispatch_nsw1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
    }
    
    dataframes = {}
    for name, query in data_sources.items():
        try:
            result = client.query(query)
            # This existing logic handles the queries perfectly, as each query returns a single 'mean' column.
            if result and result.get_points():
                df = pd.DataFrame(result.get_points())
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df = df[['mean']].rename(columns={'mean': name})
                dataframes[name] = df
        except Exception as e:
            logging.error(f"Error querying {name}: {e}")

    if not dataframes:
        logging.error("FATAL: No data retrieved from InfluxDB.")
        return pd.DataFrame()
        
    # Using 'outer' join correctly handles missing data, like your 2-week interchange gap, by creating NaNs.
    # The ffill().dropna() in the training function will handle these.
    return pd.concat(dataframes.values(), axis=1, join='outer')


def call_ha_api(method, endpoint, payload=None):
    # This function remains unchanged
    ha_config = CONFIG['home_assistant']
    url = f"{ha_config['url']}/api/{endpoint}"
    headers = {"Authorization": f"Bearer {ha_config['token']}", "Content-Type": "application/json"}
    try:
        if method.upper() == 'POST':
            response = requests.post(url, headers=headers, json=payload, timeout=30)
        else:
            response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {method} {url} - {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response body: {e.response.text}")
    return None

def get_entity_state(entity_id):
    # This function remains unchanged
    return call_ha_api('GET', f"states/{entity_id}")

def get_amber_spot_price_forecast():
    # This function remains unchanged
    logging.info("Retrieving Amber Electric spot price forecast...")
    entity_id = CONFIG['home_assistant']['amber_entity']
    entity_state = get_entity_state(entity_id)
    if not entity_state: return pd.DataFrame()
    forecasts = entity_state.get("attributes", {}).get("Forecasts", [])
    if not forecasts: return pd.DataFrame()
    processed = [{'datetime': pd.to_datetime(f['start_time']).round('min'), 'aemo_price_sa1': float(f['spot_per_kwh'])} for f in forecasts if 'start_time' in f and f.get('spot_per_kwh') is not None]
    if not processed: return pd.DataFrame()
    return pd.DataFrame(processed).set_index('datetime')

def add_gst(price):
    if price > 0:
        return price * CONFIG['gst_rate']
    else:
        return price

def remove_gst(price):
    if price > 0:
        return price / CONFIG['gst_rate']
    else:
        return price

def get_amber_advanced_forecast(price_key='advanced_price_predicted'):
    """
    Retrieves Amber Electric ADVANCED price forecast (mixed intervals).

    Args:
        price_key (str): The attribute key for the price forecast. 
                         Defaults to 'advanced_price_predicted'.
                         Can be 'advanced_price_low' (p30) or 'advanced_price_high' (p70).
    """
    logging.info(f"Retrieving Amber Electric ADVANCED forecast using price key: '{price_key}'...")
    entity_id = CONFIG['home_assistant']['amber_billing_entity']
    entity_state = get_entity_state(entity_id)
    if not entity_state:
        return pd.DataFrame()

    forecasts = entity_state.get("attributes", {}).get("Forecasts", [])
    if not forecasts:
        return pd.DataFrame()

    processed = []
    five_minute_intervals = []
    thirty_minute_intervals = []

    for f in forecasts:
        # Check if the requested price_key and other essential keys exist
        if (f.get(price_key) is not None and
            f.get('per_kwh') is not None and
            f.get('spot_per_kwh') is not None):

            per_kwh = float(f['per_kwh'])
            spot_per_kwh = float(f['spot_per_kwh'])
            # Use the specified price_key to get the value
            advanced_price_incl_gst = float(f[price_key])
            duration = f.get('duration', 30)

            tariff = remove_gst(per_kwh) - spot_per_kwh
            advanced_price_no_tariff = remove_gst(advanced_price_incl_gst) - tariff

            start_time = pd.to_datetime(f['start_time']).round('min')

            interval_data = {
                'datetime': start_time,
                'aemo_price_sa1': advanced_price_no_tariff
            }

            if duration == 5:
                five_minute_intervals.append(interval_data)
            else:
                thirty_minute_intervals.append(interval_data)
    
    # ... (the rest of the function remains unchanged) ...

    if five_minute_intervals:
        grouped_5min = {}
        for interval in five_minute_intervals:
            boundary = interval['datetime'].floor('30min')
            if boundary not in grouped_5min:
                grouped_5min[boundary] = []
            grouped_5min[boundary].append(interval['aemo_price_sa1'])

        for boundary, prices in grouped_5min.items():
            processed.append({
                'datetime': boundary,
                'aemo_price_sa1': sum(prices) / len(prices)
            })

    processed.extend(thirty_minute_intervals)

    if not processed:
        return pd.DataFrame(columns=['aemo_price_sa1'], index=pd.to_datetime([]).tz_localize('UTC'))

    processed.sort(key=lambda x: x['datetime'])

    now = pd.Timestamp.now(tz='UTC')
    current_interval_start = now.floor('30min')
    if not processed:
        return pd.DataFrame(columns=['aemo_price_sa1'], index=pd.to_datetime([]).tz_localize('UTC'))
        
    earliest_forecast = processed[0]['datetime']

    if earliest_forecast > current_interval_start:
        backfill_entry = {
            'datetime': current_interval_start,
            'aemo_price_sa1': processed[0]['aemo_price_sa1']
        }
        processed.insert(0, backfill_entry)

    return pd.DataFrame(processed).set_index('datetime')

def get_solcast_forecast():
    # This function is unchanged
    logging.info("Retrieving Solcast solar forecast...")
    all_forecasts = []
    for entity_id in CONFIG['home_assistant']['solcast_entities']:
        entity_state = get_entity_state(entity_id)
        if entity_state:
            all_forecasts.extend(entity_state.get("attributes", {}).get("detailedForecast", []))
    if not all_forecasts: return pd.DataFrame()
    processed = [{'datetime': pd.to_datetime(f['period_start']).round('min'), 'power_pv': int(f.get('pv_estimate', 0) * 1000)} for f in all_forecasts]
    solar_df = pd.DataFrame(processed).set_index('datetime').sort_index()
    return solar_df[~solar_df.index.duplicated(keep='first')]

def get_weather_forecast():
    # This function is unchanged
    logging.info("Retrieving BOM weather forecast...")
    ha_config = CONFIG['home_assistant']
    service_data = {"type": "hourly", "entity_id": ha_config['weather_entity']}
    response_data = call_ha_api('POST', "services/weather/get_forecasts?return_response=true", payload=service_data)
    if not response_data: return pd.DataFrame()
    service_response = response_data.get("service_response", {})
    forecasts = service_response.get(ha_config['weather_entity'], {}).get("forecast", [])
    if not forecasts: return pd.DataFrame()
    weather_df = pd.DataFrame(forecasts)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    if weather_df['datetime'].dt.tz is None:
        local_tz = pytz.timezone(CONFIG['timezone'])
        weather_df['datetime'] = weather_df['datetime'].dt.tz_localize(local_tz)
    weather_df['datetime'] = weather_df['datetime'].dt.tz_convert('UTC').dt.round('min')
    weather_df.set_index('datetime', inplace=True)
    weather_cols = {'temperature': 'temperature_adelaide', 'humidity': 'humidity_adelaide', 'wind_speed': 'wind_speed_adelaide'}
    available_cols = [col for col in weather_cols.keys() if col in weather_df.columns]
    weather_df = weather_df[available_cols].rename(columns=weather_cols)
    if weather_df.empty: return pd.DataFrame()
    weather_30min = weather_df.reindex(pd.date_range(start=weather_df.index.min(), end=weather_df.index.max(), freq='30min'))
    weather_30min = weather_30min.interpolate(method='linear').ffill().bfill()
    logging.info(f"Retrieved and interpolated {len(weather_30min)} weather forecast records.")
    return weather_30min

def get_aemo_forecast():
    """
    Retrieves and processes demand and interchange forecasts by combining two AEMO sources:
    1. A high-resolution JSON API for the immediate ~24h forecast.
    2. The 7-day outlook ZIP/CSV report for the long-term forecast.
    The data is then stitched together to form a complete, continuous forecast.
    """
    logging.info("Retrieving AEMO demand and interchange forecast (multi-source)...")
    
    # --- Part 1: Fetch Short-Term (Next ~24h) Forecast via JSON API ---
    short_term_df = _get_aemo_short_term_forecast()

    # --- Part 2: Fetch Long-Term (7-Day) Forecast via NEMWeb ZIP File ---
    long_term_df = _get_aemo_7_day_outlook_forecast()

    # --- Part 3: Combine Forecasts ---
    if short_term_df.empty and long_term_df.empty:
        logging.error("Failed to retrieve any AEMO forecast data from all sources.")
        return pd.DataFrame()
    
    if short_term_df.empty:
        logging.warning("Using only the long-term AEMO forecast.")
        return long_term_df
        
    if long_term_df.empty:
        logging.warning("Using only the short-term AEMO forecast.")
        return short_term_df

    # Combine the two dataframes. The `combine_first` method fills NaN values in 
    # short_term_df with the values from long_term_df for the same index.
    # This is perfect for appending the long-term data where the short-term one ends.
    combined_df = short_term_df.combine_first(long_term_df)

    # Ensure the data is continuous and sorted
    final_df = combined_df.resample('30min').mean().ffill().bfill()
    final_df.sort_index(inplace=True)

    logging.info(f"Successfully combined short and long-term AEMO forecasts into {len(final_df)} intervals.")
    return final_df

def _get_aemo_short_term_forecast():
    """Helper to fetch the ~24h JSON API forecast."""
    logging.info("Fetching AEMO short-term forecast (JSON API)...")
    try:
        url = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
        payload = {"timeScale": ["30MIN"]}
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not fetch short-term AEMO forecast: {e}")
        return pd.DataFrame()

    if "5MIN" not in data or not data["5MIN"]:
        return pd.DataFrame()

    df = pd.DataFrame(data['5MIN'])
    # (The rest of this function is the same as the previous correct version)
    required_cols = ['SETTLEMENTDATE', 'REGIONID', 'TOTALDEMAND', 'NETINTERCHANGE']
    df = df[df['REGIONID'].isin(CONFIG['aemo_forecast']['regions'])]
    nem_tz = pytz.timezone('Etc/GMT-10')
    df['timestamp_end'] = pd.to_datetime(df['SETTLEMENTDATE']).dt.tz_localize(nem_tz)
    df = df.sort_values(['REGIONID', 'timestamp_end']).reset_index(drop=True)
    df['duration'] = df.groupby('REGIONID')['timestamp_end'].diff().fillna(pd.Timedelta(minutes=5))
    df['timestamp'] = (df['timestamp_end'] - df['duration']).dt.tz_convert('UTC')
    df.set_index('timestamp', inplace=True)
    pivot_df = df.pivot_table(index='timestamp', columns='REGIONID', values=['TOTALDEMAND', 'NETINTERCHANGE'])
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    rename_map = {
        'TOTALDEMAND_SA1': 'total_demand_sa1', 'NETINTERCHANGE_SA1': 'net_interchange_sa1',
        'TOTALDEMAND_VIC1': 'total_demand_vic1', 'NETINTERCHANGE_VIC1': 'net_interchange_vic1',
        'TOTALDEMAND_NSW1': 'total_demand_nsw1', 'NETINTERCHANGE_NSW1': 'net_interchange_nsw1',
    }
    pivot_df.rename(columns=rename_map, inplace=True)
    return pivot_df.resample('30min').mean()

def _get_aemo_7_day_outlook_forecast():
    """Helper to fetch and parse the latest 7-Day Outlook report from NEMWeb."""
    logging.info("Fetching AEMO 7-Day Outlook forecast (NEMWeb ZIP)...")
    try:
        # 1. Scrape the directory to find the latest file
        dir_url = "https://nemweb.com.au/Reports/Current/SEVENDAYOUTLOOK_FULL/"
        response = requests.get(dir_url, timeout=30)
        response.raise_for_status()
        
        file_pattern = r"PUBLIC_SEVENDAYOUTLOOK_FULL_(\d{14})_\d+\.zip"
        files = re.findall(file_pattern, response.text)
        if not files:
            logging.error("Could not find any 7-Day Outlook ZIP files in the directory.")
            return pd.DataFrame()
            
        latest_file_timestamp = sorted(files, reverse=True)[0]
        full_filename_match = re.search(f'PUBLIC_SEVENDAYOUTLOOK_FULL_{latest_file_timestamp}.*?\.zip', response.text)
        if not full_filename_match:
             logging.error("Could not reconstruct full filename for the 7-Day Outlook.")
             return pd.DataFrame()

        zip_url = f"{dir_url}{full_filename_match.group(0)}"
        logging.info(f"Downloading latest report: {zip_url}")

        # 2. Download and extract the CSV from the ZIP in memory
        zip_response = requests.get(zip_url, timeout=60)
        zip_response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                lines = f.read().decode('utf-8').splitlines()
                header_line = None
                data_lines = []
                for line in lines:
                    if line.startswith('I,'):
                        header_line = line.split(',')
                    elif line.startswith('D,'):
                        data_lines.append(line.split(','))

                if not header_line or not data_lines:
                    logging.error("Could not find header ('I') or data ('D') rows in the CSV.")
                    return pd.DataFrame()
                
                data_df = pd.DataFrame(data_lines, columns=header_line)

    except Exception as e:
        logging.error(f"Failed during processing of AEMO 7-Day Outlook: {e}")
        return pd.DataFrame()

    # 4. Clean, process, and pivot the data
    required_cols = ['INTERVAL_DATETIME', 'REGIONID', 'SCHEDULED_DEMAND', 'NET_INTERCHANGE']
    if not all(col in data_df.columns for col in required_cols):
        logging.error("7-Day Outlook CSV is missing required columns.")
        return pd.DataFrame()

    nem_tz = pytz.timezone('Etc/GMT-10')
    datetime_format = '"%Y/%m/%d %H:%M:%S"'
    
    # --- FIX ---
    # The timestamp is the END of the interval. To get the START time for consistency,
    # we must subtract 30 minutes after parsing.
    data_df['timestamp'] = (
        pd.to_datetime(data_df['INTERVAL_DATETIME'], format=datetime_format)
        .dt.tz_localize(nem_tz) 
        - pd.Timedelta(minutes=30)
    ).dt.tz_convert('UTC')
    # --- END OF FIX ---

    data_df.set_index('timestamp', inplace=True)
    
    data_df['SCHEDULED_DEMAND'] = pd.to_numeric(data_df['SCHEDULED_DEMAND'], errors='coerce')
    data_df['NET_INTERCHANGE'] = pd.to_numeric(data_df['NET_INTERCHANGE'], errors='coerce')
    
    data_df = data_df[data_df['REGIONID'].isin(CONFIG['aemo_forecast']['regions'])]
    
    pivot_df = data_df.pivot_table(index='timestamp', columns='REGIONID', values=['SCHEDULED_DEMAND', 'NET_INTERCHANGE'])
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]

    rename_map = {
        'SCHEDULED_DEMAND_SA1': 'total_demand_sa1', 'NET_INTERCHANGE_SA1': 'net_interchange_sa1',
        'SCHEDULED_DEMAND_VIC1': 'total_demand_vic1', 'NET_INTERCHANGE_VIC1': 'net_interchange_vic1',
        'SCHEDULED_DEMAND_NSW1': 'total_demand_nsw1', 'NET_INTERCHANGE_NSW1': 'net_interchange_nsw1',
    }
    pivot_df.rename(columns=rename_map, inplace=True)
    
    return pivot_df

# --------------------------------------------------------------------------- #
# 5. MODEL TRAINING, PREDICTION & PUBLISHING FUNCTIONS
# --------------------------------------------------------------------------- #

def log_forecast_data(model_name, model_version, prediction_type, final_pred_df, future_covariates_df):
    """
    Logs data to a model-specific CSV, intelligently matching the column order
    of the existing file to prevent misalignment.
    """
    # Determine the base name for finding the log file (e.g., 'price' from 'price_p30')
    base_model_name = model_name.split('_')[0]
    logging.info(f"Logging forecast data for '{model_name}' model to '{base_model_name}' log file...")
    log_file_path = Path(CONFIG['paths'][f'{base_model_name}_forecast_log_file'])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Prepare the new data in memory ---
    log_df = final_pred_df.copy()
    if log_df.index.tz is None:
        log_df.index = log_df.index.tz_localize('UTC')

    prediction_col = log_df.columns[0]
    log_df.rename(columns={prediction_col: 'prediction'}, inplace=True)
    log_df['actual'] = np.nan
    log_df = log_df.join(future_covariates_df, how='left')
    log_df['forecast_creation_time'] = datetime.now(pytz.UTC).isoformat()
    log_df['model_name'] = model_name
    log_df['model_version'] = model_version
    log_df['prediction_type'] = prediction_type
    log_df.index.name = 'forecast_target_time'
    log_df.reset_index(inplace=True)

    # --- ROBUSTNESS FIX: Determine and Enforce Column Order ---
    final_column_order = []
    file_exists = log_file_path.exists() and os.path.getsize(log_file_path) > 0

    if file_exists:
        # File exists, so the file's header is the source of truth for the column order.
        final_column_order = pd.read_csv(log_file_path, nrows=0).columns.tolist()
    else:
        # File is new, so we define a default canonical order for its creation.
        # This will only run once for each log file.
        final_column_order = [
            'forecast_creation_time', 'forecast_target_time', 'model_name', 'model_version',
            'prediction_type', 'prediction', 'actual',
            'power_pv', 'power_pv_actual',
            'temperature_adelaide', 'temperature_adelaide_actual',
            'humidity_adelaide', 'humidity_adelaide_actual',
            'wind_speed_adelaide', 'wind_speed_adelaide_actual',
            'aemo_price_sa1',
            'hour', 'day_of_week', 'day_of_year', 'month'
        ]

    # Add any missing columns to the new data with NaN values
    for col in final_column_order:
        if col not in log_df.columns:
            log_df[col] = np.nan
    
    # Reorder the new data to precisely match the target order (from file or default)
    # and filter out any columns in the new data that aren't in the target order.
    log_df = log_df[[col for col in final_column_order if col in log_df.columns]]

    # Append to the CSV. 'header=False' is crucial for appending.
    # If the file is new, we write the header; otherwise, we don't.
    log_df.to_csv(log_file_path, mode='a', header=(not file_exists), index=False)
    logging.info(f"Successfully logged {len(log_df)} records to {log_file_path} with consistent column order.")

def train_single_model(model_name, quantile_info=None):
    """
    Trains a single model and saves it.
    Can handle standard regression or native LightGBM quantile regression.
    """
    logging.info(f"--- Running in TRAIN mode for model: {model_name} ---")
    
    # For a quantile model like 'price_p30', use the base 'price' config
    base_model_name = model_name.split('_')[0]
    model_config = CONFIG['models'][base_model_name]
    
    target_col, feature_cols = model_config['target_column'], model_config['feature_cols']
    client = InfluxDBClient(**CONFIG['influxdb'])

    try:
        # ... (all your data loading code is correct) ...
        target_lags_list = model_config.get('target_lags', 48 * 7)
        future_covariate_lags_list = model_config.get('future_covariate_lags', [0])
        training_days = model_config.get('training_history_days', CONFIG['training_history_days'])
        logging.info(f"Using a training history of {training_days} days.")
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(days=training_days)
        historical_df = get_historical_data(client, start_time, end_time)
        if historical_df.empty: raise SystemExit("Aborting: Failed to retrieve historical data.")
        model_data = historical_df[[target_col] + feature_cols].copy().ffill().dropna()
        min_val = model_data[target_col].min()
        shift_value = abs(min_val) + 1 if min_val <= 0 else 0
        model_data[target_col] = np.log(model_data[target_col] + shift_value)
        target_series = TimeSeries.from_series(model_data[target_col], freq='30min')
        covariates_ts = TimeSeries.from_dataframe(model_data, value_cols=feature_cols, freq='30min')
        sample_weight = None
        weighting_config = model_config.get('recency_weighting')
        if weighting_config and weighting_config.get('enabled'):
            logging.info("Applying recency weighting to the training data.")
            half_life_days = weighting_config.get('half_life_days', 90)
            time_delta_days = (model_data.index.max() - model_data.index).total_seconds() / (24 * 3600)
            weights_series = pd.Series(
                np.power(2, -time_delta_days / half_life_days),
                index=model_data.index
            )
            sample_weight = TimeSeries.from_series(weights_series, freq='30min')
            logging.info(f"Sample weights calculated and converted to TimeSeries with a half-life of {half_life_days} days.")
        encoders_config = model_config.get('add_encoders')

        lgbm_params = model_config['lgbm_params'].copy()
        if quantile_info:
            logging.info(f"Setting native LGBM parameters for quantile training: {quantile_info}")
            # We only need objective and alpha for quantile regression
            lgbm_params.update({k: v for k, v in quantile_info.items() if k in ['objective', 'alpha']})

        lgbm = LightGBMModel(
            lags=model_config.get('target_lags'),
            lags_future_covariates=model_config.get('future_covariate_lags'),
            output_chunk_length=model_config['forecast_horizon'],
            add_encoders=encoders_config,
            **lgbm_params
        )

        logging.info("Fitting the model...")
        lgbm.fit(series=target_series, future_covariates=covariates_ts, sample_weight=sample_weight)
        
        logging.info("Extracting feature importances...")
        true_feature_names = lgbm.lagged_feature_names

        if isinstance(lgbm.model, MultiOutputRegressor):
            importances = np.mean([estimator.feature_importances_ for estimator in lgbm.model.estimators_], axis=0)
        else:
            importances = lgbm.model.feature_importances_

        if len(true_feature_names) != len(importances):
            logging.error(f"FATAL: Length of feature names ({len(true_feature_names)}) does not match length of importances ({len(importances)})!")
        else:
            feature_importance_dict = {key: value.item() for key, value in zip(true_feature_names, importances)}
            sorted_importance = {k: v for k, v in sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)}
            importance_file_path = CONFIG['paths'][f'{model_name}_importance_file']
            with open(importance_file_path, 'w') as f:
                json.dump(sorted_importance, f, indent=4)
            logging.info(f"Feature importances saved successfully to {importance_file_path}")

        params = {'shift_value': shift_value}
        joblib.dump(lgbm, CONFIG['paths'][f'{model_name}_model_file'])
        with open(CONFIG['paths'][f'{model_name}_params_file'], 'w') as f:
            json.dump(params, f, indent=4)
        logging.info("Model and parameters saved successfully.")

    finally:
        client.close()

def train_models(base_model_name):
    """
    Orchestrator to train all configured quantile models for a base model type (e.g., 'price' or 'load').
    """
    logging.info(f"--- Starting training for all '{base_model_name}' models ---")
    model_config = CONFIG['models'][base_model_name]
    quantile_models_config = model_config.get('quantile_models')

    if not quantile_models_config:
        logging.warning(f"No 'quantile_models' defined for '{base_model_name}'. Training a single model.")
        train_single_model(model_name=base_model_name)
        return

    for model_name, quantile_info in quantile_models_config.items():
        logging.info(f"\n*** Training sub-model: {model_name} ***")
        train_single_model(model_name=model_name, quantile_info=quantile_info)

    logging.info(f"--- All '{base_model_name}' models trained successfully ---")

def _predict_simple(model, params, historical_df, future_covariates_ts, model_config):
    # This function is unchanged
    logging.info("Generating forecast using 'simple' method.")
    target_col = model_config['target_column']
    hist_df = historical_df[[target_col] + model_config['feature_cols']].copy().ffill().dropna()
    hist_df[target_col] = np.log(hist_df[target_col] + params['shift_value'])
    target_series = TimeSeries.from_series(hist_df[target_col], freq='30min')
    predictions = model.predict(n=model_config['forecast_horizon'], series=target_series, future_covariates=future_covariates_ts)
    pred_df = predictions.to_dataframe()
    pred_df[target_col] = np.exp(pred_df[target_col]) - params['shift_value']
    return pred_df

def _predict_with_dynamic_handoff(model, params, historical_df, future_covariates_df, model_config, amber_advanced_df):
    """
    Generates a forecast using the 'dynamic handoff' method, seeded with a provided
    advanced forecast dataframe.
    """
    logging.info("Generating forecast using 'dynamic handoff' method.")
    target_col = model_config['target_column']
    shift_value = params['shift_value']
    
    # The function now receives the amber_advanced_df instead of fetching it.
    if amber_advanced_df.empty:
        logging.warning("No advanced Amber data provided. Falling back to simple prediction.")
        future_covariates_ts = TimeSeries.from_dataframe(
            future_covariates_df, 
            value_cols=model_config['feature_cols'], 
            freq='30min'
        )
        return _predict_simple(model, params, historical_df, future_covariates_ts, model_config)
        
    last_good_amber_index = amber_advanced_df.index.max()
    logging.info(f"Using provided advanced forecast for {len(amber_advanced_df) / 2} hours (up to {last_good_amber_index}).")
    hist_df_log = historical_df[[target_col] + model_config['feature_cols']].copy().ffill().dropna()
    hist_df_log[target_col] = np.log(hist_df_log[target_col] + shift_value)
    amber_seed_data = future_covariates_df.loc[amber_advanced_df.index].copy()
    amber_seed_data[target_col] = np.log(amber_advanced_df[target_col] + shift_value)
    pseudo_history_df = pd.concat([hist_df_log, amber_seed_data.dropna()])
    pseudo_history_ts = TimeSeries.from_series(pseudo_history_df[target_col], freq='30min')
    
    future_covariates_ts = TimeSeries.from_dataframe(
        future_covariates_df, 
        value_cols=model_config['feature_cols'], 
        freq='30min'
    )
    
    steps_to_predict = max(0, model_config['forecast_horizon'] - len(amber_advanced_df))
    remaining_forecast_df = pd.DataFrame()
    if steps_to_predict > 0:
        logging.info(f"Model will predict the remaining {steps_to_predict / 2} hours.")
        remaining_forecast_series = model.predict(n=steps_to_predict, series=pseudo_history_ts, future_covariates=future_covariates_ts)
        remaining_forecast_df = remaining_forecast_series.to_dataframe()
    final_forecast_df = amber_advanced_df.copy()
    if not remaining_forecast_df.empty:
        remaining_forecast_df[target_col] = np.exp(remaining_forecast_df[target_col]) - shift_value
        if remaining_forecast_df.index.tz is None:
            remaining_forecast_df.index = remaining_forecast_df.index.tz_localize('UTC')
        final_forecast_df = pd.concat([final_forecast_df, remaining_forecast_df])
    return final_forecast_df

def _execute_quantile_prediction(base_model_name, historical_df, adjusted_covariates_for_prediction, use_dynamic_handoff):
    """
    Handles prediction for a model with multiple quantiles (e.g., price or load).
    It loads each sub-model, generates a forecast, and returns a dictionary of forecasts.
    """
    logging.info(f"--- Generating forecasts for all '{base_model_name}' models ---")
    
    model_config = CONFIG['models'][base_model_name]
    quantile_models_config = model_config.get('quantile_models', {})
    raw_forecasts = {}

    for model_name, quantile_info in quantile_models_config.items():
        logging.info(f"--- Processing forecast for sub-model: {model_name} ---")
        model_file_path = CONFIG['paths'][f'{model_name}_model_file']

        try:
            model = joblib.load(model_file_path)
            with open(CONFIG['paths'][f'{model_name}_params_file'], 'r') as f:
                params = json.load(f)
            logging.info(f"Loaded model '{model_name}' successfully.")
        except FileNotFoundError:
            logging.error(f"Model file for '{model_name}' not found. Skipping this forecast.")
            continue
        
        pred_df = pd.DataFrame()
        if base_model_name == 'price' and use_dynamic_handoff:
            amber_price_key = quantile_info.get('price_key', 'advanced_price_predicted')
            amber_df = get_amber_advanced_forecast(price_key=amber_price_key)
            if amber_df.empty:
                logging.warning(f"No advanced Amber data for price key '{amber_price_key}'. Skipping.")
                continue
            pred_df = _predict_with_dynamic_handoff(model, params, historical_df, adjusted_covariates_for_prediction, model_config, amber_df)
        else:
            future_covariates_ts = TimeSeries.from_dataframe(
                adjusted_covariates_for_prediction, value_cols=model_config['feature_cols'], freq='30min'
            )
            pred_df = _predict_simple(model, params, historical_df, future_covariates_ts, model_config)
        
        if not pred_df.empty:
            raw_forecasts[model_name] = pred_df.rename(columns={model_config['target_column']: model_name})

    if not raw_forecasts:
        logging.error(f"Failed to generate any forecasts for '{base_model_name}'.")
        return {}

    if base_model_name == 'price' and len(raw_forecasts) > 1:
        logging.info("Applying sorting to prevent price quantile crossing...")
        sorted_keys = sorted(quantile_models_config.keys(), key=lambda k: quantile_models_config[k]['alpha'])
        
        if all(key in raw_forecasts for key in sorted_keys):
            combined_raw_df = pd.concat([raw_forecasts[key] for key in sorted_keys], axis=1)
            sorted_values = np.sort(combined_raw_df.values, axis=1)
            sorted_df = pd.DataFrame(sorted_values, index=combined_raw_df.index, columns=sorted_keys)
            final_forecasts = {key: sorted_df[[key]] for key in sorted_keys}
            return final_forecasts
        else:
            logging.warning("Could not generate all price forecasts. Using raw forecasts without sorting.")
            return raw_forecasts
    else:
        return raw_forecasts

def _execute_single_prediction(model_name, historical_df, adjusted_covariates_for_prediction, use_dynamic_handoff):
    """
    WORKER: Executes prediction for a model type (e.g. 'price' or 'load'),
    delegating to a quantile-aware predictor if configured.
    """
    logging.info(f"\n>>> Executing prediction for model type: {model_name} <<<")

    model_config = CONFIG['models'].get(model_name, {})
    all_forecasts = {}
    prediction_type = 'simple'

    if 'quantile_models' in model_config:
        all_forecasts = _execute_quantile_prediction(
            base_model_name=model_name,
            historical_df=historical_df,
            adjusted_covariates_for_prediction=adjusted_covariates_for_prediction,
            use_dynamic_handoff=use_dynamic_handoff
        )
        if model_name == 'price':
            prediction_type = 'dynamic_handoff' if use_dynamic_handoff else 'simple'
    else:
        logging.error(f"Unknown model name '{model_name}' passed to prediction worker or not configured for quantiles.")

    if not all_forecasts:
        logging.error(f"Prediction failed for model type '{model_name}'.")

    return all_forecasts, prediction_type

def run_predictions(models_to_run, publish_hass, use_dynamic_handoff, publish_covariates):
    """
    ORCHESTRATOR: Fetches data, runs predictions for all specified model families,
    and then handles all logging, saving, and publishing.
    """
    logging.info(f"--- Prediction Orchestrator started for models: {models_to_run} ---")

    # 1. Fetch and process data ONCE (This part is unchanged)
    logging.info("Fetching all future covariate and recent historical data...")
    future_sources = {'solcast': get_solcast_forecast(), 'weather': get_weather_forecast(), 'aemo': get_aemo_forecast()}
    now = datetime.now(pytz.UTC)
    minute = 30 if now.minute >= 30 else 0
    forecast_start_time = now.replace(minute=minute, second=0, microsecond=0)
    client = InfluxDBClient(**CONFIG['influxdb'])
    try:
        history_start = forecast_start_time - timedelta(days=CONFIG['prediction_history_days'])
        history_end = forecast_start_time - timedelta(minutes=30)
        historical_df = get_historical_data(client, history_start, history_end)
        if historical_df.empty: raise SystemExit("Aborting: Failed to get recent history for prediction.")
    finally:
        client.close()
    
    future_covariates_df = pd.concat(future_sources.values(), axis=1).sort_index()
    all_feature_cols = set()
    for model_name in models_to_run:
        if model_name in CONFIG['models']:
            all_feature_cols.update(CONFIG['models'][model_name]['feature_cols'])
    
    available_cols = [col for col in all_feature_cols if col in historical_df.columns]
    historical_covariates_df = historical_df[available_cols]

    combined_covariates_df = pd.concat([historical_covariates_df, future_covariates_df])
    combined_covariates_df = combined_covariates_df[~combined_covariates_df.index.duplicated(keep='last')].sort_index()
    original_covariates_for_log = combined_covariates_df.copy()
    adjusted_covariates_df = apply_covariate_adjustments(combined_covariates_df)
    if publish_covariates:
        publish_df = adjusted_covariates_df[adjusted_covariates_df.index >= forecast_start_time]
        publish_adjusted_covariates_to_hass(publish_df)
    adjusted_covariates_for_prediction = adjusted_covariates_df.copy()
    adjusted_covariates_for_prediction.ffill(inplace=True)
    adjusted_covariates_for_prediction.bfill(inplace=True)

    # 2. Loop, execute predictions, and COLLECT results (This part is unchanged)
    all_results = {}
    for model_name in models_to_run:
        forecasts, prediction_type = _execute_single_prediction(
            model_name=model_name,
            historical_df=historical_df,
            adjusted_covariates_for_prediction=adjusted_covariates_for_prediction,
            use_dynamic_handoff=use_dynamic_handoff
        )
        if forecasts:
            all_results[model_name] = {'forecasts': forecasts, 'type': prediction_type}

    # 3. Process and SAVE all collected results (This part is unchanged)
    try:
        with open(CONFIG['paths']['prediction_output_file'], 'r') as f:
            final_output_json = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        final_output_json = {}

    for model_name, result_data in all_results.items():
        primary_key = CONFIG['models'][model_name].get('primary_model_key')
        if not primary_key or primary_key not in result_data['forecasts']:
            logging.warning(f"Primary key '{primary_key}' for model '{model_name}' not found or configured. Skipping save.")
            continue
        
        primary_pred_df_for_save = result_data['forecasts'][primary_key].copy()
        logging.info(f"Preparing to save primary forecast for '{model_name}'...")
        
        if model_name == 'price':
            primary_pred_df_for_save.rename(columns={primary_pred_df_for_save.columns[0]: 'wholesale_price'}, inplace=True)
            if not use_dynamic_handoff:
                amber_spot_df = get_amber_spot_price_forecast()
                if not amber_spot_df.empty:
                    primary_pred_df_for_save.update(amber_spot_df)
            apply_tariffs_to_forecast(primary_pred_df_for_save)
        
        primary_pred_df_for_save.index.name = 'timestamp'
        output_df = primary_pred_df_for_save.reset_index()
        output_df['timestamp'] = output_df['timestamp'].apply(lambda x: x.isoformat())
        
        final_output_json[f'{model_name}_forecast'] = output_df.to_dict('records')
        final_output_json[f'{model_name}_last_updated'] = datetime.now(pytz.UTC).isoformat()

    with open(CONFIG['paths']['prediction_output_file'], 'w') as f:
        json.dump(final_output_json, f, indent=4)
    logging.info(f"All forecasts saved to {CONFIG['paths']['prediction_output_file']}.")

    # 4. PUBLISH all forecasts to Home Assistant
    if publish_hass:
        logging.info("--- Publishing all generated forecasts to Home Assistant ---")
        for base_model_name, result_data in all_results.items():
            # --- FIX STARTS HERE ---
            model_config = CONFIG['models'][base_model_name]
            target_col = model_config['target_column'] 

            for key, forecast_df in result_data['forecasts'].items():
                entity_id_check = CONFIG['home_assistant']['publish_entities'].get(key)
                if not entity_id_check: continue

                logging.info(f"Processing and publishing '{key}' to '{entity_id_check}'...")
                publish_df = forecast_df.copy()
                
                if base_model_name == 'price':
                    # Price models are renamed to 'wholesale_price' for the tariff function
                    publish_df.rename(columns={publish_df.columns[0]: 'wholesale_price'}, inplace=True)
                    apply_tariffs_to_forecast(publish_df)
                else:
                    # All other models (e.g., load) are renamed to their generic target column
                    publish_df.rename(columns={publish_df.columns[0]: target_col}, inplace=True)
                # --- FIX ENDS HERE ---

                publish_forecast_to_hass(key, publish_df)

    
    # 5. LOG all forecasts (This part is unchanged)
    for model_name, result_data in all_results.items():
        primary_key = CONFIG['models'][model_name].get('primary_model_key')
        if primary_key and primary_key in result_data['forecasts']:
            model_version = "N/A"
            try:
                mod_time = os.path.getmtime(CONFIG['paths'][f'{primary_key}_model_file'])
                model_version = datetime.fromtimestamp(mod_time, tz=pytz.UTC).isoformat()
            except (FileNotFoundError, KeyError): pass
            
            primary_pred_df_for_log = result_data['forecasts'][primary_key].copy()
            log_forecast_data(model_name, model_version, result_data['type'], primary_pred_df_for_log, original_covariates_for_log)

    logging.info("--- Prediction Orchestrator finished ---")


def apply_tariffs_to_forecast(pred_df):
    # This function remains unchanged
    logging.info("Applying tariffs to wholesale price forecast with conditional GST...")
    try:
        with open(CONFIG['paths']['tariff_file'], 'r') as f:
            tariffs = json.load(f)
    except FileNotFoundError:
        logging.warning("tariff_profile.json not found. Skipping tariff application.")
        pred_df['general_price'] = pred_df['wholesale_price']
        pred_df['feed_in_price'] = pred_df['wholesale_price']
        return

    if not isinstance(pred_df.index, pd.DatetimeIndex):
        pred_df.index = pd.to_datetime(pred_df.index)

    if pred_df.index.tz is None:
        pred_df.index = pred_df.index.tz_localize('UTC')

    local_tz = pytz.timezone(CONFIG['timezone'])
    pred_df['local_time'] = pred_df.index.tz_convert(local_tz).time.astype(str)

    general_tariff_map = tariffs.get('general_tariff', {})
    feed_in_tariff_map = tariffs.get('feed_in_tariff', {})

    pred_df['general_tariff'] = pred_df['local_time'].map(general_tariff_map).fillna(0)
    pred_df['feed_in_tariff'] = pred_df['local_time'].map(feed_in_tariff_map).fillna(0)

    general_price_ex_gst = pred_df['wholesale_price'] + pred_df['general_tariff']
    feed_in_price_ex_gst = pred_df['wholesale_price'] + pred_df['feed_in_tariff']

    pred_df['general_price'] = np.where(general_price_ex_gst > 0, general_price_ex_gst * CONFIG['gst_rate'], general_price_ex_gst)
    pred_df['feed_in_price'] = np.where(feed_in_price_ex_gst < 0, feed_in_price_ex_gst * CONFIG['gst_rate'], feed_in_price_ex_gst)

    pred_df.drop(columns=['local_time', 'general_tariff', 'feed_in_tariff'], inplace=True)
    logging.info("Successfully applied general and feed-in tariffs with conditional GST.")

def publish_forecast_to_hass(model_key, forecast_df):
    """Publishes a forecast DataFrame to a Home Assistant entity using a model key."""
    entity_id = CONFIG['home_assistant']['publish_entities'].get(model_key)
    if not entity_id:
        logging.warning(f"No publish entity found for key '{model_key}'.")
        return
        
    logging.info(f"Publishing {model_key} forecast to Home Assistant entity: {entity_id}")
    forecast_df.index.name = 'timestamp'
    output_df = forecast_df.reset_index()
    if output_df['timestamp'].dt.tz is None:
        output_df['timestamp'] = output_df['timestamp'].dt.tz_localize('UTC')
    output_df['timestamp'] = output_df['timestamp'].apply(lambda x: x.isoformat())
    
    friendly_name_key = model_key.replace('_', ' ').title()
    attributes = {
        "forecasts": output_df.to_dict('records'), 
        "last_updated": datetime.now(pytz.UTC).isoformat(), 
        "friendly_name": f"AI {friendly_name_key} Forecast", 
        "icon": "mdi:chart-line"
    }
    
    # Use 'general_price' if it exists (for tariffed price models), otherwise use the first column
    state_col = 'general_price' if 'general_price' in forecast_df.columns else forecast_df.columns[0]
    state = round(forecast_df.iloc[0][state_col], 4)
    
    payload = {"state": state, "attributes": attributes}
    call_ha_api('POST', f"states/{entity_id}", payload=payload)
    logging.info(f"Successfully published state '{state}' and attributes to {entity_id}.")


def _get_tariff_data(entity_id, is_feed_in=False):
    # This function remains unchanged
    entity_state = get_entity_state(entity_id)
    if not entity_state: return pd.DataFrame()
    forecasts = entity_state.get("attributes", {}).get("Forecasts", [])
    if not forecasts: return pd.DataFrame()
    processed = []
    for f in forecasts:
        if f.get('per_kwh') is not None and f.get('spot_per_kwh') is not None:
            per_kwh_dollars = float(f['per_kwh'])
            spot_per_kwh_dollars = float(f['spot_per_kwh'])
            if is_feed_in:
                tariff = -remove_gst(per_kwh_dollars) - spot_per_kwh_dollars
            else:
                tariff = remove_gst(per_kwh_dollars) - spot_per_kwh_dollars
            processed.append({'datetime': pd.to_datetime(f['start_time']).round('min'), 'tariff': tariff})
    return pd.DataFrame(processed).set_index('datetime')

def update_tariffs():
    # This function is unchanged
    logging.info("--- Running in UPDATE-TARIFFS mode ---")
    local_tz = pytz.timezone(CONFIG['timezone'])
    general_tariff_df = _get_tariff_data(CONFIG['home_assistant']['amber_entity'], is_feed_in=False)
    feed_in_tariff_df = _get_tariff_data(CONFIG['home_assistant']['amber_feed_in_entity'], is_feed_in=True)
    if general_tariff_df.empty and feed_in_tariff_df.empty:
        raise SystemExit("Could not retrieve any tariff information from Amber entities.")
    final_profile = {}
    if not general_tariff_df.empty:
        general_tariff_df.index = general_tariff_df.index.tz_convert(local_tz)
        general_tariff_df['time'] = general_tariff_df.index.time
        general_profile = general_tariff_df.groupby('time')['tariff'].mean().to_dict()
        final_profile['general_tariff'] = {str(k): v for k, v in general_profile.items()}
        logging.info(f"Generated 24h general tariff profile with {len(general_profile)} entries.")
    if not feed_in_tariff_df.empty:
        feed_in_tariff_df.index = feed_in_tariff_df.index.tz_convert(local_tz)
        feed_in_tariff_df['time'] = feed_in_tariff_df.index.time
        feed_in_profile = feed_in_tariff_df.groupby('time')['tariff'].mean().to_dict()
        final_profile['feed_in_tariff'] = {str(k): v for k, v in feed_in_profile.items()}
        logging.info(f"Generated 24h feed-in tariff profile with {len(feed_in_profile)} entries.")
    try:
        with open(CONFIG['paths']['tariff_file'], 'w') as f:
            json.dump(final_profile, f, indent=4)
        logging.info(f"Successfully saved combined tariff profile to {CONFIG['paths']['tariff_file']}")
    except Exception as e:
        logging.error(f"Failed to save tariff profile: {e}")

def _backfill_single_log(model_name: str):
    """
    Helper function to backfill actuals for a single model's log file,
    including both the main prediction and all covariate forecasts.
    """
    log_file_path = Path(CONFIG['paths'][f'{model_name}_forecast_log_file'])
    
    logging.info(f"--- Backfilling model: '{model_name}' ---")
    if not log_file_path.exists() or os.path.getsize(log_file_path) == 0:
        logging.info(f"Log file for '{model_name}' not found or is empty. Skipping.")
        return

    # Load the data from the CSV file
    log_df = pd.read_csv(log_file_path)

    # --- ROBUSTNESS FIX: Explicitly convert the timestamp column after loading ---
    # This guarantees the correct dtype before we use the .dt accessor.
    try:
        log_df['forecast_target_time'] = pd.to_datetime(log_df['forecast_target_time'], format='ISO8601')
    except Exception as e:
        logging.error(f"Could not parse 'forecast_target_time' in {log_file_path}. Error: {e}")
        return
    # --- End of fix ---

    if log_df['forecast_target_time'].dt.tz is None:
        log_df['forecast_target_time'] = log_df['forecast_target_time'].dt.tz_localize('UTC')

    # Define the mapping from InfluxDB names to our '_actual' column names
    actual_col_map = {
        CONFIG['models'][model_name]['target_column']: 'actual',
        'power_pv': 'power_pv_actual',
        'temperature_adelaide': 'temperature_adelaide_actual',
        'humidity_adelaide': 'humidity_adelaide_actual',
        'wind_speed_adelaide': 'wind_speed_adelaide_actual',
    }
    
    # Find all records where at least one of the actual columns is missing
    cols_to_check = [col for col in actual_col_map.values() if col in log_df.columns]
    mask_to_fill = log_df[cols_to_check].isna().any(axis=1) & (log_df['forecast_target_time'] < datetime.now(pytz.UTC))

    records_to_fill = log_df[mask_to_fill]
    if records_to_fill.empty:
        logging.info(f"No records need backfilling for '{model_name}'.")
        return

    logging.info(f"Found {len(records_to_fill)} records with missing actuals to backfill for '{model_name}'.")
    start_time = records_to_fill['forecast_target_time'].min()
    end_time = records_to_fill['forecast_target_time'].max()

    logging.info(f"Querying InfluxDB for all actuals from {start_time} to {end_time}.")
    client = InfluxDBClient(**CONFIG['influxdb'])
    try:
        actuals_df = get_historical_data(client, start_time, end_time)
        if actuals_df.empty:
            logging.warning("No historical data found for the backfill time range.")
            return

        # Prepare actuals df by renaming columns according to our map
        actuals_renamed_df = actuals_df.rename(columns=actual_col_map)
        
        log_df.set_index('forecast_target_time', inplace=True)
        # Update fills in NaNs from the right DataFrame into the left one, matching on index
        log_df.update(actuals_renamed_df)
        log_df.reset_index(inplace=True)
        
    finally:
        client.close()
    
    log_df.to_csv(log_file_path, index=False)
    logging.info(f"Successfully saved backfilled data to {log_file_path}")

def backfill_actuals():
    """
    Runs the backfill process for all configured models.
    """
    logging.info("--- Running in BACKFILL-ACTUALS mode for all models ---")
    _backfill_single_log('price')
    _backfill_single_log('load')

def update_adjusters():
    """
    Analyzes the forecast logs to calculate time-of-day biases for covariates
    and saves them as JSON adjustment profiles.
    """
    logging.info("--- Running in UPDATE-ADJUSTERS mode ---")
    if 'adjusters' not in CONFIG:
        logging.warning("No 'adjusters' section in config.json. Nothing to do.")
        return

    # We can use either log file, as the covariate data is identical.
    # Let's prefer the load log as it's less sparse.
    log_file = Path(CONFIG['paths']['load_forecast_log_file'])
    if not log_file.exists():
        logging.error(f"Log file {log_file} not found. Cannot generate adjusters.")
        return
        
    df = pd.read_csv(log_file)
    df['forecast_target_time'] = pd.to_datetime(df['forecast_target_time'], format='ISO8601')

    history_days = CONFIG.get('adjuster_history_days')
    if history_days:
        logging.info(f"Using a rolling window of the last {history_days} days for adjuster calculation.")
        cutoff_date = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=history_days)
        df = df[df['forecast_target_time'] >= cutoff_date]
        if df.empty:
            logging.warning("No data found within the rolling window. Aborting adjuster update.")
            return

    df.set_index('forecast_target_time', inplace=True)
    df['target_time_of_day'] = df.index.time

    for cov_name, adjuster_config in CONFIG['adjusters'].items():
        logging.info(f"Generating adjuster for: {cov_name}")
        forecast_col = cov_name
        actual_col = f"{cov_name}_actual"

        if forecast_col not in df.columns or actual_col not in df.columns:
            logging.warning(f"Columns for {cov_name} not found in log. Skipping.")
            continue
            
        # Create a clean df for this specific covariate
        cov_df = df[[forecast_col, actual_col, 'target_time_of_day']].dropna()
        if cov_df.empty:
            logging.warning(f"No complete forecast/actual pairs for {cov_name}. Skipping.")
            continue

        if adjuster_config['type'] == 'additive':
            # Calculate the additive error (actual - forecast)
            cov_df['error'] = cov_df[actual_col] - cov_df[forecast_col]
            # Group by time of day and find the mean error
            bias_profile = cov_df.groupby('target_time_of_day')['error'].mean()
            logging.info(f"Calculated additive bias profile with {len(bias_profile)} entries.")
            
        elif adjuster_config['type'] == 'multiplicative':
            # For PV, a scaling factor is better. Avoid division by zero.
            cov_df_safe = cov_df[cov_df[forecast_col] > 10] # Only use forecasts > 10W for stable factors
            # Calculate scaling factor (actual / forecast)
            cov_df_safe['scaling_factor'] = cov_df_safe[actual_col] / cov_df_safe[forecast_col]
            # Replace infinite values and clip to prevent extreme adjustments (e.g., 0.1x to 2.0x)
            bias_profile = cov_df_safe.groupby('target_time_of_day')['scaling_factor'].mean()
            bias_profile.replace([np.inf, -np.inf], np.nan, inplace=True)
            bias_profile.dropna(inplace=True)
            bias_profile = bias_profile.clip(0.1, 2.0)
            logging.info(f"Calculated multiplicative bias profile with {len(bias_profile)} entries.")
        
        else:
            logging.warning(f"Unknown adjuster type '{adjuster_config['type']}' for {cov_name}. Skipping.")
            continue
            
        # Convert to a dictionary format that's JSON-friendly
        profile_dict = {str(k): v for k, v in bias_profile.to_dict().items()}

        # Save the profile to its JSON file
        try:
            with open(adjuster_config['path'], 'w') as f:
                json.dump(profile_dict, f, indent=4)
            logging.info(f"Successfully saved adjuster profile to {adjuster_config['path']}")
        except Exception as e:
            logging.error(f"Failed to save adjuster profile for {cov_name}: {e}")

def apply_covariate_adjustments(future_covariates_df):
    """
    Loads adjuster profiles and applies corrections to the future covariates DataFrame.
    """
    logging.info("Applying covariate forecast adjustments...")
    if 'adjusters' not in CONFIG:
        return future_covariates_df # Return unchanged if no adjusters are configured

    adjusted_df = future_covariates_df.copy()
    adjusted_df['time_of_day'] = adjusted_df.index.time.astype(str)

    for cov_name, adjuster_config in CONFIG['adjusters'].items():
        if cov_name not in adjusted_df.columns:
            continue
            
        try:
            with open(adjuster_config['path'], 'r') as f:
                profile = json.load(f)
        except FileNotFoundError:
            # This is not an error; the profile may not have been generated yet.
            continue
            
        logging.info(f"Applying '{adjuster_config['type']}' adjustment to '{cov_name}'...")
        
        # Create a mapping Series from the profile
        adjustment_map = pd.Series(profile)
        
        if adjuster_config['type'] == 'additive':
            # Map the time of day to the additive adjustment value, fill missing with 0
            adjustments = adjusted_df['time_of_day'].map(adjustment_map).fillna(0)
            adjusted_df[cov_name] += adjustments
            
        elif adjuster_config['type'] == 'multiplicative':
            # Map the time of day to the scaling factor, fill missing with 1 (no change)
            adjustments = adjusted_df['time_of_day'].map(adjustment_map).fillna(1.0)
            adjusted_df[cov_name] *= adjustments

    return adjusted_df.drop(columns=['time_of_day'])

def _publish_covariates_helper():
    """
    A helper function that fetches all future covariate data sources, applies
    adjustments, and publishes the results. Called by main() when needed.
    """
    logging.info("--- Publishing Adjusted Covariates ---")

    # Fetch ALL future data sources to get the maximum possible horizon
    logging.info("Fetching all future covariate data sources...")
    future_sources = {
        'solcast': get_solcast_forecast(),
        'weather': get_weather_forecast(),
        'aemo': get_aemo_forecast() # Include AEMO to ensure we use its 7-day index
    }

    # Combine them into a single DataFrame
    future_covariates_df = pd.concat(future_sources.values(), axis=1).sort_index()

    # Apply the bias adjustments
    adjusted_df = apply_covariate_adjustments(future_covariates_df)

    # Filter to only future data before publishing
    now = datetime.now(pytz.UTC)
    minute = 30 if now.minute >= 30 else 0
    forecast_start_time = now.replace(minute=minute, second=0, microsecond=0)
    
    publish_df = adjusted_df[adjusted_df.index >= forecast_start_time]

    # Call the existing, robust publishing function
    publish_adjusted_covariates_to_hass(publish_df)

def publish_adjusted_covariates_to_hass(adjusted_covariates_df):
    """
    Publishes adjusted covariate forecasts to specified Home Assistant entities.
    For each covariate, it individually truncates any trailing NaN values to
    publish only the valid forecast horizon, ensuring data is JSON compliant.
    """
    logging.info("Publishing adjusted covariates to Home Assistant...")
    if 'adjusters' not in CONFIG:
        return

    for cov_name, adjuster_config in CONFIG['adjusters'].items():
        entity_id = adjuster_config.get('publish_entity_id')
        
        if not entity_id or cov_name not in adjusted_covariates_df.columns:
            continue

        forecast_df = adjusted_covariates_df[[cov_name]].dropna()

        if forecast_df.empty:
            logging.warning(f"No valid data available for '{cov_name}' after dropping NaNs. Skipping publish.")
            continue
            
        logging.info(f"Publishing {len(forecast_df)} valid records for adjusted '{cov_name}' to {entity_id}")
        
        # --- FIX: Explicitly name the index before resetting it ---
        forecast_df.index.name = 'timestamp'
        
        # Now, reset_index() will correctly create a 'timestamp' column
        output_df = forecast_df.reset_index()
            
        if output_df['timestamp'].dt.tz is None:
            output_df['timestamp'] = output_df['timestamp'].dt.tz_localize('UTC')
        output_df['timestamp'] = output_df['timestamp'].apply(lambda x: x.isoformat())
        
        state = round(forecast_df.iloc[0][cov_name], 2)
        friendly_name = f"AI Adjusted {cov_name.replace('_', ' ').title()} Forecast"
        attributes = {
            "forecasts": output_df.to_dict('records'),
            "last_updated": datetime.now(pytz.UTC).isoformat(),
            "friendly_name": friendly_name,
            "icon": "mdi:chart-line"
        }
        
        payload = {"state": state, "attributes": attributes}
        
        response = call_ha_api('POST', f"states/{entity_id}", payload=payload)
        
        if response:
            logging.info(f"Successfully published state '{state}' and attributes to {entity_id}.")

# --------------------------------------------------------------------------- #
# 6. MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Energy Price & Load Forecasting Pipeline")
    parser.add_argument('mode', choices=[
        'train-price', 'train-load',
        'predict-price', 'predict-load', 'predict-all',
        'update-tariffs', 'backfill-actuals', 'update-adjusters'
    ], help="The mode to run the script in.")
    
    parser.add_argument('--publish-covariates', action='store_true', help="Publish adjusted covariate forecasts to Home Assistant.")
    parser.add_argument('--publish-hass', action='store_true', help="Publish FINAL forecasts to Home Assistant entities.")
    parser.add_argument('--dynamic-handoff', action='store_true', help="For 'predict-price' mode, use Amber's advanced forecast to seed the model.")
    parser.add_argument('--config', default='config.json', help="Path to the configuration file.")
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_config(args.config)

    if args.mode.startswith('predict-'):
        models = []
        if args.mode == 'predict-price':
            models = ['price']
        elif args.mode == 'predict-load':
            models = ['load']
        elif args.mode == 'predict-all':
            models = ['price', 'load']
        
        if models:
            run_predictions(
                models_to_run=models,
                publish_hass=args.publish_hass,
                use_dynamic_handoff=args.dynamic_handoff,
                publish_covariates=args.publish_covariates
            )
    
    elif args.mode.startswith('train-'):
        model_name = args.mode.split('-')[1]
        train_models(model_name)
    
    elif args.mode == 'update-tariffs':
        update_tariffs()
    elif args.mode == 'backfill-actuals':
        backfill_actuals()
    elif args.mode == 'update-adjusters':
        update_adjusters()

if __name__ == "__main__":
    main()
