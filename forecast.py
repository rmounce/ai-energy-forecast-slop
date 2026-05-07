#!/usr/bin/env python3

#import pdb

# # --------------------------------------------------------------------------- #
# 1. IMPORTS
# --------------------------------------------------------------------------- #
import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
import pytz
import requests
from aemo_session import make_aemo_session
from influxdb import InfluxDBClient
import time
import pickle

import io
import re
import zipfile

from tariff_utils import (
    ensure_utc_index,
    export_value_to_amber_feed_in_price,
    load_tariff_profile,
    tariffed_price_frame_from_wholesale_mwh,
)
from eval.retro_tier1_inference import build_feature_dict as build_tier1_feature_dict, build_long_matrix_for_model as build_tier1_long_matrix

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
from config_utils import load_config

CONFIG = load_config()
ROOT = Path(__file__).resolve().parent
GENERAL_TARIFF_MAP, FEED_IN_TARIFF_MAP, NETWORK_LOSS_FACTOR = load_tariff_profile(CONFIG, ROOT)

# Shared cached session for all AEMO/NEMWeb HTTP requests
_aemo_session = make_aemo_session()

# Set to True by --debug-tft flag to print TFT input/output diagnostic table
_DEBUG_TFT = False

# --------------------------------------------------------------------------- #
# 4. DATA FETCHING & PROCESSING FUNCTIONS
# --------------------------------------------------------------------------- #

def add_time_features(df):
    """
    Adds custom boolean features `is_daylight_saving_time` and `is_public_holiday`
    to a timezone-aware DataFrame based on its index.
    """
    if df.empty:
        return df
    
    local_tz = pytz.timezone(CONFIG['timezone'])
    # Convert index to local time to check DST/Holidays
    if df.index.tz is None:
        local_df_index = df.index.tz_localize('UTC').tz_convert(local_tz)
    else:
        local_df_index = df.index.tz_convert(local_tz)

    df['is_daylight_saving_time'] = local_df_index.map(lambda x: x.dst() > pd.Timedelta(seconds=0)).astype(int)

    sa_holidays = holidays.AU(subdiv='SA')
    dates = pd.Series(local_df_index.date)
    df['is_public_holiday'] = dates.apply(lambda d: d in sa_holidays).values.astype(int)

    return df

def get_amber_api_scaling_factor():
    """Reads tariff_profile.json to extract amber_api_scaling_factor. Defaults to 1.10."""
    try:
        with open(CONFIG['paths']['tariff_file'], 'r') as f:
            tariffs = json.load(f)
            return tariffs.get('amber_api_scaling_factor', 1.10)
    except (FileNotFoundError, json.JSONDecodeError):
        return 1.10

def get_network_loss_factor():
    """Reads tariff_profile.json to extract network_loss_factor. Defaults to 1.05."""
    try:
        with open(CONFIG['paths']['tariff_file'], 'r') as f:
            tariffs = json.load(f)
            return tariffs.get('network_loss_factor', 1.05)
    except (FileNotFoundError, json.JSONDecodeError):
        return 1.05

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

        # Estimated dump load (2x2000W heaters on smart switches, active during
        # negative price periods). Subtracted from power_load before training/
        # prediction so the model learns base load, not dump-load-inflated load.
        'power_dump_load': f'SELECT mean("mean_value") FROM "rp_30m"."power_dump_load_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
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
    df_combined = pd.concat(dataframes.values(), axis=1, join='outer')

    if 'power_dump_load' in df_combined.columns:
        df_combined['power_load'] = (
            df_combined['power_load'].fillna(0) - df_combined['power_dump_load'].fillna(0)
        ).clip(lower=0)
        df_combined.drop(columns=['power_dump_load'], inplace=True)

    return add_time_features(df_combined)


def add_tft_regime_features(df, rrp_col='aemo_price_sa1'):
    """
    Real-time version of regime features used for TFT Run 010.
    Calculates log-momentum and 30m volatility from historical RRP.
    """
    if df.empty or rrp_col not in df.columns:
        return df

    # ── Log Momentum (Rolling 2h slope of log-scaled RRP)
    # Scale factor 60.0 matches Run 010 training
    log_rrp = np.sign(df[rrp_col]) * np.log1p(np.abs(df[rrp_col]) / 60.0)
    
    # Simple rolling slope estimate (last 4 intervals = 2h)
    # y = mx + c; we just want a rough 'direction' flag
    df['rrp_log_momentum'] = log_rrp.diff(4).fillna(0) / 4.0

    # ── 30m Volatility (Standard deviation from 5m aggregates if available)
    # If 5m prices are available, they will be used in the calling function.
    # Otherwise, we use this 30m proxy:
    if 'rrp_volatility_30m' not in df.columns:
        df['rrp_volatility_30m'] = log_rrp.rolling(window=6).std().fillna(0)

    return df


def get_5m_price_history(client, start_time, end_time):
    """Fetches high-frequency 5m SA1 prices for regime features."""
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    query = f'SELECT "price" FROM "rp_5m"."aemo_dispatch_sa1_5m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\''
    try:
        result = client.query(query)
        if result and result.get_points():
            df = pd.DataFrame(result.get_points())
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df['price']  # $/MWh (matches training pipeline)
    except Exception as e:
        logging.warning(f"Error querying 5m prices: {e}")
    return pd.Series(dtype=float)


def time_sin_cos(timestamps):
    """Matches time_encodings in build_training_dataset.py."""
    t = timestamps.tz_convert("Australia/Brisbane")
    features = {}
    features["hour_sin"]   = np.sin(2 * np.pi * t.hour / 24)
    features["hour_cos"]   = np.cos(2 * np.pi * t.hour / 24)
    features["dow_sin"]    = np.sin(2 * np.pi * t.dayofweek / 7)
    features["dow_cos"]    = np.cos(2 * np.pi * t.dayofweek / 7)
    features["month_sin"]  = np.sin(2 * np.pi * (t.month - 1) / 12)
    features["month_cos"]  = np.cos(2 * np.pi * (t.month - 1) / 12)
    return pd.DataFrame(features, index=timestamps)


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

def _persist_amber_spot_5min_forecasts(entity_id, entity_state, forecasts):
    log_path = Path(CONFIG['paths'].get('amber_spot_5min_forecast_log_file', 'amber_spot_5min_forecast_log.parquet'))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    forecast_creation_time = pd.to_datetime(
        entity_state.get("last_updated") or datetime.now(pytz.UTC).isoformat(),
        utc=True,
        errors="coerce",
    )
    if pd.isna(forecast_creation_time):
        forecast_creation_time = pd.Timestamp.now(tz="UTC")
    retrieval_time = pd.Timestamp.now(tz="UTC")
    scaling_factor = float(get_amber_api_scaling_factor())

    rows = []
    for f in forecasts:
        if "start_time" not in f or f.get("spot_per_kwh") is None:
            continue
        target_time = pd.to_datetime(f["start_time"], utc=True, errors="coerce")
        if pd.isna(target_time):
            continue
        raw_spot_per_kwh = float(f["spot_per_kwh"])
        rows.append({
            "forecast_creation_time": forecast_creation_time,
            "retrieval_time": retrieval_time,
            "forecast_target_time": target_time.round("min"),
            "entity_id": entity_id,
            "duration_minutes": int(f.get("duration", 5) or 5),
            "amber_spot_per_kwh_raw": raw_spot_per_kwh,
            "aemo_price_sa1_adjusted": raw_spot_per_kwh / scaling_factor if scaling_factor else raw_spot_per_kwh,
            "amber_api_scaling_factor": scaling_factor,
        })

    if not rows:
        return

    new_df = pd.DataFrame(rows)
    key_cols = ["forecast_creation_time", "forecast_target_time", "entity_id"]
    if log_path.exists() and log_path.stat().st_size > 0:
        existing_df = pd.read_parquet(log_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(key_cols, kind="stable").reset_index(drop=True)
    combined.to_parquet(log_path, index=False)
    logging.info(
        "Logged %s Amber 5-min spot forecast rows to %s",
        len(new_df),
        log_path,
    )

def get_amber_spot_price_forecast(apply_loss_factor=True):
    # This function remains unchanged
    logging.info("Retrieving Amber Electric spot price forecast...")
    entity_id = CONFIG['home_assistant']['amber_entity']
    entity_state = get_entity_state(entity_id)
    if not entity_state: return pd.DataFrame()
    forecasts = entity_state.get("attributes", {}).get("Forecasts", [])
    if not forecasts: return pd.DataFrame()
    processed = [{'datetime': pd.to_datetime(f['start_time']).round('min'), 'aemo_price_sa1': float(f['spot_per_kwh'])} for f in forecasts if 'start_time' in f and f.get('spot_per_kwh') is not None]
    if not processed: return pd.DataFrame()
    df = pd.DataFrame(processed).set_index('datetime')
    if apply_loss_factor:
        df['aemo_price_sa1'] /= get_amber_api_scaling_factor()
    return df

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

def get_amber_advanced_forecast(price_key='advanced_price_predicted', apply_loss_factor=True):
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

    df = pd.DataFrame(processed).set_index('datetime')
    if apply_loss_factor:
        df['aemo_price_sa1'] /= get_amber_api_scaling_factor()
    return df

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
        # BOM returns naive local-time strings. During DST fall-back, the same wall-clock
        # time appears twice (e.g. 02:30 ACDT then 02:30 ACST). Build an explicit ambiguous
        # array so pandas doesn't have to guess: first occurrence = DST (True),
        # second occurrence = standard time (False). Non-ambiguous times ignore the flag.
        # nonexistent='shift_forward' handles spring-forward (October) where BOM may include
        # a time that doesn't exist in local time.
        seen: set = set()
        ambiguous = []
        for dt in weather_df['datetime']:
            ambiguous.append(dt not in seen)
            seen.add(dt)
        weather_df['datetime'] = weather_df['datetime'].dt.tz_localize(
            local_tz, ambiguous=ambiguous, nonexistent='shift_forward'
        )
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
    url = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
    payload = {"timeScale": ["30MIN"]}
    
    max_attempts = 5
    timeout_seconds = 10
    
    data = None
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                backoff = 2 ** (attempt - 2) # 1s, 2s, 4s, 8s
                logging.info(f"Retry attempt {attempt}/{max_attempts} for AEMO forecast (backoff: {backoff}s)...")
                time.sleep(backoff)
            
            response = _aemo_session.post(url, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            data = response.json()
            break # Success
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt}/{max_attempts} failed to fetch AEMO forecast: {e}")
            if attempt == max_attempts:
                logging.error(f"Final attempt failed. Could not fetch short-term AEMO forecast: {e}")
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

def _get_aemo_short_term_price_forecast(regions=['SA1', 'VIC1', 'NSW1']):
    logging.info(f"Fetching AEMO short-term price forecast for regions: {regions}...")
    url = "https://visualisations.aemo.com.au/aemo/apps/api/report/5MIN"
    payload = {"timeScale": ["30MIN"]}
    
    try:
        response = _aemo_session.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "5MIN" not in data or not data["5MIN"]:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['5MIN'])
        df = df[df['REGIONID'].isin(regions)]
        if df.empty:
            return pd.DataFrame()
            
        nem_tz = pytz.timezone('Etc/GMT-10')
        df['timestamp'] = (pd.to_datetime(df['SETTLEMENTDATE']).dt.tz_localize(nem_tz) - pd.Timedelta(minutes=30)).dt.tz_convert('UTC')
        
        pivot_df = df.pivot_table(index='timestamp', columns='REGIONID', values='RRP')
        pivot_df = pivot_df.astype(float) / 1000.0  # Raw $/MWh
        
        # Rename columns to match model expectations
        rename_map = {
            'SA1': 'pd_rrp',
            'VIC1': 'vic1_pd_rrp',
            'NSW1': 'nsw1_pd_rrp'
        }
        pivot_df.rename(columns=rename_map, inplace=True)
        
        return pivot_df.resample('30min').mean().dropna(how='all')
    except Exception as e:
        logging.error(f"Failed to fetch AEMO multi-region price forecast: {e}")
        return pd.DataFrame()

def _get_aemo_7_day_outlook_forecast():
    """Helper to fetch and parse the latest 7-Day Outlook report from NEMWeb."""
    logging.info("Fetching AEMO 7-Day Outlook forecast (NEMWeb ZIP)...")
    try:
        # 1. Scrape the directory to find the latest file
        dir_url = "https://nemweb.com.au/Reports/CURRENT/SEVENDAYOUTLOOK_FULL/"
        response = _aemo_session.get(dir_url, timeout=30)
        response.raise_for_status()
        
        file_pattern = r"PUBLIC_SEVENDAYOUTLOOK_FULL_(\d{14})_\d+\.zip"
        files = re.findall(file_pattern, response.text)
        if not files:
            logging.error("Could not find any 7-Day Outlook ZIP files in the directory.")
            return pd.DataFrame()
            
        latest_file_timestamp = sorted(files, reverse=True)[0]
        full_filename_match = re.search(f'PUBLIC_SEVENDAYOUTLOOK_FULL_{latest_file_timestamp}.*?\\.zip', response.text)
        if not full_filename_match:
             logging.error("Could not reconstruct full filename for the 7-Day Outlook.")
             return pd.DataFrame()

        zip_url = f"{dir_url}{full_filename_match.group(0)}"
        logging.info(f"Downloading latest report: {zip_url}")

        # 2. Download and extract the CSV from the ZIP in memory
        zip_response = _aemo_session.get(zip_url, timeout=60)
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
    if model_name.startswith('tft_'):
        log_file_path = Path(CONFIG['paths'].get('tft_price_forecast_log_file', 'tft_forecast_log.csv'))
    else:
        base_model_name = model_name.split('_')[0]
        log_file_path = Path(CONFIG['paths'][f'{base_model_name}_forecast_log_file'])
    
    logging.info(f"Logging forecast data for '{model_name}' model to '{log_file_path.name}' log file...")
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
            'is_daylight_saving_time', 'is_public_holiday',
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
    from darts import TimeSeries
    from darts.models import LightGBMModel
    from sklearn.multioutput import MultiOutputRegressor
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
    from darts import TimeSeries
    logging.info("Generating forecast using 'simple' method.")
    target_col = model_config['target_column']
    hist_df = historical_df[[target_col] + model_config['feature_cols']].copy()
    if 'power_pv' in hist_df.columns:
        hist_df['power_pv'] = hist_df['power_pv'].fillna(0)
    hist_df = hist_df.ffill().dropna()
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
    from darts import TimeSeries
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
    hist_df_log = historical_df[[target_col] + model_config['feature_cols']].copy()
    if 'power_pv' in hist_df_log.columns:
        hist_df_log['power_pv'] = hist_df_log['power_pv'].fillna(0)
    hist_df_log = hist_df_log.ffill().dropna()
    hist_df_log[target_col] = np.log(hist_df_log[target_col] + shift_value)
    amber_seed_data = future_covariates_df.loc[amber_advanced_df.index].copy()
    amber_seed_data[target_col] = np.log(amber_advanced_df[target_col] + shift_value)
    pseudo_history_df = pd.concat([hist_df_log, amber_seed_data.dropna()])
    pseudo_history_df = pseudo_history_df[~pseudo_history_df.index.duplicated(keep='last')]
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
    from darts import TimeSeries
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


def _get_influx_pd_prices(client, start_time, end_time):
    """Get TFT decoder price covariates from InfluxDB, matching training structure.

    Returns DataFrame with columns pd_rrp, vic1_pd_rrp, nsw1_pd_rrp (all $/MWh),
    indexed by UTC datetime, or empty DataFrame on failure.
    """
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str   = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        pd_frames = {}
        for region, col in [('SA1', 'pd_rrp'), ('VIC1', 'vic1_pd_rrp'), ('NSW1', 'nsw1_pd_rrp')]:
            q = (f"SELECT last(rrp) AS rrp FROM \"rp_30m\".\"aemo_predispatch_forecast\" "
                 f"WHERE region='{region}' AND time >= '{start_str}' AND time <= '{end_str}' "
                 f"GROUP BY time(30m) fill(null)")
            r = client.query(q)
            if r and list(r.get_points()):
                df = pd.DataFrame(r.get_points())
                df['time'] = pd.to_datetime(df['time'], utc=True)
                df.set_index('time', inplace=True)
                pd_frames[col] = df['rrp']

        if not pd_frames:
            return pd.DataFrame()
        return pd.DataFrame(pd_frames)
    except Exception as e:
        logging.warning(f"Failed to get InfluxDB PD prices: {e}")
        return pd.DataFrame()


def _get_influx_latest_pd7day_prices(client, forecast_run_time, dec_index):
    """Fetch the latest PD7Day run published at or before forecast_run_time.

    Returns (pd7_df, pd7_run_time) where pd7_df has column pd7_rrp indexed by
    dec_index. pd7_run_time is the selected run timestamp or None if unavailable.
    """
    if len(dec_index) == 0:
        return pd.DataFrame(), None

    start_str = dec_index.min().strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str   = (dec_index.max() + pd.Timedelta(minutes=30)).strftime('%Y-%m-%dT%H:%M:%SZ')

    try:
        q = (
            f"SELECT rrp FROM \"rp_30m\".\"aemo_pd7day_forecast\" "
            f"WHERE region='SA1' AND time >= '{start_str}' AND time < '{end_str}' "
            f"GROUP BY run_time"
        )
        result = client.query(q)
        runs = {}
        for key in result.keys():
            run_time_str = key[1].get("run_time") if len(key) > 1 else None
            if not run_time_str:
                continue
            run_time = pd.Timestamp(run_time_str, tz="UTC")
            if run_time > forecast_run_time:
                continue
            rows = list(result[key])
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            runs[run_time] = df.set_index("time")["rrp"].sort_index()

        if not runs:
            return pd.DataFrame(index=dec_index), None

        latest_run_time = max(runs)
        pd7_df = pd.DataFrame({"pd7_rrp": runs[latest_run_time].reindex(dec_index)})
        return pd7_df, latest_run_time
    except Exception as e:
        logging.warning(f"Failed to get latest InfluxDB PD7Day prices: {e}")
        return pd.DataFrame(index=dec_index), None


def _get_influx_sdo_demand(client, start_time, end_time):
    """Get SevenDayOutlook scheduled_demand and net_interchange for SA1.

    Returns DataFrame with columns sd_demand, sd_net_interchange (both in MW),
    indexed by UTC datetime, or empty DataFrame on failure.
    """
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str   = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    try:
        q = (f"SELECT last(scheduled_demand) AS sd_demand, "
             f"last(net_interchange) AS sd_net_interchange "
             f"FROM \"rp_30m\".\"aemo_sevendayoutlook\" "
             f"WHERE region='SA1' AND time >= '{start_str}' AND time <= '{end_str}' "
             f"GROUP BY time(30m) fill(null)")
        r = client.query(q)
        if not r or not list(r.get_points()):
            return pd.DataFrame()
        df = pd.DataFrame(r.get_points())
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        return df[['sd_demand', 'sd_net_interchange']].fillna(0.0)
    except Exception as e:
        logging.warning(f"Failed to get InfluxDB SDO demand: {e}")
        return pd.DataFrame()


def _select_tft_decoder_features(fut, dec_feature_names):
    """Project the full decoder frame into the checkpoint's expected feature layout."""
    if dec_feature_names is None:
        dec_feature_names = [
            "pd_rrp", "pd_demand", "pd_net_interchange",
            "vic1_pd_rrp", "nsw1_pd_rrp", "pd7_rrp",
            "sd_demand", "sd_net_interchange",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "horizon_norm", "predispatch_active", "pd7_generation_hour", "pd7_available",
        ]

    dec = fut.copy()
    if "covar_missing" in dec_feature_names:
        combined_available = np.maximum(
            dec.get("predispatch_active", pd.Series(0.0, index=dec.index)).values,
            ((dec.get("pd7_available", pd.Series(0.0, index=dec.index)).values > 0) &
             (dec.get("pd7_rrp", pd.Series(0.0, index=dec.index)).values != 0)).astype(np.float32),
        )
        dec["covar_missing"] = 1.0 - combined_available
        dec["pd_rrp"] = np.where(
            dec.get("predispatch_active", pd.Series(0.0, index=dec.index)).values > 0,
            dec["pd_rrp"].values,
            dec.get("pd7_rrp", pd.Series(0.0, index=dec.index)).values,
        )

    missing = [feat for feat in dec_feature_names if feat not in dec.columns]
    for feat in missing:
        dec[feat] = 0.0
    return dec[dec_feature_names].values.astype(np.float32), dec_feature_names


def _print_tft_debug(enc_rrp_raw, enc_load_raw, enc_5m_missing, dec_pd_rrp_raw, dec_index, preds_raw,
                     fut_raw=None):
    """Print side-by-side diagnostic table for TFT inputs and outputs.

    Helps diagnose systematic underestimation: shows what the model actually
    received (encoder history + decoder PREDISPATCH covariates) vs what it predicted.
    All prices in $/MWh for comparability; TFT outputs are $/MWh before /1000 conversion.
    """
    SEP = "─" * 90

    # ── Encoder summary (last 96 steps = 48h)
    print(f"\n{SEP}")
    print("TFT DEBUG — Encoder input (last 96 steps = 48h of history)")
    print(SEP)
    missing_frac = enc_5m_missing.mean() * 100
    print(f"  5m features missing: {missing_frac:.0f}% of encoder steps")
    if not enc_load_raw.empty:
        load_zeros = (enc_load_raw == 0).sum()
        print(f"  power_load zero steps: {load_zeros}/96 {'⚠️  likely missing/bfilled' if load_zeros > 10 else '✓'}")
    print(f"  rrp ($/MWh):  min={enc_rrp_raw.min():.1f}  median={enc_rrp_raw.median():.1f}  "
          f"max={enc_rrp_raw.max():.1f}  last={enc_rrp_raw.iloc[-1]:.1f}")
    print(f"\n  {'Time (UTC)':>22}  {'rrp $/MWh':>10}  {'5m_missing':>10}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
    for ts, rrp, miss in zip(enc_rrp_raw.index[-8:], enc_rrp_raw.values[-8:], enc_5m_missing.values[-8:]):
        print(f"  {str(ts):>22}  {rrp:>10.2f}  {int(miss):>10}")

    # ── Detailed per-step covariate + output table (first 30 steps = 15h)
    if fut_raw is not None:
        # Fetch Amber APF for comparison (best-effort)
        amber_apf = {}
        try:
            amber_state = get_entity_state(CONFIG['home_assistant']['amber_entity'])
            if amber_state:
                for item in amber_state.get('attributes', {}).get('Forecasts', []):
                    ts_key = pd.Timestamp(item['start_time']).tz_convert('UTC').floor('30min')
                    amber_apf[ts_key] = float(item.get('advanced_price_predicted', 0)) * 1000.0  # c/kWh→$/MWh
        except Exception:
            pass

        tz_local = pytz.timezone(CONFIG.get('timezone', 'Australia/Adelaide'))
        N = min(30, len(fut_raw))

        # Compute covar_missing the same way _select_tft_decoder_features does
        pd_active = fut_raw['predispatch_active'].values
        pd7_avail = fut_raw.get('pd7_available', pd.Series(0.0, index=fut_raw.index)).values
        pd7_rrp_v = fut_raw.get('pd7_rrp', pd.Series(0.0, index=fut_raw.index)).values
        combined_avail = np.maximum(pd_active, ((pd7_avail > 0) & (pd7_rrp_v != 0)).astype(float))
        covar_missing = 1.0 - combined_avail

        pd_rrp_v     = fut_raw['pd_rrp'].values           # debiased PREDISPATCH
        pd_rrp_raw_v = fut_raw.get('pd_rrp_raw', pd.Series(np.nan, index=fut_raw.index)).values
        combined_price = np.where(pd_active > 0, pd_rrp_v, pd7_rrp_v)

        print(f"\n{SEP}")
        print("TFT DEBUG — First 30 decoder steps (15h): debiaser + covariate inputs vs outputs")
        hdr = (f"  {'Local time':>19}  {'PD?':>3}  {'miss':>4}"
               f"  {'raw_PD':>7}  {'deb_PD':>7}  {'ratio':>6}  {'PD7':>7}  {'→model':>7}"
               f"  {'q30':>7}  {'q50':>7}  {'q70':>7}  {'AmberAPF':>9}")
        print(hdr)
        print(f"  {'─'*19}  {'─'*3}  {'─'*4}"
              f"  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*7}"
              f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*9}")
        for i in range(N):
            ts_utc = fut_raw.index[i]
            ts_loc = ts_utc.tz_convert(tz_local).strftime('%m-%d %H:%M')
            pd_a   = int(pd_active[i])
            c_miss = int(covar_missing[i])
            raw_p  = pd_rrp_raw_v[i]
            deb_p  = pd_rrp_v[i]
            pd7_p  = pd7_rrp_v[i]
            comb   = combined_price[i]
            q30    = preds_raw[i, 0]
            q50    = preds_raw[i, 1]
            q70    = preds_raw[i, 2]
            apf    = amber_apf.get(ts_utc.floor('30min'), float('nan'))
            apf_s  = f"{apf:>9.1f}" if not np.isnan(apf) else f"{'—':>9}"
            if not np.isnan(raw_p) and raw_p != 0:
                ratio_s = f"{deb_p/raw_p:>6.2f}"
            else:
                ratio_s = f"{'—':>6}"
            raw_s  = f"{raw_p:>7.1f}" if not np.isnan(raw_p) else f"{'—':>7}"
            print(f"  {ts_loc:>19}  {pd_a:>3}  {c_miss:>4}"
                  f"  {raw_s}  {deb_p:>7.1f}  {ratio_s}  {pd7_p:>7.1f}  {comb:>7.1f}"
                  f"  {q30:>7.1f}  {q50:>7.1f}  {q70:>7.1f}  {apf_s}")

        # Debiaser summary over active PD steps
        active_mask = (pd_active[:N] > 0) & (~np.isnan(pd_rrp_raw_v[:N])) & (pd_rrp_raw_v[:N] != 0)
        if active_mask.sum() > 0:
            mean_ratio = (pd_rrp_v[:N][active_mask] / pd_rrp_raw_v[:N][active_mask]).mean()
            mean_raw   = pd_rrp_raw_v[:N][active_mask].mean()
            mean_deb   = pd_rrp_v[:N][active_mask].mean()
            print(f"\n  Debiaser (PD-active steps): raw={mean_raw:.1f}  debiased={mean_deb:.1f}"
                  f"  mean_ratio={mean_ratio:.2f}  compression={(1-mean_ratio)*100:+.0f}%")
        n_pd  = int(pd_active[:N].sum())
        n_miss = int(covar_missing[:N].sum())
        print(f"  Steps 0–{N-1}: PD active={n_pd}/{N}  covar_missing={n_miss}/{N}")

    # ── Sparse decoder table (full 72h, every 2h)
    print(f"\n{SEP}")
    print("TFT DEBUG — Decoder covariates vs TFT output (next 72h, every 2h)")
    print(SEP)
    print(f"  {'Time (UTC)':>22}  {'PD_RRP $/MWh':>13}  {'TFT q30':>8}  {'TFT q50':>8}  {'TFT q70':>8}  {'delta q50-PD':>13}")
    print(f"  {'─'*22}  {'─'*13}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*13}")

    n = len(dec_index)
    show_idx = sorted(set([0, 1] + list(range(0, n, 4)) + [n - 1]))
    for i in show_idx:
        ts = dec_index[i]
        pd_rrp = float(dec_pd_rrp_raw.iloc[i]) if i < len(dec_pd_rrp_raw) else float('nan')
        q30 = preds_raw[i, 0]
        q50 = preds_raw[i, 1]
        q70 = preds_raw[i, 2]
        delta = q50 - pd_rrp
        pd_str = f"{pd_rrp:>13.2f}" if pd_rrp != 0.0 else f"{'(zero)':>13}"
        print(f"  {str(ts):>22}  {pd_str}  {q30:>8.2f}  {q50:>8.2f}  {q70:>8.2f}  {delta:>+13.2f}")

    q50_vals = preds_raw[:, 1]
    pd_vals = dec_pd_rrp_raw.values[:len(q50_vals)]
    nonzero = pd_vals != 0
    if nonzero.sum() > 0:
        mean_delta = (q50_vals[nonzero] - pd_vals[nonzero]).mean()
        print(f"\n  Mean (TFT q50 − PD_RRP) over non-zero PD steps: {mean_delta:+.2f} $/MWh")
    print(f"  TFT q50 range: {q50_vals.min():.1f} → {q50_vals.max():.1f} $/MWh")
    print(SEP + "\n")


def _execute_tactical_prediction():
    """
    Tier 1 tactical LightGBM: calibrated 0–60 min forecast at 5-min resolution.
    Returns dict with 'p5min_price', 'p5min_price_q05', 'p5min_price_q95' DataFrames
    (wholesale price in $/kWh, ready for apply_tariffs_to_forecast).
    """
    model_dir = ROOT / "models" / "lgbm_tactical"

    if not (model_dir / "lgbm_q50.pkl").exists():
        logging.warning("Tactical LightGBM models not found; skipping Tier 1 forecast.")
        return {}

    logging.info("--- Executing Tier 1 Tactical LightGBM Inference ---")

    try:
        q05_model = joblib.load(model_dir / "lgbm_q05.pkl")
        q50_model = joblib.load(model_dir / "lgbm_q50.pkl")
        q95_model = joblib.load(model_dir / "lgbm_q95.pkl")
        with open(model_dir / "conformal_deltas.json") as f:
            conformal = json.load(f)
    except Exception as e:
        logging.error(f"Tactical model load failed: {e}")
        return {}

    # ── Query data ────────────────────────────────────────────────────────────
    now_utc = datetime.now(pytz.UTC)
    t_4h = (now_utc - timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%SZ')
    t_now = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')

    client = InfluxDBClient(**CONFIG['influxdb'])
    try:
        p5min_res = client.query(
            f'SELECT rrp FROM "rp_5m"."aemo_p5min_forecast"'
            f' WHERE region=\'SA1\' AND time >= \'{t_4h}\' AND time <= \'{t_now}\''
            f' GROUP BY run_time'
        )
        act_res = client.query(
            f'SELECT price, total_demand FROM "rp_5m"."aemo_dispatch_sa1_5m"'
            f' WHERE time >= \'{t_4h}\' AND time <= \'{t_now}\''
        )
        pv_res = client.query(
            f'SELECT mean_value FROM "rp_5m"."power_pv_5m"'
            f' WHERE time >= \'{t_4h}\' AND time <= \'{t_now}\''
        )
    finally:
        client.close()

    # ── Parse P5MIN series (grouped by run_time tag) ─────────────────────────
    p5min_runs = {}
    for key in p5min_res.keys():
        run_time_str = key[1].get('run_time') if len(key) > 1 else None
        if not run_time_str:
            continue
        rows = sorted(list(p5min_res[key]), key=lambda r: r['time'])
        if len(rows) < 12:
            continue
        rt = pd.Timestamp(run_time_str, tz='UTC')
        p5min_runs[rt] = [float(r['rrp']) for r in rows[:12]]

    if not p5min_runs:
        logging.error("No P5MIN runs found in InfluxDB; Tier 1 forecast aborted.")
        return {}

    latest_run_time = max(p5min_runs.keys())
    p5min_rrp = p5min_runs[latest_run_time]  # [12] $/MWh

    prev_run_time = latest_run_time - pd.Timedelta(minutes=5)
    p5min_h0_prev = p5min_runs[prev_run_time][0] if prev_run_time in p5min_runs else np.nan

    # ── Parse actual 5m prices ────────────────────────────────────────────────
    act_rows = list(act_res.get_points()) if act_res else []
    if act_rows:
        act_df = pd.DataFrame(act_rows)
        act_df['time'] = pd.to_datetime(act_df['time'], utc=True)
        act_df = act_df.set_index('time').rename(columns={'price': 'rrp'}).sort_index()
    else:
        act_df = pd.DataFrame(columns=['rrp', 'total_demand'])
        act_df.index = pd.DatetimeIndex([], tz='UTC')

    # ── Parse PV ─────────────────────────────────────────────────────────────
    pv_rows = list(pv_res.get_points()) if pv_res else []
    if pv_rows:
        pv_df = pd.DataFrame(pv_rows)
        pv_df['time'] = pd.to_datetime(pv_df['time'], utc=True)
        pv_df = pv_df.set_index('time').sort_index()
        pv_series = pv_df['mean_value']
    else:
        pv_series = pd.Series(dtype=float)

    # ── Build scalar features at latest_run_time ─────────────────────────────
    def _asof(series, ts, default=0.0):
        try:
            v = series.asof(ts) if len(series) > 0 else np.nan
            return float(v) if pd.notna(v) else default
        except Exception:
            return default

    t1 = latest_run_time - pd.Timedelta(minutes=5)
    t2 = latest_run_time - pd.Timedelta(minutes=10)
    t6 = latest_run_time - pd.Timedelta(minutes=30)

    rrp_series = act_df['rrp'] if 'rrp' in act_df.columns else pd.Series(dtype=float)
    actual_t1  = _asof(rrp_series, t1)
    actual_t2  = _asof(rrp_series, t2)
    actual_t6  = _asof(rrp_series, t6)

    divergence_t1 = (actual_t1 - p5min_h0_prev) if not np.isnan(p5min_h0_prev) else 0.0

    if len(rrp_series) >= 6:
        rrp_to_t1 = rrp_series.loc[:t1]
        rolling_1h_std = float(rrp_to_t1.tail(12).std()) if len(rrp_to_t1) >= 6 else 0.0
        rolling_3h_max = float(rrp_to_t1.tail(36).max()) if len(rrp_to_t1) >= 6 else max(p5min_rrp)
    else:
        rolling_1h_std = 0.0
        rolling_3h_max = max(p5min_rrp)

    td_t1  = _asof(act_df['total_demand'] if 'total_demand' in act_df.columns else pd.Series(dtype=float), t1)
    pv_t1  = _asof(pv_series, t1)
    residual_demand_t1 = td_t1 - pv_t1  # units match training (MW - W ≈ MW)

    # Time features at run_time (Brisbane AEST, matching training build_tactical_dataset.py)
    brisbane_tz = pytz.timezone("Australia/Brisbane")
    rt_bne = latest_run_time.astimezone(brisbane_tz)
    hour_frac = rt_bne.hour + rt_bne.minute / 60.0
    hour_sin = float(np.sin(2 * np.pi * hour_frac / 24.0))
    hour_cos = float(np.cos(2 * np.pi * hour_frac / 24.0))
    dow_sin  = float(np.sin(2 * np.pi * rt_bne.weekday() / 7.0))
    dow_cos  = float(np.cos(2 * np.pi * rt_bne.weekday() / 7.0))

    intervals = pd.date_range(start=latest_run_time, periods=12, freq='5min', tz='UTC')
    tariffed_curve = tariffed_price_frame_from_wholesale_mwh(
        pd.Series(np.asarray(p5min_rrp, dtype=np.float64), index=intervals),
        timezone=CONFIG['timezone'],
        general_tariff_map=GENERAL_TARIFF_MAP,
        feed_in_tariff_map=FEED_IN_TARIFF_MAP,
        network_loss_factor=NETWORK_LOSS_FACTOR,
        gst_rate=CONFIG['gst_rate'],
    )
    import_curve = tariffed_curve['general_price_mwh'].to_numpy(dtype=np.float32, copy=False)
    export_curve = tariffed_curve['feed_in_price_mwh'].to_numpy(dtype=np.float32, copy=False)

    feature_dict = build_tier1_feature_dict(
        latest_run_time,
        p5min_rrp,
        p5min_h0_prev,
        act_df,
        pv_series,
    )
    X_long = build_tier1_long_matrix(q50_model, feature_dict, 12)

    # ── Predict ───────────────────────────────────────────────────────────────
    raw_q05 = q05_model.predict(X_long)
    raw_q50 = q50_model.predict(X_long)
    raw_q95 = q95_model.predict(X_long)

    # ── Apply conformal calibration per horizon (regime = per-horizon p5min) ──
    deltas       = conformal['deltas']
    spike_thresh = conformal['spike_threshold']
    low_thresh   = conformal['low_threshold']

    cal_q05 = np.empty(12, dtype=float)
    cal_q95 = np.empty(12, dtype=float)
    for h_idx in range(12):
        p5h = p5min_rrp[h_idx]
        if p5h >= spike_thresh:
            regime = 'spike'
        elif p5h < low_thresh or residual_demand_t1 < 0:
            regime = 'low'
        else:
            regime = 'normal'
        d = deltas[regime]
        cal_q05[h_idx] = raw_q05[h_idx] - d['delta_q05']
        cal_q95[h_idx] = raw_q95[h_idx] + d['delta_q95']

    logging.info(
        f"Tier 1 tactical: run_time={latest_run_time.isoformat()}, "
        f"h0 q50={raw_q50[0]:.1f} $/MWh, "
        f"h0 p5min={p5min_rrp[0]:.1f} $/MWh"
    )

    # ── Build output DataFrames ($/kWh for tariff compatibility) ─────────────
    return {
        'p5min_price':     pd.DataFrame({'wholesale_price': raw_q50 / 1000.0}, index=intervals),
        'p5min_price_q05': pd.DataFrame({'wholesale_price': cal_q05 / 1000.0}, index=intervals),
        'p5min_price_q95': pd.DataFrame({'wholesale_price': cal_q95 / 1000.0}, index=intervals),
    }


def _apply_pd_debiaser(fut_df, start_t, historical_df=None):
    """
    Applies the unified PREDISPATCH debiaser (LGBM) to the first 56 decoder steps.

    The debiaser takes prob_spike as a feature and smoothly transitions from
    compression-corrected output (low prob_spike) toward raw PREDISPATCH (high prob_spike).
    No binary routing threshold — the model learned the appropriate correction level.
    Falls back to prob_spike=0.0 (full debiasing) if classifier unavailable.
    """
    model_path = Path(__file__).resolve().parent / "models" / "pd_debiaser" / "lgbm_final.pkl"
    if not model_path.exists():
        return fut_df

    try:
        df = fut_df.head(56).copy()
        if df.empty:
            return fut_df

        run_time = start_t - pd.Timedelta(minutes=30)

        # ── Spike classifier: compute prob_spike for debiaser feature ─────────
        prob_spike = 0.0  # fallback: treat as non-spike, apply full debiasing
        clf_path = Path(__file__).resolve().parent / "models" / "spike_classifier" / "lgbm_spike_clf.pkl"
        if clf_path.exists() and historical_df is not None and not historical_df.empty:
            try:
                with open(clf_path, "rb") as f:
                    clf_bundle = joblib.load(f)
                clf_model    = clf_bundle["model"]
                clf_features = clf_bundle["features"]

                rrp_col  = "rrp" if "rrp" in historical_df.columns else None
                hist_rrp = historical_df[rrp_col].sort_index() if rrp_col else pd.Series(dtype=float)

                def _get_lag(rt, offset):
                    target = rt - offset
                    idx = hist_rrp.index.get_indexer([target], method="nearest")
                    if idx[0] >= 0:
                        actual_t = hist_rrp.index[idx[0]]
                        if abs((actual_t - target).total_seconds()) <= 900:
                            return float(hist_rrp.iloc[idx[0]])
                    return np.nan

                rt30           = pd.Timedelta(minutes=30)
                pd_dec         = df["pd_rrp"].dropna()
                pd_demand_dec  = df["pd_demand"].dropna() if "pd_demand" in df else pd.Series(dtype=float)
                rt_brisbane    = run_time.tz_convert("Australia/Brisbane")

                clf_x = {
                    "pd_rrp_h0":            float(pd_dec.iloc[0]) if len(pd_dec) > 0 else 0.0,
                    "pd_rrp_max":           float(pd_dec.max())    if len(pd_dec) > 0 else 0.0,
                    "pd_rrp_p90":           float(pd_dec.quantile(0.9)) if len(pd_dec) > 0 else 0.0,
                    "pd_demand_max":        float(pd_demand_dec.max()) if len(pd_demand_dec) > 0 else 0.0,
                    "pd_net_interchange_h0": float(df["pd_net_interchange"].iloc[0])
                                             if "pd_net_interchange" in df and not df["pd_net_interchange"].isna().all() else 0.0,
                    "actual_rrp_lag1":      _get_lag(run_time, 1 * rt30),
                    "actual_rrp_lag2":      _get_lag(run_time, 2 * rt30),
                    "actual_rrp_lag4":      _get_lag(run_time, 4 * rt30),
                    "actual_rrp_lag8":      _get_lag(run_time, 8 * rt30),
                    "actual_rrp_max_6h":    float(hist_rrp[hist_rrp.index >= run_time - pd.Timedelta(hours=6)].max())  if len(hist_rrp) > 0 else np.nan,
                    "actual_rrp_max_24h":   float(hist_rrp[hist_rrp.index >= run_time - pd.Timedelta(hours=24)].max()) if len(hist_rrp) > 0 else np.nan,
                    "hour_sin":   float(np.sin(2 * np.pi * rt_brisbane.hour / 24)),
                    "hour_cos":   float(np.cos(2 * np.pi * rt_brisbane.hour / 24)),
                    "dow_sin":    float(np.sin(2 * np.pi * rt_brisbane.dayofweek / 7)),
                    "dow_cos":    float(np.cos(2 * np.pi * rt_brisbane.dayofweek / 7)),
                    "month_sin":  float(np.sin(2 * np.pi * (rt_brisbane.month - 1) / 12)),
                    "month_cos":  float(np.cos(2 * np.pi * (rt_brisbane.month - 1) / 12)),
                }
                X_clf      = np.array([[clf_x.get(f, 0.0) for f in clf_features]], dtype=np.float32)
                prob_spike = float(clf_model.predict_proba(X_clf)[0, 1])
                logging.info(f"Spike classifier: prob_spike={prob_spike:.3f}")
            except Exception as clf_err:
                logging.warning(f"Spike classifier failed ({clf_err}); using prob_spike=0.0")

        # ── Apply unified debiaser (prob_spike is the 11th feature) ──────────
        df["horizon_steps"] = ((df.index - run_time).total_seconds() / 1800).astype(np.float32)
        t = df.index.tz_convert("Australia/Brisbane")
        df["hour_sin"]  = np.sin(2 * np.pi * t.hour / 24)
        df["hour_cos"]  = np.cos(2 * np.pi * t.hour / 24)
        df["dow_sin"]   = np.sin(2 * np.pi * t.dayofweek / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * t.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * (t.month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (t.month - 1) / 12)
        df["prob_spike"] = np.float32(prob_spike)

        feats = [
            "pd_rrp", "pd_demand", "pd_net_interchange", "horizon_steps",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
            "prob_spike",
        ]
        model    = joblib.load(model_path)
        X        = df[feats].astype(np.float32)
        debiased = model.predict(X).astype(np.float32)
        fut_df.loc[df.index, "pd_rrp"] = debiased
        logging.info(f"Applied unified PD debiaser: {len(debiased)} steps "
                     f"(prob_spike={prob_spike:.3f})")

    except Exception as e:
        logging.warning(f"PD debiaser failed: {e}")

    return fut_df


def _execute_tft_prediction(historical_df, future_covariates_df):
    """
    TFT Inference Worker: Parallel branch for Run 010.
    """
    import torch
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
        from train_tft_price import TFTPriceModel
    except ImportError:
        logging.warning("TFTPriceModel could not be imported. TFT shadow mode disabled.")
        return {}

    logging.info("--- Executing TFT Parallel Inference (Phase 7 decoder) ---")

    # ── 1. Load Model and Scalers
    paths = CONFIG['paths']
    try:
        model_path = Path(paths['tft_price_model'])
        if not model_path.exists():
            logging.error(f"TFT Model not found at {model_path}. Shadow mode aborted.")
            return {}

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        meta = checkpoint.get("meta", {})
        m_cfg = checkpoint.get("model_config", {})
        dec_feature_names = meta.get("dec_features") or meta.get("dec_feature_names")
        
        # Load the model class with correct architecture from checkpoint
        model = TFTPriceModel(
            n_enc=m_cfg.get("n_enc", 20),
            n_dec=m_cfg.get("n_dec", 13),
            d_model=m_cfg.get("d_model", 64),
            n_heads=m_cfg.get("n_heads", 4),
            n_lstm_layers=m_cfg.get("n_layers", 2),
            dropout=m_cfg.get("dropout", 0.1)
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        with open(paths['tft_price_scalers'], 'rb') as f:
            scalers = pickle.load(f)
            # Unified scalers.pkl is a dict: feature_name -> (QuantileTransformer or "log")
    except Exception as e:
        logging.error(f"Failed to load TFT model or scalers: {e}")
        return {}

    # ── 2. Prepare Encoder Input (Last 96 steps = 48h)
    hist = historical_df.tail(96).copy()
    if len(hist) < 96:
        logging.warning(f"Insufficient history for TFT (need 96 steps, got {len(hist)}). Bfilling.")
        full_idx = pd.date_range(end=hist.index.max(), periods=96, freq='30min')
        hist = hist.reindex(full_idx).bfill()

    # Base features: rename to match build_training_dataset.py feature names
    hist.rename(columns={
        'aemo_price_sa1': 'rrp',
        'temperature_adelaide': 'temp',
        'humidity_adelaide': 'humidity',
        'wind_speed_adelaide': 'wind_speed',
        'net_interchange_sa1': 'net_interchange',
        'total_demand_sa1': 'total_demand'
    }, inplace=True)
    # get_historical_data() stores aemo_price_sa1 in $/kWh (divides by 1000).
    # Training pipeline (export_parquet.py) uses $/MWh with LOG_SCALE_FACTOR=60.
    # Convert here to match training units before any feature computation.
    hist['rrp'] = hist['rrp'] * 1000.0  # $/kWh → $/MWh

    # ── 3. High-Frequency Features (5m)
    client = InfluxDBClient(**CONFIG['influxdb'])
    hist_5m = get_5m_price_history(client, hist.index.min() - timedelta(minutes=60), hist.index.max())
    client.close()

    if not hist_5m.empty:
        def get_agg(t):
            win = hist_5m.loc[t - timedelta(minutes=25):t]
            if win.empty: return 0.0, 0.0, 0.0, 0.0
            std = float(win.std()) if len(win) > 1 else 0.0
            return float(win.max()), std, float((win > 150).sum()), std

        aggs = [get_agg(t) for t in hist.index]
        hist['rrp_5m_max'] = [a[0] for a in aggs]
        hist['rrp_5m_std'] = [a[1] for a in aggs]
        hist['rrp_persistence'] = [a[2] for a in aggs]
        hist['rrp_volatility_30m'] = [a[3] for a in aggs]
        hist['rrp_5m_missing'] = 0
    else:
        for c in ['rrp_5m_max', 'rrp_5m_std', 'rrp_persistence', 'rrp_volatility_30m']: hist[c] = 0.0
        hist['rrp_5m_missing'] = 1

    hist = add_tft_regime_features(hist, rrp_col='rrp')
    hist = pd.concat([hist, time_sin_cos(hist.index)], axis=1)

    ENC_CONT = ["rrp", "total_demand", "net_interchange", "power_load", "power_pv", 
                "temp", "humidity", "wind_speed", "rrp_5m_max", "rrp_5m_std", 
                "rrp_persistence", "rrp_volatility_30m", "rrp_log_momentum"]
    
    # Scaling helper matching build_training_dataset.py
    log_scale = meta.get("log_scale_factor", 60.0)
    def transform_val(val, feat_name):
        s = scalers.get(feat_name)
        if s is None: return val
        if s == "log":
            return np.sign(val) * np.log1p(np.abs(val) / log_scale)
        # Handle single float or array
        if np.isscalar(val):
            return float(s.transform(np.array([[val]]))[0, 0])
        return s.transform(val.values.reshape(-1, 1)).flatten()

    # Snapshot raw encoder values before scaling (for debug table)
    _enc_rrp_raw = hist["rrp"].copy()           # $/MWh
    _enc_load_raw = hist.get("power_load", pd.Series(dtype=float)).copy()
    _enc_5m_missing = hist["rrp_5m_missing"].copy()

    # Fill any data gaps before scaling — the LGBM path uses ffill/dropna at
    # data load time, but the TFT path builds hist from tail(96) of raw data
    # which can have brief InfluxDB gaps at the most recent intervals.
    hist[ENC_CONT] = hist[ENC_CONT].ffill().bfill()

    # Apply enc scalers
    for feat in ENC_CONT:
        hist[feat] = transform_val(hist[feat], feat)

    TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    X_enc = hist[ENC_CONT + TIME_COLS + ["rrp_5m_missing"]].values.astype(np.float32)

    # ── 4. Prepare Decoder Input (Next 144 steps = 72h)
    # IMPORTANT: Ensure decoder starts from forecast_start_time (current window)
    # future_covariates_df passed here is actually combined_covariates (hist + future)
    # We find the first 30m boundary >= now
    now_utc = datetime.now(pytz.UTC)
    start_t = now_utc.replace(minute=30 if now_utc.minute >= 30 else 0, second=0, microsecond=0)
    
    fut = future_covariates_df[future_covariates_df.index >= start_t].head(144).copy()
    if fut.empty:
        logging.error("TFT decoder input is empty! Check future_covariates_df alignment.")
        return {}

    # TFT decoder price covariates: replicate training structure
    #   Steps 0-55  (0.5h-28h): PREDISPATCH rrp for SA1/VIC1/NSW1
    #   Steps 56-143 (28-72h): PD7Day rrp for SA1 only; VIC1/NSW1 = 0 (matches training)
    #
    # Primary source: InfluxDB (reliable, ingested every 30 min).
    # Supplement: AEMO viz API short-term price forecast (more recent but unreliable).
    forecast_run_time = start_t - pd.Timedelta(minutes=30)
    client = InfluxDBClient(**CONFIG['influxdb'])
    try:
        influx_pd_prices = _get_influx_pd_prices(client, fut.index.min(), fut.index.max())
        pd7_df, pd7_run_time = _get_influx_latest_pd7day_prices(client, forecast_run_time, fut.index)
        sdo_df = _get_influx_sdo_demand(client, fut.index.min(), fut.index.max())
    finally:
        client.close()

    if not influx_pd_prices.empty:
        fut = fut.join(influx_pd_prices, how='left')
    else:
        for c in ['pd_rrp', 'vic1_pd_rrp', 'nsw1_pd_rrp']:
            fut[c] = np.nan

    if not pd7_df.empty:
        fut = fut.join(pd7_df, how='left')
    else:
        fut['pd7_rrp'] = np.nan

    # Supplement with viz API for short-horizon steps (more current if available)
    viz_prices = _get_aemo_short_term_price_forecast()
    if not viz_prices.empty:
        viz_prices = viz_prices * 1000.0  # $/kWh → $/MWh
        for col in ['pd_rrp', 'vic1_pd_rrp', 'nsw1_pd_rrp']:
            if col in viz_prices.columns:
                # Override with viz API where both exist (more recent data)
                fut[col] = viz_prices[col].reindex(fut.index).combine_first(fut.get(col, pd.Series(dtype=float)))

    fut.rename(columns={
        'total_demand_sa1': 'pd_demand',
        'net_interchange_sa1': 'pd_net_interchange'
    }, inplace=True)

    # Ensure columns exist even if all sources failed
    for c in ['pd_rrp', 'pd_demand', 'pd_net_interchange', 'vic1_pd_rrp', 'nsw1_pd_rrp', 'pd7_rrp']:
        if c not in fut.columns:
            fut[c] = np.nan

    predispatch_active = fut['pd_rrp'].notna().astype(np.float32)
    predispatch_active.iloc[56:] = 0.0

    # Capture raw (pre-debiaser) PREDISPATCH for diagnostic comparison.
    _pd_rrp_raw_pre_debias = fut['pd_rrp'].copy() if _DEBUG_TFT else None

    # ── 3.5 Debias PREDISPATCH (Run 011+ contract)
    fut = _apply_pd_debiaser(fut, start_t, historical_df=hist)

    # Match training layout exactly: pd_rrp/VIC1/NSW1 are PREDISPATCH-only.
    fut.loc[fut.index[56:], 'pd_rrp'] = 0.0
    fut.loc[fut.index[56:], 'vic1_pd_rrp'] = 0.0
    fut.loc[fut.index[56:], 'nsw1_pd_rrp'] = 0.0

    # Fill remaining NaNs after availability flags are captured.
    fut['pd_rrp'] = fut['pd_rrp'].fillna(0.0)
    fut['pd_demand'] = fut['pd_demand'].fillna(0.0)
    fut['pd_net_interchange'] = fut['pd_net_interchange'].fillna(0.0)
    fut['vic1_pd_rrp'] = fut['vic1_pd_rrp'].fillna(0.0)
    fut['nsw1_pd_rrp'] = fut['nsw1_pd_rrp'].fillna(0.0)
    fut['pd7_rrp'] = fut['pd7_rrp'].fillna(0.0)

    # SevenDayOutlook: sd_demand + sd_net_interchange (all 144 steps, matches training)
    if not sdo_df.empty:
        fut = fut.join(sdo_df, how='left')
        fut['sd_demand'] = fut['sd_demand'].fillna(0.0)
        fut['sd_net_interchange'] = fut['sd_net_interchange'].fillna(0.0)
    else:
        logging.warning("TFT: SDO data unavailable — sd_demand/sd_net_interchange 0-filled")
        fut['sd_demand'] = 0.0
        fut['sd_net_interchange'] = 0.0

    fut = pd.concat([fut, time_sin_cos(fut.index)], axis=1)
    fut['horizon_norm'] = np.arange(len(fut), dtype=np.float32) / max(len(fut) - 1, 1)
    fut['predispatch_active'] = predispatch_active.values
    if pd7_run_time is not None:
        pd7_gen_hour = pd7_run_time.tz_convert("Australia/Brisbane").hour / 23.0
        fut['pd7_generation_hour'] = np.float32(pd7_gen_hour)
        fut['pd7_available'] = np.float32(1.0)
    else:
        logging.warning("TFT: PD7Day run unavailable — pd7_rrp 0-filled, pd7_available=0")
        fut['pd7_generation_hour'] = np.float32(0.0)
        fut['pd7_available'] = np.float32(0.0)

    # Snapshot combined decoder price signal before scaling for debug/visualisation.
    _dec_price_raw = pd.Series(
        np.where(fut["predispatch_active"].values > 0, fut["pd_rrp"].values, fut["pd7_rrp"].values),
        index=fut.index,
    )

    # Capture unscaled decoder frame for the debug table (all prices in $/MWh).
    if _DEBUG_TFT:
        _fut_raw_diag = fut[['pd_rrp', 'pd7_rrp', 'pd_demand', 'sd_demand',
                              'predispatch_active', 'pd7_available']].copy()
        _fut_raw_diag['pd_rrp_raw'] = _pd_rrp_raw_pre_debias.reindex(_fut_raw_diag.index)
    else:
        _fut_raw_diag = None

    DEC_CONT = ["pd_rrp", "pd_demand", "pd_net_interchange", "vic1_pd_rrp", "nsw1_pd_rrp",
                "pd7_rrp", "sd_demand", "sd_net_interchange"]
    # Apply dec scalers
    for feat in DEC_CONT:
        fut[feat] = transform_val(fut[feat], feat)

    X_dec, active_dec_features = _select_tft_decoder_features(
        fut,
        dec_feature_names,
    )

    # ── 5. Inference
    with torch.no_grad():
        t_enc = torch.tensor(X_enc).unsqueeze(0)
        t_dec = torch.tensor(X_dec).unsqueeze(0)
        preds_norm = model(t_enc, t_dec).squeeze(0).numpy()
    
    preds_norm = np.sort(preds_norm, axis=-1)
    
    # Inverse scaling for 'target_rrp' (which is log-scaled in Run 010)
    # qt_targ was the key in my previous code, now it's from scalers dictionary
    s_targ = scalers.get("target_rrp", "log")
    if s_targ == "log":
        preds_raw = np.sign(preds_norm) * log_scale * (np.exp(np.abs(preds_norm)) - 1.0)
    else:
        # QuantileTransformer inverse
        preds_raw = s_targ.inverse_transform(preds_norm)
    
    # ── 6. Format Return
    # Return as { 'tft_price': df_q50, 'tft_price_q30': df_q30, 'tft_price_q70': df_q70 }
    # CONVERSION: Convert $/MWh to $/kWh (divide by 1000)
    res = {}
    quants = ['tft_price_q30', 'tft_price', 'tft_price_q70']
    for i, q_name in enumerate(quants):
        # Create DataFrame matching the structure of standard price models
        # Use [float(x) for x in ...] for JSON safety
        df_q = pd.DataFrame({
            'wholesale_price': [float(x) / 1000.0 for x in preds_raw[:, i]]
        }, index=fut.index)
        
        # Add basic tariff columns if they exist in covariates (for easy integration)
        # This allows apply_tariffs_to_forecasts to work if called.
        df_q['general_tariff'] = fut.get('general_tariff', 0.0)
        df_q['feed_in_tariff'] = fut.get('feed_in_tariff', 0.0)
        
        res[q_name] = df_q

    # Publish the merged PREDISPATCH/PD7Day prices as a separate HA entity so they
    # can be visualised alongside the TFT quantile forecasts for debugging.
    res['aemo_price_forecast'] = pd.DataFrame(
        {'wholesale_price': _dec_price_raw.values / 1000.0},
        index=fut.index,
    )

    if _DEBUG_TFT:
        _print_tft_debug(
            enc_rrp_raw=_enc_rrp_raw,
            enc_load_raw=_enc_load_raw,
            enc_5m_missing=_enc_5m_missing,
            dec_pd_rrp_raw=_dec_price_raw,
            dec_index=fut.index,
            preds_raw=preds_raw,
            fut_raw=_fut_raw_diag,
        )

    logging.info(
        f"TFT Parallel Inference complete. Steps: {len(preds_raw)} "
        f"(decoder_width={len(active_dec_features)})"
    )
    return res


def _execute_pd_direct_prediction(historical_df, future_covariates_df):
    """PD-direct strategic forecast: no-ML baseline (Phase α-prime, 2026-05-05 pivot).

    Produces a 30-min, 144-step ($/kWh) strategic price forecast using:
      - debiased PREDISPATCH for steps 0–55 (uses the live `_apply_pd_debiaser` on
        InfluxDB-sourced raw PREDISPATCH — the same final-model debiaser the rest of
        forecast.py uses)
      - PD7Day capped at $300/MWh for any tail steps where it's available
      - Hour-of-day seasonal mean (from the eval module) as a final fallback

    Returned dict mirrors the TFT result shape so the publish layer can treat them
    interchangeably:
        {
          'pd_direct_price':     df_q50,
          'pd_direct_price_q30': df_q50,   # degenerate q30 = q50 (no native quantile
          'pd_direct_price_q70': df_q50,   # spread; horizon-stratified bands land in
                                           # Phase α-prime Step 3)
        }

    Live values will be very close to but not identical to the eval framework's OOF
    debiased values — the eval correctly uses OOF for leakage-free historical comparison;
    live correctly uses the final model trained on all data. See
    docs/pd_direct_publish_rfc.md for the expected delta and audit guidance.
    """
    # Lazy import so the eval module's parquet load only happens when this path runs.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval.pd_direct_baseline import (
        RESIDUAL_BANDS_PARQUET,
        apply_pd_residual_bands,
        load_pd_direct_context,
        PD7DAY_CAP_DEFAULT,
    )

    try:
        # ── 1. Build start_t and 144-step decoder window (mirror TFT path semantics)
        now_utc = datetime.now(pytz.UTC)
        start_t = now_utc.replace(minute=30 if now_utc.minute >= 30 else 0,
                                  second=0, microsecond=0)
        fut = future_covariates_df[future_covariates_df.index >= start_t].head(144).copy()
        if fut.empty:
            logging.warning("PD-direct: future_covariates_df empty after start_t slice; "
                            "skipping.")
            return {}
        if len(fut) < 144:
            logging.warning(f"PD-direct: only {len(fut)} decoder steps available "
                            f"(expected 144); short curve will be published.")

        # ── 2. Pull raw PREDISPATCH + PD7Day from InfluxDB (same queries TFT uses)
        forecast_run_time = start_t - pd.Timedelta(minutes=30)
        client = InfluxDBClient(**CONFIG['influxdb'])
        try:
            influx_pd_prices = _get_influx_pd_prices(
                client, fut.index.min(), fut.index.max())
            pd7_df, pd7_run_time = _get_influx_latest_pd7day_prices(
                client, forecast_run_time, fut.index)
        finally:
            client.close()

        if not influx_pd_prices.empty:
            fut = fut.join(influx_pd_prices, how='left')
        if not pd7_df.empty:
            fut = fut.join(pd7_df, how='left')

        # The debiaser expects 'pd_demand' / 'pd_net_interchange' columns. The TFT path
        # renames 'total_demand_sa1'/'net_interchange_sa1' from PREDISPATCH; pull those
        # in if present, otherwise leave NaN — debiaser handles missing covariates
        # gracefully (returns input unchanged on exception).
        fut.rename(columns={
            'total_demand_sa1': 'pd_demand',
            'net_interchange_sa1': 'pd_net_interchange',
        }, inplace=True)
        for c in ['pd_rrp', 'pd_demand', 'pd_net_interchange', 'pd7_rrp']:
            if c not in fut.columns:
                fut[c] = np.nan

        # ── 3. Debias PREDISPATCH steps 0–55 (modifies fut['pd_rrp'] in place)
        fut = _apply_pd_debiaser(fut, start_t, historical_df=historical_df)

        # ── 4. Build the layered q50 strategic curve in $/MWh
        residual_bands_path = RESIDUAL_BANDS_PARQUET if RESIDUAL_BANDS_PARQUET.exists() else None
        ctx = load_pd_direct_context(residual_bands_path=residual_bands_path)  # seasonal HoD + bands
        out = pd.Series(index=fut.index, dtype=np.float64)

        # Layer 1: debiased PREDISPATCH wherever available (typically steps 0–55)
        mask_pd = fut['pd_rrp'].notna()
        out.loc[mask_pd] = fut.loc[mask_pd, 'pd_rrp'].values.astype(np.float64)

        # Layer 2: PD7Day capped, fills any still-NaN steps (typically 56+)
        if 'pd7_rrp' in fut.columns:
            pd7_capped = np.minimum(
                fut['pd7_rrp'].values.astype(np.float64),
                float(PD7DAY_CAP_DEFAULT),
            )
            still_nan = out.isna() & fut['pd7_rrp'].notna()
            out.loc[still_nan] = pd7_capped[still_nan.values]

        # Layer 3: hour-of-day seasonal mean for any remaining gaps
        if out.isna().any():
            hods = out.index.hour * 2 + (out.index.minute // 30)
            seasonal_vals = ctx.seasonal_hod_table.reindex(hods).to_numpy(dtype=np.float64)
            out = out.where(~out.isna(), pd.Series(seasonal_vals, index=out.index))

        # Final guard: shouldn't happen but don't propagate NaN into HA payload.
        if out.isna().any():
            logging.warning("PD-direct: residual NaNs after all fallback layers; "
                            "filling with $60/MWh.")
            out = out.fillna(60.0)

        # ── 5. Apply empirical residual bands if available, then convert $/MWh → $/kWh.
        q30_mwh, q70_mwh = apply_pd_residual_bands(out, start_t, ctx.residual_bands)
        out_kwh = out / 1000.0
        q30_kwh = q30_mwh / 1000.0
        q70_kwh = q70_mwh / 1000.0
        df_q50 = pd.DataFrame({'wholesale_price': out_kwh.values}, index=fut.index)
        df_q50['general_tariff'] = fut.get('general_tariff', 0.0)
        df_q50['feed_in_tariff'] = fut.get('feed_in_tariff', 0.0)

        df_q30 = pd.DataFrame({'wholesale_price': q30_kwh.values}, index=fut.index)
        df_q30['general_tariff'] = fut.get('general_tariff', 0.0)
        df_q30['feed_in_tariff'] = fut.get('feed_in_tariff', 0.0)
        df_q70 = pd.DataFrame({'wholesale_price': q70_kwh.values}, index=fut.index)
        df_q70['general_tariff'] = fut.get('general_tariff', 0.0)
        df_q70['feed_in_tariff'] = fut.get('feed_in_tariff', 0.0)

        res = {
            'pd_direct_price':     df_q50,
            'pd_direct_price_q30': df_q30,
            'pd_direct_price_q70': df_q70,
        }

        # Diagnostics
        layer_pd  = int(mask_pd.sum())
        layer_p7  = int(((~mask_pd) & fut['pd7_rrp'].notna()).sum()) if 'pd7_rrp' in fut.columns else 0
        layer_seas = 144 - layer_pd - layer_p7
        logging.info(
            f"PD-direct: {len(out_kwh)} steps "
            f"(debiased PD: {layer_pd}, PD7Day: {layer_p7}, seasonal: {layer_seas}); "
            f"q50 range ${out.min():.1f}–${out.max():.1f}/MWh; "
            f"bands={'loaded' if ctx.residual_bands is not None else 'degenerate'}"
        )
        return res

    except Exception as e:
        logging.error(f"PD-direct prediction failed: {e}", exc_info=True)
        return {}


def _time_sin_cos_local(timestamps):
    """Like time_sin_cos() but uses Australia/Adelaide — matches build_load_dataset.py training."""
    t = timestamps.tz_convert("Australia/Adelaide")
    return pd.DataFrame({
        "hour_sin":  np.sin(2 * np.pi * t.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * t.hour / 24),
        "dow_sin":   np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos":   np.cos(2 * np.pi * t.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * (t.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
    }, index=timestamps)


def _execute_tft_load_prediction(historical_df, future_covariates_df):
    """
    TFT Load Inference: shadow branch for household load forecasting.

    Encoder: last 96 steps (48h) of historical load, PV, weather + calendar features.
    Decoder: 144 steps (72h) of weather forecast, PV forecast, calendar features.
    Returns: {'tft_load': df_q50, 'tft_load_q10': df_q10, 'tft_load_q90': df_q90}
    with 'power_load' column in Watts.
    """
    import pickle as _pickle
    import torch as _torch

    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
        from train_tft_load import TFTLoadModel
    except ImportError:
        logging.warning("TFTLoadModel could not be imported. Load TFT shadow mode disabled.")
        return {}

    logging.info("--- Executing TFT Load Shadow Inference ---")

    paths = CONFIG['paths']
    try:
        model_path = Path(paths.get('tft_load_model', 'models/tft_load/checkpoint_best.pt'))
        if not model_path.exists():
            logging.warning(f"TFT Load model not found at {model_path}. Shadow mode skipped.")
            return {}

        ckpt   = _torch.load(model_path, map_location="cpu", weights_only=False)
        m_cfg  = ckpt["model_config"]
        model  = TFTLoadModel(
            n_enc=m_cfg["n_enc"], n_dec=m_cfg["n_dec"],
            n_quantiles=m_cfg.get("n_quantiles", 3),
            d_model=m_cfg.get("d_model", 64),
            n_heads=m_cfg.get("n_heads", 4),
            n_lstm_layers=m_cfg.get("n_lstm_layers", 2),
            dropout=0.0,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        scalers_path = Path(paths.get('tft_load_scalers', 'data/parquet/load_scalers.pkl'))
        with open(scalers_path, "rb") as f:
            scalers = _pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load TFT load model or scalers: {e}")
        return {}

    meta      = ckpt.get("meta", {})
    quantiles = ckpt.get("quantiles", [0.10, 0.50, 0.90])

    ENC_FEATURES = meta.get("enc_feature_names", [
        "power_load", "power_pv", "temp", "humidity", "wind_speed",
        "is_public_holiday", "is_daylight_saving_time",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ])
    DEC_FEATURES = meta.get("dec_feature_names", [
        "temp", "humidity", "wind_speed", "power_pv",
        "is_public_holiday", "is_daylight_saving_time", "horizon_norm",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ])

    # ── 1. Encoder (last 96 steps of history)
    hist = historical_df.tail(96).copy()
    if len(hist) < 96:
        full_idx = pd.date_range(end=hist.index.max(), periods=96, freq="30min")
        hist = hist.reindex(full_idx).bfill().ffill()

    # Rename to match training feature names
    hist = hist.rename(columns={
        "temperature_adelaide": "temp",
        "humidity_adelaide":    "humidity",
        "wind_speed_adelaide":  "wind_speed",
    })
    # power_load in historical_df is already dump-load-corrected (get_historical_data subtracts it)
    # Ensure PV gaps are zero (night = no generation)
    hist["power_pv"] = hist.get("power_pv", pd.Series(0.0, index=hist.index)).fillna(0.0)

    hist = pd.concat([hist, _time_sin_cos_local(hist.index)], axis=1)
    hist = add_time_features(hist)

    # Fill any data gaps before scaling
    for feat in ENC_FEATURES:
        if feat not in hist.columns:
            hist[feat] = 0.0
    hist[ENC_FEATURES] = hist[ENC_FEATURES].ffill().bfill().fillna(0.0)

    def _scale(val, feat):
        sc = scalers.get(feat)
        if sc is None:
            return val
        if np.isscalar(val):
            return float(sc.transform(np.array([[val]]))[0, 0])
        return sc.transform(np.array(val).reshape(-1, 1)).flatten()

    X_enc = np.zeros((96, len(ENC_FEATURES)), dtype=np.float32)
    for j, feat in enumerate(ENC_FEATURES):
        X_enc[:, j] = _scale(hist[feat].values, feat)

    # ── 2. Decoder (next 144 steps of future covariates)
    now_utc = datetime.now(pytz.UTC)
    start_t = now_utc.replace(minute=30 if now_utc.minute >= 30 else 0, second=0, microsecond=0)
    fut = future_covariates_df[future_covariates_df.index >= start_t].head(144).copy()
    if len(fut) < 144:
        # Pad with last known values if forecast horizon is short
        needed = 144 - len(fut)
        last_t = fut.index[-1] if not fut.empty else start_t
        pad_idx = pd.date_range(last_t + pd.Timedelta("30min"), periods=needed, freq="30min")
        fut = pd.concat([fut, pd.DataFrame(index=pad_idx)])
        fut = fut.ffill().bfill()

    fut = fut.rename(columns={
        "temperature_adelaide": "temp",
        "humidity_adelaide":    "humidity",
        "wind_speed_adelaide":  "wind_speed",
    })
    fut["power_pv"] = fut.get("power_pv", pd.Series(0.0, index=fut.index)).fillna(0.0)

    fut = pd.concat([fut, _time_sin_cos_local(fut.index)], axis=1)
    fut = add_time_features(fut)
    fut["horizon_norm"] = np.arange(len(fut)) / 144.0

    for feat in DEC_FEATURES:
        if feat not in fut.columns:
            fut[feat] = 0.0
    fut[DEC_FEATURES] = fut[DEC_FEATURES].ffill().bfill().fillna(0.0)

    X_dec = np.zeros((144, len(DEC_FEATURES)), dtype=np.float32)
    for j, feat in enumerate(DEC_FEATURES):
        X_dec[:, j] = _scale(fut[feat].values[:144], feat)

    # ── 3. Inference
    with _torch.no_grad():
        t_enc     = _torch.tensor(X_enc).unsqueeze(0)
        t_dec     = _torch.tensor(X_dec).unsqueeze(0)
        preds_norm = model(t_enc, t_dec).squeeze(0).numpy()  # [144, Q]
    preds_norm = np.sort(preds_norm, axis=-1)

    # Inverse-transform: StandardScaler on power_load
    load_sc    = scalers["power_load"]
    preds_W    = load_sc.inverse_transform(
        preds_norm.reshape(-1, 1)
    ).reshape(preds_norm.shape).clip(min=0.0)            # [144, Q]

    # ── 4. Format return
    dec_index = fut.index[:144]
    q_names   = ["tft_load_q10", "tft_load", "tft_load_q90"]
    res = {}
    for i, q_name in enumerate(q_names):
        res[q_name] = pd.DataFrame(
            {"power_load": preds_W[:, i].tolist()},
            index=dec_index,
        )

    logging.info(f"TFT Load Inference complete. q50 median={np.median(preds_W[:, 1]):.0f}W")
    return res


def _expand_forecast_to_publish_interval(df, source_interval_minutes, publish_interval_minutes):
    """
    Expand a forecast DataFrame from its native interval to a finer publish interval
    by repeating each source row across the publish grid.
    """
    if publish_interval_minutes == source_interval_minutes:
        return df.copy()
    if publish_interval_minutes <= 0 or source_interval_minutes <= 0:
        raise ValueError("Forecast intervals must be positive")
    if source_interval_minutes % publish_interval_minutes != 0:
        raise ValueError(
            f"Cannot expand {source_interval_minutes}m source interval into {publish_interval_minutes}m publish interval"
        )

    repeats = source_interval_minutes // publish_interval_minutes
    publish_delta = pd.Timedelta(minutes=publish_interval_minutes)
    rows = []
    expanded_index = []
    for ts, row in df.iterrows():
        for offset in range(repeats):
            expanded_index.append(ts + offset * publish_delta)
            rows.append(row.to_dict())

    expanded_df = pd.DataFrame(rows, index=pd.DatetimeIndex(expanded_index))
    expanded_df.index.name = df.index.name
    return expanded_df


def _build_combined_forecast_items(
    p50_df,
    low_df,
    high_df,
    source_interval_minutes,
    publish_interval_minutes=None,
):
    """
    Convert three DataFrames (each with 'wholesale_price' in $/kWh) into a list of
    Amber-compatible forecast dicts. Applies tariffs to each to get general/feed-in price.

    Returns (general_items, feed_in_items) — two lists of dicts with Amber field names.
    """
    publish_interval_minutes = publish_interval_minutes or source_interval_minutes
    interval_delta = pd.Timedelta(minutes=publish_interval_minutes)
    general_items  = []
    feed_in_items  = []

    frames = {'p50': p50_df.copy(), 'low': low_df.copy(), 'high': high_df.copy()}
    for key, df in frames.items():
        df = ensure_utc_index(df)
        if 'wholesale_price' not in df.columns:
            df.rename(columns={df.columns[0]: 'wholesale_price'}, inplace=True)
        df = _expand_forecast_to_publish_interval(df, source_interval_minutes, publish_interval_minutes)
        apply_tariffs_to_forecast(df)
        frames[key] = df

    for ts in frames['p50'].index:
        gp50  = float(frames['p50'].loc[ts, 'general_price'])
        glow  = float(frames['low'].loc[ts, 'general_price'])
        ghigh = float(frames['high'].loc[ts, 'general_price'])
        fp50  = float(frames['p50'].loc[ts, 'feed_in_price'])
        flow  = float(frames['low'].loc[ts, 'feed_in_price'])
        fhigh = float(frames['high'].loc[ts, 'feed_in_price'])

        general_items.append({
            'start_time': ts.isoformat(),
            'end_time':   (ts + interval_delta).isoformat(),
            'advanced_price_predicted': round(gp50,  6),
            'advanced_price_high':      round(ghigh, 6),
            'advanced_price_low':       round(glow,  6),
            'per_kwh':                  round(gp50,  6),
        })
        # Boundary conversion only: internal feed_in_price is positive export value;
        # Amber-style HA feed-in sensors encode earning as negative prices.
        feed_in_items.append({
            'start_time': ts.isoformat(),
            'end_time':   (ts + interval_delta).isoformat(),
            'advanced_price_predicted': round(export_value_to_amber_feed_in_price(fp50),  6),
            'advanced_price_high':      round(export_value_to_amber_feed_in_price(fhigh), 6),
            'advanced_price_low':       round(export_value_to_amber_feed_in_price(flow),  6),
            'per_kwh':                  round(export_value_to_amber_feed_in_price(fp50),  6),
        })

    return general_items, feed_in_items


def _build_haeo_price_forecast_items(
    p50_df,
    source_interval_minutes,
    publish_interval_minutes=None,
):
    """
    Convert a p50 wholesale price DataFrame into HAEO/HAFO-style forecast points.

    Returns (import_items, export_items), where both use positive economic values:
      - import native_value: cost to import in $/kWh
      - export native_value: revenue to export in $/kWh; negative means paying to export
    """
    publish_interval_minutes = publish_interval_minutes or source_interval_minutes
    df = ensure_utc_index(p50_df)
    if "wholesale_price" not in df.columns:
        df = df.rename(columns={df.columns[0]: "wholesale_price"})
    df = _expand_forecast_to_publish_interval(df, source_interval_minutes, publish_interval_minutes)
    apply_tariffs_to_forecast(df)

    import_items = []
    export_items = []
    for ts, row in df.iterrows():
        import_items.append({
            "datetime": ts.isoformat(),
            "native_value": round(float(row["general_price"]), 6),
        })
        export_items.append({
            "datetime": ts.isoformat(),
            "native_value": round(float(row["feed_in_price"]), 6),
        })
    return import_items, export_items


def _publish_haeo_forecast_sensor(entity_id, items, label, *, interval_minutes):
    state = round(float(items[0]["native_value"]), 6) if items else 0.0
    now_iso = datetime.now(pytz.UTC).isoformat()
    payload = {
        "state": state,
        "attributes": {
            "forecast": items,
            "last_updated": now_iso,
            "friendly_name": label,
            "icon": "mdi:chart-line",
            "unit_of_measurement": "$/kWh",
            "device_class": "monetary",
            "forecast_interval_minutes": int(interval_minutes),
            "forecast_count": len(items),
            "forecast_convention": "haeo_positive_import_export",
            "timestamp_convention": "UTC ISO-8601",
        },
    }
    call_ha_api("POST", f"states/{entity_id}", payload=payload)
    logging.info(f"Published {entity_id} (state={state}, {len(items)} forecast points).")


def _publish_combined_price_forecasts(tactical_results, tft_results):
    """
    Combine Tier 1 (0–55 min, 5-min) and Tier 2 (60 min–72h, 30-min) into two
    Amber-compatible shadow sensors:
      sensor.ai_combined_general_price_forecast   — buy/general price
      sensor.ai_combined_feed_in_price_forecast   — sell/feed-in price

    Tier 1 quantile mapping: q05→low, q50→predicted, q95→high
    Tier 2 quantile mapping: q30→low, q50→predicted, q70→high
    """
    t1_p50  = tactical_results.get('p5min_price')
    t1_low  = tactical_results.get('p5min_price_q05')
    t1_high = tactical_results.get('p5min_price_q95')
    t2_p50  = tft_results.get('tft_price')
    t2_low  = tft_results.get('tft_price_q30')
    t2_high = tft_results.get('tft_price_q70')

    if any(df is None for df in [t1_p50, t1_low, t1_high, t2_p50, t2_low, t2_high]):
        logging.warning("Combined forecast skipped: one or more required model outputs missing.")
        return

    # Tier 1: all 12 steps, already 5-minute native cadence
    t1_end = t1_p50.index[-1]
    gen_t1, fin_t1 = _build_combined_forecast_items(t1_p50, t1_low, t1_high, 5, 5)

    # Tier 2: steps after Tier 1 ends
    t2_p50_after  = t2_p50[t2_p50.index > t1_end]
    t2_low_after  = t2_low[t2_low.index > t1_end]
    t2_high_after = t2_high[t2_high.index > t1_end]

    if t2_p50_after.empty:
        logging.warning("Combined forecast: no Tier 2 steps beyond Tier 1 window.")
        gen_all, fin_all = gen_t1, fin_t1
    else:
        gen_t2, fin_t2 = _build_combined_forecast_items(
            t2_p50_after,
            t2_low_after,
            t2_high_after,
            30,
            5,
        )
        gen_all = gen_t1 + gen_t2
        fin_all = fin_t1 + fin_t2

    logging.info(
        f"Combined forecast: {len(gen_t1)} Tier-1 (5-min) + "
        f"{len(gen_all) - len(gen_t1)} Tier-2 (published 5-min) = {len(gen_all)} total intervals"
    )

    now_iso = datetime.now(pytz.UTC).isoformat()
    for entity_id, items, label in [
        ('sensor.ai_combined_general_price_forecast', gen_all, 'AI Combined General Price Forecast'),
        ('sensor.ai_combined_feed_in_price_forecast', fin_all, 'AI Combined Feed In Price Forecast'),
    ]:
        state = round(items[0]['advanced_price_predicted'], 6) if items else 0.0
        payload = {
            'state': state,
            'attributes': {
                'Forecasts':      items,
                'last_updated':   now_iso,
                'friendly_name':  label,
                'icon':           'mdi:chart-line',
            },
        }
        call_ha_api('POST', f'states/{entity_id}', payload=payload)
        logging.info(f"Published {entity_id} (state={state}, {len(items)} intervals).")

    # HAEO/HAFO-style canonical sensors. These use positive import cost and
    # positive export revenue, avoiding Amber's negative feed-in convention.
    mpc_import_t1, mpc_export_t1 = _build_haeo_price_forecast_items(t1_p50, 5, 5)
    if t2_p50_after.empty:
        mpc_import_all, mpc_export_all = mpc_import_t1, mpc_export_t1
    else:
        mpc_import_t2, mpc_export_t2 = _build_haeo_price_forecast_items(t2_p50_after, 30, 5)
        mpc_import_all = mpc_import_t1 + mpc_import_t2
        mpc_export_all = mpc_export_t1 + mpc_export_t2

    dh_import_all, dh_export_all = _build_haeo_price_forecast_items(t2_p50, 30, 30)

    for entity_id, items, label, interval_minutes in [
        (
            "sensor.ai_mpc_import_price_forecast",
            mpc_import_all[:168],
            "AI MPC Import Price Forecast",
            5,
        ),
        (
            "sensor.ai_mpc_export_price_forecast",
            mpc_export_all[:168],
            "AI MPC Export Price Forecast",
            5,
        ),
        (
            "sensor.ai_dh_import_price_forecast",
            dh_import_all[:144],
            "AI Day Ahead Import Price Forecast",
            30,
        ),
        (
            "sensor.ai_dh_export_price_forecast",
            dh_export_all[:144],
            "AI Day Ahead Export Price Forecast",
            30,
        ),
    ]:
        _publish_haeo_forecast_sensor(entity_id, items, label, interval_minutes=interval_minutes)


def _publish_pd_direct_price_forecasts(tactical_results, pd_direct_results):
    """PD-direct sibling of _publish_combined_price_forecasts (Phase α-prime Step 2).

    Same Tier-1 + strategic combination logic, same Amber-shaped + HAEO-style entity
    schemas, same tariff plumbing. Only differences:
      - Strategic source is PD-direct (debiased PREDISPATCH + PD7Day + seasonal HoD)
        instead of TFT q50.
      - Entity names use the `ai_pd_direct_*` prefix so live A/B comparison against the
        existing `ai_*` family is possible without flipping any selector.
      - Friendly names say "AI PD-Direct ..." instead of "AI Combined ..." / "AI MPC ...".

    Quantile bands are degenerate in this first cut (q30=q70=q50). Phase α-prime Step 3
    will replace these with horizon-stratified empirical residual bands without
    requiring any change to this publishing path.
    """
    t1_p50  = tactical_results.get('p5min_price')
    t1_low  = tactical_results.get('p5min_price_q05')
    t1_high = tactical_results.get('p5min_price_q95')
    t2_p50  = pd_direct_results.get('pd_direct_price')
    t2_low  = pd_direct_results.get('pd_direct_price_q30')
    t2_high = pd_direct_results.get('pd_direct_price_q70')

    if any(df is None for df in [t1_p50, t1_low, t1_high, t2_p50, t2_low, t2_high]):
        logging.warning("PD-direct combined forecast skipped: one or more required "
                        "model outputs missing.")
        return

    # Tier 1: 12 steps × 5 min (already 5-minute native cadence)
    t1_end = t1_p50.index[-1]
    gen_t1, fin_t1 = _build_combined_forecast_items(t1_p50, t1_low, t1_high, 5, 5)

    # Strategic: any 30-min steps after Tier 1 ends
    t2_p50_after  = t2_p50[t2_p50.index > t1_end]
    t2_low_after  = t2_low[t2_low.index > t1_end]
    t2_high_after = t2_high[t2_high.index > t1_end]

    if t2_p50_after.empty:
        logging.warning("PD-direct combined forecast: no strategic steps beyond "
                        "Tier 1 window.")
        gen_all, fin_all = gen_t1, fin_t1
    else:
        gen_t2, fin_t2 = _build_combined_forecast_items(
            t2_p50_after, t2_low_after, t2_high_after, 30, 5,
        )
        gen_all = gen_t1 + gen_t2
        fin_all = fin_t1 + fin_t2

    logging.info(
        f"PD-direct combined forecast: {len(gen_t1)} Tier-1 (5-min) + "
        f"{len(gen_all) - len(gen_t1)} strategic (published 5-min) = "
        f"{len(gen_all)} total intervals"
    )

    now_iso = datetime.now(pytz.UTC).isoformat()
    for entity_id, items, label in [
        ('sensor.ai_pd_direct_combined_general_price_forecast', gen_all,
         'AI PD-Direct Combined General Price Forecast'),
        ('sensor.ai_pd_direct_combined_feed_in_price_forecast', fin_all,
         'AI PD-Direct Combined Feed In Price Forecast'),
    ]:
        state = round(items[0]['advanced_price_predicted'], 6) if items else 0.0
        payload = {
            'state': state,
            'attributes': {
                'Forecasts':     items,
                'last_updated':  now_iso,
                'friendly_name': label,
                'icon':          'mdi:chart-line',
            },
        }
        call_ha_api('POST', f'states/{entity_id}', payload=payload)
        logging.info(f"Published {entity_id} (state={state}, {len(items)} intervals).")

    # HAEO/HAFO-style canonical sensors with positive import cost / positive export
    # revenue. EMHASS template branches read these directly when the source selector is
    # set to `ai_pd_direct`.
    mpc_import_t1, mpc_export_t1 = _build_haeo_price_forecast_items(t1_p50, 5, 5)
    if t2_p50_after.empty:
        mpc_import_all, mpc_export_all = mpc_import_t1, mpc_export_t1
    else:
        mpc_import_t2, mpc_export_t2 = _build_haeo_price_forecast_items(t2_p50_after, 30, 5)
        mpc_import_all = mpc_import_t1 + mpc_import_t2
        mpc_export_all = mpc_export_t1 + mpc_export_t2

    dh_import_all, dh_export_all = _build_haeo_price_forecast_items(t2_p50, 30, 30)

    for entity_id, items, label, interval_minutes in [
        (
            "sensor.ai_pd_direct_mpc_import_price_forecast",
            mpc_import_all[:168],
            "AI PD-Direct MPC Import Price Forecast",
            5,
        ),
        (
            "sensor.ai_pd_direct_mpc_export_price_forecast",
            mpc_export_all[:168],
            "AI PD-Direct MPC Export Price Forecast",
            5,
        ),
        (
            "sensor.ai_pd_direct_dh_import_price_forecast",
            dh_import_all[:144],
            "AI PD-Direct Day Ahead Import Price Forecast",
            30,
        ),
        (
            "sensor.ai_pd_direct_dh_export_price_forecast",
            dh_export_all[:144],
            "AI PD-Direct Day Ahead Export Price Forecast",
            30,
        ),
    ]:
        _publish_haeo_forecast_sensor(entity_id, items, label, interval_minutes=interval_minutes)


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

    # Combine history and future
    combined_covariates_df = pd.concat([historical_covariates_df, future_covariates_df])

    # ----------------------------------------------------------------------- #
    # --- DST FIX START ----------------------------------------------------- #
    # During DST transitions, pd.concat might degrade the index from
    # DatetimeIndex to a generic object Index if inputs aren't perfectly
    # aligned timezone-wise. We must force it back to UTC DatetimeIndex
    # so that .index.time works later.
    logging.info("Enforcing UTC DatetimeIndex on combined covariates (DST safety).")
    combined_covariates_df.index = pd.to_datetime(combined_covariates_df.index, utc=True)
    # --- DST FIX END ------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    # Deduplicate and sort
    combined_covariates_df = combined_covariates_df[~combined_covariates_df.index.duplicated(keep='last')].sort_index()
    
    combined_covariates_df = add_time_features(combined_covariates_df)

    original_covariates_for_log = combined_covariates_df.copy()

    adjusted_covariates_df = apply_covariate_adjustments(combined_covariates_df)
    if publish_covariates:
        publish_df = adjusted_covariates_df[adjusted_covariates_df.index >= forecast_start_time]
        publish_adjusted_covariates_to_hass(publish_df)
    adjusted_covariates_for_prediction = adjusted_covariates_df.copy()
    adjusted_covariates_for_prediction.ffill(inplace=True)
    adjusted_covariates_for_prediction.bfill(inplace=True)
    amber_spot_df = pd.DataFrame()

    if 'price' in models_to_run:
        try:
            amber_entity_id = CONFIG['home_assistant']['amber_entity']
            amber_entity_state = get_entity_state(amber_entity_id)
            if amber_entity_state:
                amber_forecasts = amber_entity_state.get("attributes", {}).get("Forecasts", [])
                if amber_forecasts:
                    _persist_amber_spot_5min_forecasts(amber_entity_id, amber_entity_state, amber_forecasts)
                    amber_spot_df = get_amber_spot_price_forecast()
        except Exception as e:
            logging.error(f"FATAL ERROR capturing Amber 5-min spot forecasts: {e}", exc_info=True)

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

    # 2b. Execute Tier 1/2 parallel models (sandboxed — existing forecasts safe on failure)
    if 'price' in models_to_run:
        tactical_results = {}
        try:
            tactical_results = _execute_tactical_prediction()
            if tactical_results:
                all_results['p5min_tactical'] = {'forecasts': tactical_results, 'type': 'lgbm_tactical'}
        except Exception as e:
            logging.error(f"FATAL ERROR in Tier 1 tactical execution: {e}", exc_info=True)

        try:
            tft_results = _execute_tft_prediction(historical_df, adjusted_covariates_for_prediction)
            if tft_results:
                all_results['tft_price'] = {'forecasts': tft_results, 'type': 'tft_pytorch'}
        except Exception as e:
            logging.error(f"FATAL ERROR in TFT Price shadow execution: {e}", exc_info=True)

        if publish_hass:
            try:
                _publish_combined_price_forecasts(
                    tactical_results if 'p5min_tactical' in all_results else {},
                    all_results.get('tft_price', {}).get('forecasts', {}),
                )
            except Exception as e:
                logging.error(f"FATAL ERROR in combined forecast publish: {e}", exc_info=True)

        # PD-direct shadow forecast (Phase α-prime Step 2). Built and published
        # in its own try/except so a failure here cannot affect the existing TFT
        # publish path. Kept local to this scope (not added to all_results) because
        # the JSON-save loop below expects each entry to be a registered model in
        # CONFIG; PD-direct is HA-publish-only for now. RFC:
        # docs/pd_direct_publish_rfc.md.
        try:
            pd_direct_results = _execute_pd_direct_prediction(
                historical_df, adjusted_covariates_for_prediction)
        except Exception as e:
            logging.error(f"FATAL ERROR in PD-direct execution: {e}", exc_info=True)
            pd_direct_results = {}

        if publish_hass and pd_direct_results:
            try:
                _publish_pd_direct_price_forecasts(
                    tactical_results if 'p5min_tactical' in all_results else {},
                    pd_direct_results,
                )
            except Exception as e:
                logging.error(f"FATAL ERROR in PD-direct publish: {e}", exc_info=True)

    if 'load' in models_to_run:
        try:
            tft_load_results = _execute_tft_load_prediction(historical_df, adjusted_covariates_for_prediction)
            if tft_load_results:
                all_results['tft_load'] = {'forecasts': tft_load_results, 'type': 'tft_pytorch'}
        except Exception as e:
            logging.error(f"FATAL ERROR in TFT Load shadow execution: {e}", exc_info=True)

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
        
        if 'price' in model_name:
            primary_pred_df_for_save.rename(columns={primary_pred_df_for_save.columns[0]: 'wholesale_price'}, inplace=True)
            # Only apply Amber spot override for the base 'price' model if handover is NOT used
            if model_name == 'price' and not use_dynamic_handoff:
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
                
                if 'price' in base_model_name:
                    # Price models are renamed to 'wholesale_price' for the tariff function
                    if 'wholesale_price' not in publish_df.columns:
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
    # Floor to 30-min before tariff lookup so 5-min intervals (e.g. 12:25) match the
    # tariff map keys (e.g. 12:00 / 12:30) rather than falling through to fillna(0).
    pred_df['local_time'] = (
        pred_df.index.tz_convert(local_tz).floor('30min').time.astype(str)
    )

    general_tariff_map = tariffs.get('general_tariff', {})
    feed_in_tariff_map = tariffs.get('feed_in_tariff', {})

    pred_df['general_tariff'] = pred_df['local_time'].map(general_tariff_map).fillna(0)
    pred_df['feed_in_tariff'] = pred_df['local_time'].map(feed_in_tariff_map).fillna(0)

    loss_factor = get_network_loss_factor()

    general_price_ex_gst = (pred_df['wholesale_price'] * loss_factor) + pred_df['general_tariff']
    feed_in_price_ex_gst = (pred_df['wholesale_price'] * loss_factor) + pred_df['feed_in_tariff']

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
    # This function extracts pure network tariffs
    entity_state = get_entity_state(entity_id)
    if not entity_state: return pd.DataFrame()
    forecasts = entity_state.get("attributes", {}).get("Forecasts", [])
    if not forecasts: return pd.DataFrame()
    processed = []
    
    api_scaling = get_amber_api_scaling_factor()
    net_loss = get_network_loss_factor()
    
    for f in forecasts:
        if f.get('per_kwh') is not None and f.get('spot_per_kwh') is not None:
            per_kwh_dollars = float(f['per_kwh'])
            spot_per_kwh_dollars = float(f['spot_per_kwh'])
            
            # Strip the 1.10 API inflation down to Raw AEMO scale
            raw_spot = spot_per_kwh_dollars / api_scaling
            # Scale it up to the True Network Loss scale (1.08)
            true_loss_spot = raw_spot * net_loss
            
            if is_feed_in:
                tariff = -remove_gst(per_kwh_dollars) - true_loss_spot
            else:
                tariff = remove_gst(per_kwh_dollars) - true_loss_spot
            processed.append({'datetime': pd.to_datetime(f['start_time']).round('min'), 'tariff': tariff})
    return pd.DataFrame(processed).set_index('datetime')

def _create_complete_profile(tariff_df, local_tz, tariff_type_for_logging):
    """
    WORKER: Takes a DataFrame of future tariff data and creates a complete,
    robust 24-hour (48-interval) time-of-day profile.

    This is achieved by:
    1. Using the next available 48 data points (24 hours).
    2. Creating a full, generic 48-point time index (00:00 to 23:30).
    3. Re-indexing the tariff data against this complete index.
    4. Filling any gaps (like from a DST transition) to ensure a full profile.
    """
    # Convert index to the local timezone
    tariff_df.index = tariff_df.index.tz_convert(local_tz)

    # Take the next 24 hours (48 points) of available data
    profile_source_df = tariff_df.sort_index().head(48).copy()
    
    num_points = len(profile_source_df)
    if num_points < 48:
        logging.warning(
            f"Using an incomplete forecast for {tariff_type_for_logging} tariff profile ({num_points}/48 points available). "
            "The profile will be generated but may be less accurate."
        )

    if profile_source_df.empty:
        logging.error(f"No source data available to build {tariff_type_for_logging} tariff profile.")
        return {}

    # Create a profile mapping time-of-day to the tariff value.
    # .mean() correctly handles the "fall back" DST hour by averaging the duplicates.
    profile_source_df['time'] = profile_source_df.index.time
    tariff_by_time = profile_source_df.groupby('time')['tariff'].mean()

    # --- DST ROBUSTNESS ---
    # Create a complete, generic time index for a full 24-hour day (48 intervals)
    full_time_index = pd.Index(
        pd.date_range(start='2000-01-01', periods=48, freq='30min').time,
        name='time'
    )

    # Reindex our potentially gappy profile against the complete index.
    # This creates NaNs for any missing times (e.g., during a DST jump).
    complete_tariff_series = tariff_by_time.reindex(full_time_index)

    # Use back-fill then forward-fill to robustly patch any gaps.
    # This is the safest way to ensure the entire series is filled.
    complete_tariff_series = complete_tariff_series.bfill().ffill()
    
    logging.info(f"Generated complete 24h {tariff_type_for_logging} tariff profile with {len(complete_tariff_series)} entries.")
    
    # Convert to the final dictionary format
    final_profile = complete_tariff_series.to_dict()
    return {str(k): v for k, v in final_profile.items()}


def _calculate_amber_api_scaling_factor():
    logging.info("Calculating Amber API scaling factor from live forecasts...")
    
    amber_df = get_amber_spot_price_forecast(apply_loss_factor=False)
    if amber_df.empty:
        logging.warning("No Amber spot forecast available for scaling factor calculation.")
        return None
        
    pd_df = _get_aemo_short_term_price_forecast(regions=['SA1'])
    if pd_df.empty: return None
    aemo_df = pd_df.rename(columns={'pd_rrp': 'aemo_price_sa1'})
    if aemo_df.empty:
        logging.warning("No AEMO price forecast available for scaling factor calculation.")
        return None
        
    joined_df = amber_df.join(aemo_df, rsuffix='_aemo').dropna()
    if joined_df.empty:
        logging.warning("No overlapping intervals between Amber and AEMO price forecasts.")
        return None
        
    valid_df = joined_df[joined_df['aemo_price_sa1_aemo'].abs() > 0.01].copy()
    if valid_df.empty:
        logging.warning("No valid AEMO prices > $0.01 available for scaling factor calculation.")
        return None
        
    valid_df['ratio'] = valid_df['aemo_price_sa1'] / valid_df['aemo_price_sa1_aemo']
    median_ratio = valid_df['ratio'].median()
    
    if 0.95 <= median_ratio <= 1.25:
        logging.info(f"Successfully calculated Amber API scaling factor: {median_ratio:.4f}")
        return float(median_ratio)
    else:
        logging.warning(f"Calculated Amber API scaling factor {median_ratio:.4f} is outside reasonable bounds [0.95, 1.25]. Ignoring.")
        return None

def _calculate_forecasted_network_loss_factor():
    logging.info("Calculating network loss factor from forward feed-in forecast...")
    try:
        api_scaling = get_amber_api_scaling_factor()
        entity_id = CONFIG['home_assistant']['amber_feed_in_entity']
        entity_state = get_entity_state(entity_id)
        
        if not entity_state:
            logging.warning(f"Could not retrieve state for {entity_id}")
            return None
            
        forecasts = entity_state.get("attributes", {}).get("Forecasts", [])
        if not forecasts:
            logging.warning("No forecasts found in entity attributes")
            return None
            
        records = []
        for f in forecasts:
            if f.get('per_kwh') is not None and f.get('spot_per_kwh') is not None:
                per_kwh_dollars = float(f['per_kwh'])
                spot_per_kwh_dollars = float(f['spot_per_kwh'])
                raw_spot = spot_per_kwh_dollars / api_scaling
                
                if abs(raw_spot) > 0.01:
                    derived_nlf = -remove_gst(per_kwh_dollars) / raw_spot
                    start_time = pd.to_datetime(f['start_time']).tz_convert(CONFIG['timezone'])
                    records.append({
                        'hour': start_time.hour,
                        'derived_nlf': derived_nlf
                    })
                    
        df = pd.DataFrame(records)
        if df.empty:
            logging.warning("No valid forecast points found for loss factor calculation")
            return None
            
        # Zero tariff hours: 0-9, 16, 21-23
        mask_zero_tariff = (df['hour'] < 10) | (df['hour'] == 16) | (df['hour'] >= 21)
        df_zero = df[mask_zero_tariff]
        
        if df_zero.empty:
            logging.warning("No zero-tariff forecast points found for loss factor calculation")
            return None
            
        median_nlf = df_zero['derived_nlf'].median()
        
        if 0.95 <= median_nlf <= 1.25:
            logging.info(f"Successfully calculated network loss factor from forecast: {median_nlf:.4f}")
            return float(median_nlf)
        else:
            logging.warning(f"Forecast-derived network loss factor {median_nlf:.4f} is outside reasonable bounds. Ignoring.")
            return None
    except Exception as e:
        logging.error(f"Error calculating forecasted loss factor: {e}")
        return None

def update_tariffs():
    """
    ORCHESTRATOR: Fetches future tariff data and uses a helper function
    to generate robust, complete 24-hour profiles for general and feed-in rates.
    """
    logging.info("--- Running in UPDATE-TARIFFS mode ---")
    local_tz = pytz.timezone(CONFIG['timezone'])
    
    # 1. Fetch all available future tariff data
    general_tariff_df = _get_tariff_data(CONFIG['home_assistant']['amber_entity'], is_feed_in=False)
    feed_in_tariff_df = _get_tariff_data(CONFIG['home_assistant']['amber_feed_in_entity'], is_feed_in=True)

    if general_tariff_df.empty and feed_in_tariff_df.empty:
        raise SystemExit("Could not retrieve any tariff information from Amber entities.")

    final_profile = {}

    try:
        with open(CONFIG['paths']['tariff_file'], 'r') as f:
            old_profile = json.load(f)
            if 'amber_api_scaling_factor' in old_profile:
                final_profile['amber_api_scaling_factor'] = old_profile['amber_api_scaling_factor']
            if 'network_loss_factor' in old_profile:
                final_profile['network_loss_factor'] = old_profile['network_loss_factor']
    except (FileNotFoundError, json.JSONDecodeError):
        final_profile['amber_api_scaling_factor'] = 1.10
        final_profile['network_loss_factor'] = 1.05

    new_api_scaling = _calculate_amber_api_scaling_factor()
    if new_api_scaling is not None:
        final_profile['amber_api_scaling_factor'] = new_api_scaling
        
    new_network_loss = _calculate_forecasted_network_loss_factor()
    if new_network_loss is not None:
        final_profile['network_loss_factor'] = new_network_loss

    # 2. Generate profile for general tariff, if data exists
    if not general_tariff_df.empty:
        final_profile['general_tariff'] = _create_complete_profile(
            general_tariff_df, local_tz, "general"
        )

    # 3. Generate profile for feed-in tariff, if data exists
    if not feed_in_tariff_df.empty:
        final_profile['feed_in_tariff'] = _create_complete_profile(
            feed_in_tariff_df, local_tz, "feed-in"
        )

    # 4. Save the combined profile to disk
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
    parser.add_argument('--debug-tft', action='store_true', help="Print TFT encoder/decoder inputs and output side-by-side (for diagnosing underestimation).")
    parser.add_argument('--config', default='config.json', help="Path to the configuration file.")
    args = parser.parse_args()

    global CONFIG, _DEBUG_TFT
    CONFIG = load_config(args.config)
    _DEBUG_TFT = getattr(args, "debug_tft", False)

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
