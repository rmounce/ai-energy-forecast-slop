#!/usr/bin/env python3

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
    # This function remains unchanged
    logging.info(f"Querying historical data from {start_time.date()} to {end_time.date()}")
    start_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
    data_sources = {
        'power_load': f'SELECT mean("mean_value") FROM "rp_30m"."power_load_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'power_pv': f'SELECT mean("mean_value") FROM "rp_30m"."power_pv_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'temperature_adelaide': f'SELECT mean("mean_value") FROM "rp_30m"."temperature_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'humidity_adelaide': f'SELECT mean("mean_value") FROM "rp_30m"."humidity_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'wind_speed_adelaide': f'SELECT mean("mean_value") FROM "rp_30m"."wind_speed_adelaide" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
        'aemo_price_sa1': f'SELECT mean("price") / 1000 FROM "rp_30m"."aemo_dispatch_sa1_30m" WHERE time >= \'{start_str}\' AND time <= \'{end_str}\' GROUP BY time(30m)',
    }
    dataframes = {}
    for name, query in data_sources.items():
        try:
            result = client.query(query)
            if result:
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

def get_amber_advanced_forecast():
    # This function remains unchanged
    logging.info("Retrieving Amber Electric ADVANCED price forecast (mixed intervals)...")
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
        if (f.get('advanced_price_predicted') is not None and
            f.get('per_kwh') is not None and
            f.get('spot_per_kwh') is not None):

            per_kwh = float(f['per_kwh'])
            spot_per_kwh = float(f['spot_per_kwh'])
            advanced_price_incl_gst = float(f['advanced_price_predicted'])
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

def create_time_features(df):
    # This function is unchanged
    df_out = df.copy()
    df_out['hour'] = df_out.index.hour
    df_out['day_of_week'] = df_out.index.dayofweek
    df_out['day_of_year'] = df_out.index.dayofyear
    df_out['month'] = df_out.index.month
    return df_out

# --------------------------------------------------------------------------- #
# 5. MODEL TRAINING, PREDICTION & PUBLISHING FUNCTIONS
# --------------------------------------------------------------------------- #

# --- MODIFIED FUNCTION ---
def log_forecast_data(model_name, model_version, prediction_type, final_pred_df, future_covariates_df):
    """
    Logs model inputs and outputs to a CSV, ensuring a consistent column structure.
    """
    logging.info("Logging forecast data and covariates...")
    log_file_path = Path(CONFIG['paths']['forecast_log_file'])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    log_df = final_pred_df.copy()

    if log_df.index.tz is None:
        log_df.index = log_df.index.tz_localize('UTC')

    # Rename the specific prediction column for this model run
    prediction_col = log_df.columns[0]
    log_df.rename(columns={prediction_col: f'{model_name}_prediction'}, inplace=True)

    # Join with the features used for this prediction
    log_df = log_df.join(future_covariates_df, how='left')

    # Add metadata
    log_df['forecast_creation_time'] = datetime.now(pytz.UTC).isoformat()
    log_df['model_version'] = model_version
    log_df['model_name'] = model_name
    log_df['prediction_type'] = prediction_type

    log_df.index.name = 'forecast_target_time'
    log_df.reset_index(inplace=True)

    # --- ROBUSTNESS FIX ---
    # Define the complete, ideal set of columns for the log file
    all_possible_cols = [
        'forecast_creation_time', 'forecast_target_time', 'model_name',
        'model_version', 'prediction_type',
        'price_prediction', 'price_actual',
        'load_prediction', 'load_actual'
    ]
    # Add all feature columns that could exist
    all_possible_cols.extend(sorted(list(future_covariates_df.columns)))

    # Ensure all columns exist, filling with NaN if they don't.
    for col in all_possible_cols:
        if col not in log_df.columns:
            log_df[col] = np.nan

    # Reorder the dataframe to match the ideal structure
    final_cols_order = [col for col in all_possible_cols if col in log_df.columns]
    log_df = log_df[final_cols_order]

    try:
        # If the file doesn't exist or is empty, write the header.
        header = not log_file_path.exists() or os.path.getsize(log_file_path) == 0
        log_df.to_csv(log_file_path, mode='a', header=header, index=False)
        logging.info(f"Successfully logged {len(log_df)} records to {log_file_path}")
    except Exception as e:
        logging.error(f"Failed to write to forecast log file: {e}")


def train_single_model(model_name):
    """
    Trains a single model and saves it.
    """
    logging.info(f"--- Running in TRAIN mode for model: {model_name} ---")
    model_config = CONFIG['models'][model_name]
    target_col, feature_cols = model_config['target_column'], model_config['feature_cols']
    client = InfluxDBClient(**CONFIG['influxdb'])

    try:
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

        model_data_featured = create_time_features(model_data)
        all_feature_cols = feature_cols + ['hour', 'day_of_week', 'day_of_year', 'month']

        target_series = TimeSeries.from_series(model_data_featured[target_col], freq='30min')
        covariates_ts = TimeSeries.from_dataframe(model_data_featured, value_cols=all_feature_cols, freq='30min')

        lgbm = LightGBMModel(
            lags=24 * 7,
            lags_future_covariates=[0],
            output_chunk_length=model_config['forecast_horizon'],
            **model_config['lgbm_params']
        )
        logging.info("Fitting the model...")
        lgbm.fit(series=target_series, future_covariates=covariates_ts)

        logging.info("Extracting feature importances...")
        if isinstance(lgbm.model, MultiOutputRegressor):
            importances = np.mean([estimator.feature_importances_ for estimator in lgbm.model.estimators_], axis=0)
        else:
            importances = lgbm.model.feature_importances_
        
        target_lags_list = lgbm.lags['target']
        lag_names = [f'{target_col}_lag{lag}' for lag in target_lags_list]
        covariate_names = list(covariates_ts.columns)
        true_feature_names = lag_names + covariate_names

        if len(true_feature_names) != len(importances):
            logging.error("FATAL: Length of feature names does not match length of importances!")
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

def _predict_with_dynamic_handoff(model, params, historical_df, future_covariates_df, model_config):
    # This function is unchanged
    logging.info("Generating forecast using 'dynamic handoff' method.")
    target_col = model_config['target_column']
    shift_value = params['shift_value']
    amber_advanced_df = get_amber_advanced_forecast()
    if amber_advanced_df.empty:
        logging.warning("No advanced Amber data available. Falling back to simple prediction.")
        all_feature_cols = model_config['feature_cols'] + ['hour', 'day_of_week', 'day_of_year', 'month']
        future_covariates_ts = TimeSeries.from_dataframe(future_covariates_df, value_cols=all_feature_cols, freq='30min')
        return _predict_simple(model, params, historical_df, future_covariates_ts, model_config)
    last_good_amber_index = amber_advanced_df.index.max()
    logging.info(f"Using Amber advanced forecast for {len(amber_advanced_df) / 2} hours (up to {last_good_amber_index}).")
    hist_df_log = historical_df[[target_col] + model_config['feature_cols']].copy().ffill().dropna()
    hist_df_log[target_col] = np.log(hist_df_log[target_col] + shift_value)
    amber_seed_data = future_covariates_df.loc[amber_advanced_df.index].copy()
    amber_seed_data[target_col] = np.log(amber_advanced_df[target_col] + shift_value)
    pseudo_history_df = pd.concat([hist_df_log, amber_seed_data.dropna()])
    pseudo_history_ts = TimeSeries.from_series(pseudo_history_df[target_col], freq='30min')
    all_feature_cols = model_config['feature_cols'] + ['hour', 'day_of_week', 'day_of_year', 'month']
    future_covariates_ts = TimeSeries.from_dataframe(future_covariates_df, value_cols=all_feature_cols, freq='30min')
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

def predict_with_model(model_name, publish_to_hass=False, use_dynamic_handoff=False):
    # This function is largely the same, but now passes data to the modified logger
    logging.info(f"--- Running in PREDICT mode for model: {model_name} ---")
    model_config = CONFIG['models'][model_name]
    model_file_path = CONFIG['paths'][f'{model_name}_model_file']
    try:
        model = joblib.load(model_file_path)
        with open(CONFIG['paths'][f'{model_name}_params_file'], 'r') as f: params = json.load(f)
        mod_time = os.path.getmtime(model_file_path)
        model_version = datetime.fromtimestamp(mod_time, tz=pytz.UTC).isoformat()
        logging.info(f"Loaded model '{model_name}'. Version (timestamp): {model_version}")
    except FileNotFoundError:
        raise SystemExit(f"Model for '{model_name}' not found. Please run 'train-{model_name}' first.")

    now = datetime.now(pytz.UTC)
    minute = 30 if now.minute >= 30 else 0
    forecast_start_time = now.replace(minute=minute, second=0, microsecond=0)
    logging.info(f"Aligning forecast to start at current interval: {forecast_start_time}")

    logging.info("Fetching all future covariate data...")
    future_sources = {'solcast': get_solcast_forecast(), 'weather': get_weather_forecast()}
    prediction_type = 'simple'
    if use_dynamic_handoff and model_name == 'price':
        prediction_type = 'dynamic_handoff'
    else:
        future_sources['amber_spot'] = get_amber_spot_price_forecast()

    future_covariates_df = pd.concat(future_sources.values(), axis=1).sort_index()
    future_covariates_df = future_covariates_df[future_covariates_df.index >= forecast_start_time]

    time_featured_df = create_time_features(future_covariates_df)
    missing_cols = [col for col in model_config['feature_cols'] if col not in time_featured_df.columns]
    if missing_cols: raise SystemExit(f"FATAL: Missing future data for features: {missing_cols}. Check HA connection.")

    original_covariates_for_log = time_featured_df.copy()
    time_featured_df = time_featured_df.ffill().bfill()
    time_featured_df.dropna(inplace=True)
    if time_featured_df.empty: raise SystemExit(f"FATAL: Could not create a complete future covariate set.")

    client = InfluxDBClient(**CONFIG['influxdb'])
    try:
        historical_df = get_historical_data(client, forecast_start_time - timedelta(days=CONFIG['prediction_history_days']), forecast_start_time - timedelta(minutes=30))
        if historical_df.empty: raise SystemExit("Aborting: Failed to get recent history for prediction.")
    finally:
        client.close()

    if prediction_type == 'dynamic_handoff':
        final_pred_df = _predict_with_dynamic_handoff(model, params, historical_df, time_featured_df, model_config)
    else:
        all_feature_cols = model_config['feature_cols'] + ['hour', 'day_of_week', 'day_of_year', 'month']
        future_covariates_ts = TimeSeries.from_dataframe(time_featured_df, value_cols=all_feature_cols, freq='30min')
        final_pred_df = _predict_simple(model, params, historical_df, future_covariates_ts, model_config)

    log_forecast_data(
        model_name=model_name,
        model_version=model_version,
        prediction_type=prediction_type,
        final_pred_df=final_pred_df,
        future_covariates_df=original_covariates_for_log
    )

    if model_name == 'price':
        final_pred_df.rename(columns={model_config['target_column']: 'wholesale_price'}, inplace=True)
        if not use_dynamic_handoff and 'amber_spot' in future_sources and not future_sources['amber_spot'].empty:
            final_pred_df.update(future_sources['amber_spot'])
        apply_tariffs_to_forecast(final_pred_df)

    logging.info(f"--- Prediction complete for {model_name} ---")

    final_pred_df.index.name = 'timestamp'
    output_df = final_pred_df.reset_index()
    if output_df['timestamp'].dt.tz is None:
        output_df['timestamp'] = output_df['timestamp'].dt.tz_localize('UTC')
    output_df['timestamp'] = output_df['timestamp'].apply(lambda x: x.isoformat())
    output_data = output_df.to_dict('records')
    try:
        with open(CONFIG['paths']['prediction_output_file'], 'r') as f: all_predictions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError): all_predictions = {}
    all_predictions[f'{model_name}_forecast'] = output_data
    all_predictions[f'{model_name}_last_updated'] = datetime.now(pytz.UTC).isoformat()
    with open(CONFIG['paths']['prediction_output_file'], 'w') as f: json.dump(all_predictions, f, indent=4)
    logging.info(f"Saved {model_name} forecast to {CONFIG['paths']['prediction_output_file']}.")
    print("\nForecast Head:")
    print(final_pred_df.head())
    if publish_to_hass:
        publish_forecast_to_hass(model_name, final_pred_df)

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

def publish_forecast_to_hass(model_name, forecast_df):
    # This function is unchanged
    entity_id = CONFIG['home_assistant']['publish_entities'].get(model_name)
    if not entity_id: return
    logging.info(f"Publishing {model_name} forecast to Home Assistant entity: {entity_id}")
    forecast_df.index.name = 'timestamp'
    output_df = forecast_df.reset_index()
    if output_df['timestamp'].dt.tz is None:
        output_df['timestamp'] = output_df['timestamp'].dt.tz_localize('UTC')
    output_df['timestamp'] = output_df['timestamp'].apply(lambda x: x.isoformat())
    attributes = {"forecasts": output_df.to_dict('records'), "last_updated": datetime.now(pytz.UTC).isoformat(), "friendly_name": f"AI {model_name.capitalize()} Forecast", "icon": "mdi:chart-line"}
    state_col = 'general_price' if model_name == 'price' else forecast_df.columns[0]
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

# --- MODIFIED FUNCTION ---
def backfill_actuals():
    """
    Updates the forecast log with actual observed values from InfluxDB.
    This version is more robust and handles missing columns gracefully.
    """
    logging.info("--- Running in BACKFILL-ACTUALS mode ---")
    log_file_path = Path(CONFIG['paths']['forecast_log_file'])
    if not log_file_path.exists() or os.path.getsize(log_file_path) == 0:
        logging.warning("Forecast log file does not exist or is empty. Nothing to backfill.")
        return

    log_df = pd.read_csv(log_file_path, parse_dates=['forecast_target_time'])

    if log_df.empty:
        logging.info("Log file is empty after reading. Nothing to backfill.")
        return

    if log_df['forecast_target_time'].dt.tz is None:
        log_df['forecast_target_time'] = log_df['forecast_target_time'].dt.tz_localize('UTC')

    now = datetime.now(pytz.UTC)

    # --- ROBUSTNESS FIX ---
    # Check for columns before creating masks
    price_to_fill = pd.DataFrame()
    if 'price_prediction' in log_df.columns and 'price_actual' in log_df.columns:
        price_mask = log_df['price_actual'].isna() & (log_df['forecast_target_time'] < now) & log_df['price_prediction'].notna()
        price_to_fill = log_df[price_mask]
        logging.info(f"Found {len(price_to_fill)} price records to backfill.")

    load_to_fill = pd.DataFrame()
    if 'load_prediction' in log_df.columns and 'load_actual' in log_df.columns:
        load_mask = log_df['load_actual'].isna() & (log_df['forecast_target_time'] < now) & log_df['load_prediction'].notna()
        load_to_fill = log_df[load_mask]
        logging.info(f"Found {len(load_to_fill)} load records to backfill.")

    if price_to_fill.empty and load_to_fill.empty:
        logging.info("No past forecast records require backfilling at this time.")
        return

    # Consolidate time range for a single efficient query
    all_times_to_fill = pd.concat([price_to_fill['forecast_target_time'], load_to_fill['forecast_target_time']]).unique()
    start_time = pd.to_datetime(all_times_to_fill.min())
    end_time = pd.to_datetime(all_times_to_fill.max())

    logging.info(f"Fetching actuals from {start_time} to {end_time} to backfill.")
    client = InfluxDBClient(**CONFIG['influxdb'])
    try:
        actuals_df = get_historical_data(client, start_time, end_time)
        if actuals_df.empty:
            logging.warning("No historical data returned for the backfill period.")
            return

        actuals_df.rename(columns={'aemo_price_sa1': 'price_actual', 'power_load': 'load_actual'}, inplace=True)

        # Use the dataframe index for efficient updating
        log_df.set_index('forecast_target_time', inplace=True)
        if 'price_actual' in actuals_df.columns:
            log_df.update(actuals_df[['price_actual']])
        if 'load_actual' in actuals_df.columns:
            log_df.update(actuals_df[['load_actual']])
        log_df.reset_index(inplace=True)

    finally:
        client.close()

    try:
        log_df.to_csv(log_file_path, index=False)
        logging.info("Successfully saved backfilled data to log file.")
    except Exception as e:
        logging.error(f"Failed to save backfilled log file: {e}")


# --------------------------------------------------------------------------- #
# 6. MAIN EXECUTION BLOCK
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Energy Price & Load Forecasting Pipeline")
    parser.add_argument('mode', choices=['train-price', 'train-load', 'predict-price', 'predict-load', 'update-tariffs', 'backfill-actuals'], help="The mode to run the script in.")
    parser.add_argument('--publish-hass', action='store_true', help="Publish forecasts as attributes to Home Assistant entities.")
    parser.add_argument('--dynamic-handoff', action='store_true', help="For 'predict-price' mode, use Amber's advanced forecast to seed the model.")
    parser.add_argument('--config', default='config.json', help="Path to the configuration file.")
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_config(args.config)

    if args.mode == 'train-price':
        train_single_model('price')
    elif args.mode == 'train-load':
        train_single_model('load')
    elif args.mode == 'predict-price':
        predict_with_model('price', args.publish_hass, args.dynamic_handoff)
    elif args.mode == 'predict-load':
        predict_with_model('load', args.publish_hass)
    elif args.mode == 'update-tariffs':
        update_tariffs()
    elif args.mode == 'backfill-actuals':
        backfill_actuals()

if __name__ == "__main__":
    main()
