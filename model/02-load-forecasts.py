import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Home Assistant Configuration
HA_URL = "https://REDACTED"  # Change to your HA URL
HA_TOKEN = "REDACTED"  # Get from HA Profile > Long-lived access tokens

# Configuration constants
ADELAIDE_TZ = pytz.timezone('Australia/Adelaide')
WEATHER_ENTITY_ID = "weather.adelaide_hourly"
AMBER_ENTITY_ID = "sensor.amber_30min_forecasts_general_price"
SOLCAST_ENTITIES = [
    "sensor.solcast_pv_forecast_forecast_today",
    "sensor.solcast_pv_forecast_forecast_tomorrow", 
    "sensor.solcast_pv_forecast_forecast_day_3",
    "sensor.solcast_pv_forecast_forecast_day_4"
]

# Headers for API requests
headers = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json"
}

def call_ha_service(service_domain: str, service_name: str, data: Dict = None, 
                   target: Dict = None, return_response: bool = False) -> Dict:
    """
    Call a Home Assistant service with improved error handling.
    """
    url = f"{HA_URL}/api/services/{service_domain}/{service_name}"
    if return_response:
        url += "?return_response=true"
    
    payload = {}
    if data:
        payload.update(data)
    if target: 
        payload["target"] = target
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {url} - {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        raise

def get_entity_state(entity_id: str) -> Dict:
    """Get the state and attributes of an entity with error handling."""
    url = f"{HA_URL}/api/states/{entity_id}"
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get entity state for {entity_id}: {e}")
        raise

def interpolate_weather_to_30min(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate hourly weather data to 30-minute intervals using appropriate methods.
    
    Args:
        weather_df: DataFrame with hourly weather data
        
    Returns:
        DataFrame with 30-minute intervals
    """
    if weather_df.empty:
        return weather_df
    
    logger.info(f"Interpolating weather data from {len(weather_df)} hourly to 30-minute intervals")
    
    # Create 30-minute time index
    start_time = weather_df.index.min()
    end_time = weather_df.index.max()
    thirty_min_index = pd.date_range(start=start_time, end=end_time, freq='30min')
    
    # Reindex to 30-minute intervals
    weather_30min = weather_df.reindex(thirty_min_index)
    
    # Apply different interpolation methods based on variable type
    for column in weather_30min.columns:
        if 'temperature' in column.lower():
            # Linear interpolation for temperature (gradual changes)
            weather_30min[column] = weather_30min[column].interpolate(method='linear')
        elif 'humidity' in column.lower():
            # Linear interpolation for humidity (gradual changes)
            weather_30min[column] = weather_30min[column].interpolate(method='linear')
        elif 'wind' in column.lower():
            # Forward fill for wind speed (more volatile, step-like changes)
            weather_30min[column] = weather_30min[column].fillna(method='ffill')
        else:
            # Default to linear interpolation for other variables
            weather_30min[column] = weather_30min[column].interpolate(method='linear')
    
    # Fill any remaining NaN values at the beginning with forward fill
    weather_30min = weather_30min.fillna(method='ffill')
    
    logger.info(f"Interpolated to {len(weather_30min)} records at 30-minute intervals")
    return weather_30min

def get_weather_forecast() -> pd.DataFrame:
    """Get BOM weather forecast data and interpolate to 30-minute intervals."""
    logger.info("Retrieving BOM weather forecast...")
    
    try:
        # Call the Home Assistant weather service
        forecast_data_response = call_ha_service(
            service_domain="weather",
            service_name="get_forecasts",
            data={"type": "hourly", "entity_id": WEATHER_ENTITY_ID},
            return_response=True 
        )
        
        # Parse the forecast data
        service_response_data = forecast_data_response.get("service_response", {})
        forecasts = service_response_data.get(WEATHER_ENTITY_ID, {}).get("forecast", [])
        
        if not forecasts:
            logger.warning("No weather forecast data found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        weather_df = pd.DataFrame(forecasts)
        
        # Process datetime column
        weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
        
        # Handle timezone conversion properly
        if weather_df['datetime'].dt.tz is None:
            # If naive datetime, assume it's in Adelaide time
            weather_df['datetime'] = weather_df['datetime'].dt.tz_localize(ADELAIDE_TZ)
        
        # Convert to UTC and round to nearest minute
        weather_df['datetime'] = weather_df['datetime'].dt.tz_convert('UTC').dt.round('min')
        weather_df.set_index('datetime', inplace=True)
        
        # Filter for next 3 days
        current_time = datetime.now(pytz.UTC)
        end_time = current_time + timedelta(days=3)
        weather_df = weather_df[weather_df.index <= end_time]
        
        # Select and rename columns
        weather_columns = {
            'temperature': 'forecast_temperature_adelaide',
            'humidity': 'forecast_humidity_adelaide',
            'wind_speed': 'forecast_wind_speed_adelaide'
        }
        
        # Keep only available columns and rename them
        available_columns = [col for col in weather_columns.keys() if col in weather_df.columns]
        weather_df = weather_df[available_columns].copy()
        weather_df.columns = [weather_columns[col] for col in available_columns]
        
        # Interpolate to 30-minute intervals
        weather_df = interpolate_weather_to_30min(weather_df)
        
        logger.info(f"Retrieved {len(weather_df)} weather forecast records (30-min intervals)")
        return weather_df
        
    except Exception as e:
        logger.error(f"Error retrieving weather forecast: {e}")
        return pd.DataFrame()

def get_amber_price_forecast() -> pd.DataFrame:
    """Get Amber Electric price forecast data with improved error handling."""
    logger.info("Retrieving Amber Electric price forecast...")
    
    try:
        # Get entity state and attributes
        entity_state = get_entity_state(AMBER_ENTITY_ID)
        
        # Extract forecasts from attributes
        forecasts = None
        if "attributes" in entity_state:
            attrs = entity_state["attributes"]
            for attr_name in ["forecasts", "Forecasts", "forecast"]:
                if attr_name in attrs:
                    forecasts = attrs[attr_name]
                    break
        
        if not forecasts:
            logger.warning("No Amber price forecast data found in entity attributes")
            return pd.DataFrame()
        
        # Process forecast data
        processed_forecasts = []
        
        for forecast in forecasts:
            try:
                # Parse start time and round to nearest minute
                start_time = pd.to_datetime(forecast.get("start_time"))
                if start_time.tz is None:
                    start_time = start_time.tz_localize('UTC')
                else:
                    start_time = start_time.tz_convert('UTC')
                start_time = start_time.round('min')
                
                # --- FIX IS HERE: Change the default from 0.0 to np.nan ---
                def safe_float(value, default=np.nan):
                    """Safely convert value to float, defaulting to NaN for missing values."""
                    if value is None:
                        return default
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                # --- FIX ENDS HERE ---
                
                per_kwh_incl_gst = safe_float(forecast.get("per_kwh"))
                per_kwh = per_kwh_incl_gst / 1.1 if not np.isnan(per_kwh_incl_gst) else np.nan
                
                spot_per_kwh = safe_float(forecast.get("spot_per_kwh"))
                
                # This will now correctly become NaN if the source is null
                advanced_price_incl_gst = safe_float(forecast.get("advanced_price_predicted"))
                advanced_price_predicted = advanced_price_incl_gst / 1.1 if not np.isnan(advanced_price_incl_gst) else np.nan
                
                renewables_val = safe_float(forecast.get("renewables"))
                
                # Calculate derived values
                # These will correctly result in NaN if any component is NaN
                tariff = per_kwh - spot_per_kwh
                advanced_price_no_tariff = advanced_price_predicted - tariff
                
                processed_forecasts.append({
                    'datetime': start_time,
                    'aemo_predispatch_forecast': spot_per_kwh,
                    'tariff': tariff,
                    'advanced_price_forecast': advanced_price_no_tariff,
                    'renewables': renewables_val,
                    'spike_status': forecast.get("spike_status", "none"),
                    'descriptor': forecast.get("descriptor", "unknown")
                })
                
            except Exception as e:
                logger.warning(f"Error processing forecast entry: {e}")
                continue
        
        if not processed_forecasts:
            logger.warning("No valid forecast entries processed")
            return pd.DataFrame()
        
        # Convert to DataFrame
        price_df = pd.DataFrame(processed_forecasts)
        price_df.set_index('datetime', inplace=True)
        
        # Filter for next 36 hours (can be generous, as we'll drop NaNs later)
        current_time = datetime.now(pytz.UTC)
        end_time = current_time + timedelta(hours=36)
        price_df = price_df[price_df.index <= end_time]
        
        logger.info(f"Retrieved {len(price_df)} price forecast records, including potential NaNs.")
        return price_df
        
    except Exception as e:
        logger.error(f"Error retrieving Amber price forecast: {e}")
        return pd.DataFrame()

def get_solcast_forecast() -> pd.DataFrame:
    """Get Solcast solar forecast data with improved error handling."""
    logger.info("Retrieving Solcast solar forecast...")
    
    try:
        all_forecasts = []
        current_time = datetime.now(pytz.UTC)
        
        for entity_id in SOLCAST_ENTITIES:
            try:
                entity_state = get_entity_state(entity_id)
                detailed_forecast = entity_state.get("attributes", {}).get("detailedForecast", [])
                
                for forecast in detailed_forecast:
                    try:
                        period_start_str = forecast.get("period_start")
                        if not period_start_str:
                            continue
                        
                        # Parse and normalize datetime
                        period_start = pd.to_datetime(period_start_str)
                        if period_start.tz is None:
                            period_start = period_start.tz_localize('UTC')
                        else:
                            period_start = period_start.tz_convert('UTC')
                        period_start = period_start.round('min')
                        
                        # Only include future forecasts
                        if period_start <= current_time:
                            continue
                        
                        # Extract and process solar data
                        def safe_float_solar(value, default=0.0):
                            """Safely convert solar value to float and handle scaling."""
                            if value is None:
                                return default
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                return default
                        
                        pv_estimate = safe_float_solar(forecast.get("pv_estimate"))
                        pv_estimate_watts = int(pv_estimate * 1000)  # Convert kW to W
                        
                        pv_estimate_10 = safe_float_solar(forecast.get("pv_estimate10")) * 1000
                        pv_estimate_90 = safe_float_solar(forecast.get("pv_estimate90")) * 1000
                        
                        all_forecasts.append({
                            'datetime': period_start,
                            'forecast_power_pv': pv_estimate_watts,
                            'pv_estimate_10': int(pv_estimate_10),
                            'pv_estimate_90': int(pv_estimate_90)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing solar forecast entry: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Could not retrieve {entity_id}: {e}")
                continue
        
        if not all_forecasts:
            logger.warning("No Solcast forecast data found")
            return pd.DataFrame()
        
        # Convert to DataFrame
        solar_df = pd.DataFrame(all_forecasts)
        solar_df.set_index('datetime', inplace=True)
        solar_df = solar_df.sort_index()
        
        # Remove duplicates (keep first occurrence)
        solar_df = solar_df[~solar_df.index.duplicated(keep='first')]
        
        # Filter for next 4 days
        end_time = current_time + timedelta(days=4)
        solar_df = solar_df[solar_df.index <= end_time]
        
        logger.info(f"Retrieved {len(solar_df)} solar forecast records")
        return solar_df
        
    except Exception as e:
        logger.error(f"Error retrieving Solcast forecast: {e}")
        return pd.DataFrame()

def test_ha_connection() -> bool:
    """Test connection to Home Assistant."""
    try:
        response = requests.get(f"{HA_URL}/api/", headers=headers, timeout=30)
        response.raise_for_status()
        logger.info("Connected to Home Assistant successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Home Assistant: {e}")
        logger.error("Please check your HA_URL and HA_TOKEN configuration")
        return False

def combine_forecast_data(forecast_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all forecast dataframes with proper handling of different time resolutions.
    
    Args:
        forecast_dataframes: Dictionary of dataframes to combine
        
    Returns:
        Combined DataFrame with outer join
    """
    if not forecast_dataframes:
        return pd.DataFrame()
    
    logger.info("Combining forecast data...")
    
    # Start with the first dataframe
    combined_df = list(forecast_dataframes.values())[0].copy()
    
    # Join all other dataframes using outer join
    for name, df in list(forecast_dataframes.items())[1:]:
        combined_df = combined_df.join(df, how='outer')
    
    # Sort by datetime index
    combined_df = combined_df.sort_index()
    
    # Filter out data older than 30 minutes
    current_time = datetime.now(pytz.UTC)
    cutoff_time = current_time - timedelta(minutes=30)
    combined_df = combined_df[combined_df.index >= cutoff_time]
    
    # Back-fill missing values (mainly for Amber forecast alignment)
    combined_df = combined_df.fillna(method='bfill')
    
    logger.info(f"Combined forecast shape: {combined_df.shape}")
    logger.info(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
    
    return combined_df

def print_forecast_summary(forecast_dataframes: Dict[str, pd.DataFrame], 
                         combined_df: pd.DataFrame) -> None:
    """Print a comprehensive summary of forecast data."""
    print("\n" + "=" * 70)
    print("FORECAST DATA SUMMARY")
    print("=" * 70)
    
    if not forecast_dataframes:
        print("No forecast data was retrieved.")
        return
    
    # Individual dataframe summary
    print("\nIndividual Forecast Sources:")
    print("-" * 70)
    for name, df in forecast_dataframes.items():
        if not df.empty:
            print(f"{name:15} : {len(df):6} records | {df.index.min()} to {df.index.max()}")
            print(f"{'':15}   Columns: {', '.join(df.columns.tolist())}")
        else:
            print(f"{name:15} : No data retrieved")
    
    # Combined dataframe summary
    if not combined_df.empty:
        print(f"\nCombined Forecast:")
        print("-" * 70)
        print(f"Shape: {combined_df.shape}")
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")
        print(f"Columns: {', '.join(combined_df.columns.tolist())}")
        
        # Missing data analysis
        print(f"\nMissing Data Analysis:")
        print("-" * 70)
        missing_data = combined_df.isnull().sum()
        missing_percentage = (missing_data / len(combined_df)) * 100
        
        for col in combined_df.columns:
            missing_count = missing_data[col]
            missing_pct = missing_percentage[col]
            print(f"{col:35} : {missing_count:5} missing ({missing_pct:5.1f}%)")
        
        # Data quality indicators
        print(f"\nData Quality Indicators:")
        print("-" * 70)
        total_records = len(combined_df)
        complete_records = len(combined_df.dropna())
        print(f"Total records: {total_records}")
        print(f"Complete records (no missing): {complete_records} ({complete_records/total_records*100:.1f}%)")
        
        # Time coverage analysis
        time_diff = combined_df.index.max() - combined_df.index.min()
        print(f"Time coverage: {time_diff}")
        
        expected_30min_records = int(time_diff.total_seconds() / 1800) + 1  # 30 minutes = 1800 seconds
        coverage_pct = (total_records / expected_30min_records) * 100
        print(f"Expected 30-min records: {expected_30min_records}")
        print(f"Coverage: {coverage_pct:.1f}%")
    
    print("\n" + "=" * 70)

def main():
    """Main execution function."""
    logger.info("Starting forecast data retrieval...")
    
    # Test connection
    if not test_ha_connection():
        return None
    
    # Dictionary to store forecast DataFrames
    forecast_dataframes = {}
    
    # Get weather forecast (with interpolation)
    weather_df = get_weather_forecast()
    if not weather_df.empty:
        forecast_dataframes['weather'] = weather_df
    
    # Get price forecast
    price_df = get_amber_price_forecast()
    if not price_df.empty:
        forecast_dataframes['price'] = price_df
    
    # Get solar forecast
    solar_df = get_solcast_forecast()
    if not solar_df.empty:
        forecast_dataframes['solar'] = solar_df
    
    # Combine all forecast data
    combined_forecast_df = combine_forecast_data(forecast_dataframes)
    
    # Print comprehensive summary
    print_forecast_summary(forecast_dataframes, combined_forecast_df)
    
    logger.info("Forecast data retrieval complete!")
    
    return {
        'individual': forecast_dataframes,
        'combined': combined_forecast_df
    }

# Main execution
if __name__ == "__main__":
    # Execute main function
    result = main()
    
    if result:
        # Make dataframes available in global scope for interactive use
        forecast_dataframes = result['individual']
        combined_forecast_df = result['combined']
        
        print("\nDataFrames available:")
        print("- forecast_dataframes: Dict of individual forecast DataFrames")
        print("- combined_forecast_df: Combined forecast DataFrame")
        print("\nReady for ML forecasting pipeline!")
    else:
        print("Failed to retrieve forecast data. Check logs for details.")
