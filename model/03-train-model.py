import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.models import LightGBMModel
from datetime import timedelta

# --- Configuration (Optimized for Stability) ---
TRAINING_DATA_CUTOFF = pd.Timestamp.now(tz='UTC') - timedelta(days=180)

# Parameters optimized for a more stable, less volatile forecast
LGBM_PARAMS = {
    "n_estimators": 150,
    "num_leaves": 31,
    "learning_rate": 0.03,      # Slightly reduced
    "max_depth": 7,             # Prevents trees from getting too complex
    "reg_alpha": 0.1,           # L1 Regularization
    "reg_lambda": 0.1,          # L2 Regularization
    "random_state": 42,
    "n_jobs": -1
}

FORECAST_HORIZON = 144
print("Configuration loaded, optimized for stability.")

# --- Define the function before it is called ---
def create_time_features(df):
    """Creates time series features from a datetime index."""
    df_out = df.copy()
    df_out['hour'] = df_out.index.hour
    df_out['day_of_week'] = df_out.index.dayofweek # Monday=0, Sunday=6
    df_out['day_of_year'] = df_out.index.dayofyear
    df_out['month'] = df_out.index.month
    df_out['is_weekend'] = (df_out.index.dayofweek >= 5).astype(int)
    return df_out


# --- Part 1: Data Preparation & Feature Engineering ---
print("\n--- Part 1: Preparing Data ---")

# 1a. Define Target and Feature Columns
TARGET_COL = 'aemo_price_sa1'
historical_rename_map = {
    'power_pv': 'Home PV Generation',
    'temperature_adelaide': 'Temperature',
    'humidity_adelaide': 'Humidity',
    'wind_speed_adelaide': 'Wind Speed'
}
forecast_rename_map = {
    'forecast_power_pv': 'Home PV Generation',
    'forecast_temperature_adelaide': 'Temperature',
    'forecast_humidity_adelaide': 'Humidity',
    'forecast_wind_speed_adelaide': 'Wind Speed'
}
FEATURE_COLS_RENAMED = list(historical_rename_map.values())
all_feature_cols = FEATURE_COLS_RENAMED + ['hour', 'day_of_week', 'day_of_year', 'month', 'is_weekend']

# 1b. Prepare and Clean Historical Data
print("Processing historical data...")
historical_data = combined_df.rename(columns=historical_rename_map)
if historical_data.index.has_duplicates:
    duplicate_count = historical_data.index.duplicated().sum()
    print(f"Found and removed {duplicate_count} duplicate timestamps from historical data.")
    historical_data = historical_data[~historical_data.index.duplicated(keep='first')]
model_data = historical_data[[TARGET_COL] + FEATURE_COLS_RENAMED].copy()
initial_rows = len(model_data)
model_data.dropna(inplace=True)
print(f"Dropped {initial_rows - len(model_data)} rows with missing values.")

# --- ROBUST LOG TRANSFORM ---
# 1c. Calculate shift value and apply shifted log transform to the target variable
min_price = model_data[TARGET_COL].min()
SHIFT_VALUE = abs(min_price) + 1 if min_price < 0 else 1
print(f"Minimum price is {min_price:.2f}. Using a shift value of {SHIFT_VALUE:.2f} for log transform.")
model_data[TARGET_COL] = np.log(model_data[TARGET_COL] + SHIFT_VALUE)
# --- END ROBUST LOG TRANSFORM ---

model_data_featured = create_time_features(model_data)

# 1d. Prepare Future Data
print("Processing future forecast data...")
future_data = combined_forecast_df.rename(columns=forecast_rename_map)
future_covariates_df_raw = create_time_features(future_data)
future_covariates_df = future_covariates_df_raw[all_feature_cols].copy()

# 1e. Combine and Impute all data to create continuous series
print("Combining all data and filling gaps...")
full_df = pd.concat([model_data_featured, future_covariates_df])
full_df = full_df[~full_df.index.duplicated(keep='first')]
full_range = pd.date_range(start=full_df.index.min(), end=full_df.index.max(), freq='30min')
full_df_reindexed = full_df.reindex(full_range)
print(f"{full_df_reindexed.isnull().sum().sum()} total missing values created by reindexing.")
print("Filling missing values using forward-fill...")
full_df_filled = full_df_reindexed.ffill()
full_df_filled = full_df_filled.bfill()
print("All missing values filled.")

# --- Part 2: Convert to Darts TimeSeries ---
print("\n--- Part 2: Converting to Darts TimeSeries objects ---")
training_data_source = full_df_filled[full_df_filled.index < combined_forecast_df.index.min()]
if TRAINING_DATA_CUTOFF:
    training_data_source = training_data_source[training_data_source.index >= TRAINING_DATA_CUTOFF]
    print(f"Training data trimmed to {len(training_data_source)} records for speed.")

# The target_series is now log-transformed
target_series = TimeSeries.from_series(training_data_source[TARGET_COL], freq='30min')
covariates_ts = TimeSeries.from_dataframe(full_df_filled, value_cols=all_feature_cols, freq='30min')

# --- Part 3: Model Training ---
print("\n--- Part 3: Training LightGBM Model ---")
lgbm = LightGBMModel(
    lags=24 * 7,
    lags_future_covariates=[0],
    output_chunk_length=FORECAST_HORIZON,
    **LGBM_PARAMS
)
print("Fitting the model on log-transformed data...")
lgbm.fit(
    series=target_series,
    future_covariates=covariates_ts
)
print("Model training complete. The model is now stored in the 'lgbm' variable.")
print(f"The SHIFT_VALUE of {SHIFT_VALUE:.2f} must be used in the next cell to reverse the transform.")
