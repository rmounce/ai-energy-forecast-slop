# The trained 'lgbm' model, 'training_data_source', 'full_df_filled', 'combined_forecast_df', 
# 'TARGET_COL', 'covariates_ts', and SHIFT_VALUE must be available from the previous cell.

# --- Configuration ---
PRICE_CAP_KWH = 17.10 # Market Price Cap
PRICE_FLOOR_KWH = -1.00 # Market Floor Price

# --- Part 4: Generate Forecast with Dynamic Handoff and Constraints ---
print('\n--- Part 4: Generating Forecast with Dynamic Logic ---')

# 4a. Dynamically determine the handoff point from the Amber forecast
print("Finding last valid data point in Amber forecast...")
last_good_amber_index = combined_forecast_df['advanced_price_forecast'].last_valid_index()

if last_good_amber_index is not None:
    amber_forecast_df = combined_forecast_df.loc[:last_good_amber_index, ['advanced_price_forecast']].copy()
    amber_forecast_df.rename(columns={'advanced_price_forecast': TARGET_COL}, inplace=True)
    print(f"Using Amber forecast for {len(amber_forecast_df) / 2} hours (up to {last_good_amber_index}).")
else:
    amber_forecast_df = pd.DataFrame()
    print("Warning: No valid Amber forecast data found.")

# 4b. Prepare the "pseudo-history" using the dynamic Amber forecast
pseudo_history_df = training_data_source.copy() # This is already log-transformed
if not amber_forecast_df.empty:
    amber_covariates = full_df_filled.loc[amber_forecast_df.index]
    amber_seed_data = amber_covariates.copy()
    # Apply the same log transform to the Amber data before appending
    amber_seed_data[TARGET_COL] = np.log(amber_forecast_df[TARGET_COL] + SHIFT_VALUE)
    pseudo_history_df = pd.concat([pseudo_history_df, amber_seed_data])
pseudo_history_ts = TimeSeries.from_series(pseudo_history_df[TARGET_COL], freq='30min')
print(f"Log-transformed pseudo-history created, ending at {pseudo_history_ts.end_time()}.")

# 4c. Calculate how many steps to predict, respecting the weather forecast length
desired_steps = FORECAST_HORIZON - len(amber_forecast_df)
max_possible_steps = len(covariates_ts) - len(pseudo_history_ts)
steps_to_predict = max(0, min(desired_steps, max_possible_steps))
print(f"--> Will predict {steps_to_predict / 2} hours.")

# 4d. Perform the "one-go" forecast for the constrained number of steps
if steps_to_predict > 0:
    remaining_forecast_series = lgbm.predict(
        n=steps_to_predict,
        series=pseudo_history_ts,
        future_covariates=covariates_ts
    )
    remaining_forecast_df = pd.DataFrame(
        remaining_forecast_series.values(),
        index=remaining_forecast_series.time_index,
        columns=[TARGET_COL] # This column currently contains LOG-TRANSFORMED data
    )
    if remaining_forecast_df.index.tzinfo is None:
        remaining_forecast_df.index = remaining_forecast_df.index.tz_localize('UTC')
else:
    remaining_forecast_df = pd.DataFrame()

# 4e. Combine the forecasts and reverse the transformation
print("Combining forecasts and reversing the log transform...")
if not amber_forecast_df.empty:
    # We use the original, untransformed Amber data for the final result
    final_forecast_df = pd.concat([amber_forecast_df, remaining_forecast_df])
else: 
    final_forecast_df = remaining_forecast_df

# Reverse the transform on the entire series using the SHIFT_VALUE from Cell 1
if not final_forecast_df.empty:
    # The Amber part is already in real units, the predicted part is in log-space.
    # We only need to reverse the transform on the predicted part.
    if not remaining_forecast_df.empty:
        final_forecast_df.loc[remaining_forecast_df.index, TARGET_COL] = np.exp(remaining_forecast_df[TARGET_COL]) - SHIFT_VALUE

# As a final safety net, clip the forecast to the absolute market limits
if not final_forecast_df.empty:
    print("Clipping final forecast to market price limits...")
    final_forecast_df[TARGET_COL] = final_forecast_df[TARGET_COL].clip(
        lower=PRICE_FLOOR_KWH, upper=PRICE_CAP_KWH
    )
print("Final forecast prepared successfully.")

# --- Part 5: Visualize the Final, Data-Aware Forecast ---
print("\n--- Part 5: Visualizing Results ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(18, 8))

if not final_forecast_df.empty:
    historical_plot_start = final_forecast_df.index.min() - timedelta(days=3)
    # Note: We need to use the original, untransformed `combined_df` for plotting historical actuals
    untransformed_history = combined_df.rename(columns=historical_rename_map)
    untransformed_history[TARGET_COL][historical_plot_start:final_forecast_df.index.min()].plot(
        ax=ax, label="Historical Actual Price", color='gray', alpha=0.8
    )
    final_forecast_df[TARGET_COL].plot(ax=ax, label=f"Final Forecast ({len(final_forecast_df) / 2} hours)", color='purple', lw=2)
    if not amber_forecast_df.empty:
        ax.axvline(amber_forecast_df.index.max(), color='red', linestyle='--', label='Handoff to LGBM')
else:
    print("Final forecast is empty, skipping plot.")

ax.set_title(f"AEMO SA1 Spot Price: Stable Forecast", fontsize=16)
ax.set_xlabel("Date / Time", fontsize=12)
ax.set_ylabel("Spot Price ($/kWh)", fontsize=12)
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

if not final_forecast_df.empty:
    print("\n--- Forecast Summary ---")
    print(f"Total forecast length: {len(final_forecast_df) / 2} hours")
    print(final_forecast_df.describe())
