#!/usr/bin/env python3

import json
import shutil
import statistics
from datetime import time

def smooth_tariff_profile(file_path: str = "tariff_profile.json"):
    """
    Smooths tariff data by calculating the median value for predefined time buckets
    and applying that median back to all entries in that bucket.

    Time buckets are defined as:
    - Peak: 17:00 - 20:59
    - Solar sponge: 10:00 - 15:59
    - Off peak: All other times

    Args:
        file_path (str): The path to the tariff profile JSON file.
    """
    raw_file_path = file_path.replace(".json", "_raw.json")

    # 1. Save a copy of the original file
    try:
        shutil.copy(file_path, raw_file_path)
        print(f"Successfully created a backup: {raw_file_path}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while creating the backup: {e}")
        return

    # 2. Load the tariff data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Define time buckets
    peak_start = time(17, 0)
    peak_end = time(20, 59)
    solar_start = time(10, 0)
    solar_end = time(15, 59)

    # Process both general and feed-in tariffs
    for tariff_key in ['general_tariff', 'feed_in_tariff']:
        if tariff_key not in data:
            continue

        tariff_data = data[tariff_key]
        
        # Lists to hold values for each bucket
        peak_values = []
        solar_sponge_values = []
        off_peak_values = []

        # 3. Sort values into buckets
        for time_str, value in tariff_data.items():
            current_time = time.fromisoformat(time_str)
            if peak_start <= current_time <= peak_end:
                peak_values.append(value)
            elif solar_start <= current_time <= solar_end:
                solar_sponge_values.append(value)
            else:
                off_peak_values.append(value)

        # 4. Calculate the median for each bucket
        median_peak = round(statistics.median(peak_values), 4) if peak_values else 0
        median_solar = round(statistics.median(solar_sponge_values), 4) if solar_sponge_values else 0
        median_off_peak = round(statistics.median(off_peak_values), 4) if off_peak_values else 0
        
        # 5. Create the new tariff profile with median values
        smoothed_tariff = {}
        for time_str in tariff_data.keys():
            current_time = time.fromisoformat(time_str)
            if peak_start <= current_time <= peak_end:
                smoothed_tariff[time_str] = median_peak
            elif solar_start <= current_time <= solar_end:
                smoothed_tariff[time_str] = median_solar
            else:
                smoothed_tariff[time_str] = median_off_peak
        
        # Update the data with the smoothed profile
        data[tariff_key] = smoothed_tariff

    # 6. Save the new smoothed data back to the original file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully smoothed and updated '{file_path}'")


# --- Example Usage ---
# To run this code, first save your data as `tariff_profile.json`.
# Then, execute this Python script in the same directory.
if __name__ == "__main__":
    smooth_tariff_profile("tariff_profile.json")
