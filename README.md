# AI-Powered Energy Forecast for Home Assistant

Fittingly, most of this code was also AI generated.

This project provides a comprehensive forecasting pipeline for home energy management. It leverages a LightGBM machine learning model (baseline) and a Temporal Fusion Transformer (TFT, in development) to predict future energy load and electricity prices. The script integrates deeply with Home Assistant and InfluxDB to gather data, train models, and publish actionable forecasts back to Home Assistant entities. This enables smarter home automation, such as optimizing battery charging/discharging or running high-consumption appliances during low-price periods.

## Features

*   **Dual Forecasting Models:** Independently predicts both household energy consumption (load) and wholesale electricity prices.
*   **Dynamic Price Forecasting:** Features a unique "dynamic handoff" mode that seeds the price forecast with Amber Electric's high-resolution advanced forecast, using the ML model to predict beyond Amber's horizon.
*   **Rich Data Integration:**
    *   Fetches historical data from an **InfluxDB v1** database.
    *   Pulls future covariate data from **Home Assistant** entities, including:
        *   **Solcast** for solar PV generation forecasts.
        *   Bureau of Meteorology (**BOM**) for weather forecasts (temperature, humidity, wind speed).
        *   **Amber Electric** for real-time and forecast price data.
*   **Automated Tariff Calculation:** Automatically updates and applies complex network tariffs and GST to wholesale price forecasts, providing an accurate final cost.
*   **Command-Line Interface:** A single, robust script (`forecast.py`) with clear modes for training, prediction, and updating tariffs.
*   **External Configuration:** All settings, keys, and paths are managed in an external `config.json` file, keeping secrets out of the main script.

## How It Works

The system operates in a cyclical fashion:

1.  **Data Collection (Past):** Historical data for energy load, PV generation, weather, and AEMO prices is stored in InfluxDB. Continuous Queries are used to automatically downsample raw data into 30-minute averages for training.
2.  **Data Collection (Future):** For predictions, the script calls the Home Assistant API to get the latest forecast data from Solcast, BOM, and Amber Electric.
3.  **Training:** In `train` mode, the script loads years of historical data from InfluxDB to train the LightGBM models for price and load. The trained models (`.pkl`) and their parameters are saved locally. This should be run periodically (e.g., weekly or monthly) to keep the models current.
4.  **Prediction:** In `predict` mode, the script:
    *   Loads the pre-trained model.
    *   Gathers the latest future data from Home Assistant.
    *   Generates a 72-hour forecast at a 30-minute resolution.
    *   Applies tariffs and GST to the price forecast.
    *   Saves the detailed forecast to `predictions.json`.
    *   (Optional) Publishes the forecast to dedicated sensors in Home Assistant. This should be run on a schedule (e.g., every 30 minutes via a cron job).

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.9+
    *   An running InfluxDB v1 instance.
    *   A running Home Assistant instance with the required integrations (Amber Electric, Solcast, a weather provider).

2.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Set up Python Environment:**
    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
    ```

4.  **Configure InfluxDB:**
    Set up your Home Assistant to feed data into InfluxDB. Then, create the continuous queries below in your InfluxDB instance to automatically create the 30-minute summary data required by the script.

5.  **Create Configuration:**
    Copy the example configuration file and edit it with your own details.
    ```bash
    cp config.example.json config.json
    nano config.json
    ```
    *   Fill in your InfluxDB and Home Assistant credentials.
    *   Ensure all `entity_id`s match your Home Assistant setup.

## Usage

All commands are run from the script's directory with the virtual environment activated.

#### 1. Update Tariffs
This fetches the latest tariff information from Amber and creates a 24-hour profile. Run this once to create the initial file, and then periodically if tariffs change.
```bash
python3 forecast.py update-tariffs
```

#### 2. Train the Models
Train the models using your historical data. This can take some time. Run this once initially, and then schedule it to run weekly or monthly.
```bash
# Train the price model
python3 forecast.py train-price

# Train the load model
python3 forecast.py train-load
```

#### 3. Run Predictions
Generate forecasts and publish them to Home Assistant. In production this is split:
the **price** path runs event-driven on Amber APF state changes via
`ai-energy-listener.service` (with a 30-min idle heartbeat as fallback), and the **load**
path runs every 30 min on `ai-energy-predict.timer`.

```bash
# Run all models (price + load) and publish to HA (manual / one-shot)
python3 forecast.py predict-all --publish-hass --dynamic-handoff

# Price only or load only (manual/testing)
python3 forecast.py predict-price --publish-hass --dynamic-handoff
python3 forecast.py predict-load --publish-hass
```
*Omit `--publish-hass` to run locally without writing to HA.*
*Use `--config /path/to/config.json` for a non-default config.*

## Data Pipeline Configuration

#### Home Assistant (`configuration.yaml`)
Your Home Assistant instance should be configured to push data to InfluxDB.
```yaml
influxdb:
  host: YOUR_INFLUXDB_HOST
  port: 8086
  database: hass
  username: user
  password: YOUR_PASSWORD
  max_retries: 3
  default_measurement: state
  tags:
    source: hass
```

#### InfluxDB Continuous Queries
These queries downsample your raw data into the 30-minute intervals the model uses for training.
```sql
CREATE CONTINUOUS QUERY cq_5m_to_30m ON hass BEGIN SELECT mean(mean_value) AS mean_value, min(min_value) AS min_value, max(max_value) AS max_value INTO hass.rp_30m.power_load_30m FROM hass.rp_5m.power_load_5m GROUP BY time(30m), source_metadata_id, entity_id END
CREATE CONTINUOUS QUERY cq_raw_to_5m ON hass BEGIN SELECT mean(value) AS mean_value, min(value) AS min_value, max(value) AS max_value INTO hass.rp_5m.power_load_5m FROM hass.rp_raw.sensor__power WHERE entity_id = 'sigen_plant_consumed_power' GROUP BY time(5m), entity_id END
CREATE CONTINUOUS QUERY cq_power_load_without_deferrable_raw_to_5m ON hass BEGIN SELECT mean(value) AS mean_value, min(value) AS min_value, max(value) AS max_value INTO hass.rp_5m.power_load_without_deferrable_5m FROM hass.rp_raw.sensor__power WHERE entity_id = 'power_consumed_without_deferrable_loads' GROUP BY time(5m), entity_id END
CREATE CONTINUOUS QUERY cq_power_load_without_deferrable_5m_to_30m ON hass BEGIN SELECT mean(mean_value) AS mean_value, min(min_value) AS min_value, max(max_value) AS max_value INTO hass.rp_30m.power_load_without_deferrable_30m FROM hass.rp_5m.power_load_without_deferrable_5m GROUP BY time(30m), entity_id END
CREATE CONTINUOUS QUERY cq_aemo_5m_sa1_to_30m ON hass BEGIN SELECT mean(price) AS price INTO hass.rp_30m.aemo_dispatch_sa1_30m FROM hass.rp_5m.aemo_dispatch_sa1_5m GROUP BY time(30m) END
CREATE CONTINUOUS QUERY cq_weather_temp_30m ON hass BEGIN SELECT mean(value) AS mean_value INTO hass.rp_30m.temperature_adelaide FROM hass.rp_raw.sensor__temperature WHERE entity_id = 'adelaide_west_terrace_ngayirdapira_temp' GROUP BY time(30m), entity_id END
CREATE CONTINUOUS QUERY cq_weather_humidity_30m ON hass BEGIN SELECT mean(value) AS mean_value INTO hass.rp_30m.humidity_adelaide FROM hass.rp_raw.sensor__humidity WHERE entity_id = 'adelaide_west_terrace_ngayirdapira_humidity' GROUP BY time(30m), entity_id END
CREATE CONTINUOUS QUERY cq_weather_wind_30m ON hass BEGIN SELECT mean(value) AS mean_value INTO hass.rp_30m.wind_speed_adelaide FROM hass.rp_raw.sensor__wind_speed WHERE entity_id = 'adelaide_west_terrace_ngayirdapira_wind_speed_kilometre' GROUP BY time(30m), entity_id END
CREATE CONTINUOUS QUERY cq_pv_5m_to_30m ON hass BEGIN SELECT mean(mean_value) AS mean_value, min(min_value) AS min_value, max(max_value) AS max_value INTO hass.rp_30m.power_pv_30m FROM hass.rp_5m.power_pv_5m GROUP BY time(30m), source_metadata_id, entity_id END
CREATE CONTINUOUS QUERY cq_pv_raw_to_5m ON hass RESAMPLE FOR 1d BEGIN SELECT mean(value) AS mean_value, min(value) AS min_value, max(value) AS max_value INTO hass.rp_5m.power_pv_5m FROM hass.rp_raw.sensor__power WHERE entity_id = 'sensor.solcast_pv_forecast_power_now' GROUP BY time(5m), entity_id fill(0) END
CREATE CONTINUOUS QUERY cq_aemo_raw_sa1_to_5m ON hass BEGIN SELECT mean(value) * 1000 AS price INTO hass.rp_5m.aemo_dispatch_sa1_5m FROM hass.rp_raw.sensor__monetary WHERE entity_id = 'aemo_5min_current_price_sa' GROUP BY time(5m) END
CREATE CONTINUOUS QUERY cq_dump_load_raw_to_5m ON hass BEGIN SELECT mean(value) AS mean_value, min(value) AS min_value, max(value) AS max_value INTO hass.rp_5m.power_dump_load_5m FROM hass.rp_raw.sensor__power WHERE entity_id = 'estimated_dump_load_power' GROUP BY time(5m), entity_id END
CREATE CONTINUOUS QUERY cq_dump_load_5m_to_30m ON hass BEGIN SELECT mean(mean_value) AS mean_value, min(min_value) AS min_value, max(max_value) AS max_value INTO hass.rp_30m.power_dump_load_30m FROM hass.rp_5m.power_dump_load_5m GROUP BY time(30m), entity_id END
```

## Status and Next Work

**2026-06-15.** Production price source is the APF/LightGBM incumbent
(`amber_apf_lgbm`, logged in `price_forecast_log.csv` as `model_name='price'`).
Work on the APF-free replacement paths (`pd_direct`, `model_a_hybrid`,
`lgbm_strategic`) is suspended, not fully abandoned. They remain in the repo for
reference and possible deliberate revival, but they are not currently trusted as
replacement paths and should not be used as evidence about APF extrapolation.

The current price-forecast source contract lives in
`docs/price_forecast_sources.md` and `eval/price_source_contracts.py`.
Historical APF-free plans and abandonment notes live in `docs/roadmap.md`.
Structural critique of the TFT line is in `docs/tft_price_forecast.md`.

Near-term work, in order:

-   **APF-tail residual work:** evaluate improvements to the existing
    APF-backed extrapolation path, currently the STPASA tail residual-correction
    probe over `28.5-72h`.
-   **Source-contract hygiene:** keep eval scripts explicit about whether they
    are scoring `amber_apf_lgbm` or an APF-free suspended path.
-   **Production switchability:** keep the HA price-source plumbing useful for
    controlled comparisons, but do not treat suspended paths as promotion
    candidates without a fresh trust/revival decision.

Older infrastructure work that remains relevant regardless of which forecast wins:

-   **Holistic dispatch simulation (Phase 6):** the eval framework (`rolling_mpc_eval.py`,
    Window A/B tariffed gates, Amber yardstick) is what grades all of the above.
-   **Test framework (Phase 8):** regression tests against canned fixtures.
-   **Event-driven service (Phase 7):** ~~replace systemd timers with a persistent process
    using HA WebSocket subscriptions.~~ **Initial scope landed 2026-05-27** as
    `ai-energy-listener.service`, which drives `predict-price` on Amber APF state changes;
    see [docs/event_driven_predict_price_plan.md](docs/event_driven_predict_price_plan.md).
    Remaining work would be event-driving more of the pipeline if a use-case emerges.

## Acknowledgements
The initial version of the core `forecast.py` script was generated with assistance from Google's Gemini.
