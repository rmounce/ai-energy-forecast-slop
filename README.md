# AI-Powered Energy Forecast for Home Assistant

Fittingly, most of this code was also AI generated.

This project provides the price and household-load forecasts used by an EMHASS home-energy
optimisation stack. Production uses LightGBM quantile models. Historical TFT, PD-direct, tactical,
and other APF-free experiments remain in the repository as evaluation evidence, but are not active
forecast publishers.

## Production Summary — 2026-08-10

- MPC: 14h × 5-min; reads Amber forecasts directly. No AI price inference.
- Day-ahead price: 72h × 30-min; Amber APF near horizon, LightGBM extrapolation to 72h.
- Day-ahead load: 72h × 30-min; LightGBM base-load p65, with planned HWC load added by HA.
- Price refresh: HA WebSocket event via `ai-energy-listener.service`; 30-min idle heartbeat.
- Load refresh: `ai-energy-predict.timer` at `:01` and `:31`.
- Weekly training creates versioned candidates and never auto-promotes them. See the
  [production runbook](docs/price/production_hardening_runbook.md).

Canonical current-state reference:
[docs/prod_pipeline_critical_path.md](docs/prod_pipeline_critical_path.md).

Pipeline consolidation work: [plan](docs/energy_pipeline_architecture_plan.md),
[offline payload replay](docs/energy_pipeline_payload_replay.md),
[resident price shadow](docs/energy_pipeline_resident_price.md),
[current checkpoint](docs/energy_pipeline_source_cache.md). Production routing remains unchanged;
long-run memory validation and complete upstream freshness metadata remain migration gates.
Shadow [result acceptance](docs/energy_pipeline_price_acceptance.md) rejects changed/expired inputs;
production publication remains with the incumbent listener.
Each shadow run uses one [frozen tariff profile](docs/energy_pipeline_tariff_snapshot.md).
Optional `--state-file` saves a [durable accepted shadow checkpoint](docs/energy_pipeline_accepted_store.md);
recovery requires fresh reconciliation and never restores publication authority.
Optional `--publication-db` rehearses [publication recovery locally](docs/energy_pipeline_publication.md)
with SQLite outputs and fresh input checks; no HA writes. Optional `--handoff-shadow`
builds [DH/MPC payloads from a frozen bundle view](docs/energy_pipeline_handoff.md); no solver requests.
[Weather admission](docs/energy_pipeline_weather_coverage.md) preserves incumbent tail fill within a one-hour cap.
Weather capture checks [hourly freshness markers](docs/energy_pipeline_bom_freshness.md) for observed
updates; the HA integration patch remains unapplied, so live provider clocks remain unknown.

## Features

*   **Dual Forecasting Surfaces:** Independently predicts household base load and the 72-hour wholesale-price curve.
*   **Dynamic Price Forecasting:** Features a unique "dynamic handoff" mode that seeds the price forecast with Amber Electric's high-resolution advanced forecast, using the ML model to predict beyond Amber's horizon.
*   **Rich Data Integration:**
    *   Fetches historical data from an **InfluxDB v1** database.
    *   Pulls future covariate data from **Home Assistant** entities, including:
        *   **Solcast** for solar PV generation forecasts.
        *   Bureau of Meteorology (**BOM**) for weather forecasts (temperature, humidity, wind speed).
        *   **Amber Electric** for real-time and forecast price data.
*   **Automated Tariff Calculation:** Automatically updates and applies complex network tariffs and GST to wholesale price forecasts, providing an accurate final cost.
*   **Command-Line Interface:** A single, robust script (`forecast.py`) with clear modes for training, prediction, and updating tariffs.
*   **External Configuration:** All settings, keys, and paths are managed in an external `config.yaml` file, keeping secrets out of the main script.

## How It Works

The system operates in a cyclical fashion:

1.  **Data Collection (Past):** Historical data for energy load, PV generation, weather, and AEMO prices is stored in InfluxDB. Continuous Queries are used to automatically downsample raw data into 30-minute averages for training.
2.  **Data Collection (Future):** For predictions, the script calls the Home Assistant API to get the latest forecast data from Solcast, BOM, and Amber Electric.
3.  **Training:** The weekly systemd job loads up to two years of historical data from InfluxDB and trains the LightGBM price and load quantiles. The current implementation writes the live `.pkl` artifacts directly; do not treat a completed retrain as proof of improvement.
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
    cp config.example.yaml config.yaml
    nano config.yaml
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
# Train isolated candidates (the weekly timer uses these commands)
./forecast.py train-price-candidate
./forecast.py train-load-candidate

# Promotion is an explicit operator decision; inspect the candidate report first.
./forecast.py validate-bundle --family price --bundle <id>
# screening.json identifies a SHA-256-hashed CSV/Parquet row file; see the runbook.
./forecast.py screen-bundle --family price --bundle <id> --metrics screening.json
./forecast.py promote-bundle --family price --bundle <id>
./forecast.py rollback-bundle --family price
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
*Use `--config /path/to/config.yaml` for a non-default config.*

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
CREATE CONTINUOUS QUERY cq_pv_raw_to_5m ON hass RESAMPLE FOR 1d BEGIN SELECT mean(value) AS mean_value, min(value) AS min_value, max(value) AS max_value INTO hass.rp_5m.power_pv_5m FROM hass.rp_raw.sensor__power WHERE entity_id = 'solcast_pv_forecast_power_now' GROUP BY time(5m), entity_id fill(0) END
CREATE CONTINUOUS QUERY cq_aemo_raw_sa1_to_5m ON hass BEGIN SELECT mean(value) * 1000 AS price INTO hass.rp_5m.aemo_dispatch_sa1_5m FROM hass.rp_raw.sensor__monetary WHERE entity_id = 'aemo_5min_current_price_sa' GROUP BY time(5m) END
CREATE CONTINUOUS QUERY cq_dump_load_raw_to_5m ON hass BEGIN SELECT mean(value) AS mean_value, min(value) AS min_value, max(value) AS max_value INTO hass.rp_5m.power_dump_load_5m FROM hass.rp_raw.sensor__power WHERE entity_id = 'estimated_dump_load_power' GROUP BY time(5m), entity_id END
CREATE CONTINUOUS QUERY cq_dump_load_5m_to_30m ON hass BEGIN SELECT mean(mean_value) AS mean_value, min(min_value) AS min_value, max(max_value) AS max_value INTO hass.rp_30m.power_dump_load_30m FROM hass.rp_5m.power_dump_load_5m GROUP BY time(30m), entity_id END
```

## Status and Next Work

- Fresh technique assessment: [price/load accuracy review, 2026-10-04](docs/forecast_accuracy_review_2026-10-04.md).
  Ranked APF-assisted/APF-free and load experiments; evaluation exports need refresh before new comparisons.

- Keep `amber_apf_lgbm` as the production price source.
- APF-free price research is paused. Revival requires a written hypothesis and fixed evaluation
  matrix; see [docs/price/README.md](docs/price/README.md).
- TFT-load is a suspended historical candidate, not a live shadow; see
  [docs/tft_load_forecast.md](docs/tft_load_forecast.md).
- Remaining hardening risk: candidate quality reports are deliberately ineligible until identical-row
  incumbent metrics and inference smoke evidence are supplied. See the production runbook.

## Acknowledgements
The initial version of the core `forecast.py` script was generated with assistance from Google's Gemini.

Isolated solver checkpoint: [audit and result contract](docs/energy_pipeline_solver_isolation.md).
Recorded core solves and [historical DH→MPC chain](docs/energy_pipeline_solver_chain.md) verified;
[economic replay gaps/PV provenance and next work](docs/economic_replay_checkpoint_2026-10-04.md).
[Seven-day measured targets and causal load-calibration trial](docs/measured_economic_actuals_2026-10-04.md)
now available; longer-horizon p65 MAE improves 16–21%, no measured savings yet.
[Parallel economic comparisons](docs/parallel_economic_findings_2026-10-04.md): observed tariff
accounting, fixed-endpoint load and fixed-forecast terminal sensitivities; APF archive pilot.
[Sequential one-hour MPC comparison](docs/sequential_economic_replay_2026-10-04.md) now scored
against common measurements: near-zero load-calibration gain here; baseline execution mismatch open.
[Control fidelity audit](docs/control_fidelity_audit_2026-10-04.md) reproduces three live commands;
identifies minute cadence and refreshed DH/HWC parents as major replay differences.
[Minute replay pilots](docs/minute_economic_replay_2026-10-04.md) now preserve inventory and
recorded activation delays; no meaningful terminal-lock-in gain in the bounded30m comparison.
[Regime selection and DH source admission](docs/economic_regimes_and_dh_admission_2026-10-04.md)
now freeze three stress windows; four reconstructed DH origins have load timestamps one half-hour
behind price/PV despite equal array lengths. Input alignment precedes new model comparisons.
[Own-battery DH feedback pilots](docs/dh_feedback_economic_replay_2026-10-04.md) now retain aligned
parents across rollover; two stress subwindows show no terminal-lock-in gain. HWC remains exogenous;
next causal net-energy calibration and export device-delivery diagnosis.
[Causal load feedback comparison](docs/causal_load_feedback_results_2026-10-04.md) now completes
72 core solves: forecast/path changes but no 15-minute cash/inventory gain. Quantile loss is mixed;
[Verified continuation](docs/continued_feedback_and_delivery_2026-10-04.md) extends export execution
to40min/96 solves: still zero cash/inventory gain. Boundary timing and conversion residual explain
parts of the export-delivery discrepancy. [Recorded EMS delivery replay](docs/ems_delivery_fidelity_2026-10-04.md)
reduces net-export error≈87%. [Energy reconciliation and timing pilot](docs/ems_energy_reconciliation_2026-10-05.md)
identify a moving BMS SoC denominator and omitted DC overhead. Holding accepted commands adds
~5c credit but spends~0.20kWh inventory. [Timing with own optimizer feedback](docs/ems_timing_feedback_2026-10-05.md)
now completes30min/72 solves: similar tradeoff, no demonstrated net gain; longer inventory use remains the gate.
[Installed MPC formatter parity](docs/mpc_publication_audit_2026-10-05.md) passes all60 saved MPC cases;
missing-anchor clock reconstruction remains a separate diagnostic before longer replay admission.
