# System Architecture

This document describes the end-to-end architecture of the AI energy forecasting pipeline. It is intended as a living reference for understanding how all the pieces fit together.

---

## Overview

The pipeline provides 72-hour forecasts of electricity prices and household power consumption for input into [EMHASS](https://emhass.readthedocs.io/), an energy management optimiser. EMHASS produces a battery charge/discharge schedule which Home Assistant automation then executes on a SiG Energy (Sigenergy) inverter/battery system.

The system is located in Adelaide, South Australia (AEMO region SA1), and uses Amber Electric as the retail electricity provider.

---

## High-Level Data Flow

```
External Data Sources                    Internal Data Store
─────────────────────                    ───────────────────
AEMO (nemosis / NEMWeb)  ──┐
HA SQLite (statistics)   ──┤──► ingest/*.py ──► InfluxDB (hass db)
Amber Electric CSV       ──┘                         │
                                                      │ historical (30m)
                                                      ▼
                                             forecast.py  ◄── future covariates from HA:
                                             (train/predict)    Solcast PV, BOM weather,
                                                      │         Amber price forecast
                                                      │
                             ┌────────────────────────┤
                             │                        │
                             ▼                        ▼
                     predictions.json        HA sensor entities
                     *_forecast_log.csv      sensor.ai_price_forecast
                                             sensor.ai_load_forecast
                                                      │
                                                      ▼
                                               EMHASS optimiser
                                             (MPC day-ahead plan)
                                                      │
                                                      ▼
                                           HA automation reads mpc_*
                                           entities, calls EMS script
                                                      │
                                                      ▼
                                           Sigenergy inverter / battery
                                           (charge, discharge, standby, etc.)
```

---

## Scheduled Jobs (systemd)

Unit files are tracked in [`systemd/`](systemd/) in this repo. To install or update them:

```bash
sudo cp systemd/ai-energy-*.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
```

Three pairs of `.service` + `.timer` units drive the pipeline:

| Timer | Schedule | What it runs |
|---|---|---|
| `ai-energy-predict.timer` | Every 30 min (`:01` and `:31`) | `forecast.py predict-all --dynamic-handoff --publish-hass --publish-covariates` |
| `ai-energy-train.timer` | Monday 12:00 | `forecast.py train-load && forecast.py train-price` |
| `ai-energy-update-tariffs.timer` | Daily 00:00 | `forecast.py update-tariffs && smooth_tariffs.py && forecast.py backfill-actuals && forecast.py update-adjusters` |

All services run as user `saltspork`, `WorkingDirectory=/home/saltspork/src/ai-energy-forecast-slop`, activate `.venv` before running. Training is `Nice=19` (lowest CPU priority).

---

## Components

### `forecast.py` — Monolithic Orchestrator (~2,300 lines)

The core script. All behaviour is driven by subcommands:

| Subcommand | When | What it does |
|---|---|---|
| `train-price` | Weekly | Trains p30/p50/p70 LightGBM price models on 2 years of 30-min InfluxDB data |
| `train-load` | Weekly | Trains p50/p65/p75 LightGBM load models |
| `predict-all` | Every 30 min | Fetches covariates, runs all models, applies tariffs/GST, saves JSON, publishes to HA |
| `predict-price` | (manual) | Price only |
| `predict-load` | (manual) | Load only |
| `update-tariffs` | Daily midnight | Fetches 24h+ Amber tariff data, builds smoothed 48-slot profile |
| `backfill-actuals` | Daily midnight | Fills in actual measured values in the forecast log CSVs |
| `update-adjusters` | Daily midnight | Computes rolling 30-day bias corrections for weather covariates |

Key flags: `--dynamic-handoff`, `--publish-hass`, `--publish-covariates`, `--config`

#### Model Details

- **Framework:** Darts (time series library) + LightGBM quantile regression
- **Horizon:** 144 steps = 72 hours at 30-minute resolution
- **Price quantiles:** p30, p50 (median), p70
- **Load quantiles:** p50, p65, p75
- **Target lags:** 1–6, 12, 24, 48–49, 96–97, 336–337 (1h to 14 days)
- **Future covariate window:** ±4 lags around each step
- **Recency weighting:** exponential decay (price: 180-day half-life; load: 90-day half-life)
- **Log transform:** applied to price targets to handle negative/volatile prices
- **Anti-crossing:** quantile outputs are sorted to prevent p30 > p50

#### Dynamic Handoff (price only)

1. Amber Electric provides a 36-hour advanced forecast (mixed 5-min + 30-min intervals)
2. Script averages 5-min intervals to 30-min, uses this as pseudo-historical seed
3. LightGBM model extends the forecast beyond Amber's horizon

#### Covariate Adjustments

Weather forecasts (BOM) and PV forecasts (Solcast) exhibit systematic biases. Daily at midnight, `update-adjusters` computes time-of-day additive bias corrections from the last 30 days of the forecast log. These are stored in `adjuster_temperature.json`, `adjuster_humidity.json`, `adjuster_wind_speed.json` and applied at prediction time before model input.

#### Tariff Pipeline

`update-tariffs` → builds `tariff_profile_raw.json` (48 slots of real observed Amber data)
→ `smooth_tariffs.py` buckets by Peak/Solar Sponge/Off-Peak and replaces each with the bucket median → `tariff_profile.json` (the smoothed version actually used)

At prediction time, `apply_tariffs_to_forecast()` adds network loss factor and conditionally applies GST (10%) to produce final consumer prices.

---

### `ingest/` — Data Ingestion Scripts

These are run manually or ad-hoc (no systemd timer). They populate InfluxDB from historical sources.

| Script | Source | InfluxDB destination |
|---|---|---|
| `ingest-ha-data.py` | HA SQLite `statistics` table (metadata ID 55: consumed power) | `rp_30m.power_load_30m`, `rp_5m.power_load_5m` |
| `ingest-ha-pv-data.py` | HA SQLite (metadata IDs 56, 269: PV generation) | `rp_30m.power_pv_30m`, `rp_5m.power_pv_5m` |
| `ingest-ha-weather-data.py` | HA SQLite (temp 281, humidity 278, wind 277) | `rp_30m.temperature_adelaide`, etc. |
| `ingest-nem-data.py` | AEMO via `nemosis` library | `rp_30m/5m.aemo_dispatch_{sa1,vic1,nsw1}_{30m,5m}` |
| `ingest-nem-csv.py` | Local AEMO CSV files | Same measurements, backfill path |
| `backfill_amber_csv.py` | Amber Electric CSV export | (backfill path) |
| `backfill_pv_solcast.py` | Solcast historical | `power_pv_30m` |
| `patch_pv_gaps.py` | Interpolation | Fills gaps in `power_pv_30m` |
| `update_solcast.py` | HA history of EMHASS curtailment events | Updates `solcast-generation.json` export-limiting flags |

**Note:** The ingest scripts contain hardcoded credentials and are not in active automated use — they are one-time or occasional tools.

---

### InfluxDB Schema

Database: `hass`, InfluxDB v1.x

**Retention policies:**
- `rp_5m` — 5-minute resolution
- `rp_30m` — 30-minute resolution (primary for ML training)
- `rp_raw` — raw/immediate data (fed by HA's InfluxDB integration)
- `autogen` — legacy

**Key measurements (30-minute):**

| Measurement | Fields |
|---|---|
| `power_load_30m` | `mean_value`, `min_value`, `max_value` |
| `power_pv_30m` | `mean_value`, `min_value`, `max_value` |
| `temperature_adelaide` | `mean_value` |
| `humidity_adelaide` | `mean_value` |
| `wind_speed_adelaide` | `mean_value` |
| `aemo_dispatch_sa1_30m` | `price`, `total_demand`, `net_interchange` |
| `aemo_dispatch_vic1_30m` | same |
| `aemo_dispatch_nsw1_30m` | same |

Continuous queries in InfluxDB downsample raw → 5m → 30m automatically for ongoing data. See `README.md` for the CQ definitions.

---

### `smooth_tariffs.py`

Standalone utility that smooths the tariff profile to remove day-to-day volatility:

1. Loads `tariff_profile.json`
2. Classifies each 30-min slot: Peak (17:00–20:59), Solar Sponge (10:00–15:59), Off-Peak (all others)
3. Replaces all slots in each bucket with the bucket median
4. Saves smoothed to `tariff_profile.json`, original backed up to `tariff_profile_raw.json`

---

### `hass/` — Home Assistant Integration (backup copies)

These files live in HA but are backed up here. They are **not loaded directly from this directory** — they must be manually imported/updated in HA. URLs are redacted before committing.

| File | Purpose |
|---|---|
| `package-emhass.yaml` | HA package: template sensors for price/feed-in blending, `rest_command` to trigger EMHASS day-ahead optimisation with Jinja-built JSON payload |
| `automation-sigenergy-emhass.yaml` | HA automation: reads EMHASS `mpc_*` output entities every 5 min, evaluates battery control scenarios (grid charge, PV curtail, discharge, standby), calls EMS script |
| `script-sigenergy-ems.yaml` | HA script: sets SiG Energy EMS parameters (control mode, charge/discharge limits, SoC cutoffs, export limits) by writing to Sigenergy number/select entities |

#### `package-emhass.yaml` in detail

This is the most complex HA file. It does:

1. **`sensor.amber_effective_general_price`** — blends current Amber spot price with a risk-weighted advanced forecast (controlled by `input_number.emhass_weight_buy_forecast`, range −1 to +1). Adds DNSP free-tier adjustment (+1c during 10:00–16:00 if allowance > 0).

2. **`sensor.amber_effective_feed_in_price`** — same logic for feed-in (export) price.

3. **`rest_command.emhass_dayahead_optim`** — builds a JSON payload via Jinja2 and POSTs to EMHASS MPC endpoint. The payload includes:
   - PV forecast: Solcast p10/p50/p90 blended by `input_number.emhass_weight_pv_forecast`, with 65W fixed loss applied
   - Load forecast: from `sensor.ai_load_forecast_high` (p65 model)
   - Price forecast: from `sensor.ai_price_forecast` (p50), blended with p30/p70 by `input_number.emhass_weight_buy_forecast`

---

### Output Files

| File | Contents |
|---|---|
| `predictions.json` | Latest full forecast (price + load, all quantiles, 144 steps each, ~64KB) |
| `price_forecast_log.csv` | Historical predictions + actuals for price (~330MB, growing) |
| `load_forecast_log.csv` | Historical predictions + actuals for load (~340MB, growing) |
| `tariff_profile.json` | Smoothed 48-slot daily tariff profile |
| `tariff_profile_raw.json` | Unsmoothed tariff data (pre-smoothing backup) |
| `adjuster_*.json` | Weather covariate bias corrections by time-of-day |
| `price_model.pkl` / `price_p30_model.pkl` / `price_p70_model.pkl` | Trained price models (~128MB each) |
| `load_model.pkl` / `load_p65_model.pkl` / `load_p75_model.pkl` | Trained load models (~82MB each) |
| `*_importance.json` | Feature importances from last training run |

Note: several older model files exist (`price_p10`, `price_p20`, `price_p50`, `price_p80`, `price_p90`, `load_p60`) from previous experiments — only p30/p50/p70 (price) and p50/p65/p75 (load) are currently active.

---

## Configuration

`config.json` (from `config.example.json`) holds all credentials and settings:
- InfluxDB host/port/db/credentials
- Home Assistant URL + long-lived token
- Entity IDs for all HA sensors
- Model paths
- Per-model hyperparameters (n\_estimators, lags, horizon, quantiles, recency weighting)
- Adjuster settings

**Credential management:** `config.json` is git-ignored. The ingest scripts have credentials hardcoded (historical — they predate `config.json`). The HA YAML files need URL redaction before committing.

---

## Known Pain Points

1. **`forecast.py` is a monolith.** At ~2,300 lines it handles training, prediction, tariff management, logging, bias correction, and HA publishing. Hard to navigate and test.

2. **`hass/package-emhass.yaml` Jinja complexity.** The EMHASS REST command payload is built entirely in Jinja2 template syntax inside a YAML string. It's ~350 lines of logic that is hard to debug, diff, and maintain.

3. **Ingest scripts are disconnected.** They have hardcoded credentials, no systemd timers, and are run manually or ad-hoc. There is no clear trigger or documented procedure for keeping InfluxDB current beyond the HA CQs.

4. **HA backups require manual redaction.** Every time `hass/` files are committed, URLs must be manually redacted. This creates friction and risk.

5. **Forecast log CSVs are very large.** `price_forecast_log.csv` and `load_forecast_log.csv` are each ~330–340MB and growing. They are not git-ignored (or if they are, they live outside the repo). This could become a problem.

6. **Multiple stale model files.** Experimental quantile variants (p10, p20, p50, p80, p90 for price; p60 for load) remain on disk, consuming ~1.3GB that isn't actively used.

7. **No error alerting.** If a systemd timer fails silently, the forecasts go stale and EMHASS gets no updated inputs. There is no notification mechanism.

8. **`analyse.ipynb`** is a 35MB notebook committed to the repo — likely containing output data.

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML framework | [Darts](https://unit8co.github.io/darts/) + LightGBM |
| Time series DB | InfluxDB v1.x |
| Home automation | Home Assistant |
| Energy optimiser | EMHASS (MPC mode) |
| Battery hardware | SiG Energy (Sigenergy) inverter + ESS |
| Energy retailer | Amber Electric (SA1 region) |
| Market data | AEMO NEM via `nemosis` + NEMWeb CSV |
| Solar forecasting | Solcast |
| Weather | Bureau of Meteorology (BOM) via HA integration |
| Job scheduling | systemd timers |
| Runtime | Python 3.13, `.venv` |
