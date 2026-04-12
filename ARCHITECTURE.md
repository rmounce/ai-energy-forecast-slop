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

Six pairs of `.service` + `.timer` units drive the pipeline:

**Forecast pipeline:**

| Timer | Schedule | What it runs |
|---|---|---|
| `ai-energy-predict.timer` | Every 30 min (`:01` and `:31`) | `forecast.py predict-all --dynamic-handoff --publish-hass --publish-covariates` |
| `ai-energy-train.timer` | Monday 12:00 | `forecast.py train-load && forecast.py train-price` |
| `ai-energy-update-tariffs.timer` | Daily 00:00 | `forecast.py update-tariffs && smooth_tariffs.py && forecast.py backfill-actuals && forecast.py update-adjusters` |

**AEMO data collection (added 2026-04-10):**

| Timer | Schedule | What it runs |
|---|---|---|
| `ai-energy-pd7day.timer` | 3×/day (07:20, 12:55, 18:05 AEST) | `ingest/ingest-pd7day.py --fetch` |
| `ai-energy-predispatch.timer` | Every 30 min (`:12` and `:42`) | `ingest/ingest-predispatch.py --fetch` |
| `ai-energy-sevendayoutlook.timer` | Every 30 min (`:15` and `:45`) | `ingest/ingest-sevendayoutlook.py --fetch` |
| `ai-energy-p5min.timer` | Every 5 min (`:02/:07/:12/…/:57`) | `ingest/ingest-p5min.py --fetch` — SA1/VIC1/NSW1 5-min predispatch → `rp_5m.aemo_p5min_forecast` |

All services run as user `saltspork`, `WorkingDirectory=/home/saltspork/src/ai-energy-forecast-slop`, activate `.venv` before running. Training is `Nice=19` (lowest CPU priority).

---

## Components

### `forecast.py` — Monolithic Orchestrator (~2,300 lines)

The core script. All behaviour is driven by subcommands:

| Subcommand | When | What it does |
|---|---|---|
| `train-price` | Weekly | Trains price quantile models on 2 years of 30-min InfluxDB data |
| `train-load` | Weekly | Trains load quantile models |
| `predict-all` | Every 30 min | Fetches covariates, runs all models, applies tariffs/GST, saves JSON, publishes to HA |
| `predict-price` | (manual) | Price only |
| `predict-load` | (manual) | Load only |
| `update-tariffs` | Daily midnight | Fetches 24h+ Amber tariff data, builds smoothed 48-slot profile |
| `backfill-actuals` | Daily midnight | Fills in actual measured values in the forecast log CSVs |
| `update-adjusters` | Daily midnight | Computes rolling 30-day bias corrections for weather covariates |

Key flags: `--dynamic-handoff`, `--publish-hass`, `--publish-covariates`, `--config`

#### Internal Module Boundaries

The file contains ~28 functions that fall naturally into these logical groups:

| Group | Functions | Responsibility |
|---|---|---|
| **Config / utilities** | `load_config`, `add_time_features`, `add_gst`, `remove_gst` | Shared utilities |
| **InfluxDB** | `get_historical_data` | Historical data for training and backfill |
| **HA API** | `call_ha_api`, `get_entity_state` | Generic HA HTTP wrappers |
| **Data fetching** | `get_amber_spot_price_forecast`, `get_amber_advanced_forecast`, `get_solcast_forecast`, `get_weather_forecast`, `get_aemo_forecast`, `_get_aemo_short_term_forecast`, `_get_aemo_short_term_price_sa1`, `_get_aemo_7_day_outlook_forecast` | Future covariate data from external sources |
| **Training** | `train_single_model`, `train_models` | Model fitting and serialisation |
| **Prediction** | `_predict_simple`, `_predict_with_dynamic_handoff`, `_execute_quantile_prediction`, `_execute_single_prediction` | Inference |
| **Tariffs** | `get_amber_api_scaling_factor`, `get_network_loss_factor`, `_get_tariff_data`, `_create_complete_profile`, `_calculate_amber_api_scaling_factor`, `_calculate_forecasted_network_loss_factor`, `update_tariffs`, `apply_tariffs_to_forecast` | Tariff profile construction and application |
| **Adjusters** | `update_adjusters`, `apply_covariate_adjustments` | Weather covariate bias correction |
| **Logging** | `log_forecast_data`, `backfill_actuals`, `_backfill_single_log` | Forecast log CSVs |
| **Publishing** | `publish_forecast_to_hass`, `publish_adjusted_covariates_to_hass` | Push results to HA |
| **Orchestrators** | `run_predictions`, `main` | Top-level entry points |

One function — `_publish_covariates_helper` — is defined but never called (orphaned).

#### Model Details

- **Framework:** Darts (time series library) + LightGBM quantile regression
- **Horizon:** 144 steps = 72 hours at 30-minute resolution
- **Active price quantiles:** p30, p50 (median), p70 — configured in `config.json`
- **Active load quantiles:** p50, p65, p75 — configured in `config.json`
- **Target lags:** 1–6, 12, 24, 48–49, 96–97, 336–337 (1h to 14 days)
- **Future covariate window:** ±4 lags around each step
- **Recency weighting:** exponential decay (price: 180-day half-life; load: 90-day half-life)
- **Log transform:** applied to price targets to handle negative/volatile prices
- **Anti-crossing:** quantile outputs are sorted to prevent p30 > p50

Several older model files exist on disk (price: p10, p20, p50, p80, p90; load: p60) from past experiments — these are not referenced by the current `config.json` and can be deleted.

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

**Active automated scripts** (have systemd timers, use `config.json`):

| Script | Schedule | Source | InfluxDB destination |
|---|---|---|---|
| `ingest-pd7day.py --fetch` | 3×/day | AEMO NEMWeb `PD7DAY/PRICESOLUTION` | `rp_30m.aemo_pd7day_forecast` (tags: region, run_time; fields: rrp $/MWh) |
| `ingest-predispatch.py --fetch` | Every 30 min | AEMO NEMWeb `Predispatch_Reports` | `rp_30m.aemo_predispatch_forecast` (tags: region, run_time; fields: rrp, total_demand, net_interchange) |
| `ingest-sevendayoutlook.py --fetch` | Every 30 min | AEMO NEMWeb `SEVENDAYOUTLOOK_FULL` | `rp_30m.aemo_sevendayoutlook` (tags: region, run_time; fields: scheduled_demand, scheduled_capacity, net_interchange, scheduled_reserve) |
| `ingest-p5min.py --fetch` | Every 5 min | AEMO NEMWeb `P5_Reports` | `rp_5m.aemo_p5min_forecast` (tags: region, run_time; fields: rrp, total_demand, net_interchange) — SA1/VIC1/NSW1 |

Each script also has a `--backfill-archive` mode that imports historical weekly ZIPs from NEMWeb. Backfills were completed 2026-04-10 covering March 2025–April 2026 for PREDISPATCH and SEVENDAYOUTLOOK, and February–April 2026 for PD7Day (no older archive exists).

**NEMSEER/NEMWeb historical backfill** (writes directly to Parquet, not InfluxDB):

| Script | Coverage | Notes |
|---|---|---|
| `ingest/backfill_predispatch_nemseer.py` | April 2024 – February 2025 | Uses NEMSEER library for pre-Aug 2024 (MMSDM archive); direct HTTP for Aug 2024+ (AEMO restructured the archive format). Output merged into `data/parquet/aemo_predispatch_sa1.parquet`. Cache in `data/nemseer_cache/` (~0.5GB). |

**Manual/ad-hoc scripts** (hardcoded credentials, run once or occasionally):

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
| `power_dump_load_30m` | `mean_value`, `min_value`, `max_value` — estimated dump load (2×2000W heaters on smart switches, no power monitoring); subtracted from `power_load_30m` before model training/prediction |
| `power_dump_load_5m` | `mean_value`, `min_value`, `max_value` — intermediate 5m aggregation fed by CQ |
| `power_pv_30m` | `mean_value`, `min_value`, `max_value` |
| `temperature_adelaide` | `mean_value` |
| `humidity_adelaide` | `mean_value` |
| `wind_speed_adelaide` | `mean_value` |
| `aemo_dispatch_sa1_30m` | `price` ($/MWh), `total_demand`, `net_interchange` |
| `aemo_dispatch_vic1_30m` | same |
| `aemo_dispatch_nsw1_30m` | same |
| `aemo_pd7day_forecast` | `rrp` ($/MWh) — tags: `region`, `run_time` |
| `aemo_predispatch_forecast` | `rrp` ($/MWh), `total_demand`, `net_interchange` — tags: `region`, `run_time` |
| `aemo_sevendayoutlook` | `scheduled_demand`, `scheduled_capacity`, `net_interchange`, `scheduled_reserve` — tags: `region`, `run_time` |

Continuous queries in InfluxDB downsample raw → 5m → 30m automatically for ongoing data. See `README.md` for the CQ definitions.

---

### `smooth_tariffs.py`

Standalone utility called by the nightly `ai-energy-update-tariffs` service. Smooths the tariff profile to remove day-to-day volatility:

1. Loads `tariff_profile.json`
2. Classifies each 30-min slot: Peak (17:00–20:59), Solar Sponge (10:00–15:59), Off-Peak (all others)
3. Replaces all slots in each bucket with the bucket median
4. Saves smoothed to `tariff_profile.json`, original backed up to `tariff_profile_raw.json`

---

### `data/` and `train/` — TFT Price Model (in development)

A new price forecasting model is being developed to replace the LightGBM+Amber APF approach. Full design rationale, options considered, literature references, and next steps are documented in **[docs/tft_price_forecast.md](docs/tft_price_forecast.md)**. Longer-term speculative ideas (spike-aware dispatch, direct value optimisation, ensemble methods) are captured in **[docs/ideas.md](docs/ideas.md)**.

**Summary:**
- Encoder: 96 steps (2 days) × 20 features — historical price/demand/load/PV/weather (8) + 5-min volatility aggregates (4: `rrp_5m_max`, `rrp_5m_std`, `rrp_persistence`, `rrp_volatility_30m`) + `rrp_log_momentum` (slope of last 4 log-steps) + time encodings (6) + `rrp_5m_missing` flag (1)
- Decoder: 144 steps (72h) × 13 features — SA1 PREDISPATCH/PD7Day (4) + VIC1/NSW1 PREDISPATCH prices (2) + `pd_demand`, `pd_net_interchange` + time encodings (6) + `covar_missing` flag (1)
- Covariate construction: Option B (run-aligned) — each training sample uses the PREDISPATCH run issued at the encoder/decoder boundary, exactly matching inference
- Masked loss: each decoder step independently masked; handles variable PREDISPATCH horizon and growing PD7Day history
- Stratified eval benchmark: 900 fixed samples (spike + low/negative + seasonal normal) for durable cross-run comparison

**Data pipeline:**
1. `ingest/ingest-predispatch.py`, `ingest/ingest-pd7day.py`, `ingest/ingest-p5min.py` → InfluxDB (ongoing, systemd timers)
2. `ingest/backfill_predispatch_nemseer.py` → extends PREDISPATCH parquet back to 2024-04 (run once; ~2 min from cache)
3. `data/export_parquet.py` → SA1/VIC1/NSW1 PREDISPATCH + actuals + 5m volatility agg (use `--actuals-5m` to refresh just actuals without touching NEMSEER-backfilled PREDISPATCH)
4. `data/build_stratified_eval.py` → fixed 900-sample benchmark index (run once; `--force` to regenerate)
5. `data/build_training_dataset.py` → numpy arrays for training (18 enc / 13 dec features)
6. `train/train_tft_price.py` → model checkpoint at `models/tft_price/`
7. `train/evaluate_tft.py --eval-set stratified` → nMAPE (all/base/spike) + quantile calibration vs LightGBM

**Status (2026-04-12):** Run 010 in progress (**Log-Scaling** + **q30/50/70** + **2022 Backfill**). Stratified eval benchmark used to close the spike nMAPE gap vs LightGBM.

---

### `model/` — Offline Training Scripts (exploratory, not part of automated pipeline)

These scripts were used to develop and validate the ML approach before it was integrated into `forecast.py`. They are **not called by any systemd service** and are not kept in sync with the main script.

| Script | Purpose |
|---|---|
| `01-load-historical.py` | Load and explore historical data from InfluxDB |
| `02-load-forecasts.py` | Fetch and combine future covariates (has its own implementations, different from `forecast.py`) |
| `03-train-model.py` | Train a single model with hardcoded parameters |
| `04-predict.py` | Run predictions and visualise with matplotlib |

These scripts duplicate logic from `forecast.py` (especially the prediction and training code) but are not config-driven and would need manual synchronisation if the pipeline changes. Consider them as reference material rather than active code.

---

### Jupyter Notebooks (exploratory, not part of automated pipeline)

| Notebook | Purpose |
|---|---|
| `analyse.ipynb` (36MB) | Post-hoc analysis of forecast accuracy; loads `load_forecast_log.csv` and re-implements covariate adjustment logic inline. Large due to embedded output data. |
| `price-history.ipynb` (184KB) | Historical AEMO price distribution analysis via `nemosis`. Fully independent of `forecast.py`. |

Neither notebook imports from `forecast.py`. They are standalone exploratory tools.

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

The systemd services load secrets from `.env` in the repo root (git-ignored). This file must be created manually:

```ini
# .env
HC_PREDICT_URL=https://hc-ping.com/<your-uuid>
```

---

## Known Pain Points

1. **`forecast.py` is a monolith.** At ~2,300 lines it handles training, prediction, tariff management, logging, bias correction, and HA publishing. The natural module boundaries are clear (see table above) but the code is not yet split. Hard to navigate and test.

2. **`hass/package-emhass.yaml` Jinja complexity.** The EMHASS REST command payload is built entirely in Jinja2 template syntax inside a YAML string. It's ~350 lines of logic that is hard to debug, diff, and maintain.

3. **Ingest scripts are disconnected.** They have hardcoded credentials, no systemd timers, and are run manually or ad-hoc. There is no clear trigger or documented procedure for keeping InfluxDB current beyond the HA CQs.

4. **HA backups require manual redaction.** Every time `hass/` files are committed, URLs must be manually redacted. This creates friction and risk.

5. **Forecast log CSVs are very large.** `price_forecast_log.csv` and `load_forecast_log.csv` are each ~330–340MB and growing. They live outside the repo (git-ignored) but are depended on by `backfill-actuals` and `update-adjusters`.

6. **Stale model files on disk.** Experimental quantile variants (price: p10, p20, p50, p80, p90; load: p60) are no longer referenced by `config.json` and consume ~1.3GB.

7. **`analyse.ipynb`** is a 36MB notebook with embedded output data committed to the repo. Should either have outputs stripped or be moved outside the repo.

8. **`_publish_covariates_helper()`** is defined in `forecast.py` but never called — orphaned dead code.

9. **`model/` scripts are out of sync.** The four scripts in `model/` duplicate logic from `forecast.py` with hardcoded parameters. They are useful as reference but misleading as "current" code.

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML framework | [Darts](https://unit8co.github.io/darts/) + LightGBM (existing); PyTorch LSTM encoder-decoder (TFT, in development) |
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
