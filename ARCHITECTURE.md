# System Architecture

This document describes the end-to-end architecture of the AI energy forecasting pipeline. It is intended as a living reference for understanding how all the pieces fit together.

---

## Overview

The pipeline provides 72-hour forecasts of electricity prices and household power consumption for input into [EMHASS](https://emhass.readthedocs.io/), an energy management optimiser. EMHASS produces a battery charge/discharge schedule which Home Assistant automation then executes on a SiG Energy (Sigenergy) inverter/battery system.

The system is located in Adelaide, South Australia (AEMO region SA1), and uses Amber Electric as the retail electricity provider.

---

## High-Level Data Flow

```
External Sources                             Internal Data Store
────────────────                             ───────────────────
AEMO NEMweb (direct scraping):
  P5MIN        every 5 min   ─────────────► ingest/*.py ──► InfluxDB (hass db)
  PREDISPATCH  every 30 min  ──────────────────────────────────────┤
  PD7Day       3×/day        ──────────────────────────────────────┤
  SevenDayOutlook every 30m  ──────────────────────────────────────┤
HA → InfluxDB CQs (load, PV, weather):  ─────────────────────────┘
                                                        │
                         ┌──────────────────────────────┤
                         │                              │
                  every 5 min (Tier 1)          every 30 min (Tier 2)
                         │                              │
                         ▼                              ▼
                  Tactical LightGBM           APF/LightGBM price (incumbent)
                  (0–60 min q05/50/95)        PD-direct price (canonical Tier 2)
                         │                    TFT load / LightGBM load
                         └──────────┬─────────────────┘
                                    │ ◄── HA future covariates (Solcast,
                                    │     BOM weather, Amber dynamic handoff)
                                    │
             ┌──────────────────────┤
             │                      │
             ▼                      ▼
     predictions.json       HA sensor entities:
     *_forecast_log.csv     sensor.ai_p5min_price_forecast  (tactical, 5-min)
                            sensor.ai_price_forecast        (APF/LightGBM p50, incumbent)
                            sensor.ai_price_forecast_low/high
                            sensor.ai_pd_direct_price_forecast (canonical Tier 2)
                            sensor.ai_load_forecast
                            sensor.ai_combined_*_price_forecast  (TFT-based, shadow)
                                    │
                                    ▼
                             EMHASS optimiser
                       day-ahead (72h × 30-min)
                       MPC (14h × 5-min, re-runs every 5 min)
                                    │
                                    ▼
                        HA automation → EMS script
                                    │
                                    ▼
                        Sigenergy inverter / battery
                        (charge, discharge, standby)
```

---

## Scheduled Jobs (systemd)

Unit files are tracked in [`systemd/`](systemd/) in this repo and symlinked into `~/.config/systemd/user/` — edits to the repo files take effect after `systemctl --user daemon-reload`. To set up on a new machine:

```bash
mkdir -p ~/.config/systemd/user
for f in systemd/ai-energy-*.{service,timer}; do
  ln -sf "$(pwd)/$f" ~/.config/systemd/user/
done
systemctl --user daemon-reload
systemctl --user enable --now ai-energy-*.timer
systemctl --user enable --now ai-energy-listener.service   # event-driven price refresh
sudo loginctl enable-linger "$USER"   # keep units running after logout
```

Seven pairs of `.service` + `.timer` units plus one event-driven daemon drive the pipeline:

**Forecast pipeline:**

| Unit | Schedule | What it runs |
|---|---|---|
| `ai-energy-listener.service` | Event-driven (Amber APF state change in HA; 30-min idle heartbeat) | `forecast.py predict-price --dynamic-handoff --publish-hass` — added 2026-05-27, see [docs/event_driven_predict_price_plan.md](docs/event_driven_predict_price_plan.md) |
| `ai-energy-predict.timer` | Every 30 min (`:01` and `:31`) | `forecast.py predict-load --publish-hass --publish-covariates` — price path moved to the listener 2026-05-27; cadence aligned with the 30-min InfluxDB CQ granularity |
| `ai-energy-train.timer` | Monday 12:00 | `forecast.py train-load && forecast.py train-price` |
| `ai-energy-update-tariffs.timer` | Daily 00:00 | `forecast.py update-tariffs && smooth_tariffs.py && forecast.py backfill-actuals && forecast.py update-adjusters` |

**AEMO data collection (added 2026-04-10):**

| Timer | Schedule | What it runs |
|---|---|---|
| `ai-energy-pd7day.timer` | 3×/day (07:20, 12:55, 18:05 AEST) | `ingest/ingest-pd7day.py --fetch` |
| `ai-energy-predispatch.timer` | Every 30 min (`:12` and `:42`) | `ingest/ingest-predispatch.py --fetch && forecast.py publish-pd-direct --publish-hass` — chained so Tier 2 PD-direct refreshes within ~1 min of each AEMO PREDISPATCH publish (added 2026-05-13) |
| `ai-energy-sevendayoutlook.timer` | Every 30 min (`:15` and `:45`) | `ingest/ingest-sevendayoutlook.py --fetch` |
| `ai-energy-p5min.timer` | Every 5 min (`:02/:07/:12/…/:57`) | `ingest/ingest-p5min.py --fetch && forecast.py publish-tactical --publish-hass` — Tier 1 refresh + Tier 2 cache republish |

All units run as systemd user units (`systemctl --user`), `WorkingDirectory=/home/saltspork/src/ai-energy-forecast-slop`, activate `.venv` before running. Training is `Nice=19` (lowest CPU priority). Linger is enabled so units run without an active login session.

---

## Components

### `forecast.py` — Monolithic Orchestrator (~3,500 lines)

The core script. All behaviour is driven by subcommands:

| Subcommand | When | What it does |
|---|---|---|
| `train-price` | Weekly | Trains price quantile models on 2 years of 30-min InfluxDB data |
| `train-load` | Weekly | Trains load quantile models |
| `predict-all` | (manual) | Fetches covariates, runs both price+load models, applies tariffs/GST, saves JSON, publishes to HA. Production splits this into event-driven `predict-price` (via `ai-energy-listener.service`) and timer-driven `predict-load`. |
| `publish-tactical` | Every 5 min | Cheap Tier 1 LGBM refresh; reuses cached Tier 2 PD-direct; updates stitched/canonical AI sensors |
| `publish-pd-direct` | Every 30 min (after PREDISPATCH ingest) | Refreshes Tier 2 PD-direct + canonical AI MPC/DH bundle + raw AEMO stitched. Cheap: skips TFT/load/Solcast/weather. Wall-clock ≈ 30s. |
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
| **Config / utilities** | `add_time_features`, `add_gst`, `remove_gst` | Shared utilities (`load_config` delegated to `config_utils.py`) |
| **HA API** | `call_ha_api`, `get_entity_state` | Generic HA HTTP wrappers |
| **Data fetching** | `get_amber_spot_price_forecast`, `get_amber_advanced_forecast`, `get_solcast_forecast`, `get_weather_forecast`, `get_aemo_forecast`, `_get_aemo_short_term_forecast`, `_get_aemo_short_term_price_sa1`, `_get_aemo_7_day_outlook_forecast` | Future covariate data from external sources |
| **Training** | `train_single_model`, `train_models` | Model fitting and serialisation |
| **Prediction** | `_predict_simple`, `_predict_with_dynamic_handoff`, `_execute_quantile_prediction`, `_execute_single_prediction`, `_execute_tactical_prediction` (Tier 1 LightGBM, 0–60 min), `_execute_pd_direct_prediction` (canonical Tier 2 price), `_execute_tft_load_prediction` (TFT load). `_execute_tft_prediction` still importable but no longer invoked by `predict-all` (disabled 2026-05-13). | Inference |
| **Tariffs** | `get_amber_api_scaling_factor`, `get_network_loss_factor`, `_get_tariff_data`, `_create_complete_profile`, `_calculate_amber_api_scaling_factor`, `_calculate_forecasted_network_loss_factor`, `update_tariffs`, `apply_tariffs_to_forecast` | Tariff profile construction and application |
| **Adjusters** | `update_adjusters`, `apply_covariate_adjustments` | Weather covariate bias correction |
| **Logging** | `log_forecast_data`, `backfill_actuals`, `_backfill_single_log` | Forecast log CSVs |
| **Publishing** | `publish_forecast_to_hass`, `publish_adjusted_covariates_to_hass`, `_build_combined_forecast_items`, `_publish_combined_price_forecasts` | Push results to HA |
| **InfluxDB helpers** | `get_historical_data`, `_get_influx_sdo_demand` | InfluxDB queries |
| **Orchestrators** | `run_predictions`, `main` | Top-level entry points |

Dead code removed: `_publish_covariates_helper` (never called) and `model/` (01–04-*.py exploratory scripts, superseded by `forecast.py`).

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

### `eval/` — Evaluation and Backtesting

See **[eval/README.md](eval/README.md)** for full documentation including the Phase 6
holistic dispatch simulation design.

| Script | Purpose |
|--------|---------|
| `dispatch_simulator.py` | Rolling MPC LP backtester — price-only today, net_load extension in Phase 6 |
| `compare_tft_dispatch.py` | TFT vs LightGBM dispatch comparison (Phase 3) |
| `compare_load_forecast.py` | Load forecast comparison |
| `eval_load_overnight.py` | Load TFT overnight ramp diagnostics |

### `tests/` — Test Framework

See **[tests/README.md](tests/README.md)** for full documentation including Phase 8 design
and fixture capture instructions.

Two layers: fast unit tests (no external deps, <60s) + financial eval gate (requires
InfluxDB, thresholds set by Phase 6). **Both must pass before Phase 5 sub-tasks 4–8 resume.**

---

### `data/` and `train/` — TFT Price Model (V4, shadow mode)

> **2026-05-05 status.** TFT iteration is paused. A live `--debug-tft` run on 2026-05-05
> showed Run 011b outputting 30–50% below its own debiased PREDISPATCH input even with the
> debiaser passive — i.e. the TFT compresses on its own, not just because of the debiaser.
> The 2026-05-05 `run011b_active_15` retrain was rejected on Window A/B `netload_tariffed`
> gates. No further TFT training is to be launched until a no-ML "PD-direct" baseline has
> been measured through the same gates. Full plan in `docs/roadmap.md` (top section,
> 2026-05-05 Strategic Pivot); structural critique in `docs/tft_price_forecast.md`.

A TFT price model has been trained and published to HA in shadow mode alongside the APF/LightGBM incumbent. The active production checkpoint is Run 011b (+9.7% vs amber_apf_lgbm baseline); Phase 7 decoder expansion attempts (Run 014, Run 015) failed the holistic eval gate and were not promoted; the 2026-05-05 active15 retrain was also rejected. **TFT price inference was disabled in `predict-all` on 2026-05-13** (strategic pivot — PD-direct is canonical Tier 2, TFT price shadow was redundant and burning CPU). The model checkpoint, log, and `_execute_tft_prediction` function are retained for the eval/MPC harness (`model_a_hybrid` source) and to keep the door open for a future revival. The HA shadow entities `sensor.ai_tft_price_forecast(_low/_high)` were removed in the same change. Full design rationale, options considered, literature references, and next steps are documented in **[docs/tft_price_forecast.md](docs/tft_price_forecast.md)**. Longer-term speculative ideas (spike-aware dispatch, direct value optimisation, ensemble methods) are captured in **[docs/ideas.md](docs/ideas.md)**.

**Summary:**
- Encoder: 96 steps (2 days) × 20 features — historical price/demand/load/PV/weather (8) + 5-min volatility aggregates (4: `rrp_5m_max`, `rrp_5m_std`, `rrp_persistence`, `rrp_volatility_30m`) + `rrp_log_momentum` + time encodings (6) + `rrp_5m_missing` flag (1)
- Decoder: 144 steps (72h) × 18 features — PREDISPATCH-only `pd_rrp`/demand/interchange for steps 0–55, parallel `pd7_rrp` across all 144 steps, VIC1/NSW1 PREDISPATCH prices, SevenDayOutlook demand/interchange, time encodings (6), `horizon_norm`, `predispatch_active`, `pd7_generation_hour`, `pd7_available` *(Phase 7 dataset + Run 014 checkpoint; active production model remains Run 011b until eval gate passes)*
- Covariate construction: Option B (run-aligned) — each training sample uses the PREDISPATCH run issued at the encoder/decoder boundary, exactly matching inference
- Masked loss: each decoder step independently masked; handles variable PREDISPATCH horizon and growing PD7Day history
- Stratified eval benchmark: 900 fixed samples (spike + low/negative + seasonal normal) for durable cross-run comparison
- Quantiles: q5/q10/q50/q90/q95/q99

**Data pipeline (Tier 2 TFT):**
1. `ingest/ingest-predispatch.py`, `ingest/ingest-pd7day.py`, `ingest/ingest-sevendayoutlook.py`, `ingest/ingest-p5min.py` → InfluxDB (ongoing, systemd timers)
2. `ingest/backfill_predispatch_nemseer.py` → PREDISPATCH parquet back to 2022 (run once; ~2 min from cache)
3. `data/export_parquet.py` → SA1/VIC1/NSW1 PREDISPATCH + actuals + 5m volatility agg + SevenDayOutlook (use `--actuals-only` for routine refreshes to preserve NEMSEER backfill)
4. `train/train_pd_debiaser.py` → Phase 1a OOF debiaser; outputs `data/parquet/debiased_pd_rrp_oof.parquet`
5. `data/build_stratified_eval.py` → fixed 900-sample benchmark index (run once; `--force` to regenerate)
6. `data/build_training_dataset.py` → numpy arrays for training (20 enc / 15 dec features)
7. `train/train_tft_price.py` → model checkpoint at `models/tft_price/`
8. `train/evaluate_tft.py --eval-set stratified` → nMAPE (all/base/spike) + quantile calibration vs LightGBM

**Data pipeline (Tier 1 tactical — Phase 2):**
1. `ingest/backfill_p5min_nemseer.py` → P5MIN forecasts 2024-04 → 2026-03 (run once; NEMSEER + direct ARCHIVE)
2. `data/export_parquet.py --p5min` → `actuals_sa1_5m.parquet` (raw 5-min dispatch prices)
3. `data/build_stratified_eval_tactical.py` → fixed 1,600-sample benchmark index (500 spike ≥$300, 300 low/negative, 800 seasonal normal; run once)
4. `data/build_tactical_dataset.py` → numpy arrays for Tier 1 LightGBM; X [210k, 24], y [210k, 12], long-format 2.2M rows after horizon expansion
5. `train/train_lgbm_tactical.py` → 3 LightGBM quantile models (q5/q50/q95), long-format with horizon as feature

**Phase 3 (dispatch simulator):**
6. `eval/dispatch_simulator.py` → rolling MPC LP backtester (scipy HiGHS, 40 kWh/10 kW); evaluates oracle/P5MIN/LightGBM q50 on stratified eval set
7. `eval/compare_tft_dispatch.py` → TFT vs LightGBM dispatch comparison on 130 overlapping 30-min boundary runs

**Phase 4 (conformal calibration):**
8. `train/calibrate_conformal.py` → conditional conformal δ corrections; `models/lgbm_tactical/conformal_deltas.json`

**Status (2026-04-20):** Phases 1–9 + Phase 6 + Phase 8 complete. All financial gates pass — `tier1_tier2_hybrid` (Run 011b + binary spike routing) overall +9.7% vs amber_apf_lgbm baseline ✅. Active production model: Run 011b checkpoint. Phase 7 decoder expansion has now been trained twice: Run 014 (18-feature checkpoint) failed the interim holistic eval (**−35.3% overall vs amber_apf_lgbm**), and the follow-up flat-wMAPE ablation Run 015 failed even harder (**−65.9% overall**). Run 011b therefore remains the incumbent, and flat horizon weighting is not promoted. See `docs/roadmap.md`, `docs/tft_price_forecast.md`, and `docs/training_runs.md`.

---

### `data/` and `train/` — TFT Load Model (in development)

A TFT model for household load prediction, intended to shadow and eventually replace the existing Darts/LightGBM load model. The existing model uses manual lag engineering (t-48, t-96, t-336 etc.) to capture daily/weekly seasonality; TFT replaces this with attention.

**Architecture:**
- Encoder: 96 steps (48h lookback) — `power_load`, `power_pv`, temperature/humidity/wind, time features
- Decoder: 144 steps (72h) — temperature/humidity/wind forecasts (BOM), Solcast PV forecast, time features, holidays
- Target: `power_load` at each future step
- Quantiles: q10/q50/q90 (3 quantiles; symmetric uncertainty)
- Loss: horizon-weighted quantile loss (exponential decay, half-life ~24 steps / 12h) — shorter-term accuracy prioritised for EMHASS

**Key difference from price TFT:** No PREDISPATCH equivalent. Decoder covariates are purely weather + time — cleaner architecture. Target is positive-and-bounded so no log transform needed.

**Data pipeline (Load TFT):**
1. `data/export_load_dataset.py` → pull `power_load_30m`, `power_pv_30m`, weather from InfluxDB → parquet
2. `data/build_load_dataset.py` → encoder/decoder numpy arrays, MinMax scalers, train/val split
3. `train/train_tft_load.py` → TFT checkpoint at `models/tft_load/`
4. Shadow implementation in `forecast.py` → `_execute_tft_load_prediction()`; publishes to `sensor.ai_tft_load_forecast`

**Current production checkpoint:** `models/tft_load/checkpoint_best.pt` (Run 005, epoch 32). Overall MAE 234W.

**Known issue — overnight 48h morning ramp inversion:** Step 72 (6:30am day+2) is predicted lower than step 60 (3:30am day+2), which is physically implausible. Root cause: `HorizonWeightedQuantileLoss` with tau=48 gives step 72 only 22% gradient weight — the model's time-of-day encoding at this horizon is too weak. Run 006 (planned) will add a gradient floor (`--horizon-floor 0.25`) so all steps beyond ~32h retain at least 25% weight. See `docs/tft_load_forecast.md` for full run history and promotion criteria.

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

> Production terminal-SoC policy (DH offset feedback, MPC inheritance, eval-vs-production
> correspondence): see [docs/production_soc_policy.md](docs/production_soc_policy.md).

| File | Purpose |
|---|---|
| `package-emhass.yaml` | HA package: template sensors for price/feed-in blending, the `script.emhass_dayahead_optim` / `script.emhass_mpc` wrappers that compute soc_init/soc_final and persist them to helper input_numbers, and the underlying `rest_command.emhass_dayahead_optim` / `rest_command.emhass_mpc` that POST a Jinja-built JSON payload to EMHASS |
| `automation-sigenergy-emhass.yaml` | HA automation: reads EMHASS `mpc_*` output entities every 5 min, evaluates battery control scenarios (grid charge, PV curtail, discharge, standby), calls EMS script |
| `script-sigenergy-ems.yaml` | HA script: sets SiG Energy EMS parameters (control mode, charge/discharge limits, SoC cutoffs, export limits) by writing to Sigenergy number/select entities. On a **mode change** it sequences writes to avoid power bursts: lowers every limit to `min(current, target)`, waits (≤3 s) for the async grid/PCS limits to land, flips the mode, waits (≤3 s) for the mode to read back, then raises limits to target. When the mode is unchanged it writes targets directly. |
| `automation-sigenergy-master-limit.yaml` | HA automation: the async writer that propagates `input_number.desired_*` helpers to the Sigenergy `*_limitation` entities, applying the Amber negative-feed-in (→0) override, capturing externally-imposed SAPN flexible export limits into `input_number.flexible_export_limit`, and running a 5-min failsafe. `script-sigenergy-ems.yaml` writes the `desired_*` helpers; this automation does the actual hardware write for the four grid/PCS limits. |
| `automation-sigenergy-export-ramp-hold.yaml` | HA automation: works around the Sigen grid-export-limit register's internal **~300 W/s ramp**. On a **decrease** of `number.sigen_plant_grid_export_limitation` (from EMHASS desired, Amber negative-price, or SAPN flexible limit — the master-limit controller folds all into this register), it holds actual grid export at the new target by closed-loop capping the *instant* PCS export limit (`PCS_cap = p_pcs + p_grid + E_target`, event-driven on grid-power updates) for the modelled ramp duration, then releases. This is the actual fix for the transient over-export/burst when reducing export; up-ramps are tolerated. |

#### `package-emhass.yaml` in detail

This is the most complex HA file. It does:

1. **`sensor.amber_effective_general_price`** — blends current Amber spot price with a risk-weighted advanced forecast (controlled by `input_number.emhass_weight_buy_forecast`, range −1 to +1). Adds DNSP free-tier adjustment (+1c during 10:00–16:00 if allowance > 0).

2. **`sensor.amber_effective_feed_in_price`** — same logic for feed-in (export) price.

3. **`script.emhass_dayahead_optim` / `script.emhass_mpc`** — wrapper scripts that compute `soc_init_pct` and `soc_final_pct` from the prior DH plan plus the live SoC, persist the chosen soc_init to `input_number.dh_last_soc_init` / `input_number.mpc_last_soc_init`, then fire the corresponding `rest_command` with the values as parameters. **Automations must call these scripts, not the rest_commands directly.** See [docs/production_soc_policy.md](docs/production_soc_policy.md) for the formulas (DH self-correction chain, MPC plan-relative deviation, force-charge top-balance bias).

4. **`rest_command.emhass_dayahead_optim` / `rest_command.emhass_mpc`** — build a JSON payload via Jinja2 and POST to the EMHASS endpoint. The payload includes:
   - `soc_init` / `soc_final` — passed in as `soc_init_pct` / `soc_final_pct` parameters from the wrapping script.
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

**Credential management:** `config.json` is committed with secrets stripped (`influxdb.password` and `home_assistant.token` are empty strings). Real secrets live in `config.secrets.json` (git-ignored), which `config_utils.load_config()` deep-merges at runtime. See `config.secrets.json.example` for the required structure. The HA YAML files need URL redaction before committing.

The systemd services load secrets from `.env` in the repo root (git-ignored). This file must be created manually:

```ini
# .env
HC_PREDICT_URL=https://hc-ping.com/<your-uuid>
```

---

## Roadmap

See **[docs/roadmap.md](docs/roadmap.md)** for: phase status, design principles, CI/CD gate
design, training weighting methodology, execution layer (tail-risk overrides), and known
open issues.

---

## Known Pain Points

1. **`forecast.py` is a monolith.** At ~3,500 lines it handles training, prediction, tariff management, logging, bias correction, HA publishing, and both Tier 1/Tier 2 inference. The natural module boundaries are clear (see table above) but the code is not yet split. Hard to navigate and test. Refactoring is gated on Phase 8 (test framework) to avoid regressions.

2. **`hass/package-emhass.yaml` Jinja complexity.** The EMHASS REST command payload is built entirely in Jinja2 template syntax inside a YAML string. It's ~350 lines of logic that is hard to debug, diff, and maintain.

3. **Ad-hoc ingest scripts are disconnected.** The manual/historical backfill scripts (`ingest-ha-data.py`, `ingest-nem-data.py`, etc.) are run ad-hoc with no systemd timers. The automated ingest scripts (predispatch, p5min, pd7day, sevendayoutlook) all use `config_utils.load_config()` and run via systemd.

4. **HA backups require manual redaction.** Every time `hass/` files are committed, any private hostnames or URLs must be manually redacted. This creates friction and risk.

5. **Forecast log CSVs are very large.** `price_forecast_log.csv` and `load_forecast_log.csv` are each ~330–340MB and growing. They live outside the repo (git-ignored) but are depended on by `backfill-actuals` and `update-adjusters`.

6. **Stale model files on disk.** Experimental quantile variants (price: p10, p20, p50, p80, p90; load: p60) are no longer referenced by `config.json` and consume ~1.3GB.

7. **`analyse.ipynb`** is a 36MB notebook with embedded output data committed to the repo. Should either have outputs stripped or be moved outside the repo.

8. ~~**`_publish_covariates_helper()`** is defined in `forecast.py` but never called~~ — removed (75c5c84).
9. ~~**`model/` scripts are out of sync.**~~ — removed (75c5c84).

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
