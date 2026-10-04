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
                         │
                         ▼
                  APF/LightGBM price (incumbent)
                  LightGBM load forecast
                  (suspended/archived: tactical Tier 1,
                   PD-direct, TFT price, TFT load shadow)
                         │
                                    │ ◄── HA future covariates (Solcast,
                                    │     BOM weather, Amber dynamic handoff)
                                    │
             ┌──────────────────────┤
             │                      │
             ▼                      ▼
     predictions.json       HA sensor entities:
     *_forecast_log.csv     sensor.ai_price_forecast        (APF/LightGBM p50, incumbent)
                            sensor.ai_price_forecast_low/high
                            sensor.ai_load_forecast
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
| `ai-energy-listener.service` | Event-driven (Amber APF state change in HA; 30-min idle heartbeat) | `forecast.py predict-price --dynamic-handoff --publish-hass` — added 2026-05-27, see [docs/price/event_driven_predict_price_plan.md](docs/price/event_driven_predict_price_plan.md) |
| `ai-energy-predict.timer` | Every 30 min (`:01` and `:31`) | `forecast.py predict-load --publish-hass --publish-covariates` — price path moved to the listener 2026-05-27; cadence aligned with the 30-min InfluxDB CQ granularity |
| `ai-energy-train.timer` | Monday 12:00 | `forecast.py train-load && forecast.py train-price` |
| `ai-energy-update-tariffs.timer` | Daily 00:00 | `forecast.py update-tariffs && forecast.py backfill-actuals && forecast.py update-adjusters` |

**AEMO data collection (added 2026-04-10):**

| Timer | Schedule | What it runs |
|---|---|---|
| `ai-energy-pd7day.timer` | 3×/day (07:20, 12:55, 18:05 AEST) | `ingest/ingest-pd7day.py --fetch` |
| `ai-energy-predispatch.timer` | Every 30 min (`:12` and `:42`) | `ingest/ingest-predispatch.py --fetch` — AEMO PREDISPATCH ingest only; PD-direct publish archived 2026-06-15 |
| `ai-energy-sevendayoutlook.timer` | Every 30 min (`:15` and `:45`) | `ingest/ingest-sevendayoutlook.py --fetch` |
| `ai-energy-p5min.timer` | Every 5 min (`:02/:07/:12/…/:57`) | `ingest/ingest-p5min.py --fetch` — AEMO P5MIN ingest only; tactical Tier 1 publish archived 2026-06-15 |

All units run as systemd user units (`systemctl --user`), `WorkingDirectory=~/src/ai-energy-forecast-slop`, activate `.venv` before running. Training is `Nice=19` (lowest CPU priority). Linger is enabled so units run without an active login session.

---

## Components

### `forecast.py` — Monolithic Orchestrator (~3,500 lines)

The core script. All behaviour is driven by subcommands:

| Subcommand | When | What it does |
|---|---|---|
| `train-price` | Weekly | Trains price quantile models on 2 years of 30-min InfluxDB data |
| `train-load` | Weekly | Trains load quantile models |
| `predict-all` | (manual) | Fetches covariates, runs both price+load models, applies tariffs/GST, saves JSON, publishes to HA. Production splits this into event-driven `predict-price` (via `ai-energy-listener.service`) and timer-driven `predict-load`. |
| `publish-tactical` | Archived | Former Tier 1 LGBM refresh; now a no-op after the 2026-06-15 soft archive. |
| `publish-pd-direct` | Archived | Former PD-direct + canonical AI shadow publisher; now a no-op after the 2026-06-15 soft archive. |
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
| **Prediction** | `_predict_simple`, `_predict_with_dynamic_handoff`, `_execute_quantile_prediction`, `_execute_single_prediction`. Archived inference helpers remain for tactical Tier 1, PD-direct, TFT price, and TFT load, but normal `predict-price` / `predict-load` no longer invoke them after the 2026-06-15 soft archive. | Inference |
| **Tariffs** | `get_amber_api_scaling_factor`, `get_network_loss_factor`, `_get_tariff_data`, `_create_complete_profile`, `_calculate_amber_api_scaling_factor`, `_calculate_forecasted_network_loss_factor`, `update_tariffs`, `apply_tariffs_to_forecast` | Tariff profile construction and application |
| **Adjusters** | `update_adjusters`, `apply_covariate_adjustments` | Weather covariate bias correction |
| **Logging** | `log_forecast_data`, `backfill_actuals`, `_backfill_single_log` | Forecast log CSVs |
| **Publishing** | `publish_forecast_to_hass`, `publish_adjusted_covariates_to_hass`, `_build_combined_forecast_items`, `_publish_combined_price_forecasts` | Push results to HA |
| **InfluxDB helpers** | `get_historical_data`, `_get_influx_sdo_demand` | InfluxDB queries |
| **Orchestrators** | `run_predictions`, `main` | Top-level entry points |

Dead code removed: `_publish_covariates_helper` (never called) and `model/` (01–04-*.py exploratory scripts, superseded by `forecast.py`).

#### Model Details

### Production bundle lifecycle

Production artifacts live under `models/production/<family>/bundles/<bundle_id>/`. A candidate is
written to a temporary directory, hashed in its manifest, and atomically renamed. `active.json`
is an atomic pointer; prediction resolves it once per family per run, so a pointer change cannot
mix quantile generations. `promote-bundle` and `rollback-bundle` are explicit operator commands.
Prediction exits nonzero on contract or publication failure; local output replacement occurs only
after a complete family validates.

Both price and load quantile families use the same alpha-ordered monotonic rearrangement before
the 144-point contract check. The identical policy is used by live inference and bundle smoke;
crossing independently trained curves therefore cannot make those two paths disagree.

Candidate promotion evidence is derived from a SHA-256-identified CSV/Parquet row set containing
candidate, incumbent, and actual values on identical issue/target rows. Fixed family horizon
buckets, finite values, units, MAE, bias, pinball loss, and empirical coverage are validated in
code; operator-supplied summary claims are ignored.

- **Framework:** Darts (time series library) + LightGBM quantile regression
- **Horizon:** 144 steps = 72 hours at 30-minute resolution
- **Active price quantiles:** p30, p50 (median), p70 — configured in `config.yaml`
- **Active load quantiles:** p50, p65, p75 — configured in `config.yaml`
- **Target lags:** 1–6, 12, 24, 48–49, 96–97, 336–337 (1h to 14 days)
- **Future covariate window:** ±4 lags around each step
- **Recency weighting:** exponential decay (price: 180-day half-life; load: 90-day half-life)
- **Log transform:** applied to price targets to handle negative/volatile prices
- **Anti-crossing:** quantile outputs are sorted to prevent p30 > p50

Several older model files exist on disk (price: p10, p20, p50, p80, p90; load: p60) from past experiments — these are not referenced by the current `config.yaml` and can be deleted.

#### Dynamic Handoff (price only)

1. Amber Electric provides a 36-hour advanced forecast (mixed 5-min + 30-min intervals)
2. Script averages 5-min intervals to 30-min, uses this as pseudo-historical seed
3. LightGBM model extends the forecast beyond Amber's horizon

#### Covariate Adjustments

Weather forecasts (BOM) and PV forecasts (Solcast) exhibit systematic biases. Daily at midnight, `update-adjusters` computes time-of-day additive bias corrections from the last 30 days of the forecast log. These are stored in `adjuster_temperature.json`, `adjuster_humidity.json`, `adjuster_wind_speed.json` and applied at prediction time before model input.

#### Tariff Pipeline

`update-tariffs` reconstructs the deterministic time-of-day tariff adders from Amber's `per_kwh`/`spot_per_kwh` forecasts:

- **`amber_api_scaling_factor`** is computed as a sanity check (Amber-vs-AEMO ratio) but its true value is the GST rate, so it is **snapped to exactly GST (1.1)** when in tolerance; a drift outside logs loudly and keeps the measured value (signals Amber changed their spot basis).
- **Loss factor + fixed adders** come from a **pooled per-band OLS** (`tariff_utils.fit_shared_slope`): one shared slope (the network loss factor) is fit across every interval of both legs at once, with a separate intercept (fixed adder) per (leg, Peak/Solar-Sponge/Off-Peak) bucket. This pools rounding noise far better than per-interval reconstruction + median. If the slope is poorly determined (flat-spot day) the previous loss factor is retained; if the fit is unavailable it falls back to median-smoothing the per-interval reconstruction (`smooth_tariff_maps`).
- GST convention: import leg carries GST only when a net cost; the feed-in leg is GST-free in both directions (see `docs/tariff_gst_regime.md`).

Outputs: the per-interval reconstruction → `tariff_profile_raw.json` (diagnostic); the OLS band-fit → `tariff_profile.json` (production). `tariff_profile.json` is written last and only on success, so a failure leaves the previous good profile intact.

At prediction time, `apply_tariffs_to_forecast()` applies the network loss factor and the GST convention above to produce final consumer prices.

---

### `ingest/` — Data Ingestion Scripts

**Active automated scripts** (have systemd timers, use `config.yaml`):

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
| HA InfluxDB integration + CQs | `sensor.power_consumed_without_deferrable_loads` | `rp_30m.power_load_without_deferrable_30m`, `rp_5m.power_load_without_deferrable_5m` |
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
| `power_load_30m` | `mean_value`, `min_value`, `max_value` — gross site consumed power, retained for history |
| `power_load_without_deferrable_30m` | `mean_value`, `min_value`, `max_value` — preferred model load input from HA `sensor.power_consumed_without_deferrable_loads`; excludes both HWC heat sources (heat-pump compressor and resistive element) and all dump loads |
| `power_dump_load_30m` | `mean_value`, `min_value`, `max_value` — aggregate dump load: estimated 2×2000W fan heaters plus metered 2×2150W nominal oil heaters; retained for fallback subtraction against older `power_load_30m` history |
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

### Tariff estimation (`tariff_utils.fit_shared_slope`, `smooth_tariff_maps`)

Both folded into `update-tariffs` (the standalone `smooth_tariffs.py` is gone). Buckets are Peak (17:00–20:59), Solar Sponge (10:00–15:59), Off-Peak (all others).

- **Primary — pooled OLS** (`fit_shared_slope`): fits one shared slope (network loss factor) plus a per-(leg, bucket) intercept (fixed adder) across all intervals at once. Statistically the BLUE for this linear reconstruction; pools per-interval rounding noise (Amber quantizes to 0.01 c/kWh) and yields the loss factor with a standard error.
- **Fallback — bucket median** (`smooth_tariff_maps`): replaces each slot with its bucket median of the per-interval reconstruction; scalars pass through. Used only when the OLS fit is unavailable.

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

### `data/` and `train/` — Archived TFT Price / Tactical Price Tracks

> **2026-06-15 status.** Active work on APF-free price paths is suspended, not
> fully abandoned. The code and history remain for reference and possible
> deliberate revival, but normal `predict-price` no longer runs tactical Tier 1,
> PD-direct, raw AEMO stitched, canonical AI MPC/DH, or TFT price shadow
> inference. The P5MIN/PREDISPATCH/PD7Day/SevenDayOutlook ingest timers stay
> active because they are useful historical inputs and may support future APF-tail
> residual work.

> **2026-05-05 status.** TFT iteration is paused. A live `--debug-tft` run on 2026-05-05
> showed Run 011b outputting 30–50% below its own debiased PREDISPATCH input even with the
> debiaser passive — i.e. the TFT compresses on its own, not just because of the debiaser.
> The 2026-05-05 `run011b_active_15` retrain was rejected on Window A/B `netload_tariffed`
> gates. No further TFT training is to be launched until a no-ML "PD-direct" baseline has
> been measured through the same gates. Full plan in `docs/roadmap.md` (top section,
> 2026-05-05 Strategic Pivot); structural critique in `docs/archive/price_forecast_2026/tft_price_forecast.md`.

The TFT price, tactical Tier 1, and PD-direct tracks have all produced useful
evidence, but they are no longer active production/shadow publishers. The active
price path is the APF/LightGBM extrapolation surfaced through
`sensor.ai_price_forecast(_low/_high)`. The archived checkpoints, logs, eval
scripts, and helper functions remain for explicit revival/comparison work only;
normal production runs should not consume `p5min_tactical`, `pd_direct`,
`model_a_hybrid`, or `lgbm_strategic` when evaluating APF extrapolation. Full
TFT price rationale is documented in **[docs/archive/price_forecast_2026/tft_price_forecast.md](docs/archive/price_forecast_2026/tft_price_forecast.md)**;
longer-term speculative ideas are captured in **[docs/ideas.md](docs/ideas.md)**.

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

**Status (2026-04-20):** Phases 1–9 + Phase 6 + Phase 8 complete. All financial gates pass — `tier1_tier2_hybrid` (Run 011b + binary spike routing) overall +9.7% vs amber_apf_lgbm baseline ✅. Active production model: Run 011b checkpoint. Phase 7 decoder expansion has now been trained twice: Run 014 (18-feature checkpoint) failed the interim holistic eval (**−35.3% overall vs amber_apf_lgbm**), and the follow-up flat-wMAPE ablation Run 015 failed even harder (**−65.9% overall**). Run 011b therefore remains the incumbent, and flat horizon weighting is not promoted. See `docs/roadmap.md`, `docs/archive/price_forecast_2026/tft_price_forecast.md`, and `docs/archive/price_forecast_2026/training_runs.md`.

---

### `data/` and `train/` — Archived TFT Load Model

> **2026-06-15 status.** EMHASS still consumes the LightGBM load forecast. TFT
> load shadow inference is disabled in normal `predict-load` runs and retained
> only for reference/revival work.

A TFT model for household load prediction, originally intended to shadow and
eventually replace the existing Darts/LightGBM load model. The existing model
uses manual lag engineering (t-48, t-96, t-336 etc.) to capture daily/weekly
seasonality; TFT replaces this with attention.

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
4. Archived shadow implementation in `forecast.py` → `_execute_tft_load_prediction()`.

**Archived checkpoint:** `models/tft_load/checkpoint_best.pt` (Run 005, epoch 32). Overall MAE 234W.

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

The hot-water controller moved to the sibling `../hwc` repository on 2026-08-10. This
repository consumes its published `sensor.hwc_power_plan` as a Home Assistant contract;
the controller code, runtime state, dashboard, and HWC-specific HA configuration live there.

> Production terminal-SoC policy (DH offset feedback, MPC inheritance, eval-vs-production
> correspondence): see [docs/production_soc_policy.md](docs/production_soc_policy.md).

| File | Purpose |
|---|---|
| `emhass.yaml` | HA package: template sensors for price/feed-in blending, the `script.emhass_dayahead_optim` / `script.emhass_mpc` wrappers that compute soc_init/soc_final and persist them to helper input_numbers, and the underlying `rest_command.emhass_dayahead_optim` / `rest_command.emhass_mpc` that POST a Jinja-built JSON payload to EMHASS |
| `curtailment_policy.yaml` | HA package: timestamp-aware remaining-local-day PV-curtailment sensor, ordered HWC/dump-load/proactive-PV threshold helpers, ordering validation, and invalid-policy notification. HWC consumes these HA entities and fails closed when they are unavailable. |
| `automation-sigenergy-emhass.yaml` | HA automation: reads EMHASS `mpc_*` output entities every 5 min, evaluates battery control scenarios (grid charge, PV curtail, discharge, standby), calls EMS script. From 10:00 inclusive to 16:00 exclusive local time, zero-grid charging plans and battery-full PV-curtailment plans use maximum self-consumption with discharge disabled, so a transient PV shortfall imports rather than reversing into battery discharge before the next MPC solve. Outside that cheap-tariff window, battery-full curtailment may fall back to battery discharge. |
| `hass/packages/sigenergy_ems.yaml` | HA package: downstream Sigenergy execution layer. Defines `script.configure_sigen_ems_state`, `script.sigen_apply_limits`, the master-limit controller, export-ramp-hold controller, and export-ramp-release automation together so their ordering/timing/helper invariants live in one file. Upstream EMHASS policy remains in `automation-sigenergy-emhass.yaml`. This package replaces the retired split files; do not load both forms in HA at once. The repo `hass/packages/` directory is intended to mirror HA's `config/packages/`; filenames use HA-safe underscore package slugs. |

> **Helpers required in HA:**
> - `input_number.transient_pcs_export_cap` (min 0, max 100, step 0.001, **kW**). The
>   export ramp-hold writes it; `apply_limits` mins it with `desired_pcs_export_limit`
>   **but only while `timer.sigen_export_ramp` is active**, so the helper's resting value
>   is irrelevant (a value of 0 can no longer strangle PCS export when idle).
> - `timer.sigen_export_ramp` (a Timer helper) — the single "hold active" signal; its
>   duration is overridden per `timer.start` call by the ramp-hold automation.
> - Existing desired-limit helpers: `input_number.desired_export_limit`,
>   `input_number.desired_import_limit`, `input_number.desired_pcs_export_limit`,
>   `input_number.desired_pcs_import_limit`, and `input_number.flexible_export_limit`.
>
> These helpers are intentionally not declared in `hass/packages/sigenergy_ems.yaml` yet, so
> existing HA UI-created helper history is preserved and migration avoids duplicate
> entities. They can be moved into the package later if full YAML ownership is desired.

**Sigenergy timing policy:**

- **Command-settle delays:** use `00:00:00.1` (100 ms) after ordered Sigen service
  calls when HA must give the inverter a small physical serialization window before
  issuing the next potentially unsafe write. Keep these delays short and explicit;
  do not add multi-second sleeps for command ordering without evidence.
- **Readback waits:** use bounded `wait_template` checks when HA state readback can
  confirm that a guard limit or mode has landed. Current EMS transition waits use
  `timeout: "00:00:02"` with `continue_on_timeout: true`; these are maximum waits,
  not fixed delays.
- **Sensor fallback waits:** event-driven loops may use longer timeouts only as
  fallbacks when a sensor does not update. The export-ramp loop waits up to 8 s for a
  fresh grid-power update, but normally proceeds on the next event.
- **Modelled process timers:** do not shorten these for responsiveness. The
  `timer.sigen_export_ramp` duration models the inverter's internal grid-export-limit
  ramp (`delta / 300 W/s + 3 s`, capped at 60 s), not command serialization.

#### `emhass.yaml` in detail

This is the most complex HA file. It does:

1. **`sensor.amber_effective_general_price`** — blends current Amber spot price with a risk-weighted advanced forecast (controlled by `input_number.emhass_weight_buy_forecast`, range −1 to +1). Adds DNSP free-tier adjustment (+1c during 10:00–16:00 if allowance > 0).

2. **`sensor.amber_adjusted_confirmed_feed_in_price` / `sensor.amber_effective_feed_in_price`** — separates confirmed Amber export history with local economic adjustments from the control price. The effective sensor uses the adjusted confirmed value while valid, then falls back to the weighted and adjusted next forecast.

3. **`sensor.emhass_current_pv_input_mode`** — classifies the live PV measurement as `measured`, `transition`, `pv_limit`, or `export_limit` from applied Sigenergy limits and physical power flows. Battery absorption headroom uses the lower of the configured EMS charge limit and `sensor.sigen_inverter_max_battery_charge_power`, so the inverter's top-of-charge capability taper can establish a binding export limit before derived SoC reaches the fallback threshold. MPC reconstructs available PV with `max(measured, Solcast)` only during a limit/telemetry transition or while a limit is physically binding. A prior EMHASS curtailment plan, negative import price, and battery SoC alone are not curtailment evidence. The classifier is tariff-time-independent; the separate 10:00–16:00 grid-import preference remains downstream execution policy.

4. **`script.emhass_dayahead_optim` / `script.emhass_mpc`** — wrapper scripts that compute `soc_init_pct` and `soc_final_pct` from the prior DH plan plus the live SoC, persist the chosen soc_init to `input_number.dh_last_soc_init` / `input_number.mpc_last_soc_init`, then fire the corresponding `rest_command` with the values as parameters. **Automations must call these scripts, not the rest_commands directly.** See [docs/production_soc_policy.md](docs/production_soc_policy.md) for the formulas (DH self-correction chain and MPC plan-relative deviation).

5. **`rest_command.emhass_dayahead_optim` / `rest_command.emhass_mpc`** — build a JSON payload via Jinja2 and POST to the EMHASS endpoint. The payload includes:
   - `soc_init` / `soc_final` — passed in as `soc_init_pct` / `soc_final_pct` parameters from the wrapping script.
   - PV forecast: Solcast p10/p50/p90 blended by `input_number.emhass_weight_pv_forecast`, with 65W fixed loss applied
   - Load forecast: base load from `sensor.ai_load_forecast_high` (p65 model), plus planned
     HWC compressor power from `sensor.hwc_power_plan` added in the day-ahead EMHASS payload.
     The load model itself uses `power_load_without_deferrable_30m`, which excludes both the
     HWC heat-pump compressor and resistive element, so HWC demand is not learned as ordinary
     household load. Only the planned compressor load is added back to the EMHASS forecast;
     reactive resistive-element events are not forecast loads.
   - Price forecast: from `sensor.ai_price_forecast` (p50), blended with p30/p70 by `input_number.emhass_weight_buy_forecast`
   - Battery costs: scalar zero charge weight; HA-supplied discharge weight. EMHASS
     `0.17.7+` breaks economically equivalent solutions toward later PV curtailment.
     Battery-power PWL stress is disabled; inverter AC-power PWL stress remains enabled.

---

### Output Files

| File | Contents |
|---|---|
| `predictions.json` | Latest full forecast (price + load, all quantiles, 144 steps each, ~64KB) |
| `price_forecast_log.csv` | Historical predictions + actuals for price (~330MB, growing) |
| `load_forecast_log.csv` | Historical predictions + actuals for load (~340MB, growing) |
| `tariff_profile.json` | Smoothed 48-slot daily tariff profile |
| `tariff_profile_raw.json` | Raw per-interval tariff reconstruction (diagnostic; pre-smoothing) |
| `adjuster_*.json` | Weather covariate bias corrections by time-of-day |
| `price_model.pkl` / `price_p30_model.pkl` / `price_p70_model.pkl` | Trained price models (~128MB each) |
| `load_model.pkl` / `load_p65_model.pkl` / `load_p75_model.pkl` | Trained load models (~82MB each) |
| `*_importance.json` | Feature importances from last training run |

Note: several older model files exist (`price_p10`, `price_p20`, `price_p50`, `price_p80`, `price_p90`, `load_p60`) from previous experiments — only p30/p50/p70 (price) and p50/p65/p75 (load) are currently active.

---

## Configuration

`config.yaml` (from `config.example.yaml`) holds all credentials and settings:
- InfluxDB host/port/db/credentials
- Home Assistant URL + long-lived token
- Entity IDs for all HA sensors
- Model paths
- Per-model hyperparameters (n\_estimators, lags, horizon, quantiles, recency weighting)
- Adjuster settings

**Credential management:** `config.yaml` is committed with secrets stripped (`influxdb.password` and `home_assistant.token` are empty strings). Real secrets live in `config.secrets.yaml` (git-ignored), which `config_utils.load_config()` deep-merges at runtime. See `config.secrets.yaml.example` for the required structure. The HA YAML files need URL redaction before committing.

The systemd services load secrets from `.env` in the repo root (git-ignored). This file must be created manually:

```ini
# .env
HC_PREDICT_URL=<existing-healthcheck-ping-url>
```

Jobs write status locally; one minute-level aggregate reports their combined state to this
existing Healthchecks check. This keeps one remote check while ensuring an individual job failure
cannot be cleared by another job's success. See
[`docs/healthchecks.md`](docs/healthchecks.md) for monitored jobs and freshness limits.

---

## Roadmap

Current implementation priority:
[production forecast hardening](docs/price/production_hardening_plan_2026-08-10.md).

`docs/roadmap.md` is historical experiment context, not the active implementation plan.

---

## Known Pain Points

1. **Candidate quality evidence is incomplete.** Candidate bundles, atomic promotion, rollback,
   and candidate-only weekly training are implemented, but reports remain ineligible until
   identical-row metrics and inference smoke evidence are supplied.

2. **HA publication is not transactional.** A later entity POST can fail after earlier entities
   changed; the command reports the potentially changed entity IDs and exits nonzero.

3. **Historical/live covariates differ.** Training uses realised PV/weather/demand and historical
   STPASA selection does not exactly reproduce live forecast issuance. Existing screening results
   remain useful, but are not fully causal promotion evidence.

4. **`forecast.py` is a monolith.** It handles training, prediction, tariff management, logging,
   bias correction, HA publishing, and archived inference paths. Refactor only behind behavioural
   tests; do not combine broad decomposition with production hardening.

5. **`hass/packages/emhass.yaml` Jinja complexity.** The EMHASS REST command payload is built
   entirely in Jinja2 template syntax inside YAML strings and is hard to debug, diff, and maintain.

6. **Ad-hoc ingest scripts are disconnected.** The manual/historical backfill scripts (`ingest-ha-data.py`, `ingest-nem-data.py`, etc.) are run ad-hoc with no systemd timers. The automated ingest scripts (predispatch, p5min, pd7day, sevendayoutlook) all use `config_utils.load_config()` and run via systemd.

7. **HA backups require manual redaction.** Every time `hass/` files are committed, any private hostnames or URLs must be manually redacted. This creates friction and risk.

8. **Forecast log CSVs are very large.** `price_forecast_log.csv` is over 1GB and
   `load_forecast_log.csv` is over 800MB as of 2026-08-10. They are git-ignored but remain runtime
   dependencies for actual backfill, adjusters, and evaluation.

9. **Historical model artifacts are bulky.** TFT checkpoints and suspended LightGBM/debiaser
   bundles remain useful as evidence but need explicit retention rules; do not confuse them with
   active production artifacts.

---

## Technology Stack

| Layer | Technology |
|---|---|
| ML framework | [Darts](https://unit8co.github.io/darts/) + LightGBM (production); retained PyTorch TFT experiments (suspended) |
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

The proposed consolidation across Amber, Python services, EMHASS, HA and HWC is tracked in
[the energy pipeline architecture plan](docs/energy_pipeline_architecture_plan.md).
Status: initial assessment; no production cutover.

Offline Python DH/MPC payload extraction and validation: [replay runbook](docs/energy_pipeline_payload_replay.md). Production solve ownership remains in HA.

Resident price calculation-only successor: [shadow worker](docs/energy_pipeline_resident_price.md). Model reuse, independent source caches, input lineage/freshness evidence, [frozen tariffs](docs/energy_pipeline_tariff_snapshot.md) and measured memory reclamation implemented in shadow. [Result acceptance](docs/energy_pipeline_price_acceptance.md) rejects changed/expired inputs before advancing shadow state. Target-filtered STPASA reads reduce archive expansion with exact feature parity. [BOM hourly freshness patch and capture adapter](docs/energy_pipeline_bom_freshness.md): adapter checks observed updates; integration patch unapplied, atomic association unverified. Opt-in [durable accepted checkpoint](docs/energy_pipeline_accepted_store.md) added; recovery always requires fresh reconciliation. [Publication transaction](docs/energy_pipeline_publication.md) rehearsed in SQLite only; production publisher unchanged. [DH/MPC handoff](docs/energy_pipeline_handoff.md) uses the accepted bundle for DH; MPC retains separately identified Amber/existing DH parents, with solve permission disabled. [Weather admission](docs/energy_pipeline_weather_coverage.md) allows at most one hour of absent tail targets while preserving incumbent post-adjustment fill. Long-run memory validation and complete upstream freshness metadata remain cutover gates. [Current evidence/resume gates](docs/energy_pipeline_source_cache.md); production ownership unchanged.

Isolated solver checkpoint: [audit and result contract](docs/energy_pipeline_solver_isolation.md).
Recorded core solves and [historical DH→MPC chain](docs/energy_pipeline_solver_chain.md) verified;
[economic replay gaps/PV provenance and next work](docs/economic_replay_checkpoint_2026-10-04.md).
[Seven-day measured targets and causal load-calibration trial](docs/measured_economic_actuals_2026-10-04.md)
now available; bounded time-weighted raw telemetry, original model covariates/CQs unchanged.
Offline residual correction fits completed past targets only; no production model/policy promotion.
[Economic comparisons](docs/parallel_economic_findings_2026-10-04.md) share one identity-checked
isolated core runner. Observed accounting distinguishes MQTT feed states from raw APF attributes;
historical APF pilot retains revision receipts separately from quoted targets.
[Sequential replay](docs/sequential_economic_replay_2026-10-04.md) reuses direct core solves in
one isolated container, evolving simulated inventory per arm with lagged telemetry and as-of APF.
Fixed DH/HWC/risk inputs and delivered-PV lower bound remain explicit limits; no savings promotion.
[Control fidelity audit](docs/control_fidelity_audit_2026-10-04.md) reconstructs recorded inputs:
three pinned core checkpoints reproduce live commands≤0.005W; cadence/parent input differences identified.
[Minute replay](docs/minute_economic_replay_2026-10-04.md) now uses raw event-grid targets and
historical activation delays; conditional archived parents, own inventory, shared isolated batch runner.
[DH source admission](docs/economic_regimes_and_dh_admission_2026-10-04.md) reconstructs as-of
production price/load/Solcast/settings; exact UTC target-grid validation catches four half-hour load
lags. Static capture fallbacks identified explicitly; no production admission change yet.
[Own-DH event replay](docs/dh_feedback_economic_replay_2026-10-04.md) persists each arm's SoC,
trajectories/anchors/reground/offset, delays parent/command activation, and holds accepted parents
on bad alignment. HWC exogenous; execution uses runtime capacity, not base-config placeholder.
[Load calibration experiment](docs/causal_load_feedback_results_2026-10-04.md) freezes p65 corrections
at matched forecast creation, applies only admitted DH base load, and retains incumbent terminal
policy. Separate saved-plan/future-target auditor; no scored future labels in replay decision inputs.
