# Data Sources & Pipeline Audit

**Last updated: 2026-04-18**

---

## AEMO Sources (NEMweb direct scraping)

| Source | NEMweb path | InfluxDB measurement | RP | Fields | Ingest timer | Pub lag |
|--------|-------------|---------------------|-----|--------|--------------|---------|
| **P5MIN** | `CURRENT/P5_Reports/` | `aemo_p5min_forecast` | `rp_5m` | `rrp` ($/MWh), `total_demand`, `net_interchange` | `*:02,07,12,17,22,27,32,37,42,47,52,57` | ~1–2 min |
| **PREDISPATCH** | `CURRENT/Predispatch_Reports/` | `aemo_predispatch_forecast` | `rp_30m` | `rrp`, `total_demand`, `net_interchange` | `*:12,42` | ~10 min |
| **PD7Day** | `CURRENT/PD7Day/` | `aemo_pd7day_forecast` | `rp_30m` | `rrp` | 3×/day (07:20, 12:55, 18:05 AEST) | ~10 min |
| **SevenDayOutlook** | `CURRENT/SEVENDAYOUTLOOK_FULL/` | `aemo_sevendayoutlook` | `rp_30m` | `scheduled_demand`, `scheduled_capacity`, `net_interchange`, `scheduled_reserve` | `*:01,31` | ~5 min |
| **Dispatch actuals SA1** | NEMOSIS / NEMweb archive | `aemo_dispatch_sa1_5m`, `aemo_dispatch_sa1_30m` | `rp_5m`, `rp_30m` | `price`, `total_demand`, `net_interchange` | Manual backfill | — |
| **Dispatch actuals VIC1/NSW1** | NEMOSIS / NEMweb archive | `aemo_dispatch_vic1_30m`, `aemo_dispatch_nsw1_30m` | `rp_30m` | same | Manual backfill | — |

**Reliability:** NEMweb CURRENT polling is adequate for a 5-min pipeline. The supplemental AEMO viz API (`visualisations.aemo.com.au`) times out regularly — treat as opportunistic only.

---

## Home Assistant Sources (REST pull each predict run)

| HA entity | Source | Fields consumed | Update freq | Consumers |
|-----------|--------|-----------------|-------------|-----------|
| `weather.woodville_west_hourly` | BOM via HA weather integration | `temperature`, `humidity`, `wind_speed` (next 48h) | ~hourly | All model decoders |
| `sensor.solcast_pv_forecast_forecast_today/tomorrow/day_3/day_4` | Solcast HA integration | `pv_estimate` (kW → W) per 30-min slot | ~30 min | TFT Load decoder, LightGBM load future covariate |
| `sensor.amber_30min_forecasts_general_price` | Amber HA integration | `Forecasts[].spot_per_kwh` (scaled ×1.10) | ~30 min | LightGBM price dynamic handoff seed |
| `sensor.amber_30min_forecasts_feed_in_price` | Amber HA integration | same structure | ~30 min | Feed-in tariff context |
| `sensor.amber_billing_interval_forecasts_general_price` | Amber HA integration | 5-min + 30-min forecasts | ~5 min | (available; not used as primary model input) |

**All HA interaction is REST polling** (`GET /api/states/`, `POST /api/states/`, `POST /api/services/weather/get_forecasts`). No websocket subscriptions. HA WebSocket API is available for a future event-driven architecture.

**Amber vs direct AEMO**: The AI pipeline scrapes AEMO NEMweb directly for dispatch/forecast data. Amber prices flow HA → AI only (no double-scraping). If "Amber Express" replaces amber2mqtt, verify it doesn't duplicate P5MIN data already ingested by `aemo-p5min`.

---

## InfluxDB HA-fed measurements (continuous queries)

| Source | InfluxDB measurement | RP | Fields | CQ from |
|--------|---------------------|-----|--------|---------|
| Household load | `power_load_5m`, `power_load_30m` | `rp_5m`, `rp_30m` | `mean_value` (W) | Gross HA consumed power; retained for history |
| Household load without deferrable loads | `power_load_without_deferrable_5m`, `power_load_without_deferrable_30m` | `rp_5m`, `rp_30m` | `mean_value` (W) | HA `sensor.power_consumed_without_deferrable_loads` → `rp_raw` → CQs |
| Solar PV | `power_pv_5m`, `power_pv_30m` | `rp_5m`, `rp_30m` | `mean_value` (W) | HA sensor → `rp_raw` → CQs |
| Dump load | `power_dump_load_30m` | `rp_30m` | `mean_value` (W) | HA sensor → `rp_raw` → CQ |
| Temperature | `temperature_adelaide` | `rp_30m` | `mean_value` (°C) | HA sensor → `rp_raw` → CQ |
| Humidity | `humidity_adelaide` | `rp_30m` | `mean_value` (%) | HA sensor → `rp_raw` → CQ |
| Wind speed | `wind_speed_adelaide` | `rp_30m` | `mean_value` (m/s) | HA sensor → `rp_raw` → CQ |

Forecast/export code prefers `power_load_without_deferrable_30m`. Where that newer series is
missing, it falls back to `power_load_30m` minus `power_dump_load_30m` so older history remains
usable.

The production load forecast is therefore a **base household load forecast**. The EMHASS
day-ahead payload reconstructs the expected whole-site load by adding the planned HWC
compressor block from `sensor.hwc_power_plan` at payload-preparation time. If the 48h HWC
plan is shorter than the 72h day-ahead horizon, the final 24h of planned HWC power is
repeated as a plausible anchor. The dump load is not added because it is opportunistic
negative-price behaviour, not scheduled deferrable demand.

---

## Data cascade by model

### Tier 1 — Tactical LightGBM (`_execute_tactical_prediction`)

Inputs: last 4h of 5-min data.

```
rp_5m.aemo_p5min_forecast    → p5min rrp (12 steps × 5-min)
rp_5m.aemo_dispatch_sa1_5m   → actuals: rrp (divergence feature), total_demand
rp_5m.power_pv_5m            → residual demand = total_demand − pv
```

Features built: `p5min_rrp`, `aemo_divergence_t-1`, `rolling_1h_std`, `rolling_3h_max`, `residual_demand`, time features, `is_imputed_p5min`, `is_intervention`.

### Tier 2 — TFT Price (`_execute_tft_prediction`)

Encoder (last 96 steps = 48h):
```
rp_30m.aemo_dispatch_sa1_30m   → rrp (converted $/kWh → $/MWh for TFT)
rp_30m.*                        → total_demand, net_interchange, power_load, power_pv, weather
rp_5m.aemo_dispatch_sa1_5m     → rrp_5m_max, rrp_5m_std, rrp_persistence, rrp_volatility_30m
```

Decoder (next 144 steps = 72h):
```
rp_30m.aemo_predispatch_forecast  → pd_rrp, vic1_pd_rrp, nsw1_pd_rrp (steps 0–55, 0.5–28h)
rp_30m.aemo_pd7day_forecast       → pd_rrp fill (steps 56–143, 28–72h)
rp_30m.aemo_sevendayoutlook       → sd_demand, sd_net_interchange (all 144 steps)
combined_covariates_df             → pd_demand, pd_net_interchange, weather, time, horizon_norm
```

Scalers: `models/tft_price/scalers.pkl` (must match `data/parquet/scalers.pkl` — kept in sync manually after dataset rebuilds).

### LightGBM Price & Load (`_execute_single_prediction`)

Both consume `get_historical_data()` (10 days, 30-min) + `combined_covariates_df` (72h future).

Price model additionally: Amber spot override via `get_amber_advanced_forecast()` when `--dynamic-handoff` is set.

### TFT Load (`_execute_tft_load_prediction`)

Encoder: power_load, power_pv, weather, calendar (48h history).
Decoder: Solcast PV forecast + BOM weather forecast + calendar (72h).

---

## Freshness requirements

| Data path | Maximum tolerable staleness | Failure mode |
|-----------|----------------------------|--------------|
| P5MIN → Tier 1 | 5 min | Tactical uses wrong dispatch regime |
| PREDISPATCH → TFT decoder steps 0–55 | 30 min | TFT gets wrong short-horizon signal |
| SevenDayOutlook → TFT decoder all steps | 30 min | TFT gets wrong demand signal |
| Amber prices → LightGBM handoff | 30 min | Reverts to no-handoff (gracefully degraded) |
| BOM weather → all decoders | 60 min | Mild degradation |
| Solcast → load/TFT decoders | 30 min | PV mismatch in load forecast |

---

## Ingest timer alignment (UTC)

```
:00       P5MIN dispatch interval starts
:02       aemo-p5min ingest (2-min AEMO publish buffer)
:01       aemo-sevendayoutlook ingest
:12       aemo-predispatch ingest
:01,:31   ai-energy-predict runs
```

**Known gap:** `ai-energy-predict` at `:01` runs before PREDISPATCH ingest at `:12`. The TFT decoder gets PREDISPATCH from the *previous* 30-min boundary (~29 min old). This is within tolerance. Moving predict to `:14,:44` would give PREDISPATCH fresh from `:12` and SDO from `:01`. Deferred — see housekeeping.

---

## EMHASS triggering model

### Day-ahead (DH): 72h × 30-min

- Triggered by HA automation when `sensor.ai_load_forecast` updates (~1 min after predict run)
- Inputs: `sensor.ai_price_forecast` (p50), p30/p70 for risk adjustment, `sensor.ai_load_forecast_high` (p65), Solcast PV
- SOC snapshot: `battery_soc_30_minute` (start of current 30-min interval)
- Produces: `sensor.dh_p_load_forecast`, `sensor.dh_p_pv_forecast`, `sensor.dh_soc_batt_forecast`

### MPC: 14h × 5-min

- Uses a 14h horizon with 5-minute periods
- First re-plan in each 5-minute cycle is triggered at approximately `:00:30` when the
  confirmed spot price update lands (timing floats slightly with source latency)
- Additional MPC re-plans then occur at roughly `:01:30`, `:02:30`, `:03:30`, and `:04:30`
  before the next 5-minute boundary
- Inputs: `sensor.amber_5min_forecasts_extended_general_price` (Amber APF), DH load/PV (interpolated 30→5 min), current SOC
- DH→MPC handoff: `periods_into_30min = (utcnow.minute % 30) // 5` skips elapsed 5-min sub-intervals — correctly "backdates" plan to start of current interval

### Battery action timing

- Battery action updates are asynchronous to MPC solves
- Action is refreshed on the 5-minute boundaries (`:00:00`, `:05:00`, …) even if MPC has not
  produced a newer plan yet
- If a fresh MPC plan arrives between boundaries and changes the requested action, the battery
  controller can update immediately; otherwise it keeps following the most recent plan

### 5-min/30-min scheduling inaccuracy in retired combined shadow sensor

The MPC calculates per-period energy as:
```jinja2
{%- set forecast_end = forecast_start + timedelta(minutes=5) %}
```
This assumes every `Forecasts` item is 5-min. The original combined AI shadow sensor used
Tier 1 items at genuine 5-minute cadence, then published Tier 2 as 30-minute items, which
would have caused the MPC to underweight Tier 2 periods by 6x.

**Status:** fixed on `2026-04-24` and later superseded. The old Amber-shaped AI combined
sensors were retired on `2026-05-08`; the canonical MPC import/export publisher still expands
30-minute Tier 2 steps into six identical 5-minute points before tariff application and Home
Assistant publication.
