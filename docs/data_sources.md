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
| Household load | `power_load_5m`, `power_load_30m` | `rp_5m`, `rp_30m` | `mean_value` (W) | HA sensor → `rp_raw` → CQs |
| Solar PV | `power_pv_5m`, `power_pv_30m` | `rp_5m`, `rp_30m` | `mean_value` (W) | HA sensor → `rp_raw` → CQs |
| Dump load | `power_dump_load_30m` | `rp_30m` | `mean_value` (W) | HA sensor → `rp_raw` → CQ |
| Temperature | `temperature_adelaide` | `rp_30m` | `mean_value` (°C) | HA sensor → `rp_raw` → CQ |
| Humidity | `humidity_adelaide` | `rp_30m` | `mean_value` (%) | HA sensor → `rp_raw` → CQ |
| Wind speed | `wind_speed_adelaide` | `rp_30m` | `mean_value` (m/s) | HA sensor → `rp_raw` → CQ |

Dump load is subtracted from household load in `get_historical_data()` before any model sees it.

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

- Runs every 5 min (HA automation `minutes: /5`)
- First re-run triggered at ~:00:30 into each interval when amber2mqtt updates spot price
- Inputs: `sensor.amber_5min_forecasts_extended_general_price` (Amber APF), DH load/PV (interpolated 30→5 min), current SOC
- DH→MPC handoff: `periods_into_30min = (utcnow.minute % 30) // 5` skips elapsed 5-min sub-intervals — correctly "backdates" plan to start of current interval

### 5-min/30-min scheduling inaccuracy in combined shadow sensor

The MPC calculates per-period energy as:
```jinja2
{%- set forecast_end = forecast_start + timedelta(minutes=5) %}
```
This assumes every `Forecasts` item is 5-min. `sensor.ai_combined_general_price_forecast` has Tier 1 items (genuine 5-min) followed by Tier 2 items (genuine 30-min). The MPC will underweight Tier 2 periods by 6×.

**Fix (pending):** In `_build_combined_forecast_items`, when `interval_minutes=30`, emit 6 identical 5-min dict entries per step instead of one 30-min entry. No MPC YAML changes required.
