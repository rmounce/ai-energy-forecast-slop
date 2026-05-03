# Home Assistant Entity Inventory

All sensors, helpers, and binary sensors relevant to the energy pipeline.
Categorised by source. Live state values are snapshots from 2026-05-03.

---

## amber2mqtt (MQTT bridge to Amber Electric API)

Entity naming: `sensor.amber_5min_*`, `sensor.amber_30min_*`, `sensor.amber_billing_interval_*`

Forecast lists use attribute `Forecasts` (capital F). Each item has fields:
`start_time`, `end_time`, `per_kwh`, `spot_per_kwh`, `advanced_price_low`,
`advanced_price_predicted`, `advanced_price_high`, `duration`, `spike_status`,
`descriptor`, `estimate`, `renewables`, `nem_date`, `type`

Sign convention: feed-in prices are **negative** (Amber convention: cost of exporting = negative).

### Current interval sensors

| Entity | Description | Key attributes |
|---|---|---|
| `sensor.amber_5min_current_general_price` | Current 5-min import price ($/kWh incl. tariff) | `end_time` (UTC), `per_kwh`, `spot_per_kwh` |
| `sensor.amber_5min_current_feed_in_price` | Current 5-min export/feed-in price (negative = revenue) | `end_time`, `per_kwh` |
| `sensor.amber_5min_current_aemo_spot` | Raw SA1 AEMO spot price | — |
| `sensor.amber_5min_current_general_descriptor` | Price descriptor string (`extremelyLow`, `neutral`, `spike`, …) | — |
| `sensor.amber_5min_current_feed_in_descriptor` | Feed-in descriptor | — |
| `sensor.amber_5min_current_spike` | Spike status (`none`, `potential`, `spike`) | — |
| `sensor.amber_5min_current_period_time` | Start time of current interval | — |
| `sensor.amber_5min_current_last_updated` | Timestamp of last amber2mqtt update | — |

### Forecast sensors

| Entity | Items | Resolution | Horizon | APF? | Used by |
|---|---|---|---|---|---|
| `sensor.amber_5min_forecasts_general_price` | ~10 | 5-min | ~50 min | ✓ | `package-emhass.yaml` (effective price blending) |
| `sensor.amber_5min_forecasts_feed_in_price` | ~10 | 5-min | ~50 min | ✓ | `package-emhass.yaml` (effective feed-in blending) |
| `sensor.amber_5min_forecasts_extended_general_price` | 288 | 5-min | ~24h | ✓ | `package-emhass.yaml` (MPC REST payload) |
| `sensor.amber_5min_forecasts_extended_feed_in_price` | 288 | 5-min | ~24h | ✓ | `package-emhass.yaml` (MPC REST payload) |
| `sensor.amber_30min_forecasts_general_price` | 74 | 30-min | ~37h | ✓ | `config.json` as `amber_entity`; `forecast.py` LightGBM seed |
| `sensor.amber_30min_forecasts_feed_in_price` | 74 | 30-min | ~37h | ✓ | `config.json` as `amber_feed_in_entity` |
| `sensor.amber_billing_interval_forecasts_general_price` | ~56 | mixed 5+30-min | ~24h | ✓ | `config.json` as `amber_billing_entity`; `forecast.py` APF-seeded LightGBM quantiles |
| `sensor.amber_billing_interval_forecasts_feed_in_price` | ~56 | mixed 5+30-min | ~24h | ✓ | — |
| `sensor.amber_5min_forecasts_aemo_price` | ~10 | 5-min | ~50 min | — | — |
| `sensor.amber_30min_forecasts_aemo_price` | 74 | 30-min | ~37h | — | — |
| `sensor.amber_billing_interval_forecasts_aemo_price` | ~56 | mixed | ~24h | — | — |

APF = Amber Price Forecast quartile bands (`advanced_price_low/predicted/high`).

### Individual period sensors (unused)

`sensor.amber_5min_period_1_general_price` through `_12_*`, plus `_feed_in_price` and `_aemo_spot_price` variants.
36 sensors total. Expose the next 12 × 5-min intervals as scalar sensors. Not referenced by any code or automation in this repo.

---

## Amber Express (native HA integration)

Entity naming: `sensor.amber_express_home_*`

Runs in parallel with amber2mqtt, consuming the same Amber API. Not referenced by any
code or automation in this repo — present but **redundant**.

Forecast lists use attribute `forecasts` (lowercase). APF is nested:
`item.advanced_price_predicted = {low: X, predicted: Y, high: Z}` (vs amber2mqtt's flat fields).
Feed-in prices are **negative** (same convention as amber2mqtt).

### Price sensors

| Entity | Description | Items | Horizon |
|---|---|---|---|
| `sensor.amber_express_home_general_price` | Current import price + simple `forecast` list (`{time, value}` pairs, local TZ) | 57 | ~24h |
| `sensor.amber_express_home_general_price_detailed` | Current import price + detailed `forecasts` (mixed 5+30-min, APF nested, UTC) | 57 | ~24h |
| `sensor.amber_express_home_feed_in_price` | Current feed-in + simple forecast list | 57 | ~24h |
| `sensor.amber_express_home_feed_in_price_detailed` | Current feed-in + detailed forecasts | 57 | ~24h |
| `sensor.amber_express_home_renewables` | Current renewables % | — | — |
| `sensor.amber_express_home_forecast_horizon` | Forecast horizon in hours | — | — |

### Status / control

| Entity | Description |
|---|---|
| `binary_sensor.amber_express_home_price_spike` | True when spike is active or imminent |
| `binary_sensor.amber_express_home_demand_window` | True during Amber demand response window |
| `select.amber_express_home_pricing_mode` | Amber smart pricing mode selector (current: `advanced_price_predicted`) |
| `sensor.amber_express_home_api_status` | API health (`OK`) |
| `sensor.amber_express_home_confirmation_delay` | Seconds between interval start and confirmed price |
| `sensor.amber_express_home_confirmation_lag` | Lag between AEMO dispatch and Amber API update |
| `sensor.amber_express_home_rate_limit_remaining` | API calls remaining in current window |
| `sensor.amber_express_home_rate_limit_reset` | Time API rate limit resets |
| `sensor.amber_express_home_site` | Site name (`SA Power`) |

---

## forecast.py (published by AI pipeline scripts)

All sensors use `forecasts` (lowercase) attribute unless noted. UTC ISO 8601 timestamps.
Published by `systemd/ai-energy-forecast.{service,timer}` (every 5 min) and
`ai-energy-dayahead.{service,timer}` (every 30 min).

### Price forecasts

| Entity | Description | Items | Resolution | Horizon | Attribute |
|---|---|---|---|---|---|
| `sensor.ai_price_forecast` | Tier 2 TFT q50 general import price ($/kWh) | 144 | 30-min | 72h | `forecasts` |
| `sensor.ai_price_forecast_high` | Tier 2 TFT q90 | 144 | 30-min | 72h | `forecasts` |
| `sensor.ai_price_forecast_low` | Tier 2 TFT q10 | 144 | 30-min | 72h | `forecasts` |
| `sensor.ai_aemo_price_forecast` | Tier 2 TFT q50 as raw AEMO $/MWh (pre-tariff) | 144 | 30-min | 72h | `forecasts` |
| `sensor.ai_p5min_price_forecast` | Tier 1 LightGBM q50 ($/kWh) | 12 | 5-min | 60 min | `forecasts` |
| `sensor.ai_p5min_price_forecast_high` | Tier 1 LightGBM q90 | 12 | 5-min | 60 min | `forecasts` |
| `sensor.ai_p5min_price_forecast_low` | Tier 1 LightGBM q10 | 12 | 5-min | 60 min | `forecasts` |
| `sensor.ai_tft_price_forecast` | TFT standalone q50 (no LightGBM hybrid) | 144 | 30-min | 72h | `forecasts` |
| `sensor.ai_tft_price_forecast_high` | TFT standalone q90 | 144 | 30-min | 72h | `forecasts` |
| `sensor.ai_tft_price_forecast_low` | TFT standalone q10 | 144 | 30-min | 72h | `forecasts` |

**Amber-shaped compatibility sensors** (for legacy EMHASS template compatibility):

| Entity | Description | Items | Attribute format |
|---|---|---|---|
| `sensor.ai_combined_general_price_forecast` | Hybrid AI forecast in amber2mqtt `Forecasts` format | 870 | `Forecasts` (capital F); items have `advanced_price_predicted/high/low`, `per_kwh`, `start_time`, `end_time` |
| `sensor.ai_combined_feed_in_price_forecast` | Same, feed-in (negative values) | 870 | `Forecasts` |

**Canonical HAEO-style sensors** (new, for source-selector switch):

| Entity | Description | Items | Resolution | Horizon | Attribute format |
|---|---|---|---|---|---|
| `sensor.ai_mpc_import_price_forecast` | Hybrid MPC import price (positive $/kWh) | 168 | 5-min | 14h | `forecast` (lowercase); items have `datetime` (UTC), `native_value` |
| `sensor.ai_mpc_export_price_forecast` | MPC export revenue (positive = revenue; negative when export costs money at negative spot) | 168 | 5-min | 14h | `forecast` |
| `sensor.ai_dh_import_price_forecast` | TFT DH import price (positive $/kWh) | 144 | 30-min | 72h | `forecast` |
| `sensor.ai_dh_export_price_forecast` | TFT DH export revenue (positive = revenue) | 144 | 30-min | 72h | `forecast` |

### Load forecasts

| Entity | Description | Items | Resolution |
|---|---|---|---|
| `sensor.ai_load_forecast` | Household load q50 (W) | 144 | 30-min |
| `sensor.ai_load_forecast_high` | Load q75 (W) | 144 | 30-min |
| `sensor.ai_load_forecast_p75` | Load p75 variant (W) | 144 | 30-min |
| `sensor.ai_tft_load_forecast` | TFT load model q50 (W) | 144 | 30-min |
| `sensor.ai_tft_load_forecast_high` | TFT load q90 (W) | 144 | 30-min |
| `sensor.ai_tft_load_forecast_low` | TFT load q10 (W) | 144 | 30-min |

### Weather forecasts

| Entity | Description | Items | Resolution |
|---|---|---|---|
| `sensor.ai_temperature_forecast` | Temperature °C | 144 | 30-min |
| `sensor.ai_humidity_forecast` | Relative humidity % | 144 | 30-min |
| `sensor.ai_wind_forecast` | Wind speed km/h | 144 | 30-min |

---

## HA template sensors (package-emhass.yaml)

Derived sensors computed by HA template engine. Recalculate on state change.

| Entity | Description | Depends on |
|---|---|---|
| `sensor.amber_effective_general_price` | Two-knob blended import price — blends `advanced_price_predicted/high/low` toward buy weight, then takes max of blended vs `per_kwh`. Used as spot reference for automations. | `sensor.amber_5min_current_general_price`, `sensor.amber_5min_forecasts_general_price`, `input_number.emhass_weight_buy_forecast` |
| `sensor.amber_effective_feed_in_price` | Two-knob blended export price — same blend logic, then applies SAPN free-export tier (+$0.01/kWh in 10am–4pm window while allowance > 0) | `sensor.amber_5min_current_feed_in_price`, `sensor.amber_5min_forecasts_feed_in_price`, `input_number.emhass_weight_sell_forecast`, `input_number.sapn_free_exports` |
| `sensor.emhass_selected_mpc_price_source` | Active MPC source name (`amber` / `ai_shadow`) + entity references | `input_select.emhass_mpc_price_source` |
| `sensor.emhass_selected_dh_price_source` | Active DH source name | `input_select.emhass_dh_price_source` |
| `sensor.ai_mpc_price_forecast_status` | `ready` / `not_ready` + import/export counts and first values for MPC canonical sensors | `sensor.ai_mpc_import/export_price_forecast` |
| `sensor.ai_dh_price_forecast_status` | Same for DH canonical sensors | `sensor.ai_dh_import/export_price_forecast` |

---

## EMHASS output sensors

Published by EMHASS after each optimisation run. Prefixed `dh_` (day-ahead, 30-min/72h)
and `mpc_` (MPC, 5-min/14h). Values are current-interval scalars; forecast array is in
the `forecasts` attribute.

| Entity group | Description |
|---|---|
| `sensor.dh_p_batt_forecast` | Planned battery power (W); negative = charging |
| `sensor.dh_p_grid_forecast` | Planned grid import (W) |
| `sensor.dh_p_pv_forecast` | Planned PV generation (W) |
| `sensor.dh_p_load_forecast` | Planned load (W) |
| `sensor.dh_p_hybrid_inverter` | Planned inverter AC output (W) |
| `sensor.dh_p_pv_curtailment` | Planned PV curtailment (W) |
| `sensor.dh_soc_batt_forecast` | Planned battery SoC (%) |
| `sensor.dh_unit_load_cost` | Current-interval import price used by EMHASS ($/kWh) |
| `sensor.dh_unit_prod_price` | Current-interval export price used by EMHASS ($/kWh) |
| `sensor.dh_total_cost_fun_value` | Total optimisation objective value |
| `sensor.dh_optim_status` | EMHASS solver status (`Optimal`, `Infeasible`, …) |
| `sensor.mpc_*` | Same set, 5-min resolution, 14h horizon |

---

## Input helpers

### Source selectors

| Entity | Options | Default | Purpose |
|---|---|---|---|
| `input_select.emhass_mpc_price_source` | `amber`, `ai_shadow` | `amber` | Switch MPC REST payload between Amber 5-min and AI canonical sensors |
| `input_select.emhass_dh_price_source` | `amber_lgbm_extrapolated`, `ai_shadow` | `amber_lgbm_extrapolated` | Switch DH REST payload between Amber 30-min LightGBM and AI canonical sensors |

### Tuning knobs

| Entity | Description | Range |
|---|---|---|
| `input_number.emhass_weight_buy_forecast` | Risk aversion for import price blending (positive = bias toward `advanced_price_high`) | −1 to +1 |
| `input_number.emhass_weight_sell_forecast` | Risk aversion for export price blending | −1 to +1 |
| `input_number.emhass_weight_pv_forecast` | Risk aversion for PV forecast blending | −1 to +1 |
| `input_number.emhass_weight_battery_discharge` | Battery discharge cost penalty in EMHASS objective | 0.0+ |
| `input_number.emhass_weight_forecast_probability` | EMHASS stochastic optimisation probability weight | 0–1 |
| `input_number.emhass_day_ahead_forecast_probability_weight` | DH probability weight | 0–1 |
| `input_number.emhass_target_soc_offset` | Offset added to EMHASS-derived target SoC | — |
| `input_number.emhass_dayahead_soc_init` | Initial SoC sent to DH EMHASS solve | % |

### Battery state tracking

| Entity | Description |
|---|---|
| `input_number.battery_soc_30_minute` | Battery SoC (%) updated every 30 min (DH initial SoC source) |
| `input_number.battery_soc_5_minute` | Battery SoC (%) updated every 5 min (MPC initial SoC source) |
| `input_number.battery_soc_min_buffer` | Minimum SoC buffer (%) |
| `input_number.battery_soc_min_export` | Minimum SoC before export allowed (%) |
| `input_number.battery_soc_min_target` | Minimum overnight SoC target (%) |

### Other

| Entity | Description |
|---|---|
| `input_number.sapn_free_exports` | Remaining SAPN free-export daily allowance (kWh); decremented by automation |
| `input_text.emhass_battery_action` | Current EMHASS-recommended battery action string |

---

## Inverter / battery (sigenergy2mqtt)

| Entity | Description |
|---|---|
| `sensor.sigen_0_plant_battery_soc` | Live battery SoC (%) from Sigenergy via MQTT |
| `sensor.sigen_0_inverter_1_battery_soc` | Per-inverter battery SoC (%) |
| `sensor.sigenergy2mqtt_modbus_*_2` | MQTT bridge diagnostics (cache hits, read latency, errors). The `_2` suffix is the active instance; bare names are a stale/offline instance. |

---

## Solcast (PV forecast integration)

| Entity | Description |
|---|---|
| `sensor.solcast_pv_forecast_forecast_today` | Today's PV generation forecast (kWh); `detailedForecast` attribute with 30-min `{period_start, pv_estimate, pv_estimate10, pv_estimate90}` items |
| `sensor.solcast_pv_forecast_forecast_tomorrow` | Tomorrow's PV forecast |
| `sensor.solcast_pv_forecast_forecast_day_3/4/5` | Days 3–5 PV forecasts |
| `sensor.solcast_pv_forecast_power_now/in_30_minutes/in_1_hour` | Near-term power (W) |
| `sensor.solcast_pv_forecast_forecast_remaining_today` | Remaining generation today (kWh) |
| `sensor.solcast_pv_forecast_peak_forecast_today/tomorrow` | Peak W forecast |
| `sensor.solcast_pv_forecast_peak_time_today/tomorrow` | Time of peak |
| `sensor.solcast_pv_forecast_api_limit/used/last_polled` | API quota tracking |
| `sensor.solcast_pv_forecast_dampening` | Whether dampening is active |
| `binary_sensor.solcast_suppress_auto_dampening` | Manual dampening suppression |
| `select.solcast_pv_forecast_use_forecast_field` | Which field to use (`estimate` / `estimate10` / `estimate90`) |

---

## Notes

**Feed-in sign convention:** amber2mqtt, Amber Express, and `sensor.ai_combined_feed_in_price_forecast` all use **negative values for export revenue** (Amber convention). The new canonical HAEO sensors (`sensor.ai_dh_export_price_forecast`, `sensor.ai_mpc_export_price_forecast`) use **positive values for export revenue** — a negative value means the export costs money (e.g. during negative spot price periods).

**amber2mqtt vs Amber Express coexistence:** Both are currently running and querying the Amber API simultaneously. amber2mqtt is the active production source; Amber Express entities are not consumed by any code or automation in this repo. See `docs/ideas.md` for migration notes.
