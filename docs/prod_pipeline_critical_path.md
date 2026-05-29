# Production pipeline — critical path

**Scope**: fresh input data → battery dispatch command via EMHASS.

**Out of scope** (deliberately): the shadow forecast stack (PD-direct,
canonical AI MPC/DH bundle, AEMO stitched, TFT-load, p5min tactical),
the AEMO PREDISPATCH/PD7Day/SEVENDAYOUTLOOK ingest paths (they feed the
shadow stack and the eval framework only), the daily housekeeping
(`update-tariffs`, `update-adjusters`, `backfill-actuals`), and weekly
model training. Those run but are not load-bearing for dispatch *today*.

Pipeline written as it stands 2026-05-29, after the EMHASS script-wrapper
refactor (DH self-correction + persisted soc_init helpers) landed.

## MPC tier — 14h horizon

EMHASS MPC re-optimises continuously on a 14-hour horizon using **raw
Amber forecasts directly**. No AI inference in this path. MPC's horizon
is fully covered by Amber's 16.5h APF, so LGBM extrapolation isn't
needed.

1. Amber publishes a new 5-min forecast (~every 5 min, NEM dispatch
   cadence).
2. HA's Amber integration updates
   `sensor.amber_5min_forecasts_general_price` and `..._feed_in_price`.
3. HA automation **"EMHASS MPC optim on 5min price update"** fires on
   either entity changing, or on a time-pattern `:25` fallback between
   price updates so the plan keeps tracking SoC/load drift.
4. Automation calls `script.emhass_mpc`. The script: interpolates the
   prior DH plan at three timestamps (5-min boundary / utcnow / +14h),
   computes deviation against the live derived SoC, derives
   `soc_init_pct` and `soc_final_pct`, writes the chosen soc_init to
   `input_number.mpc_last_soc_init` (diagnostic), then fires
   `rest_command.emhass_mpc` with the values as parameters → EMHASS
   computes battery plan → `rest_command.emhass_publish_data_mpc`
   publishes results. See `docs/production_soc_policy.md` for the formula.
5. Automation triggers
   `automation.battery_ems_control_based_on_emhass_forecasts` → Sigenergy
   script writes battery setpoints.

**Latency from Amber publish → battery setpoint**: a few seconds.

## DH tier — 72h horizon

EMHASS day-ahead optimises 72 hours ahead and needs both:
- A 72h **price curve** — Amber APF only covers 16.5h, so the tail is
  LGBM-extrapolated.
- A 30-min **load forecast** — LGBM quantile model.

### Price feed (event-driven, sub-10s)

1. Amber publishes a new APF.
2. HA's Amber integration updates
   `sensor.amber_billing_interval_forecasts_general_price`.
3. `ai-energy-listener.service` (long-lived asyncio daemon, HA WebSocket
   subscriber) receives the `state_changed` event, debounces 1s, spawns
   `forecast.py predict-price --dynamic-handoff --publish-hass`.
4. `forecast.py`: startup + data fetch (Solcast, weather, AEMO 5MIN
   future-covariate API, InfluxDB price history, Amber) → LGBM quantile
   inference → tariff application.
5. Publishes `sensor.ai_price_forecast`, `..._low`, `..._high` (the
   "Legacy LGBM publish: …s (prod-critical fast path)" line in the
   journal marks this point).

**Latency from Amber publish → `sensor.ai_price_forecast` updated**:
~7-8s end-to-end. ~6-7s of that is `forecast.py` startup + multi-source
data fetch; LGBM inference itself plus the HA publish is <1s.

The shadow stack (PD-direct + canonical AI bundle + AEMO stitched +
TFT-load, ~30s) runs after the legacy publish and does not block any
prod sensor.

### Load feed (timer-driven, 30-min cadence)

1. `ai-energy-predict.timer` fires twice an hour at `:01` and `:31`.
   Offset gives InfluxDB's `cq_raw_to_5m` → `cq_5m_to_30m` continuous
   queries time to flush the just-closed 30-min bucket.
2. `forecast.py predict-load --publish-hass --publish-covariates` runs:
   data fetch + LGBM quantile inference.
3. Publishes `sensor.ai_load_forecast`, `..._high`, `..._p75`.

### Optimisation

4. HA automation **"EMHASS dayahead optim on AI forecast update"** fires
   on state change of either `sensor.ai_price_forecast` (~5×/hr) or
   `sensor.ai_load_forecast` (2×/hr).
5. Calls `script.emhass_dayahead_optim`. The script: recomputes
   `emhass_target_soc_offset` from the prior DH plan's +48h..+72h peak SoC
   (squashed-in feedback loop, formerly a separate HA automation),
   interpolates the prior DH plan at `utcnow()` (anchored on
   `dh_last_soc_init`), computes signed deviation against the live derived
   SoC, derives `soc_init_pct = anchor + deviation` and
   `soc_final_pct = anchor + emhass_target_soc_offset + deviation`, writes
   both updated helper values back to `input_number.dh_last_soc_init` and
   `input_number.emhass_target_soc_offset`, then fires
   `rest_command.emhass_dayahead_optim` with the values as parameters →
   EMHASS computes 72h plan → `rest_command.emhass_publish_data_dh`
   publishes it. See `docs/production_soc_policy.md` for the formulas.

**Latency from forecast publish → DH plan ready**: a couple of seconds
inside EMHASS plus the 1s defensive action delay.

**Note:** the automation also still writes
`input_number.sigen_plant_battery_state_of_charge → input_number.emhass_dayahead_soc_init`
before calling the script. The write is now a no-op (the script reads the live
SoC sensor directly and the helper is no longer consumed) but is left in place
for diagnostic continuity.

## End-to-end summary

| Tier | Horizon | Refresh trigger | Forecast freshness (post Amber publish) |
|---|---|---|---|
| MPC | 14h | Raw Amber 5-min entity state_changed + every-min `:25` fallback | seconds |
| DH (price) | 72h | `sensor.ai_price_forecast` state_changed | ~7-8s |
| DH (load) | 72h | `sensor.ai_load_forecast` state_changed | up to 30 min (timer-driven) |

## What's *not* in critical path today

Worth listing because it's easy to assume otherwise:

- **The AI MPC/DH bundle sensors** (`sensor.ai_mpc_*_price_forecast`,
  `sensor.ai_dh_*_price_forecast`, `sensor.ai_pd_direct_*`,
  `sensor.ai_aemo_price_forecast`). Published by `predict-price`'s shadow
  stack. Visible in HA, but EMHASS is configured to read raw Amber for
  MPC and `sensor.ai_price_forecast` (legacy LGBM) for DH — not these.
- **TFT-load shadow**. Generated by `predict-load`'s shadow tail; used
  only by the offline eval framework.
- **AEMO ingest timers** (`ai-energy-p5min`, `ai-energy-predispatch`,
  `ai-energy-pd7day`, `ai-energy-sevendayoutlook`). Feed
  `aemo_*_sa1.parquet` files in `data/` plus PD-direct (shadow). Not
  consumed by the prod LGBM-extrapolation path.
- **`forecast.py publish-tactical`** (chained to p5min ingest) and
  **`publish-pd-direct`** (chained to predispatch ingest). Both refresh
  shadow sensors only.
