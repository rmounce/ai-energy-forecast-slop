# HWC modelling — handover

Handover for an implementer continuing the **heat-pump hot water (HWC)** scheduling work
(Aquatech RAPID X6), via EMHASS or a replacement. Read alongside, in order:
`docs/hwc_emhass.md` (design/spec + PAUSED banner), `docs/hwc_thermal_characterisation.md`
(measured physics/COP), `docs/emhass_shared_state_race.md` (the EMHASS bug we hit + fix
status). Project memory: `project_heat_pump_hot_water` (+ `feedback_secrets_practice`,
`feedback_influxdb_cli`).

## Goal

Schedule the HWC unit to minimise cost against the import-price + weather forecasts,
**separately from the home battery**. v1 = *modelling/plan only* (publish a plan, no
actuation). Actuation (via Local Tuya) and a resistive-element "dump load" on negative
prices are later phases.

## TL;DR state (2026-06-01)

A working planner exists and produces sensible plans, but is **paused** while we (a) make
the COP model accurate and (b) decide the modelling engine. The EMHASS shared-state race
that originally forced the pause is **fixed and deployed**, so the pause is now about
*accuracy*, not safety.

| Thing | State |
|---|---|
| `hwc_planner.py` (planner) | committed `d9f171f`, **works**, timer **disabled** |
| `ai-energy-hwc.timer` (user systemd unit) | **disabled/stopped** (was briefly enabled) |
| EMHASS metadata race | **fixed + deployed** (`emhass:metadata-race-20260601`); see race doc |
| COP characterisation | done & committed `5c1ab55` (measured COP ≈ 2.4) |
| Engine decision (EMHASS vs custom) | **OPEN** — direction agreed: *recalibrate now, decide later* |
| Recalibration (`carnot_efficiency` 0.45→~0.38) | **not yet applied** — next concrete step |

## What's committed

- `d9f171f` — HWC planner v1 (`hwc_planner.py`), `config.json` `hwc` block, systemd units,
  unit tests, ApexCharts card, spec doc.
- `c61520e` — paused the planner; documented the EMHASS shared-state race.
- `5c1ab55` — `hwc_cop_analysis.py` (reusable COP sweep), `data/hwc_cop_cycles.csv`,
  `docs/hwc_thermal_characterisation.md`.

Pre-existing uncommitted changes in the working tree (`docs/ha_entity_inventory.md`,
`docs/prod_pipeline_critical_path.md`, `docs/production_soc_policy.md`,
`eval/results/load_overnight_eval.json`, `hass/packages/emhass.yaml`) are **not ours** —
leave them alone.

## How it works (architecture)

- `hwc_planner.py` runs as a Python script (`systemd/ai-energy-hwc.{service,timer}`, every
  30 min), *not* a Jinja `rest_command` (deliberate — wet-bulb + clock-aligned arrays are
  fragile in Jinja). It: reads tank temp + the import-price forecast
  (`sensor.ai_dh_import_price_forecast`) + BOM weather from HA; computes **wet-bulb** (Stull)
  and clock-aligned `draw_off`/temperature arrays; POSTs `naive-mpc-optim` to the shared
  EMHASS instance with `set_use_battery:false`/`set_use_pv:false` and one `thermal_battery`
  deferrable load; lets EMHASS publish the plan.
- Loads config via `config_utils.load_config()` (merges untracked `config.secrets.json`).
- Pure helpers are unit-tested: `tests/unit/test_hwc_planner.py` (`.venv/bin/python -m pytest`).

## Measured findings that MUST shape the model (the important part)

From `docs/hwc_thermal_characterisation.md` (telemetry analysis; reproduce with
`hwc_cop_analysis.py`):

1. **Stratified charging.** The control probe (`sensor.heat_pump_temperature`) sits flat for
   ~half the reheat while the condenser temp (`sensor.aquatech_exhaust_temperature`) climbs
   50→82 °C. COP is governed by tank **state-of-charge / condensing temperature, NOT the
   probe reading**. A single-node tank model can't represent this.
2. **Fixed-speed compressor + EEV** (confirmed by smooth power rise, no inverter step).
   So for scheduling, the reheat is effectively a **~2 h block** of ~700 W; the key quantity
   is *electrical energy + duration as a function of (start temp, target, wet-bulb)* — a
   measurable low-dimensional curve, not an intra-cycle trajectory the unit can't follow.
3. **Real COP ≈ 2.4–3.0 to 60 °C** (datasheet 4.68 is to 55 °C at rating conditions — far
   too optimistic). The **55→60 °C legionella tail alone is COP ~1.75** — the expensive part.
   ⇒ heating to 60 °C should be infrequent (legionella only); routine target lower.
4. **No onboard power meter.** Power is proxied from `sensor.remaining_power_load` − baseline,
   valid only on clean windows. **Getting a real circuit meter is the #1 accuracy lever.**

## The open decision: modelling engine

Stock EMHASS `thermal_battery` is a **single-node tank with fixed-supply Carnot COP** — it
can model neither the stratification nor the SoC-dependent condensing temp. Agreed direction:
**recalibrate now, decide the engine later** once we have more calibration data. Options when
deciding:

- **(A) Recalibrate + keep EMHASS** — set `carnot_efficiency` ≈ 0.38 (from 0.45) and an
  effective `supply_temperature` > 60; accept a cycle-average COP. Least work; OK if a single
  average COP is good enough (plausible, since the unit runs as a block).
- **(B) Purpose-built block optimiser** — model the reheat as a fixed-speed block with measured
  energy/duration vs conditions; find cheapest placement subject to availability + legionella.
  Most accurate for this unit; small bespoke MILP/enumeration; drops EMHASS for HWC.
- **(C) Enhance EMHASS COP** — add SoC/condensing-temp-dependent supply temperature in EMHASS
  (the owner already patches EMHASS — see race doc). Keeps one framework; more EMHASS surgery;
  still single-node.

The calibration data (power meter + accumulating clean cycles via `hwc_cop_analysis.py`) is
the engine-independent long pole — gather it regardless.

## Concrete next steps (suggested order)

1. **Recalibrate** `config.json` → `hwc.thermal.carnot_efficiency` 0.45 → ~0.38 (and revisit
   `supply_temperature`). Cheap; removes the optimism. (Agreed; not yet done.)
2. **Lower the routine target** below 60 °C (e.g. 55) with a separate periodic 60 °C legionella
   reheat — the COP data shows the 55→60 tail is dear. Reconsider `desired/min_temperatures`
   and the once-daily assumption (Aquatech suggest a main 10:00–18:00 window + a morning boost).
3. **Keep gathering clean COP cycles** (`python hwc_cop_analysis.py`); pursue a dedicated power
   meter; build COP(target, wet-bulb, start) — then make the engine decision (A/B/C) on evidence.
4. **Fix the `wet_bulb` column** in `hwc_cop_analysis.py` (the `humidity_adelaide` join — wrong
   RP/tag, currently empty).
5. **Re-enable** only when ready: confirm EMHASS runs the fixed image (it does:
   `emhass:metadata-race-20260601`), then `systemctl --user enable --now ai-energy-hwc.timer`.
   The planner still uses `entity_save` against the shared store — now safe because of the
   deployed fix; do NOT run it against a stock image that predates the fix.

## Gotchas / operational notes

- **Secrets:** never put secrets — *including internal hostnames* — in tracked `config.json`.
  The EMHASS URL lives in untracked `config.secrets.json` (`hwc.emhass_base_url`), merged by
  `config_utils.load_config`; placeholder in `config.secrets.json.example`. (We leaked the
  internal domain once — don't repeat.)
- **InfluxDB:** no sudo; `docker exec influxdb influx -username user -password <see config.secrets.json>
  -database hass -execute "..."`, or use `InfluxDBClient` via `load_config()`. Tank temp =
  measurement `sensor__temperature` entity `heat_pump_temperature`; binary sensors log on
  change → query with `fill(previous)`, not `fill(0)`. Useful HWC channels: `aquatech_exhaust_temperature`
  (condenser), `aquatech_compressor`, `aquatech_defrost`, `aquatech_four_way_valve`,
  `aquatech_element`, `aquatech_temperature` (ambient), `remaining_power_load` (power proxy).
- **EMHASS double-prefix:** EMHASS prepends `publish_prefix` to your `custom_*_id`, so the
  config uses *bare* names (`sensor.predicted_temp`) → published as `sensor.hwc_predicted_temp`.
  Forecast arrays live in entity *attributes* (values are strings).
- **Battery isolation:** the HWC optim must keep `set_use_battery:false`/`set_use_pv:false`
  (runtime-overridable via EMHASS `associations.csv`). The battery (DH + per-minute MPC) shares
  the EMHASS instance; the now-fixed race was between concurrent `entity_save` publishes.
- **Fan-speed regime:** fan reduced for quiet mode (F30 25→10, F35 55→30). COP calibration is
  specific to this setting — confirm the change date before mixing cycles across regimes.
- **Sandbox:** `docker exec` and direct InfluxDB/EMHASS network calls need the command sandbox
  disabled (filesystem/network restrictions); plain repo edits/tests do not.
