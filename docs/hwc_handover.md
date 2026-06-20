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

## TL;DR state (2026-06-20)

A working daemonised planner is active. The default engine is a direct fixed-speed block
planner; EMHASS thermal-battery mode remains as a fallback/comparison path. Dedicated
Athom metering is live for the HWC compressor circuit.

| Thing | State |
|---|---|
| `services/hwc_daemon.py` | event-driven planner/executor daemon, **enabled/active** |
| old `ai-energy-hwc.{service,timer}` | **obsolete/removed**; daemon owns HWC planning now |
| EMHASS metadata race | **fixed + deployed** (`emhass:metadata-race-20260601`); see race doc |
| COP characterisation | updated through `2026-06-18`; 12/14 clean cycles, five recent Athom-metered cycles |
| Engine decision (EMHASS vs custom) | custom block planner is now default; EMHASS kept as fallback |
| Recalibration (`carnot_efficiency` 0.45→0.38) | **applied** (`6af7f5f`); `supply_temperature` still needs review |
| COP analyzer `wet_bulb` column | **fixed** (`6af7f5f`); regenerate `data/hwc_cop_cycles.csv` when needed |
| Execution layer | integrated in `services/hwc_daemon.py`; old executor timer removed |
| EMHASS load input | LGBM load excludes HWC/dump loads; HA EMHASS payload adds planned HWC compressor power back in |
| Running compressor policy | planner scores stop/continue candidates using `block_planner.stop_cost_aud`; no unconditional run lock |

## What's committed

- `d9f171f` — HWC planner v1 (`hwc_planner.py`), `config.json` `hwc` block, systemd units,
  unit tests, ApexCharts card, spec doc.
- `c61520e` — paused the planner; documented the EMHASS shared-state race.
- `5c1ab55` — `hwc_cop_analysis.py` (reusable COP sweep), `data/hwc_cop_cycles.csv`,
  `docs/hwc_thermal_characterisation.md`.

Pre-existing uncommitted changes in the working tree (`docs/emhass_shared_state_race.md`,
`docs/ha_entity_inventory.md`,
`docs/prod_pipeline_critical_path.md`, `docs/production_soc_policy.md`,
`eval/results/load_overnight_eval.json`, `hass/packages/emhass.yaml`) are **not ours** —
leave them alone.

## How it works (architecture)

- `hwc_planner.py` runs as a Python script under the event-driven HWC daemon, *not* a
  Jinja `rest_command`. It reads tank temp, EMHASS-prepared import-price series
  (`sensor.mpc_unit_load_cost` + `sensor.dh_unit_load_cost`), and BOM weather from HA;
  builds clock-aligned `draw_off`/temperature arrays; creates a 48h fixed-speed block plan;
  and publishes the plan sensors directly to HA.
- Published HWC planned power is modelled compressor watts, not a flat nameplate value.
  `config.json` uses the 2026-06-19 Athom fit: `740 W @ tank 50 °C / wet-bulb 12.5 °C`,
  `+15 W/°C` tank, `+1.5 W/°C` wet-bulb, clamped `650–930 W`.
- Battery EMHASS still receives the household load forecast through HA Jinja. That forecast
  is now base load with deferrable loads excluded; `hass/packages/emhass.yaml` adds
  `sensor.hwc_power_plan` back into the day-ahead `load_power_forecast` at payload time.
- EMHASS fallback mode is still available with `hwc.planner: "emhass"`, but the block planner
  is the default because it matches the unit's fixed-speed behavior and avoids fragmented
  compressor starts.
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
4. **Dedicated circuit meter added.** Current power input should come from raw Athom
   channel 2 (`sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2`).
   Older calibration history used
   `sensor.remaining_power_load` − baseline, valid only on clean windows.

## Modelling engine decision

Stock EMHASS `thermal_battery` is a **single-node tank with fixed-supply Carnot COP** — it
can model neither the stratification nor the SoC-dependent condensing temp, and live plans
showed a too-short 48-point output, no terminal inventory contract, and fragmented starts.
The default is now option B:

- **(A) Recalibrate + keep EMHASS** — set `carnot_efficiency` ≈ 0.38 (from 0.45) and an
  effective `supply_temperature` > 60; accept a cycle-average COP. Least work; OK if a single
  average COP is good enough (plausible, since the unit runs as a block).
- **(B) Purpose-built block optimiser** — model the reheat as a fixed-speed block with measured
  energy/duration vs conditions; find cheapest placement subject to availability + legionella.
  Most accurate for this unit; small bespoke enumeration; default for HWC.
- **(C) Enhance EMHASS COP** — add SoC/condensing-temp-dependent supply temperature in EMHASS
  (the owner already patches EMHASS — see race doc). Keeps one framework; more EMHASS surgery;
  still single-node.

The calibration data (power meter + accumulating clean cycles via `hwc_cop_analysis.py`) is
the engine-independent long pole — gather it regardless.

## Concrete next steps (suggested order)

1. **Review `supply_temperature`** now that `hwc.thermal.carnot_efficiency` is 0.38. The
   cheap optimism fix is applied, but the effective supply temperature may need to be >60 °C
   to approximate the measured condensing-temperature tail.
2. **Revisit the compressor power curve** after more varied Athom-metered cycles, especially
   colder wet-bulb days and deeper cold starts. Current fields:
   `compressor_power_reference_w`, `compressor_power_wet_bulb_slope_w_per_c`,
   `compressor_power_tank_slope_w_per_c`, min/max clamps.
3. **Lower the routine target** below 60 °C (e.g. 55) with a separate periodic 60 °C legionella
   reheat — the COP data shows the 55→60 tail is dear. Reconsider `desired/min_temperatures`
   and the once-daily assumption (Aquatech suggest a main 10:00–18:00 window + a morning boost).
4. **Keep gathering clean COP cycles** (`python hwc_cop_analysis.py`); pursue a dedicated power
   meter; build COP(target, wet-bulb, start) — then make the engine decision (A/B/C) on evidence.
5. **Regenerate `data/hwc_cop_cycles.csv`** with `hwc_cop_analysis.py` once InfluxDB access is
   available; the humidity query now reads aggregate `rp_30m.humidity_adelaide` without the
   invalid `entity_id` filter.
6. **Monitor the daemon**: `systemctl --user status ai-energy-hwc-daemon.service`
   and `journalctl --user -u ai-energy-hwc-daemon.service`. The old modelling timer and
   executor timer are obsolete; do not re-enable them.
7. **Execution dry runs**: `python hwc_executor.py --dry-run` still exercises the standalone
   executor helper, but live execution is handled by `services/hwc_daemon.py`.

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
  `aquatech_element`, `aquatech_temperature` (ambient),
  `athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2` (Athom channel 2
  power), `remaining_power_load` (older residual power proxy).
- **EMHASS double-prefix:** EMHASS prepends `publish_prefix` to your `custom_*_id`, so the
  config uses *bare* names (`sensor.predicted_temp`) → published as `sensor.hwc_predicted_temp`.
  Forecast arrays live in entity *attributes* (values are strings).
- **Battery isolation:** the HWC optim must keep `set_use_battery:false`/`set_use_pv:false`
  (runtime-overridable via EMHASS `associations.csv`). The battery (DH + per-minute MPC) shares
  the EMHASS instance; the now-fixed race was between concurrent `entity_save` publishes.
- **Running compressor stop cost:** if `binary_sensor.aquatech_compressor` is already `on`,
  the block planner now compares stop/continue candidates. Objective =
  energy cost + `block_planner.stop_cost_aud` per compressor stop. This allows interrupting
  a current run only when a later valid plan beats the configured stop cost.
- **HWC/DH coherence:** battery DH snapshots `sensor.hwc_power_plan` immediately before the
  DH solve (`sensor.emhass_dh_hwc_power_plan_snapshot`) and uses that snapshot when adding
  planned HWC compressor power into DH load. MPC uses the same snapshot when subtracting HWC
  back out of DH load and adding the native 5-minute HWC shape into the MPC load forecast.
  This deliberately favours DH/MPC coherence and a stable 5-minute HWC block over using the
  freshest HWC plan between DH solves. If the snapshot is missing after HA restart, templates
  fall back to the live HWC plan until the next DH run refreshes the snapshot.
- **Fan-speed regime:** fan reduced for quiet mode (F30 25→10, F35 55→30). COP calibration is
  specific to this setting — confirm the change date before mixing cycles across regimes.
- **Sandbox:** `docker exec` and direct InfluxDB/EMHASS network calls need the command sandbox
  disabled (filesystem/network restrictions); plain repo edits/tests do not.
