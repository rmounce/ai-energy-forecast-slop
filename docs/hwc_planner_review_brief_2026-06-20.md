# HWC planner independent review brief - 2026-06-20

Purpose: brief an independent reviewer on the current heat-pump hot water (HWC) planner,
including goals, implementation, fixed bugs, and known weak spots.

## Compressed State

- Unit: Aquatech RAPID X6 Gen2 heat-pump hot water.
- Control: Home Assistant `water_heater.aquatech` via Local Tuya.
- Compressor sensor: `binary_sensor.aquatech_compressor`.
- Dedicated power meter: Athom channel 2,
  `sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2`.
- Live service: `systemctl --user status ai-energy-hwc-daemon.service`.
- Main code:
  - `hwc_planner.py`: fixed-speed block planner, HA publishing.
  - `hwc_executor.py`: executes published plan.
  - `services/hwc_daemon.py`: event-driven replanning/execution loop.
  - `config.json`: active HWC model/control config.
  - `hass/packages/emhass.yaml`: battery EMHASS integration.
- Current temporary experiment:
  - `hwc.thermal.min_temp = 50` instead of normal `45`.
  - `hwc.block_planner.stop_cost_aud = 0.02` instead of normal `0.05`.
  - Goal: force top-up opportunities and observe Aquatech start/stop behaviour.
  - Restore after review: `min_temp = 45`, `stop_cost_aud = 0.05`.

## Goals

- Schedule HWC compressor runs against import-price and weather forecasts.
- Keep HWC planning separate from home battery optimisation.
- Feed planned HWC compressor power back into EMHASS load forecasts.
- Preserve native 5-minute HWC power shape for MPC.
- Avoid fragmented compressor cycling unless price benefit exceeds configured transition cost.
- Reach 60 C for legionella/target satisfaction when required.
- Eventually run routine lower-temperature top-ups, with separate 60 C policy.

## Non-Goals

- Do not model HWC as a battery in the home battery optimiser.
- Do not let LGBM household load forecasting learn HWC schedule/load.
- Do not use dump-load behaviour as a scheduled deferrable load.
- Do not assume EMHASS thermal-battery mode is production-quality for this unit.

## Current Implementation

### Planner

- `hwc_planner.py` builds a 48h, 5-minute fixed-speed compressor plan.
- Inputs:
  - tank temp from HA.
  - import-price grids prepared for EMHASS.
  - BOM weather converted to wet-bulb.
  - configured morning draw-off profile.
  - compressor state when planning starts.
- Output:
  - `sensor.hwc_power_plan` with `deferrables_schedule`.
  - `sensor.hwc_predicted_temp` with predicted tank/probe temperatures.
  - `sensor.hwc_unit_load_cost`.
  - `sensor.hwc_wet_bulb_forecast`.
- Power model:
  - empirical compressor watts as function of tank temp and wet-bulb.
  - current config: `740 W @ tank 50 C / wet-bulb 12.5 C`,
    `+15 W/C` tank, `+1.5 W/C` wet-bulb, clamped `650-930 W`.
- Temperature model:
  - simple single-state/probe-style model.
  - known mismatch: real tank is stratified; probe can sit flat while condenser temp rises.

### Planner Stages

- `_choose_daily_main_blocks(...)`
  - choose main daytime block to satisfy daily target, normally 60 C unless already satisfied.
- `_repair_min_temperature(...)`
  - add top-up heat before forecast temp drops below `thermal.min_temp`.
- `_repair_terminal_temperature(...)`
  - add late-horizon heat to satisfy terminal target/inventory heuristic.
- `_refresh_planned_power(...)`
  - convert binary heat slots to modelled compressor watts.
- `build_block_plan(...)`
  - chooses between seed schedules.
  - scores final plan by energy cost plus stop cost.

### Stop-Cost Objective

- Config: `hwc.block_planner.stop_cost_aud`.
- Objective: `energy_cost + planned_stop_count * stop_cost_aud`.
- `planned_stop_count` counts transitions from compressor-on to compressor-off.
- If compressor is already running, current compressor state is part of the initial state.
- Intent:
  - allow interrupting a run.
  - discourage short cycles.
  - avoid hard locks/min-runtime heuristics where possible.

### Executor

- Inside planned block:
  - set `operation_mode = heat_pump`.
  - set target temperature from block-end predicted temp, clamped by config.
- Outside planned block:
  - call `water_heater.turn_off`.
- Assumption under observation:
  - `off -> heat_pump` starts promptly.
  - `turn_off` stops promptly.
- Executor logs each HA command for later comparison with:
  - `binary_sensor.aquatech_compressor`.
  - Athom channel 2 power.

### EMHASS Integration

- LGBM base load excludes HWC and dump load.
- Day-ahead EMHASS:
  - snapshots `sensor.hwc_power_plan`.
  - adds 30-minute downsampled HWC compressor plan to load.
- MPC:
  - starts from DH load contract.
  - subtracts the same HWC snapshot at 30-minute resolution.
  - adds native 5-minute HWC plan shape.
  - keeps first two slots as live load.
  - rescales remaining load to conserve total DH energy.
- Snapshot coherence is preferred over freshest HWC plan between DH solves.

## Measured Behaviour

- Real COP to 60 C is much lower than datasheet rating.
- 55 -> 60 C tail is expensive, roughly COP ~1.75 in measured cycles.
- Fixed-speed compressor behaviour observed; no inverter-style modulation.
- Fan speed was reduced for quiet mode; calibration depends on this setting.
- Athom metering gives clean circuit power from 2026-06-18 onward.

## Fixed Bugs / Lessons

- **CT direction bug**
  - Symptom: Athom power sign was negative.
  - Cause: CT clamp installed backwards.
  - Fix: CT corrected physically; raw channel power now positive.

- **Unnecessary HA template sensors**
  - Symptom: proposed HWC/cooktop template sensors duplicated raw Athom entities.
  - Decision: use raw Athom channel entities directly.

- **DH/MPC HWC coherence bug**
  - Symptom: MPC could subtract one HWC plan and add another, causing load dips.
  - Fix: DH snapshots HWC plan; MPC uses same snapshot for subtraction and 5-minute addback.

- **MPC conservation bug**
  - Symptom: first two live-load MPC slots changed total forecast energy.
  - Fix: scale remaining MPC horizon so total energy matches DH contract.

- **HA snapshot attribute type bug**
  - Symptom: Jinja expected JSON string; HA native attrs could be lists.
  - Fix: templates handle both string and native list attributes.

- **DH HWC downsampling bug**
  - Symptom: positional chunking could misalign HWC power with DH timestamps.
  - Fix: timestamp-bucket downsampling.

- **Running compressor disappeared from plan**
  - Symptom: compressor was physically running while freshly published plan only showed a future block.
  - Cause: replanning ignored current compressor state.
  - First fix: hard-lock running compressor into plan.
  - Current state: replaced hard lock with stop-cost candidate scoring.

- **Stop cost not respected inside local repairs**
  - Symptom: top-up split into two short blocks even with `stop_cost_aud = 0.05`.
  - Cause: local repair functions scored candidates by energy cost only; stop cost applied only
    after a complete schedule was already built.
  - Fix: local repairs now use `_schedule_objective_delta(...)`.
  - Result: early split top-up merged after fix at `stop_cost_aud = 0.05`; still merged at `0.02`.

## Known Shortcomings

- **Heuristic-heavy planner**
  - Main block, min-temp repair, terminal repair are sequential greedy stages.
  - Earlier stages constrain later stages.
  - No guarantee of global optimality.

- **`min_temp` is not a strict global constraint**
  - In current temporary experiment, `min_temp = 50`, but forecast still reaches about 46 C
    near 2026-06-22 10:00.
  - Repair heuristic is not a formal constraint solver.

- **Terminal target is a heuristic**
  - `terminal_target = current` and reserve penalties compensate for finite-horizon artefacts.
  - This is not a principled value function.

- **No explicit block metadata**
  - Executor infers intent from watts/time.
  - Published plan does not yet carry `block_kind`, `target_temp_c`, `source`, or interruptability.
  - This makes reviews and dashboards harder.

- **Device semantics not fully proven**
  - Need confirm:
    - lower setpoint support below 55 C.
    - `off -> heat_pump` start latency.
    - `turn_off` stop latency.
    - mode differences: `heat_pump`, `eco`, `high_demand`, `performance`, `electric`.

- **Temperature model is too simple**
  - Unit/tank is stratified.
  - Single probe/state cannot represent condenser temperature / SoC tail.
  - Current model is good enough for block timing but not physically complete.

- **Stop cost is blunt**
  - Stop cost discourages cycling, but does not model:
    - start wear.
    - minimum compressor runtime.
    - command latency.
    - defrost or ambient-dependent cycling limits.
  - This is intentional until the unit proves it needs hard constraints.

- **Config is JSON**
  - JSON has no comments.
  - Temporary experiment note uses a leading underscore key.
  - YAML support would improve operational config readability, but should be a separate change.

## Review Questions

- Should the planner be replaced with a dynamic-programming/MILP formulation?
- If yes, what state should be included?
  - time index.
  - tank temp or SoC bin.
  - compressor on/off.
  - daily 60 C satisfied flag.
  - optional terminal inventory value.
- Should stop cost be modelled as:
  - stop cost,
  - start cost,
  - transition cost,
  - or explicit minimum runtime?
- Should `min_temp` and 60 C satisfaction be hard constraints?
- How should lower routine targets and periodic 60 C reheats be represented?
- Is block metadata enough as an intermediate fix before a full optimiser rewrite?
- Should HWC planning remain custom, or should EMHASS be extended with a better thermal model?

## Candidate Next Architecture

Preferred review target: small bespoke dynamic planner.

- State:
  - `time_step`.
  - `temperature_or_soc_bin`.
  - `compressor_on`.
  - `main_60_satisfied_today`.
- Actions:
  - compressor on/off.
- Cost:
  - import energy cost.
  - transition cost.
  - soft terminal inventory value if needed.
- Constraints:
  - daily/periodic 60 C.
  - minimum temperature floor.
  - optional device limits only after confirmed.
- Output:
  - 5-minute power plan.
  - explicit `planned_blocks` metadata.

Why this may be better:

- Stop/start costs are native objective terms.
- Current compressor state is just initial state.
- Short cycles are discouraged globally.
- `min_temp` can be a real constraint.
- Terminal behaviour can become explicit rather than patched by heuristics.

## Current Operational Commands

```bash
systemctl --user status ai-energy-hwc-daemon.service
journalctl --user -u ai-energy-hwc-daemon.service -n 100 --no-pager
.venv/bin/python -m pytest tests/unit/test_hwc_planner.py tests/unit/test_hwc_executor.py tests/unit/test_hwc_daemon.py
```

## Files For Reviewer

- `hwc_planner.py`
- `hwc_executor.py`
- `services/hwc_daemon.py`
- `config.json`
- `hass/packages/emhass.yaml`
- `docs/hwc_handover.md`
- `docs/hwc_thermal_characterisation.md`
- `docs/hwc_emhass.md`
- `tests/unit/test_hwc_planner.py`
- `tests/unit/test_hwc_executor.py`
- `tests/unit/test_hwc_daemon.py`
