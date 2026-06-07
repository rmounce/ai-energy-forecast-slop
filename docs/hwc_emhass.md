# Heat-Pump Hot Water (HWC) Scheduling

**Status:** v1 = modelling only (no actuation). Timer enabled 2026-06-01. The default planner
is now a direct fixed-speed block planner that publishes the HA plan sensors itself.
**Operational guard:** the old EMHASS thermal-battery path remains available with
`hwc.planner: "emhass"`, but it still uses HWC `entity_save` alongside the battery and must run
only against an EMHASS build with the shared metadata race fix. See
`docs/emhass_shared_state_race.md`.
**Started:** 2026-05-31
**Owner workstream memory:** `project_heat_pump_hot_water`

Schedule a newly-installed heat-pump hot water (HPWH) unit intelligently against the
import-price and weather forecasts, optimised **separately** from the home battery (to
keep both optimisations fast). v1 produces a *plan only* — a predicted tank-temperature
trajectory and on/off schedule published as HA sensors. Actuation comes later.

This is the successor concept to the abandoned HVAC attempt in `actrl/statoptim.py`
(given up because the house had little thermal mass and per-room solar gain wrecked the
temperature model). A hot water tank avoids both problems: real bounded thermal mass, no
solar gain, ambient = outdoor air.

---

## Hardware

- **Unit:** Aquatech RAPID X6 Gen2 (same compressor/electronics as Hydrotherm Dynamic X8
  Gen6, shorter tank). Located **outdoors**.
- **Tank:** 225 L.
- **Compressor:** ~700–900 W electrical input. Reaches 60 °C unaided.
- **Resistive element:** 1800 W, needed only for 60→70 °C. **Out of scope for v1**
  (future: "dump load" on negative prices).
- **Datasheet COP (vs wet-bulb, recovery 15→55 °C):**

  | Wet-bulb | Heating cap | COP | Recovery 15→55 |
  |---|---|---|---|
  | 0 °C | 1500 W | 2.5 | 445 min |
  | 7.5 °C | 2180 W | 3.45 | 285 min |
  | 15 °C | 3165 W | 4.68 | 198 min |
  | 25 °C | 4000 W | 5.65 | 145 min |

  Electrical input is roughly flat (~600–710 W); the big swing is in *speed/COP*, so a
  warm-afternoon reheat is both cheaper per kWh and ~3× faster than a cold one. This
  conveniently coincides with the cheap midday / high-PV window on Amber in SA.

## Home Assistant entities (Local + cloud Tuya)

| Entity | Role | Notes |
|---|---|---|
| `sensor.heat_pump_temperature` | tank temperature (model state) | cloud Tuya; **logged to InfluxDB** — use for now |
| `water_heater.aquatech` | Local Tuya climate entity | has `current_temperature` attr; history may not be logged; future local-only path |
| `binary_sensor.aquatech_compressor` | compressor on/off | state-change logging → query InfluxDB with `fill(previous)`, not `fill(0)` |

**Current control today:** an HA timer flips the unit eco→standard between 10:00–16:00
daily; observed reheat ~11:00–13:00 to 60 °C. This is what v2 actuation will replace.

---

## Planner model

The default planner is purpose-built for this unit. It treats the Aquatech as a fixed-speed
block heater, not a continuously variable thermal store:

- 72h horizon (`horizon_steps: 144` at 30 min).
- dynamic standing loss: `standing_loss_ua_kw_per_c × (tank_temp - dry_bulb)` rather than
  EMHASS's constant loss term.
- fixed heat rate (`heat_rate_c_per_hour`) and full-power schedule steps.
- one main daytime block per local day, chosen inside `block_planner.main_window_*`.
- repair boosts only if the model would breach the minimum temperature floor.
- terminal inventory contract: default `terminal_target: "current"`, so the end-of-horizon
  tank temperature is at least the current temperature.

It publishes the same HA entities as the former EMHASS path:
`sensor.hwc_predicted_temp`, `sensor.hwc_power_plan`, and `sensor.hwc_unit_load_cost`.

## EMHASS thermal-battery model (fallback)

**EMHASS is v0.17.5** (standalone docker `ghcr.io/davidusb-geek/emhass`, config in
`/opt/dockerfiles/emhass/`). Verified that this version ships the **new physics-based
thermal model** (Carnot COP, `draw_off_demand`, `volume`, `thermal_loss`) under
`def_load_config[k]["thermal_battery"]`, and that it runs in **`naive-mpc-optim`** (the
thermal logic lives in the shared `perform_optimization` core, dispatched on
`params["type"] == "thermal_battery"`). **No EMHASS upgrade required.**

COP per timestep: `COP = carnot_efficiency × T_supply_K / (T_supply_K − T_outdoor_K)`,
clamped ≥ 1. Tank dynamics:
`T[t+1] = T[t] + conversion × (COP[t]·P[t]/1000 − draw_off[t] − loss[t])`.

### Wet-bulb vs dry-bulb (resolved)

Physically, COP follows **wet-bulb** (evaporator pulls latent heat from humid air) while
standing loss follows **dry-bulb** (tank shell → ambient, ∝ tank temp − air temp). But in
**tank mode** (`draw_off_demand` present) EMHASS treats `thermal_loss` as a **constant**
(`optimization.py:602`), and the single outdoor-temp input feeds **only the COP**. So:

- **Feed wet-bulb** as the outdoor-temperature forecast (computed from the temp + humidity
  forecast we already have). This is the only place outdoor temp matters in tank mode and
  it's wet-bulb's domain; it's also self-consistent with `carnot_efficiency` calibrated
  against wet-bulb.
- Set `thermal_loss` to a constant representative value. Accept that its real
  dry-bulb/tank-temp dependence is unmodelled — it's the smallest term (~0.12 kW vs
  ~0.5–1.4 kW heating, ~1.3 kWh draws), so this is second-order. If seasonal swing ever
  looks material, recalibrate `thermal_loss` seasonally (cheap) rather than switching to
  building-heating mode (expensive — would lose native `draw_off_demand`).

### Starting parameters (calibrate from logs)

Derived from the first ~1.5 days of InfluxDB history (Adelaide local):

- Standing loss **~0.45 °C/h** (59.5 °C @Sat 16:30 → 53.4 °C @Sun 06:00) ⇒
  `thermal_loss ≈ 0.12 kW` (225 L × 4.186 × 0.45 / 3600).
- Reheat **~5–5.6 °C/h to 60 °C** in current cool ambient (slower than datasheet's
  to-55 °C rows, as expected for the hard last 5 °C).
- Morning shower **~08:00–09:00**, ~5 °C drop beyond standing loss ≈ **~1.3 kWh** draw.

| Param (`thermal_battery` dict) | Start value | Source |
|---|---|---|
| `volume` | 0.225 (m³) | datasheet |
| `density` / `heat_capacity` | 997 / 4.184 | water |
| `supply_temperature` | ~60 °C | target |
| `carnot_efficiency` | ~0.38 | measured-cycle calibration (vs wet-bulb) |
| `thermal_loss` | ~0.12 kW | **standing-loss data** |
| nominal power (elec) | ~800 W | datasheet compressor input |
| `min_temperatures` | 45 °C | hard floor (required non-empty) |
| `max_temperatures` | 60 °C | heat-pump-only cutoff; element >60 is out of scope |
| `desired_temperatures` | 60 °C | daily legionella target |
| `draw_off_demand` | ~1.3 kWh @ ~08:00 | **shower data** |
| `thermal_inertia_time_constant` | ~0.5 h | anti-cycling / probe lag |
| outdoor temp forecast | **wet-bulb** | from temp + humidity |

**Anti-cycling:** the EMHASS thermal model has no startup penalty (only standard deferrable
loads do), and in practice it produced fragmented starts. That is why the default planner moved
to fixed block placement.

---

## Architecture (v1)

**Vehicle (decided 2026-05-31):** a **Python script + systemd timer**, mirroring
`forecast.py` and the existing pipeline timers — *not* a Jinja rest_command. Rationale:
the payload requires clock-aligned per-timestep arrays (because `draw_off_demand` tiles
from horizon start, not clock time) and a wet-bulb computation (Stull's formula +
hourly→30-min interpolation). That is fragile and untestable in Jinja, and "hass Jinja
complexity" is a noted project pain point. Python keeps it testable. A thin
`hass/packages/emhass_hwc.yaml` still holds the published plan sensors (and any helpers).

- The script: reads current tank temp, the weather (temp/humidity), and the
  EMHASS-prepared import-price series from HA (`sensor.mpc_unit_load_cost` for the
  5-minute near term, `sensor.dh_unit_load_cost` for the 30-minute tail). This keeps HWC
  aligned with the same Jinja source selection, current-interval handling, buy-price
  weighting, rounding, and fallback logic used by the battery optimiser.
- Optional EMHASS fallback: a **separate** `naive-mpc-optim` call to the existing EMHASS
  instance with:
  - `set_use_battery: false`, `set_use_pv: false` (decoupled, fast)
  - `number_of_deferrable_loads: 1`, the load configured as a `thermal_battery`
  - `load_cost_forecast` = our import-price forecast (already encodes the cheap midday
    window); `prod_price_forecast` irrelevant
  - `start_temperature` ← `sensor.heat_pump_temperature`
  - outdoor temp ← computed **wet-bulb** series
- **Publish only**: predicted tank-temperature trajectory + on/off plan as HA sensors.
  No actuation. Run alongside reality for 1–2 weeks; tune `carnot_efficiency`,
  `thermal_loss`, `draw_off_demand` against observed reheat/standing-loss.

### Configuration & secrets

Non-secret settings live under the `hwc` key in the tracked `config.json` (entities,
thermal params, draw-off window, horizon). **Secrets — including the internal EMHASS
hostname — go in the untracked `config.secrets.json`** (`hwc.emhass_base_url`), which
`config_utils.load_config()` deep-merges over `config.json`; a placeholder is in the
tracked `config.secrets.json.example`. Never put hostnames/domains/tokens in `config.json`.
Run with `python hwc_planner.py --dry-run` to build + log the payload without POSTing.

### Published entities & visualisation

The direct block planner publishes these entities using `publish_prefix` (`hwc_`). The EMHASS
fallback has the same double-prefix gotcha as before: the prefix is prepended to the
`custom_*_id` you pass, so the custom IDs must be the *bare* names
(`sensor.predicted_temp`, `sensor.power_plan`) to avoid `sensor.hwc_hwc_…`. Published entities:

| Entity | Attribute (time series) | Item value key |
|---|---|---|
| `sensor.hwc_predicted_temp` | `predicted_temperatures` | `hwc_predicted_temp` |
| `sensor.hwc_power_plan` | `deferrables_schedule` | `hwc_power_plan` |
| `sensor.hwc_unit_load_cost` | `unit_load_cost_forecasts` | `hwc_unit_load_cost` |

Plus EMHASS's usual `sensor.hwc_optim_status`, `sensor.hwc_p_grid_forecast`, etc. The
forecast arrays are in entity *attributes* (state holds only the first value); values are
strings. Dashboard card: `hass/lovelace-hwc-apexcharts.yaml` (needs the HACS
`apexcharts-card`), with `data_generator` series + 45 °C / 60 °C annotation lines.

### First block-planner observations (2026-06-01)

The first live block plan published 144 points (72h), 3 starts, 6.8 kWh of planned compressor
energy, minimum predicted temperature ~46.6 °C, and terminal temperature ~57.2 °C against a
57.0 °C start. This addressed the EMHASS symptoms: 48-point output, decay to the 45 °C floor,
and fragmented/fractional compressor starts.

### Known v1 gaps (deliberate)

- **Standing loss is temperature-independent in tank mode.** EMHASS treats `thermal_loss`
  as a flat constant when `draw_off_demand` is present (`optimization.py:602`), so the
  model does *not* capture that real loss scales with (tank temp − ambient dry-bulb) — an
  outdoor tank loses more on a cold night and when the tank is hotter. We accept a single
  representative constant for v1 (loss is the smallest energy term). **Future
  investigation:** quantify how much this approximation costs once we have multi-season
  data — options are seasonal/dynamic `thermal_loss` recalibration (cheap) or switching to
  building-heating mode with `calculate_thermal_loss_signed` (expensive; loses native
  `draw_off_demand`).
- **Battery decoupling:** the battery optimisation is *not* told about the ~1 kWh HWC
  load. Acceptable for v1; later, feed the planned HWC draw into the battery's load
  forecast so it isn't double-counted.
- **Cloud dependence:** tank temp comes via cloud Tuya (`sensor.heat_pump_temperature`).
  Move to Local Tuya (`water_heater.aquatech`) once we confirm its history is usable.

## Roadmap

- **v1 (this doc):** modelling only — publish the plan, calibrate from logs.
- **v2:** actuate via Local Tuya / HA `water_heater` services from the published plan.
- **v3+:** feed HWC load into the battery optimisation; resistive-element dump load on
  negative prices.

## Execution Layer

`hwc_executor.py` was the first standalone actuation helper. Live execution is now handled
by `services/hwc_daemon.py` under `ai-energy-hwc-daemon.service`; the old executor timer
has been removed.

Live HA metadata observed 2026-06-01:

- entity: `water_heater.aquatech`
- modes: `off`, `heat_pump`, `eco`, `high_demand`, `performance`, `electric`
- services: `water_heater.set_temperature`, `set_operation_mode`, `turn_on`, `turn_off`
- current observed state: `eco`, setpoint `55`, compressor sensor `binary_sensor.aquatech_compressor`

The executor policy is intentionally simple:

- inside a planned block, set `hwc.actuation.operation_mode` (default `heat_pump`) and setpoint
  to the predicted temperature at the end of that contiguous block, clamped to 55-60 °C;
- just after a block, if the compressor is still running, keep the same mode/setpoint so the
  unit can finish naturally;
- outside a block, once the compressor is off, call `water_heater.turn_off` so the tank can
  drift below the unit's normal 55 °C reheat trigger.

Run `python hwc_executor.py --dry-run` to inspect the current decision without actuation.
When ready, set `hwc.actuation.enabled` true and enable the executor timer.
