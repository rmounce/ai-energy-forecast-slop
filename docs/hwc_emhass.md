# Heat-Pump Hot Water (HWC) Scheduling via EMHASS

**Status:** v1 = modelling only (no actuation). Spec/plan.
**⚠️ PAUSED (2026-06-01):** the planner + timer are **disabled**. Running HWC `entity_save`
alongside the battery corrupted EMHASS's shared `/data/entities/metadata.json` (and can
break the battery's publish). Decision: fix this **upstream in EMHASS** rather than work
around it here (atomic metadata writes / publish-without-2nd-request / prefix-scoped state
files). See `docs/emhass_shared_state_race.md`. Do not re-enable until EMHASS is fixed or
HWC is moved to a dedicated instance.
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

## EMHASS model

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
- Morning shower **~09:00–10:00**, ~5 °C drop beyond standing loss ≈ **~1.3 kWh** draw.

| Param (`thermal_battery` dict) | Start value | Source |
|---|---|---|
| `volume` | 0.225 (m³) | datasheet |
| `density` / `heat_capacity` | 997 / 4.184 | water |
| `supply_temperature` | ~60 °C | target |
| `carnot_efficiency` | ~0.45–0.5 | back-solved from datasheet (vs wet-bulb) |
| `thermal_loss` | ~0.12 kW | **standing-loss data** |
| nominal power (elec) | ~800 W | datasheet compressor input |
| `min_temperatures` | 45 °C | hard floor (required non-empty) |
| `max_temperatures` | ~62 °C | element handles >60 (required non-empty) |
| `desired_temperatures` | 60 °C | daily legionella target |
| `draw_off_demand` | ~1.3 kWh @ ~09:00 | **shower data** |
| `thermal_inertia_time_constant` | ~0.5 h | anti-cycling / probe lag |
| outdoor temp forecast | **wet-bulb** | from temp + humidity |

**Anti-cycling:** the thermal model has no startup penalty (only standard deferrable
loads do). For a once-daily 60 °C reheat with a 45 °C floor the LP should naturally
produce a single block; rely on that + a small `thermal_inertia_time_constant`. Add a
hard "max 1 start/day" guard in the actuation layer (v2), not the optimiser.

---

## Architecture (v1)

**Vehicle (decided 2026-05-31):** a **Python script + systemd timer**, mirroring
`forecast.py` and the existing pipeline timers — *not* a Jinja rest_command. Rationale:
the payload requires clock-aligned per-timestep arrays (because `draw_off_demand` tiles
from horizon start, not clock time) and a wet-bulb computation (Stull's formula +
hourly→30-min interpolation). That is fragile and untestable in Jinja, and "hass Jinja
complexity" is a noted project pain point. Python keeps it testable. A thin
`hass/packages/emhass_hwc.yaml` still holds the published plan sensors (and any helpers).

- The script: reads current tank temp + the weather (temp/humidity) and import-price
  forecasts from HA/InfluxDB, computes wet-bulb and the clock-aligned input arrays, POSTs
  to EMHASS, and publishes the resulting plan back to HA.
- A **separate** `naive-mpc-optim` call to the existing EMHASS instance with:
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

EMHASS publishes the plan to HA, prefixing every entity with `publish_prefix` (`hwc_`).
**Gotcha:** the prefix is prepended to the `custom_*_id` you pass, so the custom IDs must
be the *bare* names (`sensor.predicted_temp`, `sensor.power_plan`) to avoid a double
prefix (`sensor.hwc_hwc_…`). Published entities:

| Entity | Attribute (time series) | Item value key |
|---|---|---|
| `sensor.hwc_predicted_temp` | `predicted_temperatures` | `hwc_predicted_temp` |
| `sensor.hwc_power_plan` | `deferrables_schedule` | `hwc_power_plan` |
| `sensor.hwc_unit_load_cost` | `unit_load_cost_forecasts` | `hwc_unit_load_cost` |

Plus EMHASS's usual `sensor.hwc_optim_status`, `sensor.hwc_p_grid_forecast`, etc. The
forecast arrays are in entity *attributes* (state holds only the first value); values are
strings. Dashboard card: `hass/lovelace-hwc-apexcharts.yaml` (needs the HACS
`apexcharts-card`), with `data_generator` series + 45 °C / 60 °C annotation lines.

### First real-run observations (2026-06-01) — calibration TODO

First live optim (start 56 °C, cool ambient) behaved qualitatively correctly: coasted to
the 45 °C floor overnight, reheated in the cheapest midday window ($0.06–0.09/kWh ~10:00–
13:30), and avoided the evening peak ($0.40+). Two tuning items (anticipated; not blocking):

- **Doesn't reach 60 °C** — topped out ~59 °C; `penalty_factor` (15) is too weak vs cost to
  force the daily legionella reheat. Options: raise `penalty_factor`, or add a hard daily
  `min_temperatures` bump (e.g. ≥58 at one afternoon slot) — watch for LP infeasibility.
- **Some cycling** — heats in a few short bursts rather than one block (no thermal startup
  penalty in EMHASS). Options: raise `thermal_inertia_time_constant`, or enforce a single
  block in the actuation layer. Also the LP uses fractional power (e.g. 76 W steps); the
  real unit is ~735 W on/off — consider `semi_cont` to force 0/full.

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
- **v2:** actuate via Local Tuya (mode/setpoint), with a max-1-start/day guard.
- **v3+:** feed HWC load into the battery optimisation; resistive-element dump load on
  negative prices.
