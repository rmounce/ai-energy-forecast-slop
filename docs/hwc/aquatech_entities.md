# Aquatech RAPID X6 HPWH — entity reference

Canonical map of the Home Assistant entities exposed by the Aquatech heat-pump hot-water
unit. **The vendor's entity/display names are misleading** ("exhaust" is refrigerant not air;
"return air" is the *discharged* air not intake; "inlet" duplicates the coil; "outlet" is
dead). This doc records what each entity *actually* measures, verified empirically, so we
don't have to re-derive it.

## How this was verified

InfluxDB (`docker exec influxdb influx ... -database hass`), 2026-06-23:
- 7-day `LAST/MIN/MAX/MEAN/COUNT` per entity (ranges + duplicate detection).
- Two synchronised cycles: a **cold frosty morning** (22 Jun, ambient ~0–6 °C, coil to −16 °C,
  a defrost at ~06:38) and a **mild midday** cycle (23 Jun, ambient ~14 °C). Watching which
  signals lead/lag and cross ambient disambiguates refrigerant vs air and hot vs cold side.

## Temperature entities (the confusing ones)

The unit's manual labels its refrigerant-circuit probes **T1 Coil, T2 Ambient, T3 Exhaust,
T4 Suction, T5 Inlet** (a standard convention for this class of Chinese-OEM HPWH — the Emerald
all-in-one guide and the Rinnai DemandDuo Tuya mapping use the same set). The HA entity names
do **not** match that convention; the table below maps them.

| Entity (`sensor.aquatech_*`) | Manual sensor | What it really is | 7-day range | Verified behaviour |
|---|---|---|---|---|
| `current_temperature_local` | (tank probe) | **Tank water** (control probe) | 45–60 °C | The number the planner uses. Integer resolution. |
| `temperature` | **T2 Ambient** | **Ambient/intake air** | 4–21 °C | Tracks outdoor/room air; the unit's air source. |
| `exhaust_temperature` | **T3 Exhaust** | **Compressor discharge** (hot gas) | 6–81 °C | Cold-starts near ambient, climbs to 58–81 °C as the tank heats. Governs COP. Refrigerant, **not** air. |
| `coil_temperature` | **T1 Coil** | **Evaporator coil** (refrigerant, cold) | −16–32 °C | Holds ~2 °C running (mild), plunges to −16 °C when frosting; spikes +15→32 °C during defrost. |
| `return_air_temperature` | **T4 Suction** | **Suction line** (refrigerant returning from evaporator) — *not air* | −3–40 °C | Sits at the evaporating temp (≈ coil ±1 °C steady); slams 5→40 °C in <1 min on defrost. "Return air" is a mistranslation of "return gas" (回气). See below. |
| `inlet_temperature` | (T1 dup) | **Duplicate of `coil_temperature`** | −16–32 °C | Byte-identical to `coil_temperature` *even through the defrost transient* — same datapoint exposed twice, not a distinct sensor. |
| `inlet_temperature_local` | (T5 Inlet, broken) | Noise — not usable | 1–19 °C | A single ~12-min burst of jumping values (1–19 °C), then silent all week. The real T5 Inlet is not validly exposed. |
| `outlet_temperature` | — | **Bogus / dead** | −50 °C const | Single sample, constant −50 °C. Ignore/disable. |

### Side summary (for the heat-pump cycle)
- **Cold (evaporator) side, refrigerant:** `coil_temperature` (**T1**, coil surface) and
  `return_air_temperature` (**T4**, suction-line gas leaving the evaporator). `inlet_temperature`
  is a duplicate of T1.
- **Hot (condenser) side, refrigerant:** `exhaust_temperature` (**T3**, compressor discharge).
- **Source air:** `temperature` (**T2** ambient).
- **Product:** `current_temperature_local` (tank water).

### Why `return_air` is T4 Suction (not air, not a second coil probe)
1. **Cross-vendor mapping:** the Rinnai DemandDuo (same OEM platform) Tuya DPs expose
   Ambient / Discharge / Tank / Evaporator / **Suction line**, and explicitly *no* "return air"
   sensor — the cold-side gas reading is the suction line (refrigerant returning from the coil).
   These units don't have a return-air probe, so HA's "return_air" is the suction sensor.
2. **Steady state:** suction gas sits right at the evaporating temperature → `return_air` ≈ `coil`
   ±1 °C, exactly as observed. (Bulk air-off would also be impossible here: it sits ~1 °C *below*
   `coil`, and air cannot be colder than the coil that chills it.)
3. **Defrost:** with the fan off, `return_air` jumps 5 → 40 °C in ~52 s — only a refrigerant-line
   probe (seeing reversed hot gas) moves like that, not bulk air.

**T5 Inlet** (likely the refrigerant liquid-line inlet to the evaporator) has **no working HA
entity**: `inlet_temperature` is wired to the T1 coil datapoint (proven identical through the
defrost transient), and `inlet_temperature_local` is noise.

## What the planner consumes

Only **`sensor.aquatech_current_temperature_local`** (tank, via `hwc.tank_temp_entity`).
Ambient/wet-bulb for the thermal model come from the **weather** entity, not the unit. The
other temps are diagnostic/characterisation signals (see `hwc_cop_analysis.py`, which already
reads `exhaust`, `coil`, `return_air`, `inlet`).

## Other relevant Aquatech entities

Verified in use elsewhere:
- `binary_sensor.aquatech_compressor` — compressor running (the on/off the planner cares about).
- `binary_sensor.aquatech_defrost` — vendor defrost flag; **unreliable** (stayed 0 through the
  observed 22 Jun defrost). Prefer `aquatech_four_way_valve` as the real defrost tell.
- `binary_sensor.aquatech_four_way_valve` — reverses during defrost (observed ~2.5 min, coil
  −16 → +15 °C). Best available defrost indicator.
- `binary_sensor.aquatech_element` — resistive backup element.
- `binary_sensor.aquatech_water_flow` — draw-off in progress.
- `sensor.aquatech_flow` — **fan-speed enum** (not a flow rate); `aquatech_flow` despite the name.
- `sensor.aquatech_compressor_strength`, `sensor.aquatech_eev_position_local` — modulation /
  expansion-valve diagnostics (unverified detail).

Nominal/unverified: `aquatech_pump`, `aquatech_running`, `aquatech_fluoride_cycle`,
`aquatech_low_pressure_valve`, `aquatech_power`, `aquatech_problem`, `aquatech_connectivity`.

## Actuation quirks

Confirmed locally 2026-07-27 through HA service calls and the dedicated HWC circuit meter:

- HA advertises a 75 °C maximum, but the physical controller clamps 75 °C to **70 °C** after
  several seconds. Electric mode then runs normally at ~1.78 kW.
- From off, one compound `water_heater.set_temperature` call containing both `temperature: 70`
  and `operation_mode: electric` works.
- Do not send `set_operation_mode` immediately followed by compound `set_temperature`.
  Local Tuya reports success for both but can lose the second datapoint; reproduced result:
  `electric/60` instead of requested `electric/70`.
- HA state can lag physical element start by about a minute. HTTP success is not device
  confirmation; verify observed mode + target and retry.
- `turn_off` preserves the previous target in normal operation. One failed live sequence
  produced `off/15`; this was not reproduced consistently.

### Negative-price incident record

- **2026-07-27, 16:55–17:00:** daemon sent separate `set_operation_mode(electric)` then
  `set_temperature(75, electric)`. HA returned success but the cylinder stayed off. Controlled
  reproduction showed consecutive Local Tuya writes can lose the temperature datapoint
  (`electric/60`); a single compound call works. Also confirmed physical maximum 70 °C.
- **2026-07-28, 12:10 onward:** after an earlier element run ended with the tank at 61 °C,
  repeated compound `electric/70` requests were reflected by HA but the element stayed off and
  the circuit remained at ~1.9 W. The preceding successful run started via `performance` handover
  at 60 °C and drew ~1.78 kW. A fresh `performance/70` selection at 61 °C also failed to start
  the element. Current inference: element-capable modes share a re-trigger threshold/hysteresis
  near 60 °C; keep the requested mode armed and avoid repeated writes while above the threshold.
- Mode + target are therefore necessary but insufficient confirmation. After a start grace,
  control requires the physical element sensor at/below the expected trigger. Above it, element
  off is treated as armed hysteresis rather than an immediate command failure.
- **2026-08-02 controlled boundary test:** with the HWC daemon temporarily stopped, the tank was
  idle at an integer probe reading of 60 °C. A fresh compound `performance/70` command remained
  idle at ~1.8 W, consistent with HYBRID+'s published 50 °C whole-cycle trigger. A fresh compound
  `electric/70` at the same 60 °C reading started the element, compressor stayed off, and circuit
  power stabilised around 1.79–1.81 kW. Changing the active target to 61 °C let the controller stop
  normally at 61 °C. Re-arming `electric/70` at 61 °C then remained idle at ~1.9 W for the observed
  interval. This establishes, at the controller's integer resolution, that Element mode starts at
  60 °C but not 61 °C; the published 60 °C trigger is inclusive.
- **2026-08-02 candidate panel chords:** while `electric/70` was armed but idle at 61 °C, holding
  `M + Up` for three seconds beeped and briefly flashed the element icon, but no relay or power
  transition followed. The related-controller instructions require heating to already be active,
  so support remains unresolved. Holding `Power + Clock + Down` for five seconds also beeped, but
  produced no visible change and no immediate or delayed HA relay/power transition. Manual
  sterilisation was not started under the tested conditions.
- **2026-08-02 F66 enable boundary:** with the controller on in Standard/60 but idle at 61 °C,
  changing the documented legionella setting from `0` to `1` produced no visible response and no
  immediate or one-reporting-interval relay/power transition. F66 did not start an on-demand cycle
  when enabled in this state; its weekly counter start/reset semantics remain unknown.
  F66 was restored to its installed value `0` immediately after the observation.

## Proposed HA display-name renames

To stop future confusion (set via HA UI → entity settings, or `customize.yaml`). Entity IDs
stay as-is to avoid breaking history/automations; only friendly names change:

| Entity | Proposed display name |
|---|---|
| `sensor.aquatech_current_temperature_local` | Aquatech Tank Temp |
| `sensor.aquatech_temperature` | Aquatech Ambient Air Temp |
| `sensor.aquatech_exhaust_temperature` | Aquatech Discharge Temp (T3) |
| `sensor.aquatech_coil_temperature` | Aquatech Evaporator Coil Temp (T1) |
| `sensor.aquatech_return_air_temperature` | Aquatech Suction Line Temp (T4) |
| `sensor.aquatech_inlet_temperature` | Aquatech Coil Temp (dup of T1, disable) |
| `sensor.aquatech_inlet_temperature_local` | Aquatech Inlet (T5, noise — disable) |
| `sensor.aquatech_outlet_temperature` | Aquatech Outlet (BOGUS −50, disable) |
| `sensor.aquatech_flow` | Aquatech Fan Speed |

Recommend **disabling** `outlet_temperature` (dead) and **hiding** `inlet_temperature`
(duplicate) to declutter.

## Sources

- Unit's own manual schematic: T1 Coil, T2 Ambient, T3 Exhaust, T4 Suction, T5 Inlet.
- Same T1–T5 convention cross-checked against the [Emerald all-in-one HPWH troubleshooting
  guide](https://www.emerald.com.au/wp-content/uploads/2025/03/Emerald-Heat-Pump-All-In-One-troubleshooting-guide.pdf).
- [Rinnai DemandDuo / Enviroflo Tuya DP mapping](https://community.home-assistant.io/t/rinnai-enviroflo-heat-pump-hot-water-cylinder-tuya-mapping/1007674)
  (same OEM platform): exposes Ambient / Discharge / Tank / Evaporator / Suction-line, and
  **no** "return air" sensor — confirming `return_air_temperature` is the suction-line probe.
