# HWC thermal & efficiency characterisation (measured)

Measured behaviour of the Aquatech RAPID X6 heat-pump hot water unit, from InfluxDB
telemetry. This is the empirical ground truth behind the COP/thermal assumptions in
`docs/hwc_emhass.md`. Reproduce/extend with `hwc_cop_analysis.py` (writes
`data/hwc_cop_cycles.csv`).

The Aquatech unit was installed on **2026-05-28**. `hwc_cop_analysis.py` defaults
to that date as the earliest query bound so future sweeps do not scan unrelated
pre-installation Home Assistant history.

## Telemetry available (Local Tuya → HA → InfluxDB)

Tank/control: `sensor.heat_pump_temperature` (control probe). Refrigerant/air:
`sensor.aquatech_exhaust_temperature` (compressor discharge / condensing temp),
`sensor.aquatech_coil_temperature` + `aquatech_return_air_temperature` + `aquatech_inlet_temperature`
(evaporator/suction side, ~2–4 °C), `sensor.aquatech_temperature` (ambient). State:
`binary_sensor.aquatech_compressor`, `aquatech_defrost`, `aquatech_four_way_valve`,
`aquatech_element`. The unit does **not** meter its own power
(`sensor.aquatech_power`/`_current` barely populate); current electrical input comes from
raw Athom channel 2 (`sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2`),
with older history proxied from
`sensor.remaining_power_load` (see COP method).

## Finding 1 — the two-stage condensing-temperature rise = stratified charging

During a reheat the discharge/condensing temperature (`exhaust`) rises fast, flattens
~50 °C, then rises fast again to ~80 °C before flattening — consistently. It is **not**
defrost or the element (`defrost`/`four_way`/`element` all off through the cycle).

The tell: the **tank probe sits flat (~45–47 °C) for the first ~45 min while the condenser
temperature climbs and plateaus ~50–55 °C**, then both shoot up together. This is a
strongly **stratified tank** charged via a descending thermocline:

- **Phase 1** (gentle exhaust ~42→55 °C, probe flat): heated water is buoyant and forms a
  growing hot layer at the top; the condenser still rejects into relatively cool water →
  **low lift, high COP**. The control probe is below the thermocline, so it doesn't move.
- **Transition** (~min 50–60): the thermocline reaches the probe / cool reservoir is used up.
- **Phase 2** (exhaust steeply 57→82 °C, probe 49→60): condenser now rejects into hot water
  → **high lift, COP collapses**; the probe finally tracks up to the 60 °C setpoint.

**Implication:** the COP is governed by the condensing (exhaust) temperature, which is a
function of tank **state-of-charge**, *not* of the single control-probe reading (which is
unrepresentative for half the cycle). The evaporator side stays ~2–4 °C throughout (the EEV
holds superheat).

## Finding 2 — fixed-speed compressor (confirmed by the power profile)

Heat-pump power (baseline-subtracted) rises **smoothly** with the condensing temperature
(~520 → 870 W across the cycle) with **no step**. For a fixed-speed compressor that's
expected — constant refrigerant mass flow, so power follows discharge pressure as the
condensing temperature climbs. So the phase-2 acceleration is the water-side thermocline,
**not** an inverter speed change. (Unit confirmed fixed-speed + EEV.)

## Finding 3 — measured COP (and it's well below datasheet)

**Method:** electrical-in prefers raw Athom channel 2
(`sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2`), falling back to
`sensor.remaining_power_load` − pre/post baseline for older history, integrated
over the compressor-on window; thermal-out = tank ΔT × 225 L × 4.186 kJ/kg·K + standing loss.
Single-probe ΔT under-counts thermal (stratification), so the **hard ceiling** — elec vs the
sensible capacity of a 45→60 °C reheat (~3.9 kWh) — is the more robust bound.

Recent clean full-reheat-to-60 °C cycles (see `data/hwc_cop_cycles.csv`):

| ambient | elec (kWh) | thermal (kWh) | **apparent COP** |
|---|---|---|---|
| 13.5 °C | 1.11 | 3.33 | ~3.0 |
| 15.6 °C | 1.63 | 4.12 | ~2.5 |
| 17.2 °C | 1.53 | 3.63 | ~2.4 |
| 14.2 °C / WB 8.6 °C | 2.06 | 5.28 | ~2.6 |
| 16.5 °C / WB 15.6 °C | 1.31 | 2.97 | ~2.3 |
| 14.5 °C / WB 9.8 °C | 1.68 | 3.66 | ~2.2 |
| 55→60 °C top-up only | 0.73 | 1.27 | **~1.75** |

- **Full reheat to 60 °C: COP ≈ 2.4–3.0** (cooler ambient → higher), vs the datasheet's
  headline 4.68 (at WB 15 °C, **to 55 °C**, rating conditions). The gap is the to-60 °C tail
  + real conditions (+ possibly the reduced fan speed).
- **The 55→60 °C top-up alone is COP ≈ 1.75** — the legionella tail is the expensive part.
- A single cycle's COP **cannot exceed ~2.6** given the power drawn and the tank's 45→60 °C
  sensible capacity, regardless of probe/stratification uncertainty.

The calibration CSV currently has **12 clean cycles out of 14** (mean clean COP ≈ **2.3**);
the excluded rows are contaminated/partial windows. The five clean Athom-metered cycles from
2026-06-14 through 2026-06-18 have mean COP ≈ **2.24**.
(`wet_bulb` is populated from `rp_30m.humidity_adelaide`; regenerate the CSV after analyzer
changes before using it for calibration.) Keep `data/hwc_cop_cycles.csv` as the
machine-readable cycle table, and write `--summary-md docs/hwc_calibration_cycles.md`
when a run should be easy to review in Git.

## Fan-speed regime (calibration caveat)

Fan speed was reduced via the back-end menu (F30 25→10, F35 55→30) for quieter operation;
the unit will be left in this quieter mode. Per Aquatech this leaves capacity/recovery
roughly unchanged (the compressor sets refrigerant flow) while cutting fan noise and power.
**Any COP calibration is specific to this fan setting** — record the date the change took
effect so cycles aren't mixed across regimes. (TODO: confirm change date.) Aquatech also
suggest a main 10:00–18:00 timer plus a short morning-boost timer — relevant to the schedule
design (two reheats, not one).

## Modelling implications

1. EMHASS's `thermal_battery` uses a **single-node** tank and a **fixed `supply_temperature`**
   (60) Carnot COP. Neither matches reality: the tank stratifies, and the effective condensing
   temperature swings 50→82 °C with SoC. The datasheet-derived `carnot_efficiency ≈ 0.45–0.5`
   is too optimistic; measured cycles imply ≈ **0.36–0.40** with an effective supply temp > 60.
2. The marginal cost of the 55→60 °C tail (COP ~1.75–2) is much worse than the cycle average —
   quantitative backing for not always heating to 60 °C.
3. The unit is **fixed-speed and runs as a ~2 h block** (45→60), so for scheduling the key
   quantity is *electrical energy + duration as a function of (start temp, target, ambient/
   wet-bulb)* — a low-dimensional empirical curve we can **measure**, rather than the intra-cycle
   trajectory the unit can't be controlled to follow anyway.
4. **Biggest accuracy lever landed:** dedicated circuit metering should replace
   `remaining_power_load` for new cycles; pairing it with exhaust temperature lets us fit
   COP(condensing temp / SoC, wet-bulb) directly once enough clean cycles are collected.
5. The block planner now publishes modelled compressor watts rather than a flat nameplate
   value. The 2026-06-19 fit uses recent Athom-metered active samples:
   `740 W @ tank 50 °C / wet-bulb 12.5 °C`, `+15 W/°C` tank, `+1.5 W/°C` wet-bulb,
   clamped `650–930 W`. This reduced active-sample power MAE from about **63 W** under
   the first-pass config to about **27 W**.

## Actuation semantics (measured 2026-06-20)

Manual `water_heater.aquatech` test from `operation_mode=off`, tank 57 °C, ambient 11 °C
(daemon stopped):

- **Start above the nominal trigger works.** `set_operation_mode: heat_pump` (setpoint 60)
  starts a 57→60 top-up from `off` — the unit does not refuse to start because the tank is
  already above 55 °C. So the executor's 55 °C `setpoint_min_c` is policy, not a device limit
  (entity setpoint range is 15–75 °C, 1 °C step; `operation_list` = off/heat_pump/eco/
  high_demand/performance/electric; `supported_features` = 15).
- **Real device start/stop is fast; the Tuya compressor binary lags ~50 s on *both*
  edges.** On start, Athom ch2 power rises within a few seconds; on `turn_off`, compressor
  power drops instantly (fan stops ~30 s later, power → ~0). In both cases
  `binary_sensor.aquatech_compressor` only flips ~50 s after the real transition — a Local
  Tuya **polling lag**, not device latency. So **Athom ch2 power is the faster, authoritative
  compressor-on signal**, leading the Tuya binary by ~50 s in each direction.
- **Power magnitudes / threshold.** Fan alone ≈ 50 W (possibly low speed); the full unit
  briefly dipped as low as ~363 W at startup before climbing (running range up to ~930 W; cf.
  modelled clamp 650–930 W — 363 W is a startup transient). A **~250 W threshold** cleanly
  separates compressor-running from fan-only/standby for a power-based compressor-on signal.
- **EEV modulates during the ramp** (`sensor.aquatech_eev_position_local`, from the
  `water_heater` `eev` attribute) — consistent with EEV superheat control on a fixed-speed
  compressor.

**Implications:**
1. Actuation latency is negligible for control — min-runtime is a wear/COP choice, not a
   command-lag workaround. The executor's "starts/stops promptly" assumption holds at the
   device.
2. The ~50 s Tuya binary lag (both edges) justifies the daemon's off-suppression grace
   (`heat_command_grace_seconds = 600` ≫ 50 s). A future improvement is to treat **Athom ch2
   power > ~250 W** as the compressor-on signal (leads the Tuya binary in both directions),
   rather than / in addition to `binary_sensor.aquatech_compressor`.
3. **Planner direction (2026-06-20 decision):** short-cycle avoidance should *emerge from
   cost* (a per-start cost term), not from hard minimum-runtime or minimum-temperature-rise
   rules. The DP objective is therefore monetary; min_temp / 60 °C remain as
   high-penalty cost terms rather than hard locks.

See `docs/hwc_emhass.md` for the open question of whether to enhance EMHASS's COP model or use
a purpose-built block optimiser.
