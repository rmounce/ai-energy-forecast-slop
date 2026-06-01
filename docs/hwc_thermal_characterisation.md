# HWC thermal & efficiency characterisation (measured)

Measured behaviour of the Aquatech RAPID X6 heat-pump hot water unit, from InfluxDB
telemetry. This is the empirical ground truth behind the COP/thermal assumptions in
`docs/hwc_emhass.md`. Reproduce/extend with `hwc_cop_analysis.py` (writes
`data/hwc_cop_cycles.csv`).

## Telemetry available (Local Tuya → HA → InfluxDB)

Tank/control: `sensor.heat_pump_temperature` (control probe). Refrigerant/air:
`sensor.aquatech_exhaust_temperature` (compressor discharge / condensing temp),
`sensor.aquatech_coil_temperature` + `aquatech_return_air_temperature` + `aquatech_inlet_temperature`
(evaporator/suction side, ~2–4 °C), `sensor.aquatech_temperature` (ambient). State:
`binary_sensor.aquatech_compressor`, `aquatech_defrost`, `aquatech_four_way_valve`,
`aquatech_element`. The unit does **not** meter its own power
(`sensor.aquatech_power`/`_current` barely populate); power is proxied from
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

**Method:** electrical-in = `sensor.remaining_power_load` − pre/post baseline, integrated
over the compressor-on window; thermal-out = tank ΔT × 225 L × 4.186 kJ/kg·K + standing loss.
Single-probe ΔT under-counts thermal (stratification), so the **hard ceiling** — elec vs the
sensible capacity of a 45→60 °C reheat (~3.9 kWh) — is the more robust bound.

Recent clean full-reheat-to-60 °C cycles (see `data/hwc_cop_cycles.csv`):

| ambient | elec (kWh) | thermal (kWh) | **apparent COP** |
|---|---|---|---|
| 13.5 °C | 1.11 | 3.33 | ~3.0 |
| 15.6 °C | 1.63 | 4.12 | ~2.5 |
| 17.2 °C | 1.53 | 3.63 | ~2.4 |
| 55→60 °C top-up only | 0.73 | 1.27 | **~1.75** |

- **Full reheat to 60 °C: COP ≈ 2.4–3.0** (cooler ambient → higher), vs the datasheet's
  headline 4.68 (at WB 15 °C, **to 55 °C**, rating conditions). The gap is the to-60 °C tail
  + real conditions (+ possibly the reduced fan speed).
- **The 55→60 °C top-up alone is COP ≈ 1.75** — the legionella tail is the expensive part.
- A single cycle's COP **cannot exceed ~2.6** given the power drawn and the tank's 45→60 °C
  sensible capacity, regardless of probe/stratification uncertainty.

Across the last 12 days the sweep flags **4 of 6 cycles clean** (mean clean COP ≈ **2.4**);
the two excluded are a contaminated baseline (another load on) and a short partial heat.
(`wet_bulb` is populated from `rp_30m.humidity_adelaide`; regenerate the CSV after analyzer
changes before using it for calibration.)

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
4. **Biggest accuracy lever:** real electrical metering. `remaining_power_load` works only on
   clean windows; a dedicated circuit meter + the exhaust temperature would let us fit
   COP(condensing temp / SoC, wet-bulb) directly.

See `docs/hwc_emhass.md` for the open question of whether to enhance EMHASS's COP model or use
a purpose-built block optimiser.
