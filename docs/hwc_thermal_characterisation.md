# HWC thermal & efficiency characterisation (measured)

Measured behaviour of the Aquatech RAPID X6 heat-pump hot water unit, from InfluxDB
telemetry. This is the empirical ground truth behind the COP/thermal assumptions in
`docs/hwc_emhass.md`. Reproduce/extend with `hwc_cop_analysis.py` (writes
`data/hwc_cop_cycles.csv`).

The Aquatech unit was installed on **2026-05-28**. `hwc_cop_analysis.py` defaults
to that date as the earliest query bound so future sweeps do not scan unrelated
pre-installation Home Assistant history.

## Telemetry available (Local Tuya → HA → InfluxDB)

> Canonical, empirically-verified entity map: **`docs/hwc_aquatech_entities.md`**. Maps to the
> manual's T1–T5: `coil`=T1 evaporator coil, `temperature`=T2 ambient, `exhaust`=T3 discharge,
> `return_air`=**T4 suction line** (refrigerant, not "air"), `inlet` duplicates T1, T5 inlet has
> no working entity, `outlet` is dead (−50 °C). The ~2–4 °C evaporator figure below only holds in
> mild weather (coil/suction plunge to −16 °C frosting).

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

**Clean gate** (`hwc_cop_analysis.cycle_is_clean`): a cycle is a usable calibration anchor when
peak power < 1100 W and apparent COP ∈ (0.8, 3.3). A pre/post off-state **baseline-drift** check
(< 80 W) is applied *only* to power-integration-sourced cycles, where the baseline subtraction
feeds elec. Counter-sourced cycles (elec = `energy_2` meter difference) skip it: the baseline never
touches their COP, and `b_pre` is unreliable anyway because the compressor-on edge (laggy
`aquatech_compressor` sensor) lags the power ramp, so the pre-window often catches spin-up.

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

## Finding 4 — the probe-blind build phase (decisive 2-state evidence)

Segmenting reheats (`hwc_soc_extract.py batch` over the power-metered era, or
`hwc_soc_calibrate.py --mode phases` for one window; both share `segment_reheat`) shows every
reheat starts with a **probe-blind build phase**: from compressor-on the control probe (mid-tank,
~50 % height) stays roughly flat while a large slug of energy goes in — it is heating the hot zone
*above* the sensor, which the probe cannot see until the thermocline descends to it.

Across the **18 Athom-metered reheats (2026-06-14 → 06-26)** the blind phase averages **~42 % of
cycle energy** (range 20–72 % over the 16 full reheats). The duration is **not monotone in the
start probe**:

| reheat | start probe | blind phase | rise phase |
|---|---|---|---|
| 2026-06-20 | 52.0 °C (warm) | **77 min / 0.91 kWh (60 %)** | 42 min, +8.3 °C/h |
| 2026-06-26 pm | 53.6 °C (warmer) | 30 min / 0.41 kWh (43 %) | 38 min, +6.9 °C/h |
| 2026-06-25 | 48.5 °C | 62 min / 0.72 kWh (46 %) | 61 min, +9.2 °C/h |
| 2026-06-16 | 49.4 °C | 34 min / 0.40 kWh (31 %) | 64 min, +8.0 °C/h |

The decisive point: 06-20 started **warmer** than 06-25 yet had a far **larger** blind phase, and
48–49 °C starts span 34–62 min of blind work. The blind work is set by the latent hot-volume /
stratification, **not** by the probe reading — so a probe-only heat-rate curve (the rejected
"Option A") provably cannot predict reheat time or energy. This is the measured justification for
the two-state **`(V_hot, T_hot)`** tank model
(quantity vs. temperature of the hot zone), where the blind phase = `V_hot` growing at ~constant
`T_hot` and the rise phase = the thermocline crossing the sensor then `T_hot` climbing 53→60 as
COP collapses (Finding 3). `T_mains` is a model parameter (no water-side sensor logs).
The probe's 1 °C source quantisation (Tuya integers; recorder already captures every tick) bounds
COP-curve resolution — index on the wider-swinging exhaust, or average cycles, not finer logging.

## Finding 5 — standing loss (draw-confounded upper bound)

With no flow/inlet sensor, probe declines during compressor-off mix standing loss with
unobservable draws. A mid-tank probe stays at 60 °C through small draws (the thermocline only
reaches it on a large draw), so the **slowest** multi-hour declines from a full 60 °C tank are the
closest to pure standing loss: these cluster at **0.27–0.40 °C/h**, i.e. standing loss ≈
**0.3 °C/h (~75 W)** as an upper bound — plausible for a modern HPWH tank. In `(V_hot, T_hot)`
terms this acts mainly on `T_hot` (cooling the hot zone) plus slow thermocline diffusion eroding
`V_hot`. A genuine no-draw (away) window would tighten it; don't over-fit this from current data.

## Fan-speed regime (calibration caveat)

Fan speed was reduced via the back-end menu (F30 25→10, F35 55→30) for quieter operation;
the unit will be left in this quieter mode. Per Aquatech this leaves capacity/recovery
roughly unchanged (the compressor sets refrigerant flow) while cutting fan noise and power.
**Any COP calibration is specific to this fan setting** — record the date the change took
effect so cycles aren't mixed across regimes. (TODO: confirm change date.) The calibration is
also **seasonal**: all metered cycles are June-2026 (winter mains, winter wet-bulb), so the
fitted heat rates / power curve should be re-checked as ambient and mains temperatures rise
toward summer. Aquatech also
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
2. The ~50–60 s Tuya/LocalTuya poll lag (both edges) justifies the daemon's off-suppression
   grace (`heat_command_grace_seconds = 120`, i.e. ~2 poll cycles). The grace holds an `off`
   issued just after a heat command until the compressor is *confirmed running* or the grace
   expires — gated on the **current** observed compressor state (`Decision.compressor_on`), not
   an edge latch. (The earlier `compressor_seen_on_since_heat` off→on edge latch was unreliable:
   it missed the already-running case and was reset by the heat re-assertion that the
   compressor's own turn-on triggered via command-cache invalidation, so the grace degenerated
   into a blind, repeatedly re-armed 600 s timer — see git history 2026-06-24.)
3. The same poll lag — plus the sensor reading "off" during a **defrost** — makes the raw
   compressor sensor a poor `compressor_initially_on` for the *planner's* transition accounting:
   a stale "off" prices a phantom restart and can truncate the in-progress block, while a stale
   "on" after a commanded stop can re-price a spurious continue→restart. The daemon therefore
   feeds the planner a **debounced effective-running signal** (`effective_compressor_running`),
   led by command intent (commanded `off` ⇒ off immediately; commanded `heat` within the start
   grace ⇒ on) with the raw sensor as confirmation, a defrost debounce
   (`compressor_off_debounce_seconds = 600`, below-target only), and tank-at-target as a genuine
   stop. The daemon computes it once per tick and feeds the *same* value to both the planner
   (`compressor_initially_on`) and the executor's `decide()` (`effective_compressor_on`), so the
   two layers share one compressor-running view. It is *not* used by the executor's
   off-suppression, which must stay on the raw sensor to bridge the start lag — a distinct
   question ("has the start *confirmed* on the lagging sensor yet", not "is it running"). A
   future improvement is to treat **Athom ch2 power > ~250 W** as the compressor-on signal (leads
   the Tuya binary in both directions), rather than / in addition to
   `binary_sensor.aquatech_compressor`; that would shrink both the grace and the defrost debounce.
4. **Planner direction (2026-06-20 decision):** short-cycle avoidance should *emerge from
   cost* (a per-start cost term), not from hard minimum-runtime or minimum-temperature-rise
   rules. The DP objective is therefore monetary; min_temp / 60 °C remain as
   high-penalty cost terms rather than hard locks.

See `docs/hwc_emhass.md` for the open question of whether to enhance EMHASS's COP model or use
a purpose-built block optimiser.
