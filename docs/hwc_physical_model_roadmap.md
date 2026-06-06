# HWC Physical Model Roadmap

Status: superseded in part by `docs/hwc_observer_milp_path.md`.

This note records where the hot-water-control model currently stands, what we think the
important physical/economic effects are, and how to move from the current MVP planner to a
more accurate inventory/COP-aware optimiser without trying to solve everything at once.

## Current State

Live control is intentionally conservative:

- `services/hwc_daemon.py` watches Home Assistant inputs and republishes/executes plans.
- `hwc_planner.py` uses a deterministic block planner with a single effective tank
  temperature, fixed heat rate, fixed standing loss, and a 30-minute grid.
- `hwc_executor.py` turns the Aquatech on for planned blocks and off afterwards, with guards
  against very small top-up starts.
- `hwc_cop_analysis.py` extracts compressor cycles from HA/InfluxDB since the Aquatech install
  date, `2026-05-28`.
- `data/hwc_cop_cycles.csv` is the machine-readable calibration table.
- `docs/hwc_calibration_cycles.md` and `docs/hwc_model_fit.md` are human-readable snapshots.
- `hwc_stratified_model.py` is an offline two-layer tank model scaffold. It is not used by
  the live planner.
- `docs/hwc_observer_milp_path.md` defines the current next modelling path: an observer that
  estimates useful thermal inventory, followed by a shadow MILP/MPC planner.

The MVP is good enough to run and observe, but the physical model is still too simple. The
next improvements should be made offline, validated against observed cycles, then shadowed in
Home Assistant before controlling the unit.

## What We Have Learned

Observed behaviour strongly supports a stratified-tank interpretation:

- The control probe can stay nearly flat while the compressor is clearly heating water.
- Exhaust/condensing temperature rises in stages, eventually reaching roughly `70-83 C`.
- The probe then rises quickly once the thermocline reaches it.
- The compressor appears fixed-speed; power rises smoothly through the cycle rather than
  stepping between inverter speeds.
- Baseline-subtracted compressor power is roughly `700-800 W` mean on clean cycles, with
  higher values late in a cycle.
- Full reheats to `60 C` are materially more efficient than small near-target top-ups.
- Clean extracted-cycle mean COP is currently around `2.35`, but the sample is still small.

The latest calibration snapshot estimates:

- clean full-reheat median 50% probe-rise timing: `78.0 min`
- clean full-reheat median 90% probe-rise timing: `108.5 min`
- preliminary `probe_height_fraction`: `0.62`
- preliminary `thermocline_width_fraction`: `0.63`

Treat these as hints, not settled parameters.

## Modelling Direction

The target model should stop treating the tank as one scalar probe temperature. The current
preferred live-control state is useful thermal inventory:

- `E_t`: kWh above the `45 C` floor;
- `E_max`: inventory when the physical probe reaches the `60 C` shut-off event;
- optional observer internals for hot/cold layer hints and probe correction.

The existing two-layer variables remain useful offline diagnostics:

- `cold_temp_c`
- `hot_temp_c`
- `hot_fraction`
- derived `probe_temp_c`
- derived `mean_temp_c` or stored thermal energy

Heating should grow the hot layer first. The control probe should respond only when the
thermocline reaches the probe region. Draw-off should consume hot-layer useful energy first
and refill/mix with cold inlet water. Standing loss should depend on the stratified state, not
only the control-probe reading.

The economic planner should optimise over useful hot-water inventory and constraints, not only
over "probe reaches 60 C". The physical `60 C` probe event remains the source of truth for
daily target satisfaction until the observer is validated.

## COP And Compressor Power

We are not yet modelling physical COP from wet-bulb temperature in the live planner.
The block planner now has a conservative wet-bulb adjustment to empirical recovery
speed, seeded from the datasheet direction but anchored to observed cycles. We are
still collecting cycle wet-bulb data and compressor-power proxies before promoting
this into a real COP/power model.

The compressor power curve should be modelled as a function of temperatures. The most useful
temperature proxy is likely exhaust/condensing temperature, not the control-probe temperature.
For a fixed-speed compressor, the expected shape is:

```text
compressor_power_w = f(condensing_temp_proxy, evaporating_temp_proxy)
```

Practical proxies from available telemetry:

- `condensing_temp_proxy`: `sensor.aquatech_exhaust_temperature`
- `evaporating_temp_proxy`: wet bulb, `coil`, `return_air`, or `inlet`
- `tank_state`: hot fraction / thermocline phase / target temp

The next COP model should look more like:

```text
power_w = power_model(exhaust_temp_proxy, wet_bulb_or_evap_proxy)
cop = cop_model(tank_state, wet_bulb_or_evap_proxy, exhaust_temp_proxy)
thermal_kwh = electrical_kwh * cop
```

or, if direct COP fitting remains too noisy:

```text
electrical_kwh_to_target = empirical_energy_model(start_state, target, wet_bulb)
duration_to_target = empirical_duration_model(start_state, target, wet_bulb)
```

The second form may be easier and more robust with only a small number of clean cycles. The
first form is more physically meaningful and should become feasible if we get better power
metering or enough clean proxy cycles.

## Missing Parameters To Consider

Physical:

- COP as a function of wet bulb / evaporator condition and tank state.
- Compressor power as a function of condensing and evaporating temperature proxies.
- Exhaust/condensing-temperature trajectory as a function of hot-layer fraction.
- Hot-layer fraction, thermocline width, probe height, and mixing rate.
- Cold inlet temperature, including seasonal mains-water variation.
- Draw-off size, duration, timing, and mixing/refill behaviour.
- Standing loss as a function of stratified state and ambient around the unit.
- Fan-speed regime, because calibration is only valid for the current quiet fan setting.
- Element/defrost/four-way-valve state, which should either exclude or separately model cycles.
- Autonomous Aquatech hysteresis, safety reheats, compressor latch delays, and setpoint cut-off.

Economic/control:

- 5-minute tariff resolution near price troughs, if it materially changes block timing.
- Forecast uncertainty for price, weather, and draw-off demand.
- Explicit start penalty / minimum runtime / minimum lift for wear and efficiency.
- Legionella/safety policy as a constraint based on last-achieved `60 C`, not an accidental
  consequence of daily scheduling.
- Execution error feedback: planned start/end versus actual compressor on/off and actual
  achieved temperature.

## Lowest-Hanging Fruit

The order below is deliberately pragmatic. It prioritises work that can improve confidence
without destabilising the working MVP.

1. **Incremental cycle extraction**
   The current Influx extraction is too slow for routine iteration. Add explicit `--since`,
   `--until`, and append/merge behaviour so new cycles can be added without re-scraping all raw
   history.

2. **Offline cycle replay validation**
   Build `hwc_validate_stratified_model.py` to replay observed clean cycles through
   `hwc_stratified_model.py` and report error against:
   - `probe_rise_10_min`
   - `probe_rise_50_min`
   - `probe_rise_90_min`
   - end probe temperature
   - duration to target
   - electrical kWh

3. **Fit thermocline/probe parameters**
   Use observed probe-rise timing to fit `probe_height_fraction` and
   `thermocline_width_fraction`. Keep the parameter count small.

4. **Fit a simple power curve**
   Start with a regression of baseline-subtracted compressor watts against exhaust temperature
   and wet-bulb/coil proxy. This tests the "power is temperature-driven" hypothesis directly.

5. **Fit a simple COP or energy-to-target model**
   With enough clean cycles, decide whether the first useful model is:
   - COP/power step model, or
   - direct empirical `duration/kWh to target` model.

6. **Build the useful-inventory observer**
   Estimate `E_t` from compressor runtime, energy input, draw-off, standing loss, and simple
   probe-event corrections. Keep physical probe `60 C` events as the target-satisfaction source
   of truth until the observer is validated.

7. **Prototype a shadow MILP/MPC planner**
   Optimise useful inventory over a 48 h horizon with 5 min price steps, binary compressor
   action, start penalty, reserve/floor constraints, daily target deadline, and demand-based
   terminal inventory.

8. **Shadow-publish inventory/MILP diagnostics**
   Publish only the new inventory/MILP shadow once it exists. The old DP and stratified live
   shadows are retired; offline stratified validation remains for observer design.

## Review Questions

For constructive adversarial review:

1. Is the two-layer/hot-fraction state too simple, or is it the right first abstraction?
2. Is `exhaust_temperature` a defensible condensing-temperature proxy for compressor power/COP,
   or should we model it only as an observed diagnostic?
3. Is the observed control-probe rise timing enough to fit `probe_height_fraction` and
   `thermocline_width_fraction`, or do we need a different parameterisation?
4. Should the first production improvement be better inventory estimation only, or should it
   also change scheduling once shadow error is acceptable?
5. Is a direct empirical `duration/kWh to target` model more robust than a physical COP model
   at the current sample size?
6. What operating regimes should be excluded from calibration: fan-speed changes, defrost,
   element use, overlapping household loads, unusual draw-offs, or autonomous safety cycles?
7. What is the minimum useful dataset before changing live scheduling: five clean cycles,
   ten, twenty, or enough to cover multiple wet-bulb/inlet-temperature regimes?

## Guardrails

- Keep live execution conservative until shadow predictions are demonstrably better.
- Do not tune many parameters against five clean cycles.
- Preserve raw-ish cycle summaries in `data/hwc_cop_cycles.csv` so future fits are reproducible.
- Prefer small, reviewable changes: extractor -> observer -> validator -> MILP shadow ->
  planner integration.
- Keep the owner-visible HA charts as the arbiter of whether the model is improving behaviour.
