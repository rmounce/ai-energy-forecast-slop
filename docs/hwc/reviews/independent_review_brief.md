# Heat-Pump Hot Water Control: Independent Review Brief

## Purpose

We want independent advice on how best to model and control a domestic heat-pump hot-water
system against variable electricity prices and weather-dependent efficiency.

The desired output from the review is not code. It is a recommended modelling and control
approach, including state representation, optimisation method, validation plan, and the most
valuable extra data to collect.

This brief intentionally avoids describing implementation approaches already tried in this
repository, so the review can start from the underlying control problem.

## System To Control

- Unit: Aquatech RAPID X6 heat-pump hot-water system.
- Tank volume: approximately `225 L`.
- Heat source: fixed-speed heat-pump compressor, not an inverter compressor.
- Typical compressor electrical draw: roughly `700-800 W`, rising during a heating cycle.
- Useful control range:
  - comfort/safety floor currently treated as about `45 C`;
  - normal upper cut-off is `60 C`;
  - current policy keeps a daily `60 C` target because this matches the factory behaviour.
- The unit can be controlled through Home Assistant:
  - set operation mode, for example `off` or heat-pump mode;
  - set target temperature up to `60 C`;
  - observe compressor state.
- In practice, switching from `off` to heat-pump mode appears to allow starting a reheat even
  when the tank is already well above the datasheet reheat-trigger threshold.

## Available Signals

Home Assistant / InfluxDB provide:

- current tank/control-probe temperature;
- compressor on/off state;
- several unit telemetry temperatures, including a value believed to be compressor discharge
  or condenser-related temperature;
- unit ambient/air-side temperatures;
- electricity import price forecasts:
  - short-term 5-minute prices;
  - longer-horizon 30-minute prices;
- weather forecasts, from which wet-bulb temperature can be estimated;
- household-level load data that can be used as a proxy for compressor electrical power on
  clean cycles.

The unit does not currently provide a reliable dedicated electrical power measurement.

## Observed Physical Behaviour

The tank appears materially stratified during reheat.

Observed cycles commonly show:

- the main tank/control probe remains nearly flat for a substantial early part of a cycle;
- the condenser/discharge-related temperature rises while the control probe is still flat;
- later, the control probe rises as the thermocline reaches it;
- compressor electrical power rises smoothly through the cycle, consistent with a fixed-speed
  compressor working against increasing condensing temperature;
- small near-target top-ups appear less efficient than longer reheats from a cooler state.

Clean historical cycles are still limited, but current estimates suggest:

- full reheats to `60 C` have COP roughly in the `2.4-3.0` range under observed conditions;
- the late `55 C -> 60 C` portion is materially less efficient, with apparent COP around `1.75-2`;
- power and efficiency likely depend on both tank state and air-side condition, with wet-bulb
  temperature likely more relevant than dry-bulb alone for evaporator performance.

These figures should be treated as indicative rather than final calibration.

## Demand And Operating Assumptions

- Main expected hot-water draw is a morning shower, currently represented as about `1.3 kWh`
  of thermal draw-off around `08:00-09:00`.
- Actual showering is fairly regular but not guaranteed every day.
- The owner values having enough hot water for a morning shower contingency.
- Avoiding very short compressor cycles is a goal, both for efficiency and wear.
- The schedule should use low-price periods when sensible, but not at the expense of obviously
  poor physical operation.
- The system should remain robust if forecasts change or the actual tank state diverges from
  the predicted state.

## Control Objectives

Primary objectives:

- keep hot-water availability above a practical comfort/safety floor;
- satisfy the retained daily `60 C` target policy;
- minimise electricity cost using time-varying import prices;
- avoid nuisance short cycling and inefficient near-target top-ups;
- exploit better heat-pump efficiency when weather conditions are favourable;
- avoid storing excess heat too early if later heating would be cheaper or more efficient.

Secondary objectives:

- produce plans that are understandable and diagnosable in Home Assistant charts;
- support frequent replanning as forecasts and observed tank state change;
- remain computationally cheap enough for a small always-on home server;
- allow staged improvement as more measured cycles become available.

## Forecast Horizon And Resolution

A practical target is:

- 5-minute resolution where 5-minute price forecasts are available;
- a lower-resolution tail is acceptable if needed, but a uniform 5-minute grid over about
  `48 h` would be acceptable if computationally cheap;
- `48 h` horizon is likely enough to capture daily cycles and terminal inventory effects;
- any approach should define a clear terminal-state policy so the end of the horizon does not
  create artificial behaviour.

## Key Modelling Questions

1. What state representation is appropriate for planning?
   - Single effective temperature?
   - Two-layer or stratified tank state?
   - Energy inventory plus probe-temperature observation model?
   - Empirical cycle-to-target model?

2. How should the control objective represent compressor starts?
   - Explicit start cost?
   - Minimum lift/runtime constraints?
   - Efficiency penalty for near-target top-ups?

3. How should the daily `60 C` target be represented?
   - Hard constraint based on last time the tank reached `60 C`?
   - Soft penalty?
   - Constraint on observed probe temperature, estimated tank state, or useful stored energy?

4. How should heat-pump efficiency be modelled?
   - COP as a function of wet-bulb temperature and tank state?
   - Electrical energy/duration to target as a direct empirical model?
   - Compressor power as a function of condensing/discharge proxy and evaporator condition?

5. How much stratification detail is worth modelling for control?
   - Does a two-layer/thermocline model improve decisions enough to justify added complexity?
   - What parameters are identifiable from current telemetry?

6. What optimisation method is most suitable?
   - Deterministic dynamic programming?
   - Mixed-integer or linear/nonlinear programming?
   - Model predictive control with a compact empirical transition model?
   - Heuristic block search with strong validation?

7. How should uncertainty be handled?
   - Forecast price changes;
   - weather forecast error;
   - shower timing/size variability;
   - mismatch between control-probe temperature and true stored useful heat.

## Data And Validation Questions

We would like advice on:

- the minimum dataset needed before trusting a more sophisticated model;
- whether a dedicated circuit power meter is the highest-value instrumentation upgrade;
- which cycle features should be extracted and stored permanently for retuning;
- how to validate a model offline against historical cycles without overfitting;
- which operating regimes should be excluded from calibration, such as defrost, resistive
  element use, fan-speed changes, overlapping household loads, or unusual draw-off events;
- how to quantify whether a proposed optimiser is better than a fixed timer in practice.

## Desired Review Output

Please recommend:

- a first-principles or empirical model structure suitable for this system;
- the optimisation/control approach you would use;
- the constraints and objective terms you would include initially;
- what telemetry should be collected next;
- how you would validate the model and planner before allowing it to control the unit;
- which parts should be kept deliberately simple until more data exists.
