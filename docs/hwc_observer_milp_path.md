# HWC Observer + MILP Path

## Caveman Summary

- Current live baseline: keep the conservative block planner/executor.
- New direction: split model into **observer** and **planner**.
- Observer estimates useful thermal inventory `E_t` in kWh above the `45 C` floor.
- Planner should optimise inventory, not probe temperature or detailed stratification.
- First production-grade optimiser candidate: 48 h MILP MPC with 5 min price steps.
- Daily `60 C` target remains anchored to the physical probe until the observer is validated.
- Dedicated HWC circuit power meter is the highest-value instrumentation upgrade.
- Retire old live DP/stratified shadows; keep offline stratified tools for calibration insight.

## Decision

Adopt the independent-review architecture as the next serious modelling path:

```text
telemetry -> observer/state estimator -> inventory state E_t -> optimiser -> planned compressor action
```

The optimiser should not try to model thermocline shape directly. Stratification still matters,
but mainly through:

- the observation problem: probe temperature is not true stored energy;
- the efficiency problem: compressor power/COP worsens as tank inventory approaches full;
- the safety/control problem: the unit's own probe reaching `60 C` is the event we can trust.

## Why This Replaces The DP Shadow Path

The recent DP shadow used a single-node probe-temperature transition. That exposed useful
questions, but it is not the right long-term state:

- probe temperature is a poor inventory measure during stratified charging;
- fine temperature bins made the DP slow and memory-heavy;
- coarse bins created numerical artefacts in standing-loss dynamics;
- adding more heuristics around starts, terminal state, and daily targets risks rebuilding an
  opaque policy rather than a modelled controller.

Dynamic programming is not ruled out forever, but the next implementation should first build
a better state variable: useful thermal inventory. Once that exists, MILP MPC is the preferred
candidate because binary compressor dispatch, start penalties, reserve constraints, and terminal
inventory constraints are natural to express.

## Live Baseline

Keep the current block planner and executor as the stable controller while the new model is
developed. It is understandable and already has live operational feedback.

The block planner should remain conservative:

- avoid small near-target starts;
- keep the `45 C` floor;
- retain the daily `60 C` target policy;
- publish measured/predicted temperature, planned power, import-price context, and wet-bulb
  forecast for review.

## Observer

The observer maintains `E_t`, useful thermal inventory in kWh above the `45 C` floor.

Initial open-loop update:

```text
E_now = E_prev + Q_in - Q_draw - Q_loss
```

Where:

- `Q_in` comes from compressor runtime, power, and COP/energy model;
- `Q_draw` comes from expected or observed draw-off assumptions;
- `Q_loss` comes from standing-loss model;
- `E_t` is clamped to `[0, E_max]`.

Probe-based corrections should be deliberately simple at first:

- if the physical probe reaches `60 C` and the unit shuts off, set `E_now = E_max`;
- if the probe falls in a way that clearly indicates the thermocline has passed the probe,
  correct toward the calibrated inventory at probe height;
- otherwise do not over-trust single probe readings.

Until this observer is validated, `last_reached_60C` from the physical probe remains the source
of truth for daily target satisfaction.

## Calibration Data

Keep extracting and storing clean cycle summaries. The cycle table should support retuning
without repeatedly scraping raw Home Assistant history.

Highest-value next measurement:

- install a dedicated HWC circuit power meter.

Useful calibration fields:

- cycle start/end time;
- compressor runtime;
- electrical kWh;
- start/end probe temperature;
- whether the physical probe reached `60 C`;
- wet-bulb and relevant unit air-side temperatures;
- compressor power shape, once metered;
- defrost, resistive-element, and unusual draw-off flags.

Current whole-house-load proxy cycles are useful but should be treated as provisional.

## Planner Candidate

Prototype a shadow-only MILP MPC once the observer scaffold exists.

Target shape:

- horizon: about `48 h`;
- resolution: 5 min while 5 min prices are available, with a coarser tail only if needed;
- action: compressor on/off;
- state: continuous useful inventory `E_t`;
- binary variables: compressor on, start event, heating band if using piecewise COP;
- objective: energy cost + start penalty + soft floor/reserve violations;
- constraints: inventory dynamics, hardware-safe short start guard, daily target deadline,
  terminal reserve.

Initial physical simplification:

- two or three heating bands by inventory fraction;
- each band has empirical power and COP/recovery parameters as a function of wet-bulb;
- top band represents the inefficient near-`60 C` tail.

## Daily 60 C Target

Keep the policy for now: reach `60 C` daily.

Implementation rule during transition:

- observer/planner may predict inventory reaching `E_max`;
- the executor/daemon records actual physical probe `60 C` events;
- daily target satisfaction is based on the physical event until the observer is proven reliable.

The exact deadline can remain local-day based rather than a strict rolling-24h constraint unless
there is a reason to match rolling factory logic.

## Terminal Reserve

Avoid arbitrary fixed terminal inventory such as `50%`.

Prefer a demand-based terminal policy:

- enough useful inventory for expected draws shortly beyond the horizon;
- plus a comfort/safety reserve;
- or a soft penalty for ending materially below the starting/current inventory over a full
  diurnal horizon.

The terminal policy should be explicit because horizon-end artefacts have already produced bad
planner behaviour.

## Validation Gate

Keep model validation multi-criteria, not just one scalar objective.

Before any MILP shadow can control the unit, require:

- no missed comfort floor in replay/shadow;
- no missed physical daily `60 C` target;
- sensible start count and no nuisance near-target cycling;
- predicted inventory/probe behaviour matches observed cycles well enough for the intended
  decisions;
- lower simulated cost than a fixed timer or current block baseline under the same service
  constraints.

## Cleanup Consequence

The old live DP shadow and live stratified shadow are retired from the planner path.

Offline stratified modelling remains useful for understanding probe lag and for designing the
observer correction rules, but it should not be published as a competing live controller-shaped
forecast.
