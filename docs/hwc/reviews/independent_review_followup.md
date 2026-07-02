# Follow-Up To Independent HWC Review

Thanks for the review. The energy-inventory / SoC framing is useful and broadly matches the
direction we were converging on: the control probe is not the true state, and managing useful
stored heat is the real control problem.

I would like to push on a few assumptions before turning this into an implementation plan.

## 1. SoC Is The Right State, But It May Be Hard To Observe

I agree that probe temperature alone is a poor state variable. The concern is that SoC is also
not directly observed.

With only one main probe, delayed probe response, and no dedicated power meter yet, the SoC
estimate will be a latent state inferred from:

- previous compressor runtime;
- assumed thermal input;
- assumed draw-off;
- standing losses;
- occasional probe-temperature updates.

That suggests we may need an observer/state-estimator as part of the model, not just a planner
state. Otherwise the optimiser may make precise-looking decisions from a poorly identified SoC.

Question: would you explicitly implement a simple state observer, for example a Kalman-style
filter, particle filter, or deterministic correction rule, or would you keep this as a direct
empirical mapping from recent history to SoC?

## 2. Two-Layer Hot/Cold Model: What Are The Layer Temperatures?

The proposed two-layer abstraction is attractive, but "water is either at 60 C or mains
temperature" seems too stark for this use case.

In normal operation the lower/cooler part of the tank may often be in the `40-55 C` range, not
mains temperature. After partial reheats, standing losses, and incomplete draw-offs, there may
also be no clean binary split between `60 C` and cold inlet water.

Question: would you model:

- hot volume at approximately target temperature plus cold volume at mains temperature;
- hot/cold layer temperatures as separate state variables;
- a single energy inventory with a probe-observation model;
- or piecewise empirical cycle bands without explicit layer temperatures?

The answer matters because the current unit behaviour is governed by both useful stored heat
and compressor efficiency as the condensing temperature rises.

## 3. Daily 60 C Target: Probe Constraint Or SoC Constraint?

You suggested `SoC >= 98% for at least one timestep per rolling 24-hour window`.

I am cautious about replacing "the unit/control probe reached 60 C" with an estimated SoC
threshold. The factory behaviour and observed shut-off are based on the unit's own sensor, not
our inferred inventory. If we use an SoC proxy for the daily 60 C target, we may satisfy the
optimiser while failing the physical unit's safety/control logic.

My current bias is:

- comfort/reserve constraints can be SoC/useful-energy based;
- the retained daily `60 C` target should remain anchored to the observed/control probe reaching
  `60 C`, at least until the SoC estimator is well validated.

Question: do you think that distinction is sensible, or would you still convert the `60 C`
requirement entirely into an inventory constraint?

## 4. Minimum Runtime Versus Minimum Useful Lift

You recommended a minimum runtime such as 45 minutes.

We have been leaning toward a minimum useful lift / minimum useful energy rule instead. The
reason is that runtime is weather- and state-dependent: 45 minutes from a cold, efficient state
may be useful, while 45 minutes near the top of the tank may mostly force the inefficient tail.

For wear, an explicit start penalty is also attractive because it lets the optimiser trade a
short cycle against real price/safety pressure rather than forbidding it entirely.

Question: in an MILP formulation, would you prefer:

- hard minimum runtime;
- hard minimum energy/lift per start;
- start penalty only;
- or a combination, and why?

## 5. MILP Is Plausible, But The Model Must Stay Linear Enough

MILP is appealing for binary compressor dispatch and operational constraints. The pushback is
that the physically interesting parts are nonlinear or at least state-dependent:

- COP depends on wet-bulb and tank state;
- power rises during the cycle;
- probe temperature is a nonlinear observation of stratified inventory;
- draw-off changes layer composition.

These can be approximated with piecewise-linear bands, but the formulation could become complex
quickly. A dynamic-programming or graph-search approach over discrete SoC bands might be more
direct for a low-dimensional empirical model, even if MILP is cleaner for scheduling logic.

Question: what would your minimal MILP formulation look like? In particular:

- what are the state variables?
- how many SoC/COP bands?
- how is band selection linearised?
- how is terminal inventory enforced?
- how is the probe-temperature `60 C` event represented?

## 6. Terminal Constraint Should Probably Be Demand-Based

A fixed terminal `SoC >= 50%` is a useful placeholder, but it feels arbitrary.

The terminal state should probably relate to future serviceability, such as:

- enough useful hot water for the next expected morning shower;
- not materially below the current inventory after a full diurnal horizon;
- or a penalty against ending below a rolling baseline rather than a hard fixed threshold.

Question: how would you choose the terminal inventory constraint so it does not create artefacts
at the end of the 48-hour horizon?

## 7. Validation Metric Needs Comfort And Safety Terms

`$/kWh_thermal delivered` is a good efficiency metric, but it does not fully describe success.

A bad controller can score well on cost per delivered kWh by simply under-delivering heat or
cutting margins too fine. We likely need validation metrics such as:

- floor violations or near-violations;
- missed morning reserve;
- missed daily `60 C` target;
- number of compressor starts;
- cost versus fixed timer;
- forecast robustness after replanning.

Question: would you use a single scalar score for model comparison, or keep these as separate
acceptance criteria?

## 8. Instrumentation: Strong Agreement

I agree that a dedicated HWC power meter is likely the highest-value instrumentation upgrade.

Without it, model fitting is constrained by whole-house-load contamination and a small number
of clean cycles. A circuit meter would immediately improve:

- cycle energy estimates;
- compressor power curve fitting;
- COP estimates;
- detection of actual starts/stops;
- validation of optimiser cost.

## 9. Staged Rollout

I agree with the staged rollout principle: simple control first, data collection, shadow model,
then live control.

The only caveat is that a purely "cheapest 2-3 hour block between 10:00 and 16:00" timer may
miss cases where:

- the cheapest price is outside that window;
- wet-bulb efficiency materially improves later;
- the tank can safely defer heating;
- terminal inventory has value.

As a baseline it is fine. As a controller, it should probably still evaluate price, weather,
current inventory, expected draw, and daily target satisfaction.

## What I Would Like From A Second-Pass Recommendation

Please sketch the simplest production-worthy model and optimiser you would implement first,
with enough mathematical detail to distinguish it from alternatives:

- state variables;
- observation/update rule from telemetry;
- action variables;
- objective terms;
- hard constraints;
- terminal constraint;
- how COP/power bands are represented;
- how the daily `60 C` target is represented;
- minimum viable calibration data;
- expected computational cost for a `48 h` horizon with `5 min` price resolution.
