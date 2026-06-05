# HWC Dynamic-Programming Planner Design

## Caveman Summary

- Goal: replace heuristic block picker with whole-horizon optimiser.
- Mode: shadow first; live executor stays on current block planner.
- V1 model: current single-node HWC thermal transition, 5 or 10 min internal grid.
- V1 output: publish DP shadow temp/power/cost series to HA.
- Optimises: price, wet-bulb recovery, standing loss, terminal reserve, starts.
- Constraints: 45 C floor; daily 60 C target unless already satisfied.
- Short-cycle guard: start penalty plus minimum useful lift penalty, not minimum duration.
- Later: swap DP transition from single-node to validated stratified model.

## Problem

The current `hwc_planner.py` block planner is a sequence of local heuristics:

- choose a daily main block;
- repair minimum-temperature violations;
- repair terminal temperature.

This works as an MVP, but it is not a global optimiser. It can prefer an earlier run because
one 30-minute slot disappears, while not explicitly valuing:

- later warmer wet-bulb temperature;
- less time storing hot water before evening/night;
- warmer tank state later in the horizon;
- the interaction between today's run and tomorrow's reserve;
- start count and small top-up quality in one objective.

The next planner should optimise the full trajectory over the whole forecast horizon.

## Scope

V1 is deliberately narrow:

- deterministic forecast only;
- no forecast uncertainty;
- no battery co-optimisation;
- no direct EMHASS dependency;
- no live actuation from DP until shadow behaviour is reviewed.

The first implementation should prove the optimiser architecture, not solve every physical
model issue at once.

## Inputs

Use the same live inputs as the block planner:

- current tank/control-probe temperature;
- 30-minute import price forecast from `sensor.ai_dh_import_price_forecast`;
- weather forecast from `weather.woodville_west_hourly`;
- smoothed wet-bulb forecast;
- draw-off profile from config;
- daemon state `last_reached_target_at`.

Internal resampling:

- interpolate price onto DP grid;
- interpolate draw-off energy onto DP grid preserving total kWh;
- interpolate/smooth wet-bulb onto DP grid;
- publish output back on the same or coarser grid expected by HA charts.

Initial DP grid recommendation:

- `internal_step_minutes: 10` first;
- evaluate 5 minutes if 10-minute quantisation still affects decisions;
- keep published HA series at current forecast cadence unless charting needs finer detail.

## State

V1 state should be small enough for exhaustive dynamic programming:

```text
state = (
  temp_bin,
  prev_action,
  run_start_temp_bin_or_none,
  local_day_index,
  target_satisfied_today
)
```

Definitions:

- `temp_bin`: discretised control-probe temperature, e.g. 0.25 C or 0.5 C bins.
- `prev_action`: compressor off/on on previous step; needed for start penalty.
- `run_start_temp_bin_or_none`: temperature at the start of current compressor run. Used to
  penalise stopping after a tiny lift.
- `local_day_index`: derived from timestep; mainly for day-boundary target checks.
- `target_satisfied_today`: true if the simulated trajectory has reached `desired_temp`
  during the current local day, or if daemon state says the target was already reached today.

V1 should start with 0.5 C bins unless candidate plans are too coarse. Range can be bounded:

```text
min_state_temp = 35 C
max_state_temp = 60 C
```

Values below the 45 C operating floor can exist in the state space only so the optimiser can
assign a large violation penalty and recover if needed.

## Actions

```text
action in {off, heat}
```

Heat means fixed-speed heat-pump operation at `nominal_power_w`.

No variable setpoint optimisation in V1. The executor can continue using planned block end /
setpoint policy after promotion. For shadow output, heat is just planned compressor power.

## Transition

V1 transition uses the current single-node simulation logic:

```text
next_temp = f(
  temp,
  action,
  wet_bulb,
  dry_bulb_or_ambient,
  draw_off_kwh,
  standing_loss,
  heat_rate_model
)
```

This keeps optimiser behaviour testable against the existing planner. Later, replace this
transition with a validated stratified state transition.

Important details:

- clamp heating at `max_temp`;
- apply draw-off and standing loss consistently with current `simulate_block_temperatures`;
- preserve the current smoothed wet-bulb recovery-speed adjustment;
- record if `next_temp >= desired_temp` to satisfy the daily target.

## Objective

Minimise total cost:

```text
total =
  energy_cost
  + start_penalty
  + tiny_lift_penalty
  + floor_violation_penalty
  + daily_target_miss_penalty
  + terminal_reserve_penalty
```

### Energy Cost

```text
energy_cost = action_heat * nominal_power_kw * import_price * step_hours
```

This naturally captures price timing. Wet-bulb affects cost indirectly through recovery speed
and therefore future state/action choices.

### Start Penalty

Purpose: reflect wear and efficiency loss from fragmented operation.

Initial config:

```json
"start_penalty_aud": 0.05
```

Treat this as a tunable policy parameter, not a measured physical cost yet.

### Tiny-Lift Penalty

User preference: minimum useful lift is a better guardrail than minimum duration.

When transitioning from `heat` to `off`, compute:

```text
run_lift = current_temp - run_start_temp
```

If `run_lift < min_block_lift_c`, add a high but not infinite penalty.

Initial config:

```json
"min_block_lift_c": 3.0
"tiny_lift_penalty_aud": 1.00
```

This allows emergency/floor recovery if needed but makes 59 -> 60 C top-ups unattractive.

### Floor Violation Penalty

The 45 C floor should be treated as a hard safety/comfort constraint in normal cases.

```text
floor_shortfall = max(0, min_temp - temp)
penalty = floor_shortfall^2 * floor_penalty_aud_per_c2
```

Initial value should be large enough that violating the floor is dominated by heating cost.

### Daily 60 C Target

Current policy: keep the daily 60 C target because it matches factory behaviour.

At each local day boundary, if the day is not target-satisfied, add a large penalty.

For the first local day in the horizon:

- if `last_reached_target_at` maps to today, initialise `target_satisfied_today = true`;
- otherwise false.

At the final horizon boundary, also penalise a missed target for the active local day if enough
of that day lies inside the planning horizon to make a fair decision. This avoids forcing an
impossible target for a partial tail day.

### Terminal Reserve

This replaces the current terminal repair pass.

The terminal value should represent useful reserve, not an arbitrary extra block:

```text
terminal_target_temp = current_temp or configured reserve target
terminal_shortfall = max(0, terminal_target_temp - terminal_temp)
terminal_penalty = terminal_shortfall^2 * terminal_penalty_aud_per_c2
```

Open decision: whether terminal target should be current temperature, morning-shower reserve,
or enough useful energy under the future stratified model.

## Algorithm

Use standard dynamic programming on the discretised state graph.

Forward form is easiest to implement/debug:

```text
frontier = {initial_state: cost 0}
for each timestep:
  next_frontier = {}
  for each state in frontier:
    for action in {off, heat}:
      next_state = transition(state, action, inputs[t])
      step_cost = objective_increment(state, action, next_state, inputs[t])
      keep cheapest predecessor for next_state
  frontier = prune/merge by state
choose best terminal state by total cost + terminal penalty
backtrack actions
```

Merging by discretised state gives the global optimum for the chosen model/grid/objective.

Expected complexity is small:

```text
72 h at 10 min = 432 steps
35-60 C at 0.5 C = 51 temp bins
actions = 2
extra state bits small
```

Even with run-start temperature bins, this should be comfortably cheap.

## Publish Shape

Do not replace production entities initially.

Add DP shadow entities:

- `sensor.hwc_dp_predicted_temp`
- `sensor.hwc_dp_power_plan`
- `sensor.hwc_dp_plan_cost`
- optional `sensor.hwc_dp_objective_breakdown`

Attributes:

- `predicted_temperatures`: same shape as `sensor.hwc_predicted_temp`;
- `deferrables_schedule`: same shape as `sensor.hwc_power_plan`;
- `objective_breakdown`: total energy cost, start penalty, floor penalty, target penalty,
  terminal penalty;
- `planner_role: "shadow"`;
- `internal_step_minutes`;
- `temp_bin_c`;
- `generated_at`.

Charting:

- add DP power/temp as optional hidden-by-default or low-opacity series;
- keep current block planner visible until DP is trusted.

## Validation

Unit tests:

- transition matches current single-node simulator for equivalent actions;
- daily target state initialises from `last_reached_target_at`;
- floor violation is avoided when possible;
- start penalty reduces fragmentation;
- tiny-lift penalty avoids 59 -> 60 C starts when no safety need exists;
- terminal reserve affects end state without separate repair pass.

Shadow/live checks:

- compare current block plan vs DP shadow plan on the same inputs;
- store snapshots for interesting disagreements;
- inspect candidate reasons: energy cost, starts, terminal reserve, target penalty;
- compare forecast temperature to measured tank temperature after each real cycle.

Promotion gate:

- DP plan is explainable in HA charts for several days;
- no floor-risk behaviour;
- no repeated tiny cycles;
- daily 60 C policy remains satisfied;
- owner explicitly approves executor switch.

## Implementation Steps

1. Add `hwc_dp_planner.py` with pure transition/objective/solve helpers.
2. Add unit tests with tiny horizons and deterministic prices/weather.
3. Add config under `hwc.dp_planner`, default disabled/shadow-only.
4. Add `build_dp_shadow_plan(...)` call from `hwc_planner.run` when enabled.
5. Publish DP shadow entities from `_publish_block_plan` or a separate publisher.
6. Update ApexCharts sample with optional DP series.
7. Run live shadow for several days before promotion discussion.

## Open Decisions

- Internal timestep: 10 min first, or go straight to 5 min?
- Temperature bin width: 0.5 C first, or 0.25 C?
- Start penalty value: what AUD-equivalent wear/efficiency penalty is reasonable?
- Tiny-lift penalty: hard rejection except emergency, or soft penalty?
- Terminal target: current temperature, morning reserve, or configured fixed target?
- Daily target at horizon tail: how much partial-day coverage is enough to require it?
- Should DP consume block planner's smoothed wet-bulb series exactly, or own all weather
  preprocessing?

## Non-Goals For V1

- Full physical COP model.
- Compressor power curve from exhaust temperature.
- Stratified execution state.
- Stochastic/robust optimisation over price/weather uncertainty.
- Battery/HWC co-optimisation.
- Changing the live executor to DP output.
