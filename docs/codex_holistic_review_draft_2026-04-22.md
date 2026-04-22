# Codex Holistic Review Draft — 2026-04-22

Purpose: record a system-level review of the repo and current decision space, with the
current production contract treated as evidence rather than a protected constraint.

Primary objective:
- maximize real live financial performance of the battery system

Secondary objective:
- if possible, maintain a credible path that does not depend on Amber APF as a core input

This draft is grounded in:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/data_sources.md](./data_sources.md)
- [docs/roadmap.md](./roadmap.md)
- [eval/README.md](../eval/README.md)
- [docs/training_runs.md](./training_runs.md)
- [forecast.py](../forecast.py)
- [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

---

## 1. Executive View

My current view is:

- the repo is directionally right that the control problem has both a tactical and a strategic
  component
- but the current formulation appears to over-invest in producing a detailed `72h` point price
  path, and under-invest in defining the right contract between long-horizon value and
  short-horizon execution
- the strongest near-term evidence does **not** yet say "replace the whole architecture"
- it does suggest that the current system may be sitting in a local optimum where a
  better forecast vector is being asked to compensate for a not-yet-optimal control contract

If I were backing a future architecture today, I would **keep the two-timescale idea** but
would change what the strategic layer is asked to produce.

I would not treat:
- the current `30m / 72h` point forecast as sacred
- the current `14h × 5m` tactical boundary contract as complete
- the current emphasis on TFT iteration as the main remaining lever

I would treat the most important system question as:

How should downstream value of stored energy be represented and handed to the tactical
controller under the actual arrival times of confirmed price, P5MIN, PREDISPATCH, PD7Day,
and SevenDayOutlook?

---

## 2. What Looks Directionally Correct

Several repo conclusions still look sound to me.

### 2.1 Financial eval matters more than forecast error alone

This seems clearly right. The repo is already steering toward revenue and dispatch outcomes,
and that is the correct headline objective for this application.

### 2.2 A tactical short-horizon layer is real, not optional

The tactical layer is justified by the data environment itself:
- confirmed current interval price
- frequent P5MIN updates
- fast-changing local load / PV state

That means a high-refresh execution layer is not merely a convenience. It is structurally
required by the information pattern of the problem.

### 2.3 Long-horizon awareness is also real

The handoff analysis strongly suggests the tactical layer should not act as if the world ends
at `14h`. The strategic handoff improved the rolling result materially, especially on spike
days. So the answer is not "forget the long horizon." The answer is "represent it better."

### 2.4 Amber independence is plausible

The repo already has the raw ingredients for a credible Amber-independent stack:
- confirmed current dispatch actuals
- P5MIN
- PREDISPATCH
- PD7Day
- SevenDayOutlook
- local PV / load / weather

I do not see evidence that Amber APF is irreplaceable in principle. I do see evidence that
it is a strong benchmark and a useful transition input while the AEMO-native architecture is
still being validated.

---

## 3. What I Think Is Most Likely A Local Optimum

### 3.1 Treating the `72h` point forecast as the main product

This is my strongest architectural concern.

The repo currently invests significant effort in improving a detailed `72h` forecast path,
yet the tactical controller only executes a short prefix before replanning many times.

That does not make the long horizon irrelevant. It does suggest that the long horizon may be
the wrong object to optimize as a detailed executable trajectory.

For a battery controller, the strategic layer may not need to hand down:
- a precise `72h` price path

It may instead need to hand down:
- a future inventory target or target band
- a downstream value-of-energy signal
- a scarcity / reserve posture
- a quantile tilt or risk posture for the tactical path

In other words, the present system may be over-specifying the wrong object.

### 3.2 Using forecast-layer iteration to compensate for control-contract weakness

The roadmap and rolling eval history suggest an important pattern:
- long-horizon information matters
- static terminal-value surrogates helped
- restoring the strategic handoff also helped
- residual weakness remains on low / normal days

That is exactly the pattern I would expect if the controller contract is still incomplete.

I do not think the repo has yet ruled out the possibility that a meaningful share of the
remaining gap comes from how long-horizon value is expressed, not from forecast quality alone.

### 3.3 Optimizing a training objective that is only partially aligned with the live problem

The current TFT training objective and the current eval stack are asking for somewhat different
things. The repo already knows this to some extent.

My broader concern is not just one metric choice. It is that the system may still be using
"forecast quality over a long vector" as a proxy for "quality of control-relevant information."

Those are not always the same.

### 3.4 Assuming the strategic layer must itself be a rich ML forecaster

I do not think this has been established.

Given the current data maturity, especially beyond the PREDISPATCH horizon, a simpler strategic
module might outperform a more ambitious long-horizon predictor if its output contract is
closer to what the tactical controller actually needs.

Possible simpler strategic outputs:
- bias-corrected raw PREDISPATCH / PD7Day
- reserve heuristics conditioned on strategic scarcity
- a learned or semi-learned water-value / terminal-value function
- target-band forecasts rather than full path forecasts

---

## 4. What Architecture I Would Back Today

If I were redesigning from scratch under the same external data constraints, I would back a
three-layer system, but not the repo's current version of that three-layer system.

### 4.1 Layer A: execution-facing tactical controller

Purpose:
- make the actual `5-minute` decisions

Data:
- confirmed current interval price
- P5MIN
- freshest local load / PV / weather
- short tactical forecast path

Horizon:
- likely still around `2h` to `14h`, but I would not assume `14h` is optimal without testing

Output:
- direct dispatch plan over the tactical horizon

Key design point:
- this layer should receive an explicit downstream inventory signal from the strategic layer
- it should not be expected to infer all downstream value purely from a median price vector

### 4.2 Layer B: bridge / opportunity contract

Purpose:
- translate strategic value into tactical constraints or posture

This is the layer I think is currently underdeveloped.

Candidate outputs:
- terminal SoC target
- terminal SoC floor / band
- opportunity-cost scalar
- quantile blend weight
- reserve posture score

My current preference is **not** a single hard target alone. I would prefer a richer but still
simple bridge contract such as:

- terminal SoC target or target band
- plus a posture signal indicating how conservative the tactical controller should be with
  stored energy

That makes the strategic layer responsible for "how valuable inventory is later," while the
tactical layer remains responsible for local execution timing.

### 4.3 Layer C: strategic planner

Purpose:
- look beyond the tactical horizon and value future inventory

Horizon:
- probably still `24h` to `72h`

But the strategic layer's job should not necessarily be:
- emit a perfect full-path `72h` dispatch-quality price series

I would instead ask it to solve:
- how much energy should be preserved or accumulated for later windows
- how steep is downstream value-of-energy
- how confident is that posture

This can still use ML, but I would no longer make "best full 72h point forecast" the primary
success criterion.

---

## 5. Stochasticity And Uncertainty

One natural question is whether the future system should be made more explicitly stochastic.

My view is:

- some increase in stochasticity is probably desirable
- but a fully stochastic controller is not the first implementation I would reach for here

The current pipeline is mostly:
- deterministic optimization on a single forecast path

That is simple, but it leaves real information on the table. For this battery problem, the
future value of stored energy is asymmetric and uncertain. A controller that behaves as if the
median path is "the future" is probably too certain for the true economics of the problem.

### 5.1 What "more stochastic" could mean

There are several different versions of this idea.

#### Light version: risk-aware / distribution-aware control

Examples:
- strategic layer uses quantiles or quantile spreads to derive a reserve posture
- tactical layer blends away from pure `q50` when future inventory value is uncertain
- bridge contract carries both a target and a posture signal

This is the most practical near-term direction. It keeps the optimizer simple while making it
less brittle to the false precision of a single median path.

#### Medium version: scenario-lite strategic valuation

Examples:
- solve the strategic LP on `q50` and `q90`
- compare handover value-of-energy under those cases
- derive a scalar posture or opportunity-cost signal from the spread

This does **not** create a fully stochastic controller, but it can still be useful. It gives
the tactical layer a sense of downstream uncertainty without requiring full scenario trees.

#### Heavy version: full scenario-based stochastic optimization

Examples:
- generate coherent future price scenarios with probabilities
- optimize expected value and/or a risk measure across those scenarios
- choose tactical actions that are robust to distributional uncertainty

This is feasible in principle, but I would not currently make it the default recommendation for
this repo because the scenario-generation problem is harder than the optimization problem.

### 5.2 Why I would not jump straight to full stochastic programming

Three reasons matter here.

#### Quantiles are not the same as coherent scenarios

The current quantile forecasts are marginal statements about each step, not validated pathwise
samples of the future. A deterministic LP run on a full `q10` or `q90` path can still be a
useful heuristic, but it should not be mistaken for a true stochastic program.

#### The current lower tail is less trustworthy than the upper tail

Repo documentation already treats the upper tail as more usable than the lower tail for
risk-aware control. That means a first stochastic-style implementation should lean more on:
- `q50`
- `q90` or `q95`

than on symmetric `q10/q50/q90` constructions.

#### The missing piece still looks more like a bridge-contract problem

The recent repo evidence still points most strongly toward an incomplete handoff between
strategic value and tactical execution. If that is right, a lighter risk-aware bridge should be
tried before a much heavier stochastic controller.

### 5.3 What I would actually recommend

I would recommend the following sequence.

1. Make the bridge contract distribution-aware before making the optimizer fully stochastic.
2. Use strategic quantiles to derive:
   - a central handoff target
   - an upside value-of-energy signal
   - a posture / conservatism weight
3. Feed that signal into the tactical controller through:
   - target bands
   - quantile tilt
   - or a bounded opportunity-cost bias
4. Only move to full scenario-based optimization if the lighter approach clearly helps and the
   pathwise scenario generation becomes trustworthy.

### 5.4 Concrete implication for this repo

For this repo specifically, I think the most promising near-term use of quantiles is **not**
to make the whole controller fully stochastic.

It is to make the strategic layer output a richer uncertainty-aware handoff, for example:
- `soc_target_14h`
- plus a scalar downstream-value or reserve-posture signal derived from upper-tail strategic
  cases

That preserves interpretability, fits the current rolling-eval direction, and tests the
high-value architectural question without prematurely committing to a much heavier stochastic
stack.

---

## 6. Amber APF vs Plan B

### 5.1 What I would do with Amber APF

I would keep Amber APF available as:
- a benchmark
- a transition aid
- an optional ensemble input during migration

I would not design the future system so that it fundamentally requires Amber APF to function.

### 5.2 What a credible Plan B looks like

A credible Amber-independent Plan B should be framed explicitly around the AEMO-native signals.

My current candidate Plan B would be:

1. Tactical execution:
   - confirmed current interval price
   - P5MIN-driven `5m` forecast for roughly `0–60` or `0–120` minutes

2. Short strategic bridge:
   - PREDISPATCH-driven forecast or policy signal out to roughly `12–28h`

3. Long strategic reserve logic:
   - PD7Day + SevenDayOutlook used mainly to derive reserve / scarcity posture, not to force
     a highly trusted detailed path

4. Tactical controller input:
   - current known price
   - short-horizon path
   - terminal SoC target or band
   - posture / reserve weight

This is much closer to "AEMO-native tactical + strategic reserve system" than to
"Amber replacement via one-for-one 72h forecast substitution."

---

## 7. Key Architectural Judgments

### 7.1 The two-tier concept is probably right

I would not currently recommend collapsing everything into a single model or a single horizon.

### 7.2 The current tactical / strategic contract is probably incomplete

The evidence increasingly points here.

### 7.3 The current long-horizon forecast may be too detailed for the value it provides

I think this is more likely than not.

### 7.4 A simpler strategic layer may outperform a richer one if it produces the right handoff

This is a serious possibility and should be tested explicitly.

### 7.5 A more risk-aware implementation is probably desirable before a fully stochastic one

I think the repo should become more distribution-aware, but I would treat full stochastic
optimization as a later-stage option rather than the immediate next architecture.

### 7.6 The current project may still be over-weighting forecast architecture relative to
timing and control design

I think this is likely true.

---

## 8. Experiments I Would Prioritize Next

These are the highest-value experiments I would currently run.

### 8.1 Make the rolling MPC eval the primary architecture gate

Not just as an additional track, but as the main system-design gate.

Reason:
- it is much closer to the actual control problem than one-shot `72h` dispatch simulation

### 8.2 Compare strategic output contracts, not just strategic forecast models

I would compare:
- exact terminal target
- terminal floor / band
- target + posture signal
- target + quantile tilt
- target + opportunity-cost scalar

This would answer the currently central question more directly than another round of model
iteration alone.

### 8.3 Add a scenario-lite strategic valuation experiment

I would test a limited uncertainty-aware bridge before attempting full stochastic control.

Examples:
- derive posture from `q50` vs `q90` strategic handoff value
- compare central handoff target alone vs target-plus-upside-spread
- test whether upper-tail strategic value improves low / normal day execution posture

This is a clean way to ask whether "more stochastic behavior" is actually helping before paying
the full complexity cost of a scenario-based optimizer.

### 8.4 Benchmark a simpler strategic layer against TFT

Examples:
- bias-corrected raw PREDISPATCH / PD7Day
- LightGBM strategic reserve predictor
- heuristic scarcity / reserve policy from SDO + PD7Day

If one of these matches or beats the current strategic TFT in rolling MPC terms, the future
architecture should probably simplify.

### 8.5 Test whether the tactical horizon itself is wrong

The review brief is right to question this.

I would test at least:
- shorter tactical horizon
- current tactical horizon
- a tactical horizon with explicit strategic boundary target

### 8.6 Keep Amber APF in the ablation table while building the AEMO-native path

Not because it is sacred, but because it is the strongest practical comparator for the near
horizon. The repo should know exactly which part of Amber value it has replaced, and which
part still remains.

---

## 9. What I Would Not Do Next

I would not:

- spend the next cycle primarily on improving full `72h` point accuracy without also changing
  the control contract
- assume the current TFT path is the only viable strategic solution
- treat EMHASS integration shape as the architecture to preserve
- assume the remaining gap must be solved inside the forecast model
- jump directly to full stochastic programming before testing lighter uncertainty-aware bridge
  mechanisms

---

## 10. Current Recommendation

My current recommendation is:

1. Keep the two-timescale framing
2. Recast the strategic layer as a **future inventory valuation** layer, not primarily a full
   detailed dispatch-grade `72h` price path layer
3. Keep Amber APF as an optional transition benchmark, but explicitly develop an AEMO-native
   Plan B that does not depend on it
4. Make the next major experiments compare **handoff contracts** and **control formulations**
   at least as aggressively as forecast architectures
5. Move the system in a more **distribution-aware / risk-aware** direction, but treat full
   stochastic optimization as a later option rather than the immediate next step

If this framing is right, the future best-performing system may still contain:
- a tactical short-horizon ML forecaster
- a strategic longer-horizon ML or semi-ML planner

But the decisive design improvement is likely to be:
- a better representation of downstream value-of-energy

not merely:
- a better long-horizon median price curve
