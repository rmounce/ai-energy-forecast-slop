# Independent Review Brief — 2026-04-21

Purpose: provide a neutral, self-contained briefing for an external reviewer on the
current state of the battery-dispatch forecasting pipeline, the recent evaluation work,
and the open architectural decision around opportunity-cost-aware control.

This document is intended to be shareable without additional verbal context.

Related repo documents:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/roadmap.md](./roadmap.md)
- [eval/README.md](../eval/README.md)
- [docs/training_runs.md](./training_runs.md)

---

## 1. System Context

The system forecasts South Australian electricity prices and uses those forecasts to drive
EMHASS battery optimization for a Sigenergy inverter/battery system in Adelaide, SA1.

High-level architecture from the repo:
- Long-horizon price forecasting at `30-minute / 72-hour`
- Short-horizon tactical forecasting at `5-minute / 60-minute`
- EMHASS day-ahead optimization at `72h × 30-minute`
- EMHASS MPC optimization at `14h × 5-minute`

Relevant sources in the repo:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/roadmap.md](./roadmap.md)

### Current modeling layers

From existing repo docs:
- **Tier 1 tactical model**: LightGBM at `5m / 60min`
- **Tier 2 strategic model**: TFT price model at `30m / 72h`
- **Legacy production baseline**: `amber_apf_lgbm`
  - Amber APF provides the shorter-horizon commercial forecast
  - LightGBM extends beyond Amber's native horizon

### Current known eval status

From [docs/roadmap.md](./roadmap.md) and [eval/README.md](../eval/README.md):
- Phase 6 holistic dispatch gate: passing for the intended Tier1+Tier2 hybrid
- Tactical Tier 1 eval: passing
- Phase 7 enhanced-input TFT work: currently failing interim eval
- Rolling MPC eval (Track 10A): implemented and active

---

## 2. Production-Relevant Operating Description

The following operating description was provided during implementation work on 2026-04-21
to clarify how the deployed controller behaves. It may not yet be fully represented in
repo documentation elsewhere.

### MPC timing

- The `14h × 5-minute` MPC re-runs approximately every minute
- The most important high-frequency driver is updated real-time power load / PV measurements
- The confirmed price for the current 5-minute interval typically arrives about `~30s` into
  the interval
- After that confirmed price arrives, the MPC re-runs several more times within the same
  5-minute slot
- Battery action updates are asynchronous from MPC recomputation; the inverter continues
  following the most recent plan until a newer plan is applied

### Long/short horizon interaction

The deployed architecture is described operationally as:
- a `30m / 72h` strategic tier
- a `5m / 14h` tactical EMHASS tier
- the strategic tier influences the tactical tier through a **target SoC at the 14-hour
  boundary**

Operationally, this means:
- the tactical controller does not necessarily "know why" a future SoC should be high
- it only needs to respect the strategic target handed down from the longer-horizon layer

This production description is important because some of the recent rolling MPC experiments
do **not** yet model this exact handoff.

---

## 3. Why This Review Is Being Requested

There is currently an open architectural question:

How should long-horizon opportunity cost influence short-horizon tactical dispatch in the
future system?

This question arose because:
- the current Track 10A rolling MPC eval found a meaningful gap between the hybrid forecast
  stack and the Amber-based baseline on one long follow-up window
- a simple terminal-value surrogate materially narrowed that gap
- but the first dual-driven adaptive variant did not outperform the best static surrogate
- meanwhile, the production system already carries some long-horizon information into the
  tactical layer through a `soc_target` handoff

The review request is therefore **not** just "how to improve one metric", but:
- whether the recent surrogate experiments are revealing a real production need,
- or compensating for a mismatch between the evaluation setup and the real deployed
  two-tier control structure,
- and what the most appropriate long-term architecture should be.

---

## 4. Current Rolling MPC Eval Setup

From [docs/roadmap.md](./roadmap.md) and [eval/README.md](../eval/README.md):

### Track 10A

Purpose:
- execution-focused rolling backtest
- longest dense-history window available

Characteristics:
- `5-minute` stepping
- `14h × 5-minute` horizon
- continuous SoC carryover
- current interval price treated as known
- forecast contract:
  - first hour from Tier 1 tactical forecast
  - remaining horizon from a repeated/expanded strategic extension

Important limitation:
- this eval is not yet a full reproduction of the currently described production
  long/short-horizon handoff
- specifically, it does not yet explicitly model the strategic `14h` SoC target being handed
  to the tactical controller in the same way production is described

### Track 10B

Planned purpose:
- more faithful rolling backtest of the stitched strategic+tactical architecture

Characteristics:
- also `14h × 5-minute`
- intended to start from first PD7Day availability (`2026-02-09`)
- shorter historical coverage, therefore lower statistical power than Track 10A

---

## 5. Relevant Factual Results To Date

### 5.1 Track 10A baseline result

Window A (`2025-07-21 → 2025-09-01`):
- `model_a_hybrid`: **$2.585/day**
- `amber_apf_lgbm`: **$2.523/day**
- hybrid: **+2.4%**

Window B (`2025-09-01 → 2025-10-13`):
- `model_a_hybrid`: **$2.134/day**
- `amber_apf_lgbm`: **$2.406/day**
- hybrid: **−11.3%**

### 5.2 Behavioral diagnosis from Window B

From repo results summarized in [docs/roadmap.md](./roadmap.md) and [eval/README.md](../eval/README.md):

- On `low` days, the hybrid ended with less stored energy than amber
- On `normal` days, the hybrid had weaker SoC posture and worse monetization
- On `spike` days, the hybrid was active but bought less cheaply and sold less expensively

In other words, the follow-up window suggested:
- not merely a lack-of-activity problem
- but poorer inventory positioning and/or poorer timing of energy use

### 5.3 Static terminal-value surrogate sweep

These runs added a fixed end-of-horizon value on stored energy in the rolling MPC eval.

Window B results:
- `0 $/MWh`: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- `50 $/MWh`: hybrid **$2.254/day** vs amber **$2.393/day** (**−5.8%**)
- `100 $/MWh`: hybrid **$2.306/day** vs amber **$2.385/day** (**−3.3%**)
- `150 $/MWh`: hybrid **$2.287/day** vs amber **$2.383/day** (**−4.0%**)
- `200 $/MWh`: hybrid **$2.249/day** vs amber **$2.355/day** (**−4.5%**)

Observed fact:
- a fixed terminal value materially reduced the hybrid's loss on this window
- best result in this coarse sweep was around **`100 $/MWh`**

### 5.4 Dual-driven adaptive sweep

These runs used the LP shadow price of initial SoC as an adaptive signal, then applied a
scaled terminal-energy value based on that signal.

Window B results:
- `dual 0.5`: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- `dual 1.0`: hybrid **$2.140/day** vs amber **$2.411/day** (**−11.3%**)
- `dual 1.5`: hybrid **$2.256/day** vs amber **$2.364/day** (**−4.6%**)
- `dual 2.5`: hybrid **$2.236/day** vs amber **$2.360/day** (**−5.3%**)
- `dual 3.0`: hybrid **$2.245/day** vs amber **$2.363/day** (**−5.0%**)

Observed fact:
- the dual-driven controller improved over the no-bias case at higher scales
- but it did **not** beat the best static terminal-value surrogate (`100 $/MWh`)

### 5.5 Implication that remains unresolved

The static surrogate appears useful in the current Track 10A eval.

What remains unresolved is whether that means:
- the future production system should adopt an explicit opportunity-cost bias in the tactical
  controller,

or instead:
- the current Track 10A eval is missing some of the strategic boundary information that the
  real production controller already uses, and the surrogate is compensating for that omission.

---

## 6. Open Design Options

The following options are the current live design directions.

These are described neutrally; they are not ranked here.

### Option A — Opportunity cost enters through the tactical LP objective

Description:
- keep the forecast path fixed, e.g. q50 or the current tactical/strategic forecast path
- modify the tactical optimization objective so that stored energy at the horizon boundary
  has an explicit value
- the value may be fixed, or derived from an adaptive signal such as the SoC shadow price

Examples:
- fixed terminal-energy value (`100 $/MWh` surrogate)
- dual-driven terminal-energy value

Potential advantages:
- straightforward to implement and test
- isolates execution-policy effects from forecast changes
- already shown to change outcomes in Track 10A

Potential concerns:
- may be acting as a surrogate for missing strategic handoff information in the eval
- may duplicate what the strategic SoC target is already intended to do in production
- current dual-driven version did not outperform the best fixed surrogate

### Option B — Opportunity cost influences forecast quantile selection / blending

Description:
- keep the tactical LP objective unchanged
- let long-horizon opportunity cost or risk posture influence which forecast quantile, or
  quantile blend, is presented to the tactical optimizer

Examples:
- blend `q50` toward `q90` when future inventory value is high
- use a more conservative future price view when preserving charge is strategically important

Potential advantages:
- uses the model's uncertainty outputs directly
- can be layered on top of the current strategic SoC-target handoff
- may align better with the production two-tier structure

Potential concerns:
- depends on quantile quality/calibration
- more tightly couples forecasting and control policy
- attribution is less clean than Option A

### Option C — Combine A and B

Description:
- use both:
  - strategic / opportunity-cost-aware quantile selection or blending
  - and LP-side inventory-value biasing

Potential advantages:
- most expressive
- may capture both uncertainty awareness and explicit inventory value

Potential concerns:
- highest complexity
- hardest to attribute gains or failures
- easiest to overfit

### Option D — Rework the rolling eval first to better match production handoff

Description:
- before committing to A, B, or C, modify the rolling MPC eval so that the tactical controller
  receives the strategic SoC target at the 14h boundary in a way that more faithfully reflects
  the described production system

Potential advantages:
- clarifies whether the gains from Option A are real production gains or merely eval-surrogate gains
- may be necessary before architecture choices can be judged fairly

Potential concerns:
- Track 10B has shorter historical coverage
- higher-fidelity evaluation may come with lower statistical power

---

## 7. Specific Questions For Independent Review

An external reviewer may wish to address some or all of the following:

1. Given the current production description, is the fixed terminal-value surrogate most likely:
   - a valid production direction,
   - an eval surrogate for missing strategic boundary information,
   - or both?

2. Is the current rolling Track 10A setup sufficiently aligned with the intended production
   architecture to draw architectural conclusions from the terminal-value experiments?

3. If the long-horizon strategic tier already hands down a `14h` SoC target, what is the most
   principled way to add opportunity-cost awareness:
   - LP objective bias,
   - quantile/risk-policy tilt,
   - both,
   - or neither?

4. If quantile/risk-policy tilt is the preferred direction, what is the cleanest control
   contract between strategic and tactical layers?

5. What minimal evaluation changes would best distinguish:
   - "better control policy"
   from
   - "surrogate compensation for an eval mismatch"?

6. Are there other formulations of the problem that are more appropriate than A/B/C as framed here?

---

## 8. Summary

Facts established by current repo work:
- the current hybrid stack underperformed the Amber baseline on one important 6-week Track 10A window
- a fixed tactical terminal-value surrogate narrowed that gap substantially
- the first dual-driven adaptive variant did not beat the best static surrogate
- the production system is described as already using a strategic `14h` SoC target to shape
  tactical behavior

The unresolved issue is therefore architectural, not merely numerical:

How should long-horizon opportunity cost be represented in the future system, and how much of
the current surrogate effect is revealing a real production need versus compensating for a gap
between the rolling eval and the production control structure?
