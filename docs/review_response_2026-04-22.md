# Response To Holistic Review — 2026-04-22

Purpose: record the architectural implications of the holistic review in
[docs/codex_holistic_review_draft_2026-04-22.md](./codex_holistic_review_draft_2026-04-22.md)
and how it changes current priorities.

This note is not a rebuttal. It is a compact statement of what the review appears to
change in the repo's working roadmap.

Related:
- [docs/codex_holistic_review_draft_2026-04-22.md](./codex_holistic_review_draft_2026-04-22.md)
- [docs/roadmap.md](./roadmap.md)
- [docs/option_b_plan_2026-04-22.md](./option_b_plan_2026-04-22.md)
- [docs/track10a_handoff_analysis_2026-04-22.md](./track10a_handoff_analysis_2026-04-22.md)

---

## Headline

The holistic review does **not** argue that the whole current architecture should be
discarded.

It does argue that the repo is now more likely to be bottlenecked by an incomplete
**strategic-to-tactical contract** than by the absence of another round of long-horizon
forecast iteration.

In short:
- keep the two-timescale framing
- de-emphasize the idea that the strategic layer's main product must be a detailed
  `72h` executable point forecast
- prioritize experiments that improve how downstream value-of-energy is represented and
  handed to the tactical controller

---

## What The Review Seems To Confirm

The review broadly supports several existing repo directions:
- financial outcome is the right headline objective
- a tactical short-horizon execution layer is structurally justified
- long-horizon awareness is real and should not be removed
- Amber APF should be treated as a benchmark / transition aid, not as the desired
  permanent dependency

So the main implication is **reprioritization**, not a reset.

---

## What The Review Challenges

The strongest challenge is to the current strategic-layer framing.

The review suggests that the repo may be over-investing in:
- improving a detailed `30m / 72h` price path

while under-investing in:
- defining the right bridge between long-horizon value and tactical execution

The strategic layer may ultimately be better judged by how well it provides:
- inventory targets or target bands
- future value-of-energy
- reserve / scarcity posture
- uncertainty-aware handoff signals

rather than by the quality of a detailed median path alone.

---

## Current Interpretation

The review strengthens the case for the following reading of recent Track 10A results:

1. The earlier no-handoff rolling eval was missing a real production-relevant boundary
   condition.
2. Restoring the strategic `14h` SoC handoff corrected part, but not all, of the gap.
3. The remaining weakness is now more plausibly a **contract / posture / control**
   problem than a pure "better 72h q50" problem.

This does **not** prove that forecast quality no longer matters.

It does suggest that the next highest-value experiments should compare:
- handoff contracts
- uncertainty-aware bridge signals
- simpler strategic outputs

before assuming that another major strategic forecaster iteration is the dominant lever.

---

## Roadmap Reprioritization

### Promote

Promote the following:
- **rolling MPC eval** as the primary architecture gate
- **handoff-contract experiments** as first-class design work
- **scenario-lite / upper-tail-aware bridge signals** ahead of heavier stochastic methods
- **simpler strategic baselines** that are judged by rolling MPC outcomes, not just
  forecast metrics

### De-emphasize

De-emphasize the assumption that the next major gain must come from:
- improving the strategic layer as a full-path `72h` point forecaster

This remains a valid avenue, but no longer appears to be the default next bet.

---

## Practical Next Experiments

The review suggests the following ordering.

### 1. Compare strategic output contracts directly

Examples:
- exact terminal target
- floor / band
- target + posture signal
- target + quantile tilt
- target + bounded opportunity-cost scalar

### 2. Run scenario-lite bridge experiments

Examples:
- derive a posture signal from strategic `q50` vs `q90` upside
- compare target-only vs target-plus-upside-posture
- test whether upper-tail-aware posture improves `low` / `normal` days in rolling MPC

### 3. Benchmark simpler strategic layers

Examples:
- bias-corrected raw PREDISPATCH / PD7Day
- heuristic reserve / scarcity policy
- simple strategic learner producing posture / value outputs instead of a full path

### 4. Revisit tactical horizon length explicitly

The review explicitly questions whether the current tactical horizon should be taken as
given. That should now be treated as a real experiment, not only a background assumption.

---

## Amber APF Position

The review is broadly consistent with the repo's current secondary objective:
- move toward an Amber-independent system if that can be done without sacrificing too much
  live financial performance

The practical implication is:
- keep Amber APF in the ablation table and as a transition benchmark
- do not treat it as the desired long-term dependency

---

## Current Recommendation

For now, the most defensible working direction is:

1. Keep the two-timescale architecture
2. Keep Amber APF as a benchmark, not a design anchor
3. Treat the strategic layer less as a "perfect 72h path" layer and more as a
   **future inventory valuation / posture** layer
4. Make the next major experiments compare **handoff contracts and bridge signals**
   at least as seriously as forecast models
5. Prefer **distribution-aware / risk-aware** bridge work before any jump to full
   stochastic optimization

That is the main roadmap change implied by the holistic review.
