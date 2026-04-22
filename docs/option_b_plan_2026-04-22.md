# Production-Aligned Option B Plan — 2026-04-22

Purpose: define a concrete next-step plan for **Option B** after the Track 10A
strategic-handoff rerun.

This is intended to be the production-aligned path for the next controller experiment:
keep the existing two-tier structure, keep the strategic `14h` SoC handoff, and add a
second handoff carrying **risk / opportunity posture** into the tactical price path via
quantile selection or blending.

Related documents:
- [docs/track10a_handoff_analysis_2026-04-22.md](./track10a_handoff_analysis_2026-04-22.md)
- [docs/roadmap.md](./roadmap.md)
- [docs/independent_review_brief_2026-04-21.md](./independent_review_brief_2026-04-21.md)

---

## 1. What Option B Means Here

Option B does **not** replace the current strategic/tactical split.

The intended control contract remains:
- **Strategic tier (`30m / 72h`)**
  - decides the longer-horizon energy posture
  - hands down a `14h` strategic SoC target
- **Tactical tier (`5m / 14h`)**
  - executes minute-to-minute / 5-minute decisions within that boundary condition

Option B adds a second strategic signal:
- a **risk / opportunity posture**
- used to choose or blend forecast quantiles before the tactical LP is solved

So the tactical controller still optimizes against a price vector, but that vector is no
longer always a pure q50 path.

---

## 2. Why This Is The Right Next Experiment

The current evidence suggests:
- the missing strategic `14h` SoC handoff was a real evaluation defect
- restoring that handoff materially improved Window B
- the hybrid still remains weaker than amber on `low` and `normal` days
- the remaining problem therefore looks less like "missing boundary state" and more like
  **ordinary-day execution posture / price-path shape**

That makes Option B the cleanest next production-aligned experiment because it:
- preserves the architecture already described for production
- avoids relying on a tactical LP salvage-value trick as the main design
- directly tests whether a different **forecast posture** helps the tactical layer make
  better ordinary-day decisions while respecting the strategic boundary

---

## 3. Important Constraints From Existing Repo Knowledge

### Quantile calibration constraints

Known from repo documentation:
- **Tier 2 upper tail is usable:** `q90/q95/q99` are well-calibrated enough for risk-aware
  use
- **Tier 2 lower tail is not yet safe:** `q05/q10` are structurally biased due to log-space
  compression of negative prices

Implication:
- early Option B work should bias **upward / conservative** via `q50 → q90`-style tilts
- do **not** build the first production-facing mechanism around `q05/q10`

### Available model artifacts

Already present in the repo:
- Tactical LightGBM quantiles:
  - `models/lgbm_tactical/lgbm_q05.pkl`
  - `models/lgbm_tactical/lgbm_q50.pkl`
  - `models/lgbm_tactical/lgbm_q95.pkl`
- Strategic TFT quantiles:
  - checkpoint outputs include `q05/q10/q50/q90/q95/q99`

Implication:
- the first Option B eval can use **real quantile curves on both tiers**
- no retraining is required just to test the mechanism

---

## 4. Production-Aligned Option B Contract

The cleanest near-term contract is:

1. Strategic tier computes the existing `soc_target_14h`
2. Strategic tier also computes a scalar **posture weight** `w`
3. Tactical controller receives:
   - the SoC boundary condition
   - the posture weight `w`
4. Tactical effective price path is formed by blending quantiles before LP solve

Recommended first form:

`effective_price = q50 + w * (q_hi - q50)`

Where:
- `q_hi` is the conservative / preserve-inventory quantile
- `w ∈ [0, 1]`
- `w = 0` means pure q50
- `w = 1` means pure upper-quantile path

For the first experiment:
- Tactical first hour: blend `q50` toward `q95`
- Strategic extension: blend `q50` toward `q90` or `q95`

Rationale:
- tactical LightGBM only has `q05/q50/q95`, so `q95` is the natural upper tail there
- TFT has both `q90` and `q95`; starting with `q90` may be less jumpy than `q95`

---

## 5. Recommended Experimental Sequence

### Stage B0 — Mechanism check with fixed blend weights

Goal:
- prove the mechanism on the **handoff-enabled Track 10A baseline**
- separate "does quantile tilt help?" from "can we derive the right tilt dynamically?"

Implementation:
- add fixed blend controls to rolling MPC eval, for example:
  - `--tier1-quantile-blend 0.0`
  - `--tier2-quantile-blend 0.0`
  - or one shared `--opportunity-blend-weight`
- evaluate a small sweep such as:
  - `0.00`
  - `0.25`
  - `0.50`
  - `0.75`
  - `1.00`

Primary readout:
- does a handoff-enabled hybrid improve on Window B, especially `low` / `normal`, without
  breaking Window A?

This stage should use the current strategic handoff baseline as the control, not the older
no-handoff Track 10A.

### Stage B1 — Strategic-only dynamic posture

Goal:
- keep the architecture clean by deriving posture from the **strategic** horizon only

Candidate posture signals:
- spread between strategic upper and median tail:
  - `max(q90 - q50)` over hours `14h+`
- boundary-value gap near the handoff:
  - `mean(q90 - q50)` over a window around `14h..24h`
- strategic scarcity score:
  - high when the long horizon shows expensive downstream opportunities that justify
    inventory preservation

Implementation shape:
- map the strategic signal into `w ∈ [0, 1]`
- keep the mapping simple and monotonic at first

Example:
- below threshold: `w = 0`
- above threshold: linearly ramp toward `1`

### Stage B2 — Decouple tactical and strategic weights

If B0/B1 help:
- use one weight for the first hour tactical curve
- use another weight for the 1h–14h strategic extension

Reason:
- the first hour is about execution quality and may need a milder tilt
- the 1h–14h extension is where preserving downstream opportunity may matter more

### Stage B3 — Productionization path

Only after B0/B1/B2 show value:
- add the posture signal to the live controller contract
- publish needed upper-tail entities if not already exposed in `forecast.py`
- document how the tactical layer consumes both:
  - `soc_target_14h`
  - posture / blend weight

---

## 6. Scope Boundaries

To keep attribution clean, the first Option B implementation should **not** simultaneously:
- add LP terminal salvage value
- add dual-driven LP bias
- add lower-tail charging logic
- alter the strategic handoff rule

Those can be revisited later if needed, but the first production-aligned test should isolate:

**Does risk-aware quantile posture improve the residual post-handoff weakness?**

---

## 7. Concrete Code Changes Expected

The first implementation likely belongs in `eval/rolling_mpc_eval.py`.

Expected work items:
- extend TFT inference helper to return arbitrary quantile columns, not just q50
- load tactical `q05/q50/q95` models instead of only q50
- add provider methods for:
  - `tier1_q95(...)`
  - `tft_q90_expanded(...)` and/or `tft_q95_expanded(...)`
  - blended first-hour and blended strategic-extension curves
- add CLI flags for blend-weight experiments
- keep strategic SoC handoff enabled and unchanged during Option B sweeps

Secondary docs to update when this lands:
- [eval/README.md](../eval/README.md)
- [docs/roadmap.md](./roadmap.md)

---

## 8. Success Criteria

The next Option B experiment should be judged against the **handoff-enabled** baseline.

Minimum success bar:
- improve Window B overall versus handoff-enabled q50 hybrid
- specifically improve `low` and/or `normal` regimes
- avoid materially degrading Window A

Interpretation guide:
- if fixed upward tilt helps, the remaining weakness is plausibly forecast-posture related
- if fixed tilt does not help, the remaining weakness is more likely tactical execution
  policy, ordinary-day forecast quality, or both

---

## 9. Current Recommendation

Recommended next coding step:

1. Implement **Stage B0** only
2. Use the handoff-enabled Track 10A baseline
3. Start with **upper-tail-only** blending
4. Keep the strategic SoC handoff intact
5. Defer dynamic posture derivation until fixed-weight evidence exists

This keeps the next experiment:
- production-aligned
- easy to interpret
- low-risk to implement
- consistent with the current documentation and the external review feedback
