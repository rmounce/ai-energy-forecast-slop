# Option B Fixed-Blend Sweep Results — 2026-04-23

Purpose: record the first fixed-weight Option B sweep on the **handoff-enabled** Track 10A
Window B baseline.

Related:
- [docs/option_b_plan_2026-04-22.md](./option_b_plan_2026-04-22.md)
- [docs/track10a_handoff_analysis_2026-04-22.md](./track10a_handoff_analysis_2026-04-22.md)
- [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

Artifacts:
- [blend 0.25 summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_exact_blend025_q90_summary_vs_baseline.csv:1)
- [blend 0.50 summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_exact_blend050_q90_summary_vs_baseline.csv:1)
- [blend 0.75 summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_exact_blend075_q90_summary_vs_baseline.csv:1)
- [blend 1.00 summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_exact_blend100_q90_summary_vs_baseline.csv:1)

---

## Headline

The first fixed-weight Option B sweep was a clear **negative result**.

Blending the handoff-enabled hybrid forecast upward from `q50` toward `q90` made Window B
worse at every tested blend weight. Higher blend weights degraded performance monotonically.

This does **not** support a simple production rule of:

- "just tilt the tactical/strategic path toward the upper tail"

---

## Setup

Window:
- `2025-09-01 → 2025-10-13`

Control baseline:
- Track 10A
- strategic `14h` SoC handoff enabled
- `exact` terminal target mode

Sweep:
- Tier 1 first hour: `q50 + w * (q95 - q50)`
- Tier 2 extension: `q50 + w * (q90 - q50)`
- tested `w ∈ {0.25, 0.50, 0.75, 1.00}`

Reference baseline from the prior handoff rerun:
- handoff-enabled q50 hybrid: **$2.271/day**
- amber baseline: **$2.451/day**
- hybrid gap: **−7.4%**

---

## Overall Results

Against `amber_apf_lgbm`:

- `blend 0.25`: hybrid **$2.232/day** vs amber **$2.451/day** (**−8.9%**)
- `blend 0.50`: hybrid **$1.923/day** vs amber **$2.451/day** (**−21.5%**)
- `blend 0.75`: hybrid **$1.579/day** vs amber **$2.451/day** (**−35.6%**)
- `blend 1.00`: hybrid **$1.224/day** vs amber **$2.451/day** (**−50.0%**)

Against the handoff-enabled q50 hybrid baseline (`$2.271/day`):

- `blend 0.25`: **−1.7%**
- `blend 0.50`: **−15.3%**
- `blend 0.75`: **−30.5%**
- `blend 1.00`: **−46.1%**

The shape is monotonic: more upper-tail tilt causes worse overall financial performance on
this window.

---

## Regime View

### Low days

Low days deteriorated sharply with increasing blend weight.

- `blend 0.25`: **$0.994/day**
- `blend 0.50`: **$0.809/day**
- `blend 0.75`: **$0.687/day**
- `blend 1.00`: **$0.511/day**

Amber reference:
- **$1.476/day**

### Normal days

Normal days also degraded as the blend increased.

- `blend 0.25`: **$1.730/day**
- `blend 0.50`: **$1.543/day**
- `blend 0.75`: **$1.143/day**
- `blend 1.00`: **$0.868/day**

Amber reference:
- **$1.843/day**

### Spike days

Spike days did **not** justify the fixed blend either.

- `blend 0.25`: **$4.582/day**
- `blend 0.50`: **$3.928/day**
- `blend 0.75`: **$3.386/day**
- `blend 1.00`: **$2.683/day**

Amber reference:
- **$4.632/day**

`blend 0.25` was close to amber on spike days, but not enough to offset the low/normal
damage. Higher weights degraded spike days substantially as well.

---

## Behavioral Read

The fixed upper-tail tilt appears to make the tactical controller too conservative with
inventory:

- lower dispatch intensity
- more inventory carried
- weaker monetization on ordinary days

In other words, the controller behaves as if future upside is too important too often.

This is consistent with a bridge signal that is too blunt:
- useful downstream-value awareness is likely state-dependent
- a fixed global upper-tail tilt is not selective enough

---

## Interpretation

This result narrows the architecture space in a useful way.

### What it does rule out

It rules out the naive version of Option B:
- fixed `q50 → q90` blending as a general-purpose production bridge

### What it does not rule out

It does **not** rule out all distribution-aware / posture-aware bridges.

More selective forms may still be valid, for example:
- dynamic posture signals derived from strategic upside spread
- target + posture contracts
- bounded opportunity-cost scalars
- scenario-lite strategic valuation

---

## Practical Consequence

The repo should treat this sweep as a **negative ablation** and move on from fixed-weight
upper-tail blend sweeps.

Recommended next direction:

1. keep the strategic `14h` SoC handoff as the aligned baseline
2. stop treating fixed q50→q90 tilt as a live candidate
3. move to:
   - dynamic / selective posture signals, or
   - alternate bridge-contract experiments, or
   - simpler strategic-output baselines

That is a more promising follow-on than further hand-tuning of fixed blend weights.
