# Codex Holistic Review Follow-Up — 2026-04-23

Purpose: provide a short follow-up briefing for the same reviewer after the next round of
rolling MPC results.

This is intentionally narrower than the previous review draft. It is meant to answer:

- what changed since the last review
- what that seems to rule out
- what the next decision point is

Related:
- [docs/codex_holistic_review_draft_2026-04-22.md](./codex_holistic_review_draft_2026-04-22.md)
- [docs/track10a_handoff_analysis_2026-04-22.md](./track10a_handoff_analysis_2026-04-22.md)
- [docs/option_b_sweep_results_2026-04-23.md](./option_b_sweep_results_2026-04-23.md)
- [docs/roadmap.md](./roadmap.md)

---

## 1. What Changed Since The Last Review

Two things have now happened.

### A. Strategic handoff was restored in Track 10A

The rolling MPC eval now includes the strategic `14h` SoC handoff.

Window B changed from:
- pre-handoff hybrid: **$2.134/day**
- amber: **$2.406/day**
- gap: **−11.3%**

to:
- handoff-enabled hybrid: **$2.271/day**
- amber: **$2.451/day**
- gap: **−7.4%**

Interpretation:
- the earlier no-handoff eval really was missing a production-relevant boundary condition
- but restoring that boundary condition did **not** eliminate the residual gap

This was consistent with your earlier reading that part of the earlier surrogate effect was
an eval-alignment issue.

### B. Naive fixed-weight Option B failed

We then tested the simplest production-aligned version of Option B:

- keep the strategic `14h` SoC handoff
- tilt the tactical/strategic price path upward from `q50` toward `q90`
- use fixed blend weights

Sweep results on the same Window B:
- `blend 0.25`: hybrid **$2.232/day** vs amber **$2.451/day** (**−8.9%**)
- `blend 0.50`: hybrid **$1.923/day** vs amber **$2.451/day** (**−21.5%**)
- `blend 0.75`: hybrid **$1.579/day** vs amber **$2.451/day** (**−35.6%**)
- `blend 1.00`: hybrid **$1.224/day** vs amber **$2.451/day** (**−50.0%**)

Relative to the handoff-enabled q50 hybrid baseline (`$2.271/day`), every tested blend was
worse, and the degradation was monotonic.

Interpretation:
- a fixed global upper-tail tilt is too blunt
- it causes over-preservation of inventory
- it hurts `low` and `normal` days badly
- and at higher weights it also hurts `spike` days

So the simple "just lean toward q90" implementation is now ruled out as a live candidate.

---

## 2. What Seems Narrowed By This

The space now looks tighter in a useful way.

### What now seems unlikely

- the next gain is **not** likely to come from a naive always-on quantile tilt
- the next gain is **not** likely to come from treating the strategic layer as merely a
  source of a different full price vector

### What still seems plausible

- the strategic layer still needs to express downstream value somehow
- but that expression likely has to be **selective / state-dependent**
- a richer bridge contract still seems more plausible than a pure fixed price-path tilt

This feels consistent with the previous review's claim that the bridge contract is the real
underdeveloped part of the architecture.

---

## 3. Current Decision Point

The immediate question is now:

What should be tested next as the most principled follow-on to:
- a positive handoff-alignment result
- and a negative fixed-blend result?

The leading candidates currently look like:

### Option 1 — Dynamic / selective posture signal

Keep:
- strategic `14h` SoC handoff

Add:
- a dynamic posture or conservatism scalar derived from strategic upside, for example from
  `q50` vs `q90`

Use that signal selectively rather than as a fixed global blend.

### Option 2 — Alternate bridge contracts

Examples:
- target band instead of exact target
- target + posture signal
- target + bounded opportunity-cost scalar
- floor/band plus selective posture

### Option 3 — Simpler strategic outputs

Instead of asking the strategic layer for a rich `72h` executable path, benchmark simpler
strategic outputs such as:
- reserve / scarcity posture
- value-of-energy proxy
- bias-corrected raw market signals

### Option 4 — Revisit tactical horizon

Treat the tactical horizon itself as a live variable rather than a fixed assumption.

---

## 4. What We Are Asking You

Given this new evidence, what branch would you back next?

More specifically:

1. Does the negative fixed-blend result strengthen your prior view that the core missing
   piece is the **bridge contract**, not the forecast path itself?

2. If so, what is the most principled next experiment:
   - dynamic posture signal
   - alternate terminal-state contract
   - simpler strategic output
   - tactical-horizon rethink
   - something else

3. If you had to choose **one next experiment only**, what would it be?

4. Has any of the new evidence changed your view on whether the strategic layer should be
   framed more as:
   - future inventory valuation / reserve posture
   rather than
   - a detailed long-horizon price-path generator?

---

## 5. Current Leaning, Stated Cautiously

The repo's current leaning is:

- your earlier review looks more right than less right after this round
- the positive handoff result supports the importance of strategic boundary information
- the negative fixed-blend result argues against a naive "just tilt the path" solution
- so the next promising direction appears to be a **dynamic or richer bridge contract**
  rather than another fixed-weight path experiment

But that is exactly the point we would like checked before proceeding further.
