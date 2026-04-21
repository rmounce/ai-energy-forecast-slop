# Track 10A Handoff Analysis — 2026-04-22

Purpose: record what changed when the strategic `14h` SoC handoff was added to the
Track 10A rolling MPC eval, and what did **not** change.

Related artifacts:
- [Window B pre-handoff summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_summary_vs_baseline.csv:1)
- [Window B handoff exact summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_exact_summary_vs_baseline.csv:1)
- [Window B handoff floor summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_floor_summary_vs_baseline.csv:1)
- [Window B pre-handoff behavior](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_behavior_prices_behavior_summary_vs_baseline.csv:1)
- [Window B handoff exact behavior](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_followup_6week_handoff_exact_behavior_summary_vs_baseline.csv:1)
- [Window A handoff exact summary](/home/saltspork/src/ai-energy-forecast-slop/eval/results/rolling_mpc_eval_tracka_windowA_handoff_exact_summary_vs_baseline.csv:1)

---

## Headline

Adding the strategic `14h` SoC handoff improved the bad Window B result, but it did not
eliminate the hybrid's underperformance versus `amber_apf_lgbm`.

Window B overall:
- Pre-handoff: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- Handoff exact: hybrid **$2.271/day** vs amber **$2.451/day** (**−7.4%**)
- Handoff floor: hybrid **$2.271/day** vs amber **$2.450/day** (**−7.3%**)

This means the missing boundary condition was a real part of the problem, but not the whole
problem.

---

## What The Handoff Fixed

The clearest improvement was on `spike` days.

Window B `spike` regime:
- Pre-handoff hybrid: **$4.073/day**
- Handoff exact hybrid: **$4.484/day**
- Amber under handoff exact: **$4.632/day**

Gap vs amber on `spike` shrank from about **−11.8%** pre-handoff to about **−3.2%** with
handoff exact.

Behaviorally, the handoff changed the hybrid on spike days in a direction that looks more
strategic:
- opening SoC increased further
- closing SoC increased further
- average discharge price improved substantially
- average charge price also became less negative, but the discharge improvement dominated

This is consistent with the idea that the restored strategic boundary condition helped the
hybrid preserve and deploy energy more appropriately around high-value events.

---

## What The Handoff Did Not Fix

The hybrid remained weaker on `low` and `normal` days.

Window B `low` regime:
- Amber handoff exact: **$1.476/day**
- Hybrid handoff exact: **$1.348/day**

Window B `normal` regime:
- Amber handoff exact: **$1.843/day**
- Hybrid handoff exact: **$1.597/day**

These residual gaps are still material.

### Low days

Compared with pre-handoff, the hybrid on low days:
- opened with much more SoC
- closed with much more SoC
- built more inventory over the day

But profitability still worsened slightly relative to amber.

Interpretation:
- the handoff repaired the "arrive with energy" behavior
- but the hybrid still appears to buy / hold inventory less effectively than amber in this regime

### Normal days

Compared with pre-handoff, the hybrid on normal days:
- opened with much more SoC
- closed with much more SoC
- improved mean daily PnL somewhat

But it still lagged amber materially.

Interpretation:
- restoring the strategic boundary helped inventory posture
- but ordinary-day monetization remains weaker than amber even after that correction

---

## Exact vs Floor

On Window B, `exact` and `floor` terminal target enforcement produced almost identical
overall outcomes.

That suggests the major gain came from restoring the strategic handoff at all, not from the
precise choice between exact terminal target and floor-style enforcement.

---

## Window A Sanity Check

Window A remained slightly favorable to the hybrid after adding the handoff:
- Amber handoff exact: **$2.630/day**
- Hybrid handoff exact: **$2.636/day**

So the handoff did not collapse the earlier positive result. It mainly changed the
interpretation of Window B by showing that some, but not all, of the earlier deficit was due
to missing boundary-state information.

---

## Current Interpretation

The handoff result supports three conclusions:

1. The original no-handoff Track 10A setup was missing a real production-relevant piece of
the architecture.

2. The static terminal-value surrogate was at least partly compensating for that omission.

3. Even after restoring the strategic handoff, the hybrid still has a residual weakness on
`low` and `normal` days.

So the remaining question is no longer "does the tactical layer need any long-horizon
information at all?" The answer to that appears to be yes.

The remaining question is:

What best addresses the post-handoff residual weakness:
- forecast shape / quantile posture,
- tactical execution policy on ordinary days,
- or both?

---

## Practical Consequence

From this point forward, Track 10A comparisons should treat the **handoff-enabled** setup as
the aligned baseline.

Future experiments on:
- LP-side opportunity-cost bias
- quantile/risk-policy tilt
- combined approaches

should be judged against the handoff-enabled version, not the older no-handoff variant.
