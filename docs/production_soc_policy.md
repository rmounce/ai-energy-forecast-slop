# Production Terminal SoC Policy

**Last updated: 2026-05-08**

This document captures the *production* behaviour of how the Sigenergy battery's
target State-of-Charge (SoC) is set across the day-ahead (DH) and 14-hour MPC layers,
and the relationship to what `eval/rolling_mpc_eval.py` simulates. Most of the
mechanism lives in HA automations and `input_number` entities outside this repo, so
this doc is the source of truth for what the production stack actually does.

Cross-references:

- HA Jinja that consumes the offset: `hass/package-emhass.yaml` (DH `soc_final` block
  around line 493; MPC inheritance block around line 535).
- Earlier reviewer Q3 on MPC inheritance: `docs/architecture_review_2026-04-20.md`
  Q3 (this doc supersedes the partial picture there).
- Architecture overview: `ARCHITECTURE.md`.

## The two-layer setup

The dispatch optimisation runs in two layers:

- **DH (Day-Ahead):** EMHASS optimisation over the next ~72 hours at 30-minute
  resolution. Produces `sensor.dh_soc_batt_forecast.battery_scheduled_soc` — a planned
  SoC trajectory from now to +72h.
- **MPC:** Tighter EMHASS optimisation over the next 14h at 5-minute resolution. Runs
  every 5 minutes. Inherits its terminal SoC target from the DH plan (see "MPC
  inheritance" below).

The two layers share the same EMHASS LP machinery; they differ in horizon, resolution,
and how `soc_final` is set.

## DH: how the 72h target SoC is derived

DH passes `soc_final = clamp(current_soc + emhass_target_soc_offset, min_soc, 100%)`
into EMHASS as the SoC EMHASS should target at the end of the 72h horizon.

The interesting piece is `input_number.emhass_target_soc_offset`. It is **not** static
or user-set in normal operation. An external HA automation (lives in the user's main
HA configuration, not in this repo) **continuously adjusts the offset** based on
whether the most recent DH plan reaches ~98% within the last 24 hours of the 72h
forecast. Concretely:

- If the latest DH-planned trajectory **does not** hit ~98% within +48h..+72h, the
  offset is **incremented** so the next DH solve targets a higher terminal SoC.
- If the trajectory **does** hit ~98% comfortably, the offset is **decremented**.

This is a feedback loop that anchors the terminal target near 100% with a positive
bias toward maintaining higher SoC. The target itself is a moving target — it may
never be reached in practice — but the loop keeps the LP biased upward against
"bleeding" energy over the 3-day horizon.

The intent is operational: a household battery is more useful with energy in it than
empty, especially under uncertain forecasts and intermittent solar. The offset
feedback enforces that preference *as a soft constraint at +72h*, propagated back to
the LP's choices throughout the 72h via the EMHASS objective.

## MPC: how the 14h target SoC is inherited

MPC reads the most recent DH-planned SoC trajectory at +14h and uses that as
`soc_final`. The Jinja in `hass/package-emhass.yaml` does this:

1. Look up `dh_soc_batt_forecast` planned SoC at the future target time (now + 14h).
2. Compute `deviation = (actual_soc_now − planned_soc_now) + 0.5` (a small positive
   bias so being slightly ahead of plan stays counted as ahead).
3. Take `positive_deviation_only = max(0, deviation)` — credit forward only.
4. Set `soc_final = clamp(planned_soc_future + positive_deviation_only, 0, 100%)`.

So the MPC target at +14h is *whatever the DH plan shows at +14h*, plus a one-way
slack: if reality has put us ahead of plan, carry that forward; never pass a *lower*
target than the DH plan would suggest.

**Important consequence:** the +14h MPC target is **not** itself anchored at any
particular value (high or low). It is whatever the offset-adjusted DH trajectory
happens to specify at +14h. On a day with strong evening peaks, the DH plan may
discharge through +14h to capture them — the +14h target will be low. On a flat day
with no opportunities, the DH plan may keep SoC level — the +14h target will be high.
The high-SoC bias enters at +72h, not at +14h.

## Relationship to `eval/rolling_mpc_eval.py`

The eval framework simulates a 14h LP very similar to the production MPC, with one
relevant gap.

What the eval matches:

- 14h × 5-minute LP at each step. ✓
- `--strategic-soc-handoff` with `--strategic-target-mode exact` mirrors the
  inheritance pattern: solve a 72h strategic LP, extract its planned SoC at +14h,
  pass that as the 14h LP terminal target. ✓

What the eval does **not** match:

- The eval's strategic LP solves the 72h horizon with **no terminal SoC constraint
  at +72h**. Production's DH solves with a soft target ~98% at the end (via the
  offset feedback loop).

Effect of the gap: the +72h terminal constraint propagates back through the LP
solution, but only weakly. The LP can satisfy "end at 98%" by recharging anywhere in
the back half (between +14h and +72h). It does not necessarily raise the +14h target.
In practice the eval's +14h target is **a few kWh lower** than production's would be
in flat-tail scenarios (e.g. winter when PD7Day data is missing and the seasonal HoD
fallback dominates the back half of the strategic curve), and **roughly equivalent**
in typical scenarios where the curve itself motivates inventory holding.

This means dispatch evaluations of forecast sources that rely on flat tails (any
pre-2026-02-09 historical window for PD-direct, since that's when PD7Day backfill
starts) should be expected to under-estimate the terminal SoC achievable in
production. Eval-only "battery depleted to 4 kWh" may translate to "battery preserved
to ~10–15 kWh" once production's terminal-end constraint is enforced.

This was noted but not fixed in the 2026-05-08 Phase α-prime Step 4 work — the WA7
SoC depletion observed in eval is plausibly attenuated in live operation by the
production offset feedback. Worth a targeted experiment if the question becomes
load-bearing for a promotion decision, but not blocking shadow-and-compare deployment.

## What this means for shadow forecasts

Phase α-prime Step 2 publishes PD-direct as a shadow alongside the existing
TFT-backed `ai_shadow` family. The shadow forecasts do not control dispatch; they are
inspected and (eventually) compared. Because the production high-SoC bias only
manifests when the forecast *is* driving dispatch (via the DH solve's terminal
target), the shadow path is unaffected: a shadow forecast can show whatever shape it
naturally produces, and the production bias only matters when comparing what *would*
have happened under each forecast had it been driving DH.

## Open questions

- The HA automation that adjusts `emhass_target_soc_offset` is not synced into this
  repo. The increment/decrement step size, hysteresis behaviour, and exact "reaches
  98% within last 24h" detection logic are all defined in the user's main HA
  configuration. Worth syncing when convenient — they are part of the production
  control contract.
- A future `eval/rolling_mpc_eval.py` flag (e.g.
  `--strategic-terminal-soc-target-kwh`) could optionally constrain the strategic LP
  to end at ~95–98% SoC, mirroring the production constraint. Not currently
  implemented; deferred until needed for a specific decision.
