# Production Terminal SoC Policy

**Last updated: 2026-05-29**

This document captures the *production* behaviour of how the Sigenergy battery's
target State-of-Charge (SoC) is set across the day-ahead (DH) and 14-hour MPC layers,
and the relationship to what `eval/rolling_mpc_eval.py` simulates. Most of the
mechanism lives in HA scripts and `input_number` entities outside this repo, so
this doc is the source of truth for what the production stack actually does.

Cross-references:

- DH script (computes soc_init/soc_final, writes `dh_last_soc_init`, fires the
  rest_command): `script.emhass_dayahead_optim` in `hass/package-emhass.yaml`.
- MPC script (computes soc_init/soc_final, writes `mpc_last_soc_init`, fires the
  rest_command): `script.emhass_mpc` in `hass/package-emhass.yaml`.
- The underlying `rest_command.emhass_dayahead_optim` and `rest_command.emhass_mpc`
  now expect `soc_init_pct` and `soc_final_pct` as parameters. They must be called
  via the scripts above, not directly.
- Earlier reviewer Q3 on MPC inheritance: `docs/architecture_review_2026-04-20.md`
  Q3 (this doc supersedes the partial picture there).
- Architecture overview: `ARCHITECTURE.md`.

## The two-layer setup

The dispatch optimisation runs in two layers:

- **DH (Day-Ahead):** EMHASS optimisation over the next ~72 hours at 30-minute
  resolution. Produces `sensor.dh_soc_batt_forecast.battery_scheduled_soc` — a planned
  SoC trajectory from now to +72h.
- **MPC:** Tighter EMHASS optimisation over the next 14h at 5-minute resolution. Runs
  every ~1 minute. Inherits its terminal SoC target from the DH plan plus a one-way
  lock-in slack (see "MPC: how soc_init and soc_final are set" below).

Both layers run inside an HA `script` wrapper that computes `soc_init` and
`soc_final` from the prior DH plan plus the live SoC, then fires the corresponding
`rest_command` against EMHASS. The wrappers also persist the chosen `soc_init` to
`input_number.dh_last_soc_init` (DH) and `input_number.mpc_last_soc_init` (MPC) so
the next run can correctly anchor the plan.

## DH: how soc_init and soc_final are set

DH applies a **self-correction** every run:

```
anchor    = dh_last_soc_init                       (the soc_init passed to the previous DH run)
deviation = actual_soc_now − interp_planned_now    (signed)
soc_init  = clamp(anchor + deviation, min_soc, 100%)
soc_final = clamp(anchor + emhass_target_soc_offset + deviation, min_soc, 100%)
```

where:

- `actual_soc_now` is the live SoC from
  `sensor.sigen_plant_battery_state_of_charge_derived`.
- `interp_planned_now` linearly interpolates the prior DH plan
  (`sensor.dh_soc_batt_forecast.battery_scheduled_soc`) at `utcnow()`. The
  interpolation is anchored at (plan_start, `dh_last_soc_init`) — the published
  plan only carries end-of-interval values, so the start anchor has to come from
  the persisted helper.
- `emhass_target_soc_offset` is the externally-adjusted +72h offset (see below).

The deviation is signed (can lift or lower both endpoints). The chain is bounded
in practice by EMHASS's own SoC clamping plus MPC's live-SoC re-grounding on every
dispatch cycle, so no stable external truth-anchor (e.g. a periodic SoC snapshot)
is needed.

Bootstrap: when `dh_last_soc_init` is 0 (first run after deploy, or HA restart
with no restored state), the anchor falls back to the live derived SoC sensor.
With no prior plan, deviation = 0 and `soc_init = live_soc` on the first cycle;
from the second cycle onward the chain is established.

**30-minute boundary re-grounding (2026-05-30):** when the DH script fires
within 60 seconds of a wall-clock 30-minute boundary (xx:00 or xx:30), it
re-grounds the anchor to live SoC and forces `deviation = 0`, breaking the
chain for that single solve. EMHASS's DH quantization snaps to the new
boundary at exactly that moment, and the chained anchor + prior-plan
interpolation otherwise interact across the boundary to produce a ~1pp jump
in the next solve's `max_tail`. Re-grounding cuts that interaction. The chain
restarts cleanly from `live_soc` on the post-re-grounding solve. Re-grounding
catches the price-triggered solves (~10–30 s past the boundary) but not the
load-triggered solves at xx:01 / xx:31 (~75–95 s past).

## The +72h offset feedback loop

`input_number.emhass_target_soc_offset` is **not** static or user-set in normal
operation. The automation `"EMHASS — Update target SoC offset"` in
`hass/package-emhass.yaml` fires on every `sensor.dh_soc_batt_forecast` state
change (i.e., after every DH solve publishes) and adjusts the offset based on
whether the just-published plan reaches ~97.5% in the **clock-anchored** tail
(intervals whose end > `now + 48h`):

```
cutoff     = now + 47h30m                # interval start cutoff (so end > now+48h)
tail       = [ p.dh_soc_batt_forecast for p in battery_scheduled_soc
               if as_datetime(p.date) > cutoff ]
max_tail   = max(tail)
new_offset = current_offset + (97.5 − max_tail)
```

The clock-anchored window replaces the original `[-48:]` index slice
(2026-05-30). The slice version shifted *both* ends of the wall-clock window
by 30 min at each quantization boundary (drops front, adds back), feeding
boundary noise into the offset; the clock-anchored window only shifts the
back end, matches the conceptual definition of "the plan's tail beyond
now+48h", and pairs with the script's boundary re-grounding (see "DH" above)
to keep the offset chart smooth across quantization shifts. If the cutoff
falls past `plan_end` (e.g. unusually stale plan), the offset is emitted
unchanged.

- If the latest DH-planned trajectory **does not** hit ~97.5% within +48h..+72h,
  the offset is **incremented** so the next DH solve targets a higher terminal
  SoC.
- If the trajectory **does** hit ~97.5% comfortably, the offset is
  **decremented**.

The next DH solve picks up the new value from the input_number.

History:

- Pre-2026-05-29: this loop lived in an HA-UI automation (archived at
  `hass/automation-emhass-target-soc-offset.yaml.bak`).
- 2026-05-29: squashed into `script.emhass_dayahead_optim` (pre-solve reads of
  the prior plan).
- 2026-05-30: reverted to a post-publish automation, now in-repo in
  `hass/package-emhass.yaml`. The pre-solve variant added one cycle of lag to
  the feedback loop and caused visible ~1.2pp sawtooth oscillations at every
  30-min load-forecast boundary.
- 2026-05-30 (same day): post-revert measurement showed a residual ~1pp dip
  caused not by the trigger model but by the script-wrapper refactor's
  chained anchor interacting with the prior-plan interpolation at the
  quantization boundary. Two complementary fixes shipped: (a) script-level
  boundary re-grounding (see "DH" above), and (b) clock-anchored window in
  the offset automation (replacing the original `[-48:]` index slice). The
  pre-script-wrapper baseline had ~0.05pp boundary deltas; these two changes
  together aim to restore that.

This is a feedback loop that anchors the terminal target near 100% with a positive
bias toward maintaining higher SoC. The target itself is a moving target — it may
never be reached in practice — but the loop keeps the LP biased upward against
"bleeding" energy over the 3-day horizon.

The intent is operational: a household battery is more useful with energy in it
than empty, especially under uncertain forecasts and intermittent solar. The
offset feedback enforces that preference *as a soft constraint at +72h*,
propagated back to the LP's choices throughout the 72h via the EMHASS objective.

## MPC: how soc_init and soc_final are set

MPC reads the most recent DH-planned SoC trajectory and applies plan-relative
adjustments at the start and end of its 14h horizon:

```
real_soc          = live SoC from sensor.sigen_plant_battery_state_of_charge_derived
bias_pct          = clamp((real_soc − 90) / (99.99 − 90), 0, 1) * 0.20
                    # ramped force-charge bias — see "Force-charge top-balance bias" below
effective_soc_pct = 100 if real_soc >= 100 else max(0, real_soc − bias_pct)
deviation         = effective_soc_pct − planned_soc_at_now    (signed)
positive_only     = max(deviation, 0)
soc_init          = clamp(planned_soc_at_boundary + deviation, 0, 100%)
soc_final         = clamp(planned_soc_at_future + positive_only, 0, 100%)
```

where:

- `planned_soc_at_boundary` is the prior DH plan interpolated at the most recent
  5-minute boundary (the start of EMHASS's first planning interval).
- `planned_soc_at_now` is the prior DH plan interpolated at `utcnow()` (no
  quantisation). Using the live timestamp removes the 5-min step-jumps that a
  quantised lookup would introduce in the deviation signal.
- `planned_soc_at_future` is the prior DH plan interpolated at the end of the
  168-step (14h) horizon.
- All three lookups are anchored at (plan_start, `dh_last_soc_init`); same
  end-of-interval-only data problem as DH.

So the MPC `soc_init` is "the planned SoC at the start of EMHASS's first interval,
lifted by the signed deviation between live SoC and the planned-now value".
Equivalently, when the plan slope is flat over the partial interval
`[quantized_now, utc_now]`, `soc_init` reduces to `effective_soc_pct` — matching
the pre-self-correction behaviour. The `soc_final` keeps the original positive-only
lock-in lead: lift the planned future SoC only when we're ahead of plan; never
lower the target.

**Important consequence:** the +14h MPC target is **not** itself anchored at any
particular value (high or low). It is whatever the offset-adjusted DH trajectory
happens to specify at +14h, possibly lifted by the lock-in lead. On a day with
strong evening peaks, the DH plan may discharge through +14h to capture them — the
+14h target will be low. On a flat day with no opportunities, the DH plan may keep
SoC level — the +14h target will be high. The high-SoC bias enters at +72h, not at
+14h.

## Force-charge top-balance bias

`effective_soc_pct` deflates the live SoC by a **ramped** bias that scales
linearly from 0pp at SoC ≤ 90% to 0.20pp at SoC ≥ 99.99% (and is pinned at the
100% reporting ceiling above 100%):

```
bias_pct = clamp((real_soc − 90) / (99.99 − 90), 0, 1) * 0.20
```

Because `effective_soc_pct` feeds the deviation calc, and the deviation in turn
feeds both `soc_init` and `soc_final`, the deflation propagates through both
anchors. When the bias is active near the top of the window, the result is a
small synthetic energy deficit (~60 Wh on a 30 kWh battery at full bias) that
EMHASS must close somewhere in its 14h horizon.

**Why it's there** (two intertwined purposes):

1. **Top-balance:** at very high SoC (e.g. 99.9%) EMHASS would otherwise plan only
   a tiny charge in the final 5-min interval to reach 100% — but the battery cells
   need slightly more power than the strict SoC math implies in order to top-balance.
   The bias inflates the headroom and gives EMHASS more power budget in those final
   moments.
2. **Reaching 100% at all:** without the bias, EMHASS would happily spread charging
   across the day (quadratic power cost penalty rewards flattening), then pivot
   from "charge from PV" to "export to grid" as late-afternoon prices rise — often
   before the battery reaches 100%. The synthetic deficit keeps the LP on the
   "charge" side of that corner for longer, so SoC actually tops out at 100% on
   most sunny days.

**Why the ramp** (chosen 2026-05-29): the previous flat 0.20pp always-on bias
caused a persistent small-import side effect (~50 Wh/cycle, ~$0.01-0.02/cycle,
~$3-15/year) during pure-self-consume periods, even when SoC sat far from the
top-balance window. Confining the bias to the high-SoC ramp eliminates that
side effect at low/mid SoC while preserving the LP nudge through the
charge→export pivot. The 90% lower endpoint was chosen to ramp in before the
typical late-afternoon pivot point. If the SoC-reaches-100% behaviour regresses
on sunny days, the lower endpoint may need to drop further (e.g. 85%);
empirical validation is needed before relying on the ramp in winter conditions
or after any change to the EMHASS power-cost penalty.

## Persistence helpers

Two `input_number` helpers are written by the script wrappers each run:

- `input_number.dh_last_soc_init` — written by `script.emhass_dayahead_optim`.
  Used by both DH (next run's chain anchor) and MPC (prior plan's start anchor for
  interpolation, because the published `dh_soc_batt_forecast` entity only carries
  end-of-interval values).
- `input_number.mpc_last_soc_init` — written by `script.emhass_mpc`. No current
  consumer; published for diagnostics only.

The legacy `input_number.battery_soc_5_minute` and `input_number.battery_soc_30_minute`
snapshots are **no longer used** by the EMHASS scripts. They remain populated by
external automations as pure diagnostic mirrors of the live SoC at 5- and 30-min
boundaries.

## Relationship to `eval/rolling_mpc_eval.py`

The eval framework simulates a 14h LP very similar to the production MPC, with
several relevant gaps.

What the eval matches:

- 14h × 5-minute LP at each step. ✓
- `--strategic-soc-handoff` with `--strategic-target-mode exact` mirrors the
  inheritance pattern: solve a 72h strategic LP, extract its planned SoC at +14h,
  pass that as the 14h LP terminal target. ✓

What the eval does **not** match:

- The eval's strategic LP solves the 72h horizon with **no terminal SoC constraint
  at +72h**. Production's DH solves with a soft target ~98% at the end (via the
  offset feedback loop).
- The eval does not apply the ramped force-charge top-balance bias (0 → 0.20pp
  across SoC 90–99.99%).
- The eval does not apply the DH self-correction chain across consecutive solves
  (each eval step is a clean re-solve from current SoC).
- The eval does not apply MPC's plan-relative `soc_init` lift; it passes the live
  SoC directly.

Effect of the +72h-constraint gap: the +72h terminal constraint propagates back
through the LP solution, but only weakly. The LP can satisfy "end at 98%" by
recharging anywhere in the back half (between +14h and +72h). It does not
necessarily raise the +14h target. In practice the eval's +14h target is **a few
kWh lower** than production's would be in flat-tail scenarios (e.g. winter when
PD7Day data is missing and the seasonal HoD fallback dominates the back half of
the strategic curve), and **roughly equivalent** in typical scenarios where the
curve itself motivates inventory holding.

This means dispatch evaluations of forecast sources that rely on flat tails (any
pre-2026-02-09 historical window for PD-direct, since that's when PD7Day backfill
starts) should be expected to under-estimate the terminal SoC achievable in
production. Eval-only "battery depleted to 4 kWh" may translate to "battery
preserved to ~10–15 kWh" once production's terminal-end constraint is enforced.

This was noted but not fixed in the 2026-05-08 Phase α-prime Step 4 work — the WA7
SoC depletion observed in eval is plausibly attenuated in live operation by the
production offset feedback. Worth a targeted experiment if the question becomes
load-bearing for a promotion decision, but not blocking shadow-and-compare
deployment.

## What this means for shadow forecasts

Phase α-prime Step 2 publishes PD-direct as a shadow alongside the existing
TFT-backed `ai_shadow` family. The shadow forecasts do not control dispatch;
they are inspected and (eventually) compared. Because the production high-SoC
bias only manifests when the forecast *is* driving dispatch (via the DH solve's
terminal target), the shadow path is unaffected: a shadow forecast can show
whatever shape it naturally produces, and the production bias only matters when
comparing what *would* have happened under each forecast had it been driving DH.

## Open questions

- ~~The HA automation that adjusts `emhass_target_soc_offset` is not synced
  into this repo.~~ Resolved 2026-05-30: now lives as the
  `"EMHASS — Update target SoC offset"` automation in `hass/package-emhass.yaml`.
  The dead `battery_soc_30_minute` clamp from the original Jinja (already
  commented out) was dropped in the migration; the offset remains unclamped.
- A future `eval/rolling_mpc_eval.py` flag (e.g.
  `--strategic-terminal-soc-target-kwh`) could optionally constrain the strategic
  LP to end at ~95–98% SoC, mirroring the production constraint. Not currently
  implemented; deferred until needed for a specific decision.
- The DH self-correction chain (anchor = `dh_last_soc_init` + deviation) is not
  reflected in eval either. Each eval step is a clean re-solve; consecutive solves
  do not share state via a persisted anchor. Expected to matter less than the
  +72h gap, but unmeasured.
- The force-charge bias ramp (90 → 99.99% true SoC) shipped 2026-05-29. The
  90% lower endpoint is provisional — needs empirical validation on sunny days
  through a charge→export pivot window to confirm SoC still reliably reaches
  100%. If it doesn't, the lower endpoint should drop (e.g. 85%).
