# Charge-Lever Controller Plan — 2026-06-22

Direction (not yet a build spec) for responding to the charging-deferral /
phantom-trough failure documented in
`charging_deferral_phantom_trough_2026-06-22.md`. Recorded so the reasoning
survives; implementation is deliberately deferred (n=1 event, validate first).

## Core decision

Do **not** try to fix the point price forecast, and do **not** add an absolute
wholesale-price floor (hard-coded $/MWh thresholds don't travel across
seasons/regimes — rejected). Instead, automate the lever the owner pulled by
hand on 22 Jun: the `k_buy` / `k_sell` quantile-blend knobs
(`input_number.emhass_weight_buy_forecast` / `..._sell_forecast`).

Architecture: decouple **signal** from **actuator**.

- **Actuator** — `k_buy` (and `k_sell`). One control surface. `k_buy` blends the
  import-cost vector from p50 toward the high quantile
  (`load_cost = p50 + (high − p50)·k_buy`).
- **Signal(s)** — whatever indicates an unusual / tight regime. The regime work
  (renewable availability, post-cap state) becomes an *input to the lever*, not a
  new forecast feature. "Detect unusual regime" and "respond by trusting the
  upside quantile more" stay cleanly separated.

## Why this is principled, not a workaround

The 22 Jun failure was *overconfidence* in a mean-reverting trough, not merely a
wrong mean. `k_buy` is a control over which quantile MPC trusts. So
"uncertain / tight regime → trust the upside quantile more → raise `k_buy`" is
the standard response: uncertainty → risk aversion → quantile shift. Correcting
the mean fights the symptom; widening which quantile MPC acts on addresses the
pathology. This is also why folding the regime signal into the lever (rather than
into the forecast) is the lighter and more defensible path.

## Thresholds: relocate, don't pretend to eliminate

A signal→`k_buy` map still has parameters. The goal is to put them on
**relative / normalised** quantities, not absolute ones:

- *Trough recession* (endogenous): the projected minimum of `load_cost_forecast`
  rose AND moved later across N successive MPC runs. Unit-free, self-calibrating,
  needs no new data. On 22 Jun the projected min ran 0.225 → 0.247 → 0.280 →
  0.391 over the morning — a clean monotone signature.
- *Availability anomaly* (exogenous): renewable availability (STPASA / UIGF, per
  `aemo_renewable_availability_discovery_2026-06-14.md`) as a z-score vs that
  source's own recent distribution. Travels across seasons.
- *Post-cap state*: simplest possible regime flag — hold a higher `k_buy` floor
  for the rest of a day on which a dispatch cap occurred.

Absolute wholesale-price thresholds are explicitly out of scope — that is the
part of the rejected option the owner objected to, and relative signals fix it.

## Wire the signal to the forecast, not to spot price (Trigger A vs Trigger B)

There are two distinct ways a signal could move a load, and they behave very
differently across the battery and the HWC planner:

- **Trigger A — "the current interval is cheap, act now."** Reacts to realised
  spot price of the present step.
- **Trigger B — reshape the whole forecast vector (raise `k_buy`), and let the
  optimiser's *cheapest block* move.** This is what the manual lever does, and
  what this controller automates.

The HWC DP planner (now in the sibling `../hwc` repository) carries a per-start `transition_cost_aud`
($0.05/start, no hard minimum-runtime — 2026-06-20 decision) and is **immune to
Trigger A by design**: a single 5-min cheap step would have to be ~$0.81/kWh
cheaper than heating elsewhere to justify a dedicated start (0.74 kW × 5 min =
0.062 kWh; $0.05 / 0.062 kWh). The barrier amortises away over a block
(~$0.07/kWh over 1 h, ~$0.013/kWh over 5 h), so HWC will commit a *sustained*
cheap block but never chase a momentary dip. The battery has ~no transition cost
and would chase single intervals — so an opportunistic spot-price trigger is
wrong for HWC and only debatable for the battery.

Trigger B is **neutral to the transition cost**: HWC commits ~one heating block
per day regardless; reshaping the forecast only moves *where* that block lands.
And because `k_buy` blends each interval toward *its own* high quantile, it lifts
wide-uncertainty far-out intervals (the phantom afternoon trough) more than the
tight near-term morning — which is exactly what pulls the cheapest block forward.
This is why the same lever helps both loads, and why the 22 Jun HWC miss was
*not* shielded by the transition cost: the cheap window was a sustained 5-hour
block, so the barrier was negligible and the failure was a mislocated block, not
a chased dip (see the HWC corroboration in
`charging_deferral_phantom_trough_2026-06-22.md`).

Implication for wiring: the controller signal must act on the **forecast vector
(Trigger B)** — never on spot price. A spot-price trigger would be correctly
ignored by HWC and would make the battery twitchy. One signal feeding the
forecast-reshaping lever serves both loads; an opportunistic current-price path
serves neither well.

## Candidate signal priority

1. **Endogenous trough-recession detector first** — free, no new data, directly
   encodes "this trough is illusory." Build and evaluate this alone.
2. **STPASA availability anomaly** — only if recession-detection alone proves too
   slow or too noisy on replay. This is the first concrete test of whether the
   renewable-availability feature earns its place.
3. **Post-cap floor** — cheap backstop / sanity bound, can co-exist.

## Validation before any closed loop

This converts a manually-driven knob into an automatic loop on top of EMHASS, so
validate the same way counterfactuals have been validated here:

- **Shadow-log** what `k_buy` *would* have been set to from the signal(s); do not
  let it act yet.
- **Replay** against 22 Jun (must lift `k_buy` through the morning) and several
  quiet days (must NOT twitch).
- Only promote to acting once the recession signal is shown to lift on 22 Jun
  without false positives on normal days.

Open question to answer with the replay: **does the endogenous recession signal
alone suffice, or is the exogenous availability input genuinely needed?**

## Explicitly deferred / out of scope

- No absolute price-threshold charging rule.
- No model retrain / new production forecast feature for this.
- No raising the *default* `k_buy` floor globally — that taxes every normal day
  to fix the rare one. Response must be conditional, not a blanket shift.
- n=1: do not retune the broader forecast off a single event.
