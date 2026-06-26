# Charging Deferral / Phantom-Trough Failure — 2026-06-22

Post-cap day case study: an empty battery, a daytime charging window wasted,
and a forecast bias that no available price source corrected. Companion to the
2026-06-21/22 cap-event review (the APF-liquidation failure mode); this doc is
the *deferral* failure mode that followed it the same morning.

## Caveman Summary

- The 07:00 ACST cap ($20,300/MWh) drained the battery (MPC `soc_init` 4.8% at
  07:00, 1.1% by 07:30). The whole "normal" charging window (≈08:00–17:00 ACST)
  was a recharge problem: the only question was *when* to charge.
- MPC deferred charging all morning, chasing a forecast price trough that sat
  ~3h ahead and kept receding. It barely charged through the genuinely cheap
  window (08:00–12:30, ~12%→25% SoC, mostly solar trickle), then charged hard
  from 13:30 onward — straight into the *rising* afternoon ramp.
- The trough was a forecast artifact. For a fixed 14:00 ACST target the
  buy-price forecast sat pinned at ~$0.23/kWh for seven hours and only
  converged to the realised ~$0.49 in the final 1–2h. It under-called the
  interval by ~2×.
- Independent AEMO SA1 dispatch confirms reality went the other way: a real
  trough at 10:00–10:30 (~$220/MWh), then a steady ramp to ~$650/MWh by 15:30
  (5-min prints to $875–964). Optimal action was charge 08:00–11:00; the
  forecast pointed at 14:00.
- Owner intervened manually via `input_number.emhass_weight_buy_forecast` /
  `..._sell_forecast`, cranking both from baseline (0.1 / 0.2) to max (1.0)
  between 12:47 and 13:55 ACST. This forced charging — but only *after* the
  cheap window had passed.
- Key finding for future direction: the **local** `dynamic_handoff` forecast
  did NOT help. It projected an even deeper midday trough than the Amber APF
  (~$0.13 vs ~$0.23 at the same target) and would have deferred *harder*. This
  is a shared, regime-blind bias — not a "pick a better source" problem.

## Timeline (ACST)

SoC from logged MPC `soc_init`:

| ACST  | 08:00 | 09:00 | 10:30 | 12:00 | 12:30 | 13:00 | 13:30 | 14:30 | 15:30 | 16:30 | 17:00 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| SoC % | 11.9  | 12.8  | 17.8  | 23.7  | 25.1  | 26.5  | 39.3  | 50.8  | 71.9  | 88.3  | 92.1  |

Charging is flat through the cheap window and only ramps after the owner's
manual intervention near 13:00, into the expensive afternoon.

## The phantom trough (fixed-target forecast trace)

MPC optimises against `sensor.ai_price_forecast` (the APF-derived p50/low/high
source, blended by the `k_buy`/`k_sell` knobs). Tracing the import-price
forecast for a **fixed** 14:00 ACST target across successive MPC runs:

```
made 07:00 ACST (7.0h out): $0.228   ← projected
made 09:00 ACST (5.0h out): $0.233
made 11:00 ACST (3.0h out): $0.247   pinned ~$0.23 for SEVEN hours
made 12:00 ACST (2.0h out): $0.280
made 13:00 ACST (1.0h out): $0.391
made 14:00 ACST (0.0h out): $0.489   ← reality, ~2× the morning projection
```

Same shape held for the 13:30 and 15:00 targets: a flat low projection that
only corrects in the last ~2 hours. Per-run, the projected minimum kept
getting both more expensive and later (0.225 → 0.247 → 0.280 → 0.391) — the
signature of an illusory, receding trough.

## Independent confirmation (AEMO SA1 dispatch, 30-min mean of 5-min)

```
ACST    SA1 $/MWh   max 5m
07:00    20300       20300   ← morning cap, battery drained here
07:30     4492       20300
08:00      346         353
09:00      250         299
10:00      222         257   ← genuine trough (charge HERE)
10:30      219         250
11:00      278         310
12:00      297         312
13:00      309         312
13:30      559         964   ← afternoon ramp begins
15:00      581         875
15:30      653         875   ← afternoon peak
16:30      384         491
```

Reality: a real cheap window at 10:00–10:30, then a sustained ramp. The
forecast's "cheap afternoon" was exactly when prices were highest.

## Manual lever-pulls (HA recorder)

Baseline through the morning: `k_buy`=0.1, `k_sell`=0.2,
`weight_battery_discharge`=0.04 (unchanged all day).

```
12:47–12:52  buy 0.1→0.5, sell 0.2→0.5
13:06        buy/sell → 0.6
13:43–13:55  buy/sell ramped to 1.0 (max), held until 16:21
16:21–17:43  wound back down to baseline
```

Mechanically, raising `k_buy` blends the whole import-cost vector toward its
*high* quantile (`load_cost = p50 + (high − p50)·k_buy`), which lifts the
phantom trough until it no longer undercuts "now," so MPC stops deferring. It is
the correct lever but blunt, global, manual, and laggy — the owner spent ~15
minutes hand-cranking it and still charged into the expensive ramp.

## Why the local stack would not have saved it

Checked the local `dynamic_handoff` price forecast (`price_forecast_log.csv`)
for the same fixed targets. Different price basis (lower level), so compare
*shape*, not level. For the 14:00 ACST target:

```
made 09:30 ACST : $0.13
made 12:00 ACST : $0.16
made 13:00 ACST : $0.21
made 14:00 ACST : $0.32
```

It projected an even deeper midday trough than the APF and ramped just as late.
On this day it would have deferred charging *harder*, not less. Both sources
reverted to the normal diurnal solar-trough shape and neither captured that the
post-cap supply tightness would persist into the afternoon.

## HWC corroboration — the same bias hit the hot water

The heat-pump hot water (HWC) DP planner consumes the same APF-derived import
forecast (`sensor.ai_dh_import_price_forecast`) and made the analogous mistake
the same day. Behaviour (ACST):

- Tank rode down 52 → 45 °C by 07:30 (hit the soft `min_temp` floor).
- Floor-rescue heat 07:38–08:14 to 47 °C.
- Sat flat at 47 °C **right through the cheap window (08:00–13:00)**.
- Did its main 47 → 59 °C heat-up at **13:21–15:24, in the expensive afternoon**.

The two benign explanations were checked and neither excuses the timing:

- *COP (wet-bulb)*: rose ~5 °C (08:00) → ~10 °C (14:00), worth only ~+6–7 %
  heat-rate / ~−1 % power — nowhere near the ~2× import-price difference
  ($0.35 → $0.70/kWh).
- *Solar self-consumption*: PV was **better mid-morning** (peaks ~3.6–3.7 kW at
  10:00–10:30) than mid-afternoon (~1 kW at 14:00, ~0.4 kW by 15:30). Solar
  argued for heating earlier; much of the afternoon heat-up was grid import.

The 07:00 cap-tail floor-rescue is *not* a planner fault — the cap was
unforecastable and the soft floor (`min_temp_penalty 0.1`) worked as intended.
Avoidable cost is small (~$0.40–1.00; it is hot water). The value is as
**evidence**: the failure is not battery-specific — the same phantom-trough bias
deferred *every* price-following load through the cheap morning into the
expensive afternoon.

Note on sensitivity: the HWC DP planner carries a per-start `transition_cost_aud`
($0.05/start) that makes it immune to chasing a *momentary* dip (a single 5-min
step would need to be ~$0.81/kWh cheaper to justify a start). That did **not**
protect it here, because the cheap window was a sustained 5-hour block (barrier
~$0.013/kWh) and the miss was a mislocated heating block, not a chased dip. See
`charge_lever_controller_plan_2026-06-22.md` (Trigger A vs Trigger B) for how the
controller signal must therefore act on the forecast vector, not spot price.

## Failure mode (for posterity)

On a tight-supply day following a cap, every available price forecast (Amber
APF and the local LGBM stack alike) reverts to the seasonal/diurnal mean and
projects a midday/afternoon price trough that does not occur. A cost-minimising
MPC rationally defers charging to chase the cheaper-looking trough; the trough
recedes faster than the clock advances; the battery misses the genuinely cheap
morning window and ends up charging into the rising afternoon. The only current
mitigation is the owner manually lifting `k_buy`, which is reactive and too late.

This is distinct from the 2026-06-21 APF-liquidation case (where the APF was
roughly *right* about elevated overnight feed-in and correctly drove discharge,
but gave zero lead time on the caps). Here the APF is *wrong* about the
afternoon, in a way the local stack does not fix.

## Candidate directions (not prescriptions)

These were carried into and refined in `charge_lever_controller_plan_2026-06-22.md`
(which is the live plan; the list below is the as-captured snapshot).

- **Regime awareness**: after a cap / on flagged tight-supply days, automatically
  distrust the projected afternoon trough — widen forecast dispersion or lift the
  cost floor instead of relying on manual `k_buy`.
- **Trough-recession detector**: when the projected minimum gets both more
  expensive and later across successive runs, treat it as illusory and down-weight
  it. *(Now the lead candidate — endogenous, no new data.)*
- ~~**Absolute-cheap opportunism**: charge when prices are already cheap in
  absolute terms.~~ *Rejected by owner: hard-coded $/MWh thresholds don't travel
  across seasons/regimes, and a spot-price trigger is wrong for HWC (transition
  cost). The lever must act on the forecast vector, not spot price.*

## Reproduction

- EMHASS runtime-param dump for the window:
  `docker logs emhass --since 2026-06-21T21:30:00Z --until 2026-06-22T08:30:00Z`
  (each MPC run logs `Passed runtime parameters: {...}` with `soc_init`,
  `load_cost_forecast`, `prod_price_forecast`, `weight_battery_*`).
- AEMO actuals: `rp_5m.aemo_dispatch_sa1_5m` field `price` (InfluxDB).
- Lever history: HA recorder `/api/history/period` for
  `input_number.emhass_weight_buy_forecast` / `..._sell_forecast`.
- Local forecast: `price_forecast_log.csv`, `prediction_type=dynamic_handoff`.
