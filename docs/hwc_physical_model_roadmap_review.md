# HWC Physical Model Roadmap — review

Previous-implementer review of `docs/hwc_physical_model_roadmap.md` (the roadmap invites it).
Grounded in the actual code (`hwc_stratified_model.py`, `hwc_planner.py`, `services/hwc_daemon.py`)
and the calibration snapshots (`docs/hwc_calibration_cycles.md`, `docs/hwc_model_fit.md`,
`data/hwc_cop_cycles.csv`) as of `083f16f`.

**Use this file for back-and-forth.** Reply inline under each point (e.g. `> RESP:`); I'll keep
my replies under `>> PREV:`.

## Verdict

The roadmap is well-shaped. The staged path (extract → validate → fit → shadow → integrate),
the guardrails, and the honesty about a 5-cycle sample are all right. My critique is mostly
"the data already in the repo should change two of your conclusions," plus one live-safety flag.

## 1. The fit has quietly collapsed the stratified model back toward single-node

`hwc_model_fit.md` estimates `thermocline_width_fraction = 0.63`, `probe_height_fraction = 0.62`.
Put those into `probe_temp_c()`: the probe starts rising at `hot_fraction ≈ 0.305` and saturates
at `≈ 0.935` — a gentle ramp across essentially the whole charge, **not** the "flat, then sharp
jump when the thermocline arrives" story in the narrative. So the fitted two-layer model behaves
almost like a single-node tank with a first-order lag.

Not a reason to abandon stratification — but it reframes Review Q1. The validator (step 2) should
explicitly benchmark **stratified vs single-node-with-lag** shadow error. If the extra state
doesn't measurably cut error, it's parameters you have to defend against 5 cycles. The fit is
hinting it may not earn its keep yet.

> RESP: Agreed. The current `probe_height_fraction` / `thermocline_width_fraction` estimates
> should be treated as weak hints, not as a defended physical fit. The validator should include
> at least three baselines: current single-node, single-node-with-lag, and the two-layer model.
> If the two-layer state does not materially improve probe trajectory and end-state error, it
> should remain an explanatory model rather than becoming planner state.

## 2. Fit-method units error: time-fraction ≠ hot-fraction

`probe_height_fraction` is derived "from median 50% probe-rise *timing* as a fraction of cycle
duration." But `probe_temp_c()` is a function of `hot_fraction` (an *energy* state), not elapsed
time. Time maps to hot-fraction linearly only if heat-input rate **and** lift are constant — and
they aren't: exhaust/condensing temp climbs 30→82 °C through a cycle, so both instantaneous
useful-heat-to-tank and the effective lift change, plus standing loss and draw-off. You're seeding
a state-space parameter with a time-domain statistic.

The replay-fit in steps 2–3 is the right mechanism — just don't pre-anchor on the timing estimate
as if it's physically meaningful; let the replay own those parameters. (This is the honest answer
to Review Q3: probe-rise timing alone is not enough.)

> RESP: Agreed. This is the strongest technical correction in the review. I used timing fraction
> as a convenient seed, but the model parameter is state/energy-space, not time-space. The roadmap
> should demote those values to diagnostics and require replay fitting before using them. The fit
> report should probably label them "initial guesses" or "shape diagnostics" rather than
> "parameter estimates".

## 3. `exhaust_temperature` is confounded, not exogenous (Review Q2)

`power = f(exhaust, wet_bulb)` and `exhaust = f(hot_fraction)` are not independent. Exhaust/
condensing temp rises largely *because* the water it rejects heat into is getting hotter — it's
partly an **output** of tank state, not a driver. Regress power on exhaust and you'll get a fit,
but it's partly circular and won't answer "what power will I draw starting from temp X."

For a fixed-speed compressor the cleaner causal chain is condensing ≈ f(water temp being heated),
evaporating ≈ f(wet-bulb/coil). Model condensing off the **hot-layer water temperature** and keep
`exhaust` as a validation diagnostic. So: lean to "observed diagnostic," as Q2 already half-suspects.

> RESP: Agreed with the causal framing. `exhaust_temperature` is valuable because it is close to
> the hidden condensing state, but fitting `power = f(exhaust, wet_bulb)` and then using exhaust as
> an independent planning input would be circular. Better: model condensing/exhaust as an output of
> tank state plus evaporator condition, use observed exhaust for validation, and only then derive
> power/COP. A purely empirical power regression may still be useful for sanity checks on observed
> cycles, but not as the planner's causal model.

## 4. At n=5, fit what you measure — not COP (Review Q5) — strongest recommendation

The `thermal_kwh`/COP column is computed from **probe ΔT** + standing loss — but the probe doesn't
represent tank energy mid-reheat (that's the whole point of stratification). So per-cycle COP
inherits exactly the bias the stratified model exists to fix. The spread shows it: nominally-clean
full reheats give COP 1.75 / 2.56 / 3.03 under similar conditions — that's probe-energy mismatch,
not real COP variance.

Meanwhile `elec_kwh` (∫ baseline-subtracted power over compressor-on) and `duration` are measured
**directly**. So prefer the empirical `duration/kWh-to-target` model over a fitted COP model — it's
robust at small N because it doesn't route through the probe. Promote it to the primary near-term
fit; treat physical COP as the later, metering-gated goal.

> RESP: Agreed. Near-term modelling should fit observed duration and electrical kWh to target,
> not inferred COP. COP remains useful language for interpreting efficiency, but it should not be
> the optimisation target while thermal output is estimated from the same misleading probe we are
> trying to correct. I would update the roadmap order so empirical duration/kWh-to-target comes
> before physical COP fitting.

## 5. No power meter caps everything above

Row `2026-05-30` (baseline 820 W, hp_mean 130 W, COP 12.6, correctly `clean=False`) shows one
contaminated baseline nuking a cycle. Until there's a real circuit meter, every watt/COP number
carries a baseline-subtraction error bar that **more cycles won't shrink**. State in the roadmap
that step 4 (power-curve fit) is gated on metering quality, not cycle count.

Reframe Review Q7: track **coverage** (wet-bulb × start-temp × fan-regime), not a cycle count.
Twenty single-regime cycles won't help; five across regimes might.

> RESP: Agreed. Cycle count alone is the wrong gate. We need a coverage table by start/probe temp,
> target, wet-bulb/evaporator proxy, fan regime, and contamination status. The power-curve step
> should be explicitly gated on metering quality. With `remaining_power_load`, we can do broad
> cycle-level electrical-energy estimates on clean windows, but detailed within-cycle power curves
> are fragile.

## 6. Live-safety flag: we're already paying for the thing the analysis told us to stop

From the code, not the roadmap:

- **Actuation is enabled (`c8fcc7e`) and the live planner targets 60 °C daily** — `desired_temp:
  60`, terminal repair to 60, and no legionella-cadence logic in `hwc_planner.py`. The COP analysis
  concluded the 55→60 tail runs at COP ~1.75–1.89 and routine should be ~55 with *periodic* 60.
  **The expensive behaviour we identified is exactly what's running live.** The roadmap defers this
  to step 8, but "routine 55 + weekly 60" is a pure config-policy change, decoupled from the
  stratified model, and is the biggest immediate $-win. Pull it forward and do it now.
- **Failure-mode fallback (Q10):** the daemon has reconnect/backoff and the post-heat off-
  suppression grace (good), but I don't see an explicit "stale plan / lost HA → guaranteed fixed
  daytime reheat" path. If a stale plan just stops issuing commands, the unit can be stranded off.
  Agreed principle was *never leave the tank cold* — confirm the daemon degrades to a fixed window
  rather than silently going quiet.

> RESP: Partly agreed, with one owner-policy caveat. Technically, yes: live actuation is enabled
> and `desired_temp` is `60`. The daemon now suppresses already-satisfied local dates via
> `last_reached_target_at`, and `terminal_target` is `current`, but the daily main target is still
> `60`. The efficient policy is likely routine lower target plus explicit periodic `60`.
>
> I would not silently change that to weekly `60` without owner confirmation because the owner
> previously framed reaching `60` as the main-run completion/safety criterion. But I agree this
> should be pulled forward as a near-term config-policy decision independent of stratified modelling:
> implement explicit cadence support, make routine target and sanitation target separate, then choose
> the cadence deliberately.
>
> On stale-plan fallback: confirmed. There is heartbeat-triggered replanning and WebSocket
> reconnect/backoff, but no explicit "planner unavailable / HA unavailable / stale plan -> fixed
> daytime safe reheat" fallback. If planning keeps failing, `_run_planner()` logs and returns; if
> executor cannot load a valid plan, `_run_executor()` logs and returns. That does not satisfy
> "never leave the tank cold". This should be a near-term safety issue before relying on the daemon.

## 7. Smaller notes

- `apply_heat` pins the hot layer to `hot_target_c = 60`, but the real hot layer exceeds 60
  (exhaust to 82). Harmless for probe prediction; wrong if you later compute *stored useful energy*
  — fix before step 8.
- Capturing `element_on`/`defrost_on`/`four_way_on` for exclusion (Review Q6) is exactly right. Add
  overlapping-household-load contamination (the 05-30 baseline blowout) to the exclusion list
  explicitly.
- The guardrail "keep execution conservative until shadow is better" is in mild tension with
  daily-60 being live. Resolve it in words: live = the dumb-but-safe block planner (and lower its
  target now); stratified = shadow-only until it beats live.

> RESP: Agreed on all three. `hot_target_c=60` is acceptable for probe-timing experiments but wrong
> for stored-energy accounting once we care about water above 60 or high-temperature top layers.
> Household-load contamination should be promoted to an explicit exclusion criterion. And the
> guardrail wording should distinguish the live simple planner from the stratified model: live may
> change by explicit owner-approved policy/config; stratified remains shadow-only until validated.

## Answers to the roadmap's Review Questions

1. **Two-layer too simple / right first abstraction?** Right abstraction *in principle*, but the
   current fit nearly degenerates to single-node-with-lag (§1). Make the validator prove the extra
   state pays before building planning on it.
2. **`exhaust` as condensing proxy?** Diagnostic, not driver — it's confounded with tank state (§3).
3. **Probe-rise timing enough to fit the two params?** No — timing ≠ hot-fraction (§2). Fit by replay.
4. **First prod improvement: prediction only, or also scheduling?** Prediction/shadow only for the
   stratified model. But do one scheduling change now that needs no model: routine 55 + periodic 60 (§6).
5. **Empirical duration/kWh more robust than physical COP at this N?** Yes, clearly (§4).
6. **Regimes to exclude?** fan-speed change, defrost, element, four-way-valve, overlapping household
   load, unusual draw-offs, autonomous safety cycles — all yes. Element/defrost/four-way already
   flagged in the table; add household-load contamination.
7. **Minimum dataset before changing live scheduling?** Wrong axis — measure regime coverage, not
   count (§5). And the 55/60 policy change doesn't need any of it.

## Suggested near-term order (delta to the roadmap)

1. Config-only now: routine 55 °C + periodic (weekly) 60 °C; floor ≥ ~48. Independent of the model.
2. Confirm/repair the daemon stale-plan fallback to a fixed daytime window.
3. Steps 1–2 of the roadmap as written (incremental extractor, replay validator) — but have the
   validator benchmark stratified **vs** single-node-with-lag.
4. Swap the primary fit target from COP to empirical kWh/duration-to-target.
5. Power-curve / physical-COP work: gate on a real meter, not on cycle count.

— Previous implementer (modelling/characterisation phase).

> RESP: Proposed merged near-term order:
>
> 1. Decide owner policy for routine target and sanitation cadence. I support implementing the code
>    so routine target and `60 C` cadence are explicit, but the actual cadence should be a deliberate
>    owner choice rather than inferred from the model discussion.
> 2. Add daemon stale-plan/failure fallback. This is a safety/control reliability issue, not a model
>    accuracy issue.
> 3. Add incremental cycle extraction and contamination/coverage reporting.
> 4. Build replay validator with current single-node, single-node-with-lag, and two-layer benchmarks.
> 5. Make empirical duration/electrical-kWh-to-target the primary near-term fit.
> 6. Keep power-curve/COP work as a diagnostic until metering quality is better or the clean-proxy
>    dataset has enough regime coverage.
