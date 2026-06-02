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

## 3. `exhaust_temperature` is confounded, not exogenous (Review Q2)

`power = f(exhaust, wet_bulb)` and `exhaust = f(hot_fraction)` are not independent. Exhaust/
condensing temp rises largely *because* the water it rejects heat into is getting hotter — it's
partly an **output** of tank state, not a driver. Regress power on exhaust and you'll get a fit,
but it's partly circular and won't answer "what power will I draw starting from temp X."

For a fixed-speed compressor the cleaner causal chain is condensing ≈ f(water temp being heated),
evaporating ≈ f(wet-bulb/coil). Model condensing off the **hot-layer water temperature** and keep
`exhaust` as a validation diagnostic. So: lean to "observed diagnostic," as Q2 already half-suspects.

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

## 5. No power meter caps everything above

Row `2026-05-30` (baseline 820 W, hp_mean 130 W, COP 12.6, correctly `clean=False`) shows one
contaminated baseline nuking a cycle. Until there's a real circuit meter, every watt/COP number
carries a baseline-subtraction error bar that **more cycles won't shrink**. State in the roadmap
that step 4 (power-curve fit) is gated on metering quality, not cycle count.

Reframe Review Q7: track **coverage** (wet-bulb × start-temp × fan-regime), not a cycle count.
Twenty single-regime cycles won't help; five across regimes might.

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
