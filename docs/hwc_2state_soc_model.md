# HWC two-state `(V_hot, T_hot)` tank model — design spec

Status: **agreed design, not yet implemented** (2026-06-26). Supersedes the FULL/TOP-UP
heat-rate latch in `hwc_dp_planner.py` once built. Empirical basis:
[hwc_thermal_characterisation.md](hwc_thermal_characterisation.md) Findings 3–5. Root-cause that
motivated the rewrite: [hwc_short_cycle_review_2026-06-26.md](hwc_short_cycle_review_2026-06-26.md).

## Why a second state

The current model indexes heat-rate on the single mid-tank probe via a binary FULL/TOP-UP regime
latched at `top_up_start_temp_c = 53 °C`. Two problems, both measured:

- **The probe does not determine reheat work.** Across 18 metered reheats the probe-blind build
  phase (compressor running while the probe is ~flat, heating the hot zone *above* the sensor)
  averages ~42 % of cycle energy and is **non-monotone in the start probe** — a 52.0 °C start had
  the largest blind phase (77 min), a 53.6 °C start had 30 min (Finding 4). A probe-only heat-rate
  curve provably cannot predict reheat time/energy.
- **The 53 °C latch is a discontinuity the DP arbitrages across replans** → the short-cycle limit
  cycle. Any boundary that flips a rate/cost as a step is structurally the same hazard.

A stratified tank needs **quantity** and **temperature** of the hot zone as separate states.

## State

`(V_hot, T_hot)` plus the existing scheduling flags:

- `V_hot ∈ [0, 1]` — fraction of the tank above the thermocline (the hot zone).
- `T_hot` — temperature of that hot zone (°C).
- `compressor_on`, `satisfied_today` — as today.
- `T_mains` — model **parameter** (no water-side sensor logs); seasonal constant.

DP state becomes `(V_hot_bin, T_hot_bin, compressor_on, satisfied_today)`. The `regime` dimension
is **removed** — continuity replaces the latch.

## Dynamics (compressor on)

Two physical regimes, continuous (no latch), driven by where the energy goes:

- **Build (blind):** while the hot zone has not filled the tank, delivered heat grows `V_hot` at
  ~constant `T_hot` (the condenser pushes water to a roughly fixed delivery temp; high COP, low
  condensing temp). `dV_hot/dt = COP(T_cond)·P_elec / (cap · (T_hot − T_mains))`.
- **Rise:** once `V_hot = 1`, further heat raises `T_hot` (53 → 60), COP collapsing with condensing
  temp (Finding 3: ~2.3 → 1.75). `dT_hot/dt = COP(T_cond)·P_elec / cap_full`.

`cap = 222 L · 0.997 · 4.186 / 3600 ≈ 0.257 kWh/K`. `COP` is indexed on the **exhaust/condensing
temperature** (Findings 2–4), not the quantised probe. `P_elec` from the published power model.

## Observation map (probe)

`probe = g(V_hot, T_hot, T_mains)` — the probe reads `T_hot` when the thermocline is above the
sensor (`V_hot` above sensor height ≈ 0.50) and `T_mains` below, through a **finite-width smooth
transition** (sigmoid / linear ramp), *not* a hard threshold.

> **Critical:** the transition must be smooth. A step at `V_hot ≈ 0.5` is structurally identical to
> the 53 °C FULL/TOP-UP latch and would let the DP arbitrage the boundary across replans — the same
> short-cycle bug, reincarnated. The map lives in the **observation**, never in the cost. Real
> finite width is supported by the draw cliff bottoming at ~35 °C, not mains (Finding 5).

Sensor height is uncertain (manual implies 45–55 %); modelled at **0.50** (split the difference).

## Standing loss & draws

- **Standing loss** ≈ 0.3 °C/h (~75 W) upper bound (Finding 5), acting mainly on `T_hot` plus slow
  thermocline diffusion eroding `V_hot`.
- **Draws** drop `V_hot` at fixed `T_hot` (hot off the top, mains in the bottom). Exogenous; in the
  planner they come from a **conservative (pessimistic) daily draw prior** — see below.

## Production architecture (the unobservability is embraced, not estimated)

`V_hot` is unobservable in the interior (the probe can't see it; no flow sensor). We do **not** build
a state observer — it would drift and lie, because draws that stay above the probe are invisible to
it. Instead:

1. **Conservative draw prior.** Between resets, decrement `V_hot` by an *over-estimating* daily draw
   schedule — not a historical average. A heavier-than-average draw that stays above the probe fires
   no watermark and is otherwise invisible, so the prior is the only guard; it must assume the worst.
2. **Watermark resets** (the state is exactly observable at two crossings; snap to them, drift
   erases):
   - **Top** (`V_hot → 1`): compressor reaches 60 °C and cycles off.
   - **Middle** (`V_hot → ~0.5`): probe crosses the cliff downward past the sensor.
3. **Planner.** DP simulates the `(V_hot, T_hot)` physics forward from the tracked state under the
   conservative draw prior, picking the cheapest on/off sequence to meet the daily probe-60
   obligation before the unit's own internal timer would force it at an expensive time.
4. **Executor.** Heats when the DP says "heat"; stops when the DP says "off" **or** the hardware
   hits 60 °C, whichever is first. This decouples the health/obligation target from estimator
   precision — we can never fall short of 60 regardless of a bad `V_hot` estimate.

**Why pessimism is free.** Asymmetric cost: under-budgeting `V_hot` → the run overshoots its booked
slots → panic-buy peak power (expensive). Over-budgeting → the tank hits 60 early, the executor
stops, and the unused (cheap, contiguous) slots vanish at ≈zero cost. The physical ceiling caps the
run. (≈zero, not exactly: heating to 60 early eats some standing loss before the draw, and slot
prices aren't perfectly flat — second-order.)

**The bias is economic, not a safety margin (decided 2026-06-26, reviewer sign-off).** The step-1
replay's ~3 °C-low prediction (the sharp build→rise boundary) is **kept on purpose**. The earlier
"a 1 °C-optimistic miss lands at 59 °C and fails Legionella" framing is wrong here: the system is
**closed-loop** — we replan continuously (the next tick sees probe < 60 and books more) and the
Aquatech forces its own daily probe-60 cycle as a hardware backstop, so health compliance is
structurally guaranteed regardless of model error. Bias in either direction is therefore an
**economic** error, not a compliance one. Pessimism remains correct because the asymmetry is
economic: the hardware-60 ceiling truncates over-provisioning for ≈free, while under-provisioning
forces make-up heat at a replan-chosen, possibly-adverse time. In the SA duck-curve
**declining-price-then-spike** case (a hard 16:00 step), a pessimistic DP front-loads extra cheap
slots — placed optimally by the price-aware DP, truncated by the hardware once the tank hits 60 —
and is safely full *before* the spike; a centred model gets caught short *into* the spike. So the
bias is a temporal buffer against price cliffs. Hence: do **not** centre the boundary (Option 1).
`T_hot0`-as-delivery-temp (Option 2) stays in the back pocket for the one place pessimism could cost
real money — a deep draw making the DP hallucinate a ~4 h recovery and panic-buy peak — a one-line
fix to deploy only if that's observed.

## What still needs fitting (as data accumulates)

The structure is settled; parameters are first-cut. Refine via `hwc_soc_extract.py batch` /
`hwc_soc_calibrate.py` as cycles log:

- Blind-phase COP vs condensing temp (the probe can't give it; needs the exhaust proxy).
- `g` crossing height + transition width from more draw events (one deep event so far).
- `T_mains` seasonal value; standing-loss split between `T_hot` and `V_hot` (needs a no-draw window).
- The conservative draw prior (magnitude + timing) from the draw history.
- **The build→rise boundary** (surfaced by the step-1 replay): the sharp all-volume-then-all-temp
  switch under-reaches 60 °C by ~3 °C. A blended transition region near the thermocline crossing,
  and/or tracking `T_hot` as the condenser **delivery temperature** rather than the under-reading
  mid-probe, are the candidate fixes — a model-structure call for the reviewer, not a parameter.

## Implementation sketch

1. **DONE (2026-06-26)** — Standalone forward model `step(V_hot, T_hot, on, dt) -> (V_hot', T_hot')`
   + `g()` observation map, as a pure module **with no DP wiring**: `hwc_soc_model.py`
   (`SoCParams`, `step`, `probe_temp`, `apply_draw`, `cop_rise`, `replay_reheat`, `fit_v_hot0`).
   Unit-tested in `tests/unit/test_hwc_soc_model.py` (18 deterministic invariants: finite-width
   smooth `g`, the 0.50 midpoint = ~35 °C cliff anchor, build-before-rise ordering, boundary-split
   energy conservation, standing-loss rate, draws, clamping, the `V_hot0`-from-blind-energy
   estimator, synthetic replay). Validated against real cycles via `hwc_soc_calibrate.py` over the
   metered span (**all 18 reheats**, first-cut parameters):
   - **`--mode replay` — the honest, predictive validation.** `V_hot0` is *not* fit to the probe
     (circular); it is inferred from the **measured blind-phase energy** (`v_hot0_from_blind_energy`:
     the energy delivered while the probe is flat reveals the cold water displaced), making the
     build duration match by construction and the rise phase + final probe a genuine **prediction**:
     final-probe |err| **mean 2.8 °C / max 7.2 °C**, systematically ~3 °C *under* 60. (Fitting
     `V_hot0` to the probe instead flatters to ~1.2 °C, but that latent then absorbs the error — not
     a real test.)
   - **The systematic under-bias is the sharp build→rise boundary.** The first-cut model sends *all*
     energy to `V_hot` until `V_hot = 1`, then *all* to `T_hot`; pinning `V_hot0` to the blind energy
     leaves only the rise energy for temperature, while the real tank blends the two near the
     thermocline crossing (some `T_hot` rise during late build). So the model reaches 60 a little
     short. **Next levers (deferred — reviewer/data):** soften the build→rise boundary (a blended
     transition region), and/or carry `T_hot0` as the condenser **delivery temperature** (~53–54 °C)
     rather than the mid-probe `p0` — the probe under-reads `T_hot` whenever it sits below the
     thermocline, which inflates the apparent rise the model must produce.
   - **`--mode fit`** calibrates the probe-map geometry (`sensor_height`, `g_width`) under
     energy-pinned `V_hot0`; on current data it slides to the bound (`sensor_height → 0.55`), i.e.
     the ~18 cycles don't yet pin it — only the few deep-draw starts put the probe *in* the
     transition where height is identifiable. Defaults stay at the physical anchors (0.50 / 0.10 /
     `cop_build` 2.6); don't bake in the bound-hitting fit.
2. **IN PROGRESS — DP core done (slice 1, 2026-06-26, flag-gated, off by default).**
   `hwc.dp_planner.soc_model: true` routes `build_dp_plan` to `_build_dp_plan_soc`: DP state
   `(v_hot_bin, t_hot_bin, on, sat)` (no `regime`), transitions via `hwc_soc_model.step`,
   obligations on `probe_temp(V_hot, T_hot)`. The seed `(V_hot0, T_hot0)` is a new `soc_state0`
   param (the daemon tracker supplies it in slice 2; standalone falls back to probe≈T_hot +
   a conservative `V_hot`). **Render unchanged** — published power/temps still come from
   `_refresh_planned_power`/`assemble_plan_dict` off the chosen binary, so the executor/EMHASS
   contract is identical; the chosen binary is additionally forward-simulated to attach
   `soc_v_hot`/`soc_t_hot`/`soc_probe` diagnostics (visual only). Flag-off path byte-identical
   (existing DP suite is the guard). Tests: `tests/unit/test_hwc_dp_planner_soc.py`. The legacy
   `regime` path stays the default until parity; `_build_dp_plan_soc` duplicates scaffolding for
   the migration window and folds back when the latch is deleted.
3. **NEXT (slice 2)** — `V_hot` tracker (conservative prior + watermark resets) in `HwcDaemon`
   (`services/hwc_daemon.py`, persisted in `data/hwc_daemon_state.json`), feeding `soc_state0` and
   publishing the `soc_*` diagnostics to HA.
4. Executor hardware-60 ceiling (partly present as the existing min-off/grace logic).
5. Migrate, don't alias: delete FULL/TOP-UP + the legacy DP path once parity is shown on the
   metered reheats and in live shadow.

Steps 3–5 still gated on owner go-ahead; slice 1 ships behind an off-by-default flag.
