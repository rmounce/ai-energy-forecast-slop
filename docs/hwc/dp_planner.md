# HWC DP planner

Status: **default and only optimiser** (`hwc.planner == "dp"`). The heuristic block planner
was removed 2026-06-24; EMHASS (`hwc.planner: "emhass"`) is retained as a fallback.
Code: `hwc_dp_planner.py`. Tests: `tests/unit/test_hwc_dp_planner.py`.

## Why (vs. the now-removed block planner)

- Block planner replan: ~12 s (compressor off) / **~225 s (compressor on)** at 576×5 min —
  seed fan-out × repairs re-simulating the full horizon. Too slow for event-driven replan.
- Heuristic stages (main block → min-temp repair → terminal repair) were greedy, not global;
  `min_temp` was a soft repair that could still be violated.
- DP at 576×5 min: **~350 ms, independent of compressor state**. ~600× faster on the
  compressor-on case. Global optimum within the binned state space.

## Design (2026-06-20 decisions)

- **Pure monetary objective.** No hard min-runtime, no min-lift. Short cycles discouraged
  *only* by a per-start `hwc.transition_cost_aud`. See
  [thermal_characterisation.md] "Planner direction".
- **DP picks the binary on/off sequence only.** Published power/temps come from the exact
  shared `hwc_planner` model (`_refresh_planned_power` + `simulate_block_temperatures` via
  `assemble_plan_dict`). Temp binning is an internal cost/feasibility approximation; it never
  reaches published numbers — the published plan is the exact-model render of the chosen
  on/off sequence.
- **State:** `(temp_bin, compressor_on, satisfied_today)`.
  - The heat rate is a **continuous function of the current modelled temp** (the
    `hwc_planner._heat_rate_c_per_hour` taper), so there is **no `regime` dimension to carry**.
    The former FULL/TOP-UP latch (which carried the block-start regime so a cold reheat kept the
    fast rate past `top_up_start_temp_c`) was removed 2026-06-27: it existed only to hide a step in
    the rate, and the step is now a continuous taper (band `top_up_start_temp_c ± heat_rate_taper_width_c/2`,
    default 50–56 °C). With the rate depending on current temp alone, continuing a run and starting
    a fresh one at the same temp compute the same rate — the asymmetry the 53 °C short-cycle bug
    arbitraged no longer exists. (The earlier two-state `(V_hot, T_hot)` model was the other
    candidate for this; it was shelved — see [2state_soc_model.md](2state_soc_model.md).)
  - `satisfied_today` for the daily 60 C obligation; resets at local midnight.
- **Costs:** import energy + `transition_cost_aud` on each off→on edge.
- **Soft high-penalty obligations (not locks; degrade gracefully on cold start):**
  - `min_temp` floor — penalty per °C below, each step.
  - daily `desired_temp` (60 C) by `main_window_end`, skipped for `main_satisfied_dates`.
  - terminal: small penalty below `terminal_target`.
- **Compressor-on = initial state** (`compressor_on` seeded). No seed enumeration, and no regime to
  seed — the rate is read from the current temp, so a continuing run and a fresh off→on start agree
  at any temp by construction (this is the structural fix for the 53 °C short-cycle limit cycle;
  background in `docs/hwc/reviews/short_cycle_review_2026-06-26.md`).

## Config (`hwc.dp_planner`, all optional; code defaults shown)

- `temp_bin_c` 0.25 — temperature bin width.
- `min_temp_penalty_aud_per_c` 5.0 — penalty per °C below `min_temp`, **per step**. The code
  default 5.0 makes the floor effectively hard (single-step yield price ≈ penalty × heat_rate ÷
  compressor_power ≈ penalty × 9.4 AUD/kWh → ~$47/kWh). **Production overrides to 0.1**
  (yields above ~$0.94/kWh single-step) so the tank rides through high prices; the per-step
  accumulation still limits how long/deep it sits under-floor. Raise toward 0.2–0.3 to make
  under-floor excursions rarer; keep ≥ ~1.0 to treat the floor as near-hard.
- `desired_penalty_aud_per_c` 1.0
- `terminal_penalty_aud_per_c` 0.05
- `survivors_per_state` 1 — DP survivors kept per binned state. **Leave at 1.** `2` also keeps
  the highest-temp ("run a bit longer") path; present only by owner request and **not
  objectively helpful** — see "Multi-survivor" below.
- `soc_model` false — **shelved** opt-in two-state `(V_hot, T_hot)` decision model (routes to
  `_build_dp_plan_soc`). Kept behind the flag for a possible future revisit but **off in
  production**: on the live watch its plans were less plausible than the continuous-rate
  single-temperature path (the build phase flatlines the temperature-driven power model). See
  [2state_soc_model.md](2state_soc_model.md). When on it publishes `soc_v_hot`/`soc_t_hot`/
  `soc_probe` diagnostics and reads a `soc:` sub-dict of model params; the daemon supplies the seed
  `(V_hot0, T_hot0)`. The short-cycle discontinuity it was meant to address is instead handled by
  the continuous `_heat_rate_c_per_hour` taper on the default path.

## Multi-survivor (off by default — kept by request)

Default keeps **one** survivor per state (min-cost). Two alternatives were tried 2026-06-21:

- **Full Pareto frontier (cost↓, temp↑):** unbounded survivors → **7.5 h CPU** on one
  576×5min horizon. Infeasible; not shipped.
- **Bounded 2-survivor (min-cost + max-temp), `survivors_per_state: 2`:** swept start-temp
  44–61 °C × compressor on/off on a real horizon with a $2.87/kWh price spike. Changed
  decisions in **37/70** scenarios but was **net-negative on `objective_cost_aud`**
  (~1 c/plan, mixed sign; worse ~26, better ~11). It errs warmer/safer, never risks an
  obligation, runs ~2× (still sub-second) — so harmless, just not helpful.

Why "more states → worse" isn't a paradox: the DP minimises `J = energy + transition +
soft penalties`; the superset guarantee improves *that*. The reported `objective_cost_aud`
excludes the penalties, so the extra freedom "buys" obligation margin the money metric
prices at zero. If sub-bin precision ever matters, shrink `temp_bin_c` (symmetric), don't
add survivors. The `survivors_per_state: 2` path is retained only because the owner wanted
it available; safe to delete if never enabled.

Reads from `hwc` top-level: `transition_cost_aud`, `main_window_end`, `main_satisfied_dates`
(the last injected at runtime by the daemon). `transition_cost_aud`/`main_window_end` moved up
from the removed `block_planner` section on 2026-06-24.
Reuses from `thermal`: rate/power model (incl. the `top_up_start_temp_c` /
`heat_rate_taper_width_c` rate taper), `min_temp`, `desired_temp`, `max_temp`, `terminal_target`.

## How to A/B

- Flip `config.yaml` `hwc.planner` to `"dp"`, restart `ai-energy-hwc-daemon.service`.
- Output entities/attributes are identical (`sensor.hwc_power_plan`,
  `sensor.hwc_predicted_temp`), so executor + EMHASS integration are unchanged.

## Known limits / TODO

- Binning makes the DP internal temp an approximation of the exact replay; published plan is
  exact. Penalties are tuned, not derived.
- **Reheat durations are knowingly approximate.** A probe-only heat rate provably cannot
  predict per-cycle reheat time/energy ([thermal_characterisation.md] Finding 4: the
  probe-blind build phase is 20–72 % of cycle energy and non-monotone in the start probe).
  The taper smears this under a pessimistic ceiling and closed-loop replanning + the
  hardware-60 ceiling absorb the residual — a deliberate trade, not a bug. **Watch item:**
  planned-vs-actual reheat duration from the SQLite cycle store is the instrument for
  detecting if this variance ever costs real money (overruns into expensive slots, deep-draw
  over-booking); that evidence, not model aesthetics, is what would justify revisiting a
  richer state (see [2state_soc_model.md] for why the last attempt was shelved).
- **Every on-slot is priced at `load_cost` as a grid import** — the DP has no concept of PV
  surplus/curtailment or negative feed-in prices. Known gap with an agreed design:
  [surplus_negative_price.md] (curtailment → DP-planned; negative price → executor
  override).
- **Calibration is winter-fit, single fan-speed regime.** Heat rates, the power curve and the
  taper anchors come from ~18 June-2026 cycles at the quiet fan setting
  ([thermal_characterisation.md] "Fan-speed regime"). Mains temp and wet-bulb drift the
  real rates seasonally; the pessimistic bias buys margin, but expect a re-fit toward summer.
- Compressor-on signal still reads the lagging Tuya binary; switch to Athom ch2 power
  (>~250 W) — separate follow-up (see [thermal_characterisation.md]).
- Daily-60 only (no N-day legionella variant) — low-stakes per midday price/wet-bulb.
