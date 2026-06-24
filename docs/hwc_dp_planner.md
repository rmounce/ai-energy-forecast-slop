# HWC DP planner

Status: **implemented, opt-in** (`hwc.planner == "dp"`); production default stays `block`.
Code: `hwc_dp_planner.py`. Tests: `tests/unit/test_hwc_dp_planner.py`.

## Why

- Block planner replan: ~12 s (compressor off) / **~225 s (compressor on)** at 576×5 min —
  seed fan-out × repairs re-simulating the full horizon. Too slow for event-driven replan.
- Heuristic stages (main block → min-temp repair → terminal repair) are greedy, not global;
  `min_temp` was a soft repair that could still be violated.
- DP at 576×5 min: **~350 ms, independent of compressor state**. ~600× faster on the
  compressor-on case. Global optimum within the binned state space.

## Design (2026-06-20 decisions)

- **Pure monetary objective.** No hard min-runtime, no min-lift. Short cycles discouraged
  *only* by a per-start `block_planner.transition_cost_aud`. See
  [hwc_thermal_characterisation.md] "Planner direction".
- **DP picks the binary on/off sequence only.** Published power/temps come from the exact
  `hwc_planner` model (`_refresh_planned_power` + `simulate_block_temperatures` via
  `assemble_plan_dict`). Temp binning is an internal cost/feasibility approximation; it never
  reaches published numbers. A DP plan is byte-for-byte comparable to a block plan.
- **State:** `(temp_bin, compressor_on, regime, satisfied_today)`.
  - `regime` (full-reheat vs top-up) carried because the heat-rate model latches on the
    *block-start* temp (cold reheat keeps full rate past `top_up_start_temp_c`).
  - `satisfied_today` for the daily 60 C obligation; resets at local midnight.
- **Costs:** import energy + `transition_cost_aud` on each off→on edge.
- **Soft high-penalty obligations (not locks; degrade gracefully on cold start):**
  - `min_temp` floor — penalty per °C below, each step.
  - daily `desired_temp` (60 C) by `main_window_end`, skipped for `main_satisfied_dates`.
  - terminal: small penalty below `terminal_target`.
- **Compressor-on = initial state** (regime + `compressor_on` seeded). No seed enumeration.

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

Reuses from `block_planner`: `transition_cost_aud`, `main_window_end`, `main_satisfied_dates`.
Reuses from `thermal`: rate/power model, `min_temp`, `desired_temp`, `max_temp`,
`top_up_start_temp_c`, `terminal_target`.

## How to A/B

- Flip `config.yaml` `hwc.planner` to `"dp"`, restart `ai-energy-hwc-daemon.service`.
- Output entities/attributes are identical (`sensor.hwc_power_plan`,
  `sensor.hwc_predicted_temp`), so executor + EMHASS integration are unchanged.

## Known limits / TODO

- Binning makes the DP internal temp an approximation of the exact replay; published plan is
  exact. Penalties are tuned, not derived.
- Compressor-on signal still reads the lagging Tuya binary; switch to Athom ch2 power
  (>~250 W) — separate follow-up (see [hwc_thermal_characterisation.md]).
- Daily-60 only (no N-day legionella variant) — low-stakes per midday price/wet-bulb.
