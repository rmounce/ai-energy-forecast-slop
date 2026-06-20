# HWC planner — independent review response (2026-06-20)

Response to `docs/hwc_planner_review_brief_2026-06-20.md`. Scope reviewed:
`hwc_planner.py`, `hwc_executor.py`, `services/hwc_daemon.py`, `config.json`,
`hass/packages/emhass.yaml`, `docs/hwc_thermal_characterisation.md`, and the three
HWC unit-test modules. All 59 unit tests pass; the brief's "fixed bugs" are genuinely
fixed and the EMHASS integration matches its description.

## Headline: the planner misses its own latency budget when the compressor is on

Measured `build_block_plan` at live resolution (576 steps × 5 min, real `config.json`
thermal params, synthetic diurnal price/weather):

| initial state | wall time |
|---|---|
| `compressor_initially_on=False` | ~12 s |
| `compressor_initially_on=True` | ~225 s (3.7 min) |

The daemon sets `compressor_initially_on=True` whenever the unit runs and arms a replan
on every tank move ≥ 0.3 °C. `minimum_replan_interval_seconds = 60`. So during an active
reheat — when the tank moves fastest and triggers the most replans — one replan takes
~4 min and pins a core. Plans go stale exactly when they matter most. `nice -n 19` keeps
the box responsive but does not remove the lag.

Two compounding structural causes:

1. **Seed fan-out** (`_running_compressor_seed_schedules`): when on, builds ~17 seeds
   (one per possible stop slot at 5-min) and runs the full 3-stage repair pipeline on
   each — the ~18× gap above.
2. **Repairs re-simulate the whole horizon inside their inner loops.**
   `_repair_min_temperature` is ~O(lookback² · n) per pass × 8 passes; with
   `lookback = round(18/(5/60)) = 216` and n = 576 that is ~200 M float ops in pure
   Python per replan, before the seed multiplier.

This is the strongest single argument for the DP rewrite: the heuristic is the wrong
computational shape for a 5-min/48-h grid. A DP over (time, temp-bin, on/off) is
O(n · bins · 2) and makes "compressor currently running" a free initial condition with
no seed enumeration.

Interim mitigations if the rewrite is deferred: cap seeds to {stop-now, finish-block,
stop-at-next-price-edge}; cache one base simulation per stage and apply incremental
updates; clamp the min-temp `lookback` to a physical pre-heat window (~9 h, not 18 h).

## min_temp violation is a specific tuning trap

`_repair_min_temperature` heats only up to `boost_target` and only in the overnight
window by default. Heating to a `boost_target` equal to (or barely above) `min_temp`
**cannot** hold `min_temp` once standing loss + the morning draw-off are applied before
the 10:00 main window — reproduced in the synthetic run (min_pred 49.0–49.5 vs
min_temp 50). Invariant worth recording: *a soft floor that heats only to the floor
cannot hold the floor across any later loss/draw.* A DP with min_temp as a hard per-step
constraint removes the class.

## Stop-cost has a horizon-edge asymmetry

`_schedule_stop_count` counts on→off transitions, so a block running to the **last** slot
costs zero stops — a mild bias toward ending the horizon running. Also the per-stage
`_schedule_objective_delta` calls inside the repairs default `compressor_initially_on=False`,
so the initial-on state is honoured only in the final top-level scoring, not in the greedy
repair decisions. Both disappear if you charge a **start cost** (0→1 inrush/wear event)
instead of a stop count.

## Actuation has limited authority over real stops

Executor hands the device a setpoint = block-end *predicted* temp + `turn_off` at block
end. But the characterisation shows the probe sits flat 45–47 °C for ~45 min during
stratified charging, so the real unit stops on whichever fires first: its own thermostat
hitting the probe setpoint, or the plan's `turn_off`. Stop-cost optimisation only
co-owns the stop decision until `off→heat_pump` / `turn_off` latency and setpoint-vs-probe
behaviour are measured. The current experiment is the right way to gather that.

## Live-safety gap

`actuation.enabled = true` **and** `min_block_duration_minutes = 0` (5-min minimum block)
**and** low `stop_cost_aud`. The only thing preventing 5-min cycles on a fixed-speed
compressor is a tuned cost. Fine as a supervised experiment, but a **hard** minimum
runtime should land before the system is left unattended.

## Smaller notes

- Executor is stateless on its last command; the 60-s periodic tick re-issues
  `set_operation_mode` + `set_temperature` every minute through a block and resets the
  off-suppression latch each time. Prefer commanding only on decision *change*.
- Block detection is power-threshold inference (`_block_bounds`). Safe today
  (`compressor_power_min_w 650 ≫ threshold 100`) but exactly what explicit block metadata
  would remove — do that *before* the DP so the DP emits the metadata format from day one.
- EMHASS integration checks out structurally (DH snapshot feeds both DH base-load and the
  MPC subtract/add-back; string-or-list attrs handled; 48→72 h repeat present). The 30-min
  bucket arithmetic in the Jinja was not line-by-line audited.

## Answers to the Review Questions

- **Replace with DP/MILP?** Yes — **DP**, not MILP: 1-D state in temperature, no solver
  dependency, free value function. Latency makes it a performance fix too.
- **State?** `(time_step, temp_or_SoC_bin, compressor_on)`, plus a heat-regime bit
  (full-reheat vs top-up) because the current model's heat rate depends on the *block
  start* temp, not the current temp — so (temp, on) alone is non-Markovian. Optional
  `main_60_satisfied_today` only if daily semantics are kept.
- **Stop vs start vs transition vs min-runtime?** Model **start cost** + an explicit
  **hard minimum runtime**. Drop stop-count (edge asymmetry). Compressor-on = initial state.
- **Hard constraints for min_temp / 60 °C?** Yes to both: min_temp as a hard per-step
  floor; 60 °C as a hard "satisfied by deadline" constraint.
- **Lower routine targets + periodic 60 °C?** Two constraint families on one DP: a
  continuous comfort floor + a sparse hard 60 °C deadline every N days.
- **Block metadata first?** Yes — de-risks the executor and fixes the DP output contract
  in advance.
- **Custom vs extend EMHASS?** Stay custom. EMHASS `thermal_battery` is single-node with
  fixed `supply_temperature` Carnot COP; measured COP collapses with state-of-charge
  (condensing temp 50→82 °C, 55→60 tail COP ~1.75). Single-node can't represent that.

## Suggested ordering

1. **Now (safety):** hard minimum-runtime floor; restore `min_temp = 45` /
   `stop_cost_aud = 0.05` when the experiment ends.
2. **Near-term (low risk):** publish explicit block metadata; switch executor off
   watt-inference.
3. **Rewrite:** the bespoke DP (start cost + hard min-runtime, hard min_temp, hard 60 °C
   deadline, soft terminal value). Fixes the 225-s latency, the min_temp violations, and
   the stop-cost edge artifacts together.

Caveat: latency seconds are from synthetic price/weather; the ~18× compressor-on gap is
structural (seed fan-out) and will hold. Confirm on the box with
`time .venv/bin/python hwc_planner.py --dry-run` while the compressor is running.
