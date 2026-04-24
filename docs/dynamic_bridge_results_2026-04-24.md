# Dynamic Bridge Results — 2026-04-24

Purpose: record the first completed Track 10A runs for the new
**dynamic bridge-contract** variants added on 2026-04-23.

Related:
- [docs/dynamic_bridge_experiment_plan_2026-04-23.md](./dynamic_bridge_experiment_plan_2026-04-23.md)
- [docs/roadmap.md](./roadmap.md)
- [docs/codex_review_response_2026-04-23.md](./codex_review_response_2026-04-23.md)

---

## 1. Runs Completed

All four runs completed successfully on the `2025-09-01 -> 2025-10-13` Window B slice:

- `rolling_mpc_eval_tracka_followup_6week_handoff_exact_refresh`
- `rolling_mpc_eval_tracka_followup_6week_dynterm_100`
- `rolling_mpc_eval_tracka_followup_6week_dynterm_200`
- `rolling_mpc_eval_tracka_followup_6week_dynband_100`

All exit codes were `0`.

---

## 2. Main Result

The three new dynamic bridge variants produced the **same economic outcome** as the
handoff-enabled `exact` baseline.

### Baseline handoff refresh

- `amber_apf_lgbm`: **$2.4511/day**
- `model_a_hybrid`: **$2.2706/day**
- hybrid vs amber: **−7.4%**

### Dynamic terminal bridge, scale 1.0

- `amber_apf_lgbm`: **$2.4511/day**
- `model_a_hybrid`: **$2.2706/day**
- hybrid vs amber: **−7.4%**

### Dynamic terminal bridge, scale 2.0

- `amber_apf_lgbm`: **$2.4511/day**
- `model_a_hybrid`: **$2.2706/day**
- hybrid vs amber: **−7.4%**

### Dynamic upward band, target scale 1.0

- `amber_apf_lgbm`: **$2.4511/day**
- `model_a_hybrid`: **$2.2706/day**
- hybrid vs amber: **−7.4%**

So, in this first pass, these dynamic bridge variants were **economically inert** on
Track 10A Window B.

---

## 3. Activation Diagnostics

Even though the final PnL matched the handoff baseline, the bridge signals were not all zero.

### Dynamic terminal bridge, scale 1.0

For `model_a_hybrid`:
- mean dynamic terminal adder: `0.0459 $/kWh`
- max dynamic terminal adder: `0.2246 $/kWh`
- positive terminal-adder steps: `11450 / 12096`

### Dynamic terminal bridge, scale 2.0

For `model_a_hybrid`:
- mean dynamic terminal adder: `0.0918 $/kWh`
- max dynamic terminal adder: `0.4492 $/kWh`
- positive terminal-adder steps: `11450 / 12096`

### Dynamic upward band, target scale 1.0

For `model_a_hybrid`:
- mean dynamic target uplift: `2.96 kWh`
- max dynamic target uplift: `40.0 kWh`
- positive target-uplift steps: `3652 / 12096`

So the new signals were present, but they did **not** change realized economic outcome in this
eval configuration.

### Raw action comparison

Using [eval/compare_rolling_mpc_raw.py](../eval/compare_rolling_mpc_raw.py), the raw Track 10A
outputs were compared directly against the handoff-enabled baseline.

Observed:
- `handoff_exact_refresh` vs `dynterm_100`: `charge_kw`, `discharge_kw`, `soc_kwh`, and
  `step_pnl` changed on **0** steps beyond numerical noise
- `handoff_exact_refresh` vs `dynterm_200`: `charge_kw`, `discharge_kw`, `soc_kwh`, and
  `step_pnl` changed on **0** steps beyond numerical noise
- `handoff_exact_refresh` vs `dynband_100`: `charge_kw`, `discharge_kw`, `soc_kwh`, and
  `step_pnl` changed on **0** steps

What did change:
- terminal contract metadata columns such as `dynamic_terminal_adder_per_kwh`,
  `dynamic_target_uplift_kwh`, and `max_terminal_soc_kwh`

So the stronger statement is:

These variants did not merely land on similar final economics; they produced the **same tactical
dispatch path** as the handoff-enabled baseline, up to floating-point noise.

---

## 4. Current Interpretation

This result is best read as a **formulation lesson**, not a broad rejection of dynamic bridge
contracts.

The `dynterm_100` and `dynterm_200` runs used `--strategic-target-mode exact`. Under an exact
terminal SoC constraint, adding a terminal energy value is mostly unable to influence the LP:
all feasible solutions end at the same terminal SoC, so the terminal value contributes roughly
the same constant to each feasible solution. That explains why the dynamic terminal adder was
active in diagnostics but did not change realized dispatch or PnL.

The `dynband_100` run gave the optimizer permission to finish above the q50 target, but it did
not add a positive reason to value extra terminal inventory. The optimizer therefore appears to
have stayed at the lower q50 bound. That makes this a weak test of "target band plus strategic
value"; it was only a test of "target band permission."

What these runs rule out:
- dynamic terminal value **combined with an exact terminal SoC target**, as implemented here
- dynamic upward band **without a value signal**, as implemented here

What they do **not** yet rule out:
- richer bridge contracts
- `band + dynamic terminal value`, where the optimizer has both permission and incentive to
  end above the q50 target
- `floor + dynamic target uplift`, where the q90-derived target gap forces a stricter terminal
  floor
- different tactical control formulations
- a more structural change in how strategic information is handed down

The practical takeaway is narrower:

The first dynamic bridge variants were **active but non-decisive**. They did not improve on the
handoff-enabled Track 10A baseline, and they did not degrade it either. The next test should
combine permission and incentive, or force a different terminal contract, before drawing a
stronger conclusion about dynamic bridge contracts.

Recommended next pilots:
- `--strategic-target-mode band --dynamic-bridge-target-scale 1.0 --dynamic-bridge-terminal-scale 1.0`
- `--strategic-target-mode floor --dynamic-bridge-target-scale 1.0`

Both should be tried on a short pilot window before any full 6-week rerun.

---

## 5. Follow-up 2-day Pilots

Two short pilots were run over `2025-09-01 -> 2025-09-03` after adding better
multiprocessing diagnostics and `--mp-start-method auto`:

- `rolling_mpc_eval_pilot_band_term_20260424`
- `rolling_mpc_eval_pilot_floor_20260424`

Both used:
- `--sources amber_apf_lgbm,model_a_hybrid`
- `--baseline-source amber_apf_lgbm`
- `--strategic-soc-handoff`
- `--workers 2`
- `--mp-start-method auto`

The multiprocessing path selected `fork` on Linux, emitted worker startup diagnostics, and
completed cleanly. This is the first positive validation that the improved multi-worker path
can run a real rolling MPC pilot without the earlier silent-startup failure mode.

### Pilot economics

Both pilots produced the same economics:

- `amber_apf_lgbm`: **$9.598/day**
- `model_a_hybrid`: **$9.850/day**
- hybrid vs amber: **+2.6%**

This 2-day slice is therefore favorable to the hybrid, but it is too short to draw
architecture-level conclusions.

### Pilot activation diagnostics

For `model_a_hybrid`:

- `band + dynamic terminal value`
  - positive target-uplift steps: `103 / 576`
  - positive terminal-adder steps: `563 / 576`
  - mean dynamic target uplift: `0.218 kWh`
  - mean dynamic terminal adder: `0.100 $/kWh`
- `floor + dynamic target uplift`
  - positive target-uplift steps: `103 / 576`
  - positive terminal-adder steps: `0 / 576`
  - mean dynamic target uplift: `0.218 kWh`

### Raw action comparison

Comparing `band + dynamic terminal value` against `floor + dynamic target uplift` with
[eval/compare_rolling_mpc_raw.py](../eval/compare_rolling_mpc_raw.py) showed:

- `charge_kw`: **0** changed steps
- `discharge_kw`: **0** changed steps
- `soc_kwh`: **0** changed steps
- `step_pnl`: **0** changed steps

Only terminal-contract metadata changed, notably:

- `dynamic_terminal_adder_per_kwh`: `562` changed steps
- `terminal_energy_value_per_kwh`: `562` changed steps
- `min_terminal_soc_kwh`: `35` changed steps

So these two pilots were dispatch-identical to each other. The `band + terminal` formulation
did add a value signal, but on this slice it still did not alter the realized control action
relative to the stricter `floor` formulation.

Important limitation:

These pilots were not yet compared against a same-window handoff-enabled `exact` q50 baseline.
They show that both variants can beat Amber on this particular 2-day slice and that they are
identical to each other, but they do not yet prove either variant improves over the existing
handoff baseline.

---

## 6. Same-Window Exact vs Extra-Band Follow-Up

Two additional short pilots were then run over the same `2025-09-01 -> 2025-09-03` window:

- `rolling_mpc_eval_pilot_exact_20260424`
- `rolling_mpc_eval_pilot_extra_band_20260424`

The purpose of this pair was:
- establish the same-window handoff-enabled `exact` q50 baseline
- test the new `extra_band` formulation, where terminal value applies only to the energy
  above the q50 floor inside the uplift band

Both runs completed successfully in detached `tmux`, with `--workers 2` and
`--mp-start-method auto`.

### Economics

Both pilots again produced the same economics:

- `amber_apf_lgbm`: **$9.598/day**
- `model_a_hybrid`: **$9.850/day**
- hybrid vs amber: **+2.6%**

So `extra_band` did not improve on the same-window `exact` handoff baseline.

### Activation diagnostics

For `model_a_hybrid`:

- `exact`
  - positive target-uplift steps: `0 / 576`
  - positive terminal-adder steps: `0 / 576`
- `extra_band`
  - positive target-uplift steps: `103 / 576`
  - positive terminal-adder steps: `563 / 576`
  - mean dynamic target uplift: `0.218 kWh`
  - mean dynamic terminal adder: `0.100 $/kWh`

### Raw action comparison

Comparing `exact` vs `extra_band` with
[eval/compare_rolling_mpc_raw.py](../eval/compare_rolling_mpc_raw.py) showed:

- `charge_kw`: **0** changed steps
- `discharge_kw`: **0** changed steps
- `soc_kwh`: **0** changed steps
- `step_pnl`: **0** changed steps

Changed metadata included:

- `dynamic_terminal_adder_per_kwh`: `562` changed steps
- `extra_terminal_energy_value_per_kwh`: `100` changed steps
- `dynamic_target_uplift_kwh`: `35` changed steps
- `extra_terminal_energy_kwh`: `35` changed steps
- `max_terminal_soc_kwh`: `35` changed steps

So the new `extra_band` formulation did activate and did bind on some steps, but it still did
not change realized tactical dispatch relative to the same-window `exact` baseline.

### Current reading

This is a stronger negative result than the earlier `band + terminal(all)` pilot comparison.
The issue is not merely that value was applied to the wrong part of terminal inventory. Even the
more selective "value only the extra band above q50" formulation left the first-step tactical
actions unchanged on this pilot window.

What this still does **not** prove:
- that all stronger bridge contracts are exhausted
- that higher scales or q95-derived bridge signals are useless
- that the remaining Amber gap is definitely not in the strategic-to-tactical contract

What it does prove:
- a more selective terminal-value scope alone is not enough to move dispatch on this pilot slice
- future bridge experiments should be treated as increasingly diagnostic rather than assumed to
  be promising production candidates

---

## 7. Options From Here

The next decision should be made around dispatch-changing behavior, not only summary PnL.

### Option 1 - Same-window exact baseline comparator

Completed on `2026-04-24`.

Observed:
- `exact` vs `extra_band`: dispatch-identical
- `exact` economics matched the earlier `floor` and `band + terminal(all)` pilots on the same
  slice

Implication:
- we now have a same-window handoff comparator, and the bridge-only search space has narrowed
  again

### Option 2 - Stronger floor / terminal scale probe

Run a small scale probe:

- `floor`, target scale `2.0`
- `band + terminal(all)`, target scale `1.0`, terminal scale `2.0`
- `band + terminal_scope=extra_band`, target scale `1.0`, terminal scale `1.0`
- `band + terminal_scope=extra_band`, target scale `1.0`, terminal scale `2.0`
- optionally q95 as the dynamic bridge upper quantile

Purpose:
- test whether the bridge signal is structurally capable of moving the LP when made stronger
- keep the window short so this is a behavioral diagnostic, not a tuned production candidate

Risk:
- stronger scales may over-preserve inventory and recreate the fixed q50->q90 path-tilt failure

### Option 3 - Move from point targets to a value curve

Replace the current scalar terminal target/value experiments with a richer piecewise terminal
value curve derived from the strategic solve.

Purpose:
- represent the marginal value of stored energy at the 14h boundary more directly
- avoid brittle behavior from a single exact target or a single static terminal adder

Note:
- `terminal_scope=extra_band` is a lightweight first step in this direction: it values only
  the energy above the q50 floor inside the uplift band, rather than all terminal inventory

Risk:
- more implementation complexity
- requires careful eval design to avoid overfitting a small number of windows

### Option 4 - Pause bridge-contract tuning and diagnose forecast/control residuals

Use the handoff-enabled q50 baseline as the current reference and inspect where it still loses
to Amber on Window B.

Purpose:
- determine whether the remaining gap is forecast quality, tactical execution, or strategic
boundary behavior
- avoid repeatedly tuning terminal contracts if the residual issue is elsewhere

Candidate diagnostics:
- compare charge/discharge price capture by source
- inspect days where Amber ends with materially different SoC
- compare strategic q50/q90 target paths against actual profitable opportunities

Recommended order:

1. Treat the bridge-only pilot results as mostly negative.
2. Decide whether one final stronger short-window probe is still worth the compute.
3. If not, pivot to residual diagnostics before designing a richer value-curve handoff.

---

## 8. Process Takeaway

The long-run launch shape itself is now also better understood:

- `--workers 1` was stable for detached long runs
- `PYTHONUNBUFFERED=1` was important so progress appeared in logs
- jumping straight to a `6-week` batch remains expensive and should still be preceded by a
  shorter pilot window whenever possible
- after the foreground pilot mistake on 2026-04-24, the operational rule is stricter:
  anything beyond a trivial smoke test should run in detached `tmux` with log and exit-code
  files so progress remains inspectable outside the assistant session
