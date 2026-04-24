# Dynamic Bridge Experiment Plan — 2026-04-23

Purpose: capture the first implementation pass for the post-review
**dynamic bridge-contract** experiments in Track 10A, and record the initial
long-run evaluation shape that was tried.

Related:
- [docs/roadmap.md](./roadmap.md)
- [docs/codex_review_response_2026-04-23.md](./codex_review_response_2026-04-23.md)
- [docs/option_b_sweep_results_2026-04-23.md](./option_b_sweep_results_2026-04-23.md)

---

## 1. What Is Now Implemented

`eval/rolling_mpc_eval.py` now supports a dynamic bridge signal derived from the
strategic hybrid model's `q50` and upper quantile (`q90` or `q95`).

New CLI controls:
- `--strategic-target-mode {exact,floor,band}`
- `--dynamic-bridge-upper-quantile {0.90,0.95}`
- `--dynamic-bridge-target-scale <float>`
- `--dynamic-bridge-terminal-scale <float>`
- `--dynamic-bridge-terminal-scope {all,extra_band}`

The strategic bridge signal is computed from:
- `target_q50`: 14h terminal SoC target from the strategic q50 curve
- `target_qhi`: 14h terminal SoC target from the strategic upper-quantile curve
- `shadow_q50`: initial-SoC shadow price from the strategic q50 solve
- `shadow_qhi`: initial-SoC shadow price from the strategic upper-quantile solve

Derived quantities exposed in the raw output:
- `strategic_target_gap_kwh = max(0, target_qhi - target_q50)`
- `strategic_shadow_gap_per_kwh = max(0, shadow_qhi - shadow_q50)`

These are then used in two production-aligned ways:

1. **Dynamic target contract**
- `floor` mode: minimum terminal SoC becomes `target_q50 + scale * target_gap`
- `band` mode: minimum terminal SoC remains `target_q50`, and maximum terminal
  SoC becomes `target_q50 + scale * target_gap`

2. **Dynamic terminal-value bias**
- added tactical terminal-energy value becomes
  `scale * strategic_shadow_gap_per_kwh`

3. **Scoped terminal-value bias**
- `all`: terminal value applies to all end-of-horizon stored energy
- `extra_band`: terminal value applies only to the energy above the q50 floor,
  bounded by the q50->qhi uplift band

The existing strategic q50 SoC handoff remains the baseline contract.

---

## 2. First Read From Smoke Testing

A short smoke run completed successfully with the new `band` mode.

A second short smoke run also completed successfully with the new
**dynamic terminal-value** path enabled.

Observed on that small sample:
- `target_q90` was **not** above `target_q50`, so the target-gap-derived uplift
  stayed at zero
- `shadow_q90` was above `shadow_q50`, so the shadow-gap signal was non-zero
- the dynamic terminal adder was active on every step in the smoke slice
  (`mean ~= 0.033 $/kWh`, `max ~= 0.033 $/kWh`)

Interpretation:
- the **target-band / target-floor** variant is now implemented, but it may only
  activate on a subset of windows
- the **dynamic terminal-value** variant looks more likely to produce an active
  signal on the first long sweep

This is not yet an evaluation result, just an implementation sanity check.

---

## 3. Initial Long-Run Batch

Use the same Window B follow-up slice as the recent handoff and fixed-blend work.

Common baseline settings:
- `--sources amber_apf_lgbm,model_a_hybrid`
- `--strategic-soc-handoff`
- `--tier1-quantile-blend 0`
- `--tier2-quantile-blend 0`

### Run A — Handoff baseline refresh

Purpose:
- refresh the current comparison point using the implemented codepath

Settings:
- `--strategic-target-mode exact`
- no dynamic bridge scales

### Run B — Dynamic terminal-value bridge

Purpose:
- test the reviewer's preferred "state-dependent bridge signal through the
  terminal contract" with the least structural change

Executed first scales:
- `--dynamic-bridge-terminal-scale 1.0`
- `--dynamic-bridge-terminal-scale 2.0`

Keep:
- `--strategic-target-mode exact`

### Run C — Dynamic upward band

Purpose:
- test whether selective headroom above the q50 target helps the tactical layer
  preserve inventory without another full-path tilt

Executed first setting:
- `--strategic-target-mode band`
- `--dynamic-bridge-target-scale 1.0`

If the target-gap signal is mostly dormant, this run will tell us quickly.

### Run D — Dynamic floor uplift

Purpose:
- test a more conservative version than the upward band

Planned first setting:
- `--strategic-target-mode floor`
- `--dynamic-bridge-target-scale 1.0`

---

## 4. Command Templates Used

These should be run via the agreed `tmux` + logfile flow outside the sandbox.

### Baseline refresh

```bash
nice -n19 python eval/rolling_mpc_eval.py \
  --start 2025-09-01T00:00:00Z \
  --end 2025-10-13T00:00:00Z \
  --sources amber_apf_lgbm,model_a_hybrid \
  --tft-checkpoint models/tft_price/checkpoint_run014_phase7_best.pt \
  --tft-scalers models/tft_price/scalers_run014_phase7.pkl \
  --strategic-soc-handoff \
  --strategic-target-mode exact \
  --workers 1 \
  --output-prefix rolling_mpc_eval_tracka_followup_6week_handoff_exact_refresh
```

### Dynamic terminal-value bridge

```bash
nice -n19 python eval/rolling_mpc_eval.py \
  --start 2025-09-01T00:00:00Z \
  --end 2025-10-13T00:00:00Z \
  --sources amber_apf_lgbm,model_a_hybrid \
  --tft-checkpoint models/tft_price/checkpoint_run014_phase7_best.pt \
  --tft-scalers models/tft_price/scalers_run014_phase7.pkl \
  --strategic-soc-handoff \
  --strategic-target-mode exact \
  --dynamic-bridge-upper-quantile 0.90 \
  --dynamic-bridge-terminal-scale 1.0 \
  --workers 1 \
  --output-prefix rolling_mpc_eval_tracka_followup_6week_dynterm_100
```

```bash
nice -n19 python eval/rolling_mpc_eval.py \
  --start 2025-09-01T00:00:00Z \
  --end 2025-10-13T00:00:00Z \
  --sources amber_apf_lgbm,model_a_hybrid \
  --tft-checkpoint models/tft_price/checkpoint_run014_phase7_best.pt \
  --tft-scalers models/tft_price/scalers_run014_phase7.pkl \
  --strategic-soc-handoff \
  --strategic-target-mode exact \
  --dynamic-bridge-upper-quantile 0.90 \
  --dynamic-bridge-terminal-scale 2.0 \
  --workers 1 \
  --output-prefix rolling_mpc_eval_tracka_followup_6week_dynterm_200
```

### Dynamic upward band

```bash
nice -n19 python eval/rolling_mpc_eval.py \
  --start 2025-09-01T00:00:00Z \
  --end 2025-10-13T00:00:00Z \
  --sources amber_apf_lgbm,model_a_hybrid \
  --tft-checkpoint models/tft_price/checkpoint_run014_phase7_best.pt \
  --tft-scalers models/tft_price/scalers_run014_phase7.pkl \
  --strategic-soc-handoff \
  --strategic-target-mode band \
  --dynamic-bridge-upper-quantile 0.90 \
  --dynamic-bridge-target-scale 1.0 \
  --workers 1 \
  --output-prefix rolling_mpc_eval_tracka_followup_6week_dynband_100
```

---

## 5. Outcome Of The First Batch

The completed results are summarized in
[docs/dynamic_bridge_results_2026-04-24.md](./dynamic_bridge_results_2026-04-24.md).

Headline outcome:
- the handoff-enabled baseline remained at **$2.2706/day** for `model_a_hybrid`
- the dynamic terminal-value runs (`scale=1.0`, `scale=2.0`) landed on the same
  economic result
- the dynamic upward-band run (`scale=1.0`) also landed on the same economic result

Current interpretation:
- the simple first dynamic bridge variants were **active but non-decisive**
- they did not improve on the handoff-enabled Track 10A baseline
- they also did not obviously break anything
- further work likely needs either a different bridge contract or a more
  structural tactical-control change

Follow-up implementation now added:
- `extra_band` terminal scope gives `band` mode a more selective value signal:
  the LP can value only the terminal energy above the q50 floor instead of
  all terminal inventory
- this is closer to the intended "permission + incentive" contract than the
  earlier `band + terminal(all)` formulation

---

## 6. Operational Note For Future Runs

Avoid jumping straight to another full `6-week` rolling MPC batch unless the run shape has
already been proven on a shorter window.

Recommended workflow:
- first run a very short smoke check to confirm the code path works at all
- then run an intermediate pilot window, e.g. `1-3 days`
- only then launch the full `2025-09-01 -> 2025-10-13` batch if:
  - progress logging is advancing
  - CPU usage looks healthy
  - exit-code capture is verified
  - the run configuration is worth the long wait
  - the detached launch shape has already been proven on that exact worker mode

Rationale:
- these jobs are expensive in wall-clock time
- launch/configuration mistakes can waste many hours before they are noticed
- shorter pilot windows are usually enough to validate process management and catch obvious
  pathologies before committing to the full window

Observed process lesson from this batch:
- `--workers 1` was the stable choice for these detached long runs
- `PYTHONUNBUFFERED=1` was important so progress was visible in logs

Updated process lesson after the 2026-04-24 pilots:
- `--mp-start-method auto` now prefers `fork` on Linux and completed a real `2-day` pilot with
  `--workers 2`
- multi-worker runs should still be validated on a short pilot before any long batch
- anything beyond a trivial foreground smoke test should run in detached `tmux`, with:
  - stdout/stderr logged under `eval/results/`
  - an explicit `.exitcode` file
  - a self-closing session or clear session naming
- direct assistant-attached runs are only appropriate for tiny checks where losing the frontend
  session would not hide useful progress from the user
