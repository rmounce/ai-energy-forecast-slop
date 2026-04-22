# Dynamic Bridge Experiment Plan — 2026-04-23

Purpose: capture the first implementation pass for the post-review
**dynamic bridge-contract** experiments in Track 10A, and define the first long
runs worth launching.

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

## 3. Recommended First Long Runs

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

Suggested first scales:
- `--dynamic-bridge-terminal-scale 1.0`
- `--dynamic-bridge-terminal-scale 2.0`

Keep:
- `--strategic-target-mode exact`

### Run C — Dynamic upward band

Purpose:
- test whether selective headroom above the q50 target helps the tactical layer
  preserve inventory without another full-path tilt

Suggested first setting:
- `--strategic-target-mode band`
- `--dynamic-bridge-target-scale 1.0`

If the target-gap signal is mostly dormant, this run will tell us quickly.

### Run D — Dynamic floor uplift

Purpose:
- test a more conservative version than the upward band

Suggested first setting:
- `--strategic-target-mode floor`
- `--dynamic-bridge-target-scale 1.0`

---

## 4. Suggested Command Templates

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
  --workers 2 \
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
  --workers 2 \
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
  --workers 2 \
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
  --workers 2 \
  --output-prefix rolling_mpc_eval_tracka_followup_6week_dynband_100
```

---

## 5. Current Recommendation

If there is only time or patience for a small number of long runs, prioritize:

1. dynamic terminal-value bridge
2. dynamic upward band
3. dynamic floor uplift

Reason:
- the small smoke check suggests the shadow-gap signal is more likely to be
  active than the target-gap signal
- the fixed full-path blend has already been ruled out
- these variants directly test the bridge-contract hypothesis without another
  path-tilt sweep
