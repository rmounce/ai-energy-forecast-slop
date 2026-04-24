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

---

## 4. Current Interpretation

This is a negative result for these first dynamic bridge formulations.

What it rules out:
- a simple shadow-gap-driven terminal value, as implemented here
- a simple q50/q90 target-gap-driven upward terminal band, as implemented here

What it does **not** yet rule out:
- richer bridge contracts
- different tactical control formulations
- a more structural change in how strategic information is handed down

The practical takeaway is narrower:

The first dynamic bridge variants were **active but non-decisive**. They did not improve on the
handoff-enabled Track 10A baseline, and they did not degrade it either.

---

## 5. Process Takeaway

The long-run launch shape itself is now also better understood:

- `--workers 1` was stable for detached long runs
- `PYTHONUNBUFFERED=1` was important so progress appeared in logs
- jumping straight to a `6-week` batch remains expensive and should still be preceded by a
  shorter pilot window whenever possible
