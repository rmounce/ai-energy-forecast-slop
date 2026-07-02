# Price forecasting (APF-free workstream) — index

The effort to forecast SA1 wholesale prices from open AEMO data (P5MIN, PREDISPATCH,
PD7Day, SDO, STPASA), with Amber's APF as the yardstick to beat rather than an input.
Suspended since ~2026-06; production remains `amber_apf_lgbm` (Amber APF near-horizon +
LGBM extrapolation with STPASA covariates). Strategy history lives in `docs/roadmap.md`
(2026-05-05 Strategic Pivot onward); source registry in `eval/price_source_contracts.py`.

## Living docs

| Doc | What it is |
|---|---|
| [price_forecast_sources.md](price_forecast_sources.md) | **Start here** — source contracts: which price path answers which question |
| [apf_free_price_forecast_review_2026-07-02.md](apf_free_price_forecast_review_2026-07-02.md) | Latest holistic review of the suspended work + recommended revival plan |
| [production_forecast_switch_plan.md](production_forecast_switch_plan.md) | HA source selectors, canonical AI entities, rollback procedure |
| [event_driven_predict_price_plan.md](event_driven_predict_price_plan.md) | Live production behaviour: predict-price triggered on Amber APF state changes |
| [tft_price_forecast.md](tft_price_forecast.md) | TFT design record + the 2026-05-05 Structural Critique (TFT retired; checkpoint kept) |
| [pd_direct_publish_rfc.md](pd_direct_publish_rfc.md) | PD-direct shadow publishing design (suspended path, revival candidate) |
| [training_runs.md](training_runs.md) | Training run ledger |

## Dated audits and experiment records

| Doc | What it settled |
|---|---|
| [pd_direct_debiaser_audit_2026-05-13.md](pd_direct_debiaser_audit_2026-05-13.md) | Raw PREDISPATCH beat the frozen debiaser post-alignment-fix |
| [price_forecast_bias_audit_2026-05-14.md](price_forecast_bias_audit_2026-05-14.md) | LGBM/TFT bias vs realised RRP |
| [lgbm_residual_driver_audit_2026-06-14.md](lgbm_residual_driver_audit_2026-06-14.md) | Incumbent residual drivers: interchange error, renewables, PV — not demand |
| [aemo_renewable_availability_discovery_2026-06-14.md](aemo_renewable_availability_discovery_2026-06-14.md) | STPASA REGIONSOLUTION found + backfilled for the 72h tail |
| [alignment_fix_retrain_2026-05-11.md](alignment_fix_retrain_2026-05-11.md) | Debiaser alignment-fix retrain and promotion |
| [session_summary_2026-05-15.md](session_summary_2026-05-15.md) | Consolidation of the 2026-05-15 window |
| [option_b_plan_2026-04-22.md](option_b_plan_2026-04-22.md), [option_b_sweep_results_2026-04-23.md](option_b_sweep_results_2026-04-23.md) | Run-aligned covariate construction sweep |
| [dynamic_bridge_experiment_plan_2026-04-23.md](dynamic_bridge_experiment_plan_2026-04-23.md), [dynamic_bridge_results_2026-04-24.md](dynamic_bridge_results_2026-04-24.md) | Tier1→Tier2 dynamic bridge experiment |
| [counterfactual_pilot_2026-04-25.md](counterfactual_pilot_2026-04-25.md), [rolling_eval_fidelity_pilot_2026-04-25.md](rolling_eval_fidelity_pilot_2026-04-25.md), [rolling_eval_fidelity_full_windows_2026-04-25.md](rolling_eval_fidelity_full_windows_2026-04-25.md) | Rolling-MPC eval fidelity checks |
| [run011b_recovery_2026-04-25.md](run011b_recovery_2026-04-25.md) | Run 011b checkpoint recovery |
| [shadow_forecast_artifact_manifest_2026-05-12.md](shadow_forecast_artifact_manifest_2026-05-12.md) | Shadow forecast artifact inventory |
| [tariff_aware_tier1_candidate_2026-04-27.md](tariff_aware_tier1_candidate_2026-04-27.md) | Tariff-aware Tier 1 candidate assessment |
| [track10a_handoff_analysis_2026-04-22.md](track10a_handoff_analysis_2026-04-22.md) | Track 10A handoff analysis |

[reviews/](reviews/) holds point-in-time review correspondence (independent review briefs
and responses, codex holistic reviews, the Phase 7 adversarial review, the debiaser spike-guard
review). Note: some local reviewer notes remain gitignored at `docs/review_*.md`.

## Current state (2026-07-02)

- **Production price source:** `amber_apf_lgbm` — Amber APF + LGBM extrapolation, STPASA
  covariates since 2026-06-16. Tier 1 tactical publishes but is not consumed by EMHASS.
- **Suspended APF-free paths:** `pd_direct`, `p5min_tactical`, `model_a_hybrid`,
  `lgbm_strategic` — retained for reference, revivable deliberately.
- **TFT price:** retired (structural critique); checkpoint + scalers kept on disk.
- **If reviving:** follow the first-steps sequence in the 2026-07-02 review — score the
  accumulated live PD-direct shadow logs before building anything.
