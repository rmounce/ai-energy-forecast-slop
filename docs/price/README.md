# Price forecasting — current state

## Status — 2026-08-10

APF-free price-forecast research is **paused**. No path is under active investigation.

- Production price source: `amber_apf_lgbm` (Amber APF near horizon + LGBM extrapolation).
- Active load forecasting remains in this repository and is out of scope for the price-review pause.
- `pd_direct`, `p5min_tactical`, `model_a_hybrid`, and `lgbm_strategic` are retained as
  reference/evaluation paths only.
- Do not start a retrain or source switch without a written hypothesis and an evaluation plan.
- Current implementation priority is reliability and artifact promotion safety, not a new source.

## Review entry points

| Read first | Purpose |
|---|---|
| [health_check_brief_2026-08-10.md](health_check_brief_2026-08-10.md) | Scope, production boundary, evidence, and questions for the next holistic review |
| [price_forecast_sources.md](price_forecast_sources.md) | Source contracts and active-vs-reference paths |
| [event_driven_predict_price_plan.md](event_driven_predict_price_plan.md) | Live `predict-price` behaviour |
| [production_forecast_switch_plan.md](production_forecast_switch_plan.md) | HA selectors and rollback procedure |
| [production_hardening_plan_2026-08-10.md](production_hardening_plan_2026-08-10.md) | Delegable implementation plan and return-review gate |
| [../prod_pipeline_critical_path.md](../prod_pipeline_critical_path.md) | Canonical live production routing |
| [../roadmap.md](../roadmap.md) | Historical roadmap; not a current action plan |

## Archive

Experiment records, model-design history, run ledgers, and review correspondence are preserved
under [../archive/price_forecast_2026/](../archive/price_forecast_2026/). They are evidence for
the health check, not an active backlog.
