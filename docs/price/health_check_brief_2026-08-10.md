# Price forecast health check — review brief

## Scope

Review the price-forecast approach as deployed and as retained for comparison. Do not change
production during the review.

- **Production:** `amber_apf_lgbm`; live price publication is event-driven via
  `ai-energy-listener.service`.
- **Unchanged:** active load forecasting, its services, and Home Assistant consumers remain in
  scope as system context but are not candidates for this price-path cleanup.
- **Reference paths:** `pd_direct`, `p5min_tactical`, `model_a_hybrid`, `lgbm_strategic`.
- **Paused:** APF-free experimentation. A restart needs a written hypothesis, a fixed evaluation
  matrix, and a reversible deployment plan.

## Review questions

1. Is `amber_apf_lgbm` still the defensible production baseline on current data?
2. Are the retained comparison paths reproducible from their documented artifacts and inputs?
3. Does the rolling-MPC evaluation measure the dispatch decision it claims to measure, including
   tariff, battery, and terminal-SoC assumptions?
4. Which retained paths are worth refreshing, and which can be retired after evidence capture?

## Evidence map

| Evidence | Location |
|---|---|
| Source contracts | [price_forecast_sources.md](price_forecast_sources.md) |
| Evaluation implementation and commands | [../../eval/README.md](../../eval/README.md) |
| Canonical retained outputs | `eval/results/` |
| Historical run ledger and artifacts | [../archive/price_forecast_2026/training_runs.md](../archive/price_forecast_2026/training_runs.md), [artifact manifest](../archive/price_forecast_2026/shadow_forecast_artifact_manifest_2026-05-12.md) |
| Latest pre-pause holistic review | [../archive/price_forecast_2026/apf_free_price_forecast_review_2026-07-02.md](../archive/price_forecast_2026/apf_free_price_forecast_review_2026-07-02.md) |
| Historical review correspondence | [../archive/price_forecast_2026/reviews/](../archive/price_forecast_2026/reviews/) |

## Guardrails

- Compare sources on the same data window, tariff configuration, battery assumptions, and
  starting/terminal SoC policy.
- Treat stale or unreproducible artifacts as evidence gaps, not performance evidence.
- Preserve a result manifest for any new evaluation: command, source/model artifact, window,
  inputs, and output paths.
- Keep production source selectors unchanged until a candidate clears the agreed gate.

## Outcome

The review retained `amber_apf_lgbm` and prioritised production reliability over source research.
Implementation handoff:
[production_hardening_plan_2026-08-10.md](production_hardening_plan_2026-08-10.md).
