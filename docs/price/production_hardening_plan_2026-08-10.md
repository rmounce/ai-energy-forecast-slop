# Production Forecast Hardening Plan — 2026-08-10

## Objective

Make price/load prediction and model retraining fail closed, observable, atomic, and reversible
without changing forecast models, HA entity names, EMHASS routing, tariffs, or control policy.

This plan is written for delegation to an implementation agent, followed by independent review.

## Production Baseline To Preserve

- MPC price: raw Amber; 14h × 5-min.
- Day-ahead price: `amber_apf_lgbm`; 72h × 30-min; p30/p50/p70.
- Day-ahead load: LightGBM base-load p65; 72h × 30-min; p50/p65/p75 published.
- Price trigger: `ai-energy-listener.service`.
- Load trigger: `ai-energy-predict.timer`.
- Suspended paths remain disabled: TFT price/load, PD-direct, tactical, strategic APF-free, and
  canonical AI source publishers.

Canonical routing: `docs/prod_pipeline_critical_path.md`.

## Guardrails

- Repository implementation only. Do not restart services, publish test data to HA, edit live HA
  state, or deploy systemd units.
- Do not retrain or replace the active models while implementing/tests run.
- Do not revive archived publishers or add an EMHASS source selector.
- Preserve current HA entity IDs and forecast payload schemas.
- Do not silently fall back from APF-backed price to an APF-free curve. Failed APF validation must
  leave the last known-good HA state untouched and return failure.
- Use `uv`, not raw `pip`, if dependency work becomes necessary. Prefer no new dependency.
- Update README/architecture/operational docs with any implemented behaviour change.

## Deliverables

1. Explicit prediction outcome and validation contract.
2. Correct process exit/healthcheck semantics.
3. Versioned candidate model bundles; active bundle cannot be partially overwritten.
4. Atomic promotion and one-command rollback.
5. Candidate report/manifest sufficient for a human promotion decision.
6. Unit and integration tests for failure paths.
7. Short operator runbook.

## Phase 1 — Prediction Outcome Contract

### Required design

Add a small typed result/exception boundary around production prediction. The CLI must know whether
the requested model families generated and, when requested, published a valid complete surface.

Validate before writing `predictions.json`, appending the forecast log, publishing to HA, or
reporting success:

| Contract | Price | Load |
|---|---|---|
| Required quantiles | p30, p50, p70 | p50, p65, p75 |
| Points per surface | 144 | 144 |
| Resolution | 30 min | 30 min |
| Index | UTC-aware, sorted, unique | UTC-aware, sorted, unique |
| Values | finite | finite, non-negative |
| Quantile order | p30 ≤ p50 ≤ p70 after current sorting | p50 ≤ p65 ≤ p75 |
| First target | current/next 30-min block per existing convention | same |
| Horizon | full 72h | full 72h |

Price-specific validation:

- Dynamic handoff requires non-empty Amber APF for every required quantile.
- APF timestamps must be unique, 30-min aligned after aggregation, and contiguous through the
  handoff.
- Reject APF older than a named configurable maximum age. Record the observed age.
- Validate the final APF/LGBM seam: no duplicate/missing interval and exactly 144 points.
- Remove the unreachable fallback contradiction: either call the simple fallback deliberately in
  a non-production mode or remove it. Production `--dynamic-handoff` must not silently use it.

Publication semantics:

- `--publish-hass` is successful only if every required entity POST succeeds.
- Do not publish a partial quantile family. Validate the family before the first POST.
- If a later POST fails, exit nonzero and report which entities may have changed. Do not claim
  transactional HA publication; make the limitation explicit.
- A dry run without `--publish-hass` succeeds after generation validation and local output write.
- Log one structured summary containing run ID, model family, source, model bundle ID, APF age,
  point counts, target range, publication result, and elapsed time. Do not log secrets.

### CLI acceptance

- `predict-price --dynamic-handoff --publish-hass` exits nonzero if APF is absent/invalid, any
  quantile is missing, validation fails, or any required HA write fails.
- `predict-load --publish-hass` follows the equivalent load contract.
- Failed generation does not overwrite `predictions.json` or append forecast logs.
- Failed publication may retain validated local diagnostics, but must not ping success healthchecks.

## Phase 2 — Listener And Service Semantics

Update `services/ha_listener.py` so:

- `last_run_at` means last validated successful publish, not last completed child.
- Healthcheck is pinged only after a validated successful publish.
- A failed child is retried on a bounded failure cadence without a tight loop. Use a named constant;
  default 5 minutes.
- APF events arriving during a run coalesce into one pending follow-up run.
- Timeout, nonzero exit, and shutdown paths cannot leave orphan subprocesses/tasks.
- Logs distinguish trigger, generation failure, publication failure, timeout, retry, and success.

Keep the 30-minute idle heartbeat and current WebSocket reconnect behaviour.

Add tests with fake subprocess/clock/healthcheck boundaries. Do not require HA or systemd.

For `systemd/ai-energy-predict.service`, preserve shell short-circuit behaviour so the load
healthcheck runs only after a zero exit. Add a comment documenting that dependency.

## Phase 3 — Versioned Model Bundles

### Layout

Use one independently promotable family bundle for price and one for load:

```text
models/production/
  price/
    active.json
    bundles/<bundle_id>/
      manifest.json
      price_model.pkl
      price_params.json
      price_importance.json
      price_p30_model.pkl
      ...
  load/
    active.json
    bundles/<bundle_id>/
      manifest.json
      load_model.pkl
      ...
```

Exact naming may vary, but preserve these properties:

- Training writes only to a new temporary candidate directory.
- Manifest is written last after all artifacts pass validation.
- Candidate directory is atomically renamed into `bundles/`.
- `active.json` is a small pointer updated using same-filesystem `os.replace`.
- Prediction resolves one active bundle once per run; all quantiles come from it.
- Never mix artifact generations within a run.
- Retain at least the active bundle and its predecessor. Cleanup must be explicit and must never
  delete a referenced bundle.

### Manifest minimum

- bundle ID, family, created UTC, git commit, config digest;
- training start/end and row count;
- requested quantiles and artifact filenames;
- feature list/lag configuration and LightGBM parameters;
- shift values and artifact SHA-256 hashes;
- validation/smoke result and producing command;
- parent/incumbent bundle ID when known.

Do not store credentials or the merged secret configuration.

### Commands

Provide explicit non-interactive commands, names may be adjusted consistently:

```bash
./forecast.py train-price-candidate
./forecast.py train-load-candidate
./forecast.py validate-bundle --family price --bundle <id>
./forecast.py promote-bundle --family price --bundle <id>
./forecast.py rollback-bundle --family price
```

Promotion requirements:

- complete manifest and matching hashes;
- all artifacts load successfully;
- quantile/config/feature contract matches current runtime configuration;
- canned inference smoke produces a valid 144-point family;
- candidate report exists and says `eligible_for_manual_promotion: true`;
- promotion is explicit. The weekly timer must train candidates but must not auto-promote in this
  implementation.

Migration:

- Create a documented one-shot command that imports the current root-level production artifacts as
  the initial active bundles.
- After migration, update every runtime/training/eval call site in scope to the bundle resolver.
- Do not leave permanent compatibility aliases to the root `.pkl` files.
- Migration must be idempotent and refuse conflicting partial state.

## Phase 4 — Candidate Report

This hardening track must not claim that the current training data is fully causal. Historical
training uses realised PV/weather/demand and selects STPASA differently from live inference. Record
that limitation in every report.

The delegated implementation should produce a deterministic report with:

- structural/artifact checks;
- training range and feature coverage, including STPASA coverage for price;
- model size and inference smoke timing;
- fixed recent time-split metrics where feasible:
  - price: MAE, bias, and pinball loss by 0–16.5h, 16.5–28h, 28–48h, 48–72h;
  - load: MAE, bias, pinball loss, and empirical quantile coverage by 0–24h, 24–48h, 48–72h;
- explicit statement that these metrics are screening evidence, not causal promotion proof;
- incumbent comparison when both are evaluable on the identical rows;
- machine-readable eligibility reasons.

Initial eligibility is deliberately conservative:

- structural, load, hash, and inference checks all pass;
- no evaluated primary metric regresses more than 5% against the incumbent;
- no price horizon-bucket absolute bias worsens by more than $10/MWh;
- load p65 empirical coverage remains within 0.55–0.85 on the screening slice;
- missing comparable evidence means `eligible_for_manual_promotion: false`, not an assumed pass.

Put thresholds in version-controlled configuration/constants and test their boundary behaviour.
Do not add an automatic promotion policy.

## Phase 5 — Tests

At minimum add tests for:

- empty/stale/misaligned Amber APF;
- missing quantile, 143/145 points, gap, duplicate timestamp, NaN/inf, quantile crossing;
- load negative value and load quantile ordering;
- no local-output/log mutation after generation failure;
- HA write failure produces nonzero CLI result and suppresses healthcheck success;
- listener success, nonzero child, timeout, retry cadence, event coalescing, shutdown cleanup;
- interrupted training leaves active bundle unchanged;
- incomplete/hash-mismatched bundle cannot promote;
- atomic active-pointer update;
- prediction loads all quantiles from one bundle even if active pointer changes mid-run;
- rollback selects the previous complete bundle;
- migration idempotence and partial-state refusal;
- candidate threshold boundary cases.

Run:

```bash
./.venv/bin/python -m pytest -q
```

No test may call live HA, Amber, AEMO, InfluxDB, or systemd.

## Documentation Deliverables

Update in the same implementation commit series:

- `README.md`: candidate-versus-active training commands.
- `ARCHITECTURE.md`: bundle layout, runtime resolution, exit semantics.
- `docs/prod_pipeline_critical_path.md`: only if runtime behaviour changes.
- `docs/price/event_driven_predict_price_plan.md`: retry/outcome semantics.
- New concise operator runbook under `docs/price/` covering migrate, inspect, promote, rollback,
  and failure diagnosis.

Use short concrete commands and paths. Do not copy historical experiment narrative into current
operational docs.

## Delegated-Agent Completion Checklist

- [ ] No live service or HA changes performed.
- [ ] Active production artifacts untouched except in isolated migration tests.
- [ ] Prediction failure semantics implemented and tested.
- [ ] Listener healthcheck/retry semantics implemented and tested.
- [ ] Candidate bundles, manifests, promotion, and rollback implemented.
- [ ] Weekly unit changed to candidate training only; no auto-promotion.
- [ ] Current root-artifact migration command implemented and tested.
- [ ] Documentation updated.
- [ ] Full test suite passes.
- [ ] `git status` clean after focused commits.
- [ ] Handoff lists commits, commands run, remaining risks, and any deliberately deferred work.

## Return Review Checklist

The reviewing agent should independently:

1. Diff every production call path, not only new tests.
2. Confirm a missing APF cannot yield exit zero or a healthcheck ping.
3. Confirm validation precedes local output/log mutation and first HA POST.
4. Simulate failure after each training artifact and verify active bundle is unchanged.
5. Simulate active-pointer change during prediction and verify bundle consistency.
6. Verify rollback after two promotions.
7. Inspect manifests for secrets and reproducibility gaps.
8. Check candidate metrics use identical rows and units.
9. Run the full tests and focused failure tests.
10. Check docs against code and leave the worktree clean.

## Explicitly Deferred

- Causal/as-issued training dataset rebuild.
- New spike classifier or price-risk policy.
- TFT-load revival.
- APF-free source revival or source selector.
- Transactional multi-entity HA publication; current HA state API does not provide it.
- Live deployment, service restart, migration, promotion, or rollback execution.
