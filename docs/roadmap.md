# Pipeline Roadmap

**Last updated: 2026-04-18**

Full architecture: `ARCHITECTURE.md`. Model design rationale: `docs/tft_price_forecast.md`.
Data sources: `docs/data_sources.md`. Load TFT: `docs/tft_load_forecast.md`.

---

## Design Principles

**Predict distributions, not averages.** Battery economics are asymmetric — a model trained
to minimise MSE smooths out spikes. The pipeline predicts q5/q10 through q90/q95/q99 at
every interval. Financial regret (simulated $/day vs oracle) — not nMAPE — is the headline
evaluation metric.

**Separate concerns cleanly.** Forecaster forecasts. Optimizer optimizes. Execution layer
acts. These must not bleed into each other.

**The spike gap in TFT is structural, not fixable by tuning.** LightGBM wins on spikes
because PREDISPATCH is its direct input feature. TFT's advantage is long-horizon baseload
accuracy and calibrated uncertainty. The two-tier architecture separates these roles rather
than trying to solve both with one model.

**Raw PREDISPATCH is biased.** AEMO's forecasts have systematic errors during demand peaks
and constraint events. The pipeline corrects this explicitly via the Phase 1a OOF debiaser.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | PREDISPATCH debiaser (LightGBM OOF) | ✅ Done — MAE 325→65 $/MWh |
| 1b | TFT price model Run 011b | ✅ Done — debiased decoder, q5/10/50/90/95/99 |
| 2 | Tactical LightGBM Run 001 + P5MIN backfill | ✅ Done — q50 MAE 21.1 vs 30.7 baseline |
| 3 | Dispatch simulator baseline | ✅ Done — LightGBM +5.9% regret reduction vs P5MIN |
| 4 | Conformal calibration (Tier 1) | ✅ Done — spike q95 0.750→0.821 |
| 5 (partial) | Production routing, combined shadow sensors | ✅ Done — live in HA |
| **6** | **Holistic dispatch simulation** | **Complete** — oracle/amber_apf_lgbm/p5min/TFT/hybrid all evaluated |
| **8** | **Test framework** | **Complete** (42 tests passing) — Layer 2 financial gate passing ✅ |
| 5 (remainder) | HA tail-risk automations, CI/CD gate, model updates | Paused — pending Phase 6+8 |
| 7 | Event-driven predict service | Deferred — after Phase 8 |

**Hard gate:** Phase 6 and Phase 8 must both pass before Phase 5 remainder resumes.
**STATUS: Both gates passing as of 2026-04-19. Phase 5 remainder is now unblocked.**
Reason: without a financial baseline and regression tests, pipeline changes cannot be
validated against the ultimate goal (profit).

**Amber APF replacement prerequisite (beyond Phase 6+8):** Amber APF bundles a real-time
confirmed-price feed *and* a longer-horizon forecast. The 30-min/72h Phase 6 eval validates
the strategic component only. A separate **5-min tactical eval** (Tier 1 LGBM vs naive
persistence at 5-min resolution) is required before Amber APF can be switched off in
production. See `eval/README.md` → "Tactical Eval (Pass A)".

**Tactical eval results (2026-04-19):**
- Pass A (accuracy): Tier 1 beats naive **+24.4%** MAE overall, all horizons h0–h11. ✅
- Pass B (dispatch): Tier 1 revenue beats naive on all strata (spike +0.7%, low +15.7%, normal +4.2%). ✅

Both tactical gates pass. Combined with Phase 6 (30-min/72h strategic), the eval dual
prerequisite for Amber APF replacement is met. See `eval/README.md` → "Tactical Eval".

**Note — Amber APF 5-min comparison not yet possible:** Historical Amber APF 5-min forecasts
were never logged; only the 30-min combined forecast exists in `price_forecast_log.csv`.
Comparing Tier 1 directly against Amber APF at 5-min resolution requires new ingest logging
(~1–2 months accumulation). Documented in `docs/ideas.md` → "Amber APF 5-min forecast
logging". Not blocking — naive persistence is the gate baseline, not Amber APF.

**Phase 6 results** (July 2025–March 2026, 811 windows, price-only LP MPC):

| Source | All $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $6.00 | $11.97 | $2.77 | $2.12 |
| **Amber APF + LGBM (baseline)** | **$2.99** | **$6.82** | **$0.89** | **$0.52** |
| Tier 1 + TFT hybrid (spike classifier, threshold=0.65) | $3.28 (+9.7%) | $7.31 (+7.2%) | $1.18 (+32.6%) | $0.52 (+0.4%) |
| TFT Tier 2 q50 (standalone, archived) | $3.18 (+6.6%) | $7.22 (+5.8%) | $1.10 (+23.6%) | $0.41 (−21.1%) |
| P5MIN naive | $0.09 | $0.17 | −$0.01 | $0.13 |

*Results use frozen actuals parquet (`eval/results/holistic_eval_actuals.parquet`, 2026-04-19). See `eval/export_holistic_actuals.py`.*

**Gate status (2026-04-19, tier1_tier2_hybrid):** ALL GATES PASS ✅. Phase 5 remainder unblocked.

Debiaser routing: replaced scalar 1000 $/MWh spike guard with upstream LightGBM spike
classifier (`train/train_spike_classifier.py`, threshold=0.65). Classifier features: recent
actual RRP lags + PREDISPATCH summary + time. Val ROC-AUC 0.722. See `docs/review_debiaser_spike_guard.md`.

**Caveat on eval statistics:** The 811 eval windows are drawn from a dense every-6h grid,
giving 66h of overlap between neighbors. Results are directionally robust but not 811
independent trials; tight thresholds (e.g. −2% normal) should not be over-interpreted.

---

## CI/CD: Two-Tier Promotion Gate (Phase 5 sub-task 6)

Applies to weekly model retrains. Not yet implemented — deferred to after Phase 6+8 gate.

**Gate 1 — Rolling window (14 days):** Simulated financial regret on the most recent 14
days must match or beat the incumbent model.

**Gate 2 — Golden set (partitioned, never contaminated):**
- **Set A** (training allowed, decay floor ≥50%): June 2022 Energy Crisis
- **Set B** (eval-only, never trained): February 2024 SA Storm Islanding event

Set A and Set B must remain mutually exclusive at every retrain. The gate tests
generalisation to unseen regime-change events, not memorisation of Set A.

---

## Training: Event-Stratified Importance Weighting

**Problem with pure exponential decay:** Downweights 2022 energy crisis events (market
suspensions, extreme spikes, deep negatives) precisely when their training signal is most
valuable. Spike physics do not change over time; only baseload bid-stack behaviour does.

**Solution (implemented):**
- Baseload regime (−$50 to $150): exponential decay, half-life ~90 days
- Extreme events (price < −$50 or > $150): decay floor at 50% of most-recent-sample weight

This prevents catastrophic forgetting of 2022 crisis data while keeping recent market
dynamics dominant. Implemented in `train/train_tft_price.py` and `train/train_lgbm_tactical.py`.

---

## Execution Layer

**EMHASS (baseload arbitrage):** Receives Tier 2 q50 price vector via
`sensor.ai_price_forecast`. LP maximises daily arbitrage net of cycle degradation
(~$0.05/kWh marginal penalty).

**HA tail-risk overrides (Phase 5 sub-task 4, not yet implemented):**
1. Spike defence: when `q95 − q50` exceeds opportunity cost of holding charge →
   set `backup_reserve = 100%` to prevent premature discharge before a spike.
2. Oversupply capture: when `q50 − q5` justifies forced grid charging ahead of EMHASS
   schedule → override charge threshold.

**EMHASS dual variable (Phase 5 sub-task 5):** The mathematically correct denominator
for opportunity cost is the shadow price of the SOC constraint (`constraint.pi` in PuLP).
EMHASS does not expose this via API. Preferred path: fork/upstream to extract it.
Fallback: two LP solves with finite difference (2× overhead).

---

## Known Open Issues

**Tier 2 lower-tail calibration:** q05/q10 are structurally biased — log-scaling compresses
negative values. Do not use Tier 2 q05/q10 for dispatch thresholds until conformal
calibration is applied. Upper tail (q90/q95/q99) is well-calibrated.

**NEM intervention pricing:** `is_intervention` boolean must mask `aemo_divergence` to 0
in Tier 1 when AEMO issues market directions. Also: provisional vs final settlement
corrections (up to 4 days post-interval) not yet tracked in ingest pipeline.

**Reserve margin for Tier 2 decoder:** SevenDayOutlook demand is biased low during
heatwaves — exactly when reserve margin tightens. Adding rolling actual-vs-forecast
demand divergence as an encoder feature would help. Blocked on SDO capacity column
being absent from current parquet export.

**Debiaser endogeneity:** PREDISPATCH debiaser trained on endogenous targets (actual
prices are partly caused by generators responding to PREDISPATCH). Not fixable; document,
monitor residuals, retrain aggressively during structural transition periods.
