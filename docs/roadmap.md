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
| **8** | **Test framework** | **Layer 1 complete** (29 tests) — Layer 2 gate enabled, normal stratum failing |
| 5 (remainder) | HA tail-risk automations, CI/CD gate, model updates | Paused — pending Phase 6+8 |
| 7 | Event-driven predict service | Deferred — after Phase 8 |

**Hard gate:** Phase 6 and Phase 8 must both pass before Phase 5 remainder resumes.
Reason: without a financial baseline and regression tests, pipeline changes cannot be
validated against the ultimate goal (profit).

**Amber APF replacement prerequisite (beyond Phase 6+8):** Amber APF bundles a real-time
confirmed-price feed *and* a longer-horizon forecast. The 30-min/72h Phase 6 eval validates
the strategic component only. A separate **5-min tactical eval** (Tier 1 LGBM vs naive
persistence at 5-min resolution) is required before Amber APF can be switched off in
production. See `eval/README.md` → "Tactical Eval (Pass A)".

**Tactical eval Pass A result (2026-04-19):** Tier 1 LGBM beats p5min_naive by **+24.4%**
overall (19–30% by stratum) at all horizons h0–h11. Gate ✅ passes. Pass B (dispatch value
simulation via `eval_tier1_dispatch.py`) still needed for complete tactical gate.

**Phase 6 results** (July 2025–March 2026, 811 windows, price-only LP MPC):

| Source | All $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $6.00 | $11.97 | $2.77 | $2.12 |
| **Amber APF + LGBM (baseline)** | **$2.99** | **$6.82** | **$0.89** | **$0.52** |
| Tier 1 + TFT hybrid (debiased + spike guard) | $3.22 (+7.8%) | $7.34 (+7.6%) | $1.08 (+21.1%) | $0.41 (−21.7%) |
| TFT Tier 2 q50 (standalone, archived) | $3.18 (+6.6%) | $7.22 (+5.8%) | $1.10 (+23.6%) | $0.41 (−21.1%) |
| P5MIN naive | $0.09 | $0.17 | −$0.01 | $0.13 |

**Gate status (2026-04-19, tier1_tier2_hybrid):** Overall/spike/low pass. Normal stratum
fails (−21.7% vs −2% threshold). Phase 5 remainder blocked until normal stratum resolves.

Normal root cause: debiaser spike guard (1000 $/MWh) blocks correction of PREDISPATCH
overestimates > 1000 in flat-price windows. Without spike guard: normal +4.8% ✅ but
spike −16.4% ❌. The spike guard threshold is a heuristic — needs principled derivation
from training residuals or retraining with spike-aware loss. See TODO in
`eval/retro_tft_inference.py`.

**Caveat on eval statistics:** The 811 eval windows are drawn from a dense every-6h grid,
giving 66h of overlap between neighbors. Results are directionally robust but not 811
independent trials; tight thresholds (e.g. −2% normal) should not be over-interpreted.

**Next steps to fix normal gate (priority order from independent review, 2026-04-18):**

1. **Fix debiaser inference path (prerequisite before any model work)**
   Training uses OOF-debiased `pd_rrp` at decoder steps 0–55; live inference
   (`forecast.py:_get_influx_pd_prices`) and retro eval (`eval/retro_tft_inference.py`)
   both feed raw PREDISPATCH. This violates the training contract and may be the primary
   cause of flat-price overestimation. Fix both paths, rerun Phase 6 eval before
   drawing conclusions about the normal gate.
   - `eval/retro_tft_inference.py`: substitute `debiased_pd_rrp_oof.parquet` at steps 0–55
   - `forecast.py`: apply `models/pd_debiaser/lgbm_final.pkl` to live PREDISPATCH before TFT

2. **After re-evaluating with corrected inference:**
   - If normal gate substantially improves → relax gate tolerance (see below) and proceed
   - If normal gate still fails badly → build `lgbm_strategic` (30-min/72h LGBM on debiased
     PREDISPATCH) as q50 dispatch signal; use TFT solely for tail-risk quantiles (q05/q95)

3. **Gate tolerance relaxation (independent review recommendation)**
   The normal stratum penalty is −$0.14/day while spike/low gains are +$0.55/day combined.
   After the debiaser fix, consider widening normal tolerance from −2% to −10% with an
   explicit financial-asymmetry justification comment in the gate test.

4. **Deprioritized — q50 conformal calibration**: existing conformal infrastructure
   (`train/calibrate_conformal.py`) targets Tier 1 tail coverage, not TFT median bias.
   Statistically murky for correcting a log-skewed q50 point estimate. Defer.

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
