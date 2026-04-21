# Pipeline Roadmap

**Last updated: 2026-04-21**

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
| 9 | LightGBM strategic model (30-min/72-hour) | **Complete** — TFT wins on spikes; LightGBM wins on normal. Archived as exploration. |
| 5 (remainder) | HA tail-risk automations, CI/CD gate, model updates | Paused — deprioritised; Phase 7 active |
| **7** | **Enhanced Input TFT — parallel PREDISPATCH + PD7Day decoder** | **Active** — Run 014 failed interim eval (−35.3%); Run 015 flat-wMAPE ablation also failed harder (−65.9%) |
| **10A** | **Rolling MPC Eval — Model A / execution track** | **In progress** — `eval/rolling_mpc_eval.py` added; two 6-week Track A comparisons plus behavior diagnostics completed, with the follow-up window showing a clear hybrid loss vs amber |
| **10B** | **Rolling MPC Eval — full Phase 7 / planning track** | **Planned** — shorter-history stitched Tier1+Tier2 backtest from first PD7Day availability (`2026-02-09`) |

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

Companion net-load run (same 811 windows, load+PV from frozen actuals): oracle $4.73, amber $1.72, hybrid $2.01 (+16.9%). Lower absolute values than price-only because net-load objective replaces pure arbitrage. Results in `eval/results/holistic_eval_results_netload.csv`.

**Phase 9 — LightGBM strategic (complete, 2026-04-20):** Trained a 30-min/72-hour LightGBM
model (q5/q50/q95, PREDISPATCH covariates steps 0–55) and evaluated through holistic dispatch simulation.
- **Pass 1** (no spike routing): overall −27.5%, spike −31.7%, normal +13.4%. Spike failure mirrors TFT pre-classifier.
- **Pass 2** (consistent routing in training + inference): overall −32.4%, spike −37.5%, normal +13.6%. Routing made spike worse.
- **Conclusion:** LightGBM strategic has structural limitations for spikes (no attention/memory beyond 28h PREDISPATCH window; bimodal training distribution). TFT hybrid is the better architecture for spikes. Confirms TFT's spike superiority is structural, not just an artefact of routing. LightGBM strategic archived as exploration only. See `eval/README.md` → "LightGBM Strategic Model".

**Gate status (2026-04-19, tier1_tier2_hybrid):** ALL GATES PASS ✅. Phase 5 remainder unblocked.

Debiaser routing: replaced scalar 1000 $/MWh spike guard with upstream LightGBM spike
classifier (`train/train_spike_classifier.py`, threshold=0.65). Classifier features: recent
actual RRP lags + PREDISPATCH summary + time. Val ROC-AUC 0.722. See `docs/review_debiaser_spike_guard.md`.

**Caveat on eval statistics:** The 811 eval windows are drawn from a dense every-6h grid,
giving 66h of overlap between neighbors. Results are directionally robust but not 811
independent trials; tight thresholds (e.g. −2% normal) should not be over-interpreted.

---

## Rolling MPC Eval: Two-Track Plan

The one-shot 72h holistic eval remains useful as a coarse regression screen, but it does not
match how EMHASS actually operates. The production controller uses a 14h × 5-min MPC that
replans repeatedly while carrying SoC forward. To align evaluation with that destination, the
rolling backtest is split into two tracks.

### Track 10A — Model A / execution track

**Purpose:** evaluate the part of the architecture that most directly influences executed
dispatch decisions.

- Time step: `5 minutes`
- Horizon presented to MPC: `14h × 5-min`
- Forecast contract: Tier 1 native `5m / 60min` forecast for the first 12 steps, then a
  near-horizon strategic extension for the remaining horizon, expanded from `30m` steps into
  repeated `5m` slots as needed
- Statefulness: continuous SoC carryover across the whole backtest
- Refresh semantics: current-interval price treated as known; forecast path refreshed on the
  eval timescale when new forecast-bearing source data is available
- Historical scope: use the longest dense-history window available (PREDISPATCH/P5MIN/actuals),
  not constrained by PD7Day availability

**Why first:** this track directly tests the execution-facing component, gives far more history
 than the PD7Day-constrained full Phase 7 setup, and should be the first rolling gate to build.

**Observed results so far:**
- **Window A** (`2025-07-21 → 2025-09-01`): `model_a_hybrid` **$2.585/day** vs `amber_apf_lgbm`
  **$2.523/day** (**+2.4%**)
- **Window B** (`2025-09-01 → 2025-10-13`): `model_a_hybrid` **$2.134/day** vs `amber_apf_lgbm`
  **$2.406/day** (**−11.3%**)

**Regime interpretation:**
- Window A: hybrid won on `spike`, roughly tied on `low`, and lost on `normal`
- Window B: hybrid lost on `low`, `normal`, `spike_moderate`, and the single `spike_extreme` day

**Current conclusion:** Track 10A is informative but not yet decisive. The hybrid does **not**
show a robust, window-stable edge over Amber on the execution track. The main persistent
weakness remains `normal` days, and the earlier apparent spike advantage did not hold in the
follow-up window.

**Behavioral diagnosis from Window B (`rolling_mpc_eval_tracka_followup_6week_behavior_prices_behavior_summary_vs_baseline.csv`):**
- `low`: hybrid charged less than amber and ended days with materially less stored energy
  (`soc_delta` **7.50 kWh** vs **10.78 kWh**), suggesting weaker low-price energy accumulation;
  average charge price was slightly better than amber, so the issue looks more like under-building
  inventory than obviously buying at the wrong moments
- `normal`: hybrid charged slightly more but discharged less, started with lower SoC, and
  depleted less over the day (`soc_delta` **−2.64 kWh** vs **−5.17 kWh**), consistent with a
  weaker SoC posture and poorer monetisation of stored energy on ordinary days; realised
  discharge prices were also worse than amber
- `spike`: hybrid was **more** active than amber (more charge, more discharge, higher average
  dispatch, higher opening and closing SoC) yet still earned less; realised charge prices were
  less negative and realised discharge prices were lower than amber, pointing to timing/forecast-shape
  errors rather than simple under-activity

**Working hypothesis:** the hybrid's current execution-track weakness is not just "too little
 dispatch". It appears to combine (1) weaker SoC build on low-price days, (2) less effective
 monetisation of stored energy on normal days, and (3) mistimed charge/discharge around spike
 opportunities. The new realised buy/sell price diagnostics strengthen the spike-day mistiming
 hypothesis: the hybrid is active enough, but captures a worse spread.

**Potential remediation direction:** before changing forecast architecture again, test whether
the MPC objective can be biased by **opportunity cost of energy**, ideally using the LP dual
for the SoC constraint (shadow price). The intent is to discourage locally attractive discharge
when future value-of-energy is high, without hard-coding brittle spike heuristics. This should
be treated as an execution-layer experiment, not a replacement for Track 10A's forecast
comparison gate. `eval/rolling_mpc_eval.py` now has an experimental
`--terminal-energy-value-mwh` hook to support this ablation with a simple salvage-value proxy
before wiring in a true dual-driven policy.

**Opportunity-cost sweep (Window B, salvage-value proxy):**
- `0 $/MWh`: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- `50 $/MWh`: hybrid **$2.254/day** vs amber **$2.393/day** (**−5.8%**)
- `100 $/MWh`: hybrid **$2.306/day** vs amber **$2.385/day** (**−3.3%**)
- `150 $/MWh`: hybrid **$2.287/day** vs amber **$2.383/day** (**−4.0%**)
- `200 $/MWh`: hybrid **$2.249/day** vs amber **$2.355/day** (**−4.5%**)

**Interpretation:** the salvage-value proxy materially reduced the hybrid's execution gap,
with the best result at roughly **`100 $/MWh`**. That is enough to establish proof of concept:
execution policy is a meaningful part of the Track 10A deficit, not just forecast quality.
The next step is to replace the static proxy with a **dual-driven opportunity-cost policy**
rather than keep sweeping fixed salvage values.

**Data-quality note:** results are now based on full coverage for all sources after adding Amber
target-time normalization plus finite-gap curve repair. The first 6-week Amber run used
**241 repaired curves** with **0 skipped steps**; the follow-up 6-week run required **0**
repairs and also had **0 skipped steps**.

### Track 10B — Full Phase 7 / planning track

**Purpose:** evaluate the stitched strategic architecture in the form closest to the intended
 production system.

- Time step: `5 minutes`
- Horizon presented to MPC: `14h × 5-min`
- Forecast contract: Tier 1 `5m / 60min` + Tier 2 `30m / 72h`, with each Tier 2 30-minute
  step repeated across six 5-minute slots
- Statefulness: continuous SoC carryover
- Historical scope: starts at **`2026-02-09`**, the first date where PD7Day exists, so the
  planning-layer inputs are actually exercised

**Interpretation:** this track is the closest match to the desired production architecture, but
it has much shorter historical coverage and therefore lower statistical power.

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

**Why it now matters again:** the latest Track 10A follow-up window suggests the current
execution weakness is not simply forecast inactivity. On spike days the hybrid moved plenty
of energy but bought less cheaply and sold less expensively than amber. That makes a
shadow-price-informed execution bias a credible next experiment: use the marginal future
value of stored energy to resist premature discharge and improve spread capture, especially
when near-term forecasts understate downstream opportunity.

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
