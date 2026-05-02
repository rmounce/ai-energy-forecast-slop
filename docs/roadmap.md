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

**Dual-driven variant (now scaffolded in eval):** `eval/rolling_mpc_eval.py` supports
`--dual-terminal-scale`, which probes the LP's initial-SoC shadow price each step and then
re-solves using a terminal-energy value proportional to that shadow price. This gives a
controller whose inventory bias adapts to the forecast curve instead of staying fixed at one
hand-tuned salvage value.

**Dual-driven sweep (Window B):**
- `dual 0.5`: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- `dual 1.0`: hybrid **$2.140/day** vs amber **$2.411/day** (**−11.3%**)
- `dual 1.5`: hybrid **$2.256/day** vs amber **$2.364/day** (**−4.6%**)
- `dual 2.5`: hybrid **$2.236/day** vs amber **$2.360/day** (**−5.3%**)
- `dual 3.0`: hybrid **$2.245/day** vs amber **$2.363/day** (**−5.0%**)

**Interpretation:** the raw dual signal is directionally useful, but the current
`probe shadow price → scaled terminal value` controller still does **not** beat the best static
surrogate (`100 $/MWh`, **−3.3%**). On this window the dual-driven policy appears too weak at
low scales and not well-shaped enough at higher scales. Treat the static terminal-value trick as
a **useful surrogate** for missing long-horizon opportunity-cost information in Track 10A, not
yet as the intended production design.

**Architectural implication:** for the eventual production-facing design, the more promising
direction is likely to combine the current two-tier SoC-target handoff with an
**opportunity-cost-aware quantile / risk policy** rather than relying solely on LP terminal-value
biasing. In other words: Option A was diagnostically useful, but the likely destination now looks
more like **B** or **B+C**, not a pure A-only controller.

**Independent review checkpoint (2026-04-21):** an external review, based on the repo briefing
alone, concluded that the Track 10A terminal-value experiments are most likely exposing an
**end-of-horizon artifact** caused by the current eval not modeling the strategic `14h` SoC
handoff. On that reading, the static `100 $/MWh` surrogate is primarily compensating for a
missing boundary condition rather than identifying a ready-to-ship production feature.

**Updated priority:** implement the strategic `14h` SoC handoff in the rolling eval before
continuing with more surrogate tuning. After that alignment step, re-run the follow-up window and
then reassess whether the remaining gap, if any, calls for LP biasing (A), quantile/risk-policy
tilt (B), or a combination (C).

**Strategic handoff result (Track 10A, Window B):**
- pre-handoff: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- handoff `exact`: hybrid **$2.271/day** vs amber **$2.451/day** (**−7.4%**)
- handoff `floor`: hybrid **$2.271/day** vs amber **$2.450/day** (**−7.3%**)

**Interpretation:** restoring the strategic `14h` SoC handoff does close a meaningful part of
the Window B gap, which supports the view that some of the earlier terminal-value benefit was
compensating for a missing boundary condition in Track 10A. However, the gap does **not**
disappear. That means the earlier surrogate result was not *only* an eval artifact.

**Regime effect of handoff (Window B, exact mode):**
- `spike`: hybrid gap improves materially, from **−11.8%** pre-handoff to **−3.2%**
- `low`: hybrid remains weaker than amber (**−8.7%**)
- `normal`: hybrid remains weaker than amber (**−13.3%**)

**Current reading:** the strategic handoff should remain part of the rolling-eval baseline from
this point onward. With that alignment restored, the remaining work is no longer "fix missing
boundary state"; it is to understand and improve the residual `low` / `normal` weakness, and to
decide whether the next production-facing mechanism should be quantile/risk-policy tilt (B) or a
combined approach (C).

**Next planned experiment:** a production-aligned **Option B** path is now documented in
[docs/option_b_plan_2026-04-22.md](./option_b_plan_2026-04-22.md). The near-term plan is to keep
the strategic `14h` SoC handoff as the baseline contract and test whether an
opportunity-cost-aware **upper-tail quantile blend** improves the residual `low` / `normal`
weakness on handoff-enabled Track 10A before introducing any dynamic posture logic.

**Fixed-blend Option B result (2026-04-23):** the first handoff-enabled fixed-weight sweep is a
clear negative result. See [docs/option_b_sweep_results_2026-04-23.md](./option_b_sweep_results_2026-04-23.md).
Blending the hybrid path upward from `q50` toward `q90` made Window B worse at every tested
weight:
- `blend 0.25`: hybrid **$2.232/day** vs amber **$2.451/day** (**−8.9%**)
- `blend 0.50`: hybrid **$1.923/day** vs amber **$2.451/day** (**−21.5%**)
- `blend 0.75`: hybrid **$1.579/day** vs amber **$2.451/day** (**−35.6%**)
- `blend 1.00`: hybrid **$1.224/day** vs amber **$2.451/day** (**−50.0%**)

**Updated reading:** this does **not** kill the broader idea of distribution-aware bridge
signals, but it does rule out the naive version of Option B. Fixed global q50→q90 tilting is
too blunt and causes the controller to over-preserve inventory, especially on `low` and
`normal` days. The next experiments should therefore move away from fixed blend sweeps and
toward:
- dynamic / selective posture signals
- alternate bridge contracts
- simpler strategic-output baselines

**Reviewer follow-up implication (2026-04-23):** the latest follow-up response in
[docs/codex_review_response_2026-04-23.md](./codex_review_response_2026-04-23.md) sharpens
that conclusion further. The recommended next move is **not** another full-path quantile tilt.
Instead, keep the strategic `14h` SoC handoff as the baseline contract and add a
**dynamic, state-dependent bridge signal** derived from strategic upper-tail value.

**Recommended next experiment:** use the strategic `q50` and `q90` outputs to derive a bounded
"downstream upside" or "future inventory value" scalar, then apply that scalar through the
tactical **terminal contract** rather than the entire forecast path. The first production-aligned
variants to test are:
- dynamic terminal SoC floor uplift
- dynamic target band width / posture
- bounded terminal-energy-value bias that only activates when strategic upside is high

**Current priority order:**
1. Treat handoff-enabled `q50` as the Track 10A baseline.
2. Stop running fixed q50→q90 path sweeps.
3. Prototype a dynamic bridge-contract experiment before revisiting broader stochastic or
   path-tilt ideas.

**First dynamic bridge result (2026-04-24):** the first completed dynamic bridge-contract
variants did **not** improve on the handoff-enabled baseline. See
[docs/dynamic_bridge_results_2026-04-24.md](./dynamic_bridge_results_2026-04-24.md).
On Window B:
- handoff refresh: hybrid **$2.2706/day** vs amber **$2.4511/day** (**−7.4%**)
- dynamic terminal bridge `scale=1.0`: hybrid **$2.2706/day** vs amber **$2.4511/day**
- dynamic terminal bridge `scale=2.0`: hybrid **$2.2706/day** vs amber **$2.4511/day**
- dynamic upward band `scale=1.0`: hybrid **$2.2706/day** vs amber **$2.4511/day**

**Important nuance:** these were not purely dormant code paths. The dynamic terminal adder was
active on most steps, and the dynamic band sometimes widened materially, but the realized
economic outcome was unchanged. The result is therefore best read as a **formulation lesson**:
`exact` terminal targets leave terminal value little room to matter, while `band` without a
value signal gives the optimizer permission to finish higher without giving it a reason to do so.

**Comparator confirmation:** direct raw-parquet comparison now shows that these variants were not
just economically similar; they were dispatch-identical to numerical noise. Comparing the handoff
baseline to `dynterm_100`, `dynterm_200`, and `dynband_100` showed **0 changed steps** in
`charge_kw`, `discharge_kw`, `soc_kwh`, and `step_pnl`. The changed columns were terminal-contract
metadata, not executed control actions.

**Updated reading:** this narrows the next-step search, but it does not rule out dynamic bridge
contracts. The simple fixed path tilt is too blunt, and the first dynamic bridge runs did not
combine the right constraint/value ingredients. The next useful pilots should be short-window
diagnostics that verify dispatch actually changes before any full 6-week rerun:
- `band + dynamic terminal value`, giving the optimizer both permission and incentive to hold
  extra terminal inventory
- `floor + dynamic target uplift`, forcing a stricter q90-informed terminal floor
- a small raw-output comparator that checks `charge_kw`, `discharge_kw`, and `soc_kwh` deltas
  before promoting a variant to a long run

**Follow-up 2-day pilot result (2026-04-24):** the first two short pilots completed over
`2025-09-01 -> 2025-09-03` after improving the multi-worker path. See
[docs/dynamic_bridge_results_2026-04-24.md](./dynamic_bridge_results_2026-04-24.md).
Both pilots used `--workers 2 --mp-start-method auto`; on Linux this selected `fork`, emitted
worker startup diagnostics, and completed cleanly.

Pilot economics:
- `amber_apf_lgbm`: **$9.598/day**
- `model_a_hybrid` with `band + dynamic terminal value`: **$9.850/day** (**+2.6%**)
- `model_a_hybrid` with `floor + dynamic target uplift`: **$9.850/day** (**+2.6%**)

Raw comparison between the two pilot variants showed **0 changed steps** in `charge_kw`,
`discharge_kw`, `soc_kwh`, and `step_pnl`. The variants changed terminal-contract metadata but
not executed dispatch relative to each other. This is a useful behavioral result, but not yet a
promotion signal, because there was no same-window `exact` q50 handoff comparator in this pilot
batch.

**Immediate next options:**
- run a same-window handoff-enabled `exact` q50 baseline pilot, then compare raw outputs against
  `floor` and `band + terminal`
- if all three are dispatch-identical, run one stronger short-window probe such as `floor`
  target scale `2.0`, `band + terminal` terminal scale `2.0`, or q95 bridge upper quantile
- if stronger probes still fail to move useful dispatch, pause bridge-contract tuning and
  diagnose residual Window B losses against Amber before designing a richer value-curve handoff

**Same-window exact vs selective-band follow-up (2026-04-24):** the next two detached 2-day
pilots completed over the same `2025-09-01 -> 2025-09-03` window:
- `rolling_mpc_eval_pilot_exact_20260424`
- `rolling_mpc_eval_pilot_extra_band_20260424`

`extra_band` is the new selective formulation where dynamic terminal value applies only to the
terminal energy above the q50 floor inside the uplift band, rather than to all terminal
inventory.

Economics:
- `amber_apf_lgbm`: **$9.598/day**
- `model_a_hybrid exact`: **$9.850/day**
- `model_a_hybrid extra_band`: **$9.850/day**

Raw comparison between `exact` and `extra_band` again showed **0 changed steps** in
`charge_kw`, `discharge_kw`, `soc_kwh`, and `step_pnl`. What changed were bridge metadata
columns such as:
- `dynamic_terminal_adder_per_kwh`
- `extra_terminal_energy_value_per_kwh`
- `dynamic_target_uplift_kwh`
- `extra_terminal_energy_kwh`

Updated reading:
- the more selective "value only the extra band above q50" formulation did activate and did bind
  on some steps
- but it still did not alter first-step tactical dispatch on this pilot slice
- so the bridge-only search space is narrowing further, and any remaining bridge experiments
  should be treated as increasingly diagnostic

Revised next-step choice:
- either run one final stronger short-window bridge probe
- or pivot from bridge tuning to diagnosing the residual Amber gap more directly

**Rolling eval fidelity pilot (2026-04-25):** after fixing the AI combined publisher
interval mismatch, the next implementation step was a minimum production-fidelity upgrade
to Track A:
- actual `30m` load/PV expanded to `5m` net load
- separate tariffed import and feed-in price curves
- same `14h x 5m` rolling MPC and same strategic handoff semantics

This was first tested on the same short `2025-09-01 -> 2025-09-03` exact-handoff slice in
two otherwise-matched runs:
- `rolling_mpc_eval_pilot_exact_priceonly_20260424`
- `rolling_mpc_eval_pilot_exact_netload_20260424`

Result:
- under `price_only`, hybrid beat Amber by **+2.6%**
- under `netload_tariffed`, hybrid lost to Amber by **-6.3%**

This is an important architecture signal. The more production-like economics gate changed
both dispatch behavior and the observed ranking on the same window.

Raw comparator confirmation:
- `step_pnl`: `1152` changed rows
- `soc_kwh`: `892` changed rows
- `discharge_kw`: `295` changed rows
- `charge_kw`: `289` changed rows

So the fidelity upgrade is not merely re-accounting the same controller behavior; it changes
the tactical actions taken.

Behavioral read from the tariffed pilot:
- both sources were similar on `2025-09-01`
- the gap mostly came from `2025-09-02`
- Hybrid finished the second day with about `8.0 kWh` more stored energy
- Amber exported materially more energy on that day (`14.73 kWh` vs `7.33 kWh`) and earned
  about `$0.46` more pnl

Updated implication:
- the production-fidelity rolling gate now deserves priority over further bridge-only tuning
- the next diagnostics should focus on why Amber monetizes inventory better on this slice
  under tariffed site economics
- future control experiments should be judged primarily under `netload_tariffed`, with
  `price_only` kept as a secondary comparability lens rather than the main architectural gate

**Full-window tariffed follow-up (2026-04-25):** the next batch extended the tariffed exact-handoff
comparison beyond the initial 2-day pilot. See
[docs/rolling_eval_fidelity_full_windows_2026-04-25.md](./rolling_eval_fidelity_full_windows_2026-04-25.md).

Finished runs:
- `rolling_mpc_eval_pilot_exact_netload_7day_20260425`
- `rolling_mpc_eval_tracka_followup_6week_netload_exact_20260425`
- `rolling_mpc_eval_tracka_followup_6week_priceonly_exact_refresh_20260425`
- `rolling_mpc_eval_tracka_windowa_6week_netload_exact_20260425`

Results:
- `7-day netload_tariffed` (`2025-09-01 -> 2025-09-08`):
  - Amber: **$1.476/day**
  - Hybrid: **$0.952/day**
  - hybrid vs amber: **-35.5%**
- `Window B price_only exact refresh`:
  - Amber: **$2.451/day**
  - Hybrid: **$2.271/day**
  - hybrid vs amber: **-7.4%**
- `Window B netload_tariffed exact`:
  - Amber: **$0.427/day**
  - Hybrid: **$0.298/day**
  - hybrid vs amber: **-30.2%**
- `Window A netload_tariffed exact`:
  - Amber: **-$0.570/day**
  - Hybrid: **-$0.901/day**
  - hybrid vs amber: **-58.1%**

This materially changes the roadmap interpretation for the currently tested local asset set:
- under `price_only`, the hybrid looked acceptable in Window A and only moderately weak in Window B
- under `netload_tariffed`, Amber beats the hybrid on the 7-day pilot, full Window B, and full Window A

So the stronger production-fidelity gate is not just narrowing the advantage; it reverses the
old optimistic reading across both major windows.

Important caveat:
- these tariffed rolling results used the locally available **Run 014 Phase 7** TFT asset
- repo docs still treat **Run 011b + binary routing** as the incumbent
- therefore these results should be treated as **provisional / artifact-limited** until the
  strongest intended incumbent asset is recovered or reproduced and re-run under the same gate

6-week diagnostic read:
- Window B: Amber earns about **$8.0** more export revenue, with similar import cost, and the
  hybrid finishes the full window with **less** final stored energy (`0.0 kWh` vs `9.1 kWh`)
- Window A: Amber again wins mostly on export monetization, while the hybrid saves a little on
  import/degradation but not nearly enough to offset the weaker realized spread

Updated implication:
- the main bottleneck now looks more like tactical control / inventory monetization under
  realistic site economics than another round of generic 72h path tuning
- bridge-only experiments should remain paused until Amber-vs-Hybrid residual diagnostics are
  better understood under the tariffed gate
- `netload_tariffed` should be treated as the primary rolling architecture gate going forward
- architecture conclusions should remain provisional until the intended incumbent checkpoint is
  re-run under that gate
- the next diagnostic harness should be crossed tactical/strategic counterfactuals under
  `netload_tariffed`, not another bridge sweep:
  - `model_a_hybrid`
  - `hybrid_tactical_amber_strategic`
  - `amber_tactical_hybrid_strategic`
  - `amber_apf_lgbm`

**Crossed counterfactual pilot (2026-04-25):** the first 2-day pilot using the recovered
snapshot-backed Run 011b-era TFT asset is documented in
[docs/counterfactual_pilot_2026-04-25.md](./counterfactual_pilot_2026-04-25.md).

Window B `netload_tariffed` (`2025-09-01 -> 2025-09-03`):
- `amber_apf_lgbm`: **$6.311/day**
- `amber_tactical_hybrid_strategic`: **$6.095/day**
- `hybrid_tactical_amber_strategic`: **$5.750/day**
- `model_a_hybrid`: **$5.739/day**

Window A `netload_tariffed` (`2025-07-21 -> 2025-07-23`):
- `amber_apf_lgbm`: **-$1.533/day**
- `amber_tactical_hybrid_strategic`: **-$1.666/day**
- `model_a_hybrid`: **-$1.897/day**
- `hybrid_tactical_amber_strategic`: **-$1.985/day**

Interpretation:
- on the tariffed gate, swapping Amber strategic target into Hybrid recovers little
- swapping Hybrid strategic target into Amber degrades Amber only modestly
- on these pilots, the Amber-vs-Hybrid gap looks more **tactical-curve-shaped** than
  **strategic-target-shaped**

So the crossed-counterfactual pilots currently point toward near-horizon tactical monetization
quality as the larger bottleneck, with strategic handoff still relevant but probably
second-order.

**Window B 7-day crossed-counterfactual confirmation (2026-04-25):**
- `amber_apf_lgbm`: **$1.476/day**
- `amber_tactical_hybrid_strategic`: **$1.370/day**
- `hybrid_tactical_amber_strategic`: **$0.967/day**
- `model_a_hybrid`: **$0.841/day**

This longer Window B run strengthened the same interpretation as the 2-day pilots:
- swapping Amber strategic target into Hybrid recovers only a small fraction of the gap
- swapping Hybrid strategic target into Amber degrades Amber only modestly
- the larger deficit still appears to be tactical-curve / near-horizon monetization quality

**7-day same-target tactical residual read (2026-04-25):**
Holding the Hybrid strategic target fixed:
- `amber_tactical_hybrid_strategic`: **$1.370/day**
- `model_a_hybrid`: **$0.841/day**
- gap: about **+$0.53/day** for Amber tactical
- same final SoC for both: **`26.589 kWh`**

Tariffed decomposition of that tactical-only gap:
- Amber tactical has about **$2.83 lower import cost**
- Amber tactical has about **$1.03 higher export revenue**
- Hybrid tactical underperforms on **6 of 7 days**

Missed-export diagnostics on the same run show high-feed-in intervals where:
- Amber exports about **7.9–8.4 kW**
- Hybrid exports **0 kW**
- strategic targets are effectively **zero for both**
- step-0 wholesale forecast is also the **same**

So the next bottleneck is now best described as:
- tactical curve / tactical objective interaction under tariffed site economics
- especially missed export monetization and weaker import-side economics

That makes the next likely intervention class:
- tactical monetization work
- not another bridge-only strategic-target experiment

Reviewer follow-up on `2026-04-26` sharpened two tactical points:
- the current gap should be described less as generic "forecast accuracy" and more as
  **short-horizon curve shape for economically asymmetric decisions**
- `price_only` should remain in the comparison table as a decomposition/debug lens even while
  `netload_tariffed` remains the primary architecture gate, to reduce the risk of overfitting
  conclusions to one fixed retail tariff structure

The same reviewer also recommended the next tactical ablation order:
- split buy/sell tactical curves first (eval-only)
- only then consider learned tariff-aware tactical calibration if the split-curve ablation closes
  too little of the Amber gap

**Split-curve tactical ablation update (2026-04-27):**

The first split-curve batch substantially narrowed that search space.

Window B `netload_tariffed`, 2-day, same-target tactical comparison:
- baseline `amber_tactical_hybrid_strategic`: **$6.088/day**
- Hybrid sell-only `0.25`: **$4.650/day**
- Hybrid sell-only `0.50`: **$2.988/day**
- Hybrid sell-only `0.75`: **-$2.625/day**
- Hybrid buy-only `0.50`: **$2.643/day**
- Hybrid buy+sell `0.50 / 0.50`: **-$3.925/day**
- Hybrid sell-only `0.50` + urgency `500 / 1h / 50%`: **$5.961/day**

Window B `netload_tariffed`, 7-day:
- baseline `amber_tactical_hybrid_strategic`: **$1.386/day**
- Hybrid sell-only `0.25`: **$0.573/day**
- Hybrid sell-only `0.50`: **$0.235/day**
- Hybrid sell-only `0.75`: **-$1.452/day**
- Hybrid buy-only `0.50`: **$0.322/day**
- Hybrid buy+sell `0.50 / 0.50`: **-$1.542/day**

Window A sanity:
- baseline `amber_tactical_hybrid_strategic`: **-$1.673/day**
- Hybrid sell-only `0.50`: **-$2.324/day**

Interpretation:
- broad split buy/sell shaping is **too blunt**
- buy-side shaping should be deprioritized
- broad buy+sell shaping should be deprioritized
- pure sell-side shaping is still harmful versus the unshaped Hybrid tactical baseline, even if
  it is less bad than buy-only or buy+sell shaping
- the first genuinely promising follow-up is now **conditional** export-side shaping:
  - sell-only `0.50`
  - plus strict urgency during already-high feed-in conditions

So the tactical hypothesis has tightened again. The next question is no longer:
- “do separate buy/sell curves help in general?”

It is now:
- “can a narrowly selective export-side heuristic survive the same-target tariffed gate?”

That question has now been tested twice:

1. **Combined selective candidate**  
   sell-only `0.50` + urgency `500 / 1h / 50%`

   Window B `7-day`:
   - baseline `amber_tactical_hybrid_strategic`: **$1.385/day**
   - unshaped Hybrid: **$0.841/day**
   - shaped Hybrid: **$1.074/day**

   This closed only about **40–45%** of the same-target tactical gap, and trigger activations
   were highly concentrated on `2025-09-01`.

   Window A sanity was materially worse:
   - `2-day`: **-$1.909/day** vs baseline **-$1.658/day**
   - `7-day`: **-$1.542/day** vs baseline **-$1.176/day**

2. **Trigger-only falsification**  
   same urgency trigger, but outside triggered intervals the sell curve remained exactly
   unshaped

   Window B:
   - `2-day`: **$5.609/day** vs baseline **$6.085/day**
   - `7-day`: **$0.792/day** vs baseline **$1.385/day**

   Window A:
   - `2-day`: **-$1.909/day** vs baseline **-$1.658/day**
   - `7-day`: **-$1.542/day** vs baseline **-$1.176/day**

Interpretation:
- the trigger-only variant did **not** preserve the earlier Window B gain
- it also remained worse on Window A
- so the clean selective hypothesis is effectively falsified

That means the heuristic export-side shaping family has probably reached its ceiling.
The most justified next step is now:
- **tariff-aware tactical features / calibration**

The least invasive first version of that, per the old implementer’s suggestion, is:
- add effective import rate and effective export rate features to the Tier 1 tactical model
- keep the current training pipeline and architecture otherwise unchanged
- then re-run the tariffed same-target tactical comparisons before considering any more complex
  tariff-aware objective changes

**Tariff-aware Tier 1 branch start (2026-04-27):**

The first implementation pass now begins with explicit deterministic tariff-derived features in
the Tier 1 tactical dataset and inference path, rather than another heuristic shaping layer.

Current scope:
- derive effective import/export price features directly from the existing P5MIN wholesale curve
- use the current tariff contract:
  - general tariff profile
  - feed-in tariff profile
  - network loss factor
  - GST rule
- keep the Tier 1 LightGBM architecture and training objective unchanged for this first pass

First-pass feature block:
- `eff_import_price_h0`
- `eff_feed_in_price_h0`
- `eff_import_price_1h_mean`
- `eff_import_price_1h_max`
- `eff_import_price_1h_spread`
- `eff_feed_in_price_1h_mean`
- `eff_feed_in_price_1h_max`
- `eff_feed_in_price_1h_spread`

Important framing:
- this is a **current-tariff backtest**, not yet a historically tariff-versioned backtest
- the tariff contract is deterministic enough to reconstruct without a separate historical
  effective-rate log
- if tariff rules change materially later, training/eval artifacts should record the tariff
  version or be explicitly described as current-tariff experiments

This branch is meant to answer the smallest next question:
- does making tariff asymmetry explicit to Tier 1 improve the `netload_tariffed`
  same-target tactical gap before we consider a separate post-forecast calibrator or a new
  tariff-aware learning objective?

**First tariff-aware candidate (`lgbm_tactical_tariffaware_v1`)**

The first current-tariff candidate has now been trained into a separate model directory:
- `models/lgbm_tactical_tariffaware_v1/`

This branch should be described as a **current-tariff counterfactual backtest**:
- tariff-aware features are reconstructed from the current deterministic tariff contract
- they are useful for architecture choice
- but they should not be silently described as historical tariff truth if tariff rules later
  change

Offline training signal is mixed but plausible:
- baseline validation `q50` MAE: **21.08**
- tariff-aware candidate validation `q50` MAE: **21.21**
- baseline stratified-eval `q50` MAE: **109.96**
- tariff-aware candidate stratified-eval `q50` MAE: **108.43**

Interpretation:
- generic price MAE did **not** improve on the ordinary time-ordered validation slice
- but the stratified hold-out, which overweights the harder market regimes we actually care
  about tactically, improved a little
- so this branch should be judged primarily through the `netload_tariffed` rolling gate, not
  by headline MAE alone

The first clean short-window rolling comparison is documented in
[docs/tariff_aware_tier1_candidate_2026-04-27.md](./tariff_aware_tier1_candidate_2026-04-27.md).

**Compatibility checkpoint:** adding tariff-aware Tier 1 features changed the tactical
inference contract from a legacy `25`-column long matrix to a new `33`-column long matrix.
A compatibility layer now adapts the inference frame to the target tactical model so that:

- legacy `models/lgbm_tactical` still receive `25` columns
- `models/lgbm_tactical_tariffaware_v1` receive `33` columns

This restores clean side-by-side rolling evaluation under the same harness.

**First clean A/B result (`2026-04-27`):**

Window B `2-day` (`2025-09-01 -> 2025-09-03`):
- Amber tactical + Hybrid strategic: `6.085/day`
- legacy Tier 1 Hybrid: `5.717/day`
- tariff-aware Tier 1 candidate: `5.995/day`

Window A `2-day` (`2025-07-21 -> 2025-07-23`):
- Amber tactical + Hybrid strategic: `-1.658/day`
- legacy Tier 1 Hybrid: `-1.909/day`
- tariff-aware Tier 1 candidate: `-1.952/day`

Current read:
- the first tariff-aware candidate materially improves the export-heavy Window B tactical gap
- it does **not** yet improve Window A, and is slightly worse there
- so tariff-aware tactical features now look justified as a direction, but not yet as a
  general solution without longer-window confirmation

**7-day confirmation (`2026-04-27`):**

Window B `7-day` (`2025-09-01 -> 2025-09-08`):
- Amber tactical + Hybrid strategic: `1.385/day`
- legacy Tier 1 Hybrid: `0.823/day`
- tariff-aware Tier 1 candidate: `0.897/day`

Window A `7-day` (`2025-07-21 -> 2025-07-28`):
- Amber tactical + Hybrid strategic: `-1.176/day`
- legacy Tier 1 Hybrid: `-1.542/day`
- tariff-aware Tier 1 candidate: `-1.570/day`

Interpretation after the longer confirmation:
- the candidate still improves Window B, but only modestly on the longer slice
- the gain closes only about `13%` of the Amber gap on Window B `7-day`
- the improvement is real action quality rather than changed terminal SoC, because legacy and
  candidate finish Window B at the same final SoC
- the candidate still does not improve Window A
- so `tariffaware_v1` looks like a meaningful first positive signal, but not yet a broadly
  deployable tactical replacement

Decomposition read:
- Window B improvement is mostly export-side (`+0.48` export revenue vs legacy) with a very
  small import-cost improvement
- most of the gain is concentrated on `2025-09-01`, and regime-wise it is mainly a spike-regime
  improvement
- Window A is slightly worse in both low and spike regimes

Structural read:
- Window B has much richer export-opportunity conditions than Window A
- Window B candidate rows had feed-in `>= 300` on about `6.0%` of steps and feed-in `>= 500`
  on about `2.18%`
- Window A had only about `1.14%` and `0.15%` respectively
- Window B also had a more negative mean net load and a higher share of negative-net-load steps

That reinforces the interpretation that `tariffaware_v1` is currently behaving more like a
partial high-export regime detector than a generally improved tactical model.

So the current conclusion is:
- tariff-aware tactical features are still the right branch
- but the first feature-only pass remains too regime-specific
- the next tariff-aware refinement should be judged on whether it preserves the Window B gain
  while reducing the Window A regression, before escalating to a larger calibration or
  objective-change step

Given the reviewer and implementer feedback after this checkpoint, the most justified next
question is now slightly earlier than “build the calibrator”:
- does `tariffaware_v1` generalize beyond the flagship `2025-09-01` Window B export day?

Current critical consensus:
- branch alive
- candidate not a win
- best label is **partial regime detector**
- next cheap falsification should use:
  - Window B excluding `2025-09-01`
  - and/or a moderate-FIT middle window

Only if the candidate still helps on those slices should the next implementation branch become:
- a small post-forecast tactical calibrator with real inference-time information advantage

What to avoid from here:
- another broad tariff-aware feature sweep before the generalization question is answered
- any return to shaping or bridge-side tactical hacks
- a loss-function change before the smaller calibrator / regime test has done its job

**Overnight falsification outcome (`2026-04-28`):**

Window B excluding `2025-09-01` (`2025-09-02 -> 2025-09-09`):
- Amber tactical + Hybrid strategic: `-0.353/day`
- legacy Hybrid tactical: `-0.695/day`
- `tariffaware_v1`: `-0.691/day`
- candidate improvement vs legacy: `+0.0039/day`
- gap closure: about `1.1%`

Moderate-FIT middle window (`2025-08-12 -> 2025-08-19`):
- Amber tactical + Hybrid strategic: `0.225/day`
- legacy Hybrid tactical: `-0.217/day`
- `tariffaware_v1`: `-0.213/day`
- candidate improvement vs legacy: `+0.0037/day`
- gap closure: about `0.8%`

Interpretation:
- once the flagship `2025-09-01` day is removed, the apparent Window B gain essentially
  disappears
- the candidate also fails to show meaningful improvement on a middle export-opportunity regime
- export revenue does not consistently improve on these falsification slices

So `tariffaware_v1` is now best described as:
- **single-event-sensitive probe**

not:
- weak positive candidate
- generally useful tariff-aware tactical model

This changes the branch decision:
- stop iterating the `tariffaware_v1` feature-only path
- do **not** build a calibrator on top of `tariffaware_v1`
- treat the remaining problem as a formulation / target-construction issue

Most justified next branch:
- build an explicit oracle-action / action-regret dataset under the tariffed LP
- compare what Hybrid did, what Amber did, and what the oracle would have done under actual
  future import/export prices
- then decide whether the next model should predict action deltas, action ranking, or bounded
  tactical correction signals

Implementation checkpoint:
- [eval/build_tactical_action_regret_dataset.py](../eval/build_tactical_action_regret_dataset.py)
  now exists to build that oracle-action / action-regret dataset directly from rolling raw
  parquets using the logged SoC, terminal constraints, and actual future tariffed prices/net
  load
- [eval/analyze_tactical_action_regret.py](../eval/analyze_tactical_action_regret.py)
  now summarizes those oracle-action datasets by bucket and reports whether Hybrid or Amber is
  actually closer to the oracle first action

**First oracle-action read (2026-04-28):**
- the simple story “Hybrid should just act more like Amber on export-heavy rows” did **not**
  survive the first-action oracle check
- on Window B `7-day`, Hybrid was closer to the oracle slightly more often overall
  (`11.4%` vs Amber `10.0%`, with `78.6%` equal)
- on the high-FIT rows, the oracle sided with Hybrid much more often than Amber:
  - feed-in `>= 300`: Hybrid `30.6%`, Amber `2.5%`
  - feed-in `>= 500`: Hybrid `38.6%`, Amber `2.3%`
- outside the flagship regime, the result was mostly flat/tied rather than decisively pro-Amber

**Interpretation shift:** the remaining Amber advantage is now less likely to be “better
immediate action direction” and more likely to involve:
- multi-step path effects,
- forecast-information quality,
- or a better label family than raw first-step imitation

So the next modeling branch should not be “teach Hybrid to copy Amber’s first action.”
It should be a richer oracle-derived label around multi-step regret / state-transition value.

Builder upgrade:
- `build_tactical_action_regret_dataset.py` now also records full-horizon
  **forced-first-action objective regret** for both the observed Hybrid action and the Amber
  comparator action
- that makes the next label search more concrete: we can now ask not just “who matched the
  oracle first action?” but “whose first action left more tariffed horizon value on the table?”

**Corrected oracle-regret read (`v3`):**
- the corrected full-horizon regret rebuilds did **not** move the story back toward
  “copy Amber’s first action”
- on Window B `7-day` overall:
  - Hybrid regret: `0.00541`
  - Amber regret: `0.00605`
- on the strongest high-FIT Window B rows:
  - feed-in `>= 300`: Hybrid `0.0153`, Amber `0.0411`
  - feed-in `>= 500`: Hybrid `0.00566`, Amber `0.0567`
- so in the regime where Amber originally looked economically strongest, Amber’s **first
  action** actually leaves more full-horizon tariffed value on the table than Hybrid’s
- Amber is still slightly better on Window A `7-day` (`0.00340` vs Hybrid `0.00372`) and on
  Window B excluding `2025-09-01` (`0.00426` vs Hybrid `0.00479`)

**Interpretation shift (stronger):**
- the next branch should definitely **not** be “teach Hybrid to imitate Amber’s first action”
- the remaining gap is more likely to live in:
  - multi-step path effects,
  - forecast-information quality,
  - or a richer state-value / multi-step-regret label family

**Updated next-step consensus (reviewer + implementer):**
- first-action calibrator is now ruled out as the next primary abstraction
- the best next diagnostic pair is:
  1. forward-curve shape comparison on the high-FIT intervals where Amber wins economically
  2. forced-prefix regret with `N = 1, 3, 6, 12` pinned actions

Reasoning:
- if Hybrid gets step 0 roughly right but mean-reverts too quickly over the next several steps,
  the tactical curve shape/persistence is the issue
- if Amber only becomes better once several initial actions are pinned, the loss lives in the
  multi-step tactical path / inventory trajectory rather than in a local first-action fix
- only after those two diagnostics should a richer target be chosen, such as:
  - multi-step regret
  - marginal energy value
  - or state-transition value

**First forced-prefix result on `wb7`:**
- Amber becomes increasingly better as more of the prefix is pinned:
  - `N=1`: `-0.00043`
  - `N=3`: `-0.00150`
  - `N=6`: `-0.00351`
  - `N=12`: `-0.01147`
- so the path effect is real and grows with prefix length

But the surprise is where it does **not** come from:
- high-FIT rows (`FIT >= 300` and especially `>= 500`) still favor Hybrid, not Amber

The current leading decomposition is:
- Amber’s multi-step advantage is accumulating mainly in **negative-net-load, sub-`300` FIT**
  conditions
- `FIT < 300` and negative net load:
  - `N=1`: `-0.00389`
  - `N=3`: `-0.01168`
  - `N=6`: `-0.02287`
  - `N=12`: `-0.05101`
- `FIT < 300` with non-negative net load slightly favors Hybrid

So the next branch should stop thinking “big export spikes” and start thinking:
- medium-horizon inventory trajectory quality in ordinary export-capable periods
- especially the `30–60 minute` path through negative-net-load, non-extreme-FIT regimes

**Forced-prefix path attribution:**
- [eval/analyze_forced_prefix_path_attribution.py](../eval/analyze_forced_prefix_path_attribution.py)
  now joins forced-prefix regret rows back to rolling raw paths and summarizes charge,
  discharge, import, export, step PnL, and SoC movement by bucket
- in the key `FIT < 300` + negative-net-load bucket at `N=12`, Amber's lower-regret prefix is:
  - lower charge: `-0.083 kWh`
  - lower discharge: `-0.161 kWh`
  - lower import: `-0.302 kWh`
  - lower export: `-0.371 kWh`
  - higher prefix step PnL: `+0.022`
  - slightly higher prefix SoC delta: `+0.081 kWh`
- so the working hypothesis is now sharper than “ordinary export-capable periods”:
  Amber appears to win by **reducing churn / preserving inventory during low-to-moderate,
  often negative-FIT surplus-PV periods**, not by exporting harder
- next abstraction remains state-transition / marginal energy value, with multi-step regret as
  the label source and gate

**State-transition label branch started:**
- [eval/build_state_transition_label_dataset.py](../eval/build_state_transition_label_dataset.py)
  builds the first compact label dataset for this branch
- for each target row, it solves the realized-future tariffed oracle and compares oracle,
  target, and optional comparator prefixes over configurable horizons such as `N=6` and `N=12`
- emitted labels include:
  - oracle-vs-target SoC delta
  - oracle-vs-target throughput/churn
  - oracle-vs-target import/export energy
  - oracle-vs-target prefix PnL
  - optional finite-difference initial-SoC value via `--soc-finite-diff-kwh`
- quick smokes passed on Window B raw data; full-window label builds are LP-heavy and should use
  detached `tmux` plus logs/exitcode files before being left unattended
- the builder now supports numeric target-bucket filters; the current first full label build
  should use `--feed-in-max-mwh 300 --net-load-max-kw 0 --horizons 6,12`

**First full target-bucket state-label run:**
- completed `state_transition_wb7_fitlt300_negload` with `700` filtered starts and `1,400`
  horizon rows (`N=6`, `N=12`)
- [eval/analyze_state_transition_labels.py](../eval/analyze_state_transition_labels.py)
  summarizes the label dataset by horizon
- all rows are in the intended `FIT < 300` + negative-net-load bucket
- at `N=12`, oracle minus Hybrid averages:
  - SoC delta: `-0.443 kWh`
  - throughput/churn: `-0.540 kWh`
  - import: `-0.466 kWh`
  - export: `+0.005 kWh`
  - prefix PnL: `+0.004`
- at `N=12`, Amber minus Hybrid averages:
  - SoC delta: `-0.104 kWh`
  - throughput/churn: `-0.235 kWh`
  - import: `-0.303 kWh`
  - export: `-0.366 kWh`
  - prefix PnL: `+0.024`
- this corrects the previous shorthand: the signal is not mainly "preserve more inventory"
  and not "export harder"; it is **reduce uneconomic churn and grid exchange during low-FIT
  surplus-PV periods**
- likely first model target should be a bounded churn / grid-exchange discipline signal or
  state-transition value label, with marginal-SoC finite-difference labels deferred until this
  cheaper label distribution is understood

**Curtailment correction:**
- the low-FIT surplus-PV branch exposed that the first `netload_tariffed` LP had no explicit
  curtailment variable, so negative net load not used for battery charging was forced to export
- `solve_lp_dispatch()` now includes `curtail_kw` in site-flow mode
- with split load/PV inputs, curtailment is bounded by available PV:
  `0 <= curtail_kw <= pv_kw`
- with net-load-only inputs, curtailment is bounded by visible surplus:
  `0 <= curtail_kw <= max(0, -net_load_kw)`
- rolling `netload_tariffed` now uses split load/PV when available and includes first-step
  curtailment in the grid-flow balance
- regret/label builders now carry curtailment into path metrics and prefer split load/PV inputs
  when newer rolling raw outputs include them
- quick smoke:
  - negative feed-in with `2.5 kW` surplus: `export=0`, `curtail=2.5`
  - positive feed-in with `2.5 kW` surplus: `export=2.5`, `curtail=0`
  - negative import/feed-in with `3.0 kW` load and `2.0 kW` PV: `curtail=2.0`,
    `import=3.0`
- remaining limitation: callers that provide only net load can still curtail only visible surplus;
  full PV turn-off while site load is positive requires split load/PV inputs
- corrected Window B `7-day` rerun completed on `2026-05-01`:
  - `amber_apf_lgbm`: `1.611/day`
  - `amber_tactical_hybrid_strategic`: `1.619/day`
  - `hybrid_tactical_amber_strategic`: `1.078/day`
  - `model_a_hybrid`: `1.040/day`
- compared with the pre-curtailment run, all sources improved but the same-target tactical gap
  remained: about `+0.579/day` for `amber_tactical_hybrid_strategic` over `model_a_hybrid`
  versus about `+0.529/day` before
- corrected target-bucket labels (`FIT < 300`, negative net load) still favor reduced churn /
  grid exchange; at `N=12`, oracle minus Hybrid averages:
  - SoC delta: `-0.400 kWh`
  - throughput/churn: `-0.639 kWh`
  - import: `-0.563 kWh`
  - export: `-0.122 kWh`
  - curtail: `-0.009 kWh`
  - prefix PnL: `+0.023`
- implication: curtailment support improves eval fidelity, but it does not explain away the
  remaining Hybrid tactical loss; proceed with a short-horizon inventory-discipline / churn
  reduction target rather than returning to spike-export or first-action correction branches
- physical-feasibility audit on the corrected raw run passed with `0` violations for all sources
  at tolerance `1e-6`, covering simultaneous charge/discharge, simultaneous planned/realized
  import/export, grid-balance residuals, SoC transition residuals/bounds, curtailment greater
  than available PV, and import/export power bounds
- implication: no further simulator refinement is currently blocking the modeling branch; keep
  future simulation work focused on specific audit failures or explicitly modeled inverter
  constraints
- first diagnostic state-value model probe completed on the corrected target-bucket labels:
  - tool: `eval/train_state_transition_value_model.py`
  - prefix value / PnL label: validation MAE improves by about `3.8%` over a median baseline,
    `R2 ~= 0.084`, sign accuracy `~94.5%`
  - SoC-delta label: validation MAE improves by about `7.2%`, `R2 ~= 0.038`, sign accuracy
    `~85.7%`
  - direct throughput/import/export/curtail labels are not yet strong enough; most do not beat
    the baseline on MAE
- implication: the inventory-discipline branch has weak but real signal, but should not be wired
  into control from this single target-bucket slice; broaden the label set or improve target
  shaping before attempting an MPC bias
- first label-broadening batch completed on `2026-05-01` with all exit codes `0`:
  - added `FIT < 300` / non-negative-net-load labels and `FIT >= 300` labels from the corrected
    curtailment raw run
  - pooled those labels with the original `FIT < 300` / negative-net-load target bucket in the
    diagnostic LightGBM scaffold
  - the broadened path labels are informative, but the pooled model is effectively at baseline:
    prefix-value MAE improves by only `0.046%` overall on validation with `R2 ~= -0.033`, while
    SoC-delta, throughput, import, export, and curtail labels are at or worse than baseline
  - implication: the original target bucket still looks like a local signal, but a single broad
    state-value bias across coarse regimes is not supported; next work should either narrow the
    corrector to the ordinary surplus-PV regime where the gap was found or make the model/labels
    explicitly regime conditioned
- marginal-SoC finite-difference target-bucket run completed on `2026-05-01`:
  - added a `+1 kWh` initial-SoC finite-difference value label for the corrected `FIT < 300` /
    negative-net-load bucket
  - label distribution is plausible: mean `0.082 $/kWh`, median `0.084 $/kWh`, IQR about
    `0.050` to `0.115 $/kWh`
  - first diagnostic LightGBM fit is not controller-ready: validation MAE improves by about
    `19.8%`, but validation `R2 ~= -3.57`, train `R2 ~= 0.89`, and sign accuracy is unchanged
    from baseline
  - implication: the finite-difference label is useful as a diagnostic / possible future target,
    but the first learned marginal-value model overfits the narrow slice; do not wire it into
    dispatch without better target shaping, regularization, or broader regime-aware training data
- first eval-only inventory-discipline control hook is implemented:
  - `solve_lp_dispatch()` accepts a control-only `throughput_cost_adder_per_kwh`
  - `rolling_mpc_eval.py` exposes a `model_a_hybrid_inventory_bias` source alias plus
    `--inventory-discipline-*` gates for source label, feed-in price, net load, and cycle-cost
    adder
  - this is deliberately not a learned finite-difference controller; it is a small bounded
    churn-friction probe for the ordinary surplus-PV regime
  - smoke on `2025-09-01T00:00Z -> 03:00Z`, `netload_tariffed`, `50 $/MWh` adder, exact strategic
    handoff passed with full coverage and `32/36` gate activations
  - smoke result was directionally large (`model_a_hybrid_inventory_bias` beat baseline Hybrid on
    that tiny slice by about `$1.40` total) but also materially changed final SoC, so the next
    step is a small sweep on the normal Window B / Window A gates rather than treating the smoke
    as evidence of a deployable setting
- first inventory-discipline sweep completed on `2026-05-01` with all exit codes `0`:
  - windows: Window B 2-day (`2025-09-01 -> 2025-09-03`) and Window A 2-day
    (`2025-07-21 -> 2025-07-23`)
  - cycle-cost adders: `25`, `50`, `75`, `100 $/MWh`
  - Window B result is negative at every tested setting:
    - baseline Amber tactical + Hybrid strategic: `6.253/day`
    - unguarded Hybrid: `5.905/day`
    - guarded Hybrid: `4.963`, `4.181`, `2.111`, `1.977/day`
  - Window A result is only slightly positive versus unguarded Hybrid and still below Amber:
    - Amber tactical + Hybrid strategic: `-1.052/day`
    - unguarded Hybrid: `-1.330/day`
    - guarded Hybrid: best about `-1.313/day`
  - implication: a blunt regime-gated cycle-friction guard is falsified as a production path; it
    suppresses too much useful Window B surplus/export behavior. Keep the hook for diagnostics,
    but do not pursue broad churn friction as the next deployment candidate without a sharper
    opportunity-aware gate.
- tactical-model diagnostic pivot is now implemented as the next branch:
  - added `eval/analyze_tier1_dispatch_relevant_errors.py` to inspect Tier 1 forecast shape from
    rolling raw parquet outputs before training another model candidate
  - the script records the contract mismatch explicitly: Tier 1 is trained on raw wholesale RRP,
    while the production gate is tariffed site economics with separate import/feed-in prices
  - outputs include source/bucket forecast errors, Amber-minus-Hybrid pairwise diagnostics, top
    event rows, an enriched row parquet, and a model metadata audit for legacy vs tariff-aware
    Tier 1 artifacts
  - Window B 7-day smoke joined the `N=12` forced-prefix attribution and state-transition labels
    successfully under prefix `tier1_dispatch_relevance_wb7_smoke_20260501`
  - first smoke read keeps the focus on ordinary surplus-PV path quality: in `FIT < 300` with
    negative net load, Amber has lower forced-prefix regret more often than Hybrid (`36.0%` vs
    `23.1%`) while doing less charge/discharge/import/export over the pinned prefix; the tool also
    exposes a meaningful feed-in 1h shape gap in that bucket
  - fast diagnostic pass across Window A 7-day, Window B excluding `2025-09-01`, and the midfit
    slice also completed for both legacy and tariff-aware raw files
  - the result is not a clean universal target-bucket story: Window B excluding `2025-09-01` still
    shows Amber ahead in `FIT < 300` / negative-net-load step PnL, but Window A and midfit are flat
    to negative in that bucket, while Amber's edge often shifts to non-negative-net-load or
    high-FIT buckets
  - `tariffaware_v1` barely changes the dispatch-relevant forecast-shape fingerprints versus
    legacy on these slices, which reinforces the earlier read that it is not a strong production
    base
  - implication: do not train a narrow `FIT < 300` / negative-net-load tactical model yet; first
    inspect event rows and add a fuller h0-h11 forecast-vector diagnostic so the model branch is
    aimed at the actual misranking mechanism rather than a bucket label alone
  - added `eval/analyze_tier1_tactical_vector_errors.py` for that fuller h0-h11 diagnostic:
    - reconstructs Amber APF and Tier 1 q50 first-hour tactical vectors from local inputs
    - compares per-horizon tariffed feed-in/general-price errors by dispatch bucket
    - supports `--max-times` for smaller smokes; real-window runs should use `nice -n 19`
  - Window B 7-day legacy vector pass completed under prefix `tier1_vector_wb7_legacy_20260501`
  - first read: the main shape difference is horizon-dependent, not a simple level bias. In
    `FIT < 300` / negative net load, Amber's feed-in curve is lower than Hybrid's from h1-h11
    by roughly `3-7 $/MWh`, while in `FIT >= 300` Amber is much higher than Hybrid in the first
    15-20 minutes and materially lower later in the hour. This explains why a broad bucket or
    act-now classifier is too crude for the next model branch.
  - the broader vector batch completed on `2026-05-01` with exit code `0` under
    `tier1_vector_*_20260501` prefixes:
    - Window A 7-day, legacy and `tariffaware_v1`
    - Window B excluding `2025-09-01`, legacy and `tariffaware_v1`
    - moderate-FIT middle window, legacy and `tariffaware_v1`
    - Window A and Window B 6-week Track A tactical windows, legacy
  - this broader pass weakens the neat target-bucket story:
    - `FIT < 300` / negative net load is still relevant in Window B excluding `2025-09-01`,
      but it is flat-to-negative for Amber on Window A, midfit, and the two 6-week windows on
      the immediate step-PnL lens
    - Amber's clearer local edge often appears in `FIT >= 300` or `FIT < 300` /
      non-negative-net-load buckets
    - `tariffaware_v1` remains nearly fingerprint-identical to legacy, reinforcing that the
      feature-only tariff-aware branch is not a strong production base
  - the apparent tension with forced-prefix regret is useful rather than contradictory:
    vector diagnostics measure local first-hour curve/step behavior, while forced-prefix regret
    measures multi-step path value after pinning the controller
  - implication: do **not** train a narrow `FIT < 300` / negative-net-load model or revive
    `tariffaware_v1`; the next model-side candidate should use a richer multi-step path/value
    label or target transformation that can distinguish profitable surplus-PV preparation from
    wasteful churn, with vector diagnostics used as the falsification lens
  - added optional h0-h11 vector-feature ingestion to
    [eval/train_state_transition_value_model.py](../eval/train_state_transition_value_model.py)
    so the state-value probe can test whether full first-hour Tier 1 curve shape contains
    learnable multi-step path-value signal without leaking realized future prices
  - full target-bucket vector probe completed under
    `state_transition_wb7_fitlt300_negload_vector_probe_20260502`:
    - joined `86` vector features from `tier1_vector_wb7_legacy_20260501_tier1_vector_rows.parquet`
      to all `1,404` target-bucket label rows
    - validation prefix-PnL MAE improved only `0.4%` over baseline (`R2 ~= 0.078`)
    - validation SoC-delta MAE improved `5.3%` (`R2 ~= 0.080`)
    - throughput, import, export, and curtail labels were at or below baseline
  - feature importance used some h0-h11 shape features, but the dominant predictors remained
    time, SoC, strategic target, PV/net-load, and existing 4h forecast summaries
  - implication: the full first-hour vector has diagnostic value, but it is not by itself a
    strong enough supervised signal for a production dispatch-shape model. The next model branch
    should improve the label/target formulation or broaden regime-aware training data before
    attempting another controller hook.
  - added [eval/train_state_transition_direction_model.py](../eval/train_state_transition_direction_model.py)
    as a zero-inflated label diagnostic: instead of predicting exact sparse dollar/kWh deltas,
    it classifies directional path-change events such as `pnl_gain`, `throughput_down`,
    `grid_exchange_down`, and `soc_down`
  - target-bucket direction probe on corrected `FIT < 300` / negative-net-load labels:
    - with h0-h11 vector features, `grid_exchange_down` validation ROC AUC `0.788`, average
      precision lift `1.88x`
    - without vector features, `grid_exchange_down` improved to ROC AUC `0.825`, average
      precision lift `2.30x`
    - other labels were weaker but still more informative than the corresponding regression
      labels: `pnl_gain` ROC AUC `0.680`, `throughput_down` `0.630`, `soc_down` `0.682`
  - pooled-regime direction probe across `FIT < 300` negative net load, `FIT < 300`
    non-negative net load, and `FIT >= 300`:
    - most labels collapsed, but `grid_exchange_down` remained strong with ROC AUC `0.781`,
      average precision lift `2.58x`, precision `0.985`, recall `0.393`
  - implication: the most promising next abstraction is no longer a broad scalar state-value
    regressor. It is a narrow event-gated **grid-exchange reduction** signal: identify when the
    oracle wants materially less import+export exchange over the next 30-60 minutes, then test a
    bounded MPC nudge only in those high-confidence events.
  - eval-only grid-exchange gate implemented in
    [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py):
    - new source alias: `model_a_hybrid_grid_exchange_gate`
    - new `--grid-exchange-reduction-*` flags consume an external direction-model prediction
      CSV/parquet, threshold `grid_exchange_down` scores by timestamp, and apply a bounded
      throughput-cost nudge only for eligible source/time pairs
    - raw outputs record gate activation, score, threshold, horizon, and applied cost
  - short Window B smoke (`2025-09-01T01:00Z -> 03:00Z`, score `>= 0.8`, `50 $/MWh` nudge)
    passed with full coverage:
    - Amber tactical + Hybrid strategic: `6.431/day`, final SoC `25.675 kWh`
    - ungated Hybrid: `5.330/day`, final SoC `26.185 kWh`
    - grid-exchange-gated Hybrid: `16.299/day`, final SoC `19.949 kWh`
    - activations: `19 / 24` steps, mean score `0.911`
  - interpretation: the hook is mechanically effective and hits a real action channel, but the
    tiny smoke also shows the danger: it can harvest near-term PnL by suppressing charge/import
    and leaving the battery much lower. The next run must judge gap closure with final SoC,
    import/export decomposition, and Window A sanity before treating this as a candidate.

**Holistic review implication (2026-04-22):** the latest system-level review in
[docs/codex_holistic_review_draft_2026-04-22.md](./codex_holistic_review_draft_2026-04-22.md)
argues that the repo may now be closer to a local optimum where strategic forecast
iteration is compensating for an under-specified strategic-to-tactical contract. The
response note in [docs/review_response_2026-04-22.md](./review_response_2026-04-22.md)
records the current interpretation:
- keep the two-timescale framing
- make rolling MPC eval the primary architecture gate
- prioritize bridge-contract experiments and scenario-lite / upper-tail-aware posture
  signals
- explicitly benchmark simpler strategic outputs, not only richer full-path strategic
  forecasters

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

**Current sidecar gate checkpoint (2026-05-02):** the strongest near-production probe is no
longer a broad state-value regression. It is an eval-only `grid_exchange_down` event gate:
train a saved direction-model bundle, score ordinary rolling raw rows out-of-band, then let
`rolling_mpc_eval.py` consume the scored event file. The saved pooled bundle is modest but real
on validation (`ROC AUC 0.781`, `AP lift 2.58x`, high precision at the selected threshold).

A tiny Window B smoke with scored sidecar signals confirmed the full plumbing works:
`model_a_hybrid_grid_exchange_gate` activated on `21 / 24` steps and improved immediate
`netload_tariffed` PnL on `2025-09-01T01:00Z -> 03:00Z`, but it left final SoC about `6 kWh`
below Amber. Treat this as proof the channel is controllable, not proof the policy is safe.
Next gate: short Window A/B tests must include final SoC valuation and import/export/degradation/
curtailment decomposition before any longer run or production-facing interface.

**Follow-up:** the throughput-cost implementation was the wrong control abstraction for the
`grid_exchange_down` label. A 6h Window B smoke improved immediate PnL but mostly by suppressing
charge and leaving the battery much lower (`29.03 kWh` vs Amber `39.77 kWh`). Window A did not
change dispatch despite activations. A static `100 $/MWh` terminal value did not mitigate the
Window B inventory drain under exact handoff.

The eval hook now also supports direct import/export flow cost via
`--grid-exchange-reduction-flow-cost-mwh`. The first 2h Window B smoke is better aligned with the
label and less severe than the throughput proxy, but still spends inventory (`21.48 kWh` final SoC
vs Amber `25.67 kWh`). Do not launch longer sidecar-gate batches until the activation policy or
control action is constrained enough that short smokes preserve terminal inventory.

**SoC-guard result:** a first-action comparative guard now exists via
`--grid-exchange-reduction-max-next-soc-drop-kwh`. It blocks the gated solve when its next-step SoC
would be lower than the ungated solve by more than the configured tolerance. On the 2h Window B
flow-cost smoke, both `0.00 kWh` and `0.25 kWh` tolerances preserved SoC but collapsed the gated
source back to ungated Hybrid PnL. This means the current flat flow-cost hook has no demonstrated
production value once inventory spend is disallowed. Next work should change the control action or
train a model-side path-shape target, not scale up this hook.

**Window A finite-difference state-label result (2026-05-02):** the corrected Window A `7-day`
batch completed successfully under prefix
`rolling_mpc_eval_counterfactual_windowa_7day_netload_011b_curtail_20260502`, then built
`+1 kWh` finite-difference labels for `FIT < 300` negative net load, `FIT < 300` non-negative
net load, and `FIT >= 300`.

Dispatch result:
- `amber_apf_lgbm`: `-1.116/day`, final SoC `15.400 kWh`
- `model_a_hybrid`: `-1.125/day`, final SoC `12.732 kWh`
- `amber_tactical_hybrid_strategic`: `-0.994/day`, final SoC `4.222 kWh`
- `hybrid_tactical_amber_strategic`: `-1.691/day`, final SoC `25.432 kWh`

The full Hybrid is effectively level with Amber on this corrected Window A slice. The
Amber-tactical / Hybrid-strategic crossed source has the best immediate PnL, but it spends
inventory heavily and is not a clean production signal.

The pooled continuous finite-difference value model did not validate as useful:
- finite-difference marginal-SoC value MAE improved only `0.2%` vs baseline, with `R2 = -0.403`
- prefix PnL, SoC delta, throughput, import, export, and curtail regressions were all at or below
  baseline

The pooled direction model retained limited ranking signal:
- `pnl_gain`: ROC AUC `0.735`, AP lift `2.00x`
- `soc_down`: ROC AUC `0.704`, AP lift `2.64x`
- `grid_exchange_down`: ROC AUC `0.587`, AP lift `1.41x`
- `throughput_down`: ROC AUC `0.392`, AP lift `0.85x`

Implication: this weakens the case for a broad state-value or direct sidecar-control branch.
The next production-relevant diagnostic should compare WA/WB label and feature distributions and
test cross-window event ranking. If the strongest WB signal is regime-specific, keep it as a
diagnostic lens rather than a production policy.

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
