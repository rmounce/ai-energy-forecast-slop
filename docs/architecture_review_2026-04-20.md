# Architecture Review: AI Energy Forecast Pipeline
**Date:** 2026-04-20  
**Purpose:** Independent expert review request — honest assessment of whether we are solving the right problems.

---

## System Context

Single-household energy management system in Adelaide, SA1 (AEMO NEM). Hardware: Sigenergy 10 kW inverter + 40 kWh battery, rooftop PV, Amber Electric retail (spot price pass-through). The goal is to maximise battery arbitrage: charge when prices are low, discharge when prices are high.

**The decision loop in production:**

1. Every 30 minutes: `forecast.py` runs TFT + LightGBM price/load models → writes `predictions.json`
2. Simultaneously: EMHASS day-ahead optimiser (72h × 30-min) builds an initial battery schedule
3. Every 5 minutes: EMHASS MPC re-plans with latest 14h × 5-min view, executes one dispatch step
4. HA automation reads EMHASS MPC output → sets Sigenergy inverter mode (charge/discharge/standby)

**Key insight:** the 14h MPC is the primary decision maker. The 72h day-ahead plan is an initial schedule that the 5-min MPC replans continuously. In practice, dispatch decisions at hours 30–72 of the 72h forecast are almost never executed as planned — they are revised many times before they matter.

---

## Current Architecture Summary

### Three-tier price forecast

| Tier | Model | Horizon | Role |
|---|---|---|---|
| Tier 1 | LightGBM tactical | 0–60 min (12 × 5-min) | P5MIN correction for real-time dispatch |
| Tier 2 | TFT price (Run 011b) | 0–72h (144 × 30-min) | PREDISPATCH-debiased strategic forecast |
| — | PREDISPATCH debiaser | 0–28h | OOF LightGBM bias correction for TFT decoder inputs |

### Hybrid dispatch in production (proposed)

- Steps 0–1 (0–60 min): Tier 1 LightGBM q50
- Steps 2–143 (30 min–72h): TFT q50
- Spike routing: if `prob_spike > 0.65` → raw PREDISPATCH bypasses debiaser for TFT decoder input

### Eval gate (Phase 6 holistic dispatch simulation)

811 fixed 72h windows (300 spike, 300 low, 211 normal), sampled Jul 2025–Mar 2026. Metric: simulated battery arbitrage revenue vs `amber_apf_lgbm` baseline (Amber APF for first ~28h, LGBM to 72h).

**Best result to date (Run 011b + binary spike routing):**

| Stratum | vs amber_apf_lgbm |
|---|---|
| All | +9.7% |
| Spike | +7.2% |
| Low | +32.6% |
| Normal | +0.4% |

---

## Tough Questions

### 1. Is the eval measuring what EMHASS actually does?

**The eval** simulates: make one 72h forecast, dispatch greedily for 72h using only that forecast.

**Production** does: make a 72h forecast for initial planning, then replan every 5 minutes using a 14h MPC window. Errors at hours 30–72 are corrected ~60 times before execution.

**Implication:** The holistic eval penalises long-horizon forecast errors as if dispatch decisions are locked in. In production, these errors are largely irrelevant. A model that is excellent at 0–14h and mediocre at 14–72h would score badly on the eval but perform well in production.

The eval is a valid *relative comparison* tool (better model should still rank better) but the absolute +9.7% number does not translate to a production revenue estimate. A rolling MPC eval (contiguous windows, SoC carryover, 5-min replanning) would be more representative.

---

### 2. Is the training objective aligned with dispatch value?

**Training metric:** `pw_wMAPE` = price-weighted + horizon-weighted nMAPE.  
Horizon weights: tau=14 steps → 4h step: 0.56× baseline weight; 16h: 0.10×; 28h: 0.02×; beyond 28h: effectively 0.

**Observable consequence (Run 013, currently training):**

| Epoch | 4h nMAPE | 72h nMAPE | Unweighted nMAPE | pw_wMAPE |
|---|---|---|---|---|
| 1 | 56.7% | 59.4% | 57.9% | 51.26% |
| 6 | 40.6% | 90.3% | 73.5% | 39.40% (best) |
| 12 | 40.4% | 99.9% | 79.6% | 39.42% |

The model is learning to nail near-horizon accuracy while the 72h nMAPE approaches 100% (random). pw_wMAPE reports "improving" because it barely weights the horizon it's destroying. The training metric is actively driving the model toward a degenerate solution at long horizons.

**If MPC is the real decision maker** (Question 1), this may not matter in production. But it makes the holistic eval result unreliable as a signal — the eval rewards 72h accuracy, the training metric discourages it.

---

### 3. Is PD7Day actually useful given 3.5% training coverage?

PREDISPATCH only reaches ~28h. PD7Day covers 72h but only started being ingested in February 2026 — meaning steps 57–144 (28h–72h) of the decoder have ~3.5% training coverage. The model cannot reliably learn from these steps. Combined with the near-zero training weights at those steps, the TFT is essentially doing mean reversion beyond 28h.

**Question:** Is the 72h forecast horizon providing any real value, or is it just reassuring packaging around a 28h model?

---

### 4. Does stacking three ML models help or compound errors?

Current architecture for spike windows:
1. Spike classifier: `P(spike)` from LightGBM on PREDISPATCH features
2. Debiaser: LightGBM OOF correction of PREDISPATCH, conditioned on `prob_spike`
3. TFT: encoder-decoder with debiased PREDISPATCH as decoder input

Each model introduces its own error. The debiaser corrects a bias in the TFT's input, but was trained on data generated by the TFT (OOF), creating coupling. Changing the debiaser requires retraining the TFT, and vice versa. Three-way comparison results:

| Config | All | Spike | Normal |
|---|---|---|---|
| 011b + binary routing (best) | +9.7% | +7.2% | +0.4% |
| 011b + unified debiaser (no TFT retrain) | −9.9% | −16.2% | +4.2% |
| 012 + unified debiaser (TFT retrain, epoch 2) | −28.3% | −29.9% | −44.5% |

The unified debiaser degraded results significantly even after retraining, with Run 012 barely training (converged at epoch 2 out of 100). The system is brittle to debiaser changes.

**Counter-argument:** The binary routing approach (+9.7%) does work. The brittleness emerged from an architectural change attempt, not from the existing architecture.

---

### 5. Is normal stratum performance acceptable?

Normal stratum = flat-price, typical SA1 days. These represent the majority of production hours. The best result is +0.4% vs baseline. This is effectively noise.

**What does +0.4% mean in practice?** If the baseline earns $0.52/day on normal days and we get +0.4%, that's $0.002/day improvement. Within any reasonable confidence interval this is indistinguishable from zero.

The gain is entirely driven by spike windows (+7.2%) and low/negative price windows (+32.6%). The model provides no value on the most common day type.

**Root cause:** On normal days, PREDISPATCH is already a reasonable forecast. The TFT learns to pass it through. The question is whether the spike-stratum gains are large enough and stable enough to justify the complexity.

---

### 6. What is this worth in dollar terms?

Rough estimate (for context, not commitment):
- Oracle (ceiling) earns $6.00/window = ~$2.19/day
- Baseline (`amber_apf_lgbm`) earns $2.99/window = ~$1.09/day  
- Hybrid earns $3.28/window = ~$1.20/day (+$0.11/day vs baseline)

Annual gain estimate: ~$106/year over a naive baseline that already uses Amber APF + LightGBM (0.29 $/window × ~365 days). This is before accounting for the complexity of maintaining the pipeline vs simply using Amber APF.

*Note: these are retrospective estimates on sampled windows — not production measurements. The actual production gain may be higher or lower.*

---

### 7. Is the spike gate sample stable?

The holistic eval uses 300 spike windows, each independent (no SoC carryover). Run-to-run variance on 300 samples of a volatile metric (arbitrage revenue on SA1 spike days) is unknown. A bootstrap confidence interval on the +9.7% result has not been computed.

It is possible that +9.7% is within the noise of a different 300-window sample. This should be quantified before treating the result as a reliable gate.

---

### 8. Is PREDISPATCH the right input signal, or is Amber APF better?

The baseline (`amber_apf_lgbm`) uses Amber's Advanced Price Forecast (APF) directly for the first ~28h, then extends with LGBM. Amber APF is commercially produced, well-calibrated, and includes real-time confirmed prices. The TFT instead takes raw PREDISPATCH (which is known to be biased) and debiases it.

PREDISPATCH is available earlier and includes demand/interchange features that Amber APF does not expose. But Amber APF has better accuracy at 0–14h (the MPC horizon that actually matters). 

**Has anyone checked whether raw Amber APF + TFT extension beats debiased PREDISPATCH + TFT?** This comparison has not been done.

---

## Summary Assessment

| Question | Status |
|---|---|
| Is +9.7% vs baseline real? | Plausible but unquantified confidence interval |
| Does 72h eval match production (MPC)? | No — eval over-penalises long-horizon errors |
| Is training metric aligned with dispatch? | No — pw_wMAPE actively destroys 72h accuracy |
| Is spike gain worth the complexity? | Unclear — only meaningful on ~37% of windows |
| Normal stratum value | Effectively zero |
| Dollar value | ~$106/year incremental, rough estimate |
| Architecture brittleness | High — debiaser↔TFT coupling requires joint retraining |

---

## Suggested Questions for Independent Reviewer

1. Given that EMHASS MPC replans every 5 minutes over a 14h horizon, what fraction of the 72h forecast actually influences executed dispatch decisions? Does a 72h model make sense at all, or should we focus on a high-quality 14h model?

2. Should the financial eval be a rolling MPC simulation (contiguous windows, SoC carryover, 5-min replanning) rather than independent one-shot 72h windows? How much would this change the relative ranking of models?

3. The training metric (pw_wMAPE) heavily down-weights horizons beyond 4h. Is this the correct objective for an energy arbitrage application? Would a metric based directly on simulated dispatch revenue (with replanning) better align training with production use?

4. Is the OOF debiaser + spike classifier + TFT stack architecturally sound, or does the tight coupling (each component trained on outputs of others) make this inherently fragile? Would a single end-to-end model be preferable?

5. The system earns an estimated +$106/year over a baseline that already uses Amber APF (rough estimate from sampled eval windows). Is this incremental gain worth the ongoing maintenance of a multi-model pipeline? What would the gain need to be to justify the complexity?

6. Normal stratum (+0.4%) is effectively zero value. Is this a fundamental limitation of price forecasting on typical-price days, or is there an architectural fix?

---

## Implementer Responses to Reviewer Questions

### Q3: How is 72h terminal SoC passed to the 14h MPC?

**Short answer: it's a soft-but-real coupling — the reviewer's concern is valid.**

The MPC config does this every 5 minutes:

1. Read the day-ahead SoC schedule (`sensor.dh_soc_batt_forecast`) — the SoC trajectory that the 72h day-ahead optimisation planned
2. Look up what that schedule says SoC *should* be in 14h
3. Add any positive deviation (if we've charged more than planned, carry that forward as credit)
4. Pass the result as `soc_final` to the 14h MPC

```jinja
{%- set future_target_time = quantized_now + timedelta(hours=14) %}
{%- set planned_soc_future = get_planned_soc(future_target_time, ...) %}
{%- set deviation = (actual_soc_now - planned_soc_now) + 0.5 %}
"soc_final": {{ planned_soc_future + positive_deviation_only }}
```

The 14h MPC does inherit a terminal SoC target from the 72h plan. It is not a hard mechanical fence — EMHASS uses it as a cost term — but a bad 72h forecast will propagate a wrong 14h SoC target. The `+0.5` bias and positive-only deviation correction provide one-directional slack: you can be ahead of plan, but not behind.

**Concrete failure mode:** if tomorrow has a spike but the 72h forecast misses it, the day-ahead schedule will plan to discharge tonight, and the 14h MPC will inherit a low-SoC target for tomorrow morning — exactly when battery should be reserved.

**One mitigation in place:** `battery_minimum_state_of_charge` is computed dynamically as the highest of a hard floor, current SoC minus a configurable buffer, and a minimum target. This prevents full depletion even if the plan says to. But it is a floor, not a spike-aware reserve.

---

### Q4: Feature continuity at the PREDISPATCH → PD7Day splice (step 56/57)

**The splice is real and the reviewer is right to ask.**

At step 56, four of nine decoder continuous features step-change to zero:

| Steps | pd_rrp | pd_demand | pd_net_interchange | vic1_pd_rrp | nsw1_pd_rrp |
|---|---|---|---|---|---|
| 0–55 | PREDISPATCH (real) | PREDISPATCH (real) | PREDISPATCH (real) | PREDISPATCH (real) | PREDISPATCH (real) |
| 56–143 | PD7Day (real) | **0-filled** | **0-filled** | **0-filled** | **0-filled** |

The model is told about this via a `covar_missing` flag and `horizon_norm` (step position 0→1 across decoder). But those are single scalar signals — the model must learn from experience that "when `horizon_norm > 0.39` and `covar_missing=1`, ignore the zero-filled features." With only ~3.5% training coverage at steps 57–144, there are roughly 1,000 training samples with any PD7Day signal at all. That is very few examples of the splice for the attention heads to learn the transition from.

**Observable symptom:** Run 013's 72h nMAPE diverging to ~100% while 4h nMAPE improves to ~40% is consistent with the model breaking at or after the splice boundary. The training metric doesn't penalise it for this, so it goes uncorrected.

---

### Q5: Should we split into two decoupled models?

**Yes, and this is probably the right architectural pivot.**

The two use cases have fundamentally different requirements:

| | Model A (MPC execution) | Model B (day-ahead planning) |
|---|---|---|
| Horizon | 0–28h | 28–72h |
| Input | PREDISPATCH (dense, multi-feature) | PD7Day (rrp only, low coverage) |
| Loss | Dispatch regret or short-horizon MAPE | Soft directional accuracy (high/low) |
| Accuracy required | High — directly drives executed dispatch | Low — only sets a SoC target that gets revised |
| Training samples | ~15,000 | ~1,000 (growing) |

Running them as one model forces `pw_wMAPE` to trade them off, and as Run 013 demonstrates, it sacrifices the 28–72h component entirely. Splitting allows:

- **Model A** trained aggressively on short-horizon dispatch accuracy with an appropriate loss (dispatch regret, or simple unweighted MAPE over 0–28h)
- **Model B** simplified — even a calibrated mean-reversion or raw PD7Day with a lightweight correction may suffice for terminal SoC guidance, which only needs to be directionally correct
- Different update cadences and independent failure modes

The cost is two training pipelines and a blending layer. But the current architecture already carries equivalent complexity through the debiaser + spike classifier + single TFT stack — it is just hidden inside one model boundary.

---

### Q6: Is a Rolling MPC Eval feasible to build?

**Technically feasible, but constrained to ~75 days of history if PD7Day is required.**

AEMO data available:

| Source | Coverage |
|---|---|
| PREDISPATCH (parquet) | April 2024 → present, run_time-indexed |
| PD7Day | February 2026 → present only (no historical archive exists) |
| 5-min actuals | March 2024 → present |

A rolling backtester would walk forward in 30-min steps, retrieve whatever PREDISPATCH run was available at each `now`, run TFT inference, simulate one dispatch step, carry SoC forward. The PREDISPATCH run-time index makes the "what was available at time T" reconstruction exact.

**Constraint:** any model using PD7Day beyond step 56 has only ~75 days of history to evaluate against. That gives roughly 3,600 decision points — enough for a meaningful eval, but with limited spike event coverage.

**Recommended scope:** build the rolling eval for Model A only (0–28h, PREDISPATCH). This has full history back to 2024, directly tests the component that drives executed MPC decisions, and sidesteps the PD7Day coverage gap. Model B (28–72h planning) is better evaluated separately on its actual purpose — does it set better SoC targets? — rather than folded into a dispatch revenue metric where its long-horizon errors are largely irrelevant to execution.

**On trusting current results:** the +9.7% one-shot result is a valid relative ranking signal — a better model should still rank better under rolling eval — but it should not be treated as a production revenue estimate. The rolling eval is the right long-term investment before committing to the full architecture.
