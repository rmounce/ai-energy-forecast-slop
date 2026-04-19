# Design Review: Debiaser Spike Guard Threshold

**Date:** 2026-04-19  
**Status:** Needs decision before Phase 5 remainder can resume  
**Context needed:** ~10 min read  

---

## Background

The TFT (Tier 2) price model was trained with **OOF-debiased PREDISPATCH prices** at decoder
steps 0–55. The debiaser is a LightGBM model that corrects AEMO's PREDISPATCH forecasts, which
systematically overestimate prices during demand peaks (MAE improvement: 325 → 65 $/MWh on the
training set).

The problem: **the debiaser was trained mostly on non-spike data**. When PREDISPATCH reports a
genuine high price (e.g. 5000 $/MWh during a price spike), the debiaser maps it toward the mean
(e.g. 269 $/MWh) — suppressing real spike signal that the TFT needs to make good dispatch
decisions.

To prevent this, a **spike guard** was added: if `|raw_pd_rrp| > THRESHOLD`, the raw PREDISPATCH
value passes through the debiaser unchanged. Below the threshold, the debiaser correction applies.

---

## The Trade-off

The threshold is currently **1000 $/MWh** (a round-number heuristic, explicitly marked TODO in
the code). The tension:

- **Threshold too low** → blocks debiaser from correcting overestimates in the 300–1000 $/MWh
  range. These commonly occur in *normal flat-price* windows when PREDISPATCH overshoots by a
  few hundred dollars. This directly hurts normal stratum dispatch quality.

- **Threshold too high** → allows debiaser to suppress genuine spike prices. Spike windows see
  raw PREDISPATCH prices of 5000–15000 $/MWh; if the guard doesn't trigger, the debiaser maps
  these to ~270 $/MWh and the TFT sees no spike coming.

We've tested two thresholds and a no-guard baseline. The results from the holistic dispatch eval
(811 windows, Jul 2025–Mar 2026, 30-min/72h windows):

| Configuration | Normal $/day vs baseline | Spike $/day vs baseline |
|---|---|---|
| No guard (debiaser applied everywhere) | **+4.8% ✅** | −16.4% ❌ |
| Guard at 300 $/MWh | ~−28% ❌ | (not fully tested) |
| Guard at 1000 $/MWh *(current)* | −21.7% ❌ | **+7.6% ✅** |

**Current gate thresholds:** Normal must be ≥ −2% vs baseline. Spike must be ≥ −20% vs baseline.

At 1000 $/MWh: spike passes (+7.6%), normal fails (−21.7%).  
Without guard: normal passes (+4.8%), spike fails (−16.4%).

There is no threshold we've found that passes both simultaneously, because the debiaser
wasn't designed with spike awareness.

---

## Root Cause

PREDISPATCH prices in the 300–1000 $/MWh range occur in two completely different situations:

1. **Genuine high-price periods** (approaching a spike): raw PREDISPATCH is accurate; debiaser
   should leave it alone.

2. **Normal periods with PREDISPATCH noise**: PREDISPATCH sometimes overshoots to 300–800 $/MWh
   during demand peaks when actual settlement will be 50–150 $/MWh. Debiasing these is
   beneficial.

The current guard can't distinguish these cases — it only sees the magnitude of the raw price.

---

## Options

### Option A: Principled threshold from training residuals

Derive the guard threshold from the debiaser's own training data — e.g. the 95th or 99th
percentile of actual RRP during the training period, or the price level above which the debiaser's
residuals become large/biased.

**Pros:** No model retraining. Fast to implement (inspect `models/pd_debiaser/` and the OOF
parquet). Principled justification instead of a round-number heuristic.

**Cons:** Still a scalar threshold — can't resolve the genuine-spike vs noisy-overestimate
ambiguity in the 300–1000 range. May shift the balance but is unlikely to pass both gates
simultaneously if the trade-off is fundamental.

**Effort:** ~2–4 hours.

---

### Option B: Retrain debiaser with spike-aware loss

Retrain the LightGBM debiaser with a loss function that penalises suppressing high prices more
than it penalises failing to correct normal overestimates. Options include:
- Asymmetric quantile loss (predict conditional median separately for spike/normal regimes)
- Sample weighting: upweight spike examples in training
- Two-model ensemble: one debiaser for spike regime, one for normal regime

**Pros:** Addresses the root cause. If the model learns to distinguish real spikes from noisy
overestimates using features (time of day, forecast demand, recent actual prices), the scalar
threshold limitation goes away entirely.

**Cons:** Requires retraining + re-evaluating. The debiaser training data (`debiased_pd_rrp_oof.parquet`)
needs to be inspected for spike coverage — if spikes are very rare in training, even
upweighting may not be enough without synthetic augmentation.

**Effort:** ~1–2 days including eval.

---

### Option C: Loosen the normal gate threshold

Accept that the current debiaser design can't cleanly satisfy both normal and spike simultaneously,
and widen the normal gate tolerance (e.g. from −2% to −10% or −15%).

**Pros:** Immediate unblock. No model changes.

**Cons:** Weakens the gate's meaning. The current normal underperformance (−21.7%) is large enough
that it likely reflects a real dispatch quality gap, not just noise.

**Not recommended** unless both A and B are assessed as too risky before a near-term production
deadline.

---

## Question for Reviewer

Given the above, which approach do you recommend?

1. **Option A first** — quick diagnostic to see if a principled threshold can resolve it; if not,
   escalate to B.

2. **Option B directly** — treat this as a model retraining task, skip the threshold search since
   the scalar-threshold approach is likely fundamentally limited.

3. **Something else** — e.g. separate the debiaser correction from the spike guard entirely using
   a regime classifier upstream.

The pipeline is otherwise in good shape: tactical eval (5-min/1h Tier 1) passed both accuracy
and dispatch gates on 2026-04-19. Phase 6 (30-min/72h strategic) passes on all strata except
normal. Resolving the normal gate is the last blocker before Phase 5 production work resumes.
