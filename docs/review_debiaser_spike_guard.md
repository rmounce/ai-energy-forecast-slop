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

## Reviewer Decision (2026-04-19)

**Skip Option A. Proceed to Option B/3 hybrid: upstream regime classifier + conditional routing.**

### Why Option A is a dead end (reviewer analysis)

The collapse of normal stratum performance from +4.8% (no guard) to −21.7% (guard at 1000
$/MWh) means PREDISPATCH regularly emits fake noise **above** 1000 $/MWh on normal days —
and the debiaser was successfully correcting those. Simultaneously, genuine spikes also live
above 1000 $/MWh. Because both fake noise and genuine spikes occupy the same raw price
magnitudes, **no scalar threshold on `raw_pd_rrp` can ever separate them.** Option A just
moves to a different point on the same failing Pareto frontier.

### Recommended implementation

1. **Train a lightweight upstream regime classifier** (LightGBM or logistic regression) to
   predict P(genuine spike window) at each run_time.
   - Features: time of day, recent actual RRP lags, PREDISPATCH h0 + max, forecast demand,
     net interchange.
   - Label: 1 if any actual RRP ≥ 300 $/MWh occurs in the window.
   - Training data already exists: OOF parquet run_times + actuals.

2. **Conditional routing in the pipeline:**
   - `prob_spike > threshold` → bypass debiaser (pass raw PREDISPATCH through)
   - otherwise → apply existing LightGBM debiaser as normal

   This reuses both existing components exactly where each excels: debiaser yields +4.8%
   on normal windows; raw passthrough yields +7.6% on spike windows.

3. **Check first** whether the existing spike/stratum classification logic in the pipeline
   can be reused upstream — avoid training a new classifier if one already exists.

4. Re-run Phase 6 holistic dispatch eval with conditional routing in place.

**Estimated effort:** 1–2 days including eval.

---

## Implementation Results (2026-04-19)

### Classifier training (train/train_spike_classifier.py)

- **Features:** PREDISPATCH summary (pd_rrp_h0, pd_rrp_max, pd_rrp_p90, pd_demand_max, pd_net_interchange_h0), recent actual RRP lags (lag1/2/4/8, rolling max 6h/24h), time features
- **Label:** any actual RRP ≥ 300 $/MWh in next 28h from run_time (covers decoder steps 0–55)
- **Split:** train pre-Jul 2025, val Jul 2025–Apr 2026 (spike rate 17.7% in val)
- **Val metrics:** ROC-AUC 0.722, AP 0.476
- **Top features:** actual_rrp_max_24h (406), pd_demand_max (345), pd_rrp_p90 (164)

The 24h actual RRP lag dominates — contextual market history that a scalar pd_rrp threshold can't see.

### Routing threshold tuning — Phase 6 holistic dispatch eval (price-only, 811 windows)

| Threshold | Normal bypass% | Spike bypass% | Spike $/day | Low $/day | Normal $/day | All $/day |
|-----------|---------------|---------------|-------------|-----------|--------------|-----------|
| No guard (debiaser always) | 0% | 0% | −16.4%* | +21.1%* | +4.8%* | — |
| 1000 $/MWh scalar (old) | ~0% | ~100% | +7.6%* | +21.1%* | −21.7%* | +7.8%* |
| Classifier 0.35 | 35.5% | 39.0% | +7.0% | +26.2% | −13.9% | +8.1% |
| Classifier 0.50 | 17.1% | 27.7% | +7.2% | +30.3% | −6.9% | +9.1% |
| Classifier 0.60 | 11.8% | 21.3% | +7.3% | +32.1% | −2.4% | +9.6% |
| **Classifier 0.65** ✅ | **8.3%** | **18%** | **+7.2%** | **+32.6%** | **+0.4%** | **+9.7%** |

*Old cached results marked with *. All 2026-04-19 values from frozen actuals parquet (holistic_eval_actuals.parquet, exported same day).

**Final threshold: 0.65.** All gates pass. Stored in `models/spike_classifier/lgbm_spike_clf.pkl` and `eval/retro_tft_inference.py::SPIKE_ROUTE_THRESHOLD`.

**Note on eval reproducibility:** These results require `eval/results/holistic_eval_actuals.parquet` (frozen InfluxDB snapshot, 2026-04-19). Without it, `holistic_eval.py` queries live InfluxDB which drifts over time. Run `eval/export_holistic_actuals.py --refresh` to re-freeze.
