# Ideas & Future Directions

Speculative concepts worth revisiting once the core TFT pipeline is stable.
Not prioritised — captured here so they don't get lost.

---

## Battery Dispatch

### Spike-aware discharge (EMHASS-side, near-term)

Current problem: q50 forecast mean-reverts quickly after spike onset because PREDISPATCH
is biased toward normal prices once dispatch clears. The q90 spread captures price *level*
uncertainty at each step independently — it doesn't model spike *duration* autocorrelation.

Simple fix in EMHASS dispatch logic (no TFT change needed):
- Add a derived feature: "fraction of last N 30-min intervals where RRP > threshold"
- When this feature is high, the system is likely mid-spike — increase the sell-hold
  bias (e.g. push sell threshold even higher toward q90, or lock out discharge entirely
  for 1–2 intervals)
- No model retraining required; pure rule-based overlay on top of quantile forecasts

### Spike duration model

Separate companion model (classifier or survival model) trained to predict:
- "Is this the start of a new price spike?"
- "Given we're in a spike, what is P(spike continues for ≥ N more intervals)?"

Spike duration has fat tails in SA1 — the q10/q50/q90 forecast doesn't encode this
because it models price level at each step independently. A survival/hazard model on
spike run-lengths could feed directly into EMHASS dispatch as an additional signal.

### Direct value optimisation / batch RL

Instead of predicting prices and then dispatching, train a model that directly predicts
the *expected net battery revenue* of each dispatch action (charge/hold/discharge) under
a stochastic price process. This is equivalent to Q-learning over the battery state space.

Requires:
- Simulating EMHASS dispatch decisions at training time
- A differentiable model of battery SoC transitions and tariff structure
- Significantly more complexity than the current price-forecast → dispatch pipeline

Worth considering only after the price forecast itself is solid and producing real
production value. The current quantile + asymmetric threshold approach is a reasonable
proxy and much simpler to debug.

---

## Price Forecasting

### ~~Temporal sample weighting~~ ✅ Done

Event-stratified importance weighting implemented: exponential decay (half-life ~90 days)
for baseload regime (−$50 to $150), with a 50% floor for extreme events (price < −$50 or
> $150). Prevents catastrophic forgetting of 2022 energy crisis data while keeping recent
market dynamics dominant. See plan file for design rationale.

### ~~Quantile calibration — Tier 1~~ ✅ Done (Phase 4)

Conditional conformal prediction applied to Tier 1 tactical LightGBM. Stratified by
physical regime (spike / oversupply / normal). Spike q95 coverage 0.750 → 0.821.
Deltas stored in `models/lgbm_tactical/conformal_deltas.json`.

**Tier 2 TFT calibration still needed** — lower tail (q05/q10) is biased due to
log-scaling compressing negatives. Do not use Tier 2 q05/q10 for dispatch thresholds
until Tier 2 conformal calibration is complete. Deferred pending Phase 6+8 gate.

### ~~P5MIN integration~~ ✅ Done (2026-04)

Implemented as Tier 1 tactical LightGBM. `ingest/ingest-p5min.py` ingests to
`rp_5m.aemo_p5min_forecast`. Inference: `_execute_tactical_prediction()` in `forecast.py`.
Publishes `sensor.ai_p5min_price_forecast` and contributes to `sensor.ai_combined_*_price_forecast`.

### Ensemble / model averaging

Average predictions from multiple TFT checkpoints (different random seeds, different
`--horizon-decay` values, different training windows). Ensembles typically reduce variance
and improve quantile calibration. Low implementation cost once the single-model pipeline
is working.

### ~~VIC1 + NSW1 prices as decoder features~~ ✅ Done (Run 011b)

`vic1_pd_rrp` and `nsw1_pd_rrp` added to `DEC_CONTINUOUS` in training dataset and inference.
Both are populated from `rp_30m.aemo_predispatch_forecast` for VIC1/NSW1 regions.
EnergyConnect (SA1↔NSW1) expected ~2026–2027; `covar_missing` flag handles sparse NSW1 data.

### ~~SevenDayOutlook as decoder covariate~~ ✅ Done (Run 011b)

`sd_demand` and `sd_net_interchange` from `rp_30m.aemo_sevendayoutlook` (SA1 region)
added to all 144 decoder steps. Fills the gap beyond PREDISPATCH horizon (steps 56–143).
Ingested by `ingest/ingest-sevendayoutlook.py`, queried at inference via `_get_influx_sdo_demand()`.

---

## Infrastructure

### UTC-first timezone handling

The codebase mixes timezone conventions in a way that creates recurring train/inference skew:
- `time_sin_cos()` hardcodes `"Australia/Brisbane"` (UTC+10, no DST)
- `build_load_dataset.py` uses `"Australia/Adelaide"` (UTC+9:30 / UTC+10:30)
- `add_time_features()` uses `CONFIG['timezone']` = Adelaide
- InfluxDB stores timestamps in UTC; pandas operations can silently shift on DST boundaries

The right fix is to standardise on UTC throughout — compute all time features from UTC timestamps directly (e.g. `hour = (utc_hour * 60 + utc_minute) / 60.0` mapped to local-hour via a fixed offset), and only convert to local time at the display/publish boundary. This removes DST as a source of train/inference feature skew entirely.

Not urgent while models are in shadow mode, but should be resolved before any model is promoted to primary. The load TFT `_time_sin_cos_local()` workaround is a stopgap.

### Event-driven pipeline service (Phase 7)

Replace systemd timers with a single persistent `ai-energy-pipeline.service` using
HA WebSocket subscriptions and an internal scheduler. Enables: (a) Tier 1 firing within
~5s of each Amber price update, (b) in-process model caching (eliminates ~30s cold-start),
(c) natural place to wire HA tail-risk automations. See plan file for full design sketch.
Deferred until Phase 8 (test framework) is complete.

---

## Evaluation & Metrics

### ~~Spike-aware evaluation set~~ ✅ Done (Phase 2/3)

Stratified eval sets built for both Tier 1 (1,600 samples: 500 spike ≥$300, 300 low/negative, 800 normal)
and Tier 2 TFT (900 samples). Generated by `data/build_stratified_eval_tactical.py` and
`data/build_stratified_eval.py`. Durable across dataset rebuilds.

### Evaluation metric: nMAPE, revenue-weighted, or dispatch-regret?

**Why not MSE or raw MAE?**
- **MSE** squares errors. Electricity prices reach the $15,000 market cap; a handful of
  spike events would dominate the loss and make training numerically unstable. Avoid.
- **MAE** is linear but scale-dependent — a $50 error on a $100 price equals a $50 error
  on a $1,000 price. Doesn't normalise for price level.

**Current: nMAPE / wMAPE** = normalised MAE = `sum(|e|)/sum(|y|)`. Scale-invariant,
standard for electricity price evaluation. Limitation: treats relative errors symmetrically
regardless of economic impact. A 20% error on a $300 spike and on $70 baseload are equal.

**Implemented (Run 008): progressive price-weighted loss**
`weight = 1 + log1p(max(0, (price − p50_ref) / p50_ref))` applied per-step in training loss.
Chosen over sharp threshold (avoids discontinuity) and CDF-based (similar shape, simpler).
`pw_wMAPE` (price+horizon weighted) is now the headline training and early stopping metric.
`y_weights.npy` produced by `build_training_dataset.py`; loaded by `train_tft_price.py`.

**Longer-term: dispatch-regret metric**
Simulate charge/hold/discharge decisions on the forecast vs actuals; measure lost revenue
against theoretical perfect-foresight dispatch. Gold standard for this use case. Requires
a simplified but realistic EMHASS/battery model at evaluation time. Complex to implement
correctly — do after TFT is in production and producing real value.

**Pinball/quantile loss** (current training objective): keep for training. Not ideal as
standalone eval metric — not interpretable in dollar terms.

### Amber APF — CSIRO Kick-Start case study

CSIRO worked with Amber Electric on APF; case study at:
https://www.csiro.au/en/work-with-us/funding-programs/SME/CSIRO-Kick-Start/Case-studies/Amber-Electric

**Key finding (external review, 2026-04-12):** CSIRO did not create a proprietary data
source. Their approach: identify "periods of concern" where AEMO PREDISPATCH forecast high
prices but actuals came in low (typically due to generator rebidding or late-stage
constraints). They trained a model specifically to predict these PREDISPATCH divergence
events. Amber integrated this as a subset model in SmartShift.

**How our planned steps replicate this without Amber:**
- **P5MIN (Step 5):** Updates every 5 min, catches late generator rebidding that 30-min
  PREDISPATCH misses entirely. Directly addresses the "period of concern" divergence.
- **VIC1/NSW1 decoder features (Step 2e):** SA1 price divergence from PREDISPATCH is often
  driven by interconnector constraints. Adding adjacent-region prices gives the model exactly
  the constraint-divergence signal CSIRO likely used. Confirmed high-value by review.

Our Step 2e and Step 5 are already the right moves. VIC1/NSW1 first (lower effort,
no new ingest required), P5MIN after Step 3 is stable in production.
