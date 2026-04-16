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

### Temporal sample weighting

SA1 market dynamics are evolving (solar penetration, storage growth, VRE volatility).
Down-weight older training samples with `exp(-age_days / half_life)` via
`WeightedRandomSampler`. Allows aggressive historical backfilling while keeping recent
market dynamics dominant. Half-life ~90–120 days is probably right for SA1.

Implementation: add `--temporal-halflife` arg to `train/train_tft_price.py`; compute
sample ages from `run_times.npy`; pass `WeightedRandomSampler` to the training DataLoader.
This is a near-term improvement (Step 2d in the plan), not speculative.

### Quantile calibration

Are the q10/q50/q90 predictions actually calibrated? If q90 only contains actuals 80% of
the time, the user's asymmetric dispatch strategy (bias toward q90 for sell threshold) is
less conservative than intended.

Add a calibration diagnostic to `evaluate_tft.py`: for each quantile q, compute the
empirical coverage rate `P(actual ≤ pred_q)` and plot/report the calibration curve.
Well-calibrated quantiles make the dispatch strategy more reliable without any EMHASS
changes.

### P5MIN integration (near-term debiased signal)

AEMO P5MIN provides 5-min forecasts for the next 60 min, updated every 5 min. This is
the data needed to fully replace Amber APF's near-term debiased signal for 0–1h forecasts.

- `aemo_dispatch_sa1_5m` already in InfluxDB via CQ chain — bias correction training ready
- New `ingest/ingest_p5min.py` fetches from NEMweb `/Reports/Current/P5_MIN_PREDISPATCH/`
- Resolution cascade at inference: P5MIN (0–1h) → PREDISPATCH (1–28h) → PD7Day (28–72h)
- See plan file (Step 5) for full design

### Ensemble / model averaging

Average predictions from multiple TFT checkpoints (different random seeds, different
`--horizon-decay` values, different training windows). Ensembles typically reduce variance
and improve quantile calibration. Low implementation cost once the single-model pipeline
is working.

### VIC1 + NSW1 prices as decoder features (near-term, low effort)

SA1 has two high-capacity interconnectors:
- **Heywood** → VIC1 (~650MW, already operating)
- **Project EnergyConnect** → NSW1 (~800MW, expected full commission ~2026–2027,
  will be represented in NEM dispatch as a direct SA1↔NSW1 interchange)

`net_interchange` is already an encoder feature (the *result* of price differentials)
but not the *cause*. Adding VIC1 and NSW1 PREDISPATCH forecast prices as decoder features
gives the model the actual signals driving interchange — useful for spike precursors where
adjacent region prices lead SA1 convergence, and for constraint events where SA1 diverges.

VIC1 and NSW1 data are almost certainly already in InfluxDB (AEMO ingest files cover all
NEM regions; the SA1 filter is only applied at export time in `export_parquet.py`).

Implementation:
1. Verify in InfluxDB: `SELECT COUNT(rrp) FROM rp_30m.aemo_predispatch_forecast WHERE region='VIC1'` (and NSW1)
2. Add VIC1 + NSW1 exports to `data/export_parquet.py` → `aemo_predispatch_vic1.parquet`, `aemo_predispatch_nsw1.parquet`
3. In `data/build_training_dataset.py`, join both to each decoder step → add `vic1_pd_rrp` and `nsw1_pd_rrp` to `DEC_CONTINUOUS`
4. Rebuild dataset and retrain — EnergyConnect-era data will have both features populated; pre-commissioning NSW1 data will have sparse/missing NSW1 values, handled by the existing `covar_missing` flag

Note: once EnergyConnect is fully commissioned and SA1↔NSW1 interchange is represented in
NEM data, also add `nsw1_net_interchange` as an encoder feature alongside the existing
`net_interchange` (which currently represents the SA1↔VIC1 Heywood flow).

### Multi-region training

`aemo_sevendayoutlook_sa1.parquet` is already ingested and exported but not yet used in
training. AEMO's 7-day outlook provides scheduled demand and net interchange forecasts for
28h–7d ahead — this could improve the decoder's long-tail (steps 57–144) considerably,
particularly for the `pd_demand` and `pd_net_interchange` features that are currently
zero-padded beyond the PREDISPATCH horizon.

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

### APScheduler pipeline service

Replace the predict/train/Parquet-rebuild systemd timers with a single
`ai-energy-pipeline.service` using APScheduler. Enables inter-job dependencies
(e.g. "rebuild Parquet 5 min after PREDISPATCH ingest completes") without systemd
unit dependency gymnastics. See plan file for full design.

Make this change when TFT goes into production and the Parquet rebuild becomes a
dependency of the prediction path.

---

## Evaluation & Metrics

### Spike-aware evaluation set

**Recommended (external design review, 2026-04-12):** build this before the next significant
architecture change so future run comparisons are meaningful.

The problem: Run 003 wMAPE=31.2% and Run 005 wMAPE=40.6% are not comparable — the windows
differed (mild autumn vs volatile late-summer). A durable benchmark removes this ambiguity.

Rather than a rolling last-N-days val window (which may be mild or volatile by chance),
construct a stratified held-out eval set automatically selected to include:
- Major spike events (e.g. top 5% of 30-min intervals by volatility / RRP level)
- Extreme low / negative price events (heavy solar curtailment days)
- Normal diurnal cycles covering all 4 seasons

Selection criteria should be purely data-driven (no hard-coded dates), so the set is
stable and reproducible across dataset rebuilds. Candidate approach: rank all PREDISPATCH
run-times by some spike/volatility score, stratified-sample across deciles and seasons.

This would make `evaluate_tft.py` comparisons meaningful regardless of when training runs.

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
