# TFT Price Forecast: Design Rationale and Implementation Guide

**Current goal (V4 architecture, April 2026):** Replace Amber/CSIRO APF entirely with a
three-tier cascaded pipeline: (1) a multi-output LightGBM tactical model (0–60 min, 5-min
resolution), (2) a TFT strategic model (0–72h, 30-min resolution) using explicitly debiased
PREDISPATCH as its decoder covariate, and (3) a conditional conformal calibration layer
stratified by physical grid regime. See plan file for full V4 spec.

**Original goal (Runs 001–010):** Replace the current LightGBM price forecast (which uses Amber Electric's APF as a future covariate seed) with a Temporal Fusion Transformer (TFT) that learns horizon-dependent bias correction of AEMO pre-dispatch forecasts. Primary driver is **forecast accuracy**; retailer portability (removing the Amber dependency) is a side-effect.

**Paper reference:**
> Sinclair, Shepley, Hajati (2026) "Learning the Grid: Transformer Architectures for Electricity Price Forecasting in the Australian National Market". *Applied Sciences* 16(1), 75.
> GitHub: [github.com/redaxe101/TransformerApplicationNEM](https://github.com/redaxe101/TransformerApplicationNEM)

Key results: TFT achieved nMAPE 36.3–40.8% for NSW1 at 2–16h horizons. A decoder-only transformer (PatchTST variant) achieved 33.6–39.2%. Both beat AEMO's own P30 PREDISPATCH by 40–70%.

SHAP feature importance (Sinclair et al. Table 3): AEMO PREDISPATCH F_RRP alone accounts for >60% of total predictive importance. All other features (demand, weather, history) contribute only a few percent each. This is the primary justification for using PREDISPATCH as the decoder covariate rather than engineered features.

SA1 (Adelaide/South Australia) is the most volatile NEM region; expect nMAPE ~38–48% (vs 33–40% for NSW1) due to higher price spike frequency.

**Metric Integrity:** Evaluation uses a custom `nMAPE` computed against raw price targets. Point-wise nMAPE (dividing each error by specific price) is avoided to prevent explosive metrics during $0 or negative prices. Instead, a global normalization is used: `sum(|error|) / sum(|actual|)`.

---

## Data Sources

All data ingested to InfluxDB (`hass` database, `rp_30m` retention policy) by scripts in `ingest/`.

### PREDISPATCH (`aemo_predispatch_forecast`)
- **What:** AEMO's 30-min pre-dispatch price, demand, and interchange forecast
- **Frequency:** Every 30 min, at approximately :12 and :42 past the hour
- **Horizon:** Variable, ~28h into the future (see "PREDISPATCH horizon cycle" below)
- **Tags:** `region` (SA1, VIC1, NSW1), `run_time` (ISO UTC, when run was issued)
- **Fields:** `rrp` ($/MWh), `total_demand` (MW), `net_interchange` (MW)
- **Backfill (InfluxDB):** March 2025 – April 2026 (13 months, ~990K rows for SA1)
- **Backfill (NEMSEER/NEMWeb):** April 2024 – February 2025 via `ingest/backfill_predispatch_nemseer.py` — extends parquet to ~1.87M rows, 33K runs. Uses NEMSEER for pre-Aug 2024 (MMSDM archive) and direct NEMWeb HTTP for Aug 2024+ (AEMO restructured the archive format).
- **Bias note:** Structurally biased upward due to strategic generator rebidding. TFT learns this bias correction via horizon-dependent attention.

### PD7Day (`aemo_pd7day_forecast`)
- **What:** AEMO's 7-day ahead price forecast (PD7DAY/PRICESOLUTION table on NEMWeb)
- **Frequency:** 3×/day (~07:10, ~12:45, ~17:50 AEST)
- **Horizon:** ~7 days (330–367 intervals)
- **Tags:** `region`, `run_time`
- **Fields:** `rrp` ($/MWh)
- **Backfill:** February 2026 – April 2026 (~60 days, 183 runs, ~64K rows for SA1). **No older archive exists** — only the CURRENT directory (~46 days) is available on NEMWeb.
- **Bias note:** Significantly biased upward (more so than PREDISPATCH). TFT must learn correction from limited training data initially; improves passively as data accumulates.

### SevenDayOutlook (`aemo_sevendayoutlook`)
- **What:** AEMO's 7-day demand/capacity/interchange outlook
- **Note:** Contains NO price data (demand/capacity only). Useful as a covariate for 28h+ horizon demand forecasting but not used in the current price model.

### Actuals (`aemo_dispatch_sa1_30m`)
- Historical SA1 dispatch price and demand, updated via `ingest/ingest-nem-data.py`
- `price` field = RRP in $/MWh; `total_demand` and `net_interchange` in MW
- InfluxDB data goes back to 2021-12-31; local sensor data (power_load, power_pv) from 2021-12-29; weather (temp, humidity, wind_speed) from 2023-04-19
- `data/export_parquet.py --actuals-only` re-exports actuals without overwriting the NEMSEER-backfilled PREDISPATCH parquet

---

## PREDISPATCH Horizon Cycle

AEMO PREDISPATCH runs every 30 min and forecasts up to "the end of the next trading day." Since trading days are 48 intervals (24h) long and PREDISPATCH runs 48 times per day, the available forecast horizon cycles predictably:

| Run time (UTC) | Mean future intervals | Mean horizon |
|---|---|---|
| 03:00 (13:00 AEST) | ~77.5 | ~38.8h |
| 14:00 (00:00 AEST) | ~55.5 | ~27.8h |
| 00:00 (10:00 AEST) | ~35.5 | ~17.8h |
| 02:00 (12:00 AEST) | ~31.5 | ~15.8h |

**Consequence for training dataset construction:** Using a hard cutoff of `output_length=56` (28h) would include only 47.9% of runs AND would systematically exclude all runs issued between ~UTC 15:00–02:00 (AEST 01:00–12:00). This creates severe time-of-day bias: the model would never see morning forecasts during training.

**Solution adopted:** Masked loss with `output_length=144` (72h). Each sample has a per-step mask indicating which steps have valid covariates. Steps 1–32 train from ~15K samples (97.9% of runs); steps 33–56 from ~8.5K (47.9%); steps 57–144 from ~1K initially (PD7Day backfill). No time-of-day bias.

**Missing Data Handling:** Non-present covariates are padded with `0.0`. However, since features are normalized via `QuantileTransformer` (Gaussian distribution), `0.0` maps to the median price (~$50-60/MWh). To prevent the model from hallucinating median prices for missing horizons (especially the 72h tail), a `covar_missing` boolean feature is provided. The LSTM learns to ignore median-mapped inputs when `covar_missing=1.0`.

---

## Covariate Construction: Option B (Run-Aligned)

### The problem

PREDISPATCH is a sequence of overlapping runs: each run at time T covers T+30min through T+28h. At training time, for each window, you must choose which run's data to use as the decoder covariate. This choice has major theoretical consequences.

### Three options analysed

**Option A — Contemporaneous (h≈1 always)**

For each interval T, use the most recent PREDISPATCH run issued before T. Because runs are issued every 30min, this always gives the h=1 forecast at each decoder step.

- Pros: simple; Darts-compatible
- **Fatal flaw:** At inference time the decoder spans T+1 to T+144 using a single PREDISPATCH run (h=1 to h=56). The model receives h=10, h=30, h=56 signals it was never trained on. The horizon-dependent bias correction — the *entire motivation for using PREDISPATCH* — cannot be learned.
- Verdict: **Do not use.** Acceptable only as a baseline to measure, not as production architecture.

**Option B — Run-aligned (h=k per window)** ← **ADOPTED**

For each training window at run_time T, use run-T's forecasts for all decoder steps. Decoder step k sees the h=k forecast. Model experiences h=1..56 within every window — exactly matching inference.

- Pros: theoretically sound; matches inference exactly; model learns h-dependent bias correction
- Cons: requires per-window covariate variation, which Darts' flat TimeSeries API cannot express
- **Verified in Sinclair et al. source code** (`build_pipeline.py`, line `run_dt = timestamps[i + input_length]` — encoder end = decoder start; run at that timestamp used for entire decoder window)
- Verdict: **Use this.** Implemented in `data/build_training_dataset.py`.

**Option C — Horizon-as-explicit-feature**

Store (predispatch_rrp, predispatch_h) as flat series where h[T] = steps between issue and T. Under contemporaneous construction, h[T]≈1 always. Degenerates to Option A with a constant extra feature.

- Verdict: **Does not solve the mismatch.** `horizon_norm` is included as a decoder feature for a different reason (see below) but is insufficient to substitute for correct construction.

### Why Darts cannot support Option B

`TFTModel.fit(series=..., future_covariates=fc_series)` accepts a single flat `TimeSeries`. When it slices training windows, every window at position T draws `fc_series[T]` — always the same value for a given absolute timestamp, regardless of which training window is being constructed. Per-window variation is architecturally excluded.

### Why pytorch-forecasting was evaluated but not used

pytorch-forecasting's `TimeSeriesDataSet(group_ids=["run_time"])` would allow one "group" per PREDISPATCH run, giving per-run decoder variation. However:
- The `time_varying_known_reals` (future covariates) must be consistent across all timesteps in both encoder and decoder periods
- For encoder steps (past), PREDISPATCH data from the run at T is not available for those past timestamps — only the run at T's decoder-period data is available
- Mapping around this requires significant data engineering that effectively rebuilds the pre-built array approach anyway

**Decision:** Custom PyTorch LSTM encoder-decoder with pre-built numpy arrays. Simpler, faster, and gives complete control.

---

## Normalisation

### Normalisation: Log-Scaling vs QuantileTransformer

As of Run 010, the target `rrp` and decoder covariates `pd_rrp` (SA1/VIC1/NSW1) use **Log-Scaling** instead of `QuantileTransformer`.

**Rationale:** `QuantileTransformer` (Normal distribution) compresses the extreme price tail ($300 to $16,000) into a very narrow range of z-scores (z=2.4 to z=3.4). This "blinds" the loss function to the magnitude of spikes. Log-scaling preserves relative distance in the tail: a jump from $1,000 to $10,000 remains several times larger than a jump from $50 to $150.

**Implementation:**
`scaled = sign(x) * log1p(abs(x) / 60.0)`
Target quantiles are predicted in log-space and inverse-transformed for evaluation. Other features (demand, PV, weather) continue to use `QuantileTransformer`.

---

## `horizon_norm` Feature

`horizon_norm = (h-1) / (output_length-1)`, range [0,1] across decoder steps h=1..144.

This is Sinclair et al.'s "hours to delivery" feature, confirmed as the second most important SHAP feature in their ablation study. It serves two purposes:
1. Tells the model how far into the future the current decoder step is
2. Implicitly encodes which data source to trust (PREDISPATCH for small h, PD7Day for large h, zeros/missing for very large h without PD7Day coverage)

The model learns regime-switching behaviour: for h=1..56 it has PREDISPATCH with high accuracy; for h=57..144 it has PD7Day with lower accuracy; `horizon_norm` is the continuous signal that drives this.

---

## Model Architecture

LSTM encoder-decoder with Gated Residual Networks and cross-attention. Simplified TFT (Lim et al. 2021) without variable selection networks (which can be added later).

```
Input:
  Encoder [B, 96, 18]:  GRN → LSTM(d_model, n_layers)  → enc_out [B, 96, d_model]
  Decoder [B, 144, 13]: GRN → LSTM(d_model, n_layers)  → dec_out [B, 144, d_model]
                                  ↑ initialized with encoder LSTM final (h, c)

Cross-attention:  MultiheadAttention(dec_out, enc_out, enc_out) + residual + LayerNorm
Post-attention:   GRN(d_model)
Output:           Linear(d_model → 3) → [B, 144, 3]  (q30, q50, q70)
```

**Hyperparameters (defaults):**
- `d_model=64` (CPU-optimised; increase to 128+ with GPU)
- `n_heads=4`
- `n_layers=2`
- `dropout=0.1`
- Parameters at d_model=64: ~190K

**Loss:** Masked QuantileLoss averaged over valid decoder steps. `mask[h]=1` where both forecast covariate and actual target exist. quantiles: `[0.3, 0.5, 0.7]`.

**Training:**
- Adam, lr=1e-3
- CosineAnnealingLR (T_max=n_epochs, eta_min=0.01×lr)
- Early stopping on val loss, patience=7
- Gradient clipping, max_norm=1.0
- Default: 100 epochs

---

## Feature Sets

### Encoder (18 features, past 96 × 30min = 2 days)

**Base features (8, always available):**

| Feature | Source | Notes |
|---|---|---|
| `rrp` | `aemo_dispatch_sa1_30m.price` | Actual SA1 spot price $/MWh |
| `total_demand` | `aemo_dispatch_sa1_30m.total_demand` | SA1 demand MW |
| `net_interchange` | `aemo_dispatch_sa1_30m.net_interchange` | SA1 net imports MW |
| `power_load` | `rp_30m.power_load_30m` | House load W (30-min avg) |
| `power_pv` | `rp_30m.power_pv_30m` | Solar PV generation W |
| `temp` | `rp_30m.temperature_adelaide` | Adelaide temp °C |
| `humidity` | `rp_30m.humidity_adelaide` | Adelaide humidity % |
| `wind_speed` | `rp_30m.wind_speed_adelaide` | Adelaide wind speed km/h |

**5-minute volatility features (3, available from 2025-03-31; ~50% of training samples):**

| Feature | Source | Notes |
|---|---|---|
| `rrp_5m_max` | `rp_5m.aemo_dispatch_sa1_5m` | Max 5-min price in the 30-min window — captures intra-period spike peaks smoothed by 30m averaging |
| `rrp_5m_std` | same | Std of 5-min prices in the 30-min window — intra-period price chaos signal |
| `rrp_persistence` | same | Count of 5-min intervals in the last 1h with price > $150 — regime persistence detector |

Motivation (Run 007): when a spike is building, 5-min prices jump from $50 → $300 → $1000 within a single 30-min window — information the 30-min average obscures. The TFT attention mechanism can learn to selectively attend to these volatility signals. Missing steps (pre-2025-03-31) are 0-filled and flagged by `rrp_5m_missing`.

**Non-scaled features (7):**

| Feature | Notes |
|---|---|
| `hour_sin/cos` | Diurnal cycle encoding |
| `dow_sin/cos` | Weekly cycle encoding |
| `month_sin/cos` | Annual seasonality encoding |
| `rrp_5m_missing` | Binary: 1.0 where 5m data unavailable (pre-era or gap); allows model to ignore 0-filled 5m features |

### Decoder (13 features, future 144 × 30min = 72h)

| Feature | Source | Steps | Notes |
|---|---|---|---|
| `pd_rrp` | PREDISPATCH `rrp` | 1–56 | PREDISPATCH run issued AT encoder/decoder boundary |
| `pd_rrp` | PD7Day `rrp` | 57–144 | Most recent PD7Day run ≤ T (Option A; ~8h max staleness; acceptable) |
| `pd_demand` | PREDISPATCH `total_demand` | 1–56 | 0 for steps 57–144 (not in PD7Day) |
| `pd_net_interchange` | PREDISPATCH `net_interchange` | 1–56 | 0 for steps 57–144 |
| `vic1_pd_rrp` | VIC1 PREDISPATCH `rrp` | 1–56 | Adjacent region price — Heywood interconnector (~650MW) spike precursor |
| `nsw1_pd_rrp` | NSW1 PREDISPATCH `rrp` | 1–56 | Adjacent region price — EnergyConnect (~800MW, commissioning 2026–2027) |
| `hour_sin/cos` | computed | 1–144 | Future time encodings |
| `dow_sin/cos` | computed | 1–144 | |
| `month_sin/cos` | computed | 1–144 | |
| `horizon_norm` | computed | 1–144 | (h-1)/143; "hours to delivery" |
| `covar_missing` | computed | 1–144 | 1.0 where no valid covariate exists; prevents model treating 0-padded steps as median price |

---

## Dataset Stats (as of 2026-04-12, Run 006/007 rebuild)

| Metric | Value |
|---|---|
| PREDISPATCH SA1 rows | ~1,878,000 (33,844 runs, April 2024 – April 2026) |
| PREDISPATCH VIC1/NSW1 rows | ~390,000 each (decoder features, same run coverage) |
| PD7Day SA1 rows | 64,294 (183 runs, Feb 2026 – April 2026) |
| 5m dispatch SA1 rows | ~97,000 (2025-03-31 → 2026-04-12 → aggregated to 17,194 30m slots) |
| Training samples (Run 007 rebuild) | 14,736 train / 2,211 val / 431 stratified eval hold-out |
| Stratified eval set | 900 samples: 300 spike (top 5% RRP) + 200 low/negative + 400 seasonal normal |
| Steps 1–16h mask coverage | ~98% |
| Steps 1–28h mask coverage | ~87% |
| Steps 28–72h mask coverage | ~11% (growing as PD7Day accumulates) |
| 5m feature coverage | ~50% of encoder steps (pre-2025-03-31 = 0-filled + flagged) |
| Target RRP stats ($/MWh) | mean=87.8, p50=58.7, p99=~220, max=20,300 |

**Key constraint:** encoder requires 2 days of actuals before each run_time. Actuals go back to 2024-03-29 in InfluxDB (weather only to 2023-04-19, which still covers the full backfill window). PREDISPATCH runs before 2024-04-01 cannot be used as training samples.

## Production Integration (Shadow Mode)

As of April 2026, the TFT model (Run 010) is running in **Shadow Mode** within the `forecast.py` pipeline.

### Architecture
- **Worker:** `_execute_tft_prediction` in `forecast.py`.
- **Inputs:** 48h historical encoder (30m) + 72h future decoder (30m).
- **Regime Features:** High-frequency (5m) price spikes, log-momentum, and 30m volatility are engineered in real-time.
- **Safety:** Sandboxed execution via `try-except` to ensure zero impact on the primary LightGBM battery dispatch signal.

### Entities (Home Assistant)
- `sensor.ai_tft_price_forecast` (q50)
- `sensor.ai_tft_price_forecast_low` (q30)
- `sensor.ai_tft_price_forecast_high` (q70)

### Logging
Shadow predictions are logged to `tft_price_forecast_log.csv` for objective benchmarking against the local actuals.

---

## Options Considered and Discarded

### Darts TFTModel
- First implementation used Darts with `future_covariates` from InfluxDB queries
- Failed: `GROUP BY run_time` on 17,814 unique tags loaded 2.97M rows via HTTP JSON; data loading hung for 8.5 hours and never completed
- Root cause: InfluxDB is optimised for time-range queries, not high-cardinality tag enumeration
- **Solution:** Parquet cache layer (`data/export_parquet.py`) + pre-built numpy arrays

### pytorch-forecasting TimeSeriesDataSet
- Investigated as the "native" way to implement per-run training samples
- Problem: `time_varying_known_reals` must be consistent across both encoder and decoder periods. For encoder steps (past), the PREDISPATCH run-T data doesn't exist for those past timestamps. Mapping around this requires rebuilding the pre-built array approach internally.
- Also: TF/Keras was Sinclair et al.'s framework for TFT; our custom PyTorch follows their PatchTST (best model) more closely.
- **Decision:** Custom PyTorch with pre-built arrays matches the paper's data pipeline pattern exactly.

### StandardScaler normalisation
- Initial draft used `StandardScaler`
- Problem: SA1 prices have extreme right-tail spikes. StandardScaler compresses the bulk of the distribution into a narrow range. The p50 price (~60 $/MWh) would be close to the mean; spikes at 10,000+ $/MWh would be 50+ standard deviations.
- **Solution:** `QuantileTransformer(n_quantiles=2000)` confirmed in Sinclair et al.

### output_length=56 (hard cutoff at 28h)
- Initial plan was to use 56 steps as the maximum PREDISPATCH coverage
- Problem: only 47.9% of runs have ≥56 future PREDISPATCH intervals, and those runs are systematically limited to UTC 03:00–14:00 (AEST 13:00–00:00). The model would never train on morning runs, causing severe time-of-day distribution mismatch at inference.
- **Solution:** Masked loss with output_length=144. All runs contribute to short-horizon steps regardless of their PREDISPATCH horizon length.

### LightGBM + PD7Day as intermediate step
- Considered as a quick improvement before TFT is ready
- Problem: PREDISPATCH and PD7Day are structurally biased upward. Amber APF is clean and debiased. Wiring biased forecasts into LightGBM without training for horizon-dependent correction would degrade performance.
- **Decision:** Amber APF stays in place until TFT is validated.

---

## Current Status and Next Steps

### Complete (as of 2026-04-12)
1. ✅ AEMO ingest: PREDISPATCH, PD7Day, SevenDayOutlook, P5MIN → InfluxDB (systemd timers)
2. ✅ Parquet ML cache: `data/export_parquet.py` — SA1/VIC1/NSW1 PD + 5m volatility agg + actuals
3. ✅ Run-aligned dataset builder: `data/build_training_dataset.py` — 18 enc / 13 dec features
4. ✅ Training script: `train/train_tft_price.py` — AdamW + ReduceLROnPlateau + horizon-weighted loss
5. ✅ Rolling-origin evaluation: `train/evaluate_tft.py` — nMAPE (all/base/spike) + quantile calibration
6. ✅ NEMSEER/NEMWeb backfill: `ingest/backfill_predispatch_nemseer.py` (2024-04 → 2025-02)
7. ✅ Stratified eval benchmark: `data/build_stratified_eval.py` — fixed set, durable across runs
8. ✅ 17,514 training samples; 431 stratified eval hold-out; VAL_DAYS=60

### Current training setup (`train/train_tft_price.py`)

- **Optimizer:** AdamW (lr=2e-4, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=2) — fires after 2 non-improving epochs
- **Loss:** Horizon-weighted masked quantile loss: `weight_h = exp(-h / tau)`, tau=14 steps (7h).
  Short-horizon steps dominate gradients; 4h weight=0.56, 16h weight=0.10, 28h weight=0.02.
- **Early stopping:** wMAPE (horizon-weighted nMAPE, consistent with training loss); fallback val_loss

### Run 006 evaluation (Stratified Eval Set — fixed benchmark, Apr 2026)

Numbers below are from Run 006. See Run 010 table for the corrected LightGBM comparison.

| Horizon | TFT nMAPE | LightGBM | Delta | TFT base (≤$150) | TFT spike (>$150) |
|---|---|---|---|---|---|
| 1h | 79.4% | 37.9% | +41.6% ⚠️ inflated | 34.9% | 84.7% |
| 2h | 77.5% | 40.7% | +36.8% ⚠️ inflated | 37.6% | 82.8% |
| 4h | 73.8% | 43.7% | +30.1% ⚠️ inflated | 40.2% | 79.1% |
| 28h | 74.5% | 52.9% | +21.6% ⚠️ inflated | 44.0% | 79.4% |

⚠️ LightGBM deltas above are inflated ~2× — `evaluate_tft.py` used a time-window filter rather
than exact `forecast_creation_time` matching against the TFT stratified set (fixed in dc4ea19).
Corrected Run 010 deltas: +15.3% (1h), +7.4% (4h), +3.6% (28h).

**Key finding:** Run 005's apparent 1h TFT win was an artefact of the Feb–Apr val window being
anomalously favourable. The stratified benchmark reveals TFT spike nMAPE is 79–84% — much worse
than LightGBM at spike-weighted samples. Baseload accuracy is competitive. The spike gap
is the primary problem to solve.

**Run 007:** Spike nMAPE barely moved (84.6%). 5m coverage only 48.8% — most stratified
spike events predate the 5m window. **Run 008:** Progressive price-weighted loss (log-growth
up to 6.8× at market cap) — spike nMAPE unchanged (84.1%); training imbalance hypothesis
falsified. **Run 009:** Full 5m coverage (~100% after NEMSEER backfill) — spike nMAPE 86.8%,
model converged in epoch 2, suggesting features add noise. All three mitigation hypotheses
After 10 runs all mitigation approaches exhausted. Spike nMAPE structurally stuck at ~84%.
Likely a fundamental mismatch: LightGBM uses PREDISPATCH RRP directly as a feature; TFT must
infer spike onset from encoder context. TFT's production value is in **long-horizon baseload
accuracy and calibrated quantile intervals**, not spike prediction.

Full run history and calibration results: **[docs/training_runs.md](training_runs.md)**

### Completed (Runs 001–010) ✅
9. ✅ VIC1/NSW1 decoder features (Run 006)
10. ✅ Stratified eval benchmark (spike gap confirmed structural)
11. ✅ 5-min volatility encoder features (Run 007 — marginal)
12. ✅ Progressive price-weighted loss (Run 008 — no spike improvement; calibration recovered)
13. ✅ NEMSEER 5m backfill → full coverage (Run 009 — no spike improvement)
14. ✅ Log-scaling + 4yr backfill + shadow mode (Run 010 — in production shadow mode)
15. ✅ Fix eval: LightGBM stratified comparison corrected (dc4ea19)
16. ✅ Fix inference: inverse log transform + unit mismatch + PD_RRP zeros (258db6a, f2d9127)

### V4 Architecture — Next Steps
See plan file for full sequencing. Summary:

1. **Phase 1 — PREDISPATCH debiaser** (trainable now): LightGBM on (PREDISPATCH forecast →
   actual settlement) pairs. Feed debiased `pd_rrp` into TFT decoder as Run 011.
   Also add: reserve margin feature (SevenDayOutlook), `aemo_divergence` encoder feature,
   expanded quantiles (q5/q10/q50/q90/q95/q99).

2. **Phase 2 — Tactical model** (needs P5MIN backfill): Backfill P5MIN forecasts via
   NEMSEER. Train both multi-output LightGBM and a TFT variant on the tactical tier.
   Compare on financial regret over a common evaluation window; promote the winner.
   Note: the theoretical case for LightGBM is strong but has not been empirically
   confirmed — empirical comparison is required before committing.

3. **Phase 3 — Dispatch simulator**: Offline LP backtester. Financial regret minus cycle
   degradation cost. Golden set of historical crisis events. CI/CD promotion gate.
   **Note:** Golden set events must be hard-excluded from training (not merely
   downweighted) — the gate measures generalisation, not memorisation.

4. **Phase 4 — Calibration**: Conditional conformal prediction stratified by reserve margin
   (spike regime) and residual demand (oversupply regime).

5. **Phase 5 — Production routing**: Tier 1 (0–60 min) + Tier 2 (1h–72h TFT).
   HA automations for tail risk overrides. EMHASS on q50. Amber APF removal.
   Investigate exposing EMHASS LP shadow price (SOC dual variable) via fork or upstream
   PR before implementing the tail-risk override trigger — this is the correct
   opportunity cost denominator. Finite-difference approximation is fallback only.

### Known Open Issues (V4)

- **Reserve margin demand bias:** SevenDayOutlook demand forecasts are biased low
  during heatwaves — exactly when reserve margin tightens. Mitigation: add a rolling
  actual-vs-forecast demand divergence term (analogous to `aemo_divergence`) to both
  the encoder and the debiaser training pipeline.

- **Debiaser endogeneity:** The PREDISPATCH debiaser trains on actual settlement prices
  that are partially endogenous to the PREDISPATCH forecast (generators respond to it).
  This is a fundamental constraint of operating in a strategic market, not fixable by
  design. Monitor debiaser residuals; retrain more aggressively during known structural
  transition periods (battery fleet scaling, new FCAS products, generator retirements).

- **NEM intervention pricing:** Actual prices may be revised up to 4 days post-dispatch.
  Training labels and the `aemo_divergence` feature may be based on provisional prices.
  Pipeline must handle corrections when revisions arrive.

- Dispatch-regret metric: simulate charge/hold/discharge vs perfect foresight

---

## File Reference

| File | Purpose |
|---|---|
| `data/export_parquet.py` | Export InfluxDB → Parquet. `--actuals-only` skips PREDISPATCH/PD7Day (preserves backfill) |
| `data/build_training_dataset.py` | Build run-aligned numpy arrays from Parquet |
| `data/parquet/aemo_predispatch_sa1.parquet` | PREDISPATCH: InfluxDB (2025-03+) merged with NEMSEER backfill (2024-04 to 2025-02) |
| `data/parquet/aemo_pd7day_sa1.parquet` | PD7Day (Feb 2026+) |
| `data/parquet/actuals_sa1.parquet` | Actuals (dispatch + local sensors, 2024-03-29+) |
| `data/parquet/X_encoder.npy` | Built training encoder arrays [N, 96, 14] |
| `data/parquet/X_decoder.npy` | Built training decoder arrays [N, 144, 11] |
| `data/parquet/y_targets.npy` | Normalised target RRP [N, 144] |
| `data/parquet/y_targets_raw.npy` | Raw target RRP in $/MWh [N, 144] |
| `data/parquet/y_mask.npy` | Valid-step mask [N, 144] bool |
| `data/parquet/scalers.pkl` | Fitted QuantileTransformer per feature |
| `data/parquet/dataset_meta.json` | Shape/coverage metadata |
| `train/train_tft_price.py` | Training script |
| `train/evaluate_tft.py` | Rolling-origin evaluation: TFT vs LightGBM nMAPE at 1h/2h/4h/8h/16h/28h + quantile calibration |
| `docs/training_runs.md` | Persistent log of all training runs, configs, and eval results |
| `models/tft_price/checkpoint_best.pt` | Best trained model |
| `models/tft_price/training_log.csv` | Epoch-level metrics |
| `models/tft_price/evaluation_results.csv` | nMAPE comparison table (written by evaluate_tft.py) |
| `ingest/ingest-predispatch.py` | PREDISPATCH ingestion (--fetch / --backfill-archive) |
| `ingest/ingest-pd7day.py` | PD7Day ingestion (--fetch / backfill) |
| `ingest/backfill_predispatch_nemseer.py` | Historical PREDISPATCH backfill via NEMSEER (pre-Aug 2024) + direct NEMWeb (Aug 2024+) |

---

## Rebuilding the Dataset

When new data has accumulated (e.g., more PD7Day runs, or after NEMSEER backfill):

```bash
# 1a. Refresh Parquet cache from InfluxDB (actuals only — preserves NEMSEER-backfilled PREDISPATCH)
python data/export_parquet.py --actuals-only

# 1b. Or full refresh (overwrites PREDISPATCH parquet from InfluxDB — requires re-running backfill after)
python data/export_parquet.py
python ingest/backfill_predispatch_nemseer.py --start 2024-04 --end 2025-02

# 2. Rebuild training arrays
python data/build_training_dataset.py

# 3. Retrain
python train/train_tft_price.py --epochs 100

# 4. Evaluate
python train/evaluate_tft.py
```

All steps are idempotent (overwrite outputs). Typical runtime: actuals export ~5 min, backfill ~2 min (cached), build ~3 min, train ~70 min on CPU (100 epochs), evaluate ~2 min.

**Important:** `data/export_parquet.py` (without `--actuals-only`) overwrites `aemo_predispatch_sa1.parquet` with InfluxDB data only (2025-03+), losing the NEMSEER backfill. Always use `--actuals-only` for routine refreshes. The backfill cache in `data/nemseer_cache/` is gitignored but safe to keep; re-running the backfill is fast (~2 min) because all ZIPs are cached locally.
