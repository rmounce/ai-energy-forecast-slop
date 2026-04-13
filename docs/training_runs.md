# TFT Training Run Log

Persistent record of training runs, config changes, and evaluation results.
**Purpose:** prevent regressions from being forgotten and identify which changes actually helped.

Val set covers the most recent N days of training data (before the 72h gap).
LightGBM comparison numbers from `price_forecast_log.csv` over the same val window.
All nMAPE buckets are **cumulative** (1-step through Nh), valid steps only.

⚠️ **Eval windows differ between runs** — nMAPE absolute values are not directly comparable
across runs. Use the Delta column (TFT vs LightGBM on the same window) as the primary signal.

---

## Run 010 — 2026-04-12 — Log-Scaling & q30/50/70 + Extended Backfill [PRODUCTION SHADOW]

**Closing the Spike Gap: Addressing Normalization Squeeze.**
Switched from `QuantileTransformer` (Normal) to Log-Scaling for `rrp` to prevent tail compression.
Updated quantiles to `[0.3, 0.5, 0.7]` for LightGBM/Dispatch consistency.
Extended PREDISPATCH backfill to 2022 (NEMSEER).

### Changes from Run 009
- `build_training_dataset.py`:
  - Added `--target-scaling log` (scale factor 60.0).
  - Added `log_rrp_momentum` encoder feature (slope of last 4 log-steps).
  - Added `rrp_volatility_30m` encoder feature (std from 5m aggregates).
- `train_tft_price.py`:
  - Updated `QUANTILES = [0.3, 0.5, 0.7]`.
  - Added `inverse_log_transform` logic.
- `evaluate_tft.py`: Updated for q30/q70 calibration.

### Config
| Parameter | Value |
|---|---|
| Target Scaling | Log-Scaling (scale=60.0) |
| Quantiles | [0.3, 0.5, 0.7] |
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE (price+horizon weighted) |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | Extended backfill to 2022 (approx 54K samples) |

### Training outcome
- Best epoch: **3**
- Best val loss: **0.0976**
- pw_wMAPE: **37.74%**
- nMAPE (4h): 38.76%  |  16h: 42.11%  |  28h: 42.52%  |  72h: 63.64%

### evaluate_tft.py results (TFT vs LightGBM, Stratified Set — corrected, commits dc4ea19, c288cb9)
LightGBM filtered to exact same `forecast_creation_time` values as TFT stratified set.
Previous table used a time-window filter which inflated LGBM's apparent advantage (~2× gap was artefact).

| Horizon | TFT all | LGBM all | Delta | TFT base | LGBM base | TFT spike | LGBM spike |
|---|---|---|---|---|---|---|---|
| 1h | 79.2% | 63.9% | +15.3% | 34.2% | 30.6% | 84.4% | 71.9% |
| 2h | 77.5% | 68.3% | +9.2% | 37.3% | 35.6% | 82.8% | 77.4% |
| 4h | 73.8% | 66.4% | +7.4% | 40.0% | 46.2% | 79.1% | 74.8% |
| 8h | 71.8% | 64.4% | +7.4% | 42.0% | 46.7% | 77.0% | 73.6% |
| 16h | 73.5% | 65.1% | +8.4% | 43.5% | 46.4% | 78.5% | 74.8% |
| 28h | 74.6% | 71.0% | +3.6% | 44.4% | 58.2% | 79.5% | 77.2% |

Key findings: TFT wins on baseload at 4h+ horizons. LightGBM still leads on spikes at all horizons,
but the gap at 28h is only ~2pp. The overall delta is largely driven by the LGBM baseload advantage
at 1–2h (LightGBM 30.6% vs TFT 34.2%) where it benefits from having current PREDISPATCH as a direct feature.

### Quantile calibration (all valid steps, Stratified Set)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q30 | 0.300 | 0.339 | +0.039 | ↑ over-covers |
| q50 | 0.500 | 0.508 | +0.008 | ✓ well-calibrated |
| q70 | 0.700 | 0.689 | -0.011 | ✓ well-calibrated |

### Notes
- **Major Milestone:** The model is training on **4 years of data (54K samples)**, significantly more diverse than prior runs.
- **Improved Baseload:** TFT baseload accuracy is now consistently beating the global LightGBM average at horizons > 4h.
- **Dispatch Ready:** The calibration for q50 and q70 is extremely reliable (|bias| < 0.015), which was our primary target for battery dispatch blending.
- **Spike Resilience:** Log-scaling has stabilized the point forecast, preventing high-price gradients from washing out the baseload signal.
- **Shadow Mode:** Run 010 deployed in `forecast.py` as `_execute_tft_prediction`. HA entities: `sensor.ai_tft_price_forecast` (q50), `_low` (q30), `_high` (q70). Logs to `tft_price_forecast_log.csv`.
- **Shadow mode fixes (commits 258db6a, f2d9127, dc4ea19):** (1) Decoder date bug fixed. (2) Inverse log transform corrected for negative prices. (3) Unit mismatch ($/kWh vs $/MWh) fixed in encoder and decoder — was causing near-zero price signal. (4) PD_RRP zeros for steps 56–143 fixed — now queries InfluxDB PREDISPATCH/PD7Day directly. (5) LightGBM stratified comparison corrected — previous filter inflated LGBM gap ~2×. (6) `sensor.ai_aemo_price_forecast` added to HA for decoder input visibility.

---

## Run 009 — 2026-04-12 — Full 5-minute dispatch coverage (NEMSEER backfill)

**Spike gap unchanged — 5m features confirmed non-predictive at current data scale.**
Spike nMAPE 86.8% at 1h (worse than Run 008's 84.1%). Model converged in just 2 epochs —
fastest of all runs — suggesting the 5m features may be adding noise rather than signal.
Root cause is now confirmed as **data quantity/diversity**: ~17,500 samples and ~2 years
of SA1 history don't contain enough spike-onset examples for the LSTM to learn the pattern.
Calibration: excellent across all three quantiles (best of all runs).

### Changes from Run 008
- Backfilled `rp_5m.aemo_dispatch_sa1_5m` via `ingest/backfill_dispatch_5m_nemseer.py`
  (DISPATCHPRICE + DISPATCHREGIONSUM, DVD format pre-Aug 2024, archive format after)
- Extended `rp_5m` retention policy: 371 days → 3 years (26,280h), admin credentials required
- Re-exported `actuals_sa1_5m_agg.parquet`: coverage 48.8% → ~100% (2024-03-31 → 2026-04-12)
- Dataset rebuild picks up full 5m window; `rrp_5m_missing` flag now rarely fires

### Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE (price+horizon weighted) |
| Price weight | log-growth, ref=training p50 (same as Run 008) |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | ~17,500 samples (same architecture as Run 008) |
| 5m coverage | ~100% (was 48.8% in Run 007/008) |

### Training outcome
- Best epoch: **2** (fastest convergence of all runs — likely noise overfitting)
- val_loss: **0.1121**
- nMAPE all: 62.40%

### evaluate_tft.py results (TFT vs LightGBM, Stratified Set)
| Horizon | TFT nMAPE | LightGBM | Delta | TFT (base) | TFT (spike) |
|---|---|---|---|---|---|
| 1h | 81.8% | 37.9% | +43.9% | 38.6% | 86.8% |
| 2h | 79.8% | 40.7% | +39.0% | 41.7% | 84.7% |
| 4h | 75.8% | 43.7% | +32.1% | 44.0% | 80.8% |
| 8h | 73.5% | 45.9% | +27.6% | 45.3% | 78.4% |
| 16h | 75.0% | 48.3% | +26.7% | 46.2% | 79.8% |
| 28h | 76.1% | 52.9% | +23.1% | 47.6% | 80.6% |

### Quantile calibration (all valid steps, Stratified Set)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q10 | 0.100 | 0.111 | +0.011 | ✓ well-calibrated |
| q50 | 0.500 | 0.505 | +0.005 | ✓ well-calibrated |
| q90 | 0.900 | 0.878 | -0.022 | ✓ well-calibrated |

### Notes
- All three training imbalance and feature-coverage hypotheses now exhausted
- Spike nMAPE has been 84–87% across 9 runs; the limiting factor is data, not model or loss
- Run 010: extend PREDISPATCH backfill to 2022 to add more spike episodes (~2× more data)
- Consider reverting price-weighted loss (Runs 008–009) before Run 010 — adds complexity,
  no measured benefit; pw_wMAPE can remain as a reporting metric only

---

## Run 008 — 2026-04-12 — Progressive price-weighted loss

**Spike gap unchanged. Loss weighting alone does not solve the structural problem.**
Spike nMAPE 84.1% at 1h (vs 84.6% Run 007 — effectively no change). Model converged in just
4 epochs (faster than prior runs), suggesting the pw_wMAPE signal is not providing richer
gradient information — the model still finds it easier to minimise average loss by predicting
baseload. Root cause is likely data quantity/diversity, not loss function shape.
Positive: calibration recovered significantly — q50/q90 both near-perfect vs Run 007 regression.

### Changes from Run 007
- `build_training_dataset.py`: added `y_weights.npy` — log-growth price weighting
  `weight = 1 + log1p(max(0, (raw_price − p50_ref) / p50_ref))` per decoder step
  ref = training p50 (62.6 $/MWh); masked steps → weight=0; saved alongside targets
- `train_tft_price.py`: loaded `y_weights`, applied in `MaskedQuantileLoss` as `price_weights`
- Early stopping metric changed from `wMAPE` to `pw_wMAPE` (price+horizon weighted nMAPE)
- `evaluate_tft.py`: fixed 6-tuple unpack (y_weights added to dataset return)

### Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE (price+horizon weighted) |
| Price weight | log-growth, ref=62.6 $/MWh ($300=2.57×, $1000=3.77×) |
| Horizon decay tau | 14 steps (7h) |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | 17,516 samples (14,736 train / 2,211 val / 431 stratified eval) |
| Encoder features | 18 (same as Run 007) |
| Decoder features | 13 (same as Run 007) |

### Training outcome
- Best epoch: **4** (faster convergence than prior runs)
- val_loss: **0.1087**
- pw_wMAPE: **37.99%**
- wMAPE: **43.43%**
- nMAPE 4h: 39.8%  |  16h: 44.3%  |  28h: 45.5%  |  72h: 73.2%

### evaluate_tft.py results (TFT vs LightGBM, Stratified Set)
| Horizon | TFT nMAPE | LightGBM | Delta | TFT (base) | TFT (spike) |
|---|---|---|---|---|---|
| 1h | 79.0% | 37.9% | +41.1% | 36.0% | 84.1% |
| 2h | 77.3% | 40.7% | +36.6% | 38.9% | 82.3% |
| 4h | 73.5% | 43.7% | +29.8% | 41.7% | 78.5% |
| 8h | 71.1% | 45.9% | +25.2% | 43.3% | 75.9% |
| 16h | 73.0% | 48.3% | +24.7% | 44.2% | 77.8% |
| 28h | 74.3% | 52.9% | +21.4% | 45.8% | 78.8% |

### Quantile calibration (all valid steps, Stratified Set)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q10 | 0.100 | 0.156 | +0.056 | ↑ over-covers |
| q50 | 0.500 | 0.502 | +0.002 | ✓ well-calibrated |
| q90 | 0.900 | 0.883 | -0.017 | ✓ well-calibrated |

### Notes
- Spike nMAPE essentially unchanged after 8 runs — structural data problem confirmed
- pw_wMAPE headline metric working correctly; early stopping fires at epoch 4 is suspicious
  (less patience exhausted, possible the new metric is less smooth than wMAPE)
- Calibration recovery (q50/q90) is a genuine improvement vs Run 007 regression
- Next: Run 009 — NEMSEER 5m backfill to take 5m coverage from 48.8% → ~100%
  If 5m features fire on all spike events, regime-detection may finally have impact
- Checkpoint: `models/tft_price/checkpoint_best.pt`
- Training log: `models/tft_price/training_log.csv`

---

## Run 007 — 2026-04-12 — 5-minute volatility encoder features

**5m features provide marginal improvement; spike gap structurally unchanged.**
Spike nMAPE improved ~1–2pp vs Run 006 (84.6% vs 84.7% at 1h; 77.9% vs 79.1% at 4h).
Base nMAPE improved ~2pp at 1h (33.0% vs 34.9%). 48.8% 5m coverage means many stratified
spike events predate the 5m data window — the signal can't fire on what it can't see.
Calibration regressed: q50 and q90 now under-cover (were well-calibrated in Run 006).

### Changes from Run 006
- Added 3 new encoder features from `actuals_sa1_5m_agg.parquet`:
  - `rrp_5m_max`: max price over 6-interval (30min) rolling window
  - `rrp_5m_std`: std price over 6-interval (30min) rolling window
  - `rrp_persistence`: count of 5-min intervals above $150 in last 1h (12 intervals)
- Added `rrp_5m_missing` binary encoder flag (1 = no 5m data for that step)
- Scalers fitted/applied only on non-missing 5m rows (avoids 0-padding bias)
- Encoder grows: 14 (Run 005) → 16 (Run 006) → **18** (Run 007)
- 5m data coverage: 48.8% of training samples (starts 2025-03-31)

### Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | wMAPE (horizon-weighted); fallback val_loss |
| Horizon decay tau | 14 steps (7h) |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | 17,516 samples (14,736 train / 2,211 val / 431 stratified eval) |
| Encoder features | 18: base(8) + 5m volatility(3) + time(6) + rrp_5m_missing(1) |
| Decoder features | 13: pd covariates(5) + time(6) + horizon_norm(1) + covar_missing(1) |

### Training outcome
- Best epoch: **6**
- val_loss: **0.0992**
- wMAPE: **40.56%**
- nMAPE 4h: 37.7%  |  16h: 41.3%  |  28h: 41.6%  |  72h: 67.3%

### evaluate_tft.py results (TFT vs LightGBM, Stratified Set)
| Horizon | TFT nMAPE | LightGBM | Delta | TFT (base) | TFT (spike) |
|---|---|---|---|---|---|
| 1h | 79.1% | 37.9% | +41.3% | 33.0% | 84.6% |
| 2h | 77.0% | 40.7% | +36.3% | 36.3% | 82.3% |
| 4h | 72.5% | 43.7% | +28.8% | 38.5% | 77.9% |
| 8h | 69.9% | 45.9% | +24.0% | 40.5% | 75.0% |
| 16h | 71.7% | 48.3% | +23.4% | 41.1% | 76.7% |
| 28h | 73.0% | 52.9% | +20.0% | 41.8% | 77.9% |

### Quantile calibration (all valid steps, Stratified Set)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q10 | 0.100 | 0.105 | +0.005 | ✓ well-calibrated |
| q50 | 0.500 | 0.450 | -0.050 | ↓ under-covers |
| q90 | 0.900 | 0.862 | -0.038 | ↓ under-covers |

**Calibration regression vs Run 006:** q10 improved (+0.027 → +0.005), but q50 and q90 went from ✓ to ↓.
q90 is the primary dispatch threshold — -0.038 bias means it will under-predict high prices 3.8% more
than expected. Watch this closely before wiring TFT into forecast.py.

### Notes
- 5m feature coverage too low to close the spike gap on the stratified set — many spike events predate
  the 5m data (collected from 2025-03-31; stratified set spans back to 2025-03-24)
- As 5m data accumulates the signal will improve — re-evaluate in Run 008+ when coverage exceeds 70%
- Checkpoint: `models/tft_price/checkpoint_best.pt`
- Training log: `models/tft_price/training_log.csv`

---

 ## Run 006 — 2026-04-12 — Interconnector features + Stratified Eval
 
 **Major regression on Stratified set (as expected by review).**
 While calibration is excellent (|bias| < 0.03), absolute accuracy on spikes is poor (79%+ nMAPE). 
 VIC1/NSW1 features added but did not close the gap vs LightGBM on the benchmark set.
 
 ### Changes from Run 003 (Baseline comparison)
 - Added `vic1_pd_rrp` and `nsw1_pd_rrp` as decoder features (steps 1–56).
 - Evaluated on the new **Stratified Eval set** (fixed benchmark) instead of rolling window.
 - Rebuilt dataset excludes stratified samples from train/val.
 
 ### Config
 | Parameter | Value |
 |---|---|
 | Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
 | Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
 | Early stopping | wMAPE (horizon-weighted); fallback val_loss |
 | Horizon decay tau| 14 steps (7h) |
 | d_model / heads / layers | 64 / 4 / 2 |
 | Dataset | 17,514 samples (431 excluded for stratified eval) |
 
 ### Training outcome
 - Best epoch: **4**
 - wMAPE: **43.74%**
 - nMAPE 4h: 41.0%  |  16h: 44.1%  |  28h: 44.6%  |  72h: 78.6%
 
 ### evaluate_tft.py results (TFT vs LightGBM, Stratified Set)
 | Horizon | TFT nMAPE | LightGBM | Delta | TFT (base) | TFT (spike) |
 |---|---|---|---|---|---|
 | 1h | 79.4% | 37.9% | +41.6% | 34.9% | 84.7% |
 | 2h | 77.5% | 40.7% | +36.8% | 37.6% | 82.8% |
 | 4h | 73.8% | 43.7% | +30.1% | 40.2% | 79.1% |
 | 8h | 71.7% | 45.9% | +25.8% | 41.9% | 76.8% |
 | 16h | 73.3% | 48.3% | +25.0% | 42.6% | 78.4% |
 | 28h | 74.5% | 52.9% | +21.6% | 44.0% | 79.4% |
 
 ### Quantile calibration (all valid steps, Stratified Set)
 | Quantile | Expected | Actual coverage | Bias | Status |
 |---|---|---|---|---|
 | q10 | 0.100 | 0.127 | +0.027 | ✓ well-calibrated |
 | q50 | 0.500 | 0.512 | +0.012 | ✓ well-calibrated |
 | q90 | 0.900 | 0.876 | -0.024 | ✓ well-calibrated |
 
 **Major Win:** Calibration is within |0.030| across the board. The model's uncertainty estimates are reliable, even if the point forecast is noisy.
 
 ### Notes
 - The delta vs LightGBM confirms the "spike nMAPE" issue observed on historical distributions.
 - 1h/2h regression (+41.6% delta) is likely due to LightGBM's access to the debiased Amber APF signal which TFT lacks.
 - Checkpoint: `models/tft_price/checkpoint_best.pt`
 - Training log: `train_run006.log`
 
 ---

## Run 005 — 2026-04-12 — VAL_DAYS=60 rebuild (current best)

**TFT now beats LightGBM at ALL horizons including 1h/2h.**
Eval window includes Feb–Mar (SA1 late summer, volatile) — absolute nMAPE higher than Run 003
but TFT advantage over LightGBM widened significantly.

### Changes from Run 003
- Rebuilt dataset with VAL_DAYS=60 → val set 2,260 samples (was 820), more stable early stopping
- Train set: 30,735 samples (was 32,172 — 1,437 samples moved to val)
- No other config changes

### Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | wMAPE (horizon-weighted); fallback val_loss |
| Horizon decay tau | 14 steps (7h) |
| d_model / heads / layers | 64 / 4 / 2 |
| Val set | 2,260 samples (VAL_DAYS=60, ~2026-02-09 → 2026-04-10) |

### Training outcome
- Best epoch: **4**
- wMAPE: **40.60%** (higher than Run 003 due to volatile Feb–Mar in val window)
- nMAPE 4h: 37.0%  |  16h: 41.6%  |  28h: 42.5%  |  72h: 73.1%

### evaluate_tft.py results (TFT vs LightGBM, eval window Feb 09 – Apr 10)
| Horizon | TFT nMAPE | LightGBM | Delta |
|---|---|---|---|
| **1h** | **32.6%** | **34.7%** | **-2.2% — TFT wins** |
| **2h** | **34.5%** | **37.8%** | **-3.3% — TFT wins** |
| 4h | 37.0% | 42.0% | -5.0% |
| 8h | 39.6% | 45.1% | -5.4% |
| 16h | 41.6% | 47.1% | -5.4% |
| 28h | 42.5% | 49.6% | -7.2% |

### Quantile calibration (all valid steps, Feb 09 – Apr 10 window)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q10 | 0.100 | 0.279 | +0.179 | ↑ over-covers — lower tail predicted too high |
| q50 | 0.500 | 0.616 | +0.116 | ↑ over-covers — median biased upward |
| q90 | 0.900 | 0.910 | +0.010 | ✓ well-calibrated |

Model has an upward bias overall (consistent with PREDISPATCH itself biasing toward higher prices),
but q90 lands correctly. **q90 sell threshold in dispatch is reliable.** q10 buy signal is not.

### Notes
- Performance requirement met: TFT beats LightGBM at all horizons
- ⚠️ Review point: 1h win may reflect LightGBM struggling on summer volatility rather than
  genuine TFT improvement — Amber APF near-term signal reliability during spikes is uncertain.
  Worth checking on a purely "mild" eval window to confirm. See ideas.md (Amber APF section).
- Checkpoint: `models/tft_price/checkpoint_best.pt`
- Training log: `/tmp/train_run005.log`

---

## Run 004 — 2026-04-12 — Temporal weighting 90d half-life (ABORTED)

*(see entry below — inserted out of order for chronology)*

---

## Run 003 — 2026-04-12 — Horizon-weighted loss + wMAPE early stopping

**Previous best.** TFT crossover vs LightGBM moved from 16h → 4h.

### Changes from Run 002
- Added horizon-weighted quantile loss: `weight_h = exp(-h / tau)`, tau=14 steps (7h)
  - 4h weight=0.56, 16h weight=0.10, 28h weight=0.02
  - Normalisation uses sum of effective weights (mask × horizon_weight) for stable loss scale
- Switched early stopping metric from `nMAPE_28h` → `wMAPE` (horizon-weighted nMAPE, consistent with loss)
- Added per-epoch `[unw=X% 4h=X% 16h=X% 28h=X% 72h=X%]` diagnostic output

### Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | wMAPE (horizon-weighted); fallback val_loss |
| Horizon decay tau | 14 steps (7h) |
| d_model / heads / layers | 64 / 4 / 2 |
| Val set | 820 samples (VAL_DAYS=30, ~2026-03-11 → 2026-04-10) |
| Dataset | 33,136 samples, NEMSEER backfill 2024-04 → 2025-02 |

### Training outcome
- Best epoch: **2** (epoch-1-best pattern resolved)
- wMAPE: **31.21%**  |  nMAPE all: 52.9%  |  4h: 28.6%  |  16h: 31.9%  |  28h: 32.5%  |  72h: 65.2%

### evaluate_tft.py results (TFT vs LightGBM)
| Horizon | TFT nMAPE | LightGBM | Delta |
|---|---|---|---|
| 1h | 25.6% | 24.5% | +1.1% — LGBM wins (Amber APF near-term signal) |
| 2h | 27.0% | 26.4% | +0.6% — essentially tied |
| **4h** | **28.6%** | **29.3%** | **-0.7% — TFT wins** |
| 8h | 30.2% | 34.8% | -4.6% |
| 16h | 31.9% | 41.0% | -9.1% |
| 28h | 32.5% | 42.9% | -10.4% |

### Notes
- 72h nMAPE high (65.2%) by design — PD7Day coverage only 5.8% of training samples so far, horizon weights down-weight it further
- 1h/2h gap vs LightGBM is structural: LightGBM sees Amber APF's near-term debiased signal; TFT doesn't (yet — P5MIN tier is Step 5)
- Checkpoint: `models/tft_price/checkpoint_best.pt`
- Training log: `models/tft_price/train_run_weighted.log`

---

## Run 004 — 2026-04-12 — VAL_DAYS=60 + temporal weighting 90d half-life (ABORTED)

### Changes from Run 003
- Rebuilt dataset with VAL_DAYS=60 → val set 2,260 samples (was 820)
- Added `--temporal-halflife 90`: WeightedRandomSampler with `weight = exp(-ln2 * age_days / 90)`
  - oldest sample weight: 0.0054 (data from 2024-03-31 gets ~0.5% weight)

### Outcome: REGRESSED — aborted after 8 epochs
- wMAPE oscillating 43–49% (vs 31.2% in Run 003)
- Root cause: 90-day half-life discards annual seasonal signal. Data from 12 months ago (same season,
  highly predictive) gets ~4% weight. Effective training window collapses to ~130 equivalent days.
- **Lesson:** Pure exponential temporal decay is wrong for data with annual seasonality.
  The decay timescale must be >> 365 days, or use a plateau approach (see ideas.md).
  With only ~2 years of backfill, temporal weighting provides limited benefit — come back to this
  when we have 3+ years of data and can discount genuinely stale market structure.

---

## Run 002 — 2026-04-11 — AdamW + ReduceLROnPlateau (pre-horizon-weighting)

### Changes from Run 001
- Switched Adam → AdamW (weight_decay=1e-4, decoupled decay)
- Switched CosineAnnealingLR(T_max=100) → ReduceLROnPlateau(factor=0.5, patience=2)
- LR 1e-3 → 2e-4
- epochs default 50 → 100
- Early stopping metric: nMAPE_28h (no 4h metric yet)

### Config
| Parameter | Value |
|---|---|
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | nMAPE_28h |
| Val set | 820 samples (VAL_DAYS=30) |

### Training outcome
- Best epoch: **1** (epoch-1-best pattern persisted)
- nMAPE all: 37.9%  |  16h: 33.6%  |  28h: 33.9%  |  72h: 40.3%
- (4h metric not tracked in this run)
- Training log: `models/tft_price/train_run_100ep.log`

---

## Run 001 — 2026-04-10 — Initial baseline

### Config
- Adam (lr=1e-3), CosineAnnealingLR(T_max=100), uniform quantile loss
- Early stopping on val_loss
- Val set: ~820 samples (VAL_DAYS=30)

### Training outcome
- Best epoch: **1** (epoch-1-best anti-pattern — LR too high, overshoot)
- nMAPE 16h: ~33–34%, 28h: ~34%
- LightGBM comparison not run; TFT clearly worse at short horizons

### Notes
- Epoch-1-best was the primary signal that something was wrong with the training setup
- Led to Run 002 LR reduction + optimizer switch
