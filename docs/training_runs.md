# TFT Training Run Log

Persistent record of training runs, config changes, and evaluation results.
**Purpose:** prevent regressions from being forgotten and identify which changes actually helped.

Val set covers the most recent N days of training data (before the 72h gap).
LightGBM comparison numbers from `price_forecast_log.csv` over the same val window.
All nMAPE buckets are **cumulative** (1-step through Nh), valid steps only.

⚠️ **Eval windows differ between runs** — nMAPE absolute values are not directly comparable
across runs. Use the Delta column (TFT vs LightGBM on the same window) as the primary signal.

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
