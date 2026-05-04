# Training Run Log

Persistent record of training runs, config changes, and evaluation results.
**Purpose:** prevent regressions from being forgotten and identify which changes actually helped.

Val set covers the most recent N days of training data (before the gap).
All nMAPE buckets are **cumulative** (1-step through Nh), valid steps only.

⚠️ **Eval windows differ between runs** — nMAPE absolute values are not directly comparable
across runs. Use the Delta column (TFT vs LightGBM on the same window) as the primary signal.

---

## TFT Price Run 014 — 2026-04-20 — Enhanced Input TFT (parallel PREDISPATCH + PD7Day decoder)

**COMPLETE — INTERIM GATE FAILED.** First Phase 7 run with the expanded 18-feature decoder:
`pd_rrp` is now PREDISPATCH-only (0-filled beyond step 56), `pd7_rrp` carries the
latest PD7Day run across all 144 steps, `covar_missing` is replaced by
`predispatch_active`, and the decoder adds `pd7_generation_hour` + `pd7_available`.
This run intentionally kept the existing `pw_wMAPE` objective (`tau=14`) so the
architecture/input change could be evaluated in isolation against Run 011b.

### Config
| Parameter | Value |
|---|---|
| Target Scaling | Log-Scaling (scale=60.0) |
| Quantiles | [0.05, 0.10, 0.50, 0.90, 0.95, 0.99] |
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE, patience=7 |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | 54,404 samples (51,623 train, 2,211 val, 431 eval), 20 enc / **18 dec** features |
| Key change | Phase 7 decoder contract: parallel `pd_rrp` + `pd7_rrp`, `predispatch_active`, `pd7_generation_hour`, `pd7_available` |

### Training outcome
- Best epoch: **4** (early stop epoch 11, patience=7)
- Best pw_wMAPE: **42.04%**
- Best val_loss: **0.0555**
- nMAPE (all): **62.20%**
- nMAPE (4h): **41.35%**  |  16h: **45.69%**  |  28h: **45.71%**  |  72h: **71.83%**

### Interpretation

Run 014 did **not** repeat the catastrophic Run 012/013 failure mode. The 72h nMAPE stayed
in the ~68–72% range through training rather than diverging toward 100%, so the Phase 7
decoder expansion appears trainable. However, the run still peaked early (epoch 4) and did
not clearly solve the long-horizon objective issue; the existing `tau=14` horizon weighting
continues to de-emphasise the far horizon relative to the financial gate.

**Checkpoint status:** `models/tft_price/checkpoint_best.pt` now contains the 18-feature
decoder contract (`n_dec=18`, `meta.dec_features` updated accordingly). This means the
default local checkpoint path no longer matches the documented Run 011b incumbent. Treat
Run 011b as the evaluated production baseline, but do not assume `checkpoint_best.pt`
still points to it after 2026-04-20 training activity.

### Interim holistic eval (vs amber_apf_lgbm baseline)
| Stratum | Run 011b + binary routing | Run 014 enhanced input |
|---|---|---|
| All | **+9.7%** | **−35.3%** |
| Spike | **+7.2%** | **−37.1%** |
| Low | **+32.6%** | **−34.1%** |
| Normal | **+0.4%** | **−3.1%** |

*Results saved to `eval/results/holistic_eval_results_tft014_enhanced_input.csv` and
`eval/results/holistic_eval_raw_tft014_enhanced_input.parquet`. Canonical
`holistic_eval_results.csv` remains restored to the frozen Run 011b baseline.*

**Gate result:** FAILED. Phase 7 decoder expansion alone regressed badly on the interim holistic
gate, despite training stably. This does not look like the Run 012/013 catastrophic 72h-collapse
mode; instead it looks like a train/eval objective mismatch surviving the architecture change.
That makes the flat-wMAPE Run 015 ablation the immediate next step rather than optional cleanup.

### Next ablation (agreed with implementer)

**Run 015:** repeat Run 014 dataset/architecture with **flat wMAPE** (no horizon decay;
`tau=None` / `tau=∞`). Rationale: the financial/dispatch gate treats the full 72h vector as
consequential, so `tau=14` is structurally misaligned with the eval objective. Run 014 is kept
as the apples-to-apples Phase 7 comparison; Run 015 isolates the objective change.

### Commands and logs

- Dataset rebuild: `source .venv/bin/activate && PYTHONUNBUFFERED=1 nice -n 19 python data/build_training_dataset.py 2>&1 | tee /tmp/dataset_build_phase7.log`
- Training: `source .venv/bin/activate && PYTHONUNBUFFERED=1 nice -n 19 python train/train_tft_price.py 2>&1 | tee /tmp/tft_run014_phase7.log`
- Retro inference: `source .venv/bin/activate && PYTHONUNBUFFERED=1 nice -n 19 python eval/retro_tft_inference.py --overwrite 2>&1 | tee /tmp/retro_tft_run014.log`
- Holistic eval: `source .venv/bin/activate && PYTHONUNBUFFERED=1 nice -n 19 python eval/holistic_eval.py --hybrid-source --price-only --workers 12 2>&1 | tee /tmp/holistic_eval_run014.log`

---

## TFT Price Run 015 — 2026-04-20 — Flat wMAPE ablation (no horizon decay) ⚠️ checkpoint_best.pt points here — do NOT promote

**COMPLETE — FAILED.** Direct follow-up to Run 014. Keeps the Phase 7 decoder/input contract
unchanged (18-feature decoder with parallel PREDISPATCH + PD7Day) and changes only the
training objective: horizon decay disabled so every decoder step contributes equally to the
masked quantile loss.

### Motivation

Run 014 showed that the decoder expansion is trainable, but the interim financial gate still
failed badly. The agreed next ablation is to remove the `tau=14` horizon weighting because the
dispatch/EMHASS gate treats the full 72h vector as consequential. This run isolates objective
alignment without changing decoder inputs, the debiaser, or routing.

### Planned config delta vs Run 014
| Parameter | Value |
|---|---|
| Decoder contract | Same as Run 014 (18 features) |
| Loss | **Flat masked quantile loss** (`--horizon-decay 0`) |
| All other hyperparameters | Same as Run 014 unless noted otherwise |

### Command and log

- Training: `source .venv/bin/activate && PYTHONUNBUFFERED=1 nice -n 19 python train/train_tft_price.py --horizon-decay 0 2>&1 | tee /tmp/tft_run015_flat_wmape.log`

### Training outcome

- Best epoch: **2** (early stop epoch 9, patience=7)
- Best val loss: **0.0950**
- Best pw_wMAPE: **60.31%**
- wMAPE / nMAPE (all): **62.70%**
- nMAPE (4h): **44.90%**  |  16h: **46.10%**  |  28h: **45.79%**  |  72h: **72.57%**

### Initial interpretation

Flat weighting did **not** produce an obvious training-space win over Run 014. Best epoch
arrived even earlier (epoch 2 vs epoch 4), `72h` remained effectively unchanged, and the
shorter buckets were slightly worse than Run 014 at the best checkpoint. That said, the loss
definition changed enough that the training metrics alone are not the gate; the real question
is whether the resulting forecast vector performs better under dispatch simulation.

### Interim holistic eval (vs amber_apf_lgbm baseline)
| Stratum | Run 014 enhanced input | Run 015 flat wMAPE |
|---|---|---|
| All | **−35.3%** | **−65.9%** |
| Spike | **−37.1%** | **−71.0%** |
| Low | **−34.1%** | **−46.9%** |
| Normal | **−3.1%** | **−18.1%** |

*Results saved to `eval/results/holistic_eval_results_tft015_flat_wmape.csv` and
`eval/results/holistic_eval_raw_tft015_flat_wmape.parquet`. Canonical
`holistic_eval_results.csv` and `holistic_eval_raw.parquet` were restored afterward to the
frozen incumbent baseline snapshot.*

**Gate result:** FAILED, and materially worse than Run 014. Flat horizon weighting did not
repair the Phase 7 regression; it amplified it. As a result, flat wMAPE should be treated as
an explicitly rejected ablation for the current Phase 7 setup, not the new default.

**Important caveat:** this rejection is specific to the **current data regime and gate**.
PD7Day coverage in training is still sparse in the 28h–72h region, and the present holistic
eval penalises the full 72h forecast vector through one-shot dispatch simulation. If PD7Day
history becomes materially denser, or if the 30m/72h tier is later evaluated primarily as a
strategic `soc_final` guide for the downstream 5m/14h MPC tier rather than as a standalone
72h dispatch driver, flat weighting may still be worth revisiting.

### Artifact handling

- Preserved Run 014 snapshot before launch:
  `models/tft_price/checkpoint_run014_phase7_best.pt`
  `models/tft_price/scalers_run014_phase7.pkl`
- Preserved Run 015 snapshot after training:
  `models/tft_price/checkpoint_run015_flat_wmape_best.pt`
  `models/tft_price/scalers_run015_flat_wmape.pkl`

---

## TFT Price Run 012 — 2026-04-20 — Unified debiaser (prob_spike as decoder feature)

### Live Diagnostic Follow-Up — 2026-05-04

`forecast.py --debug-tft` now prints a first-30-step decoder diagnostic with raw
PREDISPATCH, debiased PREDISPATCH, compression ratio, PD7Day value, model input, TFT
q30/q50/q70, and Amber APF comparison.

Initial live diagnostic confirms a **double-compression** failure mode in the Run 011b-era
TFT stack during the inspected event:

| Stage | Mean price |
|---|---:|
| Raw PREDISPATCH | `2324 $/MWh` |
| After PREDISPATCH debiaser | `153 $/MWh` |
| TFT q50 output | `~87 $/MWh` |
| Amber APF | `~286 $/MWh` |

Observed mechanism:

- The encoder history was dominated by low solar-period prices, so the spike classifier produced
  low `prob_spike`.
- The PREDISPATCH debiaser compressed the active PREDISPATCH steps heavily (`mean_ratio ~= 0.16`,
  about `84%` compression).
- The TFT then discounted the already-compressed decoder signal further.
- Run 011b's 15-feature decoder contract has `covar_missing`, but that feature is effectively
  inactive when PREDISPATCH or PD7Day is available. It does not tell the model which decoder steps
  are genuine PREDISPATCH-backed steps.

Current architectural hypothesis for the next retrain: replace `covar_missing` with
`predispatch_active` in the 15-feature decoder contract, keeping feature count stable. The intent
is to make the PREDISPATCH-to-PD7Day quality transition explicit without repeating the broader
18-feature Run 014 expansion. Retrain the debiaser/TFT/scalers as a matched bundle and A/B against
Run 011b before changing any shadow or production forecast source.

Implementation prep:

- `data/build_training_dataset.py` now supports `--decoder-contract`.
- Default remains `phase7_18`, preserving the existing expanded Phase 7 builder.
- The controlled next-run contract is `run011b_active_15`:
  - 15 decoder features total
  - same continuous decoder width as Run 011b
  - last feature changed from `covar_missing` to `predispatch_active`
  - PD7Day tail values are folded into `pd_rrp` after the PREDISPATCH horizon, matching the old
    15-feature price-covariate shape
- Smoke checks passed:
  - `nice -n 19 ./.venv/bin/python -m py_compile data/build_training_dataset.py`
  - `nice -n 19 ./.venv/bin/python data/build_training_dataset.py --dry-run --decoder-contract phase7_18`
  - `nice -n 19 ./.venv/bin/python data/build_training_dataset.py --dry-run --decoder-contract run011b_active_15`

Recommended next long-run sequence:

```bash
./.venv/bin/python data/build_training_dataset.py --decoder-contract run011b_active_15
./.venv/bin/python train/train_tft_price.py --lr 1e-4 --patience 15
```

Before promoting any checkpoint, inspect `data/parquet/dataset_meta.json` or the checkpoint meta
and confirm:

- `decoder_contract == "run011b_active_15"`
- `n_dec_features == 15`
- `dec_features[-1] == "predispatch_active"`

**COMPLETE — FAILED.** Retrain on same dataset as Run 011b, with one structural change: the OOF
PREDISPATCH debiaser now takes `prob_spike` as an 11th feature (see PD Debiaser Run 002 below).
Spike-classified windows in the decoder training data now receive higher (less suppressed) `pd_rrp`
values, closing the train/inference distribution gap that existed in Run 011/011b.

**Motivation:** Run 011b used binary spike routing at inference only — debiased OOF for training,
raw PREDISPATCH bypass at inference for prob_spike > 0.65. The decoder never trained on high raw
pd_rrp inputs, causing out-of-distribution predictions during spike windows.

### Config
| Parameter | Value |
|---|---|
| Target Scaling | Log-Scaling (scale=60.0) |
| Quantiles | [0.05, 0.10, 0.50, 0.90, 0.95, 0.99] |
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE, patience=7 |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | 54,404 samples (51,623 train, 2,211 val, 431 eval) — same split as Run 011b |
| Key change | `debiased_pd_rrp_oof.parquet` regenerated with unified debiaser (Run 002) |

### Training outcome
- Best epoch: **2** (early stop epoch 9, patience=7)
- Best pw_wMAPE: **38.97%** (vs Run 011b: 42.38% — better)
- Best val_loss: **0.05145**
- nMAPE (4h): 40.75%  |  16h: 42.82%  |  28h: 43.63%  |  72h: 82.23%

**⚠ Scalers co-location bug discovered during eval (2026-04-20):** `train_tft_price.py` loads scalers
from `data/parquet/scalers.pkl` but saves the checkpoint to `models/tft_price/`. Inference reads
from `models/tft_price/scalers.pkl`. When the dataset was rebuilt with the new unified debiaser OOF,
`data/parquet/scalers.pkl` was updated but `models/tft_price/scalers.pkl` remained stale (Apr 18).
The `pd_rrp` scaler was fitted on the old OOF distribution, causing silent mis-scaling of spike-window
decoder inputs at inference — manifesting as All −28.3%, Normal −44.5% in the first eval attempt.
**Fix:** `train_tft_price.py` now copies scalers alongside every best-checkpoint save.

### Holistic eval (vs amber_apf_lgbm baseline) — three-way comparison
| Config | All | Spike | Low | Normal |
|---|---|---|---|---|
| Run 011b + binary routing (prior best) | +9.7% | +7.2% | +32.6% | +0.4% |
| Run 011b + unified debiaser (no retrain) | −9.9% | −16.2% | +31.9% | +4.2% |
| **Run 012 + unified debiaser** | **−28.3%** | **−29.9%** | **−9.6%** | **−44.5%** |

*Results in `eval/results/holistic_eval_results_tft012_unified_debiaser.csv`.*

**Key finding:** Run 012 is worse than the no-retrain case — the TFT retrain itself is the problem,
not just the debiaser change. The normal stratum collapse (−44.5%) indicates a fundamentally broken
model, not spike-routing confusion. Root cause: Run 012 converged at epoch 2 with training loss still
falling steeply (0.070→0.058 over 9 epochs) — the model barely trained on the new bimodal OOF
distribution. The new distribution (spike windows have high raw pd_rrp instead of suppressed OOF)
may require lower LR and more patience to fit well.

**Scaler bug note:** The catastrophic first eval attempt (All −28.3%) was attributed to a stale
`models/tft_price/scalers.pkl` but proved to be a red herring — `pd_rrp` uses a fixed log transform,
not a fitted scaler, so old and new were identical. Run 012 is genuinely this bad.

**Run 013 outcome:** Also failed — see Run 013 entry below. Unified debiaser direction abandoned.
**Prior best remains Run 011b + binary routing.**

---

## TFT Price Run 013 — 2026-04-20 — Unified debiaser, lower LR

**COMPLETE — FAILED.** Retry of Run 012 with `--lr 5e-5 --patience 15` to address early convergence.
Early stopping at epoch 21 (patience=15 exhausted). Best at epoch 6.

### Config
| Parameter | Value |
|---|---|
| All params | Same as Run 012 |
| Optimizer | AdamW **lr=5e-5** (vs 2e-4 in Run 012), weight_decay=1e-4 |
| Early stopping | pw_wMAPE, **patience=15** (vs 7 in Run 012) |

### Training outcome

| Epoch | 4h nMAPE | 16h nMAPE | 28h nMAPE | 72h nMAPE | Unweighted | pw_wMAPE | LR |
|---|---|---|---|---|---|---|---|
| 1 | 56.7% | 55.3% | 55.2% | 59.4% | 57.9% | 51.26% | 5e-5 |
| **6 (best)** | **40.6%** | **44.5%** | **44.7%** | **90.3%** | **73.5%** | **39.40%** | 5e-5 |
| 10 | 41.3% | 45.5% | 45.6% | 103.4% | 82.1% | 40.00% | 2.5e-5 |
| 21 (final) | 40.9% | 45.3% | 45.2% | 95.7% | 77.1% | 39.73% | 1.56e-6 |

- Best pw_wMAPE: **39.40%** at epoch 6 (slightly worse than Run 012's 38.97%)
- LR decayed 5× over 21 epochs (scheduler patience=2); model stalled after epoch 6
- 72h nMAPE peaked at 103% (epoch 10), partially recovered to 96% as LR decayed — never usable

### Diagnosis

Lower LR did not fix the problem. The same failure mode as Run 012 occurred: pw_wMAPE improves by
sacrificing long-horizon accuracy, because the training metric weights 28h+ steps at <2% each.
The 72h nMAPE diverging to ~100% while 4h improves to ~40% is the training objective working as
designed — it is not a hyperparameter problem.

**Root cause (confirmed):** `pw_wMAPE` with tau=14 steps is misaligned with the unified debiaser
architecture. The bimodal OOF distribution (spike windows ~5000 $/MWh, normal windows 0–500 $/MWh)
gives the TFT a hard learning problem at long horizons, but the training metric doesn't reward
solving it. No LR or patience change can address a misaligned objective.

**Direction change:** Unified debiaser + single TFT architecture abandoned. See
`docs/architecture_review_2026-04-20.md` for full assessment. The likely path forward is splitting
into two decoupled models (0–28h PREDISPATCH model for MPC; 28–72h PD7Day model for day-ahead
SoC planning) with separate, aligned loss functions.

**Active checkpoint remains Run 011b + binary routing (+9.7% holistic eval).**

---

## PD Debiaser Run 002 — 2026-04-20 — Unified debiaser (prob_spike feature)

**Replaces binary spike routing with smooth correction.** Added `prob_spike` (spike classifier
output per run_time) as an 11th input feature to the LightGBM debiaser. The model now learns
the appropriate correction level for each spike probability value rather than applying a hard
bypass at a tuned threshold (previously 0.65). Eliminates the magic number.

**Root cause addressed:** Prior architecture had a train/inference mismatch for spike windows —
OOF debiased uniformly during training, raw PREDISPATCH at inference when prob_spike > 0.65.
The unified debiaser is trained with `prob_spike` as a feature; both training and inference use
the same correction path.

### Config
| Parameter | Value |
|---|---|
| Base features (10) | pd_rrp, pd_demand, pd_net_interchange, horizon_steps, hour/dow/month sin/cos |
| New feature | prob_spike (from spike_clf_predictions.parquet, fill 0.0 if absent) |
| Objective | regression_l1 (MAE) — robust to spike outliers |
| n_estimators | 1000 (early stopping, patience=50) |
| num_leaves | 127, feature_fraction=0.8, bagging_fraction=0.8 |
| OOF folds | 5 (time-ordered, split by interval_dt) |
| Training data | aemo_predispatch_sa1.parquet + actuals_sa1.parquet (3,067,043 rows) |

### OOF validation metrics
| Category | N | Raw bias | Debiased bias | Raw MAE | Debiased MAE |
|---|---|---|---|---|---|
| Overall | 3,067,043 | −266.5 | +26.0 | 325.7 | **62.0** $/MWh |
| Horizon 1–6 (0–3h) | 331,563 | −123.6 | +25.5 | 177.0 | 58.7 |
| Horizon 7–16 (3.5–8h) | 552,512 | −177.8 | +25.9 | 231.7 | 61.3 |
| Horizon 17–32 (8.5–16h) | 882,544 | −244.1 | +26.6 | 307.3 | 63.1 |
| Horizon 33–56 (16.5–28h) | 952,135 | −349.9 | +27.8 | 417.0 | 65.5 |
| Regime: baseload | 2,180,675 | −70.0 | −7.1 | 94.9 | 28.0 |
| Regime: spike | 687,226 | −967.1 | **+153.1** | 1124.9 | 173.1 |
| Regime: oversupply | 199,142 | −0.7 | −49.9 | 95.3 | 51.0 |

**Key:** Spike regime debiased bias flipped from deeply negative (suppressed) to +153 $/MWh —
the debiaser now outputs *higher* values for spike-classified windows rather than compressing them.
Overall MAE improvement unchanged (325 → 62 $/MWh).

### Outputs
- `data/parquet/debiased_pd_rrp_oof.parquet` — regenerated (3,067,043 rows)
- `models/pd_debiaser/lgbm_final.pkl` — updated final model
- `models/pd_debiaser/metrics.json` — updated

---

## Load TFT Run 003 — 2026-04-17 — Adelaide timezone fix (inference only)

**Same dataset as Run 002.** Fixes `time_sin_cos()` at inference to use
`Australia/Adelaide` instead of Brisbane, matching `build_load_dataset.py` training.
Training result is near-identical to Run 002 (different random init).

**Config:** `train/train_tft_load.py --epochs 100 --batch-size 512`
**Dataset:** 18,769 samples (stride=4 / every 2h, 4.3y), 13 enc + 13 dec features, 90d val split
**Model:** d_model=64, 4 heads, 2 LSTM layers, 189k params, q10/q50/q90

### Results (best epoch 6 / early stop at 13)
| Metric | Value |
|---|---|
| wMAE (horizon-weighted) | **226.1 W** |
| MAE 0–24h | 226.2 W |
| MAE 24–48h | 226.4 W |
| MAE 48–72h | 229.8 W |
| Val loss | 0.1221 |

### Formal comparison vs LightGBM (`eval/compare_load_forecast.py`)
Val window: 2026-01-13 → 2026-04-13. TFT: 1,081 offline val samples. LightGBM: 4,164 production runs.

| Bucket | TFT q50 MAE | LightGBM MAE | Δ |
|---|---|---|---|
| 0–24h | **226.2 W** | 271.5 W | −45.4 W |
| 24–48h | **226.4 W** | 312.9 W | −86.5 W |
| 48–72h | **229.8 W** | 316.4 W | −86.6 W |
| Overall | **227.5 W** | 300.1 W | −72.7 W |

TFT q10/q90 coverage: 0.802 overall (0-24h: 0.802, 24-48h: 0.806, 48-72h: 0.799) — well-calibrated at target ~0.80.

**Promotion gate: PASSED on offline val metrics** — but live HA shadow predictions are suspect (see below). Hold promotion until inference bug is resolved.

### Notes
- LightGBM MAE degrades sharply beyond 24h (271W → 316W) due to static lag features.
- TFT MAE is flat across all horizons (~226–230W) — attention captures longer-range patterns.
- Results saved to `eval/results/load_forecast_comparison.json`.
- **⚠️ Overnight bias identified post-eval**: horizon-weight tau=24 (12h half-life) gives 48h steps only 1.8% gradient. Model's 48h overnight prior drifts below training Q10. See Run 004/005 for diagnosis and fix.

---

## Load TFT Run 005 — 2026-04-17 — horizon-tau=48 (overnight bias fix)

**Same dataset as Run 004.** Fixes the overnight 48h prediction bias identified after Run 003.
Root cause: `--horizon-decay 24` (12h half-life) gave 48h steps only 1.8% gradient, causing
the model to extrapolate below its training Q10 for overnight periods. tau=48 (24h half-life)
raises 48h gradient to 14%, halving the overnight bias.

**Config:** `train/train_tft_load.py --epochs 100 --batch-size 512 --horizon-decay 48`
**Dataset:** 18,769 samples (stride=4, tau=365d decay weights), 90d val split
**Model:** d_model=64, 4 heads, 2 LSTM layers, 189k params, q10/q50/q90
**Note:** `--horizon-decay 48` and `--patience 4` are now the defaults in `train_tft_load.py`.

### Results (best epoch 32 / early stop at 39)
| Metric | Value |
|---|---|
| wMAE (tau=48 weighted, not comparable to Run 003) | **234.2 W** |
| MAE 0–24h (raw, comparable) | 235.3 W |
| MAE 24–48h (raw, comparable) | 234.4 W |
| Val loss | 0.1227 |

### Overnight-stratified evaluation (`eval/eval_load_overnight.py`)
Val window: 2026-01-13 → 2026-04-13. Overnight = hours 3–6am Adelaide time.

| Horizon window | Actual mean | q50 mean | Bias | MAE | q10/q90 coverage |
|---|---|---|---|---|---|
| +24h overnight | 388 W | 364 W | **−24 W** | 69 W | 0.761 |
| +48h overnight | 388 W | 364 W | **−24 W** | 69 W | 0.772 |
| +72h overnight | 388 W | 364 W | **−24 W** | 69 W | 0.758 |
| All overnight | 388 W | 364 W | **−24 W** | 69 W | 0.765 |

Run 003/004 baseline (tau=24): bias ≈ −45W, coverage ≈ 0.795.

### Formal comparison vs LightGBM (`eval/compare_load_forecast.py`)
Val window: 2026-01-13 → 2026-04-13. TFT: 1,081 offline val samples. LightGBM: 4,164 production runs.

| Bucket | TFT q50 MAE | LightGBM MAE | Δ |
|---|---|---|---|
| 0–24h | **235.3 W** | 271.5 W | −36.2 W |
| 24–48h | **234.4 W** | 312.9 W | −78.6 W |
| 48–72h | **232.4 W** | 316.4 W | −84.0 W |
| Overall | **234.0 W** | 300.1 W | −66.1 W |

TFT q10/q90 coverage: 0.758 overall (0-24h: 0.743, 24-48h: 0.759, 48-72h: 0.772) — below 0.80 target.

### Notes
- Overnight bias halved (−45W → −24W) and non-monotonic pattern across horizons eliminated.
- Coverage dropped 0.795 → 0.758 (below 0.80 target) — gradient now spread more evenly, interval width less precisely calibrated. Acceptable for shadow mode; may need conformal correction before promotion.
- Near-term raw MAE cost: +9W vs Run 003 (235W vs 226W) — expected trade-off for spreading gradient across all horizons.
- **Live observation (2026-04-17):** morning ramp still inverted at 36h — step 72 (6:30am day+2) = 265W q50, step 120 (6:30am day+3) = 262W q50. Confirmed as residual gradient cliff (step 72 gets 22% weight), not a data or code bug. Fix: Run 006 with weight floor.
- **Promotion gate: NOT YET** — overnight coverage below 0.80, morning ramp inversion at 36h+. Re-evaluate after Run 006.

---

## Load TFT Run 004 — 2026-04-17 — Temporal decay weighting (tau=365d)

**Same dataset rebuild as Run 003.** Adds `exp(-age_days/365)` sample weights via
`WeightedRandomSampler`. Upweights recent 2025–2026 samples where overnight load is ~350W.

**Config:** `train/train_tft_load.py --epochs 100 --batch-size 512`
**Dataset:** 18,769 samples (stride=4, tau=365d decay weights added), 90d val split
**Horizon decay:** tau=24 (unchanged — this is why it didn't fix the overnight bias)

### Results (best epoch 10 / early stop at 17)
| Metric | Value |
|---|---|
| wMAE | **229.9 W** |
| MAE 0–24h | 231.3 W |
| MAE 24–48h | 229.8 W |
| Val loss | 0.1213 |

### Notes
- Slightly worse than Run 003 on all offline val metrics (+3W overall MAE).
- Overnight 48h q50 unchanged (237→236W): temporal decay alone cannot fix a 1.8% gradient horizon.
- Conclusion: tau must be increased (not sample weighting) to fix 48h predictions.

---

## Load TFT Run 002 — 2026-04-17 — Stride=4 subsampling + vectorised scaling

**Same architecture as Run 001.** Rebuilds dataset with stride=4 (every 2h) reducing
75k → 19k samples. Epoch time drops from ~158s to ~40s. Also vectorises the per-sample
scaling loop in `build_load_dataset.py`.

**Config:** `train/train_tft_load.py --epochs 100 --batch-size 512`
**Dataset:** 18,769 samples (stride=4), 13 enc + 13 dec features, 90d val split

### Results (best epoch 7 / early stop at 14)
| Metric | Value |
|---|---|
| wMAE | **227.2 W** |
| MAE 0–24h | 227.9 W |
| MAE 24–48h | 224.8 W |
| MAE 48–72h | 226.2 W |
| Val loss | 0.1204 |

### Notes
- Val loss improved significantly vs Run 001 (0.1385 → 0.1204): stride-1 was overfitting to autocorrelated windows.
- wMAE marginally worse (+2.5W) — noise from sparser val sample-to-date mapping.
- Epoch time 4× faster; no quality regression.

---

## Load TFT Run 001 — 2026-04-17 — Initial TFT load model

**Shadow implementation.** Replaces Darts/LightGBM load model's manual lag engineering
with TFT attention. 72h forecast at 30-min resolution. Trained with horizon-weighted
quantile loss (tau=24 steps / 12h) so short-horizon accuracy (0–24h) dominates.

**Config:** `train/train_tft_load.py --epochs 100 --batch-size 512 --horizon-decay 24`
**Dataset:** 75,073 samples (continuous 30-min stride, 4.3y), 13 enc + 13 dec features, 90d val split
**Model:** d_model=64, 4 heads, 2 LSTM layers, 189k params, q10/q50/q90
**Shadow entities:** `sensor.ai_tft_load_forecast` / `_low` / `_high`

### Results (best epoch 7 / early stop at 14)
| Metric | Value |
|---|---|
| wMAE (horizon-weighted) | **224.7 W** |
| MAE 0–24h | 226.2 W |
| MAE 24–48h | 228.7 W |
| MAE 48–72h | 226.7 W |
| Val loss | 0.1385 |

### Notes
- Training time ~158s/epoch (75k samples × stride 1 — highly autocorrelated).
- Epoch-to-epoch wMAE improvement was small after epoch 4; model converges quickly.
- Run 002 planned: stride=4 subsampling (~19k samples) + vectorised scaling for faster iteration.
- No comparison vs LightGBM baseline yet — see `eval/compare_load_forecast.py` (planned).

---

## Conformal Calibration Run 001 — 2026-04-16 — Phase 4 conditional conformal for Tier 1

**Per-regime additive corrections for q05 and q95.** Calibrated on val set (17,175 runs),
validated on stratified eval. Regime detection: spike if `p5min_rrp_h{h} ≥ $300` OR
`actual_rrp_t1 ≥ $300`; low if `p5min_rrp_h{h} < $0` OR `residual_demand_t1 < 0`.

### δ corrections (saved to `models/lgbm_tactical/conformal_deltas.json`)
| Regime | n (val steps) | δ_q95 | δ_q05 |
|---|---|---|---|
| spike | 703 | +$800.0 | −$73.3 (adds to q05) |
| low | 45,318 | −$0.6 (negligible) | +$23.5 |
| normal | 112,322 | +$1.1 | −$4.1 (adds to q05) |

*At inference: `adjusted_q95 = raw_q95 + δ_q95[regime]`; `adjusted_q05 = raw_q05 − δ_q05[regime]`*

### Coverage results
| Set | Regime | q05 before | q05 after | q95 before | q95 after |
|---|---|---|---|---|---|
| Val (calibration) | all | 0.949 ✓ | 0.950 ✓ | 0.948 ✓ | 0.950 ✓ |
| Stratified eval | spike | 0.997 ❌ | 0.781 ❌ | 0.750 ❌ | **0.821** ⚠ |
| Stratified eval | low | 0.911 ❌ | 0.938 ✓ | 0.974 ❌ | 0.965 ✓ |
| Stratified eval | normal | 0.959 ✓ | 0.893 ❌ | 0.935 ✓ | 0.937 ✓ |
| Stratified eval | all | 0.950 ✓ | 0.890 ❌ | 0.920 ❌ | 0.928 ❌ |

### Notes
- **If spike regime were perfectly known at inference:** spike q95 0.750 → 0.970 ✓ (oracle regime table).
  The gap between 0.970 and 0.821 is entirely due to imperfect regime detection.
- **Spike regime recall = 76%** (p5min + rrp_t1 detector). 24% of spikes are undetectable
  sudden events — fundamental limit. Those steps receive normal correction (+$1) instead of +$800.
- **Spike q05 overcorrects** (0.997 → 0.781): correction computed from only 703 val spike steps;
  non-critical for dispatch (nobody charges during a spike).
- **Calibration framework is production-ready** for Tier 1 inference. Apply δ per-horizon using
  `p5min_rrp_h{h}` and `actual_rrp_t1` for regime detection.

---

## LightGBM Tactical Run 001 — 2026-04-16 — Tier 1 baseline (3-quantile, long-format)

**First Tier 1 training run.** Multi-output LightGBM (q5/q50/q95) for 0–60 min SA1 price
correction. Long-format construction: X [191k runs × 12 horizons = 2.2M rows, 25 features].
Horizon appended as scalar feature; models learn horizon-specific AEMO bias patterns.

### Config
| Parameter | Value |
|---|---|
| Architecture | LightGBM quantile regression (3 separate models: q5, q50, q95) |
| Input format | Long format: 2,214,949 train rows / 158,343 val rows |
| Features (25) | 12× p5min_rrp, aemo_divergence_t1, actual_rrp_t{1,2,6}, rolling_{1h_std,3h_max}, residual_demand, hour/dow sin/cos, is_imputed_p5min, horizon |
| n_estimators | 2000 (early stopping) |
| learning_rate | 0.05 |
| num_leaves | 63 |
| min_child_samples | 50 |
| subsample / colsample | 0.8 / 0.8 |
| early_stopping_rounds | 50 |

### Training outcome
| Quantile | Best iter | Val pinball |
|---|---|---|
| q05 | 305 | 3.15 |
| q50 | 312 | 10.54 |
| q95 | 269 | 5.39 |

### Calibration — val set (last 60 days, 17,175 runs)
| Quantile | Expected | Actual | Bias | Status |
|---|---|---|---|---|
| q05 | 0.050 | 0.051 | +0.001 | ✓ |
| q50 | 0.500 | 0.505 | +0.005 | ✓ |
| q95 | 0.950 | 0.948 | -0.002 | ✓ |

**q50 MAE: 21.1 $/MWh vs P5MIN baseline 30.7 $/MWh → 31.4% reduction on val set.**

### Calibration — stratified eval (1,600 runs: 500 spike, 300 low, 800 normal)
| Quantile | Expected | Actual | Bias | Status |
|---|---|---|---|---|
| q05 | 0.050 | 0.050 | -0.000 | ✓ |
| q50 | 0.500 | 0.465 | -0.035 | ⚠ |
| q95 | 0.950 | 0.920 | -0.030 | ⚠ |

**q50 MAE: 110.0 $/MWh vs P5MIN baseline 187.7 $/MWh → 41.4% reduction on stratified set.**

#### By stratum
| Stratum | Runs | Mean max RRP | q50 MAE | Baseline MAE | Improvement | q05 coverage | q95 coverage |
|---|---|---|---|---|---|---|---|
| SPIKE | 543 | 1,386 $/MWh | 284.5 | 491.8 | **42.2%** | 0.034 ✓ | 0.857 ❌ |
| LOW | 591 | 2 $/MWh | 16.9 | 27.3 | **38.1%** | 0.074 ✓ | 0.954 ✓ |
| NORMAL | 466 | 142 $/MWh | 19.8 | 28.5 | **30.4%** | 0.037 ✓ | 0.950 ✓ |

### Notes
- q95 under-covers on spike stratum (0.857 vs 0.950). Same structural issue as TFT — log-scaling
  compresses spike region; Phase 4 conditional conformal calibration is the fix.
- q50 median is systematically under-predicting on spikes: corrects well but not far enough.
  Expected — quantile regression with symmetric loss doesn't fully capture right-tail skew.
- Models: `models/lgbm_tactical/lgbm_q{05,50,95}.pkl`

---

## Dispatch Sim Run 001 — 2026-04-16 — Phase 3 LP backtester baseline

**First financial regret evaluation.** Rolling MPC LP dispatch simulator over the 1,600-sample
stratified eval set. Three strategies compared: oracle (perfect foresight), P5MIN baseline,
LightGBM q50. Revenue is booked against actual prices; decisions are made using each strategy's
forecast at each 5-min step. Initial SoC fixed at 50% for all runs (5 kWh).

### Battery parameters
| Parameter | Value |
|---|---|
| Capacity | 40 kWh |
| Max charge/discharge | 10 kW |
| Charge efficiency | 0.95 |
| Discharge efficiency | 0.95 |
| Cycle degradation cost | $0.05/kWh throughput |
| Initial SoC | 20.0 kWh (50%) |

### Financial regret — stratified eval (1,600 runs)
| Stratum | n | Oracle rev | P5MIN rev | LightGBM rev | P5MIN regret | LightGBM regret | Regret reduction |
|---|---|---|---|---|---|---|---|
| SPIKE | 543 | $5.175 | $4.946 | $4.942 | $0.229 (4.4%) | $0.233 (4.5%) | −1.8% ⚠ |
| LOW | 591 | $0.080 | $0.026 | $0.044 | $0.054 (67%) | $0.036 (46%) | **+32.3%** |
| NORMAL | 466 | $0.608 | $0.585 | $0.589 | $0.023 (3.8%) | $0.019 (3.1%) | **+17.5%** |
| ALL | 1,600 | $1.963 | $1.859 | $1.865 | $0.104 (5.3%) | $0.098 (5.0%) | **+5.9%** |

*Revenue = mean per-run revenue over 12 × 5-min intervals ($)*

### Notes
- **Low stratum dominates the improvement** (+32.3% regret reduction): LightGBM's
  `aemo_divergence_t1` and rolling features detect oversupply regimes earlier than raw P5MIN,
  avoiding charging at prices that will go negative.
- **Spike stratum: essentially neutral** (−1.8%, within noise). During spike events, both
  strategies correctly predict "discharge now" — marginal q50 differences don't change the
  binary charge/discharge decision. Spike *capture* is a tail-risk problem (q95/HA override),
  not a q50 dispatch problem.
- **Spike regret vs oracle is low** ($0.114/run, 4.4%) because MPC oracle also can't see
  beyond the 12-step window; spike is typically one or two intervals of extreme price.
- Full results: `eval/results/dispatch_sim_run001.json`

---

## TFT vs LightGBM Dispatch Comparison — 2026-04-16 — Phase 3 architecture validation

**Compares TFT (30-min step-hold to 5-min) vs LightGBM q50 on 130 overlapping runs** (tactical
stratified eval runs at 30-min boundaries that exist in TFT's run_times). TFT step 0 q50 held
constant for intervals 0–5; step 1 q50 for intervals 6–11.

### Results (130 runs: 36 spike, 56 low, 38 normal)
| Stratum | n | Oracle | P5MIN | LightGBM | TFT | LightGBM regret | TFT regret | LGB vs TFT |
|---|---|---|---|---|---|---|---|---|
| SPIKE | 36 | $2.619 | $2.513 (4.0%) | $2.613 (0.2%) | $2.613 (0.2%) | 0.2% | 0.2% | ~equal |
| LOW | 56 | $0.039 | −$0.019 (150%) | −$0.009 (123%) | −$0.010 (126%) | 123% | 126% | LGB +$0.001 |
| NORMAL | 38 | $0.501 | $0.453 (9.5%) | $0.468 (6.6%) | $0.465 (7.1%) | 6.6% | 7.1% | LGB +$0.003 |
| ALL | 130 | $0.888 | $0.820 (7.7%) | $0.856 (3.6%) | $0.855 (3.7%) | 3.6% | 3.7% | LGB +$0.001 |

### Conclusion
**LightGBM and TFT are functionally equivalent for 0–60 min dispatch.** The $0.001/run revenue
difference is statistical noise on 130 runs. Both massively outperform P5MIN (3.6–3.7% vs 7.7%
regret). The architecture rationale for LightGBM as Tier 1 is **practical, not accuracy-based**:
- Native 5-min cadence (TFT is 30-min, requires step-hold approximation)
- ~10ms inference (TFT requires loading 144-step decoder + PREDISPATCH features)
- No PREDISPATCH pipeline dependency at runtime

Full results: `eval/results/tft_dispatch_comparison.json`

---

## Run 011 — 2026-04-16 — Debiased decoder + SDO features + q5/10/50/90/95/99

**Three Phase 1b improvements applied simultaneously.**
- OOF-debiased PREDISPATCH RRP replaces raw `pd_rrp` at decoder steps 0–55.
- SevenDayOutlook `sd_demand` + `sd_net_interchange` added as decoder features for all 144 steps.
- Quantiles expanded from q30/50/70 → q5/10/50/90/95/99.

### Changes from Run 010
- `build_training_dataset.py`: loads `debiased_pd_rrp_oof.parquet`, substitutes OOF-debiased
  RRP at decoder steps 0–55 (raw fallback if run absent). Adds SDO features; DEC_FEATURES 13→15.
- `train_tft_price.py`: `QUANTILES = [0.05, 0.10, 0.50, 0.90, 0.95, 0.99]`.
- `evaluate_tft.py`: p50 index derived from checkpoint quantile list (backwards-compatible).

### Config
| Parameter | Value |
|---|---|
| Target Scaling | Log-Scaling (scale=60.0) |
| Quantiles | [0.05, 0.10, 0.50, 0.90, 0.95, 0.99] |
| Optimizer | AdamW lr=2e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE (price+horizon weighted) |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | 54,404 samples (51,623 train, 2,211 val, 431 eval) |

### Training outcome
- Best epoch: **4** (early stop epoch 11, patience=7)
- Best val loss: **0.0513** (not comparable to Run 010 — 3→6 quantiles changes loss scale)
- pw_wMAPE: **39.99%**
- nMAPE (4h): 39.71%  |  16h: 45.67%  |  28h: 46.84%  |  72h: 74.55%
- **Note:** Early stopping at epoch 4 due to aggressive LR schedule (patience=2 halved LR twice
  by epoch 10). Training loss still decreasing at stop — clear under-convergence.
  Retrained as **Run 011b** with `--lr 1e-4 --patience 15`.

### evaluate_tft.py results (TFT vs LightGBM, Stratified Set)

| Horizon | TFT all | LGBM all | Delta | TFT base | LGBM base | TFT spike | LGBM spike |
|---|---|---|---|---|---|---|---|
| 1h | 77.5% | 63.9% | +13.6% | 33.8% | 30.6% | 82.6% | 71.9% |
| 2h | 75.6% | 68.3% | +7.3% | 35.4% | 35.6% | 80.9% | 77.4% |
| 4h | 71.7% | 66.4% | +5.3% | 37.8% | 46.2% | 77.0% | 74.8% |
| 8h | 69.3% | 64.4% | +4.9% | 39.9% | 46.7% | 74.4% | 73.6% |
| 16h | 71.6% | 65.1% | +6.6% | 42.9% | 46.4% | 76.4% | 74.8% |
| 28h | 73.0% | 71.0% | +1.9% | 44.7% | 58.2% | 77.5% | 77.2% |

**nMAPE 1–2pp better than Run 010 on the stratified set.** LightGBM still leads on spikes
at short horizons (1h/2h); TFT wins on baseload at 4h+.

### Quantile calibration (all valid steps, Stratified Set)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q05 | 0.050 | 0.153 | +0.103 | ❌ far over-covers |
| q10 | 0.100 | 0.200 | +0.100 | ❌ far over-covers |
| q50 | 0.500 | 0.541 | +0.041 | ↑ mild |
| q90 | 0.900 | 0.895 | -0.005 | ✓ |
| q95 | 0.950 | 0.941 | -0.009 | ✓ |
| q99 | 0.990 | 0.976 | -0.014 | ✓ |

**Upper tail (q90/q95/q99) well-calibrated.** Lower tail (q05/q10) badly miscalibrated —
the model's lower-quantile predictions are too high (conservative), meaning it underestimates
how often prices fall below q05/q10. Initially attributed to under-convergence (epoch 4).
**Confirmed structural after Run 011b** — more training did not fix it. Root cause: log-scaling
compresses the low-price region; Phase 4 conformal calibration required for q05/q10.

### Notes
- SDO covariate shift: 75% of training samples have SDO=0 (pre-2025-03); val set is 100%
  within SDO coverage (last 60 days). May contribute to mild val loss divergence.
- Retrained as Run 011b (`--lr 1e-4 --patience 15 --epochs 150`) — lower tail bias unchanged.

---

## Run 011b — 2026-04-16 — Lower LR + higher patience (convergence re-test)

**Lower tail calibration confirmed structural — more training does not fix it.**
Retrain of Run 011 with `--lr 1e-4 --patience 15` to test whether epoch-4 early stopping
caused the q05/q10 miscalibration. Best checkpoint still epoch 5; nMAPE marginally worse
than Run 011. q05 bias: +0.103→+0.096 (negligible); q10: +0.100→+0.108 (slightly worse).
Upper tail (q90/q95/q99) remains well-calibrated. **Conclusion: lower tail bias is structural,
not a convergence artefact. Requires Phase 4 conformal calibration layer.**

### Config
| Parameter | Value |
|---|---|
| Target Scaling | Log-Scaling (scale=60.0) |
| Quantiles | [0.05, 0.10, 0.50, 0.90, 0.95, 0.99] |
| Optimizer | AdamW lr=1e-4, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau factor=0.5, patience=2 |
| Early stopping | pw_wMAPE, patience=15 |
| d_model / heads / layers | 64 / 4 / 2 |
| Dataset | Same as Run 011 (54,404 samples) |

### Training outcome
- Best epoch: **5** (early stop epoch 20, patience=15 exhausted)
- Best val loss: **0.0538**
- pw_wMAPE: **42.38%**
- nMAPE (4h): 43.04%  |  16h: 48.96%  |  28h: 49.77%  |  72h: 67.28%
- LR decayed to 3.13e-6 by early stop — scheduler (patience=2) still aggressive relative
  to early-stopping patience; val loss diverging from epoch 6 onward.

### evaluate_tft.py results (TFT vs LightGBM, Stratified Set)
| Horizon | TFT all | LGBM all | Delta | TFT base | LGBM base | TFT spike | LGBM spike |
|---|---|---|---|---|---|---|---|
| 1h | 79.1% | 63.9% | +15.2% | 36.2% | 30.6% | 84.1% | 71.9% |
| 2h | 77.2% | 68.3% | +8.9% | 39.2% | 35.6% | 82.1% | 77.4% |
| 4h | 73.1% | 66.4% | +6.8% | 40.6% | 46.2% | 78.3% | 74.8% |
| 8h | 70.9% | 64.4% | +6.5% | 42.5% | 46.7% | 75.8% | 73.6% |
| 16h | 72.9% | 65.1% | +7.9% | 44.4% | 46.4% | 77.6% | 74.8% |
| 28h | 74.2% | 71.0% | +3.2% | 45.5% | 58.2% | 78.8% | 77.2% |

Marginally worse than Run 011 at short horizons (1h: 79.1% vs 77.5%) — same best epoch, same
convergence pattern. Not a meaningful regression.

### Quantile calibration (all valid steps, Stratified Set)
| Quantile | Expected | Actual coverage | Bias | Status |
|---|---|---|---|---|
| q05 | 0.050 | 0.146 | +0.096 | ❌ over-covers (vs Run 011: +0.103) |
| q10 | 0.100 | 0.208 | +0.108 | ❌ over-covers (vs Run 011: +0.100) |
| q50 | 0.500 | 0.543 | +0.043 | ↑ mild over-covers |
| q90 | 0.900 | 0.909 | +0.009 | ✓ |
| q95 | 0.950 | 0.943 | -0.007 | ✓ |
| q99 | 0.990 | 0.974 | -0.016 | ✓ |

### Notes
- **Key finding:** Lower tail bias did not improve with 3× more training budget. Convergence
  was not the cause. Log-scaling compresses the low-price region; the model cannot represent
  q05/q10 tails accurately in that space. Phase 4 conformal calibration (stratified by regime)
  is the correct and only fix.
- Upper tail excellent — q90/q95/q99 all within ±0.02. These are safe for dispatch use.
- q50 mild over-coverage (+0.043) is acceptable for EMHASS median input; conformal layer will
  tighten this too.
- **Active production checkpoint (promoted 2026-05-04).** Loaded via `models/tft_price/checkpoint_active.pt`.
  Upper tail quantiles (q90/q95/q99) reliable. Lower tail (q05/q10) should not be used for
  dispatch thresholds until Phase 4 calibration is applied.

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
