# TFT Load Forecast: Design and Run History

Shadow branch for household load prediction. Replaces the existing Darts/LightGBM load
model in `forecast.py`. Currently in shadow mode — publishes alongside (not instead of)
the legacy model.

---

## Architecture

- **Encoder:** 96 steps (48h lookback) — `power_load`, `power_pv`, temperature, humidity,
  wind speed, time features (6 sin/cos)
- **Decoder:** 144 steps (72h) — BOM weather forecasts, Solcast PV forecast, time features,
  holidays
- **Target:** `power_load` at each future step (W)
- **Quantiles:** q10/q50/q90 (symmetric uncertainty)
- **Loss:** `HorizonWeightedQuantileLoss` — exponential decay, currently tau=48 steps
- **Model size:** d_model=64, 4 heads, 2 LSTM layers, ~189k params

**Key difference from price TFT:** No PREDISPATCH equivalent. Decoder covariates are purely
weather + time — cleaner architecture. Target is positive-and-bounded so no log transform.

---

## Run History

| Run | Key change | Overall MAE | Overnight 48h bias | Notes |
|-----|-----------|-------------|---------------------|-------|
| 001 | Baseline | — | Not measured | Initial implementation |
| 002 | stride=4, tz fix | 227.5W | Not measured | Faster training, sampling fix |
| 003 | Vectorised scaling, Adelaide tz fix (inference) | 227.5W | −45W (q50 mean 344W vs actual 388W) | Same dataset as 002 |
| 004 | Temporal decay weights | 230.5W | −45W | No improvement — gradient cliff |
| 005 | horizon-decay tau=48 | 234.2W | −24W (q50 mean 364W vs actual 388W) | Best checkpoint (epoch 32) |

**Current production checkpoint:** `models/tft_load/checkpoint_best.pt` (Run 005, epoch 32)

---

## Known Issue: Overnight 48h Morning Ramp Inversion

**Symptom (observed live 2026-04-17):** Step 72 (6:30am day+2) = 265W q50, which is lower
than step 60 (3:30am day+2). The morning ramp (3–7am) is predicted to invert — physically
implausible. Overall overnight 48h mean is biased low (~364W vs actual ~388W).

**Root cause:** `HorizonWeightedQuantileLoss` with tau=48 gives step 72 only
`exp(-72/48) = 22%` gradient weight. At 36h+ horizon, the model's time-of-day encoding is
too weak to produce a monotone morning ramp. The gradient cliff localises the error
to specific steps even when the mean overnight bias appears to improve.

**Sanity check for any new run:** Does the model predict monotonically increasing load
during the 3–7am morning ramp for **all three** overnight windows (24h, 48h, 72h)?
This is a qualitative gate — quantitative MAE alone is insufficient.

---

## Run 006 Plan: Horizon Weight Floor (Option B)

**Recommended implementation:** `--horizon-floor 0.25` in `train/train_tft_load.py`.

Weight formula: `max(exp(-t/48), 0.25)` — all steps beyond ~32h retain at least 25% weight.

- Steps 0–32h: unaffected (same shape as Run 005)
- Steps 32–72h: floor kicks in; step 72 goes from 22% → 25% gradient
- No discontinuity in wMAE metric interpretation

**Expected outcomes:**
- Morning ramp inversion fixed (primary goal)
- Overnight 48h bias ≤ −15W (better than Run 005's −24W)
- Overall MAE ≤ 237W (allow small regression from Run 005's 234W)
- q10/q90 coverage ≥ 0.78

**Alternatives considered:**
- **Option A (tau=96):** More aggressive; step 72 goes to 47% but near-term MAE degrades
  further (runs 003→005 already cost +7W from this trade-off).
- **Option C (conformal post-correction):** Doesn't fix the structural signal weakness;
  use as a second layer after Option B, not standalone.

---

## Promotion Criteria

Before Load TFT replaces the Darts/LightGBM load model in production:

1. Morning ramp monotonicity: 3am < 5am < 7am for all three overnight windows (24h, 48h, 72h)
2. Overnight 48h bias ≤ −15W
3. Overall MAE ≤ 237W
4. q10/q90 coverage ≥ 0.78
5. Phase 8 regression test suite passing (holistic pipeline gate)

If coverage remains below 0.80 after Run 006: apply conformal calibration (Option C) as
a second layer before promotion.

---

## Integration Path

Phase 6 (holistic dispatch simulation) will baseline the current LightGBM load model.
Load TFT promotion is gated on Phase 6+8 completion — not on Run 006 training results
alone. The promotion decision should be validated against dispatch simulation profit, not
just MAE.
