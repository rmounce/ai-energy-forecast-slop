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

## Live Shadow Read (2026-05-09)

The TFT load shadow is consistently lower than the production LightGBM forecast, but the
available offline accuracy check says that lower forecast is probably not merely a bug.

A live-log shape comparison over the last 14 days, matching forecast runs and target
timestamps between LightGBM and the TFT load shadow, shows the TFT is systematically lower:

| Horizon bucket | Mean TFT − LightGBM |
|---|---:|
| 0-24h | -79.7 W |
| 24-48h | -104.3 W |
| 48-72h | -131.6 W |
| Overall | -105.2 W |

Matched sample: `95,615` rows across `664` runs, from roughly
`2026-04-25T09:00:00Z` to `2026-05-09T08:30:00Z`. TFT was below LightGBM on
`83.1%` of matched rows.

This confirms the user's live observation that the TFT shadow sits below the current
LightGBM forecast. It does not by itself prove either model is more accurate.

The existing offline comparison script, `eval/compare_load_forecast.py`, compares TFT
Run 005 against LightGBM rows in the same validation calendar window
(`2026-01-13` to `2026-04-13`). On that check, TFT is materially better by MAE:

| Horizon bucket | TFT q50 MAE | LightGBM MAE | TFT improvement |
|---|---:|---:|---:|
| 0-24h | 235.3 W | 270.6 W | 35.3 W |
| 24-48h | 234.4 W | 310.5 W | 76.1 W |
| 48-72h | 232.4 W | 312.4 W | 80.0 W |
| Overall | 234.0 W | 297.8 W | 63.8 W |

TFT q10/q90 coverage was `0.758`, below the nominal `0.80`, so its uncertainty bands
still need calibration. This is also not yet a live shadow backtest because historical
`tft_load` log rows were not backfilled with actuals. As of 2026-05-09, future
`tft_load_forecast_log.csv` rows are included in `forecast.py backfill-actuals`, so a
proper live accuracy comparison can accumulate from here.

Operational read: do not promote TFT load blindly, but do not abandon it. The next load
decision should be based on live backfilled accuracy over a few weeks, plus a sanity check
that the lower load forecast does not make EMHASS under-prepare in edge cases.

Repeatable diagnostic:

```bash
nice -n 19 ./.venv/bin/python eval/analyze_live_load_shadow_gap.py --days 14
```

Future logging note: `forecast.py` now writes future `tft_load` rows to
`tft_load_forecast_log.csv`, and `forecast.py backfill-actuals` now backfills that log.
Historical `tft_load` rows before this change are mixed into `tft_price_forecast_log.csv`
and have no actuals.

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
