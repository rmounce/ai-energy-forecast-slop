# Alignment-Fix Retrain Side-By-Side — 2026-05-11

Per the implementer's recommendation in `docs/timestamp_convention_audit_2026-05-11.md`,
ran a "small strategic/PD-direct side-by-side eval before any substantial retrain"
to size the impact of the forecast/actuals interval-end/interval-start mis-alignment
on a trained model.

Picked PD7Day debiaser as the smallest meaningful retrain target (~3 months of data,
fast to retrain, existing canonical artefact at `models/pd7day_debiaser/`).

## Method

Added `--actuals-shift-min` flag to `train/train_pd7day_debiaser.py`. When set to
`30`, shifts `actuals.interval_dt` by +30 min before the inner merge with PD7Day
forecasts, so each PD7Day forecast row (interval-end labelled) joins against the
actual price for the **same** half-hour (rather than the half-hour starting at the
forecast's interval-end timestamp, which is the next half-hour).

Trained both variants with identical hyperparameters and the same 4-fold time-based
CV split. Outputs:

- canonical: `models/pd7day_debiaser/` (unchanged)
- aligned: `models/pd7day_debiaser_aligned30/` (variant, parallel file structure)
- OOF series: `data/parquet/debiased_pd7day_oof{,_aligned30}.parquet`

Raw data is unchanged. Existing canonical model is unchanged. The variant exists
side-by-side for comparison.

## Result

4-fold CV mean MAE per slice ($/MWh):

| Slice | n | Raw MAE | Cap-300 baseline | Canonical model | Aligned model | Δ aligned-canonical |
|---|---:|---:|---:|---:|---:|---:|
| overall | 51,012 | 543.18 → 539.41 | 74.96 → 74.29 | **36.98** | **36.14** | **−2.3%** |
| forecast ≥ 150 | 13,495 | 1939.46 → 1911.65 | 169.57 → 167.96 | 46.08 | 45.05 | −2.2% |
| forecast ≥ 300 | 8,594 | 2978.50 → 2935.41 | 199.27 → 197.59 | 45.92 | 45.15 | −1.7% |
| forecast ≥ 980 | 5,211 | 4682.56 → 4617.10 | 195.41 → 193.84 | 48.32 | 47.56 | −1.6% |
| actual ≥ 150 | 2,693 | 2521.10 → 2513.27 | 104.66 → 104.76 | 111.02 | 109.61 | −1.3% |
| actual ≥ 300 | 74 | 3575.27 → 3922.93 | 159.06 → 171.10 | 326.66 | 331.32 | +1.4% (small sample) |

(Two-column-per-cell entries show canonical → aligned values for context.)

## Reading

- The alignment fix produces a **consistent ~2% model MAE improvement** across
  overall + every forecast-band slice. One small-sample slice (`actual ≥ 300`,
  n=74) shows a 1.4% regression — within the noise band for a 74-sample slice.
- Even the cap-300 deterministic baseline drops by ~1% (74.96 → 74.29) — the
  alignment fix is helping ALL methods, not just the trained model.
- Raw PD7Day MAE drops only ~0.7%. PD7Day values are highly autocorrelated across
  adjacent half-hours, so the per-row impact of one-bin mis-alignment is modest.
  The model can extract more signal from cleanly-aligned pairs (~2% MAE drop) than
  raw pointwise comparison suggests (~0.7% MAE drop).
- This is consistent with the implementer's audit finding: PREDISPATCH at the
  shortest horizon (h≤30m) showed an 8% raw MAE improvement with the alignment
  shift. PD7Day's longer horizon and higher per-step volatility means the
  per-row gap matters less in raw terms.

## Implications for the substantive retrain

1. **PD7Day debiaser**: gain is modest (~2%) but consistent. Worth applying when
   the model is next retrained on its own schedule. Not a fire-drill.
2. **PREDISPATCH debiaser** (`models/pd_debiaser/lgbm_final.pkl`): not retrained
   here. Audit shows alignment matters more at short horizons, so the expected
   gain for the PREDISPATCH debiaser is plausibly larger than 2% but unknown
   without a side-by-side. Same `train/train_pd_debiaser.py` has the same merge
   pattern — apply the same `--actuals-shift-min` flag pattern.
3. **TFT shadow** (already on 2026-06-05 sunset clock): the structural critique
   documented in `docs/tft_price_forecast.md` says TFT compresses peaks and
   regresses toward the encoder median regardless of training-target alignment.
   Alignment fix unlikely to rescue TFT; deferred.
4. **Residual band table** (`models/pd_residual/residual_bands.parquet`): built
   by `eval/build_pd_residual_bands.py`. Likely uses the same forecast/actuals
   merge pattern. Same flag pattern should be added; expected gain is comparable
   to or larger than the debiaser's because residual bands are essentially
   measuring forecast-vs-actual error distributions.
5. **`build_training_dataset.py`** (TFT training pipeline): the line-428 target
   lookup is the analogous bug. The TFT retrain branch from this audit needs
   the same fix.

## Recommended next moves

In priority order:

1. **Mirror the `--actuals-shift-min=30` flag in `train/train_pd_debiaser.py`**
   and produce a side-by-side comparison for the PREDISPATCH debiaser. Expected
   to be a larger improvement than PD7Day (shorter horizon, more
   alignment-sensitive). Same code change pattern; ~30 min work + retrain.
2. **Rebuild the residual band table with the alignment fix** (separate variant
   path). Likely material since bands literally measure forecast-vs-actual error.
3. **Add the alignment fix to `data/build_training_dataset.py`** behind a flag.
   When set, the produced npy variant becomes the candidate dataset for a future
   TFT retrain (if the user wants to revisit TFT).
4. **Decide on promotion**: if PREDISPATCH debiaser also shows a small-but-real
   improvement, promote both `pd_debiaser_aligned30` and `pd7day_debiaser_aligned30`
   to the canonical paths. Otherwise document as a known but not-yet-actioned
   improvement.

## What I did NOT do here

- Did **not** modify the canonical `models/pd7day_debiaser/lgbm_final.pkl` — the
  variant lives in a sibling directory.
- Did **not** modify the live publish path's choice of debiaser — `forecast.py`
  still loads the canonical model.
- Did **not** retrain TFT, residual bands, or the PREDISPATCH debiaser.
- Did **not** rebuild the npy dataset variant.

These are queued in the recommendations above. The user's standing instruction
is to proceed toward the retrain; this PD7Day result confirms the alignment fix
is worth the effort for at least one component and is a concrete reproducible
template for the others.
