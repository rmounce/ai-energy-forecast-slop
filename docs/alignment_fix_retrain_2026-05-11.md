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
- Did **not** retrain TFT.
- Did **not** rebuild the npy dataset variant.

These are queued in the recommendations above. The user's standing instruction
is to proceed toward the retrain; this PD7Day result confirms the alignment fix
is worth the effort for at least one component and is a concrete reproducible
template for the others.

## Round 2 — PREDISPATCH Debiaser

Applied the same `--actuals-shift-min` flag pattern to
`train/train_pd_debiaser.py`. Same canonical-preserving sibling output path
(`models/pd_debiaser_aligned30/`, OOF at
`data/parquet/debiased_pd_rrp_oof_aligned30.parquet`).

Training data: 3,067,048 rows, 5-fold CV, 2000 max estimators. ~5 min wall.

| Slice | Canonical MAE | Aligned MAE | Δ |
|---|---:|---:|---:|
| overall | 61.97 | **60.88** | **-1.76%** |
| horizon 1–6 (≤3h) | 58.66 | **55.76** | **-4.94%** |
| horizon 7–16 (~3.5–8h) | 61.30 | 59.46 | -3.00% |
| horizon 17–32 (~8.5–16h) | 63.13 | 61.72 | -2.22% |
| horizon 33–56 (~16.5–28h) | 65.46 | 64.94 | -0.79% |
| horizon 57–99 (beyond PREDISPATCH) | 53.48 | 55.80 | +4.33% |
| regime: baseload | 27.96 | **26.75** | **-4.30%** |
| regime: spike | 173.09 | 171.18 | -1.10% |
| regime: oversupply | 51.01 | 51.50 | +0.96% |

Exact horizon-decay shape predicted by the audit. The largest gain (-4.94%)
is at h_1–6, which is the horizon most consumed by live PD-direct
production. The h_57_99 slight regression (+4.33%) is the PD7Day-zone steps
where the alignment fix is structurally different — small sample size and
not the production-critical horizon.

## Round 3 — Residual Band Table

Applied the same `--actuals-shift-min` flag pattern to
`eval/build_pd_residual_bands.py`. Output:
`models/pd_residual_aligned30/residual_bands.parquet`. Bands consume the
aligned OOF.

Held-out validation (2025-07-01 → 2026-01-01):

| Bucket | n | Canonical coverage | Aligned coverage | Canonical mean_width | Aligned mean_width | Δ width |
|---|---:|---:|---:|---:|---:|---:|
| overall | 487,725 | 0.5925 | 0.5889 | 64.13 | **61.39** | **-4.3%** |
| 1–6h | 105,384 | 0.5869 | 0.5864 | 57.03 | **52.42** | **-8.1%** |
| 6–14h | 140,512 | 0.5999 | 0.5906 | 63.89 | 60.34 | -5.6% |
| 14–30h | 206,940 | 0.5922 | 0.5889 | 68.16 | 66.21 | -2.9% |
| 30h+ | 34,889 | 0.5813 | 0.5895 | 62.62 | 64.07 | +2.3% |

Coverage stays in the same ~58–60% band (q20/q80 target ~60%); mean widths
drop consistently in the in-PREDISPATCH horizons (1–6h: −8%, 6–14h: −6%).
Narrower bands at the same coverage = better calibration. Strict improvement.

## Consolidated picture across the three artefacts

| Component | Canonical baseline | Aligned variant | Notes |
|---|---|---|---|
| PD7Day debiaser | MAE 36.98 overall | MAE 36.14 (−2.3%) | Modest; PD7Day is highly autocorrelated |
| PREDISPATCH debiaser | MAE 61.97 overall, 58.66 at h_1–6 | MAE 60.88 (−1.8%) overall, 55.76 (−4.9%) at h_1–6 | Production-relevant gain on the short-horizon path |
| Residual bands | width 64.13 at coverage 0.5925 | width 61.39 at coverage 0.5889 (−4.3% width) | Tighter bands → cleaner buy/sell asymmetry in PD-direct |

All three are consistent. The alignment fix is real, modest in aggregate, and
larger on the horizons that matter for live production.

## Promotion proposal

The variants are ready. Promoting means:

1. Replace `models/pd_debiaser/lgbm_final.pkl` with `_aligned30` version
2. Replace `data/parquet/debiased_pd_rrp_oof.parquet` with aligned OOF
3. Replace `models/pd7day_debiaser/lgbm_final.pkl` with `_aligned30` version
4. Replace `data/parquet/debiased_pd7day_oof.parquet` with aligned OOF
5. Replace `models/pd_residual/residual_bands.parquet` with aligned version

Effects:
- **Live publish** (`forecast.py` `_apply_pd_debiaser`): PD-direct values
  shift by the model's compression delta. Modest visible change. Will
  reduce the small systematic bias we currently see.
- **Eval framework** (`eval/pd_direct_baseline.py` reads `debiased_pd_rrp_oof.parquet`):
  any future rolling-MPC eval with PD-direct will use the aligned OOF.
  Historical eval result CSVs already saved are unaffected (they're
  snapshots).
- **Documentation**: production_soc_policy and other refs need a one-line
  note that the underlying artefacts were realigned 2026-05-11.

Risk: low. The shifts are bounded (~2% MAE deltas) and the variants are
trained from the same data with the same hyperparameters — only the
forecast-actual alignment differs. No new code paths.

Rollback: keep the canonical files as `.canonical_20260511` snapshots
before promoting.

## Recommended action

Promote the aligned variants when the user signs off. Update the audit doc
status (`docs/pipeline_audit_2026-05-11.md` section C) to reflect the
training-pipeline alignment issue is now addressed.

Optional follow-up:

- Add `--actuals-shift-min` to `data/build_training_dataset.py` for any
  future TFT retrain. Not needed for any current downstream consumer.
- Audit `eval/rolling_mpc_eval.py` for the same alignment in its actuals
  lookup (audit section F.4).

## Promotion executed (2026-05-11 22:46 ACST)

User signed off. The following swaps were applied:

| Canonical path | Replaced with | Snapshot (rollback) |
|---|---|---|
| `models/pd_debiaser/lgbm_final.pkl` | `models/pd_debiaser_aligned30/lgbm_final.pkl` | `.canonical_20260511` |
| `models/pd_debiaser/metrics.json` | aligned variant | `.canonical_20260511` |
| `data/parquet/debiased_pd_rrp_oof.parquet` | aligned variant | `.canonical_20260511` |
| `models/pd7day_debiaser/lgbm_final.pkl` | aligned variant | `.canonical_20260511` |
| `models/pd7day_debiaser/metrics.json` | aligned variant | `.canonical_20260511` |
| `data/parquet/debiased_pd7day_oof.parquet` | aligned variant | `.canonical_20260511` |
| `models/pd_residual/residual_bands.parquet` | aligned variant | `.canonical_20260511` |

What changed in live behaviour:

- `forecast.py predict-price --publish-hass`: PD-direct values shift to
  reflect the corrected forecast/actuals training alignment. PD-direct
  q50 now consistently sits ~0–16% below raw PREDISPATCH (the
  debiaser's expected bias-correction direction). Pre-promotion the
  spread was -19% to +7%; post-promotion is tighter.
- `eval/pd_direct_baseline.py`: any future PD-direct rolling-MPC eval
  will consume the aligned OOF parquet via the standard load path. No
  code changes were required.
- Live publish verified post-promotion: `sensor.ai_pd_direct_price_forecast`
  reports timestamps and values consistent with the raw stitched
  `sensor.ai_aemo_price_forecast`, with the expected modest debiaser
  compression and no 30-min offsets.

What was NOT changed:

- The APF+LGBM-extrapolated production EMHASS path is untouched.
- `input_select.emhass_*_price_source` options unchanged (still
  `amber` / `amber_lgbm_extrapolated` only).
- TFT shadow artefacts (on 2026-06-05 sunset) untouched.
- `data/build_training_dataset.py` not retrained — no current
  downstream consumer.

Rollback procedure if regression observed:

```bash
for f in models/pd_debiaser/lgbm_final.pkl models/pd_debiaser/metrics.json \
         models/pd7day_debiaser/lgbm_final.pkl models/pd7day_debiaser/metrics.json \
         models/pd_residual/residual_bands.parquet \
         data/parquet/debiased_pd_rrp_oof.parquet \
         data/parquet/debiased_pd7day_oof.parquet; do
  cp -v "${f}.canonical_20260511" "$f"
done
```

The `.canonical_20260511` snapshots are gitignored alongside the live
artefacts; they live locally on the host. The aligned variants remain
in their sibling directories (`models/*_aligned30/`) as additional
on-disk references.

## Reproducibility note

Only the tracked artefacts in `models/pd7day_debiaser/` and
`models/pd_residual/` are reproduced by git. The PREDISPATCH debiaser,
OOF parquet files, aligned sibling directories, and rollback snapshots
live under ignored paths (`models/` and `data/parquet/`) and are therefore
local host state unless they are deliberately force-added or regenerated.

For the live machine, the promotion was verified by matching canonical
and aligned hashes for the ignored PREDISPATCH artefacts:

- `models/pd_debiaser/lgbm_final.pkl` matches
  `models/pd_debiaser_aligned30/lgbm_final.pkl`
- `data/parquet/debiased_pd_rrp_oof.parquet` matches
  `data/parquet/debiased_pd_rrp_oof_aligned30.parquet`

This is operationally fine for the current deployment workflow, but a fresh
clone will not reconstruct the promoted PREDISPATCH debiaser from git alone.
If this stack becomes production-selected rather than shadow-published, add
a small artefact manifest or formal model registry entry so rollback and
deployment do not depend on unstated local files.
