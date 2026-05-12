# Shadow Forecast Artefact Manifest — 2026-05-12

This manifest records the local artefacts that make up the current promoted
PD-direct shadow forecast stack after the 2026-05-11 interval-alignment fix.

The purpose is operational provenance. Some live artefacts are intentionally
gitignored under `models/` and `data/parquet/`; this file records what the live
host is using so deployment and rollback are not implicit local state.

## Scope

This manifest covers the shadow raw-wholesale price forecast used by:

- `sensor.ai_pd_direct_price_forecast`
- `sensor.ai_pd_direct_price_forecast_low`
- `sensor.ai_pd_direct_price_forecast_high`
- `sensor.ai_spot_price_forecast`
- `sensor.ai_spot_price_forecast_low`
- `sensor.ai_spot_price_forecast_high`

It does not change the current production EMHASS control selectors. At the time
of this manifest, the AI shadow price source is still published for comparison,
not selected for production control.

## Current Head

Manifest created against repo head:

- `70a3d6f` — `Clarify alignment promotion follow-ups`

Promotion commit:

- `211ae8d` — `Promote alignment-fix variants to canonical PD-direct shadow`

Supporting training/eval commits:

- `3ea05d2` — `PREDISPATCH debiaser + residual bands alignment-fix variants`
- `88d7ab0` — `PD7Day debiaser alignment-fix side-by-side`
- `6614776` — `Fix spike-classifier silent fallback on PD-direct path`
- `f184320` — `Timestamp alignment diagnostic + PD-direct -30min publish shift`
- `f8bd8ea` — `Fix PD-direct/TFT PREDISPATCH stale-run bug`

## Promoted Live Artefacts

| Role | Path | SHA-256 |
|---|---|---|
| PREDISPATCH debiaser model | `models/pd_debiaser/lgbm_final.pkl` | `1307dcbb7c68f240a606801b9bb6751461eab4540159e524a9641aaefdcc23c5` |
| PREDISPATCH debiaser metrics | `models/pd_debiaser/metrics.json` | `34677380fbff78133f939248b1ff20e4d4aad848ff537bf30241d197522b78c4` |
| PD7Day debiaser model | `models/pd7day_debiaser/lgbm_final.pkl` | `c5a30a57efb69c8499d98470e845a9904184ba314ebc00ef765ad0fc960dc0dd` |
| PD7Day debiaser metrics | `models/pd7day_debiaser/metrics.json` | `f40a9c37575b2bc8f98e50be2e864d319ec6f68353033db1534de981af7f1986` |
| PREDISPATCH residual bands | `models/pd_residual/residual_bands.parquet` | `f0e5d381092644e63610420289b8455c36bc610de25e669f2b54633ab9eac558` |
| PREDISPATCH OOF series | `data/parquet/debiased_pd_rrp_oof.parquet` | `549d7324db7cbaeda7776cc04443fc71f96836de8dd552c1e79dc3d54154468b` |
| PD7Day OOF series | `data/parquet/debiased_pd7day_oof.parquet` | `be5f3865609c38faad3010239b2589421ddc155fccce469afc4ac1d6077e5a82` |

## Aligned Variant References

The promoted canonical artefacts were copied from the alignment-fix sibling
paths. These hashes should match the canonical paths above:

| Role | Path | SHA-256 |
|---|---|---|
| PREDISPATCH aligned model | `models/pd_debiaser_aligned30/lgbm_final.pkl` | `1307dcbb7c68f240a606801b9bb6751461eab4540159e524a9641aaefdcc23c5` |
| PREDISPATCH aligned OOF | `data/parquet/debiased_pd_rrp_oof_aligned30.parquet` | `549d7324db7cbaeda7776cc04443fc71f96836de8dd552c1e79dc3d54154468b` |
| PD7Day aligned OOF | `data/parquet/debiased_pd7day_oof_aligned30.parquet` | `be5f3865609c38faad3010239b2589421ddc155fccce469afc4ac1d6077e5a82` |

## Rollback Snapshots

The previous canonical artefacts were retained locally with the
`.canonical_20260511` suffix:

| Role | Path | SHA-256 |
|---|---|---|
| Previous PREDISPATCH model | `models/pd_debiaser/lgbm_final.pkl.canonical_20260511` | `2e58fc0cb03a8837a5cad91e5d0304ab3e15a449778c99a4d9722aea133a17d3` |
| Previous PREDISPATCH metrics | `models/pd_debiaser/metrics.json.canonical_20260511` | `081f1cd72b51103ec61fcd6c43987f398f06348f36a010621e097b68937e72ac` |
| Previous PD7Day model | `models/pd7day_debiaser/lgbm_final.pkl.canonical_20260511` | `5bb79b6db313aa882ccb532b24c832b255fe07a01d77de1c3e352be497a968ef` |
| Previous PD7Day metrics | `models/pd7day_debiaser/metrics.json.canonical_20260511` | `822df3142fa637805b09c405338527f754733fa9f9870e8d2e5a9d7dfa8c5b59` |
| Previous residual bands | `models/pd_residual/residual_bands.parquet.canonical_20260511` | `f47a356e305ce41b8176ed79f02847228abc09a1bff7138b73abed462263f3e3` |
| Previous PREDISPATCH OOF | `data/parquet/debiased_pd_rrp_oof.parquet.canonical_20260511` | `40cf214a7a3e34d302448f36d5b5019d5a63e3e0c5af0fddb2611692548919f8` |
| Previous PD7Day OOF | `data/parquet/debiased_pd7day_oof.parquet.canonical_20260511` | `347722614ea6cb7a8f55031a14d3a0220f27c267d3115e7200bd5fae5883e622` |

Rollback command:

```bash
for f in models/pd_debiaser/lgbm_final.pkl models/pd_debiaser/metrics.json \
         models/pd7day_debiaser/lgbm_final.pkl models/pd7day_debiaser/metrics.json \
         models/pd_residual/residual_bands.parquet \
         data/parquet/debiased_pd_rrp_oof.parquet \
         data/parquet/debiased_pd7day_oof.parquet; do
  cp -v "${f}.canonical_20260511" "$f"
done
```

Note: these rollback snapshots are local ignored artefacts. They are not
recreated by a fresh git clone.

## Provenance

The alignment-fix variants were trained or rebuilt by adding
`--actuals-shift-min 30` to the PREDISPATCH and PD7Day training/evaluation
paths so interval-ending AEMO forecast rows join against the actual price for
the same half-hour. See:

- `docs/alignment_fix_retrain_2026-05-11.md`
- `docs/pipeline_audit_2026-05-11.md`
- `docs/timestamp_convention_audit_2026-05-11.md`

Observed side-by-side improvements before promotion:

- PREDISPATCH debiaser: overall MAE `61.97 -> 60.88` (`-1.8%`), h_1-6 MAE
  `58.66 -> 55.76` (`-4.9%`).
- PD7Day debiaser: overall MAE `36.98 -> 36.14` (`-2.3%`).
- Residual bands: held-out mean width `64.13 -> 61.39` (`-4.3%`) at similar
  coverage.

## Verification Commands

To verify the live host still matches this manifest:

```bash
sha256sum \
  models/pd_debiaser/lgbm_final.pkl \
  models/pd_debiaser/metrics.json \
  models/pd7day_debiaser/lgbm_final.pkl \
  models/pd7day_debiaser/metrics.json \
  models/pd_residual/residual_bands.parquet \
  data/parquet/debiased_pd_rrp_oof.parquet \
  data/parquet/debiased_pd7day_oof.parquet
```

To publish a fresh shadow forecast after verifying artefacts:

```bash
nice -n 19 ./.venv/bin/python forecast.py predict-price --publish-hass
```
