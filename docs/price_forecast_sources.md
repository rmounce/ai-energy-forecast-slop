# Price Forecast Sources

## Caveman Summary

- `amber_apf_lgbm` is the current APF-backed production incumbent.
- It comes from `price_forecast_log.csv` rows with `model_name='price'`.
- It is the only current source to use for APF extrapolation questions.
- `p5min_tactical`, `pd_direct`, `model_a_hybrid`, and `lgbm_strategic` are
  APF-free or APF-independent price paths.
- Work on those other price paths is suspended, not fully abandoned. They remain
  useful for reference and possible revival, but they are not currently trusted
  replacement paths.
- Do not treat APF-free strategic results as evidence that the APF extrapolation
  path improved.
- Code registry: `eval/price_source_contracts.py`.
- Main APF-tail/STPASA probe: `eval/ablate_stpasa_tail_features.py`.
- Dispatch probe artifact generator: `eval/build_stpasa_residual_price_log.py`
  adjusts only the incumbent APF tail residuals.
- Production-shaped STPASA path: `stpasa_*` future covariates in `forecast.py`
  for the incumbent `price` model. Live `config.yaml` is promoted and production
  model artifacts were retrained with these covariates on 2026-06-16.

## Source Contracts

| Source label | APF-backed? | Artifact/log | Resolution | Horizon | Current status | Use for | Do not use for |
|--------------|-------------|--------------|------------|---------|----------------|---------|----------------|
| `amber_apf_lgbm` | Yes | `price_forecast_log.csv`, `model_name='price'` | 30 min | 0-72h, 144 steps | Production incumbent / active APF extrapolation baseline with STPASA covariates | Current APF extrapolation, tail residual correction, STPASA feature value | APF-free replacement claims |
| `p5min_tactical` | No | `p5min_forecast_log.csv`, tactical sensors | 5 min | 0-60 min, 12 steps | Suspended, retained for reference | Explicit tactical-price revival work | Current production APF extrapolation evaluation |
| `pd_direct` | No | `pd_direct_forecast_log.csv`, PD-direct sensors | 30 min | 0-72h, 144 steps | Suspended, retained for reference | Explicit APF-free revival work | Evidence about APF extrapolation improvements |
| `model_a_hybrid` | No | `retro_tier1_forecasts.pkl` + `retro_tft_forecasts.pkl` | 5 min tactical prefix plus 30 min tail | 0-72h stitched curve | Suspended, retained for reference | Historical hybrid/TFT comparisons | Evidence about APF extrapolation improvements |
| `lgbm_strategic` | No | `retro_lgbm_strategic_forecasts.pkl` | 30 min | 0-72h, 144 steps | Suspended APF-free experiment | Explicit APF-free strategic experiments | APF extrapolation evaluation |

## Lineage

### `amber_apf_lgbm`

This is the as-run production incumbent. Amber commercial APF provides the
near-horizon signal, and the incumbent price LightGBM extrapolates the curve to
the full 72h strategic horizon. Since 2026-06-16, the production extrapolator
also consumes STPASA renewable availability covariates from
`data/parquet/aemo_stpasa_regionsolution_sa1.parquet`, refreshed by
`ai-energy-stpasa.timer`. In evaluation code, this source is usually named
`amber_apf_lgbm`; in the raw forecast log, its rows are selected with
`model_name='price'`.

Use this source when the question is:

- did the APF extrapolation tail improve?
- can STPASA explain or correct the 28.5-72h residual?
- does a single-stage STPASA covariate retrain improve the incumbent APF-backed
  extrapolator?
- what would the current production APF-backed path have done?

### Suspended APF-Free Paths

`p5min_tactical`, `pd_direct`, `model_a_hybrid`, and `lgbm_strategic` are not
deleted, but active work on them is suspended. The common problem was trust and
scope: the experiments did not show a clear path to a replacement forecast
source that could be relied on in production, and tactical Tier 1 is not
currently used by EMHASS. They can still be revived deliberately, but they
should not be mixed into APF extrapolation work without an explicit comparison
design.

## Guardrail

When adding or running an eval, first identify the source label and whether it is
APF-backed. If the experiment is about APF extrapolation, it should either:

- read `price_forecast_log.csv` with `model_name='price'`, or
- transform/evaluate an artifact derived directly from that logged APF-backed
  curve.

`eval/build_stpasa_residual_price_log.py` is in the second category: it keeps
the logged `amber_apf_lgbm` curve as the base and adjusts only validation-window
`28.5-72h` rows using a learned residual correction. It is a dispatch-eval
probe, not a separate APF-free forecast family.

If the run consumes `p5min_tactical`, `pd_direct`, `model_a_hybrid`,
`retro_tft_forecasts.pkl`, or `retro_lgbm_strategic_forecasts.pkl`, it is not
evaluating the APF extrapolation model unless the script explicitly compares
against `amber_apf_lgbm`.
