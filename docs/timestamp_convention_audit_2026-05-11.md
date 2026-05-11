# Timestamp Convention Audit - 2026-05-11

This note records the timestamp issues found after the HA `sensor.ai_aemo_price_forecast` chart showed an apparent 30-minute offset against Amber/AEMO.

## Confirmed Production Fixes

### Raw AEMO comparison entity

`sensor.ai_aemo_price_forecast` is a HA-facing raw upstream comparison surface. It now:

- selects one explicit latest PREDISPATCH `run_time` instead of `last(rrp)` across all run tags;
- republishes AEMO interval-ending timestamps as interval starts:
  - P5MIN: `time - 5 min`
  - PREDISPATCH / PD7Day: `time - 30 min`

This fixed the visible dashboard alignment. Live comparison against Amber billing interval forecasts showed zero-shift alignment was best after this change.

### Amber spot scaling

Amber `Forecasts[].spot_per_kwh` is scaled relative to direct raw AEMO wholesale. On 2026-05-11, the PREDISPATCH leg showed:

- `Amber spot_per_kwh / sensor.ai_aemo_price_forecast.wholesale_price ~= 1.10`
- dividing Amber `spot_per_kwh` by `amber_api_scaling_factor` reduced PREDISPATCH MAE from about `0.0083 $/kWh` to about `0.000018 $/kWh`

So raw-wholesale Lovelace comparisons should divide Amber `spot_per_kwh` by the current `tariff_profile.json` `amber_api_scaling_factor`.

## Additional Low-Risk Fix

`_get_aemo_short_term_forecast()` used the AEMO 30MIN visualisations API for demand/interchange covariates but inferred row duration from adjacent timestamps. The first row therefore used a 5-minute duration fallback and could be shifted into the wrong 30-minute bin.

It now treats every 30MIN API row as interval-ending and subtracts 30 minutes, matching `_get_aemo_short_term_price_forecast()` and the 7-day outlook parser.

## Deeper Internal Alignment Finding

A quick historical diagnostic suggests there may be a broader internal interval-end / interval-start mismatch between PREDISPATCH covariates and 30-minute actual targets.

Method:

- load `data/parquet/aemo_predispatch_sa1.parquet`
- compute PREDISPATCH step index with current training convention: `step 0 = interval_dt - run_time = 30 min`
- compare each PREDISPATCH row against `actuals_sa1.parquet` at:
  - same `interval_dt`
  - `interval_dt - 30 min`
  - `interval_dt + 30 min`

Result, MAE in $/MWh:

| PREDISPATCH step | same timestamp | minus 30 min | plus 30 min |
|---:|---:|---:|---:|
| 0 | 155.13 | 143.29 | 167.26 |
| 1 | 165.36 | 155.98 | 177.04 |
| 2 | 178.77 | 169.79 | 188.94 |
| 5 | 196.95 | 190.06 | 207.30 |
| 11 | 236.13 | 227.85 | 247.48 |
| 55 | 537.17 | 534.13 | 544.55 |

The `interval_dt - 30 min` comparison is consistently better, especially at short horizons. That is evidence that AEMO forecast rows are interval-ending while `actuals_sa1.parquet` rows may be interval-start labelled.

## Diagnostic Confirmation (2026-05-11 late)

Ran `eval/timestamp_alignment_diagnostic.py` over full available history
(2024-01-01 onward). For each AEMO-derived forecast row, compared the
forecast against `actuals_sa1.parquet.rrp` at three timestamp shifts of
the forecast row's `interval_dt`. Result is unambiguous:

Winning-shift share (sample-weighted, % of strata) per source × horizon:

| source | horizon | -30 min | +0 | +30 min |
|---|---|---:|---:|---:|
| predispatch | h<=30m | **99.4** | 0.6 | 0.0 |
| predispatch | h=30m–6h | **98.8** | 0.9 | 0.3 |
| predispatch | h=6–14h | **93.9** | 2.1 | 4.0 |
| predispatch | h=14–28h | **88.0** | 0.6 | 11.4 |
| predispatch | h=28–72h | **86.9** | 5.1 | 8.0 |
| p5min | h<=30m | **98.8** | 1.1 | 0.1 |
| p5min | h=30m–6h | **95.2** | 4.1 | 0.7 |
| pd7day | h<=30m | **93.7** | 5.1 | 1.3 |
| pd7day | h=30m–6h | **90.8** | 4.2 | 5.0 |
| pd7day | h=6–14h | **82.8** | 17.2 | 0.0 |
| pd7day | h=14–28h | **71.4** | 4.7 | 23.9 |
| pd7day | h=28–72h | **65.8** | 25.7 | 8.5 |
| pd7day | h=72h–7d | 31.5 | 25.0 | **43.5** |

Sample-weighted MAE per source × shift ($/MWh):

| source | -30 min | +0 | +30 min |
|---|---:|---:|---:|
| p5min | **52.06** | 59.46 | 71.59 |
| predispatch | **115.02** | 120.87 | 128.44 |
| pd7day | **544.52** | 548.37 | 545.48 |

Interpretation:

- `interval_dt` is **interval-end** for PREDISPATCH, P5MIN, and PD7Day.
- `actuals_sa1.parquet.time` is **interval-start**.
- HA chart surfaces and any analysis that compares forecasts against actuals
  should apply a `-interval` shift on the forecast `interval_dt` to convert
  interval-end → interval-start.
- PD7Day at horizons beyond 28h becomes noisy enough that `+30 min` even
  beats `-30 min` in some strata; this is consistent with PD7Day at long
  horizons being a categorical spike-indicator rather than a literal price
  prediction, so the timestamp alignment matters less.

## Do Not Silently Patch Yet

Do not immediately shift all PREDISPATCH/PD7Day internals in production code. The current TFT/PD-direct/strategic models and rolling evals were trained and scored under the existing convention. A blind shift would change model inputs, labels, and cached artifacts together in a way that could invalidate recent results without a clean comparison.

Recommended next branch:

1. Add a dedicated timestamp-alignment diagnostic script that computes same/minus30/plus30 MAE by source, horizon, regime, and run period.
2. Rebuild a small strategic dataset variant with PREDISPATCH/PD7Day interval starts, leaving raw parquet unchanged or writing clearly named derived columns.
3. Run one short side-by-side eval with the corrected alignment before retraining anything substantial.
4. If confirmed, rebuild/retrain the affected strategic/TFT/PD-direct artifacts and rerun the tariffed gate.

## Current Working Convention

- External HA/chart surfaces should use UTC interval starts.
- AEMO raw ingest can preserve source timestamps, but derived/exported model datasets should explicitly state whether `interval_dt` is source interval end or control interval start.
- Avoid mixing Adelaide local time with NEM time except for explicitly local features such as household load time-of-day or tariff lookups.
