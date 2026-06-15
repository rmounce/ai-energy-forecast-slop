# LGBM Residual Driver Audit — 2026-06-14

## Caveman Summary

- Question: before adding price-model features, explain `amber_apf_lgbm` residuals.
- New tool: `eval/analyze_lgbm_residual_drivers.py`.
- Controlled run:
  `./.venv/bin/python eval/analyze_lgbm_residual_drivers.py --since 2026-04-01T00:00:00Z --until 2026-05-13T00:00:00Z --output-prefix eval/results/lgbm_residual_drivers_20260614_to20260513`
- Scored rows: `255,035`; target window: `2026-04-01 00:30Z -> 2026-05-15 23:00Z`.
- Overall LGBM bias: `+$3.14/MWh`; MAE: `$36.57/MWh`.
- Time split remains the main known shape:
  - over: overnight `+$15.57`, late `+$14.08`, evening `+$11.82`
  - under: solar `-$13.95`, morning `-$5.34`
- Long horizon still over-forecasts: `24h+` bias `+$6.79/MWh`, MAE `$43.01/MWh`.
- Strongest first-pass drivers by bias spread:
  - SDO net-interchange forecast error: `$57.62/MWh`
  - actual net interchange regime: `$53.34/MWh`
  - PV forecast error: `$47.41/MWh`
  - local wind actual bucket: `$33.30/MWh` (diagnostic only; not a market-wide wind signal)
- Weak driver: demand forecast error. SDO demand-error spread is only `$10.99/MWh`.
- Data gap: no local AEMO renewable/wind/semi-scheduled availability forecast parquet found.
  Do not treat BOM local wind as a substitute for grid renewable availability.
- Recommendation: investigate/interrogate interchange and renewable availability first;
  do not retrain or add features until this decomposition is repeated with a renewable
  availability source.
- 2026-06-15 update: AEMO STPASA REGIONSOLUTION now fills the renewable
  availability gap for tail work. The APF-backed residual ablation
  `eval/ablate_stpasa_tail_features.py` scored `amber_apf_lgbm` only, not an
  APF-free replacement path. With current STPASA merged through `2026-06-15
  05:00Z`, STPASA tail join coverage was `100.0%` for `28.5-72h`; validation
  MAE improved from original `$61.09/MWh` to baseline residual-corrected
  `$31.16/MWh`, and to STPASA residual-corrected `$20.97/MWh`.

## Method

`eval/analyze_lgbm_residual_drivers.py` extends `eval/audit_price_forecast.py`.
It reads `price_forecast_log.csv` with `model_name=price`, joins realised SA1
actuals from `data/parquet/actuals_sa1.parquet`, then joins AEMO forecasts as
issued at the original forecast creation time:

- PREDISPATCH: `data/parquet/aemo_predispatch_sa1.parquet`, latest `run_time <= forecast_creation_time`
  for the same interval-end target.
- SevenDayOutlook: `data/parquet/aemo_sevendayoutlook_sa1.parquet`, same as-of rule.

Residual convention:

- `residual_mwh = prediction * 1000 - actual_rrp`
- positive residual = LGBM over-forecast
- negative residual = LGBM under-forecast

The result CSVs are generated under `eval/results/` and are ignored by default.
They can be regenerated with the command in the summary.

## Headline Tables

### Time Of Day

| Adelaide bucket | n | actual | pred | MAE | bias |
|---|---:|---:|---:|---:|---:|
| evening | 43,362 | 86.33 | 98.15 | 37.23 | +11.82 |
| late | 31,107 | 68.66 | 82.74 | 39.84 | +14.08 |
| morning | 62,075 | 43.83 | 38.48 | 35.03 | -5.34 |
| overnight | 62,194 | 57.91 | 73.48 | 37.67 | +15.57 |
| solar | 56,297 | 44.38 | 30.44 | 34.76 | -13.95 |

### Horizon

| horizon | n | actual | pred | MAE | bias |
|---|---:|---:|---:|---:|---:|
| 0-1h | 3,486 | 56.59 | 50.39 | 14.00 | -6.19 |
| 1-4h | 10,453 | 56.62 | 49.29 | 17.90 | -7.34 |
| 4-12h | 27,903 | 56.92 | 52.34 | 23.64 | -4.58 |
| 12-24h | 42,439 | 57.28 | 54.18 | 25.62 | -3.09 |
| 24h+ | 170,754 | 57.93 | 64.72 | 43.01 | +6.79 |

### Driver Rank

Bias-spread is `max(bucket bias) - min(bucket bias)` for each bucketed driver.

| driver | joined n | bias spread | MAE spread |
|---|---:|---:|---:|
| SDO net-interchange error bucket | 204,933 | 57.62 | 30.35 |
| actual net-interchange bucket | 255,035 | 53.34 | 15.60 |
| PV forecast error bucket | 254,414 | 47.41 | 23.41 |
| local wind actual bucket | 254,843 | 33.30 | 10.36 |
| actual PV bucket | 254,843 | 28.93 | 10.45 |
| actual demand bucket | 255,035 | 22.11 | 6.02 |
| PREDISPATCH net-interchange error bucket | 75,326 | 18.48 | 4.56 |
| PREDISPATCH demand error bucket | 75,326 | 17.83 | 12.20 |
| local wind error bucket | 253,365 | 17.14 | 20.84 |
| SDO demand error bucket | 204,933 | 10.99 | 8.36 |

## Interpretation

The previous bias audit result still holds on the controlled window: LGBM is not
uniformly high or low. It over-prices overnight/late/evening and under-prices
solar/morning. The strategic problem remains concentrated in the `24h+` region,
where the 72h LP sees a `+$6.79/MWh` mean over-forecast and a `$73.77/MWh`
p90 overshoot.

Interchange is the strongest market driver currently visible in local data:

- When actual net interchange is in the high-export bucket, actual prices average
  `$104.62/MWh` but LGBM predicts `$78.54/MWh`: bias `-$26.08/MWh`.
- When actual net interchange is in the high-import bucket, actual prices average
  only `$15.63/MWh` but LGBM predicts `$42.88/MWh`: bias `+$27.26/MWh`.

That pattern says the model is not just missing price level. It is getting the
price/interconnector relationship wrong in a way that changes sign by regime.
This is more actionable than a blunt bias correction.

PV forecast error is also material:

- PV forecast too high: bias `-$36.81/MWh`, MAE `$50.26/MWh`.
- PV near-zero error: bias `+$10.60/MWh`, MAE `$36.36/MWh`.

Demand forecast error is weaker. It is not zero, but it is not the first feature
track to pursue.

Local BOM wind buckets correlate with residuals, but this should be treated as a
proxy diagnostic only. It is probably standing in for broader renewable output and
weather systems. The handover recommendation was right: prefer AEMO renewable/wind
forecast or semi-scheduled availability data, with local station wind as fallback.

## STPASA Update — 2026-06-15

The renewable-availability source for this line is now AEMO STPASA
REGIONSOLUTION, normalised into
`data/parquet/aemo_stpasa_regionsolution_sa1.parquet`.

Plain-English shape:

- What it provides: SA1 regional short-term PASA forecast rows, including
  demand bands, aggregate available capacity, total intermittent generation,
  UIGF, semi-scheduled capacity, and split solar/wind UIGF/capacity.
- Resolution: 30-minute target intervals.
- Horizon: local validation showed every merged run reaches at least 72h; source
  max horizons range roughly `160-183h`.
- Cadence: hourly current `PUBLIC_STPASA` files from NEMWeb
  `Reports/CURRENT/Short_Term_PASA_Reports`; monthly MMSDM archive ZIPs remain
  useful for older history.
- Pipeline fit: join by exact target interval and latest STPASA `run_time <=
  forecast_creation_time`, then use the joined fields only as residual-correction
  features for the incumbent `amber_apf_lgbm` APF tail.

Backfill commands used:

```bash
./.venv/bin/python ingest/backfill_stpasa_regionsolution.py --source current --start 2026-05 --end 2026-06 --min-horizon-hours 72.0
./.venv/bin/python eval/ablate_stpasa_tail_features.py --since 2026-04-01T00:00:00Z --tail-start-hours 28.5 --max-horizon-hours 72.0 --output-prefix eval/results/stpasa_tail_ablation_apf_202604_to_latest
```

Current validation result:

| model | split | n | original MAE | corrected MAE | MAE delta | corrected bias |
|---|---:|---:|---:|---:|---:|---:|
| baseline residual corrector | val | 141,532 | 61.09 | 31.16 | -29.93 | +15.68 |
| baseline + STPASA residual corrector | val | 141,532 | 61.09 | 20.97 | -40.12 | +9.80 |

Interpretation: STPASA adds about `$10.19/MWh` incremental validation MAE
improvement over the non-STPASA residual corrector on this split. This is still
a residual-correction probe; it is not yet a production retrain or dispatch
financial gate.

## Data Gap Status

Local parquet inventory has:

- `aemo_predispatch_sa1.parquet`: RRP, total demand, net interchange.
- `aemo_sevendayoutlook_sa1.parquet`: scheduled demand, net interchange.
- `actuals_sa1.parquet`: actual RRP, total demand, net interchange, site PV/load,
  local BOM weather.
- `aemo_stpasa_regionsolution_sa1.parquet`: STPASA regional renewable
  availability and capacity fields for SA1.

It still does not expose realised renewable generation by technology as an
actual-vs-forecast error source. STPASA is suitable as a forward-looking tail
feature source; a later decomposition can still add realised renewable buckets if
we need explanatory diagnostics rather than forecast-only features.

## Next Steps

1. Add a renewable-availability data source if practical:
   AEMO wind/solar/renewable forecast, semi-scheduled availability, or a suitable
   regional UIGF/availability feed.
2. Rerun `analyze_lgbm_residual_drivers.py` with renewable actual/forecast buckets.
3. If interchange and renewable buckets remain strong, build a feature ablation
   backtest before retraining production LGBM.
4. Only after that, test a conservative horizon x Adelaide-bucket calibration as
   a separate dispatch ablation, not as a substitute for understanding the driver.
