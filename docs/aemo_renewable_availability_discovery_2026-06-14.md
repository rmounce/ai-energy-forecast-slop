# AEMO Renewable Availability Discovery — 2026-06-14

## Caveman Summary

- Goal: find a better renewable/wind availability source for LGBM price residual
  decomposition before adding model features.
- Critical horizon check: `PDPASA_REGIONSOLUTION` is not enough for the 72h
  extrapolation target. Latest sampled PDPASA current file covered only `0-29h`.
- Best first target for the model tail: `STPASA_REGIONSOLUTION`.
- Why: archive sample covered `28.5-172h`, including the full 72h horizon, and
  carries region-level renewable availability fields such as `UIGF`,
  `SS_SOLAR_UIGF`, `SS_WIND_UIGF`, `SS_SOLAR_CAPACITY`, and `SS_WIND_CAPACITY`.
- STPASA current path found:
  `https://nemweb.com.au/Reports/CURRENT/Short_Term_PASA_Reports/`
- Runtime caveat: current listing has `PUBLIC_STPASA_YYYYMMDDHHMM_*.zip` files
  but must be checked for freshness before live integration.
- Useful overlap/short horizon source: `PDPASA_REGIONSOLUTION`.
- PDPASA current path:
  `https://nemweb.com.au/Reports/Current/PDPASA/`
- Archive path:
  `https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/YYYY/MMSDM_YYYY_MM/MMSDM_Historical_Data_SQLLoader/DATA/`
- Archive file example:
  `PUBLIC_ARCHIVE%2523STPASA_REGIONSOLUTION%2523FILE01%2523202604010000.zip`
- Implemented archive backfill:
  `ingest/backfill_stpasa_regionsolution.py`
- Dry-run validation command:
  `./.venv/bin/python ingest/backfill_stpasa_regionsolution.py --start 2026-04 --end 2026-04 --dry-run`
- Dry-run result: April 2026 archive produced `207,360` SA1 rows, `720` runs,
  and source horizons from `16.5h` to `183.0h`, satisfying the 72h requirement.
- Output parquet target:
  `data/parquet/aemo_stpasa_regionsolution_sa1.parquet`
- Residual audit integration:
  `eval/analyze_lgbm_residual_drivers.py` joins STPASA when that parquet exists,
  validates source horizon >= requested max horizon, and validates tail join
  coverage over `28.5-72h`.
- Offline ablation:
  `eval/ablate_stpasa_tail_features.py`
- April 2026 ablation result: on held-out `28.5-72h` tail rows, baseline
  residual correction reduced MAE by `$12.12/MWh`; baseline+STPASA reduced MAE
  by `$17.37/MWh` and reduced corrected bias from `+$12.77/MWh` to
  `+$4.67/MWh`.
- Useful actual/proxy: `Dispatch_SCADA` current feed and archive `INTERMITTENT_GEN_SCADA`
  / dispatch SCADA unit output, but those are DUID-level and need unit metadata to
  aggregate SA wind/solar cleanly.
- Not first target: `INTERMITTENT_DS_RUN/PRED`.
  It is exact dispatch UIGF by DUID but April 2026 monthly files are huge
  (`RUN` ~916 MB ZIP, `PRED` ~227 MB ZIP). Use only if STPASA/PDPASA aggregate is
  not enough.
- Data model source: AEMO Electricity Data Model v5.7 identifies MMS public data,
  Demand Forecasts intermittent tables, and PASA region fields.

## Source Candidates

### 1. STPASA_REGIONSOLUTION — recommended for 72h tail

Archive files exist as:

```text
https://nemweb.com.au/Reports/CURRENT/Short_Term_PASA_Reports/
PUBLIC_STPASA_YYYYMMDDHHMM_*.zip
PUBLIC_ARCHIVE%2523STPASA_REGIONSOLUTION%2523FILE01%2523YYYYMM010000.zip
```

An April 2026 archive sample contained 207,360 rows. For the latest run in that
file, the target intervals covered:

```text
run_time:    2026-04-30 14:00:00+00:00
horizon min: 28.5h
horizon max: 172.0h
```

The same sample included the fields needed for the residual audit and model
feature trial:

```text
RUN_DATETIME
INTERVAL_DATETIME
REGIONID
DEMAND10 / DEMAND50 / DEMAND90
AGGREGATECAPACITYAVAILABLE
AGGREGATEPASAAVAILABILITY
TOTALINTERMITTENTGENERATION
DEMAND_AND_NONSCHEDGEN
UIGF
SEMISCHEDULEDCAPACITY
LOR_SEMISCHEDULEDCAPACITY
SS_SOLAR_UIGF
SS_WIND_UIGF
SS_SOLAR_CAPACITY
SS_WIND_CAPACITY
SS_SOLAR_CLEARED
SS_WIND_CLEARED
```

This is the source that matches the project reason for adding the data: it covers
the period after short pre-dispatch/APF usefulness tails off and still includes
the full 72h horizon.

Implemented parser/backfill:

```bash
./.venv/bin/python ingest/backfill_stpasa_regionsolution.py --start 2026-04 --end 2026-04 --dry-run
```

The live dry run against the April 2026 archive returned:

```text
207,360 SA1 rows
720 STPASA runs
horizon range: 16.5h -> 183.0h
```

Write the parquet by dropping `--dry-run`, then rerun the residual audit. The
audit now joins STPASA automatically when
`data/parquet/aemo_stpasa_regionsolution_sa1.parquet` exists.

Offline residual-correction ablation:

```bash
./.venv/bin/python eval/ablate_stpasa_tail_features.py \
  --since 2026-04-01T00:00:00Z \
  --until 2026-05-01T00:00:00Z \
  --output-prefix eval/results/stpasa_tail_ablation_20260615_apr
```

Held-out validation rows (`28.5-72h`, chronological split):

```text
baseline original MAE:          $35.51/MWh
baseline corrected MAE:         $23.39/MWh
baseline+STPASA corrected MAE:  $18.14/MWh
baseline corrected bias:        +$12.77/MWh
baseline+STPASA corrected bias: +$4.67/MWh
```

Interpretation: STPASA has enough incremental signal to justify a proper
tail-feature experiment. Do not wire live production use until the current
`Short_Term_PASA_Reports` feed freshness is validated.

Current feed note: NEMWeb lists `PUBLIC_STPASA_YYYYMMDDHHMM_*.zip` files under
`Short_Term_PASA_Reports`. The visible filenames are approximately hourly. Before
using it at runtime, validate the listing freshness and the latest file horizon;
archive data is already enough for backtests and feature ablation.

### 2. PDPASA_REGIONSOLUTION — short-horizon overlap only

Live current feed exists:

```text
https://nemweb.com.au/Reports/Current/PDPASA/
PUBLIC_PDPASA_YYYYMMDDHHMM_*.zip
```

One sampled current file, `PUBLIC_PDPASA_202606142300_0000000522475243.zip`,
covered only `0-29h` from its run time:

```text
run_time:    2026-06-14 13:00:00+00:00
horizon min: 0.0h
horizon max: 29.0h
```

It contained `REGIONSOLUTION` with these useful columns:

```text
RUN_DATETIME
INTERVAL_DATETIME
REGIONID
DEMAND10 / DEMAND50 / DEMAND90
AGGREGATECAPACITYAVAILABLE
AGGREGATEPASAAVAILABILITY
TOTALINTERMITTENTGENERATION
DEMAND_AND_NONSCHEDGEN
UIGF
SEMISCHEDULEDCAPACITY
LOR_SEMISCHEDULEDCAPACITY
SS_SOLAR_UIGF
SS_WIND_UIGF
SS_SOLAR_CAPACITY
SS_WIND_CAPACITY
SS_SOLAR_CLEARED
SS_WIND_CLEARED
```

This is useful, but it is not enough for the 72h extrapolation goal. Keep it for:

- overlap checks against pre-dispatch/APF
- validating consistency with STPASA near the `28-29h` handoff
- short-horizon residual diagnostics where the same field family is useful

Backfill note: April 2026 archive sample had `UIGF` but not the newer split fields
in the first rows sampled. The ingest should handle missing split columns and keep
aggregate `UIGF` as the stable baseline.

### 3. INTERMITTENT_DS_RUN / INTERMITTENT_DS_PRED — exact dispatch UIGF, heavy

AEMO data model labels these as Unconstrained Intermittent Generation Forecasts
for Dispatch. Public archive files exist:

```text
PUBLIC_ARCHIVE#INTERMITTENT_DS_RUN#FILE01#YYYYMM010000.zip
PUBLIC_ARCHIVE#INTERMITTENT_DS_PRED#FILE01#YYYYMM010000.zip
```

Sampled April 2026 headers:

```text
INTERMITTENT_DS_RUN:
RUN_DATETIME, DUID, OFFERDATETIME, ORIGIN, FORECAST_PRIORITY, ...

INTERMITTENT_DS_PRED:
RUN_DATETIME, DUID, OFFERDATETIME, INTERVAL_DATETIME, ORIGIN,
FORECAST_PRIORITY, FORECAST_MEAN, FORECAST_POE10, FORECAST_POE50, FORECAST_POE90
```

This is the most exact short-horizon wind/solar forecast source, but the monthly
ZIPs sampled for April 2026 were large:

- `INTERMITTENT_DS_RUN`: ~916 MB ZIP
- `INTERMITTENT_DS_PRED`: ~227 MB ZIP

It also requires DUID-to-region and DUID-to-fuel/technology metadata before it
can produce SA wind/solar aggregates. Use after the lighter region-level PDPASA
path if needed.

### 4. Dispatch_SCADA / INTERMITTENT_GEN_SCADA — actual/proxy side

Current Dispatch SCADA exists:

```text
https://nemweb.com.au/Reports/Current/Dispatch_SCADA/
PUBLIC_DISPATCHSCADA_YYYYMMDDHHMM_*.zip
```

Sampled header:

```text
SETTLEMENTDATE, DUID, SCADAVALUE, LASTCHANGED
```

Archive `INTERMITTENT_GEN_SCADA` exists and contains intermittent unit SCADA
availability records:

```text
RUN_DATETIME, DUID, SCADA_TYPE, SCADA_VALUE, SCADA_QUALITY, LASTCHANGED
```

Observed `SCADA_TYPE` examples: `ELAV`, `LOCL`.

This can support actual availability/generation error features, but it is more
work than PDPASA because it needs DUID metadata and careful type selection.

## Recommended Implementation Path

1. Add archive/backfill script for `STPASA_REGIONSOLUTION` first:
   - monthly DATA archive, `PUBLIC_ARCHIVE#STPASA_REGIONSOLUTION#FILE01#YYYYMM010000.zip`
   - cache under `data/nemseer_cache/stpasa/`
   - parse `I/D` rows for `STPASA,REGIONSOLUTION`
   - filter `REGIONID='SA1'`
   - write `data/parquet/aemo_stpasa_regionsolution_sa1.parquet`
   - columns: `interval_dt`, `run_time`, `uigf`, `ss_solar_uigf`,
     `ss_wind_uigf`, `ss_solar_capacity`, `ss_wind_capacity`,
     `total_intermittent_generation`, `demand50`
2. Validate current/live STPASA source:
   - list `https://nemweb.com.au/Reports/CURRENT/Short_Term_PASA_Reports/`
   - select the latest `PUBLIC_STPASA_*.zip`
   - validate it is fresh, has the same columns, and covers at least `72h`
   - if freshness is unreliable, use archive data for backtests only and keep
     runtime integration blocked until a fresh live source is identified
3. Add optional `ingest/ingest-pdpasa.py` for short-horizon overlap:
   - list `https://nemweb.com.au/Reports/Current/PDPASA/`
   - select latest unseen `PUBLIC_PDPASA_*.zip`
   - parse `I/D` rows for `PDPASA,REGIONSOLUTION`
   - filter `REGIONID='SA1'`
   - write Influx measurement `rp_30m.aemo_pdpasa_regionsolution`
   - fields: `uigf`, `ss_solar_uigf`, `ss_wind_uigf`, `ss_solar_capacity`,
     `ss_wind_capacity`, `total_intermittent_generation`, `demand50`
   - tag: `region`, `run_time`
4. Extend `data/export_parquet.py` if runtime Influx ingestion is added:
   - export `aemo_stpasa_regionsolution_sa1.parquet`
   - optionally export `aemo_pdpasa_regionsolution_sa1.parquet` for overlap
5. Extend `eval/analyze_lgbm_residual_drivers.py`:
   - join latest STPASA row as issued at `forecast_creation_time`
   - add buckets for `uigf`, `ss_wind_uigf`, `ss_solar_uigf`
   - if actual renewable proxy is available, add forecast-error buckets
   - enforce a validation that joined horizons cover `72h`
6. Rerun controlled audit:
   - same window as previous audit first: `2026-04-01T00:00Z -> 2026-05-13T00:00Z`
   - then full current log window after parquet refresh

## Source Notes

- AEMO MMS page says MMS public data is available from NEMWEB and the MMS Data
  Model describes packages/tables.
- AEMO Electricity Data Model Package Summary v5.7 lists Demand Forecasts as
  containing regional demand forecasts and intermittent generation forecasts.
- The same summary lists `INTERMITTENT_DS_PRED` as Dispatch UIGF and
  `INTERMITTENT_GEN_FCST_P5_PRED` / `INTERMITTENT_GEN_FCST_PRED` for wind/solar
  forecast predictions.
- AEMO Electricity Data Model Report v5.7 documents `SS_UIGF`,
  `SS_SOLAR_UIGF`, and `SS_WIND_UIGF` as aggregate semi-scheduled UIGF fields
  in PASA region solution output.
