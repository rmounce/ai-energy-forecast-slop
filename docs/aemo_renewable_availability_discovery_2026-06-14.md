# AEMO Renewable Availability Discovery — 2026-06-14

## Caveman Summary

- Goal: find a better renewable/wind availability source for LGBM price residual
  decomposition before adding model features.
- Best first target: `PDPASA_REGIONSOLUTION`.
- Why: public current feed, small ZIPs, 30-minute cadence, 30-minute intervals,
  region-level `UIGF`, `SS_SOLAR_UIGF`, `SS_WIND_UIGF`, `SS_SOLAR_CAPACITY`,
  `SS_WIND_CAPACITY`, and related semi-scheduled fields.
- Current path:
  `https://nemweb.com.au/Reports/Current/PDPASA/`
- Archive path:
  `https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/YYYY/MMSDM_YYYY_MM/MMSDM_Historical_Data_SQLLoader/DATA/`
- Archive file example:
  `PUBLIC_ARCHIVE#PDPASA_REGIONSOLUTION#FILE01#202604010000.zip`
- Useful fallback: `STPASA_REGIONSOLUTION` for longer horizon/backfill.
- Useful actual/proxy: `Dispatch_SCADA` current feed and archive `INTERMITTENT_GEN_SCADA`
  / dispatch SCADA unit output, but those are DUID-level and need unit metadata to
  aggregate SA wind/solar cleanly.
- Not first target: `INTERMITTENT_DS_RUN/PRED`.
  It is exact dispatch UIGF by DUID but April 2026 monthly files are huge
  (`RUN` ~916 MB ZIP, `PRED` ~227 MB ZIP). Use only if PDPASA aggregate is
  not enough.
- Data model source: AEMO Electricity Data Model v5.7 identifies MMS public data,
  Demand Forecasts intermittent tables, and PASA region fields.

## Source Candidates

### 1. PDPASA_REGIONSOLUTION — recommended

Live current feed exists:

```text
https://nemweb.com.au/Reports/Current/PDPASA/
PUBLIC_PDPASA_YYYYMMDDHHMM_*.zip
```

One sampled current file, `PUBLIC_PDPASA_202606142230_0000000522472211.zip`,
contained `REGIONSOLUTION` with these useful columns:

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

This is enough for the residual audit:

- `ss_uigf_error = forecast UIGF - actual/realised semi-scheduled generation proxy`
- `ss_wind_uigf`, `ss_solar_uigf` regime buckets
- region-level renewable availability buckets without DUID metadata
- direct join semantics: latest `RUN_DATETIME <= forecast_creation_time`,
  same `INTERVAL_DATETIME`, `REGIONID='SA1'`

Backfill note: April 2026 archive sample had `UIGF` but not the newer split fields
in the first rows sampled. The ingest should handle missing split columns and keep
aggregate `UIGF` as the stable baseline.

### 2. STPASA_REGIONSOLUTION — longer-horizon companion

Archive files exist as:

```text
PUBLIC_ARCHIVE#STPASA_REGIONSOLUTION#FILE01#YYYYMM010000.zip
```

The sampled April 2026 file had the same older region-solution field family through
`UIGF`, but not the current PDPASA wind/solar split in sampled rows. This is still
useful if the residual audit needs a 1-6 day lookahead companion to PDPASA.

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

1. Add `ingest/ingest-pdpasa.py` for current feed:
   - list `https://nemweb.com.au/Reports/Current/PDPASA/`
   - select latest unseen `PUBLIC_PDPASA_*.zip`
   - parse `I/D` rows for `PDPASA,REGIONSOLUTION`
   - filter `REGIONID='SA1'`
   - write Influx measurement `rp_30m.aemo_pdpasa_regionsolution`
   - fields: `uigf`, `ss_solar_uigf`, `ss_wind_uigf`, `ss_solar_capacity`,
     `ss_wind_capacity`, `total_intermittent_generation`, `demand50`
   - tag: `region`, `run_time`
2. Add archive/backfill script:
   - monthly DATA archive, `PUBLIC_ARCHIVE#PDPASA_REGIONSOLUTION#FILE01#YYYYMM010000.zip`
   - cache under `data/nemseer_cache/pdpasa/`
   - write `data/parquet/aemo_pdpasa_regionsolution_sa1.parquet`
3. Extend `data/export_parquet.py`:
   - export `aemo_pdpasa_regionsolution_sa1.parquet`
   - columns: `interval_dt`, `run_time`, fields above
4. Extend `eval/analyze_lgbm_residual_drivers.py`:
   - join latest PDPASA row as issued at `forecast_creation_time`
   - add buckets for `uigf`, `ss_wind_uigf`, `ss_solar_uigf`
   - if actual renewable proxy is available, add forecast-error buckets
5. Rerun controlled audit:
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
