# PEC-MI transition data capture

Status: capture and aggregate timers enabled 2026-09-24 ACST; latest canary failed and was sent to
the existing shared Healthchecks check.

## What is retained

`ingest/capture_aemo_transition_reports.py` saves original response bytes below
`data/aemo_transition/raw/` and writes one `.meta.json` sidecar per response. Sidecars keep
capture time, filename-derived report run time, HTTP `Last-Modified`, source URL, response
headers, byte count, and SHA-256. The 5MIN visualisations API response is stored as received,
with its POST payload in the sidecar.

The capture starts at `2026-09-22T00:00:00Z` by default. Change
`AEMO_TRANSITION_CAPTURE_FROM_UTC` to extend or narrow the initial report-file backfill. It polls
every five minutes and archives all matching current files since that time:

| Product | Current folder / endpoint | Expected cadence | Interconnector results |
|---|---|---:|---|
| DispatchIS | `DispatchIS_Reports` | 5 min | `REGIONSUM`, `INTERCONNECTORRES` |
| Legacy dispatch | `Dispatch_Reports` | 5 min | `DREGION`, `DINT` |
| P5MIN | `P5_Reports` | 5 min | `REGIONSOLUTION`, `INTERCONNECTORSOLN` |
| PREDISPATCHIS | `PredispatchIS_Reports` | 30 min | `REGION_SOLUTION`, `INTERCONNECTOR_SOLN` |
| Legacy PREDISPATCH | `Predispatch_Reports` | 30 min | `PDREGION`, `PDINT` |
| STPASA | `Short_Term_PASA_Reports` | hourly | `REGIONSOLUTION`, `INTERCONNECTORSOLN` (capacity flow) |
| Seven Day Outlook | `SEVENDAYOUTLOOK_FULL` | 30 min | `PEAK` regional rows |
| Visualisations API | `POST /aemo/apps/api/report/5MIN`, `timeScale=["30MIN"]` | 5 min | regional rows |

The AEMO Electricity Data Model identifies the public dispatch, P5MIN, and PREDISPATCH
interconnector result tables. The data model defines positive flow from an interconnector's
`FROMREGION`. Seven Day Outlook is used for regional demand and interchange, not as an
interconnector-result source. See the [AEMO MMS data model](https://nemweb.com.au/Reports/Current/MMSDataModelReport/Electricity/Electricity%20Data%20Model%20Report_files/Elec58.htm),
[dispatch reports](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-management-system-mms-data/dispatch),
and [pre-dispatch reports](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-management-system-mms-data/pre-dispatch).

## Canary and alerting

The canary writes `data/aemo_transition/canary_latest.json` and
`data/aemo_transition/canary_state.json`. It checks report freshness, ZIP/CSV readability,
expected regional and interconnector tables, SA1/VIC1/NSW1 rows, required flow/interchange
columns, row widths, duplicate keys, and schema-version/header changes. It records the first
report run containing `NSW1-SA1`. API and Seven Day Outlook data are also compared at their
first non-overlapping interval; a gap above 30 hours or a regional interchange jump above
3,000 MW fails the canary. For ZIP products, it checks the latest report gap and the median
cadence over the latest 13 distinct run times. It records the maximum target horizon by source
and alerts if that horizon changes by more than 30 minutes between captures.
When a new report advances its run time but keeps the same forecast end (within one minute),
the shorter remaining horizon is expected and does not alert. This occurred in the 2026-09-25
10:00 NEM-time STPASA report: the remaining horizon moved from 163 to 162 hours while both
reports ended at `2026-10-01T18:00:00Z`.

The capture records its result in ignored `data/healthcheck_status/`. The repository's
`ai-energy-healthcheck-aggregate.timer` evaluates that status with the load service and price
listener, then sends success or `/fail` to the one existing `HC_PREDICT_URL`. A success from one
job cannot clear another job's failure. Capture freshness is enforced locally at five minutes
plus three minutes; see [`docs/healthchecks.md`](../healthchecks.md).

## Initial live observation

The live captures on 2026-09-23 found `NSW1-SA1` in STPASA
`INTERCONNECTORSOLN` rows in a report run before cutover. The earliest retained run was
`2026-09-23T03:00:00Z`; its forecast target range starts at
`2026-09-24T18:30:00Z` (2026-09-25 04:30 NEM time). This is a future PASA record and does not
show that NEMDE had already switched its dispatch topology.

That STPASA table reports `CAPACITYMWFLOW`, calculated import/export limits, and associated
constraint IDs. It does not have a `MWFLOW` field, so treat its `NSW1-SA1` entry as capacity
evaluation data, not as a forecasted cleared flow. DispatchIS and PREDISPATCHIS raw downloads
contain `MWFLOW` tables, but their latest reports do not yet list `NSW1-SA1`. The first capture
also found a 2,554 MW VIC interchange change between the API and Seven Day source across their
first non-overlapping half-hour before cutover; the alert threshold is 3,000 MW to preserve that
baseline without failing each capture. DispatchIS uses `REGIONSUM` and `INTERCONNECTORRES`;
legacy dispatch uses `DREGION`/`DINT`; P5MIN uses `REGIONSOLUTION`/`INTERCONNECTORSOLN`;
PREDISPATCHIS uses `REGION_SOLUTION`/`INTERCONNECTOR_SOLN`. Legacy PREDISPATCH rows place
`PDREGION` in the package field and call the target-time field `PERIODID`; the canary accepts
both. The observed report cadences are 5 minutes for dispatch and P5MIN, 30 minutes for both
PREDISPATCH products and Seven Day Outlook, and 60 minutes for STPASA. The latest report horizons
were 0 hours for dispatch, 0.92 hours for P5MIN, 34.5 hours for PREDISPATCHIS, 35 hours for
legacy PREDISPATCH, 175.41 hours for Seven Day Outlook, and 180 hours for STPASA. The API
horizon was 34.38 hours at capture time.

At `2026-09-23T07:00:16Z`, the archive had 2,688 raw and sidecar files (187.5 MB) from the
2026-09-22 backfill start. At the observed rate this is about 4.4 GB per month; the machine had
about 2.1 TB free.

## Follow-up live canary — 2026-09-24 ACST

The `ai-energy-transition-capture.service` run at `2026-09-23T23:39:25Z` archived the current
reports and marked the canary failed. The aggregate service completed successfully and sent the
failure to the one configured Healthchecks check; `ai-energy-healthcheck-aggregate.timer` runs
once per minute and `ai-energy-transition-capture.timer` every five minutes.

Observed issues:

- Visualisations API horizon shortened from 34.38 to 17.84 hours; PREDISPATCHIS shortened from
  34.50 to 18 hours and legacy PREDISPATCH from 35 to 18 hours. STPASA shortened from 180 to 163
  hours and Seven Day Outlook from 175.41 to 162.45 hours. The reason is unconfirmed.
- The first non-overlapping VIC1 API/Seven-Day interchange difference was -3,258 MW, above the
  3,000 MW canary threshold.
- Six older report downloads returned HTTP 403 after three attempts (two legacy dispatch files
  and four P5MIN files dated 2026-09-23 UTC). Whether these files are permanently unavailable or
  the denial is transient remains unknown; their paths remain in the canary issue list.
- The current STPASA report still contains `NSW1-SA1`; this remains a forecast capacity result,
  not evidence that NEMDE has switched its dispatch topology.

A later capture at `2026-09-24T23:42:18Z` still failed: Seven Day Outlook's horizon changed from
162.95 to 162.44 hours, and the VIC1 API/Seven-Day interchange jump was -3,011 MW. The horizon
change is just over the configured 30-minute threshold; both values are checked against the
preceding capture. Their cause is unconfirmed.

These are observations, not proof that PEC-MI caused the horizon or interchange changes. Keep
the canary failure active until the current data recovers or the individual condition is reviewed.

## ST-PASA timing update — 2026-09-24

WattClarity's 2026-09-24 review reports that ST-PASA now includes `NSW1-SA1` within the short-term
window, seven days before its 2026-10-01 effective dispatch date. The current outlook reached
2026-10-02 04:00; its capacity flow was constrained to 0 MW until about midday on 2026-10-01,
then became non-zero as the `NS_` / `SN_ZERO` constraints were lifted. The article also describes
different capacity results by regional LOR study, including about 150 MW into SA in the SA study.
This is forecast capacity assessment, not cleared dispatch flow, and explains why a pre-cutover
STPASA report can already contain the new interconnector. See
[WattClarity's ST-PASA review](https://wattclarity.com.au/articles/2026/09/pec-stage-2-enters-the-st-timeframe/).

Install and enable the capture and aggregate timers. Set the existing `HC_PREDICT_URL` in `.env`;
no per-job Healthchecks records or schedule edits are needed:

```bash
sudo cp systemd/ai-energy-transition-capture.service systemd/ai-energy-transition-capture.timer /etc/systemd/system/
sudo cp systemd/ai-energy-healthcheck-aggregate.service systemd/ai-energy-healthcheck-aggregate.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-energy-transition-capture.timer
sudo systemctl enable --now ai-energy-healthcheck-aggregate.timer
```

Inspect operation and disk use:

```bash
systemctl list-timers ai-energy-transition-capture.timer
journalctl -u ai-energy-transition-capture.service --since today
cat data/aemo_transition/canary_latest.json
du -sh data/aemo_transition
df -h .
```

The currently observed InfluxDB retention policies keep parsed `rp_5m` data for 26,280 hours
(three years) and `rp_30m` data for 89,040 hours (about 10.2 years). Neither retains the
as-issued ZIP/JSON responses. The current disk had about 2.1 TB free on 2026-09-23. AEMO
PREDISPATCHIS files observed in September were roughly 0.5–1.2 MB each, so include raw archive
growth in ordinary disk monitoring.

## Production boundary

The active price bundle is `initial-root-20260810T142151Z` and its manifest validates. The
production feature contract remains the 20 features listed in `config.yaml`; this capture adds no
model feature and does not retrain or promote a model. The weekly candidate bundle dated
2026-09-21 remains separate from the active bundle.

The report parser is for canary summaries only. Keep the ZIP/JSON bytes as the record of truth;
add generic parsed interconnector storage only after confirming post-cutover report schemas.
