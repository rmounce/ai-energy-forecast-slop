# PEC-MI transition data capture

Status: implementation prepared 2026-09-23; systemd timer is not enabled yet.

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

Use one `HC_REPO_PING_KEY` project Ping Key in `.env` for Healthchecks across the repo. Each job
uses a unique slug: this capture uses `aemo-pec-mi-transition`, while the load service and price
listener use `predict-load` and `price-listener`. These slugs have independent check states, so a
success from one job cannot clear another job's failure. The `?create=1` option auto-creates each
check on its first ping; no per-job UUID needs to be copied into `.env`. Without the project key,
this capture only records failures in the systemd journal.

Healthchecks auto-created checks start with a one-day period and one-hour grace period. After the
capture's first ping creates its slug, set its expected period to five minutes with a suitable
grace time so a stopped timer is detected promptly. The existing `HC_PREDICT_URL` setting remains
a transition fallback for the load service and listener only; while they use that one URL, their
success pings can mask one another. Set `HC_REPO_PING_KEY` to move them to their separate slugs.

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

Install and enable after putting `HC_REPO_PING_KEY` in `.env` and adjusting the auto-created
capture check's period and grace time:

```bash
sudo cp systemd/ai-energy-transition-capture.service systemd/ai-energy-transition-capture.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ai-energy-transition-capture.timer
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
