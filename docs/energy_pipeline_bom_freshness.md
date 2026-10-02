# BOM hourly freshness: staged integration patch

Status: patch prepared/dry-checked, NOT applied. Production HA integration untouched.
2026-10-02 Adelaide. Parent: [freshness checkpoint](energy_pipeline_source_cache.md).

## Confirmed contract

- Existing `bureau_of_meteorology/PyBoM/collector.py` fetches daily/hourly products separately;
  successful hourly fetch stores internal `_cache['hourly_forecasts']['timestamp']`.
  Cached fallback retains that timestamp. Weather forecast service currently strips metadata.
- `sensor.py` exposes daily metadata only. Daily issue/observation clocks cannot prove hourly
  freshness: independent HTTP failures/cached fallbacks can affect each product separately.
- One read-only native hourly API probe returned aware `metadata.issue_time` and
  `metadata.response_timestamp` (2026-10-02T04:47:08Z /04:56:06Z). Provider issue is distinct from
  local successful fetch and a response timestamp. No coordinates/raw forecasts recorded here.
- The inspected AEMO short-term JSON has only `5MIN`; rows have settlement target and market
  fields but no run/issue/update clock. Its HTTP creation timestamp remains retrieval evidence,
  not provider issuance. No new acquisition owner or recurring direct BOM request added.

## Prepared change

- [bom_hourly_freshness.patch](../hass/integration_patches/bom_hourly_freshness.patch) adds two
  attributes to `WeatherHourly`: `hourly_forecast_issue_time` from cached hourly metadata and
  aware UTC `hourly_forecast_last_fetched` from its successful-fetch clock. Missing data →null.
- Property reads existing collector data only; no fetching, new entities, cache reset or retries.
  Cached fallback cannot claim a fresh clock. Forecast arrays and observation attributes unchanged.
- Based on installed `custom_components/bureau_of_meteorology/weather.py` SHA256:
  `995f92b3a275fd9f910c073984c4b97a59c9e4e36295d388e380ee658e943240`.
- `patch --dry-run -p1 -d /opt/dockerfiles/hass/config < hass/integration_patches/bom_hourly_freshness.patch`
  passes. Patched source parses; extracted actual property checked for UTC conversion, cached-clock
  preservation, advancement on successful fetch and null markers before initialization.
- Source Python patch must take effect in HA's loaded integration before claiming live evidence;
  config-entry reload alone does not establish module reload. Follow [HA recovery/reload guidance](ha_hot_reload.md)
  for any controlled deployment. No reload/restart performed in this work.

## Resident capture adapter

- `energy_pipeline/weather.py` reads hourly markers before/after the existing HA forecast service.
  Changed markers discard the capture and retry once; a second change fails refresh, retaining
  the previous cached frame/evidence/time. Failed state reads also fail refresh.
- Stable aware timestamps normalize to UTC; absent/null/malformed/naive values remain explicit
  unknown issue/fetch evidence. No capture clock substituted. Production forecast path unchanged.
- Clocks are labelled `hourly_issue_observed` / `hourly_successful_fetch_observed`: these separate
  HA reads detect observed races, not an atomic collector snapshot. Collector-to-entity update lag
  remains unverified. Validate actual attribute/array lifecycle after controlled deployment before
  treating these markers as an admission contract.
- Seven adapter checks cover stable clocks, retry/discard, repeated races, missing/invalid clocks,
  failed reads; focused pipeline regression suite: 133 passed.
- Live unpatched HA shadow, 2026-10-02 16:05 Adelaide: weather capture 0.216s, 145 weather points;
  both provider clocks explicitly unknown. Accepted 144-point quantile forecasts, 3 model loads,
  RSS 1129.4 MiB; acceptance 0.082s, exit 0. No publications/reloads/restarts. Does not validate
  patched integration lifecycle or sustained memory behaviour.
- Capture issue age and successful-fetch age separately. Choose admission budgets from observed
  cadence/coverage; no new source thresholds/pages introduced by the staged patch.
