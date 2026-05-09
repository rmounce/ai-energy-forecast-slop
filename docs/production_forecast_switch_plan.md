# Production Forecast Switch Plan

Goal: make the new AI shadow forecast source deployable without changing the
existing EMHASS/Sigenergy control path.

## Current Production Shape

Current control remains:

```text
price forecasts -> EMHASS DH/MPC -> Home Assistant automations -> inverter
```

The existing forecast sources are:

- MPC: Amber APF / Amber extended 5-minute forecast sensors.
- DH: 30-minute Amber/APF-derived price forecast with the existing
  LightGBM-extrapolated day-ahead path.

The old AI combined Amber-shaped compatibility sensors were removed on
2026-05-08. They were useful as a transitional EMHASS adapter, but they retained
Amber's confusing negative feed-in convention and duplicated the clearer
HAEO-style import/export surfaces.

## Target Source Set

Keep two forecast representations, with different risk surfaces:

1. Per-model chart/comparison sensors
   - `forecasts` attribute with model-native fields such as `wholesale_price`
   - useful for ApexCharts and side-by-side model review
   - examples: incumbent APF/LGBM, TFT, PD-direct, stitched spot forecast

2. Canonical HAEO-style import/export sensors
   - `forecast` attribute with `{datetime, native_value}` points
   - import price positive cost
   - export price positive revenue
   - UTC ISO timestamps
   - suitable for a future deliberate EMHASS source switch

Legacy Amber provider sensors still exist as inputs/rollback:

   - Amber-shaped `Forecasts[]`
   - feed-in earning encoded as negative
   - used by the current production EMHASS templates

Suggested canonical entities:

- `sensor.ai_mpc_import_price_forecast`
- `sensor.ai_mpc_export_price_forecast`
- `sensor.ai_dh_import_price_forecast`
- `sensor.ai_dh_export_price_forecast`

Implementation status as of 2026-05-09:

- `forecast.py --publish-hass` now publishes these four canonical AI price
  sensors. The old Amber-shaped `sensor.ai_combined_*` AI sensors are retired.
- MPC canonical sensors now use the current Amber-independent shadow stack:
  Tier 1 tactical LightGBM for the first 60 minutes, then PD-direct expanded to
  5-minute cadence, truncated to the 14-hour MPC horizon. If PD-direct fails to
  build, the publisher falls back to the older TFT Tier 2 bundle.
- DH canonical sensors now use the PD-direct 30-minute / 72-hour price forecast,
  with the same TFT fallback on PD-direct failure.
- PD-direct, now with the trained PD7Day q50 debiaser, publishes as per-model
  chart/comparison triplets:
  - `sensor.ai_pd_direct_price_forecast`
  - `sensor.ai_pd_direct_price_forecast_low`
  - `sensor.ai_pd_direct_price_forecast_high`
- `sensor.ai_spot_price_forecast` is a graph-friendly stitched wholesale source:
  Tier 1 5-minute spot/wholesale forecast followed by the PD-direct 30-minute
  tail.
- All four canonical sensors use HAEO-style `forecast` points with UTC
  `datetime` and positive economic `native_value` prices.
- `hass/package-emhass.yaml` declares source selectors, read-only status
  sensors, and diagnostic sensors for both MPC and DH price sources.
- The selectors currently expose only production legacy options:
  - MPC: `amber`
  - DH: `amber_lgbm_extrapolated`
- Guarded `ai_shadow` template branches exist, but the `ai_shadow` option was
  removed from the selectors to avoid accidental control routing before a
  deliberate promotion decision.
- `sensor.emhass_mpc_price_diagnostic` and `sensor.emhass_dh_price_diagnostic`
  expose side-by-side first values and 1h/24h means for both sources without
  calling EMHASS. State = currently selected source.

## Switching Model

When promoting an AI source to controllable shadow/prod, re-add explicit Home
Assistant selector options rather than editing templates for each trial:

- `input_select.emhass_mpc_price_source`
  - `amber`
  - `ai_shadow`
- `input_select.emhass_dh_price_source`
  - `amber_lgbm_extrapolated`
  - `ai_shadow`

The EMHASS request templates should read from adapter template sensors or macros
whose only job is to select the active source and normalize it into the price
arrays EMHASS expects.

Do not embed source-specific sign or timestamp conventions in the main MPC/DH
payload templates.

## Rollout Sequence

1. Publish canonical AI forecast sensors. **Done in `forecast.py`; verify against
   live HA state.**
2. Add source selectors and AI forecast health/status sensors. **Done in
   `hass/package-emhass.yaml`; selectors are currently legacy-only by design.**
3. Add read-only diagnostic template sensors. **Done: `sensor.emhass_mpc_price_diagnostic`
   and `sensor.emhass_dh_price_diagnostic` expose side-by-side first values and hourly/daily
   means for both sources.**
4. Add adapter logic to the EMHASS REST payload templates. **Done, but the AI
   branch is unreachable until an AI option is deliberately re-added.**
5. Run visual shadow mode:
   - selected source remains Amber
   - AI arrays are rendered and logged
   - no inverter behavior changes
6. Re-add an AI selector option only after the canonical control entities have
   been observed live with PD-direct content and all status checks pass.
7. Switch DH first, if desired, because it changes the strategic plan but not
   the immediate 5-minute action as directly as MPC.
8. Switch MPC only after source freshness, sign, length, and unit checks are
   visible and stable.
9. Keep one-action rollback by setting the selectors back to the legacy source.

## Acceptance Checks

Before enabling AI as a live source:

- Forecast arrays cover the full EMHASS horizon.
- All timestamps are timezone-aware UTC at the publisher boundary.
- Import/export prices are in `$ / kWh`.
- Export price is positive revenue, except when exporting costs money.
- The current interval is present or intentionally filled.
- The selected-source arrays can be inspected before submission to EMHASS.
- Missing/stale AI forecasts fall back to the legacy source.
- Amber remains available as a yardstick and rollback path.

## Non-Goals For Production v0

- No custom MPC controller.
- No state-value sidecar control hook.
- No dispatch-shape hook.
- No grid-exchange gate.
- No broad heuristic sell/export shaping.

Those branches informed the risk model, but production v0 should be a
forecast-source deployment into the existing EMHASS control path.
