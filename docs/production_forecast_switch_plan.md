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

The AI combined publisher currently emits Amber-shaped compatibility sensors:

- `sensor.ai_combined_general_price_forecast`
- `sensor.ai_combined_feed_in_price_forecast`

Those are useful for EMHASS compatibility but retain Amber's confusing feed-in
sign convention at the boundary.

## Target Source Set

Keep both forecast representations:

1. Legacy / compatibility sensors
   - Amber-shaped `Forecasts[]`
   - feed-in earning encoded as negative
   - used for compatibility with current templates

2. Canonical HAEO-style sensors
   - `forecast` attribute with `{datetime, native_value}` points
   - import price positive cost
   - export price positive revenue
   - UTC ISO timestamps
   - suitable for HAEO/HAFO-style consumers and less error-prone templates

Suggested canonical entities:

- `sensor.ai_mpc_import_price_forecast`
- `sensor.ai_mpc_export_price_forecast`
- `sensor.ai_dh_import_price_forecast`
- `sensor.ai_dh_export_price_forecast`

## Switching Model

Add explicit Home Assistant selectors rather than editing templates for each
trial:

- `input_select.emhass_mpc_price_source`
  - `amber`
  - `ai_shadow`
- `input_select.emhass_dh_price_source`
  - `amber_lgbm_extrapolated`
  - `ai_shadow`

The EMHASS request templates should read from adapter template sensors or
macros whose only job is to select the active source and normalize it into the
price arrays EMHASS expects.

Do not embed source-specific sign or timestamp conventions in the main MPC/DH
payload templates.

## Rollout Sequence

1. Publish canonical AI forecast sensors alongside the existing Amber-shaped AI
   combined sensors.
2. Add read-only diagnostic template sensors that render the selected MPC/DH
   import/export arrays without calling EMHASS.
3. Add source selectors and adapter macros/template sensors.
4. Run shadow mode:
   - selected source remains Amber
   - AI arrays are rendered and logged
   - no inverter behavior changes
5. Switch DH first, if desired, because it changes the strategic plan but not
   the immediate 5-minute action as directly as MPC.
6. Switch MPC only after source freshness, sign, length, and unit checks are
   visible and stable.
7. Keep one-action rollback by setting the selectors back to the legacy source.

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
