# HA Frontend Entity Cleanup

Date: 2026-05-11

This note records the production Lovelace references found under
`/opt/dockerfiles/hass/config/.storage/` and the recommended cleanup path. The
frontend dashboard is edited through Home Assistant, so this repo should treat
the `.storage` files as read-only reference material.

## Current Relevant Dashboard References

Production dashboard file:

- `/opt/dockerfiles/hass/config/.storage/lovelace.dashboard_battery`

Relevant chart groups:

- Around line 1019: `Spot Price Comparison`
  - Amber billing interval spot:
    `sensor.amber_billing_interval_forecasts_general_price`
  - APF/LGBM triplet:
    `sensor.ai_price_forecast_low`
    `sensor.ai_price_forecast`
    `sensor.ai_price_forecast_high`
  - PD-direct triplet:
    `sensor.ai_pd_direct_price_forecast_low`
    `sensor.ai_pd_direct_price_forecast`
    `sensor.ai_pd_direct_price_forecast_high`

- Around line 2703: `Predict Comparison Temp`
  - Amber billing interval spot:
    `sensor.amber_billing_interval_forecasts_general_price`
  - APF/LGBM low/high:
    `sensor.ai_price_forecast_low`
    `sensor.ai_price_forecast_high`
  - stitched current-best AI spot:
    `sensor.ai_spot_price_forecast`
  - TFT triplet:
    `sensor.ai_tft_price_forecast_low`
    `sensor.ai_tft_price_forecast`
    `sensor.ai_tft_price_forecast_high`

- Around line 2846: `Battery, Price & Cost Forecast`
  - AEMO decoder/debug:
    `sensor.ai_aemo_price_forecast`
  - Amber billing interval spot:
    `sensor.amber_billing_interval_forecasts_general_price`
  - APF/LGBM:
    `sensor.ai_price_forecast`
  - TFT:
    `sensor.ai_tft_price_forecast`

## Recommended Frontend Shape

For day-to-day use, keep one primary raw-wholesale comparison chart:

- Amber observed/forecast spot yardstick:
  `sensor.amber_billing_interval_forecasts_general_price`
- current-best Amber-independent AI spot:
  `sensor.ai_spot_price_forecast`
- optional legacy incumbent yardstick:
  `sensor.ai_price_forecast`

Hide or move to a diagnostics-only view:

- TFT triplet:
  `sensor.ai_tft_price_forecast(_low/_high)`
- PD-direct triplet:
  `sensor.ai_pd_direct_price_forecast(_low/_high)`
- AEMO decoder/debug:
  `sensor.ai_aemo_price_forecast`

Reasoning:

- `sensor.ai_spot_price_forecast` is now the graph-friendly stitched source:
  fresh 5-minute Tier 1 wholesale forecast followed by the 30-minute PD-direct
  tail.
- The canonical EMHASS/control surfaces are separate:
  `sensor.ai_mpc_import_price_forecast`, `sensor.ai_mpc_export_price_forecast`,
  `sensor.ai_dh_import_price_forecast`, and
  `sensor.ai_dh_export_price_forecast`.
- Per-model triplets are useful diagnostics, but keeping all of them on the
  primary dashboard makes the live system harder to read.

## Repo-Side Template Fix

`hass/package-emhass.yaml` has been updated so `sensor.ai_spot_price_forecast`
reads the current `sensor.ai_p5min_price_forecast` schema:

- preferred field: `wholesale_price`
- fallback field: `aemo_price_sa1`

The fallback is only for compatibility with older published state. The live
publisher now emits `wholesale_price`.

After syncing the package to production and restarting/reloading HA templates,
the existing `Predict Comparison Temp` chart should continue to work with the
new schema.
