# HA Frontend Entity Cleanup

Date: 2026-05-11

This note records the production Lovelace references found under
`/opt/dockerfiles/hass/config/.storage/` and the recommended cleanup path. The
frontend dashboard is edited through Home Assistant, so this repo should treat
the `.storage` files as read-only reference material.

## 2026-06-15 Archive Update

The live dashboard snapshot was archived before removing suspended price/load
shadow surfaces from the active repo-side HA package:

- `docs/archive/suspended_price_paths/lovelace.dashboard_battery_2026-06-15.json`

The dashboard still referenced `sensor.ai_spot_price_forecast(_low/_high)`,
`sensor.ai_aemo_price_forecast`, and `sensor.ai_tft_load_forecast(_low/_high)`
at archive time. Those sources are now expected to go stale unless deliberately
revived, so active dashboards should remove them and keep only production
APF/LightGBM price plus LightGBM load forecast surfaces.

## Historical Dashboard References

Production dashboard file:

- `/opt/dockerfiles/hass/config/.storage/lovelace.dashboard_battery`

Relevant chart groups after the 2026-05-11 frontend cleanup:

- Around line 1019: `Spot Price Comparison`
  - Amber billing interval spot:
    `sensor.amber_billing_interval_forecasts_general_price`
  - Raw upstream AEMO stitched:
    `sensor.ai_aemo_price_forecast`
  - Current-best AI stitched spot:
    `sensor.ai_spot_price_forecast`
  - APF/LGBM triplet:
    `sensor.ai_price_forecast_low`
    `sensor.ai_price_forecast`
    `sensor.ai_price_forecast_high`
  - PD-direct triplet:
    `sensor.ai_pd_direct_price_forecast_low`
    `sensor.ai_pd_direct_price_forecast`
    `sensor.ai_pd_direct_price_forecast_high`

- Around line 1336: `Amber Price Detail`
  - Amber import/export fields from:
    `sensor.amber_billing_interval_forecasts_general_price`
    `sensor.amber_billing_interval_forecasts_feed_in_price`

## Archived 2026-05-11 Frontend Shape

This was the recommended shape while the tactical/PD-direct shadow stack was
active. It is retained for context only; as of 2026-06-15 the active dashboard
should not depend on these archived sensors.

- Amber observed/forecast spot yardstick:
  `sensor.amber_billing_interval_forecasts_general_price`
- direct raw upstream AEMO stitched yardstick:
  `sensor.ai_aemo_price_forecast`
- current-best Amber-independent AI spot triplet:
  `sensor.ai_spot_price_forecast_low`
  `sensor.ai_spot_price_forecast`
  `sensor.ai_spot_price_forecast_high`
- optional legacy incumbent yardstick:
  `sensor.ai_price_forecast`

Important convention detail: Amber `Forecasts[].spot_per_kwh` is not directly
the same unit as the direct raw AEMO entities. Live comparison on 2026-05-11
showed Amber `spot_per_kwh / sensor.ai_aemo_price_forecast.wholesale_price`
at about `1.10` on the PREDISPATCH leg. For a raw-wholesale chart, divide the
Amber `spot_per_kwh` series by the current `amber_api_scaling_factor` (`1.1` at
the time of writing). Otherwise the Amber line will appear about 10% higher than
the direct AEMO line even when the underlying source prices agree.

Archived diagnostics from that period:

- TFT triplet:
  `sensor.ai_tft_price_forecast(_low/_high)`
- PD-direct triplet:
  `sensor.ai_pd_direct_price_forecast(_low/_high)`
Reasoning:

- `sensor.ai_spot_price_forecast(_low/_high)` was the graph-friendly stitched
  triplet: fresh 5-minute Tier 1 wholesale forecast followed by the 30-minute
  PD-direct tail.
- `sensor.ai_aemo_price_forecast` was the matching model-free stitched upstream
  AEMO surface: raw P5MIN, raw PREDISPATCH, then raw PD7Day where available.
- The now-archived canonical EMHASS/control surfaces were separate:
  `sensor.ai_mpc_import_price_forecast`, `sensor.ai_mpc_export_price_forecast`,
  `sensor.ai_dh_import_price_forecast`, and
  `sensor.ai_dh_export_price_forecast`.
- Per-model triplets are useful diagnostics, but keeping all of them on the
  primary dashboard makes the live system harder to read.

## Historical Repo-Side Template Fix

`hass/packages/emhass.yaml` was updated so `sensor.ai_spot_price_forecast`
reads the current `sensor.ai_p5min_price_forecast` schema:

- preferred field: `wholesale_price`
- fallback field: `aemo_price_sa1`

The fallback is only for compatibility with older published state. The live
publisher now emits `wholesale_price`.

The same entity later gained a direct publisher fallback in `forecast.py` via the
raw stitched AEMO path, so graph comparison does not depend on HA template
stitching for that upstream yardstick.
