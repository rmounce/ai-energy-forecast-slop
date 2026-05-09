# Repository Conventions

This project uses one canonical internal representation and keeps external
provider conventions at the boundary.

## Time

- Internal timestamps should be timezone-aware UTC.
- Persisted timestamps should use ISO-8601 UTC where practical.
- Local time should be used only for:
  - tariff schedule lookup
  - provider-specific API parsing
  - user-facing Home Assistant display/templates
  - model features that were explicitly trained in a named local timezone
- New Python code should normalize tabular time indexes through
  `tariff_utils.ensure_utc_index()` or an equivalent explicit UTC conversion.

Existing exceptions are intentional when they match model training contracts:

- Tier 1 tactical price time features use `Australia/Brisbane`.
- TFT price time features use `Australia/Brisbane`.
- TFT load time features use `Australia/Adelaide`.
- Tariff lookup uses the configured site timezone, currently
  `Australia/Adelaide`.

## Price Units

- Internal wholesale forecasts are either clearly named `$ / kWh`
  (`wholesale_price`) or `$ / MWh` (`*_mwh`).
- Internal tariffed import/export prices should use positive economic values:
  - `general_price`: import cost, positive means importing costs money.
  - `feed_in_price`: export value, positive means exporting earns money.
- Simulation columns should keep units in their names, for example
  `actual_general_price_mwh` and `actual_feed_in_price_mwh`.

## HAEO / HAFO Alignment

The canonical Home Assistant-facing convention should follow HAEO/HAFO where it
is practical:

- Import price is a positive cost in `$ / kWh`.
- Export price is a positive revenue in `$ / kWh`.
- Negative export price means the customer pays to export.
- Import and export power should be separate positive-or-zero flows where the
  integration/control surface supports that shape.
- Forecast sensors should expose a standard `forecast` attribute using
  timezone-aware ISO timestamps. Prefer the HAFO-style point shape:

```yaml
forecast:
  - datetime: "2025-01-15T10:00:00+00:00"
    native_value: 0.25
```

This repo should publish future production/shadow price forecasts in that
HAEO-compatible shape alongside any legacy Amber-compatible adapter sensors.

References:

- HAEO grid price convention:
  <https://haeo.io/user-guide/elements/grid/>
- HAFO forecast point convention:
  <https://hafo.haeo.io/user-guide/forecasters/historical-shift/>
- Amber Express HAEO compatibility note:
  <https://github.com/hass-energy/amber-express>

## Amber Feed-In Sign

Amber/Home Assistant feed-in sensors use the opposite sign convention:

- negative feed-in price means the customer earns money by exporting
- positive feed-in price means the customer pays to export

That convention should not leak into model, eval, or tariff logic. Convert only
at the Home Assistant compatibility boundary:

- `tariff_utils.export_value_to_amber_feed_in_price()`
- `tariff_utils.amber_feed_in_price_to_export_value()`

The old AI combined Amber-compatible publisher has been retired. Treat any
remaining Amber-shaped feed-in handling as a provider boundary adapter, not as
the internal convention.
