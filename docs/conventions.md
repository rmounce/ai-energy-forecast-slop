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

## Amber Feed-In Sign

Amber/Home Assistant feed-in sensors use the opposite sign convention:

- negative feed-in price means the customer earns money by exporting
- positive feed-in price means the customer pays to export

That convention should not leak into model, eval, or tariff logic. Convert only
at the Home Assistant compatibility boundary:

- `tariff_utils.export_value_to_amber_feed_in_price()`
- `tariff_utils.amber_feed_in_price_to_export_value()`

The AI combined publisher intentionally emits Amber-compatible feed-in forecast
items because the current EMHASS templates consume Amber-shaped objects. Treat
that as a boundary adapter, not as the internal convention.
