# Production price-source switch contract

**Status:** no alternate source is currently eligible or publishing. Updated 2026-08-10.

## Current Routing

| Consumer | Active source | Resolution / horizon |
|---|---|---|
| EMHASS MPC | Raw Amber 5-min forecast | 5 min / 14h |
| EMHASS day-ahead | `sensor.ai_price_forecast(_low/_high)` | 30 min / 72h |
| Day-ahead price construction | Amber APF + LightGBM tail | Amber horizon + extrapolation |

The APF-free canonical AI sensors, PD-direct sensors, AEMO stitched sensors, TFT price sensors,
and tactical sensors were disabled on 2026-06-15. `forecast.py publish-tactical` and
`publish-pd-direct` are retained no-op commands. Do not use their historical selector/readiness
plumbing as evidence that a switchable source exists today.

## Rules For Any Future Source

A source-switch implementation must be a separate, reversible project. Before adding a selector
option or control route, require:

- written source contract: lineage, APF dependence, units, sign, timestamps, resolution, horizon;
- live shadow publisher with no EMHASS consumer;
- complete forecast manifest and freshness/status entity;
- paired forecast and tariffed-dispatch evaluation against the active source;
- inventory/terminal-SoC safety result;
- failure injection for missing, stale, partial, non-finite, and misaligned arrays;
- one-action rollback to the current Amber routes.

## Minimum Runtime Guards

- All required quantile/import/export arrays exist.
- Expected point count and resolution match the consumer.
- First timestamps align and are timezone-aware UTC.
- No duplicate, missing, non-finite, or unsorted timestamps.
- Remaining horizon covers the optimisation window.
- Price units are `$ / kWh` at the HA/EMHASS boundary.
- Import is positive cost; export is positive revenue except when export itself costs money.
- Source data and published entity are within source-specific freshness limits.
- Failed validation leaves the last known-good route active and emits an observable failure.

## Rollout Order

1. Publish diagnostics only.
2. Observe live freshness and validation state through representative normal and volatile days.
3. Add an unreachable adapter branch and tests.
4. Add an explicit selector option, default unchanged.
5. Switch day-ahead first.
6. Switch MPC only under a separate approval after day-ahead evidence.
7. Keep Amber source values and rollback available throughout.

Historical PD-direct/TFT switch work is archived under
[../archive/price_forecast_2026/](../archive/price_forecast_2026/). The active implementation
track is production hardening, not source replacement; see
[production_hardening_plan_2026-08-10.md](production_hardening_plan_2026-08-10.md).
