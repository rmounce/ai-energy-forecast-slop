# Codebase Bug-Class Audit - 2026-05-12

This audit was a targeted cleanup pass after the shadow forecast promotion work. It
looked for known failure classes that have already caused production confusion:
stale forecast-run mixing, timestamp convention drift, silent fallback, and
requested-vs-effective source ambiguity.

## Findings

### 1. Stale PREDISPATCH run mixing

The dangerous Influx pattern `last(rrp) GROUP BY time(...)` is no longer present in
active fetch code. The remaining reference is a warning comment in `forecast.py`
near `_get_influx_pd_prices`.

This is a good state. Keep this as an audit checklist item because the failure mode
is subtle: grouping all PREDISPATCH rows by interval can combine rows from different
AEMO run tags and publish a stale curve that still looks fresh.

### 2. Timestamp convention drift

Current promoted PD-direct / PD7Day debias artifacts include the 2026-05-11
alignment fix and are documented in `docs/alignment_fix_retrain_2026-05-11.md`.
The canonical HA publish surfaces use UTC interval-start timestamps.

Deferred risks remain in archived or non-production paths:

- `data/build_training_dataset.py` still appears to pair interval-end forecast
  targets against actuals indexed at the same timestamp.
- `train/train_lgbm_strategic.py` has the same apparent interval-end target
  convention.
- Forecast-log diagnostics that span the 2026-05-11 publish fix can mix old and
  corrected timestamp conventions. Use `--since` or equivalent filtering when
  comparing live logs to actuals.

These are not current production blockers while TFT/strategic retraining remains
parked, but they should be fixed before reviving those branches.

### 3. Silent canonical AI source fallback

Before this audit, the canonical HAEO-style AI control sensors could publish from
the deprecated TFT Tier 2 fallback if PD-direct failed, and the HA readiness sensors
only checked count, freshness, horizon, and timestamp alignment.

That is acceptable for shadow visibility but unsafe for a future selectable
production control source: a TFT fallback could have passed readiness purely on
shape.

The cleanup now adds source provenance attributes to canonical AI sensors:

- `forecast_source`
- `tier1_source`
- `tier2_source`
- `control_ready_tier2_source`

The HA readiness sensors now require import/export to report the same `tier2_source`,
and that source must be `PD-direct` or `cached PD-direct`. A `TFT fallback` or
`cached TFT fallback` canonical publish is therefore observable but not
control-ready.

### 4. Requested-vs-effective source ambiguity

The HA selectors still expose only legacy production choices, so there is no
immediate routing risk. If `ai_shadow` is reintroduced, the templates already fall
back to legacy prices when the AI status sensor is not `ready`.

A future cleanup should add an explicit effective-source diagnostic, e.g.
`requested_source`, `effective_source`, and `fallback_reason`, so the frontend does
not imply that `ai_shadow` was actually used when the guard rejected it.

## Recommended Follow-Up Order

1. Keep the new source guard in place before reintroducing any `ai_shadow` selector
   option.
2. Add an effective-source diagnostic sensor before the first controlled AI switch.
3. Fix timestamp alignment in `data/build_training_dataset.py` and
   `train/train_lgbm_strategic.py` before any future TFT or strategic retrain.
4. Treat stale-run mixing search as a recurring pre-release audit:
   `rg "last\\(|GROUP BY time" forecast.py data eval train ingest -g '*.py'`.
