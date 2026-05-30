# RFC: Publish PD-direct as a parallel HA shadow forecast

**Status:** approved for implementation 2026-05-07. Open questions resolved at the bottom
of this file (section *Resolved decisions*). Implementation tracked in `HANDOVER.md` as
Phase α-prime Step 2.
**Date:** 2026-05-07.
**Author:** Claude.
**Context:** Phase α-prime Step 2 in `docs/roadmap.md`. The reviewer (HANDOVER 2026-05-07)
gated retirement of the existing TFT shadow on PD-direct first having a production-
compatible HA entity and a reversible source-selector switch. This RFC sketches that work
*before* any code lands so it can be reviewed cheaply.

## What's actually published today (clarification)

I had been describing "the TFT shadow" as `sensor.ai_tft_price_forecast`. That entity name
appears in `ARCHITECTURE.md` and old roadmap text but **is not produced by the code today**.
The actual published shadow forecast is a *Tier-1+TFT bundle* across six entities:

| Entity | Cadence | Length | Source |
|---|---|---|---|
| `sensor.ai_combined_general_price_forecast` | 5min | T1 12 + T2 144 | T1 q05/50/95 + TFT q30/50/70 (Amber-shaped) |
| `sensor.ai_combined_feed_in_price_forecast` | 5min | T1 12 + T2 144 | as above (sell side) |
| `sensor.ai_mpc_import_price_forecast` | 5min | 168 | T1 q50 + TFT q50 (HAEO-style, positive import) |
| `sensor.ai_mpc_export_price_forecast` | 5min | 168 | T1 q50 + TFT q50 (HAEO-style, positive export) |
| `sensor.ai_dh_import_price_forecast` | 30min | 144 | TFT q50 only (HAEO-style) |
| `sensor.ai_dh_export_price_forecast` | 30min | 144 | TFT q50 only (HAEO-style) |

The HA selectors `input_select.emhass_mpc_price_source` and `..._dh_price_source` each have
two options today: their Amber default and `ai_shadow` (which reads the entities above).

**`sensor.ai_tft_price_forecast` does not exist as a published entity** — only as a CSV log
file `tft_price_forecast_log.csv` for offline analysis. So "retire the TFT shadow" in the
operational sense means stop populating the `ai_combined_*`/`ai_mpc_*`/`ai_dh_*` family
from TFT q50.

This is a useful clarification for the retirement decision: today there is exactly one AI
shadow path, fed by Tier 1 + TFT. To make PD-direct switchable without retiring TFT first,
we need a *second parallel path*.

## Proposed design

### Naming

Add a new sibling family of entities, prefix `ai_pd_direct_`:

- `sensor.ai_pd_direct_combined_general_price_forecast`
- `sensor.ai_pd_direct_combined_feed_in_price_forecast`
- `sensor.ai_pd_direct_mpc_import_price_forecast`
- `sensor.ai_pd_direct_mpc_export_price_forecast`
- `sensor.ai_pd_direct_dh_import_price_forecast`
- `sensor.ai_pd_direct_dh_export_price_forecast`

Identical schema to existing `ai_*` siblings (HAEO-style, `forecast` attribute with
`{datetime, native_value}` items, UTC timestamps, $/kWh, positive import cost, positive
export revenue). Only the source data and entity names differ.

### Selector options

In `hass/packages/emhass.yaml`, extend each selector by **adding** an option, not replacing:

```yaml
input_select:
  emhass_mpc_price_source:
    options:
      - amber
      - ai_shadow      # existing — Tier 1 + TFT
      - ai_pd_direct   # new — Tier 1 + debiased PREDISPATCH + PD7Day/HoD tail
    initial: amber     # unchanged
  emhass_dh_price_source:
    options:
      - amber_lgbm_extrapolated
      - ai_shadow
      - ai_pd_direct
    initial: amber_lgbm_extrapolated
```

EMHASS template branches stay additive: today there are `if ai_shadow` blocks; we add
`elif ai_pd_direct` blocks that read from `sensor.ai_pd_direct_dh_*` / `..._mpc_*`. Default
behavior is unchanged for any user not actively flipping the selector.

### `forecast.py` changes

Three localised additions, each at an existing seam — no rewrites of the monolith:

1. **New result builder** — `_execute_pd_direct_prediction(historical_df, adjusted_covariates)`
   sits next to `_execute_tft_prediction()` (around line 2407) and returns a dict shaped
   identically to `tft_results`:
   ```python
   {
       'pd_direct_price':     df_q50,   # 30-min, 144 steps, wholesale $/kWh
       'pd_direct_price_q30': df_q30,   # = q50 in first cut (no spread until Step 3)
       'pd_direct_price_q70': df_q70,   # = q50 in first cut
   }
   ```
   Internally it calls `eval.pd_direct_baseline.build_pd_direct_30m_curve()` for steps
   2..144 and reuses the Tier 1 LGBM result for steps 0..1 (60 min). Same module the eval
   already uses, so live and eval can never drift.

2. **New publish function** — `_publish_pd_direct_price_forecasts(tactical_results, pd_direct_results)`
   sits next to `_publish_combined_price_forecasts()`. Direct copy with two changes:
   entity-name prefix `ai_pd_direct_*`, friendly-name "PD-Direct" in place of "AI Combined".
   The internal builders (`_build_combined_forecast_items`,
   `_build_haeo_price_forecast_items`) are reused unchanged.

3. **Orchestrator hook** — at line ~2413 inside `run_predictions`, call
   `_publish_pd_direct_price_forecasts(...)` immediately after the existing
   `_publish_combined_price_forecasts(...)`. Wrapped in its own `try/except` so a PD-direct
   publishing failure cannot break the existing TFT publish path.

No changes to `apply_tariffs_to_forecast`, `_build_combined_forecast_items`, the Tier 1
predictor, or the TFT predictor. No change to `predictions.json` schema (PD-direct is HA-
only for now; we can decide later whether to also persist it to the JSON file).

### Quantile bands (deferred to Step 3)

For Step 2 first cut, `pd_direct_price_q30 == pd_direct_price_q70 == pd_direct_price_q50`.
The Amber-shaped `ai_pd_direct_combined_*` sensors will publish identical low/predicted/
high values, which is honest — PD-direct has no native quantile spread until Step 3 wires
in horizon-stratified empirical residual bands. EMHASS template branches that consume
these entities work either way.

When Step 3 lands, the residual-band module becomes a function call inside
`_execute_pd_direct_prediction()` that produces non-degenerate q30/q70 frames. No HA
package change needed.

### Things explicitly **not** changing

- Default selector value. `amber` and `amber_lgbm_extrapolated` remain the live MPC and DH
  inputs. The user must manually flip the selector to put either AI source live.
- The existing TFT-based `ai_shadow` entity family. They keep publishing.
- Tariff application logic.
- `predictions.json` shape.
- `forecast.py` CLI surface (no new flag — PD-direct publishes whenever `--publish-hass` is
  set, gated only on the eval baseline module being importable).

### What the `predict-price --publish-hass` run looks like after this

```
--- Tier 1 LGBM tactical predict ---  (unchanged)
--- TFT predict ---                    (unchanged)
--- PD-direct compute ---              (new, fast: pure pandas/numpy lookup, ~50 ms)
--- Publish ai_combined_* / ai_mpc_* / ai_dh_* ---     (unchanged)
--- Publish ai_pd_direct_combined_* / ai_pd_direct_mpc_* / ai_pd_direct_dh_* ---  (new)
```

A user inspecting HA can now compare PD-direct against TFT side-by-side on the same
graphs without flipping anything. EMHASS/MPC sees no change unless the selector is moved.

## Open questions for review

1. **Naming.** Is `ai_pd_direct_*` the right prefix? Alternatives: `ai_baseline_*` (less
   accurate), `ai_pd_*` (ambiguous), `ai_predispatch_*` (loses the "direct, no ML" meaning).
   I prefer `ai_pd_direct_*` for explicitness.
2. **Combined-Amber-shaped sensors.** The two `ai_combined_*` entities are an Amber-shape
   compatibility convention. Worth duplicating for PD-direct, or do we only need the four
   HAEO-style entities (the ones EMHASS actually consumes)? Keeping all six gives parity
   for any HA dashboard already wired to `ai_combined_*`. Cost is small.
3. **Selector value name.** `ai_pd_direct` (matches the entity prefix) vs `pd_direct`
   (matches the eval source name). I'd go with `ai_pd_direct` for HA-side consistency with
   `ai_shadow`.
4. **Failure mode.** If the debiased PD parquet is stale (e.g. >24h old) the PD-direct
   publish should still succeed but with a status sensor flagged stale. We already have
   `sensor.ai_mpc_price_forecast_status` and `..._dh_price_forecast_status` — should
   PD-direct have parallel status sensors? Cheap to add.
5. **Live-inference debiased-PD source.** The eval uses the OOF parquet. Live inference
   uses `_apply_pd_debiaser()` inside `forecast.py` against the loaded models. These
   should be equivalent (the OOF parquet was *generated* by the debiaser running OOF over
   history), but worth verifying we use the same debiaser path live as the eval did
   historically.

## Cost / risk

- **Wall time:** ~half-day of editing + smoke. No new dependencies.
- **Production risk:** low. Additive, default selector unchanged, separate try/except.
  Rolling back is `git revert` on the `forecast.py` and `emhass.yaml` changes.
- **Quota:** modest — 6–10 well-targeted edits, then a smoke run of `predict-price
  --publish-hass --debug-tft` and a quick HA-side sanity check.

## Decision asked of reviewer

OK to proceed with this design? Specifically:

- (a) The `ai_pd_direct_*` naming and full six-entity parity.
- (b) Reusing `eval/pd_direct_baseline.py` directly from `forecast.py` (binds live and
  eval to the same code path — feature, not bug).
- (c) Deferring quantile bands to Step 3 (entity payload publishes degenerate q30=q50=q70
  in the meantime; honest, doesn't block).
- (d) Adding parallel status sensors `sensor.ai_pd_direct_mpc_price_forecast_status` and
  `..._dh_price_forecast_status` to mirror the existing pair.

---

## Resolved decisions (2026-05-07)

All five open questions resolved. Recorded here so future readers can see what was
chosen and why.

| # | Question | Decision | Rationale |
|---|---|---|---|
| 1 | Entity prefix | `ai_pd_direct_*` | Long but explicit; alternatives lose meaning or were ambiguous. |
| 2 | Six-entity parity vs only the four EMHASS-consumed | Ship all six | Dashboard cards already wired to `ai_combined_*` need a parallel for visual A/B. ~10 lines of plumbing. |
| 3 | Selector value name | `ai_pd_direct` | Matches entity prefix and existing `ai_shadow` convention. |
| 4 | Status sensors | Yes, parallel pair | Without them a stale debiased-PD parquet would silently produce stale prices. Non-negotiable for production safety. |
| 5 | Live debiaser path | `_apply_pd_debiaser` (final model `lgbm_final.pkl`), **not** the OOF parquet | The eval correctly used OOF for leakage avoidance; live should use the final model trained on all data, which is the same code path with more data. Document the small live-vs-eval delta as expected and not drift. |

### Live-vs-eval delta — what to expect

Live PD-direct values will be very close to OOF-eval values but not bitwise identical
(different fold structure, more training data in the final model). When live PD-direct
starts logging alongside actuals after Step 2 ships, the delta should sit within roughly
$1–3/MWh on typical conditions. A larger delta is a signal that something changed
(debiaser retrained, parquet stale, feature drift) and needs investigation. This is the
calibration check the eval-vs-live audit will compare against.
