# Session Consolidation — 2026-05-15

Closes the multi-thread session that spanned the load-source dispatch
counterfactual, the shadow-forecast Tier-1→Tier-2 discontinuity, the
PD-direct/LGBM bias audits, the seasonal tariff overlay, the retail-plan
comparison, the silent `model_a_hybrid` corruption fix, and the LGBM
bias-calibration sensitivity hook.

## What landed

### Forecast / production pipeline
- **`forecast.py publish-pd-direct`** publishes the Tier-2 (PD-direct)
  shadow curve immediately after each PREDISPATCH ingest. Wired into
  `systemd/ai-energy-predispatch.service`. Closes the ~24-29 min staleness
  gap that produced the visible Tier-1→Tier-2 jump in
  `sensor.ai_spot_price_forecast`.
- **TFT price inference removed from `predict-all`**. No production
  consumer; the entities were dropped from `config.json`'s
  `publish_entities`. CPU saving on every prediction cycle.
- **Tier-1 quantile staleness guard** (`e953ca4`). `tier1_quantile()` now
  returns None when the underlying `aemo_p5min_sa1` series is older than
  1h. Was silently using March-31 data on every April run because
  NEMSEER's backfill scope ends 2026-03-31 by design; live p5min data
  lives only in InfluxDB. Symptom was `model_a_hybrid` showing 100%
  `repaired_invalid_curve` in eval runs.

### Rolling-MPC eval harness
- **Load-source counterfactual plumbing**: `--load-forecast-source`,
  `_load_load_forecast_log_30m()`, `lookup_load_forecast_kw_5m()`, per-row
  `load_forecast_source`/`forecast_load_step0_kw`/`load_under_prep_*`
  columns and bucketed summary fields.
- **Seasonal RESELE feed-in overlay** (`03d7d73`). Inline
  `SUMMER_FEED_IN_OVERRIDES` (Nov–Mar 17:00–20:30 → +$0.1225/kWh) replaces
  the gitignored summer tariff JSON. DST-safe (floor in UTC, then convert).
- **`--lgbm-bias-calibration`** (`4edd2a1`, this commit). Opt-in flag
  subtracts the audit-measured per-Adelaide-bucket bias from the
  `amber_apf_lgbm` strategic curve before LP consumption. Smoke confirms
  the flag reaches the LP (42/285 discharge rows, 226/285 SoC rows differ
  vs uncalibrated). PnL delta on a calm April day was +$0.006 — bigger
  effects expected on spikier windows where over-confidence costs more.
- **Bug fixes**: empty-frame guard in `build_coverage_summary`; DST
  fall-back crash in `_tariffed_price_frame_from_wholesale_mwh`.

### New audit / analysis tools
- `eval/audit_pd_direct_debiaser.py` — PD-direct vs raw PREDISPATCH vs
  realised. Verdict in `docs/pd_direct_debiaser_audit_2026-05-13.md`:
  debiaser MAE exceeds raw PREDISPATCH MAE on this window.
- `eval/audit_price_forecast.py` — generic LGBM/TFT forecast-vs-realised.
  Verdict in `docs/price_forecast_bias_audit_2026-05-14.md`. Bug found
  and fixed: forecast log `actual` column is $/kWh, not $/MWh.
- `eval/retariff_dispatch.py` — replay an existing rolling-MPC raw_rows
  under RESELE / Flow Power / IO Energy tariff structures. Verdict in
  `docs/retail_plan_comparison_2026-05-14.md`.
- `eval/compare_loadsrc_runs.py` — side-by-side comparison helper for
  load-source A/B runs.

## Findings worth re-citing

### Load-source counterfactual
TFT-load vs LGBM-load vs actual-load matters very little for dispatch
on these tariffs:
- Winter (41 days, Apr 1 – May 12): ~$7.50 spread across load sources.
- Summer (14 days, Jan 1–15): $3.44 spread, on per-day PnL ~70× higher
  than winter because the RESELE evening credit dominates.
- Mechanism: LP over-commits to pre-charging because LGBM over-forecasts
  the overnight/evening bucket by +$12–14/MWh.

### LGBM bias profile (audit window)
Adelaide-local buckets, signed mean (pred − actual) in $/MWh:
overnight +14.73, morning −6.10, solar −13.80, evening +11.97, late +13.96.

### TFT bias profile
Short-horizon compression: 0–1h −$28.46 (over-discounts), 24h+ +$0.12
(centred). Indicates double-application of the debiaser on early steps.

### Retail plan comparison
No plan strictly dominates:
| plan       | annual $ (PV-charge) | annual $ (grid-charge) |
|---         |---:                  |---:                    |
| RESELE     | ~$1,700–1,900        | ~$1,700–1,900          |
| Flow Power | ~$2,500              | ~$250                  |
| IO Energy  | ~$2,200              | ~$1,250                |

RESELE owns spike upside (wholesale-pass-through), Flow Power owns the
predictable 17:30–19:30 evening if PV is strong, IO Energy is the
robust middle. User's preference: stay on RESELE — "Amber pays out big
during spikes [and] I feel better actively supporting the grid when it's
needed."

## What is still open

- **LGBM bias-calibration validation** beyond the smoke. To actually
  estimate the dispatch lift in summer where the LP has more leverage,
  re-run the summer 2-week or Run B v3 windows with the flag on and
  compare. Cheap, can be queued in tmux next session.
- **LP-per-plan eval** for Flow Power / IO Energy. The retariff replay
  is a lower bound because the LP was tuned for RESELE. Only worth
  doing if a switch is on the table.
- **Roadmap items α/β/γ** from `docs/roadmap.md` continue as the main
  baseline-first track; this session was tangential to that.

## Pointers
- `e953ca4` — Tier-1 staleness guard
- `4edd2a1` — `--lgbm-bias-calibration`
- `f17dae6` — RESELE/Flow/IO doc
- `71686cf` — `eval/retariff_dispatch.py`
- `380f88c` — bias audit doc + seasonal dispatch extension
- `03d7d73` — seasonal-aware MPC eval harness
