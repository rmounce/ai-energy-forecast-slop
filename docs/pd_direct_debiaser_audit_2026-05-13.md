# PD-direct Debiaser Audit — 2026-05-13

User observed that the new (post-`383d0fe`) PD-direct refresh on every PREDISPATCH
publish keeps `sensor.ai_pd_direct_price_forecast` ≤2 min behind AEMO, but the
debiased Tier 2 values are now consistently **$30+/MWh BELOW raw AEMO PREDISPATCH**
(`sensor.ai_aemo_price_forecast`). Audit answers: is the PD-direct debiaser
actually doing useful work, or is it adding error?

## Method

`eval/audit_pd_direct_debiaser.py` joins every row in
`pd_direct_forecast_log.csv` with:

- raw PREDISPATCH `rrp` (from `aemo_predispatch_sa1.parquet`, latest `run_time`
  ≤ `forecast_creation_time`, matching `interval_dt`),
- realised RRP (from `actuals_sa1.parquet`, 30-min aggregate, using
  `target_time - 30min` to align AEMO's interval-START vs the log's interval-END).

Compares MAE / signed bias / per-row "did debiaser win" frequency for the debiased
prediction vs raw PREDISPATCH against the realised actual.

**Sample**: 10,812 paired rows, target window `2026-05-08T11:00Z → 2026-05-12T22:00Z`.
Spans the alignment-fix promotion (`211ae8d`, 2026-05-11T13:22Z) so split into pre/post.

## Headline

**The debiaser is hurting more than helping, in both periods.** Raw PREDISPATCH
is closer to realised RRP on average than the debiased PD-direct curve.

| window | n | actual | raw mean | PD mean | raw MAE | PD MAE | raw bias | PD bias | debiaser helped% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PRE (<2026-05-11T13:22Z) | 8883 | $97.96 | $101.02 | **$118.66** | $29.36 | **$33.39** | +$3.06 | **+$20.70** | 41.8% |
| POST (≥promotion) | 1929 | $93.61 | $79.77 | $88.49 | $26.86 | **$29.38** | −$13.85 | −$5.13 | 48.2% |

The May 11 promotion improved calibration meaningfully — PD-direct bias
dropped from +$21/MWh to −$5/MWh, MAE dropped from $33 to $29. **But raw
PREDISPATCH still beats debiased PD-direct overall** (post-promotion MAE
$26.86 vs $29.38; "helped%" 48.2% < 50%).

## POST-Promotion Bucket Breakdown (n=1929)

| Adelaide bucket | n | actual | raw | PD-direct | raw MAE | PD MAE | helped% | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| overnight | 648 | $70 | $65 | $82 | $7.7 | **$17.8** | **31%** | debiaser **hurts** — over-corrects up |
| morning | 380 | $117 | $87 | $89 | $40.4 | $40.2 | 52% | tie — both severely under-forecast |
| solar | 305 | $138 | $75 | $88 | $73.8 | **$63.9** | **78%** | debiaser **helps** — pulls up toward unusually-high solar actuals |
| evening | 316 | $91 | $111 | $117 | $21.2 | **$26.4** | 41% | debiaser **hurts** — amplifies raw's over-forecast |
| late | 280 | $70 | $75 | $71 | $8.2 | **$7.2** | 60% | small help |

### Structural reads

1. **Overnight is the worst bucket.** Raw PREDISPATCH is nearly unbiased there
   (bias −$5.9, MAE $7.7); the debiaser adds +$18 of upward shift, more than
   doubling MAE to $17.8. Helped only 31% of intervals.
   *Why this matters operationally:* overnight is where EMHASS decides whether
   to pre-charge from the grid. A $18/MWh systematic upward shift makes
   "import overnight to discharge in the morning peak" look more profitable
   than it actually is.

2. **Evening is also hurt.** Raw already over-forecasts evening peaks by
   +$20/MWh on average; the debiaser pushes another +$5 on top. Combined
   +$25 over-forecast at evening would make EMHASS plan grid imports during
   periods that turn out not to need them.

3. **Solar is the only place the debiaser meaningfully helps.** It's helping
   because actuals in this window were unexpectedly **high during solar**
   (mean $138/MWh vs the −$10 to $40 that would normally be expected for
   PV-rich periods). The debiaser was trained on a different regime where
   PREDISPATCH systematically under-shoots solar. That historical pattern
   happened to keep working here, but the cause looks accidental rather
   than structural.

## Live Disagreement With Historical Pattern

The live snapshot taken 2026-05-13T13:00Z showed PD-direct **−$30/MWh BELOW**
raw PREDISPATCH at tonight's overnight slots (15:00–17:00Z = early Adelaide
morning). The POST overnight bucket historical pattern says PD-direct should
be **+$17/MWh ABOVE** raw. The live observation is ~$47/MWh outside the
historical overnight pattern.

Likely cause: tonight's forecasts at horizon 4-12h are in the PD7Day debiaser
region (steps 56+ of the 144-step curve), which has different bias characteristics
than PREDISPATCH and was retrained more recently. Without running a separate
audit segmented by debiaser source (PREDISPATCH vs PD7Day vs seasonal HoD), we
can't attribute the live anomaly precisely. The structural-calibration question
(see #1, #2 above) stands regardless.

## Options Considered

a) **Disable the debiaser entirely** — pass raw PREDISPATCH through as Tier 2.
   Recovers $2.5/MWh MAE on average and gives substantial overnight/evening
   improvement. Loses the solar-bucket gain ($63.9 → $73.8 MAE) but solar's
   apparent gain was a quirk of the unusual high-actuals regime in this window.
   *Lowest-risk change.*

b) **Selective bucket-aware debiaser** — apply only in solar; pass through raw
   in overnight/evening/late. Captures the solar pattern without the
   overnight/evening damage. Requires plumbing a bucket-aware switch through
   `_execute_pd_direct_prediction` and is more code than (a).

c) **Investigate today's live anomaly first** — figure out whether the
   ~$30/MWh downward correction tonight is a feature-edge-case bug or a
   genuine regime shift the new debiaser picked up that the log audit cannot
   yet reflect. Probably an `eval/` analysis splitting PD-direct contributions
   by debiaser layer (PREDISPATCH vs PD7Day vs seasonal HoD).

d) **Live with it** — leave PD-direct as canonical Tier 2 (raw PREDISPATCH isn't
   currently a production source), note the calibration gap, and queue a
   debiaser retrain when the training data spans more of the current AEMO
   regime. Aligns with the cautious 2026-05-05 strategic pivot stance
   (no further model work until baselines are measured).

## Decision

Documented only for now. No code or HA changes. Re-running the audit when
more POST-promotion data accumulates may shift the verdict — the POST sample
is currently 1,929 rows over ~1.5 days, vs the PRE sample of 8,883 rows over
~3 days. POST trend may continue improving past 50% helped%.

## Dispatch-side corroboration (2026-05-14, from Run B v3)

The 6-week rolling-MPC eval `loadsrc_B_v3_*` (same `2026-04-01 → 2026-05-12`
window, `--strategic-soc-handoff --strategic-target-mode exact`,
`amber_apf_lgbm` price source) produced PnL ≈ −$0.18 for LGBM-load and +$7.21
for actual-load over 41 days. The negative absolute is **not** load-source
driven: on the worst day (2026-04-06) `actual` lost −$2.94 and `lgbm` lost
−$3.23 — a difference of only $0.29.

What is driving the loss is the **strategic 72h LP's price-only optimisation
choosing high SoC targets in response to debiased PD-direct's inflated future
spikes**, then the per-step LP racing to hit those targets via expensive grid
imports while realised spikes either don't materialise or are smaller than
forecast.

| metric | value |
|---|---:|
| Days strategic_target_max hits 40 kWh (battery cap) | 29 / 42 (69%) |
| Daily-PnL × strategic_target_mean correlation | **−0.37** |
| Daily-PnL × actual_price_max correlation | +0.53 |
| Negative-PnL days (B_lgbm) | 26 / 42 |

Top loss days have sustained-high strategic targets (24-40 kWh mean); top
earning days have lower sustained targets (12-23 kWh mean) but still hit 40
kWh peaks. The strategic LP knows the spikes are *coming* on both kinds of
day, but on losing days it stays pre-charged through long expensive windows
between spikes.

**This is direct dispatch evidence for the debiaser-MAE finding above.**
PRE-promotion PD-direct over-forecast by +$20.70/MWh — exactly the bias that
makes the strategic LP over-confident. Even the POST window (PD bias −$5.13)
isn't enough to flip the dispatch sign: most of the 41-day window is PRE.

Option (a) — disable the debiaser, pass raw PREDISPATCH — predicts a
direct dispatch PnL improvement on this same window. A future run with
`--strategic-price-source=raw_predispatch` (would need to be added to the
harness) would verify. Wall-clock ~9h on the strategic-handoff path.

Run C v3 (`--terminal-energy-value-mwh 100`, no strategic 72h LP) avoids the
problem entirely — flat term-value PnL was +$11.95 for LGBM and +$19.62 for
actual on the same window. The $12 swing between B and C with identical
load and price inputs is the cost of the strategic LP's over-confidence
under the current debiased Tier 2 curve.

## Files

- Audit script: `eval/audit_pd_direct_debiaser.py`
- PD-direct log: `pd_direct_forecast_log.csv`
- Raw PREDISPATCH: `data/parquet/aemo_predispatch_sa1.parquet`
- Realised RRP (30-min): `data/parquet/actuals_sa1.parquet`
- Run B v3 raw rows: `eval/results/loadsrc_B_v3_{actual,lgbm_load_log}_raw.parquet`
- Run C v3 raw rows: `eval/results/loadsrc_C_v3_{actual,lgbm_load_log}_raw.parquet`

## Cross-references

- `docs/roadmap.md` §4 — TFT load / PD-direct production status
- `docs/alignment_fix_retrain_2026-05-11.md` — the May 11 promotion this audit
  is segmented around
- `docs/production_forecast_switch_plan.md` — production cadence including the
  2026-05-13 `publish-pd-direct` chained refresh
