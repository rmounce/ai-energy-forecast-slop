# Load-Source Dispatch — Run A Partial Findings, 2026-05-13

## Status

| Run | Window | Status | Wall-clock |
|---|---|---|---|
| A | 3-day, all 3 load sources, both price sources, strategic handoff exact | **Done** | ~41 min |
| B v1 | 6-week, 2 load sources, strategic handoff exact | **Crashed** (DST bug) | ~5 min |
| C v1 | 6-week, 2 load sources, terminal-value=100 | **Crashed** (same DST bug) | ~5 min |
| B v2 | Same as B v1, post-DST-fix | In flight | TBD |
| C v2 | Same as C v1, post-DST-fix | In flight | TBD |

DST bug was in `_tariffed_price_frame_from_wholesale_mwh` (L416). Fall-back at 2026-04-05 02:00 Adelaide raised `Cannot infer dst time` because `floor("30min")` re-localized through the ambiguous local hour. Fixed in `6ea7afd` by flooring in UTC before converting to local.

## Run A Headline Numbers

3-day window `2026-05-09T12:00Z → 2026-05-12T12:00Z`, 864 MPC steps, `--strategic-soc-handoff --strategic-target-mode exact`.

| price_source | load | total_pnl | mean/day | soc_final (kWh) | repaired | under-prep |
|---|---|---:|---:|---:|---:|---:|
| amber_apf_lgbm | actual | 1.541 | 0.514 | 3.51 | 0 | 0 |
| amber_apf_lgbm | lgbm_load_log | **-0.164** | -0.055 | **12.00** | 0 | 335 |
| amber_apf_lgbm | tft_load_log | 0.856 | 0.286 | 5.78 | 0 | 419 |
| model_a_hybrid | actual | 2.193 | 0.732 | 0.00 | **864** | 0 |
| model_a_hybrid | lgbm_load_log | 2.226 | 0.743 | 0.33 | **864** | 335 |
| model_a_hybrid | tft_load_log | 1.946 | 0.649 | 0.10 | **864** | 419 |

## Two Confounds Before Reading Too Much Into This

### 1. `model_a_hybrid` is producing invalid tactical curves on every step

`repaired_invalid_curve = 864 / 864` for every model_a_hybrid run. The curve repair (L870–886) ffills/bfills NaN values, so the curve isn't outright broken, but every step had at least one NaN in the Tier 1 + Tier 2 blend that required patching.

Hypothesis: TFT price inference is failing on this recent May 2026 window (training distribution boundary? missing decoder feature? recent-data ingest gap?). Not investigated tonight — the `model_a_hybrid` PnL column should be treated as untrustworthy until this is understood.

`amber_apf_lgbm` shows 0 repairs across all three load runs — that path is clean.

### 2. `--strategic-soc-handoff` is NOT load-aware

`strategic_solve_summary` (L1167–1173) calls `solve_lp_dispatch(curve, soc_init_kwh)` without any load/PV/import/export args. The strategic 72h solve is price-only arbitrage — it sets a SoC handoff target ignoring load.

Implication: load-source swap affects the per-step LP but not the strategic target. Different load forecasts → different per-step actions → different SoC trajectories chasing the same price-only strategic target. This is a structural property of the harness, not a bug — but it means the apparent "verdict" between load sources is mediated by how each interacts with a load-blind strategic target.

If we ever want to score load-aware strategic targets, that's a separate change.

## Cleanest Read: amber_apf_lgbm Only

| load | total_pnl | soc_final |
|---|---:|---:|
| actual | $1.541 | 3.51 kWh |
| lgbm_load_log | **-$0.164** | **12.00 kWh** (battery cap) |
| tft_load_log | $0.856 | 5.78 kWh |

Surprising results:

- **LGBM-load → negative PnL with terminal battery full**. The strategic-handoff target is presumably high (price-only optimisation likes ending charged). The per-step LP is fed LGBM's load forecast which is conservative (overestimates load by ~142 W mean per Codex). The LP plans for more grid import than reality needs, charges the battery, and the conservative bias compounds. Battery ends at cap; net dispatch loses money.
- **TFT-load → positive PnL, intermediate SoC**. TFT's load forecast is closer to actual on MAE, so the per-step LP plans more accurately and doesn't over-import. Better PnL.
- **actual → highest PnL, lowest SoC**. Perfect-info LP, ends most-drained because it spent the battery on profitable arbitrage rather than over-buying grid energy.

This is the OPPOSITE of the operational caveat in roadmap §4:

> User preference is conservative: stay with LGBM because over-estimating load is safer operationally than under-preparing.

The smoke says: over-estimating load *also* costs money in dispatch because the LP commits to grid imports that don't materialize.

## Under-prep Distribution

Under-prep (forecast < actual) Adelaide-bucketed, identical across price sources because it only depends on load:

| load | overnight | morning | solar | evening | late | total |
|---|---:|---:|---:|---:|---:|---:|
| lgbm_load_log | 87 | 109 | 54 | 47 | 38 | 335 |
| tft_load_log | 138 | 127 | 76 | 30 | 48 | 419 |

TFT has more under-prep events total (419 vs 335) and concentrates them in overnight + morning — matching Codex's offline finding (45% morning / 38% overnight). So the operational hazard TFT is suspected of (under-preparing for morning ramp) does show up structurally in this data.

But despite this, TFT-load **outperformed** LGBM-load on PnL — because LGBM's conservative bias was *more* economically costly than TFT's under-forecasting was operationally risky, **on this 3-day window**.

## What This 3-Day Result Does NOT Settle

- **3 days is small**. Run B v2 (6 weeks) will carry more weight.
- **model_a_hybrid is repaired-curve-broken**, so the production-equivalent price stack's load-source verdict is missing.
- **No terminal-value sensitivity yet** — Run C v2 will provide that.
- **Strategic SoC isn't load-aware**, so terminal-SoC differences across loads partly reflect this structural asymmetry.
- **`load_p65` (the actual production load surface) wasn't tested** — no history yet. The "LGBM" tested here is `load` q50, not the conservative production high.

## What to Compare in the Morning

When Run B v2 / Run C v2 land:

1. `amber_apf_lgbm` PnL across (actual, lgbm_load_log) for both runs — does load source still matter at 6-week scale?
2. Run B vs Run C — does terminal-value choice flip the verdict?
3. `model_a_hybrid` repaired-curve count — does the 6-week window show the same 100% repair rate? If so, it's not specific to recent days — it's a structural issue across the window.

## Files

- `eval/results/loadsrc_A_3day_{actual,lgbm_load_log,tft_load_log}_*.csv` (Run A outputs)
- `eval/results/loadsrc_A_3day_*.log` (combined Run A log)
- `eval/results/loadsrc_B_6week_v2_*.log` (Run B v2 in flight)
- `eval/results/loadsrc_C_6week_termval_v2_*.log` (Run C v2 in flight)
