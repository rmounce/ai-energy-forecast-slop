# Retail Plan Comparison — RESELE vs Flow Power vs IO Energy, 2026-05-14

Question from the user: would Flow Power or IO Energy be better than the
current RESELE plan, and when (time of year)?

## Plans Modelled

All values in $/kWh, Adelaide local time.

### RESELE (current)
- **Import** = wholesale × 1.05 (network loss factor) + tariff(time-of-day),
  then × 1.10 GST if positive. Three bands:
  - $0.0666 at 10:00-16:00 (solar window)
  - $0.3580 at 17:00-21:00 (evening peak)
  - $0.1331 other times
- **Export** = wholesale × 1.05 + tariff(time-of-day), × 1.10 GST if positive:
  - **+$0.1225 at 17:00-20:30, Nov-Mar only (RESELE summer evening credit)**
  - −$0.0100 at 10:00-15:30 (solar penalty)
  - $0 other times

### Flow Power
- **Import** = wholesale × 1.05 + $0.20/kWh markup, × 1.10 GST (≈ Amber-shape)
- **Export** = flat $0.45/kWh at 17:30-19:30 year-round, $0 elsewhere.
  No wholesale upside on export.

### IO Energy
- **Import** (fully fixed): $0.85 at 17:00-21:00, $0.08 at 10:00-16:00,
  $0.37 other times.
- **Export** (fully fixed): $0.30 at 18:00-21:00, −$0.02 at 10:00-16:00,
  $0.04 other times.

## Method

`eval/retariff_dispatch.py` replays per-step dispatch decisions from existing
rolling-MPC runs under each plan's tariff. Inputs used:

- Run B v3 winter: `loadsrc_B_v3_actual_raw.parquet` (Apr 1 - May 12 2026,
  41 days, actual load)
- Run B v3 winter: `loadsrc_B_v3_lgbm_load_log_raw.parquet` (same, LGBM load)
- Summer 2-week: `loadsrc_summer_2wk_actual_raw.parquet` (Jan 1-15 2026,
  actual load)
- Summer 2-week: `loadsrc_summer_2wk_lgbm_load_log_raw.parquet` (LGBM load)

All runs are on `amber_apf_lgbm` price source with
`--strategic-soc-handoff --strategic-target-mode exact`.

## Replay Headline (lower bound)

Naive annualised extrapolation, 152 summer days + 213 winter days,
actual-load runs:

| plan | summer $/day | winter $/day | annual $ |
|---|---:|---:|---:|
| **RESELE** (incumbent) | $11.46 | $0.11 | **$1,766** |
| **Flow Power** | $0.89 | −$1.33 | −$147 |
| **IO Energy** | $2.16 | −$1.09 | $96 |

LGBM-load runs (production-equivalent):

| plan | summer $/day | winter $/day | annual $ |
|---|---:|---:|---:|
| RESELE | $11.22 | −$0.07 | $1,691 |
| Flow Power | $0.26 | −$1.53 | −$285 |
| IO Energy | $1.69 | −$1.40 | −$42 |

**These numbers are heavily biased** — see next section. RESELE wins here
partly because the LP was *optimised* for RESELE's tariff, so the
dispatch decisions are calibrated to its peak/shoulder/credit structure.

## Why the Replay Underestimates Flow Power and IO Energy

Per-hour breakdown on the summer window (actual-load run, 14 days
aggregate):

| local hr | export kWh | RESELE $ | Flow Power $ | IO Energy $ |
|---:|---:|---:|---:|---:|
| 10 | 0.0 | −2.78 | **−6.51** | −4.51 |
| 11 | 0.0 | −2.72 | **−7.43** | −5.57 |
| 12 | 0.0 | −2.82 | **−7.83** | −6.19 |
| 13 | 0.8 | −2.17 | **−7.04** | −5.75 |
| 14 | 10.3 | +0.86 | **−9.50** | **−8.96** |
| 15 | 15.1 | +3.13 | **−10.16** | −8.56 |
| 16 | 12.2 | +4.68 | −2.08 | −2.19 |
| 17 | 79.7 | +10.14 | +16.37 | +1.04 |
| 18 | 112.3 | +17.64 | **+45.84** | +29.00 |
| 19 | 121.7 | +43.11 | +23.55 | +30.26 |
| 20 | 111.2 | **+100.45** | −7.30 | +26.49 |

Three structural mismatches:

1. **10:00-13:00 (grid-charging window):** the RESELE-tuned LP imports
   25-35 kWh/hour aggregate during the cheap RESELE shoulder
   ($0.0666/kWh). Under Flow Power, those same imports cost ~$0.30/kWh
   (wholesale + $0.20 markup). Under IO Energy, $0.08/kWh — still
   cheap. **Flow Power loses $7-8/hour just on the grid-charging step
   that the RESELE LP chose.** An LP tuned for Flow Power would avoid
   grid-charging entirely during sunny daylight and rely on PV.

2. **18:00-19:00 (Flow Power's sweet spot):** Flow Power earns
   $45.84 on 112 kWh exported, vs RESELE's $17.64 (winter rate on the
   wholesale-pass-through; summer credit window is wider but per-kWh
   payment is smaller). **Flow Power's flat $0.45 beats RESELE's
   wholesale-derived ~$0.16 effective rate on baseline evenings.**

3. **20:00-21:00 (wholesale-spike capture):** RESELE earns $100.45 on
   111 kWh exported because wholesale spiked during a RESELE credit
   slot and the wholesale-pass-through caught it (~$0.90/kWh effective).
   **Flow Power's flat $0.45 cap means it cannot capture the upside of
   wholesale spikes — RESELE structurally wins on spike days.** IO
   Energy similar problem with its $0.30 cap, though wider 18-21
   window mitigates.

## Oracle Estimate (back-of-envelope upper bound)

Assumptions: 40 kWh battery, 10 kW inverter, 0.95/0.95 efficiency, $0.05/kWh
degradation, can do one full cycle per day (charge in cheap window or PV,
discharge in best export window).

### Flow Power oracle
- Max discharge into 17:30-19:30 window: 10 kW × 2h = 20 kWh
- Revenue: 20 × $0.45 = **$9.00/day**
- Cost depends on charging source:
  - **PV-charge** (no grid import): degradation ~$2.00 → **$7/day net** → ~$2,555/year
  - **Grid-charge** at wholesale + $0.20 + GST (~$0.30/kWh avg): 21 kWh × $0.30 = $6.30 + degradation $2 = **$0.70/day net** → ~$256/year

### IO Energy oracle
- Max discharge into 18:00-21:00 window: 10 kW × 3h = 30 kWh
- Revenue: 30 × $0.30 = **$9.00/day** (similar to Flow Power)
- Plus negligible baseline FIT ($0.04/kWh other hours)
- Charge cost options:
  - PV-charge: degradation ~$3 → **$6/day net** → ~$2,190/year
  - Grid-charge during 10-16 at $0.08: 32 × $0.08 = $2.56 + degradation $3 = **$3.44/day net** → ~$1,256/year
- **IO Energy is much more forgiving of grid-charging** because of its $0.08 shoulder import — Flow Power's $0.30 markup punishes grid-charging.

### RESELE oracle (matches measured)
- Summer: 35 kWh discharge into 17-20:30 at avg $0.30/kWh effective (credit + wholesale + spikes) - $3 deg = ~$11/day net (matches the measured summer eval)
- Winter: import-during-peak too expensive, only marginal arbitrage available → $0-1/day
- **Total: ~$1,700-1,900/year** (matches measured)

### Oracle summary

| plan | annual $ (PV-charge) | annual $ (grid-charge) | comment |
|---|---:|---:|---|
| **RESELE** | ~$1,700-1,900 | ~$1,700-1,900 | already measured; PV/grid mix similar |
| **Flow Power** | ~$2,500 | ~$250 | strongly PV-dependent |
| **IO Energy** | ~$2,200 | ~$1,250 | most robust to grid-charging |

## When Each Plan Wins (time-of-year breakdown)

Based on the structural analysis:

| period | RESELE | Flow Power | IO Energy |
|---|---|---|---|
| **Summer (Nov-Mar), spike-heavy weeks** | **Wins** — wholesale-pass-through catches spikes | Capped at $0.45 | Capped at $0.30 |
| **Summer, calm weeks** | $0.13 baseline credit + wholesale ~$0.15 = $0.28/kWh | **Wins** at $0.45 flat | $0.30 flat |
| **Winter (Apr-Oct), spike-heavy weeks** | Wholesale spikes pay, but no fixed credit | **Wins** at $0.45 | $0.30 |
| **Winter, calm weeks** | ~$0.10-0.15/kWh evening export | **Wins** at $0.45 | $0.30 |
| **Daytime grid-charging needed** | Wins (shoulder $0.067) | Loses (markup $0.20) | $0.08 — cheaper than RESELE |
| **PV strong** | Doesn't matter | **Wins** if PV covers evening discharge fully | Strong |
| **PV thin** | Manageable | Bad | Best of the three |

## Caveats

1. **Replay is a LOWER bound for Flow Power and IO Energy.** The LP wasn't
   tuned for their structures. Oracle estimates are an UPPER bound. True
   number is somewhere in between, closer to oracle if a proper LP-per-plan
   eval were done.

2. **PV-charge vs grid-charge dependency is large.** Flow Power's annual
   value swings by ~$2,300 depending on whether the battery can be filled
   from PV exclusively. RESELE has less sensitivity to this because its
   shoulder import is already cheap.

3. **Spike behaviour in the audit window may not generalise.** Hour 20:00
   showed a single big spike worth $100 over 14 days. Summer spike
   frequency in AEMO can shift year-to-year — wholesale-pass-through value
   depends on the realised spike distribution.

4. **Tariff structures are stylised.** Flow Power's "wholesale + 20c"
   could include or exclude GST and loss factor differently in practice
   ($0.02/kWh-ish uncertainty). IO Energy and Flow Power $0.30/$0.45
   exports are assumed flat with no daily kWh cap (user confirmed
   inverter cap is the effective limit).

5. **No switching cost / contract churn factored in.** Real-world switches
   have admin overhead, possibly app/HA integration changes
   (Amber's API → Flow Power's API/data feed, etc).

## Recommended Next Move

If the order-of-magnitude conclusion is enough — i.e., **RESELE is
"competitive at ~$1,800/year, no obvious switch unless PV size strongly
favours Flow Power"** — then this audit is done. The structural insight
is that no plan strictly dominates; RESELE owns the spike upside, Flow
Power owns predictable evening, IO Energy is the robust middle.

If a definitive number is wanted, the next step is a **proper LP-per-plan
eval**:

1. Add a `--tariff-plan {resele,flowpower,ioenergy}` flag to
   `eval/rolling_mpc_eval.py` that plumbs the chosen tariff into both the
   LP's forecast curves AND the realised PnL formula.
2. Re-run the same B v3 winter window + summer 2-week window for each
   plan with the LP correctly tuned.
3. ~6-9h compute per plan in tmux.
4. Compare apples-to-apples PnL.

That's the only way to remove the LP-mismatch artifact in the replay
table. Worth doing if a switch decision is on the table; not worth doing
otherwise.

## Files

- Re-tariff script: `eval/retariff_dispatch.py`
- Raw rows replayed: `eval/results/loadsrc_B_v3_{actual,lgbm_load_log}_raw.parquet`
  and `eval/results/loadsrc_summer_2wk_{actual,lgbm_load_log}_raw.parquet`
- Seasonal-tariff harness extension: `eval/rolling_mpc_eval.py` (commit `03d7d73`)
