# Crossed Counterfactual Pilot — 2026-04-25

Purpose: document the first short-window crossed tactical/strategic counterfactual runs under
the rolling MPC harness, using the recovered snapshot-backed Run 011b-era TFT asset.

These runs are intended as a **diagnostic decomposition**, not an architecture verdict.

Related documents:
- [rolling_eval_fidelity_full_windows_2026-04-25.md](./rolling_eval_fidelity_full_windows_2026-04-25.md)
- [fresh_independent_strategy_review_followup_2026-04-25.md](./fresh_independent_strategy_review_followup_2026-04-25.md)
- [roadmap.md](./roadmap.md)

---

## 1. Asset Caveat

The strategic TFT asset used here was loaded directly from a ZFS snapshot rather than from the
current repo-local `models/tft_price/` directory.

Recovered local copies now exist under explicit provisional names:
- [models/tft_price/checkpoint_run011b_snapshot_candidate.pt](../models/tft_price/checkpoint_run011b_snapshot_candidate.pt)
- [models/tft_price/scalers_run011b_snapshot_candidate.pkl](../models/tft_price/scalers_run011b_snapshot_candidate.pkl)

Snapshot-backed candidate:
- checkpoint:
  - `/home/saltspork/.zfs/snapshot/autosnap_2026-04-17_00:00:10_daily/src/ai-energy-forecast-slop/models/tft_price/checkpoint_best.pt`
- scalers:
  - `/home/saltspork/.zfs/snapshot/autosnap_2026-04-17_00:00:10_daily/src/ai-energy-forecast-slop/models/tft_price/scalers.pkl`

Why this appears to be the correct Run 011b-era asset:
- checkpoint metadata matches the documented Run 011b note:
  - best epoch `5`
  - best val loss `0.0538`
  - 15-feature decoder contract
- snapshot `evaluation_results.csv` matches the documented Run 011b horizon table
- the current rolling harness can load the snapshot checkpoint/scalers directly and complete a
  smoke run

Still, this should be treated as a **recovered incumbent candidate** until it has been promoted
into an explicit local artifact path and used in a fuller validation pass.

---

## 2. Experiment Design

Runs launched:

### Window B, `netload_tariffed`, 2-day
- `2025-09-01 -> 2025-09-03`

### Window A, `netload_tariffed`, 2-day
- `2025-07-21 -> 2025-07-23`

### Window B, `price_only`, 2-day
- `2025-09-01 -> 2025-09-03`

### Window B, `netload_tariffed`, 7-day
- `2025-09-01 -> 2025-09-08`

Source matrix:
- `amber_apf_lgbm`
- `model_a_hybrid`
- `hybrid_tactical_amber_strategic`
- `amber_tactical_hybrid_strategic`

Interpretation of crossed sources:
- `hybrid_tactical_amber_strategic`
  - tactical curve from Hybrid
  - strategic handoff from Amber
- `amber_tactical_hybrid_strategic`
  - tactical curve from Amber
  - strategic handoff from Hybrid

---

## 3. Results

## Window B — `netload_tariffed`

- `amber_apf_lgbm`: **$12.601 total / $6.311 per day**
- `amber_tactical_hybrid_strategic`: **$12.169 total / $6.095 per day**
- `hybrid_tactical_amber_strategic`: **$11.480 total / $5.750 per day**
- `model_a_hybrid`: **$11.457 total / $5.739 per day**

Relative to Amber:
- Amber tactical + Hybrid strategic: **-$0.216/day**
- Hybrid tactical + Amber strategic: **-$0.562/day**
- Hybrid tactical + Hybrid strategic: **-$0.573/day**

## Window A — `netload_tariffed`

- `amber_apf_lgbm`: **-$3.061 total / -$1.533 per day**
- `amber_tactical_hybrid_strategic`: **-$3.326 total / -$1.666 per day**
- `model_a_hybrid`: **-$3.787 total / -$1.897 per day**
- `hybrid_tactical_amber_strategic`: **-$3.964 total / -$1.985 per day**

Relative to Amber:
- Amber tactical + Hybrid strategic: **-$0.133/day**
- Hybrid tactical + Hybrid strategic: **-$0.364/day**
- Hybrid tactical + Amber strategic: **-$0.452/day**

## Window B — `price_only`

- `amber_apf_lgbm`: **$19.163 total / $9.598 per day**
- `amber_tactical_hybrid_strategic`: **$18.775 total / $9.404 per day**
- `hybrid_tactical_amber_strategic`: **$19.099 total / $9.566 per day**
- `model_a_hybrid`: **$19.482 total / $9.758 per day**

Relative to Amber:
- Amber tactical + Hybrid strategic: **-$0.194/day**
- Hybrid tactical + Amber strategic: **-$0.032/day**
- Hybrid tactical + Hybrid strategic: **+$0.160/day**

## Window B — `netload_tariffed`, 7-day

- `amber_apf_lgbm`: **$10.325 total / $1.476 per day**
- `amber_tactical_hybrid_strategic`: **$9.588 total / $1.370 per day**
- `hybrid_tactical_amber_strategic`: **$6.765 total / $0.967 per day**
- `model_a_hybrid`: **$5.885 total / $0.841 per day**

Relative to Amber:
- Amber tactical + Hybrid strategic: **-$0.105/day**
- Hybrid tactical + Amber strategic: **-$0.509/day**
- Hybrid tactical + Hybrid strategic: **-$0.635/day**

---

## 4. Interpretation

The main signal from the tariffed gate is:

- swapping **Amber strategic -> Hybrid strategic** only hurts Amber modestly
- swapping **Hybrid strategic -> Amber strategic** helps Hybrid only marginally

That pattern holds on both 2-day tariffed pilots and strengthens on the longer 7-day Window B run.

### Window B read

In Window B `netload_tariffed`:
- Hybrid with Amber strategic target improves only from **$5.739/day** to **$5.750/day**
- Amber with Hybrid strategic target degrades from **$6.311/day** to **$6.095/day**

So the tactical curve appears to dominate the strategic handoff on this slice.

### Window B, 7-day read

The longer Window B run strengthens the same conclusion:
- Hybrid with Amber strategic improves only from **$0.841/day** to **$0.967/day**
- Amber with Hybrid strategic degrades only from **$1.476/day** to **$1.370/day**

So even on the longer slice, the first-order deficit still looks much more tactical than
strategic.

### Window A read

In Window A `netload_tariffed`:
- Amber tactical remains clearly best even with Hybrid strategic
- Hybrid tactical remains materially worse even with Amber strategic

Window A therefore agrees with Window B: the strategic target matters, but it appears
second-order relative to tactical curve quality under the tariffed gate.

### `price_only` contrast

Window B `price_only` still gives a much friendlier answer for Hybrid:
- full Hybrid beats Amber
- the crossed-source ranking is materially different

That reinforces the earlier conclusion:
- `price_only` remains useful as a decomposition/debug lens
- but it is not a trustworthy production architecture gate

---

## 5. 7-Day Tactical Residual Read

The richer tariffed diagnostics on the 7-day Window B run make the tactical read substantially
stronger.

### Same strategic target, different tactical curve

Comparing:
- `amber_tactical_hybrid_strategic`
- `model_a_hybrid`

holds the **Hybrid strategic target fixed** and changes only the tactical curve.

Result:
- Amber tactical + Hybrid strategic: **$9.588 total / $1.370 per day**
- Hybrid tactical + Hybrid strategic: **$5.885 total / $0.841 per day**
- same final SoC: **`26.589 kWh`** for both

So the tactical-only gap is about:
- **+$3.70 total**
- **+$0.53/day**

This is important because it is not explained by:
- different final inventory posture
- a more favorable strategic terminal target

### Tariffed economic decomposition

Under that same-target tactical comparison:

- `amber_tactical_hybrid_strategic`
  - import cost: **$7.28**
  - export revenue: **$35.47**
- `model_a_hybrid`
  - import cost: **$10.11**
  - export revenue: **$34.45**

So Amber tactical beats Hybrid tactical through both:
- about **$2.83** lower import cost
- about **$1.03** higher export revenue

with only a small degradation-cost offset.

That means the problem is not merely weaker export timing. The tactical damage shows up on both
the import and export side of the tariffed objective.

### Daily same-target tactical delta

The same-target daily delta table shows:
- Amber tactical beats Hybrid tactical on **6 of 7 days**
- largest positive day: **2025-09-06**, about **+$2.28**
- Hybrid tactical wins only **2025-09-05**, and only by about **$0.20**

So the tactical underperformance is broad across the week rather than being dominated by one
pathological interval.

### Missed export intervals

The missed-export report is also revealing.

Top missed-export intervals cluster around **2025-09-01 10:00–11:20 UTC**:
- actual feed-in prices are high, roughly **$410–$676/MWh**
- Amber exports about **7.9–8.4 kW**
- Hybrid exports **0 kW**
- Amber discharges at **10 kW**
- Hybrid only discharges around **1.17–1.68 kW**
- strategic targets are effectively **zero for both**
- `forecast_step0_mwh` is also the **same** in these rows

So these are not simple:
- strategic-target differences
- step-0 wholesale-price differences

Instead, they point more toward:
- the shape of the tactical forward curve across the next **1h / 4h / 14h**
- and how that curve interacts with the tariffed LP objective

Diagnostic outputs for this 7-day run were written to:
- [rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_20260425_diagnostics_overall.csv](../eval/results/rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_20260425_diagnostics_overall.csv)
- [rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_20260425_diagnostics_same_target_daily_delta.csv](../eval/results/rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_20260425_diagnostics_same_target_daily_delta.csv)
- [rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_20260425_diagnostics_missed_export.csv](../eval/results/rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_20260425_diagnostics_missed_export.csv)

---

## 6. Provisional Conclusion

These short counterfactual pilots suggest:

1. The tariffed loss is **more tactical-curve-shaped than strategic-target-shaped** on both the
   short pilots and the longer 7-day Window B confirmation run.
2. The strategic handoff is not irrelevant, but it does not appear to be the dominant source of
   the current Amber-vs-Hybrid gap on these slices.
3. The 7-day same-target tactical comparison now shows that the gap persists even with the same
   strategic target and same final SoC, which makes the tactical curve / tactical objective
   interaction the clearest current bottleneck.
4. The main next diagnostic priority should therefore remain near-horizon monetization quality,
   not another round of bridge-only strategic-target tuning.

This is still a **provisional diagnostic result**, not a final architecture verdict. But the
7-day Window B run is now a meaningful confirmation step, and it points in the same direction as
the 2-day pilots.

---

## 7. Immediate Next Questions

1. Why does the Hybrid tactical curve create both higher import cost and lower export revenue
   under the tariffed objective?
2. Are the most important tactical failures:
   - feed-in/export opportunity miss,
   - weak discharge urgency,
   - or tariff-aware curve shape/calibration?
3. Should the next intervention focus first on:
   - tactical curve calibration under tariffed economics,
   - export-aware tactical quantile/curve shaping,
   - or separate effective buy/sell tactical curves?
