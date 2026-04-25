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

---

## 4. Interpretation

The main signal from the tariffed gate is:

- swapping **Amber strategic -> Hybrid strategic** only hurts Amber modestly
- swapping **Hybrid strategic -> Amber strategic** helps Hybrid only marginally

That pattern holds on both 2-day tariffed pilots.

### Window B read

In Window B `netload_tariffed`:
- Hybrid with Amber strategic target improves only from **$5.739/day** to **$5.750/day**
- Amber with Hybrid strategic target degrades from **$6.311/day** to **$6.095/day**

So the tactical curve appears to dominate the strategic handoff on this slice.

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

## 5. Provisional Conclusion

These short counterfactual pilots suggest:

1. The tariffed short-window loss is **more tactical-curve-shaped than strategic-target-shaped**.
2. The strategic handoff is not irrelevant, but it does not appear to be the dominant source of
   the current Amber-vs-Hybrid gap on these slices.
3. The main next diagnostic priority should therefore remain near-horizon monetization quality,
   not another round of bridge-only strategic-target tuning.

This is still a **pilot result**, not a final architecture verdict. The next confirmation step
should be the same crossed counterfactual matrix on a longer Window B run using the same
snapshot-backed Run 011b candidate.

---

## 6. Immediate Next Questions

1. Does the tactical-dominant pattern persist on a longer Window B run?
2. If so, where exactly does Hybrid’s tactical curve miss Amber’s monetization?
   - missed export windows
   - weaker sell timing
   - weaker discharge magnitude
3. Does an oracle strategic target materially improve Hybrid once the tactical curve is held
   fixed?
4. Should the next intervention focus on:
   - tactical forecast shape,
   - tactical objective formulation,
   - or only then the strategic valuation layer?
