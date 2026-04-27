## Tariff-Aware Tier 1 Candidate — First Clean A/B

Date: `2026-04-27`

Purpose: record the first clean short-window comparison between the legacy Tier 1 tactical
model and the first tariff-aware Tier 1 candidate under the `netload_tariffed` rolling gate.

Scope label: this is a **current-tariff counterfactual backtest**, not a claim about exact
historical tariff truth. Effective import/export features were reconstructed from the current
deterministic tariff contract rather than from a separately persisted historical tariff ledger.

### Context

The heuristic shaping branch has now been closed. The next smallest justified intervention is
to make tariff asymmetry explicit to Tier 1 rather than continuing to hand-shape tactical
sell behavior.

The first tariff-aware candidate, `lgbm_tactical_tariffaware_v1`, adds deterministic
tariff-derived features reconstructed from the current tariff contract, including:

- current effective import price
- current effective feed-in price
- short-horizon effective import price summaries
- short-horizon effective feed-in price summaries

The candidate was trained in:

- [models/lgbm_tactical_tariffaware_v1](../models/lgbm_tactical_tariffaware_v1)

### Compatibility note

Adding tariff-aware features changed the tactical feature contract from the legacy
`25`-column long matrix to a new `33`-column long matrix.

To allow fair side-by-side rolling evaluation, inference now adapts the feature matrix to the
target tactical model:

- legacy models receive the old `25`-column contract
- tariff-aware models receive the new `33`-column contract

This avoids the earlier LightGBM feature-shape mismatch and lets the old and new Tier 1 models
be compared under the same `rolling_mpc_eval.py` harness.

### Offline signal

Training metrics were mixed but plausible:

- baseline validation `q50` MAE: `21.08`
- tariff-aware candidate validation `q50` MAE: `21.21`
- baseline stratified-eval `q50` MAE: `109.96`
- tariff-aware candidate stratified-eval `q50` MAE: `108.43`

Interpretation:

- ordinary validation MAE did not improve
- the harder stratified hold-out improved a little
- therefore the candidate should be judged primarily by the `netload_tariffed` rolling gate

### Rolling comparison

All runs below use:

- `netload_tariffed`
- exact strategic handoff
- same source pair:
  - `amber_tactical_hybrid_strategic`
  - `model_a_hybrid`
- recovered snapshot-backed `011b` candidate for Tier 2 strategic forecasting

#### Window B, 2-day (`2025-09-01 -> 2025-09-03`)

Amber tactical baseline:

- `amber_tactical_hybrid_strategic`: `6.085474/day`

Legacy Tier 1:

- `model_a_hybrid`: `5.716788/day`
- gap to Amber tactical: `-0.368686/day`
- final SoC: `23.981213 kWh`

Tariff-aware Tier 1 candidate:

- `model_a_hybrid`: `5.994913/day`
- gap to Amber tactical: `-0.090561/day`
- improvement vs legacy Tier 1: `+0.278125/day`
- final SoC: `23.969908 kWh`

Read:

- the tariff-aware candidate materially narrows the Window B tactical gap
- on this slice it gets very close to Amber tactical while ending with almost the same final SoC

#### Window A, 2-day (`2025-07-21 -> 2025-07-23`)

Amber tactical baseline:

- `amber_tactical_hybrid_strategic`: `-1.657865/day`

Legacy Tier 1:

- `model_a_hybrid`: `-1.908514/day`
- gap to Amber tactical: `-0.250649/day`
- final SoC: `23.593562 kWh`

Tariff-aware Tier 1 candidate:

- `model_a_hybrid`: `-1.952173/day`
- gap to Amber tactical: `-0.294308/day`
- change vs legacy Tier 1: `-0.043659/day`
- final SoC: `23.593562 kWh`

Read:

- Window A does not improve
- the candidate is slightly worse than the legacy Tier 1 on this slice

### Current interpretation

This first tariff-aware Tier 1 pass is a meaningful positive result, but not yet a general fix.

What it shows:

1. Explicit tariff-derived tactical features can materially improve the export-heavy Window B
   failure mode without changing the tactical model architecture.
2. The same first-pass feature set does not yet solve the broader Window A weakness.
3. The tariff-aware direction looks justified enough to continue, but only with longer-window
   confirmation and more careful regime analysis before claiming success.

### Longer-window confirmation

The first longer-window A/B confirmation used the same `netload_tariffed` gate and the same
exact strategic handoff setup.

#### Window B, 7-day (`2025-09-01 -> 2025-09-08`)

Amber tactical baseline:

- `amber_tactical_hybrid_strategic`: `1.384788/day`

Legacy Tier 1:

- `model_a_hybrid`: `0.823249/day`

Tariff-aware Tier 1 candidate:

- `model_a_hybrid`: `0.897313/day`
- improvement vs legacy Tier 1: `+0.074065/day`
- remaining gap to Amber tactical: `-0.487475/day`
- gap closure vs legacy: about `13.2%`

Important control detail:

- legacy final SoC: `26.589069 kWh`
- candidate final SoC: `26.589069 kWh`

So the candidate improvement is not explained by simply finishing with a different terminal
inventory posture.

Decomposition relative to legacy Tier 1:

- import cost: improved slightly (`-0.0579`)
- export revenue: improved more materially (`+0.4776`)
- degradation cost: slightly worse (`+0.0173`)

Read:

- the 7-day Window B gain is real
- it still looks export-side and tariff-related
- but it is much smaller than the short 2-day Window B result suggested

Day concentration:

- most of the gain comes from `2025-09-01` (`+0.555/day` vs legacy)
- smaller positive gain on `2025-09-05`
- slight regressions on `2025-09-04`, `2025-09-06`, and `2025-09-07`

Regime split:

- spike improves materially (`13.198 -> 13.755`)
- low improves only slightly
- normal gets slightly worse

This suggests the first tariff-aware pass is learning the export-heavy spike regime better, but
is not yet broadly improving tactical behavior across all market conditions.

Structural contrast between Window B and Window A also supports that read.

Window B candidate raw rows (`2016` steps) had:

- mean actual feed-in price: `71.45 $/MWh`
- feed-in `p90`: `217.25 $/MWh`
- feed-in `p99`: `638.95 $/MWh`
- negative net-load share: `36.9%`
- feed-in `>= 300`: `6.0%` of steps
- feed-in `>= 500`: `2.18%` of steps

Window A candidate raw rows (`2016` steps) had:

- mean actual feed-in price: `45.89 $/MWh`
- feed-in `p90`: `149.02 $/MWh`
- feed-in `p99`: `304.27 $/MWh`
- negative net-load share: `25.9%`
- feed-in `>= 300`: `1.14%` of steps
- feed-in `>= 500`: `0.15%` of steps

So Window B contains materially more frequent and more intense high-value export conditions
than Window A. That makes the current tariff-aware gain look more like a regime-boundary
effect than a broad tactical uplift.

Per-day high-feed-in counts reinforce the same picture.

Window B:

- `2025-09-01`: `98` steps with feed-in `>= 300`, `41` with feed-in `>= 500`
- `2025-09-02`: `5` / `2`
- `2025-09-03`: `12` / `0`
- `2025-09-04`: `6` / `1`
- `2025-09-05` to `2025-09-07`: `0` / `0`

Window A:

- `2025-07-21`: `0` / `0`
- `2025-07-22`: `0` / `0`
- `2025-07-23`: `15` / `0`
- `2025-07-24`: `2` / `2`
- `2025-07-25`: `4` / `1`
- `2025-07-26`: `0` / `0`
- `2025-07-27`: `2` / `0`

#### Window A, 7-day (`2025-07-21 -> 2025-07-28`)

Amber tactical baseline:

- `amber_tactical_hybrid_strategic`: `-1.175585/day`

Legacy Tier 1:

- `model_a_hybrid`: `-1.542099/day`

Tariff-aware Tier 1 candidate:

- `model_a_hybrid`: `-1.569593/day`
- change vs legacy Tier 1: `-0.027494/day`
- remaining gap to Amber tactical: `-0.394008/day`

Decomposition relative to legacy Tier 1:

- import cost: worse (`+0.0993`)
- export revenue: worse (`-0.1464`)
- degradation cost: slightly better (`-0.0534`)
- final SoC: higher by `+0.638 kWh`

Read:

- Window A is still not improved
- the regression is small, but it is not a clean win even on the longer slice

Regime split:

- low gets worse
- spike also gets slightly worse

So this first tariff-aware pass is not just failing on one pathological interval. It still adds
some wrong shape outside the Window B-style export-opportunity regime.

Daily deltas vs legacy also suggest a regime-boundary problem rather than a broad uplift.

Window B candidate-minus-legacy deltas:

- `2025-09-01`: `+0.555`
- `2025-09-02`: `+0.000`
- `2025-09-03`: `+0.006`
- `2025-09-04`: `-0.005`
- `2025-09-05`: `+0.064`
- `2025-09-06`: `-0.058`
- `2025-09-07`: `-0.044`

Window A candidate-minus-legacy deltas:

- `2025-07-21`: `-0.096`
- `2025-07-22`: `+0.009`
- `2025-07-23`: `+0.017`
- `2025-07-24`: `-0.068`
- `2025-07-25`: `+0.038`
- `2025-07-26`: `-0.006`
- `2025-07-27`: `-0.087`

That is:

- Window B improvement is dominated by a single very strong export-opportunity day
- Window A does not show a coherent positive pattern at all
- the current candidate therefore looks better described as a partial regime detector than as a
  generally stronger tactical forecast

### Updated interpretation

The first tariff-aware Tier 1 candidate remains a meaningful positive signal, but it is now
clearer what kind of signal it is.

What the longer runs say:

1. Explicit tariff-derived tactical features are attacking the right failure family.
2. The candidate meaningfully helps the export-heavy Window B regime, and that gain survives
   to a 7-day confirmation.
3. But the gain is much more moderate than the short 2-day result suggested, and it remains
   concentrated in spike/export-opportunity conditions.
4. Window A does not improve, so `tariffaware_v1` is not yet a general replacement for the
   legacy Tier 1 tactical model.

The current best read is therefore:

- the tariff-aware tactical direction is justified
- the first feature-only pass is not sufficient
- the next refinement should be judged on whether it preserves the Window B gain while reducing
  the Window A regression

Reviewer / implementer critical read after this checkpoint converged on the following:

- `tariffaware_v1` is best understood as a **partial regime detector**
- the branch is alive, but the candidate is not yet a win
- the strongest claim that remains justified is:
  - tariff-aware features help the tactical model recognize and exploit some high-export-value
    conditions
  - but they do not yet create a generally stronger tactical model

The main remaining uncertainty is now:

- does the candidate generalize to export-opportunity conditions beyond the very strong
  `2025-09-01`-style Window B day?
- or is it essentially a narrow-event detector dressed up as tariff awareness?

That means the next branch-decision should be based on **generalization falsification**, not on
another immediate feature sweep.

### Next recommended steps

1. Run a cheap falsification pass before another implementation branch:
   - Window B `7-day` excluding `2025-09-01`
   - and/or a moderate-FIT middle window between Window A and Window B
2. Compare:
   - legacy Hybrid
   - `tariffaware_v1`
   on those slices using the same `netload_tariffed` gate
3. If the candidate still helps on non-extreme export-opportunity days while staying neutral
   elsewhere, the branch is genuinely regime-sensitive and remains promising.
4. If the gain disappears off the flagship Window B day, treat `tariffaware_v1` as a narrow
   event detector and be more cautious about adding a calibrator on top of it.
5. Only after that should the next implementation branch be chosen:
   - small post-forecast tactical calibrator with genuine inference-time information advantage,
     or
   - another targeted tariff-aware feature reformulation if the candidate proves too narrow.
