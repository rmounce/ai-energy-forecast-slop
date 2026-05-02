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

### Overnight falsification result

The overnight falsification slices came back strongly negative for `tariffaware_v1` as a
general tactical improvement.

#### Window B excluding `2025-09-01`
`2025-09-02 -> 2025-09-09`

- `amber_tactical_hybrid_strategic`: `-0.353183/day`
- legacy `model_a_hybrid`: `-0.695252/day`
- `tariffaware_v1`: `-0.691327/day`

So the candidate improves on legacy by only `+0.003924/day`, which closes about `1.1%` of the
Amber gap.

Decomposition vs legacy:

- import cost: slightly better (`-0.1260`)
- export revenue: slightly worse (`-0.0909`)
- degradation: slightly worse (`+0.0076`)
- final SoC: higher by `+0.194 kWh`

The earlier Window B gain therefore mostly disappears once the flagship `2025-09-01` day is
removed.

#### Moderate-FIT middle window
`2025-08-12 -> 2025-08-19`

- `amber_tactical_hybrid_strategic`: `0.224917/day`
- legacy `model_a_hybrid`: `-0.217169/day`
- `tariffaware_v1`: `-0.213425/day`

So the candidate improves on legacy by only `+0.003743/day`, which closes about `0.8%` of the
Amber gap.

Decomposition vs legacy:

- import cost: slightly better (`-0.1632`)
- export revenue: worse (`-0.1899`)
- degradation: slightly better (`-0.0528`)
- final SoC: unchanged

This means the candidate does **not** meaningfully generalize to a middle regime with more
export opportunity than Window A but much less than the flagship Window B day.

### Updated conclusion after falsification

After the overnight falsification runs, the most defensible label for `tariffaware_v1` is now:

- **single-event-sensitive probe**

rather than:

- weak positive candidate
- partial regime detector
- or generally improved tactical model

The branch-level conclusion is now:

1. The broader tariff-aware tactical idea is still conceptually relevant.
2. But `tariffaware_v1` should not be extended with more feature sweeps.
3. A calibrator on top of `tariffaware_v1` is not justified, because the underlying signal has
   not generalized enough.

The current best read from reviewer and implementer feedback is that this is now a
**formulation problem**, not a simple feature-gap problem.

Likely issue:

- tariff features were added to a model still trained to predict price quantiles
- so the model can see tariff asymmetry
- but the loss still rewards price accuracy, not tariffed action value

### New next step

The next branch should be an explicit **oracle-action / action-regret diagnostic dataset**,
not another Tier 1 feature pass.

For each rolling step, under the same SoC and strategic target, record:

- what the tariffed LP would do with actual future import/export prices
- what Hybrid did
- what Amber did
- missed export value
- bad import cost
- value of charging more / less
- value of discharging more / less

Then decide whether the next model should predict:

- discharge uplift
- charge suppression
- action ranking
- or a corrected buy/sell curve

Decision logic:

- if the action errors are learnable from available real-time features, then a tactical
  calibrator or action model is justified
- if they mostly require future information, then the remaining Amber gap is primarily
  forecast-information quality rather than tactical correction

### First oracle-action read

The first oracle-action dataset runs materially changed the interpretation of this branch.

The question tested was narrower than overall PnL:

- with the same logged SoC and terminal constraints,
- and with actual future tariffed prices/net load fed to the LP,
- is Hybrid or Amber closer to the oracle **first action**?

This is not the final label family, but it is a useful falsification check for the simple
story that “Hybrid should just act more like Amber in export-heavy intervals.”

#### Window B, 7-day oracle first-action comparison

Across all `2016` steps:

- Hybrid closer to oracle: `11.4%`
- Amber closer to oracle: `10.0%`
- equal: `78.6%`

High feed-in subsets were more surprising:

- feed-in `>= 300`: Hybrid closer `30.6%`, Amber closer `2.5%`, equal `66.9%`
- feed-in `>= 500`: Hybrid closer `38.6%`, Amber closer `2.3%`, equal `59.1%`

So on the same high-FIT rows that originally made Amber look economically stronger, the
realized-future oracle first action was actually much more often closer to Hybrid than to
Amber.

#### Other windows

The same pattern weakens or disappears away from the flagship regime:

- Window B excluding `2025-09-01`: near-flat, overwhelmingly equal, tiny differences
- moderate-FIT middle window: Hybrid slightly closer overall, but still mostly equal
- Window A 7-day: no clear winner; Hybrid and Amber are both close and mostly tied

#### What this does and does not mean

This does **not** prove that Hybrid is economically better overall. It only says:

- the oracle-action dataset does not support the simple hypothesis that Amber wins because its
  first action is systematically more oracle-like on export-heavy rows
- a large part of the Amber-vs-Hybrid gap must therefore come from something subtler than
  “under-export now”

Likely alternatives now include:

1. multi-step path differences that are not captured by first-step imitation
2. forecast-information quality differences rather than local tactical correction
3. a better label family based on state-transition value or multi-step action regret

Practical implication:

- do **not** use raw first-step Amber imitation as the next training target
- use the oracle dataset to construct richer labels around multi-step regret / state value
  instead

Implementation note:

- the oracle dataset builder now records not only first-action deltas, but also
  **full-horizon forced-first-action objective regret** for both Hybrid and Amber
  actions under the realized future tariffed path
- the first stitched implementation of that regret field had an accounting inconsistency and was
  replaced with a stricter definition: solve the full horizon again with the first action pinned

### Corrected oracle-regret read (`v3`)

The corrected full-horizon oracle-regret rebuilds mostly **confirmed** the earlier first-action
read rather than reversing it.

#### Window B, 7-day

Overall:

- Hybrid closer to oracle first action: `11.4%`
- Amber closer to oracle first action: `10.0%`
- equal: `78.6%`
- mean Hybrid forced-first-action regret: `0.00541`
- mean Amber forced-first-action regret: `0.00605`

So on the corrected full-horizon label, Hybrid is still slightly better overall on this
oracle-first-action lens.

High-FIT subsets were the strongest signal:

- feed-in `>= 300`
  - Hybrid closer: `30.6%`
  - Amber closer: `2.5%`
  - Hybrid regret: `0.0153`
  - Amber regret: `0.0411`
- feed-in `>= 500`
  - Hybrid closer: `38.6%`
  - Amber closer: `2.3%`
  - Hybrid regret: `0.00566`
  - Amber regret: `0.0567`

So on the same export-heavy rows that originally made Amber look economically stronger, the
corrected realized-future oracle still says Amber's **first action** is less oracle-like than
Hybrid's.

#### Other windows

- Window B excluding `2025-09-01`
  - Hybrid regret: `0.00479`
  - Amber regret: `0.00426`
  - Amber slightly better once the flagship day is removed
- moderate-FIT middle window
  - Hybrid regret: `0.00450`
  - Amber regret: `0.00516`
  - Hybrid slightly better overall, but not strongly
- Window A 7-day
  - Hybrid regret: `0.00372`
  - Amber regret: `0.00340`
  - Amber slightly better overall

#### Updated interpretation

This narrows the conclusion further:

1. The remaining Amber-vs-Hybrid gap is **not** well described as “Hybrid should imitate
   Amber’s first action.”
2. In the strongest high-FIT Window B regime, Amber’s first action actually leaves **more**
   full-horizon tariffed value on the table than Hybrid’s.
3. Amber remains slightly better on Window A and on the non-flagship Window B slice, so the
   answer is not “Hybrid is simply better.” The gap must come from something subtler.

The most plausible remaining explanations are now:

- multi-step path effects beyond first-action imitation
- forecast-information quality rather than local first-action correction
- a richer label family based on state-transition value or multi-step regret

Practical implication:

- do **not** train the next model to copy Amber’s first action
- if a corrective modeling branch is pursued, the label should be richer than first-action
  oracle regret alone

### Updated next-step consensus

Reviewer and implementer feedback after the corrected `v3` oracle-regret pass converged on a
more specific next diagnostic sequence.

What is now ruled out:

- a first-action calibrator as the next primary abstraction
- a model trained to make Hybrid imitate Amber’s immediate dispatch action

Why:

- Hybrid is already slightly better on the corrected first-action oracle-regret lens in the
  strongest Window B high-FIT regime
- yet Amber still wins economically over some broader slices
- so the remaining gap is not well explained by “wrong first command”

Most justified next diagnostic pair:

1. **Forward-curve shape comparison** on the high-FIT intervals where Amber wins economically.
   The key question is whether Hybrid gets step 0 roughly right but mean-reverts too quickly
   over steps `2–10`, while Amber’s tactical curve stays elevated longer.

2. **Forced-prefix regret** with pinned prefixes of length:
   - `N = 1`
   - `N = 3`
   - `N = 6`
   - `N = 12`
   and optionally longer if runtime is acceptable.

Interpretation logic:

- if Amber only becomes better when several initial actions are pinned, the issue is more about
  multi-step tactical path / inventory trajectory value than about a local first-action fix
- if the curves already differ in their short-horizon persistence shape, a narrower tactical
  persistence intervention may be more justified than a wholly new target
- if neither the curve shape nor the pinned-prefix path reveals Amber’s advantage, the next
  place to look is forecast-update / receding-horizon behavior rather than local tactical labels

### Forced-prefix result on `wb7`

The first real forced-prefix batch on Window B `7-day` (`N = 1, 3, 6, 12`) sharpened the story
again.

Overall, Amber becomes increasingly better as more of the path is pinned:

- `N = 1`: Amber advantage `-0.00043`
- `N = 3`: Amber advantage `-0.00150`
- `N = 6`: Amber advantage `-0.00351`
- `N = 12`: Amber advantage `-0.01147`

Negative here means Amber has lower forced-prefix regret than Hybrid.

So the reviewer’s core hypothesis was right:

- Amber’s advantage is **not** mainly in the first action
- it emerges more clearly over a multi-step prefix

But the high-FIT subset result went the other way:

- on feed-in `>= 300`, Amber is worse at every prefix length
- on feed-in `>= 500`, Amber is worse by an even larger margin

So the path advantage is **not** coming from the flagship high-FIT rows.

The first useful non-high-FIT split shows where the Amber edge is actually accumulating.

For feed-in `< 300`:

- `N = 1`: Amber advantage `-0.00087`
- `N = 3`: Amber advantage `-0.00251`
- `N = 6`: Amber advantage `-0.00480`
- `N = 12`: Amber advantage `-0.01364`

And that is driven mainly by the **negative-net-load** portion of the same subset:

- `N = 1`: Amber advantage `-0.00389`
- `N = 3`: Amber advantage `-0.01168`
- `N = 6`: Amber advantage `-0.02287`
- `N = 12`: Amber advantage `-0.05101`

By contrast, `FIT < 300` with **non-negative** net load slightly favors Hybrid:

- `N = 1`: `+0.00094`
- `N = 3`: `+0.00292`
- `N = 6`: `+0.00567`
- `N = 12`: `+0.00780`

So the current best read is:

1. Amber’s advantage is a real **multi-step path** effect.
2. It does **not** live mainly in the highest-FIT spike rows.
3. It appears to accumulate primarily in **negative-net-load, sub-`300` FIT** conditions.

That points the next diagnosis away from “export harder into the big spikes” and toward:

- inventory trajectory quality during more ordinary export-capable periods
- medium-horizon tactical path quality
- possibly preserving/exporting energy better across clusters of moderate opportunities rather
  than only the most extreme intervals

Concrete saved decomposition from the `wb7` forced-prefix summaries:

- overall Amber advantage grows with prefix length:
  - `N=1`: `-0.00043`
  - `N=3`: `-0.00150`
  - `N=6`: `-0.00351`
  - `N=12`: `-0.01146`
- `FIT < 300` and negative net load drives most of that edge:
  - `N=1`: `-0.00389`
  - `N=3`: `-0.01168`
  - `N=6`: `-0.02287`
  - `N=12`: `-0.05101`
- `FIT < 300` with non-negative net load slightly favors Hybrid instead:
  - `N=1`: `+0.00094`
  - `N=3`: `+0.00292`
  - `N=6`: `+0.00567`
  - `N=12`: `+0.00780`

So the strongest currently supported working hypothesis is:

- Amber’s remaining edge is a **multi-step inventory/path advantage in ordinary export-capable
  periods**, not a “giant spike export” trick

### Forced-prefix path attribution

A follow-up attribution pass joined the forced-prefix regret rows back to the rolling raw paths
without rerunning the LP solves:

- [eval/analyze_forced_prefix_path_attribution.py](../eval/analyze_forced_prefix_path_attribution.py)
- outputs:
  - `eval/results/wb7_prefix_n{1,3,6,12}_forced_prefix_path_attribution.csv`
  - `eval/results/wb7_prefix_n{1,3,6,12}_forced_prefix_path_attribution_summary.csv`

For the key `FIT < 300` + negative-net-load bucket, the `N=12` attribution shows Amber's
lower-regret prefix is **not** achieved by exporting more:

- Amber minus Hybrid forced-prefix regret: `-0.05101`
- prefix charge: `-0.083 kWh`
- prefix discharge: `-0.161 kWh`
- prefix import: `-0.302 kWh`
- prefix export: `-0.371 kWh`
- prefix step PnL: `+0.022`
- prefix SoC delta: `+0.081 kWh`

The same pattern ramps with prefix length (`N=1,3,6,12`): Amber does less charging, less
discharging, less importing, and less exporting in the target bucket, while ending the prefix
with slightly more stored energy and better immediate PnL.

That sharpens the interpretation again:

- the target bucket is not simply "ordinary export opportunities"
- its average feed-in price is low/negative, so excess export can be actively unattractive
- the Amber edge looks more like **reduced churn / better surplus-PV inventory discipline**
  through low-to-moderate FIT periods

This makes the next modeling target closer to a short-horizon state-transition or marginal
energy-value label than to an export-uplift label. A useful first label family would ask:

- how much SoC should be carried 30-60 minutes forward during negative-net-load, low-FIT periods?
- when should surplus PV be stored versus exported versus ignored?
- what is the marginal value of one extra kWh in the battery at the end of the prefix?

### First target-bucket state-transition labels

The first full state-transition label build for the target bucket completed successfully:

- output prefix: `state_transition_wb7_fitlt300_negload`
- filter: `FIT < 300` and negative net load
- horizons: `N=6` and `N=12`
- rows: `700` filtered starts, `1,400` horizon rows
- summary tool: [eval/analyze_state_transition_labels.py](../eval/analyze_state_transition_labels.py)

The label read sharpens the interpretation again. In this target bucket, the realized-future
oracle does **not** say "carry more inventory than Hybrid" on average.

At `N=12`, oracle minus Hybrid averages:

- SoC delta: `-0.443 kWh`
- throughput/churn: `-0.540 kWh`
- import: `-0.466 kWh`
- export: `+0.005 kWh`
- prefix PnL: `+0.004`

At `N=12`, Amber minus Hybrid averages:

- SoC delta: `-0.104 kWh`
- throughput/churn: `-0.235 kWh`
- import: `-0.303 kWh`
- export: `-0.366 kWh`
- prefix PnL: `+0.024`

So the earlier "preserve inventory" phrase was too broad. The better current read is:

- reduce uneconomic charge/discharge churn
- reduce unnecessary grid exchange
- avoid exporting surplus PV into weak feed-in conditions unless the path value supports it
- learn a 30-60 minute state-transition discipline signal, not a simple export-uplift signal

This points the next modeling pass toward a bounded churn / grid-exchange discipline target or
state-transition value label. A finite-difference marginal-SoC run remains useful, but should be
second-pass because it roughly doubles LP solve cost.

### Curtailment correction

The low-FIT surplus-PV read exposed an important simulator limitation: the first
`netload_tariffed` LP represented grid import/export and battery charge/discharge, but did not
represent PV curtailment. With negative net load, any surplus not absorbed by battery charging
was therefore forced into grid export.

That is too restrictive for the real hybrid inverter, which can curtail export or turn PV down.

The LP has now been corrected:

- site-flow mode includes a nonnegative `curtail_kw` variable
- when separate load/PV inputs are available, curtailment is bounded by available PV:
  `0 <= curtail_kw <= pv_kw`
- the net-load fallback still supports surplus-only curtailment:
  `0 <= curtail_kw <= max(0, -net_load_kw)`
- the grid balance becomes:
  `grid_import - grid_export = load - pv + charge - discharge * eff_d + curtail`
- rolling `netload_tariffed` scoring now uses the split load/PV path when available and
  falls back to net load otherwise
- oracle/action/state-label builders carry curtailment into their path metrics and prefer split
  load/PV inputs when newer rolling raw outputs include them

Smoke behavior:

- negative feed-in with `2.5 kW` surplus: `export=0`, `curtail=2.5`
- positive feed-in with `2.5 kW` surplus: `export=2.5`, `curtail=0`
- negative import/feed-in with `3.0 kW` load and `2.0 kW` PV available:
  `curtail=2.0`, `import=3.0`

Important remaining limitation: any caller that provides only net load can still curtail only visible
surplus PV. Full PV turn-down while site load remains positive needs the split load/PV inputs now
supported by `solve_lp_dispatch()` and the rolling `netload_tariffed` harness.

The target-bucket labels were rerun under the corrected LP on `2026-05-01`:

- rolling raw: `rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_curtail_20260501_raw.parquet`
- state labels: `state_transition_wb7_fitlt300_negload_curtail_20260501_state_transition_labels.parquet`
- all three chained stages exited `0`

Window B `7-day`, corrected curtailment:

- `amber_apf_lgbm`: `1.611/day`
- `amber_tactical_hybrid_strategic`: `1.619/day`
- `hybrid_tactical_amber_strategic`: `1.078/day`
- `model_a_hybrid`: `1.040/day`

Compared with the pre-curtailment run, every source improves, but the same-target tactical gap
does not disappear:

- old `amber_tactical_hybrid_strategic` minus `model_a_hybrid`: about `+0.529/day`
- corrected-curtailment gap: about `+0.579/day`

So PV curtailment support improves simulator fidelity, but it does **not** explain away the Hybrid
tactical loss.

Corrected target-bucket state-transition labels (`FIT < 300`, negative net load) still point toward
reduced churn / grid exchange. At `N=12`, oracle minus Hybrid averages:

- SoC delta: `-0.400 kWh`
- throughput/churn: `-0.639 kWh`
- import: `-0.563 kWh`
- export: `-0.122 kWh`
- curtail: `-0.009 kWh`
- prefix PnL: `+0.023`

This strengthens the current read: the next modeling branch should target short-horizon inventory
discipline / reduced uneconomic churn, not spike export uplift, first-action imitation, or simply
more PV curtailment.

A physical-feasibility audit of the corrected raw run also passed cleanly:

- audit output: `rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_curtail_20260501_physical_feasibility_summary.csv`
- checked simultaneous charge/discharge
- checked simultaneous planned and realized import/export
- checked planned and realized grid-balance residuals
- checked SoC transition residuals and SoC bounds
- checked curtailment against available PV
- checked import/export bounds
- result: `0` violations for every source and every check at tolerance `1e-6`

That makes further simulator work a lower-priority branch for now. The corrected run is physically
clean enough to use as the next state-value / inventory-discipline modeling gate.

### First State-Value Model Probe

The first diagnostic model scaffold is now in place:

- tool: `eval/train_state_transition_value_model.py`
- labels: `state_transition_wb7_fitlt300_negload_curtail_20260501_state_transition_labels.parquet`
- model family: small LightGBM regressors
- split: time-ordered, `70%` train / `30%` validation
- features: current-time/site/forecast summary features only, plus horizon and time-of-day/day-of-week encodings
- targets:
  - `oracle_minus_target_step_pnl`
  - `oracle_minus_target_soc_delta_kwh`
  - `oracle_minus_target_throughput_kwh`
  - `oracle_minus_target_import_kwh`
  - `oracle_minus_target_export_kwh`
  - `oracle_minus_target_curtail_kwh`

Validation overall:

- prefix value / PnL: MAE improves by about `3.8%` over a median baseline, `R2 ~= 0.084`,
  sign accuracy `~94.5%`
- SoC delta: MAE improves by about `7.2%`, `R2 ~= 0.038`, sign accuracy `~85.7%`
- throughput/churn: MAE improves by only about `1.3%`, with negative `R2`
- import/export/curtail deltas do not beat the simple baseline on MAE

Top features for the two most promising labels are plausible rather than obviously spurious:
`strategic_soc_target_kwh`, `soc_prev_kwh`, time/day encodings, `actual_pv_kw`, current net load,
and `forecast_feed_in_mean_next_4h_mwh`.

Interpretation: there is weak but real learnable signal for "how much value / SoC movement is the
Hybrid path missing" in the corrected target bucket. There is not yet enough evidence to wire a
model directly into control. The next modeling step should broaden the label set across more
windows/regimes or improve target shaping before attempting an MPC bias.

### State-Value Label Broadening

The first broadening run completed on `2026-05-01`:

- `FIT < 300`, non-negative net load:
  `state_transition_wb7_fitlt300_nonnegload_curtail_20260501_state_transition_labels.parquet`
- `FIT >= 300`:
  `state_transition_wb7_fitgte300_curtail_20260501_state_transition_labels.parquet`
- pooled diagnostic model:
  `state_transition_wb7_pooled_curtail_20260501_state_value_model_metrics.csv`

All batch steps completed with exit code `0`.

The broadened label distributions are useful, but they weaken the simple pooled-model story.

For `FIT < 300` with non-negative net load, the oracle still tends to prefer less churn than
Hybrid, but the value sign differs from the original negative-net-load target bucket:

- `N=12`, `1165` rows
- oracle minus Hybrid SoC delta: `+0.322 kWh`
- oracle minus Hybrid throughput/churn: `-0.164 kWh`
- oracle minus Hybrid import: `+0.129 kWh`
- oracle minus Hybrid export: `-0.185 kWh`
- oracle minus Hybrid prefix PnL: `-0.074`
- Amber minus Hybrid prefix PnL: `+0.012`

For `FIT >= 300`, the oracle again prefers materially less movement than Hybrid, but also much
less export and worse prefix PnL over the pinned prefix:

- `N=12`, `121` rows
- oracle minus Hybrid SoC delta: `+1.305 kWh`
- oracle minus Hybrid throughput/churn: `-1.183 kWh`
- oracle minus Hybrid export: `-1.180 kWh`
- oracle minus Hybrid prefix PnL: `-0.474`
- Amber minus Hybrid prefix PnL: `+0.119`

The pooled LightGBM diagnostic did not find a useful general correction signal. On validation,
the pooled model is effectively at baseline:

- prefix value / PnL: `0.046%` MAE improvement overall, `R2 ~= -0.033`
- SoC delta: about `0.002%` worse than baseline overall, `R2 ~= -0.114`
- throughput, import, export, and curtail labels are all worse than baseline on MAE

Interpretation: the original negative-net-load target bucket still looks like a real local signal,
but it does not yet generalize as a single broad state-value model across these coarse regimes.
The next production-worthy step should either narrow the state-value corrector to the specific
ordinary surplus-PV regime where the economic gap was found, or build a more explicitly regime
conditioned label/model family. A pooled one-size-fits-all inventory-value bias is not supported by
this run.

### Marginal SoC Finite-Difference Probe

The next target-bucket run added a direct `+1 kWh` initial-SoC finite-difference label:

- run script: `run_state_value_fdiff_targetbucket_20260501.sh`
- label output:
  `state_transition_wb7_fitlt300_negload_fdiff1_curtail_20260501_state_transition_labels.parquet`
- model metrics:
  `state_transition_wb7_fitlt300_negload_fdiff1_curtail_20260501_model_state_value_model_metrics.csv`
- bucket: `FIT < 300`, negative net load, corrected curtailment Window B raw
- all steps completed with exit code `0`

Runtime was about `90` minutes. The finite-difference solve produced `1404` label rows
(`702` timestamps x two horizons). The finite-difference value was available on `1302` rows;
rows at or near the usable capacity ceiling cannot always support a full `+1 kWh` perturbation.

Label distribution:

- mean finite-difference value: `0.082 $/kWh`
- median: `0.084 $/kWh`
- interquartile range: `0.050` to `0.115 $/kWh`
- min / max: about `-0.170` to `0.413 $/kWh`

This is a useful diagnostic label: it gives the branch a direct local estimate of "what is one
more kWh in the battery worth here?" rather than relying only on path deltas such as churn or
prefix PnL.

The first diagnostic LightGBM fit, however, is not yet a strong learned value model:

- finite-difference value validation MAE improves by about `19.8%` over the median baseline
- validation `R2` is strongly negative (`~ -3.57`)
- train `R2` is high (`~ 0.89`), so the fit is over-learning the small target-bucket slice
- sign accuracy is unchanged from baseline (`~54.8%`)

The older path labels remain similar to the prior target-bucket probe:

- prefix value / PnL: validation MAE improves by about `3.8%`, `R2 ~= 0.084`
- SoC delta: validation MAE improves by about `7.2%`, `R2 ~= 0.038`

Interpretation: the finite-difference target is worth keeping as a label source, but this first
fit is not a production candidate. It argues for either better target shaping / regularization or
more regime-aware training data before attempting an MPC inventory-value bias. It does **not**
justify wiring the learned finite-difference model directly into dispatch.

### Eval-Only Inventory-Discipline Control Sweep

The first production-shaped control probe was deliberately simple: add a control-only cycle-cost
adder in the `FIT < 300` + negative-net-load regime, using the new
`model_a_hybrid_inventory_bias` rolling-eval source. This is not a learned finite-difference
controller; it is a bounded test of whether "less churn in the ordinary surplus-PV bucket" is
itself enough to improve dispatch.

Sweep setup:

- windows:
  - Window B 2-day: `2025-09-01 -> 2025-09-03`
  - Window A 2-day: `2025-07-21 -> 2025-07-23`
- source comparison:
  - `amber_tactical_hybrid_strategic`
  - `model_a_hybrid`
  - `model_a_hybrid_inventory_bias`
- cycle-cost adders: `25`, `50`, `75`, `100 $/MWh`
- result rollup: `inventory_bias_sweep_20260501_summary.csv`
- all eight rolling runs completed with exit code `0`

Window B result:

| source / setting | mean $/day |
| --- | ---: |
| Amber tactical + Hybrid strategic | `6.253` |
| unguarded Hybrid | `5.905` |
| guarded Hybrid, `25 $/MWh` | `4.963` |
| guarded Hybrid, `50 $/MWh` | `4.181` |
| guarded Hybrid, `75 $/MWh` | `2.111` |
| guarded Hybrid, `100 $/MWh` | `1.977` |

So the blunt guard makes Window B worse at every tested setting. The first `25 $/MWh` setting
already loses about `0.94/day` versus unguarded Hybrid and about `1.29/day` versus Amber
tactical + Hybrid strategic.

Window A result:

| source / setting | mean $/day |
| --- | ---: |
| Amber tactical + Hybrid strategic | `-1.052` |
| unguarded Hybrid | `-1.330` |
| guarded Hybrid, `25 $/MWh` | `-1.329` |
| guarded Hybrid, `50 $/MWh` | `-1.313` |
| guarded Hybrid, `75 $/MWh` | `-1.313` |
| guarded Hybrid, `100 $/MWh` | `-1.313` |

So the guard slightly improves Window A versus unguarded Hybrid, but not enough to approach Amber.

Interpretation: the eval hook is useful, but the broad cycle-friction hypothesis is falsified as
a production path. The guard suppresses too much useful Window B surplus/export behavior. The
remaining plausible direction is not "more friction whenever ordinary surplus PV is present"; it
would need a sharper opportunity-aware state-value gate that distinguishes wasteful churn from
profitable charging/export preparation.

### Tier 1 Tactical Vector Generalization Batch

After adding the first dispatch-relevant Tier 1 diagnostic, we ran the fuller h0-h11 vector
diagnostic across the main falsification slices and longer tactical windows:

- output prefixes: `tier1_vector_*_20260501`
- script: [eval/analyze_tier1_tactical_vector_errors.py](../eval/analyze_tier1_tactical_vector_errors.py)
- windows:
  - Window A 7-day, legacy and `tariffaware_v1`
  - Window B excluding `2025-09-01`, legacy and `tariffaware_v1`
  - moderate-FIT middle window, legacy and `tariffaware_v1`
  - Window A and Window B 6-week Track A tactical windows, legacy
- batch exit code: `0`

The result is helpful mainly because it prevents an over-simple next model branch.

Key reads:

- `tariffaware_v1` is still close to inert. Its vector fingerprints remain very close to the
  legacy Tier 1 fingerprints across the tested slices.
- `FIT < 300` with negative net load is not a universal local-error bucket. It remains relevant
  on Window B excluding `2025-09-01`, but Amber is flat-to-worse on Window A, midfit, and the
  6-week tactical windows on the immediate step-PnL lens.
- Amber's more stable local vector edge often appears in `FIT >= 300` or `FIT < 300` with
  non-negative net load.

That does not invalidate the forced-prefix result. It says the two diagnostics are looking at
different pieces of the problem:

- forced-prefix regret says the controller can lose multi-step path value when 30-60 minutes of
  behavior are pinned
- vector diagnostics say the loss is not explained by one clean bucket-specific price-curve
  level error

So the next production-shaped model candidate should **not** be a narrow
`FIT < 300` / negative-net-load retrain, and it should not build on `tariffaware_v1` as a base.
The better next branch is a richer multi-step path/value target or target transformation that can
distinguish profitable surplus-PV preparation from wasteful churn. The vector diagnostic should
remain the falsification lens for that branch.

### Vector Features In State-Value Probe

To test whether the full h0-h11 Tier 1 curve shape contains enough signal for that richer branch,
the diagnostic state-value trainer now supports optional vector-feature ingestion:

- code: [eval/train_state_transition_value_model.py](../eval/train_state_transition_value_model.py)
- test: [tests/eval/test_train_state_transition_value_model.py](../tests/eval/test_train_state_transition_value_model.py)
- option: `--vector-rows ... --vector-source model_a_hybrid`
- leakage guard: only forecast vector columns are used; realized future price columns in the
  vector diagnostic rows are ignored

First full target-bucket probe:

- output prefix: `state_transition_wb7_fitlt300_negload_vector_probe_20260502`
- labels: `state_transition_wb7_fitlt300_negload_curtail_20260501_state_transition_labels.parquet`
- vector rows: `tier1_vector_wb7_legacy_20260501_tier1_vector_rows.parquet`
- joined vector features: `86`
- matched label rows: `1,404 / 1,404`

Validation result:

| target | MAE improvement vs median baseline | R2 |
| --- | ---: | ---: |
| prefix PnL / value | `+0.4%` | `0.078` |
| SoC delta | `+5.3%` | `0.080` |
| throughput | `+1.0%` | `-0.148` |
| import | `+0.0%` | `-0.353` |
| export | `-0.5%` | `-0.066` |
| curtail | `-0.4%` | `-0.164` |

Feature importance shows the model does use some h0-h11 shape features, especially adjacent
general/feed-in curve moves. But the dominant predictors remain time, SoC, strategic target,
PV/net-load, and existing 4h forecast-summary features.

Interpretation: first-hour curve shape is useful context, but not enough by itself to produce a
strong state-value model on this single target-bucket slice. The next candidate should improve
the target/label formulation or broaden regime-aware training data before another dispatch hook.
