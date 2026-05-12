# Load-Source Dispatch Counterfactual — Plan, 2026-05-12

Test whether swapping the load forecast feeding EMHASS-equivalent dispatch
(`netload_tariffed`) changes operational outcomes enough to justify promoting
TFT load — or keeps LGBM as production. The motivating finding (Codex,
roadmap §4): TFT q50 has materially lower MAE than LGBM q50 (`146 W` vs
`211 W`) but under-forecasts 27% of the time, clustered in
morning/overnight/solar Adelaide periods. MAE alone is not a dispatch
verdict — TFT price had the same shape, and lost.

## What the Smoke Already Proved (2026-05-12)

Three runs over a 12-hour window (`2026-05-10T00:00–12:00Z`), one source
contract (`amber_apf_lgbm`), `--economic-mode netload_tariffed`:

| metric | actual | lgbm_load_log | tft_load_log |
|---|---:|---:|---:|
| total_pnl ($) | 1.981 | 1.910 | 2.044 |
| soc_final (kWh) | 4.36 | 5.03 | 3.67 |
| under-prep steps | 0 | 28 | 7 |
| under-prep mean (kW) | — | 0.039 | 0.069 |
| solar under-prep | 0 | 12 | 7 |
| evening under-prep | 0 | 16 | 0 |

Working:

- Harness end-to-end: 100% MPC coverage across all three runs.
- Three different load curves produce three distinct LP decisions (PnL,
  SoC trajectory, dispatch profile all differ).
- `actual` baseline correctly shows 0 under-prep (forecast == actual ⇒ tie).
- Adelaide target-time buckets populate as designed.

## What the Smoke Did NOT Prove

- **TFT > actual in PnL is a structural artefact, not a real signal.**
  Perfect information should be the ceiling. TFT ended at lower SoC (3.67
  kWh vs 4.36 kWh) and the LP doesn't price terminal kWh. The PnL column
  cannot be trusted for cross-source comparison until terminal value is
  handled.
- **The window doesn't cover TFT's risk regime.** Codex's 45% morning /
  38% overnight under-forecast clustering is mostly outside the
  Adelaide 09:30–21:30 covered here. Need multi-day windows to span all
  five Adelaide buckets.

## Open Design Decisions

### D1. Terminal-Value Treatment

Three options:

a) `--terminal-energy-value-mwh <window-mean wholesale price>`
   - Simplest. Prices remaining kWh at a flat $/MWh equal to the
     window-average wholesale.
   - Pros: fast, easy to interpret.
   - Cons: arbitrary endpoint; doesn't reflect production behaviour where
     terminal SoC matters more in some regimes than others.

b) `--strategic-soc-handoff --strategic-target-mode exact`
   - Production-equivalent path. Strategic optimiser handles terminal SoC
     via the 72h forward solve. This is what
     `docs/dynamic_bridge_experiment_plan_2026-04-23.md` uses for all the
     6-week MPC sweeps.
   - Pros: comparable to historic eval results; closer to live behaviour.
   - Cons: adds dependency on strategic solve which may interact with the
     load-source swap in ways we haven't tested yet.

c) Both — run (a) for the cheap-and-dirty answer and (b) for the
   production-equivalent answer; compare to see if the verdict moves.

**Recommendation**: (b) directly. The whole point of the counterfactual
is to predict production behaviour; using the production terminal
treatment is more defensible than introducing a third regime.

### D2. Source Contract Scope

Today's smoke used only `amber_apf_lgbm`. Production uses
`model_a_hybrid` (TFT-extrapolated Tier 2 + LGBM Tier 1) and the smoke
plan doesn't yet exercise that. Two options:

a) Single source: keep `amber_apf_lgbm`. Cleaner isolation of the
   load-source effect, but doesn't say anything about how load source
   interacts with the production price stack.
b) Two sources: `amber_apf_lgbm,model_a_hybrid`. Requires loading TFT
   price context (~5–10s per run). Lets us see whether load source
   interacts differently across price stacks.

**Recommendation**: (b). Marginal cost is small; the comparison is more
useful if it reflects the production price-source choice.

### D3. Window

Hard ceiling: **TFT load log starts 2026-05-09T08:43Z**, so any window
that includes `tft_load_log` is capped at ~3 days.

| window candidate | days | TFT coverage? | LGBM coverage? | wall-clock estimate (3 runs × 2 sources) |
|---|---:|:---:|:---:|---|
| 2026-05-10 → 2026-05-12 | 2.5 | yes (full) | yes (full) | ~5–10 min |
| 2026-05-09T12:00Z → 2026-05-12T12:00Z | 3.0 | yes (full) | yes (full) | ~6–12 min |
| 2026-04-01 → 2026-05-12 | 41 | no (TFT excluded) | yes (full) | ~30–45 min, LGBM-only |
| 2025-09-01 → 2025-10-13 | 42 | no (TFT excluded) | yes (full) | ~30–45 min, LGBM-only |

The 3-day TFT-included window is unfortunately small for stable PnL
signal but is the maximum we can do until backfill catches up.

**Recommendation**: split into two runs.

- **Run set A (TFT-included, short)**: 3 days, both sources, all three
  load sources (`actual`, `lgbm_load_log`, `tft_load_log`). Answers
  "does the TFT load curve produce sensibly-different dispatch from
  LGBM in the regimes TFT is suspected of mishandling?"
- **Run set B (LGBM-only, long)**: ~6 weeks of recent data, both
  sources, two load sources (`actual`, `lgbm_load_log`). Answers
  "does load-forecast quality matter for dispatch at all over a
  meaningful sample?" — the prerequisite question that doesn't depend
  on TFT.

If B shows load-source has negligible dispatch impact, A's TFT
result is moot and the whole branch can park until backfill enables a
fairer TFT window.

## Proposed Concrete Commands

After confirming D1 + D2 + D3 the morning of 2026-05-13:

```bash
# Run set A: short, TFT-included, both sources
nice -n 19 ./.venv/bin/python eval/rolling_mpc_eval.py \
  --start 2026-05-09T12:00:00Z --end 2026-05-12T12:00:00Z \
  --sources amber_apf_lgbm,model_a_hybrid \
  --tft-checkpoint models/tft_price/checkpoint_active.pt \
  --tft-scalers models/tft_price/scalers_active.pkl \
  --economic-mode netload_tariffed \
  --strategic-soc-handoff --strategic-target-mode exact \
  --load-forecast-source <actual|lgbm_load_log|tft_load_log> \
  --workers 1 \
  --output-prefix loadsrc_A_3day_<source>

# Run set B: longer, LGBM-only (no tft_load_log run)
nice -n 19 ./.venv/bin/python eval/rolling_mpc_eval.py \
  --start 2026-04-01T00:00:00Z --end 2026-05-12T00:00:00Z \
  --sources amber_apf_lgbm,model_a_hybrid \
  --tft-checkpoint models/tft_price/checkpoint_active.pt \
  --tft-scalers models/tft_price/scalers_active.pkl \
  --economic-mode netload_tariffed \
  --strategic-soc-handoff --strategic-target-mode exact \
  --load-forecast-source <actual|lgbm_load_log> \
  --workers 1 \
  --output-prefix loadsrc_B_6week_<source>
```

Both pairs are doable in well under an hour total. Neither is overnight
in scope.

## What Each Result Will/Won't Tell Us

A's verdict:

- If TFT load → meaningfully worse dispatch (PnL, under-prep events,
  SoC profile) **in morning/overnight regimes specifically**: TFT load
  inherits the same MAE-good-dispatch-bad pattern as TFT price. Park
  promotion. Roadmap §4 stands.
- If TFT load → comparable dispatch with no morning/overnight blowups:
  promotion gets a green light *for further investigation* — not a
  green light to flip. Conservatively, this just means "no dispatch
  evidence against TFT in 3 days of data."

B's verdict:

- If load source matters meaningfully (>~1% PnL spread, >~5%
  under-prep spread): the branch is worth continuing. Wait for
  backfill, repeat A on a longer window.
- If load source barely moves anything: the load-source axis is not
  the right battery economics knob. Park the branch entirely until
  there's a separate reason to revisit it.

## Constraints and Caveats

- TFT log boundary at 2026-05-09T08:43Z is the binding constraint.
- The window 2026-05-09 → 2026-05-12 straddles the alignment-fix
  promotion (`211ae8d`, 2026-05-11T13:22Z). The pre/post-promotion
  halves are not perfectly comparable on the price side; check whether
  the load-source delta differs across the boundary.
- 3 days × 2 sources × 3 load curves = 18 source-days of evidence.
  This is small. Treat A as exploratory; B is what carries weight.
- No backfill or model retrain happens here. This is dispatch-side eval
  only.
- `actuals_sa1.parquet` and `actuals_sa1_5m.parquet` were refreshed
  2026-05-12T14:13Z (from ~26 days stale). Re-check before launch in
  case more days have accumulated.

## Pre-flight Checklist

Before launching in the morning:

- [ ] Refresh actuals parquets again (cheap, ensures current to
      launch time).
- [ ] Confirm TFT price checkpoint paths still resolve to the
      production-active pair.
- [ ] Choose D1 (recommended: strategic-soc-handoff).
- [ ] Choose D2 (recommended: both sources).
- [ ] Confirm D3 split (recommended: A then B; or B first if cheap
      sanity check is more valuable).
- [ ] Decide whether to commit raw output parquets/CSVs to git. Default
      is no (the `.gitignore` already excludes `eval/results/`).

## Followups Independent of This Run

- Production-surface logging only started 2026-05-12. `load_p65` and
  `tft_load_q90` comparison runs are blocked until backfill produces
  enough actuals-backed rows. Defer to a follow-up branch.
- Spike-classifier warning text tightening (from HANDOVER §5) still
  pending.
- `eval/rolling_mpc_eval.py` actuals alignment audit still pending
  (HANDOVER §7).

## Cross-References

- Smoke output: `eval/results/loadsrc_smoke_{actual,lgbm,tft}_*.csv`
  (uncommitted, regenerable).
- Live load accuracy audit: `eval/compare_live_load_accuracy.py` +
  `eval/results/live_load_accuracy_latest.json`.
- Roadmap §4 (TFT load eval) and §5 (HA deployment after canonical
  source decision).
- HANDOVER.md and HANDOVER_QUESTIONS.md.
