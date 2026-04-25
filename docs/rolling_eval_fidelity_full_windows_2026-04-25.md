# Rolling Eval Fidelity Full-Window Results — 2026-04-25

Purpose: document the first longer-window results after upgrading Track 10A to support
`netload_tariffed` economics, and compare them with the established `price_only` exact-handoff
baseline.

These runs use:
- `14h x 5m` rolling MPC
- continuous SoC carryover
- exact strategic handoff
- sources: `amber_apf_lgbm`, `model_a_hybrid`
- TFT assets:
  - `models/tft_price/checkpoint_run014_phase7_best.pt`
  - `models/tft_price/scalers_run014_phase7.pkl`

Important caveat:
- these results use the locally available **Run 014 Phase 7** TFT asset
- repo docs still describe **Run 011b + binary routing** as the strongest incumbent and Run 014
  as a failed interim eval checkpoint
- so these tariffed results should currently be treated as **provisional / artifact-limited**
  evidence about the mechanics of the rolling gate, not yet a final verdict on the best
  AEMO-native architecture

## 1. Finished Runs

Completed on `2026-04-25`:

- `rolling_mpc_eval_pilot_exact_netload_7day_20260425`
- `rolling_mpc_eval_tracka_followup_6week_netload_exact_20260425`
- `rolling_mpc_eval_tracka_followup_6week_priceonly_exact_refresh_20260425`
- `rolling_mpc_eval_tracka_windowa_6week_netload_exact_20260425`

## 2. Headline Results

### 7-day pilot (`2025-09-01 -> 2025-09-08`, `netload_tariffed`)

- `amber_apf_lgbm`: **$10.325 total / $1.476 per day**
- `model_a_hybrid`: **$6.661 total / $0.952 per day**
- hybrid vs amber: **-35.5%**

### Window B (`2025-09-01 -> 2025-10-13`)

`price_only` exact handoff refresh:
- `amber_apf_lgbm`: **$102.938 total / $2.451 per day**
- `model_a_hybrid`: **$95.357 total / $2.271 per day**
- hybrid vs amber: **-7.4%**

`netload_tariffed` exact handoff:
- `amber_apf_lgbm`: **$17.923 total / $0.427 per day**
- `model_a_hybrid`: **$12.505 total / $0.298 per day**
- hybrid vs amber: **-30.2%**

### Window A (`2025-07-21 -> 2025-09-01`, `netload_tariffed`)

- `amber_apf_lgbm`: **-$23.941 total / -$0.570 per day**
- `model_a_hybrid`: **-$37.849 total / -$0.901 per day**
- hybrid vs amber: **-58.1%**

## 3. Provisional Conclusion

The production-fidelity economics gate materially changes the architecture picture for the
currently tested local asset set.

Previously:
- Window A looked favorable to the hybrid under `price_only`
- Window B looked moderately unfavorable under `price_only`

Now, under `netload_tariffed`:
- Amber wins on the 7-day pilot
- Amber wins on full Window B
- Amber wins on full Window A

This is no longer a narrow “one follow-up window” problem for the current Run 014-based local
setup. Under more production-like site economics, Amber is consistently ahead on both major
Track 10A windows that previously anchored the roadmap discussion.

## 4. Window B Diagnostic Read

From
`rolling_mpc_eval_tracka_followup_6week_netload_exact_20260425_diagnostics_overall.csv`:

Amber totals:
- charge energy: `1076.48 kWh`
- discharge energy: `1033.52 kWh`
- realized grid import: `309.34 kWh`
- realized grid export: `930.31 kWh`
- import cost: `$19.89`
- export revenue: `$140.62`
- degradation cost: `$102.81`
- final SoC: `9.14 kWh`

Hybrid totals:
- charge energy: `1045.39 kWh`
- discharge energy: `1013.13 kWh`
- realized grid import: `285.69 kWh`
- realized grid export: `918.37 kWh`
- import cost: `$19.77`
- export revenue: `$132.59`
- degradation cost: `$100.31`
- final SoC: `0.00 kWh`

Important interpretation:
- Amber earns about **$8.03** more export revenue
- import cost is nearly the same
- degradation cost is slightly higher for Amber, but not enough to matter
- Hybrid ends the full window with **less** final stored energy, not more

So on full Window B, the hybrid does not appear to be “losing because it preserved too much
energy.” Instead, Amber appears to be converting inventory into realized export revenue more
effectively across the whole window.

## 5. Window A Diagnostic Read

From
`rolling_mpc_eval_tracka_windowa_6week_netload_exact_20260425_diagnostics_overall.csv`:

Amber totals:
- charge energy: `1108.80 kWh`
- discharge energy: `1062.57 kWh`
- realized grid import: `661.31 kWh`
- realized grid export: `595.16 kWh`
- import cost: `$68.32`
- export revenue: `$150.17`
- degradation cost: `$105.80`
- final SoC: `10.79 kWh`

Hybrid totals:
- charge energy: `1020.66 kWh`
- discharge energy: `976.80 kWh`
- realized grid import: `609.43 kWh`
- realized grid export: `549.95 kWh`
- import cost: `$67.41`
- export revenue: `$126.88`
- degradation cost: `$97.32`
- final SoC: `12.83 kWh`

Interpretation:
- Amber’s edge in Window A is again mostly on realized export revenue
- Hybrid is slightly cheaper on import/degradation, but not enough to offset weaker export monetization
- Hybrid does finish with about `2.04 kWh` more stored energy, but that is small relative to the
  revenue gap

So Window A and Window B now point in the same direction:
- the hybrid is not obviously failing because it buys much more expensively
- it is failing because it monetizes stored energy less effectively under tariffed site economics

## 6. Revised Interpretation

This makes the next-step priority much clearer.

The most likely bottleneck is now:
- tactical control / inventory monetization under realistic site economics

Less likely primary bottlenecks:
- another round of generic 72h path tuning
- another sweep of weak bridge-contract variants
- conclusions drawn mainly from `price_only`

In other words, the full-window tariffed gate supports the reviewer’s recommendation to treat
the strategic layer more as an inventory-value layer and to make the rolling `14h x 5m` eval
the main architectural gate. But the checkpoint caveat matters: these runs should not yet be
summarized as “the AEMO-native incumbent loses” until the strongest intended incumbent asset is
recovered or reproduced and re-run under the same gate.

## 7. Immediate Next Questions

The most useful follow-up is no longer “which bridge variant next?” but:

1. Where, day by day, does Amber earn higher export revenue?
2. Does Amber discharge into better realized price periods, or simply more often?
3. Does Hybrid systematically miss profitable export windows because its tactical controller
   over-values future inventory under the current contract?
4. Are there identifiable day classes where this tariffed gap is concentrated?

## 8. Operational Notes

- All four runs completed successfully with full coverage.
- The detached-run wrapper used for this batch wrote exitcode files as `0n` instead of plain `0`,
  so the wrapper quoting needs a small fix even though the runs themselves completed cleanly.
- The new progress checkpoint files were useful during execution and should be retained for
  future long runs.
