# Rolling Eval Fidelity Pilot — 2026-04-25

Purpose: document the first like-for-like comparison between the legacy Track 10A
`price_only` objective and the new `netload_tariffed` objective on the same short
window, with the same forecast sources and strategic handoff.

Window:
- `2025-09-01 -> 2025-09-03`

Shared run shape:
- `14h x 5m` rolling MPC
- continuous SoC carryover
- sources: `amber_apf_lgbm`, `model_a_hybrid`
- strategic handoff enabled
- `--strategic-target-mode exact`
- TFT assets:
  - `models/tft_price/checkpoint_run014_phase7_best.pt`
  - `models/tft_price/scalers_run014_phase7.pkl`

Compared modes:
- `price_only`
- `netload_tariffed`

## 1. Headline Result

The production-fidelity economics gate changed the conclusion on this same 2-day slice.

`price_only`:
- `amber_apf_lgbm`: **$19.163 total / $9.598 per day**
- `model_a_hybrid`: **$19.667 total / $9.850 per day**
- hybrid vs amber: **+2.6%**

`netload_tariffed`:
- `amber_apf_lgbm`: **$12.578 total / $6.300 per day**
- `model_a_hybrid`: **$11.791 total / $5.906 per day**
- hybrid vs amber: **-6.3%**

So the new economic mode did not merely rescore the same tactical behavior. It changed the
observed ranking on the same window.

## 2. Dispatch Change Confirmation

Direct raw comparison between:
- `rolling_mpc_eval_pilot_exact_priceonly_20260424_raw.parquet`
- `rolling_mpc_eval_pilot_exact_netload_20260424_raw.parquet`

showed that the tariffed mode materially changed control actions:

- `step_pnl`: `1152` changed rows
- `soc_kwh`: `892` changed rows
- `soc_prev_kwh`: `890` changed rows
- `discharge_kw`: `295` changed rows
- `charge_kw`: `289` changed rows

Important negative result:
- dynamic bridge metadata remained unchanged in this pair
- the changed behavior came from the new objective/economic accounting, not from a different
  bridge contract

This is the key point of the pilot: the more production-like economics gate is behaviorally
active, not just cosmetically different.

## 3. Daily View Under `netload_tariffed`

### Amber APF + LGBM

`2025-09-01`:
- pnl: **$11.956**
- charge energy: `27.79 kWh`
- discharge energy: `44.97 kWh`
- realized grid import: `20.77 kWh`
- realized grid export: `40.03 kWh`
- close SoC: `1.43 kWh`

`2025-09-02`:
- pnl: **$0.623**
- charge energy: `39.06 kWh`
- discharge energy: `25.53 kWh`
- realized grid import: `13.17 kWh`
- realized grid export: `14.73 kWh`
- close SoC: `13.01 kWh`

### Hybrid

`2025-09-01`:
- pnl: **$11.626**
- charge energy: `26.95 kWh`
- discharge energy: `44.18 kWh`
- realized grid import: `21.46 kWh`
- realized grid export: `40.80 kWh`
- close SoC: `1.43 kWh`

`2025-09-02`:
- pnl: **$0.165**
- charge energy: `38.84 kWh`
- discharge energy: `17.31 kWh`
- realized grid import: `13.37 kWh`
- realized grid export: `7.33 kWh`
- close SoC: `21.02 kWh`

## 4. Behavioral Read

The gap under `netload_tariffed` appears to come mainly from `2025-09-02`.

Observed pattern:
- both sources behave similarly on `2025-09-01`
- on `2025-09-02`, the hybrid ends with much more stored energy
- but Amber monetizes materially more export energy on that second day

Numerically on `2025-09-02`:
- Amber exports `14.73 kWh`
- Hybrid exports `7.33 kWh`
- Amber day pnl beats Hybrid by about **$0.458**
- Hybrid closes with about **8.01 kWh** more energy than Amber

Interpretation:
- under production-like import/export economics, Amber appears to convert stored energy into
  realized spread more effectively on this slice
- the hybrid keeps a stronger terminal inventory posture, but that extra stored energy does not
  pay off within the 2-day window

This is directionally consistent with the reviewer’s warning that the bottleneck may now be
future inventory valuation and tactical control fidelity rather than another round of 72h path work.

## 5. What This Pilot Does And Does Not Prove

What it establishes:
- the `netload_tariffed` gate is worth keeping
- production-like economics can change both dispatch behavior and relative ranking
- the residual Amber edge is not just a reporting artifact of the old price-only eval

What it does not yet establish:
- whether Amber’s edge here is driven by better buy timing, better sell timing, or a better
  inventory-value heuristic
- whether the same ranking flip holds on a longer window
- whether the hybrid gap is fundamentally forecast-shaped or control-shaped

## 6. Immediate Follow-Up

Recommended next step:
- keep `netload_tariffed` as the primary short-window gate for the next diagnostic cycle
- compare Amber vs Hybrid behavior by day and by step under this gate before launching more
  bridge variants

Specifically:
- inspect where Amber buys lower
- inspect where Amber sells higher
- inspect where Amber chooses a different SoC posture despite similar strategic targets

Operational note:
- the runs completed successfully and wrote full result files
- the `.exitcode` files were created but ended up blank, so the tmux wrapper still needs a
  small fix even though the logs/results clearly show success
