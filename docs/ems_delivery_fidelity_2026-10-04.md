# EMS delivery fidelity — 2026-10-04

- Decision: controller timing/energy accounting precede forecast promotion. Retain APF;
  calibrated load remains offline. No production YAML change, reload, service call or device write.
- New `audit_ems_delivery.py`: replay recorded EMS mode/grid-export/PCS-export/charge/discharge limits
  against frozen raw telemetry. This is baseline fidelity, **not a counterfactual controller**.
  Recorded control trace cannot rank different forecasts or establish recoverable savings.
- Verified preceding own-feedback requests/results and continuation lineage; same initial SoC,
  runtime40.3kWh capacity and raw sample-hold targets as original replay. Carry physical SoC;
  never reset it to later measurements. Cut execution at each recorded mode/register event.
- Two supported device modes: Maximum Self Consumption; Command Discharging (PV First).
  Equilibrium DC demand derived from site load, delivered PV and grid target; apply battery/SOC,
  AC/DC conversion, inverter, recorded charge/discharge and PCS constraints through existing executor.
  Unsupported modes or missing/invalid limits fail. No charging-mode approximation.
- Charging cutoff readback comes from existing PV-mode sensor attribute `charge_limit_kw`;
  retain field-change clocks and cap DC charging. In this window it changes21→0kW at22:33:31.903809
  and0→21kW at22:36:32.867580. Ignoring it would incorrectly allow PV charging in that segment.
- Recorded PCS register already includes transient caps. Do not apply timer/helper cap a second
  time. Grid-only ablation measures PCS-cap contribution. Physical ramp/firmware response,
  fixed inverter losses and select/readback optimism remain unmodeled.

## Confirmed recorded controller behavior

Read-only Influx export Sep30 21:55–22:45 UTC, source `rp_raw`; latest prior states retained.
`export_control_history.py --include-ems-inputs` adds all six MPC control curves, action label,
EMS mode, grid status, hardware limits, desired/flexible export, ramp timer/cap, price and export
SoC guard. Missing source explicitly recorded; no current-state fallback.

- At22:05, previous plan selected P_grid=0, P_batt315.79W, load300W.
- At22:10, P_grid=0, P_batt179.16W, load269W.
- At22:15, P_grid=0, P_batt200.32W, load291W.
- YAML selects latest row date≤utcnow in each entity. These are next-slot rows from the
  preceding minute's plan. With on-grid, positive load, nonnegative battery/PV/hybrid and zero
  curtailment, choose order reaches `Self-consume from battery` before export-SOC guard.
- Archived `input_text.emhass_battery_action` changes to that label ~0.24s after each tick.
  EMS mode reports Maximum Self Consumption ~4.87s after tick. Later MPC publication/explicit
  battery-controller trigger restores Force discharge/Command Discharging in these three cases.
- Grid export register remains near9.999kW during the low-output gap; it is a **mode switch**,
  not a lower grid-export-cap command, at these ticks.

| Tick UTC | Self-consume action recorded | Self-consume mode recorded | Next MPC publication | Discharging mode recorded |
|---|---|---|---|---|
| 22:05 | 22:05:00.238744 | 22:05:04.864978 | 22:05:21.655533 | 22:05:26.867539 |
| 22:10 | 22:10:00.241263 | 22:10:04.880181 | 22:10:25.857036 | 22:10:30.887402 |
| 22:15 | 22:15:00.238424 | 22:15:04.875621 | 22:15:21.023504 | 22:15:25.874470 |

- Six supported self-consume tick selections in40min all match recorded action state; some
  matches are held unchanged labels, not evidence of six separate service executions.
  22:20 also switches to self-consumption; subsequent new MPC decisions retain it. Do not
  interpret every tick as a spurious switch or assume continued export would be optimal.
- Labels evidence branch selection; independently recorded select/number state is not proof of
  command completion or an atomic hardware snapshot. Power drops/recoveries corroborate timing.
- Current repo YAML hashed with diagnostic; not an independently captured Sep30 configuration.
  Historical minimum-export-SoC source absent in this archive. Supported self-consume subset
  does not consume it; other branch outcomes are not guessed.

## Matched execution results

Nominal15min/40min pilots actually begin22:00:23.787881 UTC, first captured solve origin;
end22:15/22:40. Same start, rates and raw telemetry for every comparison.

| Metric | First pilot | Continued pilot |
|---|---:|---:|
| Duration seconds | 876.212119 | 2,376.212119 |
| Observed net grid export kWh | 1.72249127 | 2.22454072 |
| Original ideal-command replay net export kWh | 1.85954368 | 2.50889606 |
| Recorded mode/grid/PCS replay net export kWh | 1.74352082 | 2.26288627 |
| Original export error kWh | +0.13705241 | +0.28435534 |
| Recorded-control export error kWh | +0.02102955 | +0.03834555 |
| Absolute export-error reduction | 84.7% | 86.5% |
| Observed variable export credit AUD | 0.46600089 | 0.59621092 |
| Original replay credit AUD | 0.50401427 | 0.66939740 |
| Recorded-control replay credit AUD | 0.47166147 | 0.60656171 |
| Recorded-control DC discharge / measured kWh | 1.86607061 / 1.88964816 | 2.42325488 / 2.49105379 |
| Original ending-inventory error kWh | −0.06885206 | −0.13737456 |
| Recorded-control ending-inventory error kWh | +0.07772242 | +0.14812076 |

- Grid matching improves; absolute inventory error **does not improve** and changes sign.
  DC discharge underestimated; model's fixed95% conversion, omitted fixed losses/transients,
  timestamp alignment, SoC/capacity reconciliation require independent validation.
- Difference between model credit and observed credit remains0.566c/1.035c. These are fidelity
  errors, not earned savings. Quotes are the same verified observed Amber variable rates;
  exclude wear, fixed charges and terminal inventory valuation.
- Ignoring recorded PCS cap changes modeled export only0.00002261/0.00016287kWh.
  Export-ramp PCS limiter is not a material explanation of these windows' gap. This does not
  establish its safety/value during large negative-price curtailment or other operating modes.
- A better grid fit from **recorded** controls is insufficient to certify a policy replay:
  policy changes would produce different mode/limit traces. No forecast challenger rescored
  with incumbent mode history; original own-feedback zero-gain result remains conditional.

## Reproduction

```bash
./.venv/bin/python eval/export_control_history.py \
  --start 2026-09-30T21:55:00Z --end 2026-09-30T22:45:00Z \
  --include-ems-inputs --output data/energy_replay/NEW_EMS_ARCHIVE
./.venv/bin/python eval/audit_ems_delivery.py \
  --history data/energy_replay/ems_control_history_20261004_export \
  --replay data/energy_replay/dh_load_feedback_20261004_export_15m \
    data/energy_replay/dh_load_feedback_20261004_export_30m_chunk2 \
    data/energy_replay/dh_load_feedback_20261004_export_40m_chunk3 \
  --output data/energy_replay/NEW_EMS_AUDIT.json
```

- Authoritative ignored reports: `ems_delivery_audit_20261004_export_{15m,40m}_v3.json`.
  Earlier outputs intermediate. History/bundle/report/code/YAML hashes retained; raw private
  telemetry never committed. New outputs required; no overwrites.
- Validation:65 affected tests pass, including12 EMS tests. Prior/future plan selection, YAML-supported branch, mode/DC energy balance,
  PCS/discharge/SOC constraints, unsupported states, chronological control cuts and audit scope.

## Next effort

1. Reconcile stable-mode DC energy, AC/grid output and SoC/capacity on independently selected
   windows. Estimate fixed losses and response delay only from released training observations;
   verify on separate periods. Both cash and terminal inventory need fidelity.
2. Build bounded controller challenger: retain latest accepted current command until fresh MPC
   activation instead of temporarily executing the preceding plan's next row. Re-evaluate current
   price/export/SOC/device limits; preserve safety controls. Compare through **endogenous**
   mode/limit history with wear and ending inventory. No live change from this diagnostic alone.
3. Establish actual solve/republish clocks and unchanged-helper history semantics; then continue
   inventory-constrained forecast comparisons. Preserve p65 quantile-loss gate alongside economics.
