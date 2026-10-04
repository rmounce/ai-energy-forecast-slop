# Offline EMS fallback timing sensitivity — 2026-10-05

- Decision: review timing policy after delivery/energy validation; no production change.
  This experiment conditions on frozen incumbent MPC plans. It does not rerun the optimizer
  with each arm's changed SoC, rank forecast models, or establish deployable savings.
- New `eval/replay_ems_fallback.py`: baseline selects the previous published plan's current
  target at five-minute fallback ticks; challenger retains the most recently accepted command
  until the next complete current-plan publication. Both arms execute newly accepted current
  plans, own their physical SoC, and emit their own EMS mode/grid/PCS/charge/discharge commands.
  Recorded incumbent EMS mode/register/SoC trajectories are not counterfactual controls.
- Admit only on-grid, noncharging, noncurtailing battery export/self-consumption branches.
  PV-only export, grid charging, curtailment, off-grid, missing/stale guards and mixed selected
  target clocks fail explicitly. MPC command coverage gaps >120s fail. Fresh publication activates after latest changed receipt in
  a bounded two-second envelope; unchanged entities may retain older receipt clocks.
- At every raw execution cut, apply current effective-feed receipt (≤15min), SAPN flexible
  export limit, on-grid status, own SoC and physical plant limits. Negative/zero feed prohibits
  grid export. Defaults follow current EMS script: grid10kW, PCS100kW, discharge24kW, charge21kW;
  physical inverter/battery caps remain tighter. No incumbent desired/register limit reuse.
- Historical export-SoC helper absent from raw archive. Explicit static fallback:
  Oct3 journal capture confirms5%, last changed/updated/reported Sep18 08:23:57.974267 UTC.
  Journal/record hashes retained. This is weaker than independent historical availability.
  Export guard must be at/below physical15% floor; higher guard crossing cases fail.
- Both arms assume immediate command activation after settled publication. Script completion
  delay, physical ramp, transient PCS controller and firmware behavior are omitted. Optional
  `--dc-fixed-loss-w 140` applies the existing fixed DC-overhead value as a separate ablation;
  default0 reproduces preceding economics. Current repo YAML hashed, not independently captured
  historical configuration.

## Conditioned economic sensitivity

Frozen Sep30 22:00:23.787881–22:30 UTC; verified initial15min + continuation lineage.
Same raw site-load/delivered-PV holds and observed Amber scoring rates for both arms.

| Metric | Prior-plan fallback | Hold accepted command |
|---|---:|---:|
| Variable credit AUD | 0.60476442 | 0.65450949 |
| Grid export kWh | 2.25361751 | 2.44055058 |
| DC battery throughput kWh | 2.42170759 | 2.61836161 |
| Modeled ending inventory kWh | 29.06048835 | 28.86173029 |

- Extra credit4.9745c; extra DC throughput0.196654kWh; ending inventory−0.198758kWh.
  Nominal40.3kWh plant model; this inventory is model stock, not independently validated
  observed energy from recorded percentage SoC.
- Net value = credit gain + inventory delta × ending-energy value − throughput delta × wear.
  Break-even ending-energy value25.028c/kWh before wear. Using4c/kWh DC wear as an illustrative
  sensitivity lowers it to21.070c/kWh. The configured discharge weight is not measured wear.
- Thus a larger export credit alone cannot support deployment. Reconcile capacity/energy and
  activation delay, then feed changed battery state back into future solves before judging value.
- Fixed140W DC-overhead ablation: same4.9745c extra credit; ending inventory delta−0.198759kWh;
  DC throughput delta+0.196772kWh; break-even25.0278c/kWh before wear. The timing comparison
  remains almost unchanged; absolute battery inventory decreases in both arms. Fixed losses
  do not create additional export value. This is an explicit existing-parameter sensitivity,
  not a fit to this selected window or evidence of physical calibration.

| Timing comparison | DC loss0W | DC loss140W |
|---|---:|---:|
| Extra variable credit AUD | 0.04974507 | 0.04974507 |
| Ending inventory delta kWh | −0.19875806 | −0.19875924 |
| Extra DC throughput kWh | 0.19665402 | 0.19677165 |
| Break-even inventory AUD/kWh, no wear | 0.25027953 | 0.25027804 |
| Break-even inventory AUD/kWh, illustrative4c DC wear | 0.21070296 | 0.21067804 |

- Supported subset stops before later PV-only export branches in the40min archive. No silent
  approximation or opportunistic omission of those branches.

## Reproduction and validation

```bash
./.venv/bin/python eval/replay_ems_fallback.py \
  --history data/energy_replay/ems_control_history_20261004_export \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --replay data/energy_replay/dh_load_feedback_20261004_export_15m \
    data/energy_replay/dh_load_feedback_20261004_export_30m_chunk2 \
  --output data/energy_replay/NEW_FALLBACK_TIMING.json
./.venv/bin/python -m pytest -q tests/test_ems_fallback.py
```

- Authoritative ignored reports: `ems_fallback_20261005_30m_final.json` and
  `ems_fallback_20261005_30m_loss140_final.json`; no raw private data committed.
- 15 tests: credit/inventory tradeoff; causal price/flexible caps; independent controls/SoC;
  stale/off-grid rejection; unsupported branches; physical SOC floor; settled receipt clock;
  static guard's older unchanged clock admission; stale held-command coverage; fixed DC overhead
  increases required battery energy without extra grid credit, and reduces export under a DC
  discharge constraint. Production untouched.
