# EMS timing with own optimizer feedback — 2026-10-05

- Decision: retain APF; no production promotion. Thirty-minute timing gain remains primarily
  a trade of inventory for current export credit. [Installed MPC projection parity](mpc_publication_audit_2026-10-05.md)
  passes60 saved cases; next verify activation/capture semantics
  and extend supported branches into longer scarce-inventory execution.
- `dh_feedback_replay.py --experiment ems_timing`: common forecasts and terminal policy;
  each arm owns physical SoC, full accepted MPC plan, EMS command, DH parent, anchor, reground
  block and offset. Each later installed optimizer solve receives its own state.
- Baseline selects the current target from its preceding accepted plan at five-minute ticks;
  challenger holds its accepted command until fresh MPC activation. Common historical publication
  clocks; no incumbent plan/control/SoC reset after initial seed.
- `ems_feedback_policy.py`: two-decimal consumed power projection, negative export sign,
  target-start dates, own curve selection, causal export guards and existing physical executor.
  Shared raw timeline splitter moved to `minute_core_replay.py`; all callers migrated.
- Continuation carries verified owned plan/command. Physics/controller/guard/loss contract rejects
  incompatible checkpoints. CLI requires one immutable EMS archive across chunks; different or
  extended guard histories need a historical overlap audit before admission.
- Supported on-grid noncharging/noncurtailing battery export and self-consumption branches only;
  unsupported counterfactual branches abort. Export guard5% uses older unchanged capture fallback,
  weaker than historical availability; must stay at/below each execution plant's physical floor.
  Fresh MPC coverage≤120s; effective feed≤900s through each execution segment's end.

## Installed-core result

Sep30 22:00:23.787881–22:30 UTC, actual duration1776.212119s. Two bounded batches,
72 validated DH/MPC solves, summed core solve time66.98s. Existing140W DC-overhead ablation,
nominal40.3kWh stock,99% battery and95% inverter efficiencies; no parameter fitting.

| Metric | Prior-plan fallback | Hold accepted command |
|---|---:|---:|
| Variable credit AUD | 0.60487906 | 0.65451127 |
| Grid export kWh | 2.25403899 | 2.44055427 |
| DC battery throughput kWh | 2.49110854 | 2.68744041 |
| Ending model inventory kWh | 28.99026875 | 28.79195373 |
| Ending model SoC% | 71.93615 | 71.44405 |
| Final DH offset percentage points | 0.41 | 0.90 |

- Total credit gain4.9632c; ending inventory−0.198315kWh; extra DC throughput0.196332kWh.
  Changed SoC propagates into later MPC endpoints and DH trajectories/offsets. This closes
  the frozen-plan limitation of the preceding timing sensitivity; physical limitations remain.
- First15min: gain2.6691c, ending inventory−0.097588kWh. Second chunk adds2.2941c.
  Sum cash/flows, use only final inventory difference; do not sum chunk inventory differences.
- Net AUD =0.04963221 −0.19831503×ending-energy value −0.19633188×DC wear cost.
  Break-even ending-energy value25.027c/kWh before wear,21.067c/kWh with illustrative
  4c/kWh DC wear. Configured discharge weight is not independently measured wear.
  At20c/kWh ending-energy value and4c wear: +0.2116c; at30c ending value: −1.7716c.
- Cash gain is not demonstrated system profit. Longer replay must consume/value remaining
  inventory and cover branch transitions; these terminal prices are sensitivities.

## Limits and validation

- Retrospective export selection, no seasonal/regime generalization. APF common and available;
  HWC thermal schedule exogenous; delivered PV a lower bound on available PV.
- Fixed nominal DC stock; recorded percentage SoC uses a changing BMS denominator. Ending model
  inventory is not independently measured energy.
- Instant modeled activation at historical publication clock; no physical ramp, script completion
  delay, transient PCS controller or counterfactual solve-latency measurement.
- Six consumed MPC power channels projected from validated installed-core results using expected
  formatter rounding. Installed MPC formatter parity now passes60 saved cases plus numerical
  fixtures; DH formatter evidence is validated. Coherent publication/helper admission is a modeled improvement.
- Current YAML/plant config and explicit capture fallbacks are not independent historical config.
  No HA/service/device writes, reloads, training or production changes.
- 10 new tests: later MPC/DH feedback, owned checkpoint plan/command, fixed-loss stock/contract,
  future scoring price exclusion, observed SoC exclusion, unsupported branch rejection,
  export sign/rounding, physical-floor changes, unchanged guard archive and full-segment freshness.
- 118 affected tests pass. Both new batches and old calibrated-load baseline reproduce saved
  requests/events/economics/checkpoints exactly after refactoring; no extra solves for verification.
  Source/report hashes in ignored artifacts; private raw data uncommitted.

## Reproduction

```bash
./.venv/bin/python eval/dh_feedback_replay.py \
  --history data/energy_replay/dh_source_history_20261004_export \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --start 2026-09-30T22:00:00Z --end 2026-09-30T22:15:00Z \
  --experiment ems_timing \
  --ems-history data/energy_replay/ems_control_history_20261004_export \
  --dc-fixed-loss-w 140 --output data/energy_replay/NEW_EMS_FEEDBACK_15M
# Repeat for22:15–22:30 with --resume-from NEW_EMS_FEEDBACK_15M and new output.
```

- Ignored outputs under `data/energy_replay/`: `ems_feedback_20261005_export_15m_140w/`,
  `ems_feedback_20261005_export_30m_chunk2_140w/`, `ems_feedback_20261005_export_30m_chain_140w.json`.
- Core image/optimizer/formatter pins unchanged. Disposable read-only/network-none container,
  1CPU/2GiB,≤15MPC/8DH origins/batch,150s worker/180s client; nice19.
