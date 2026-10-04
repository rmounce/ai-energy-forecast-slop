# Continued feedback and export delivery — 2026-10-04

- Decision: retain p65 calibration offline. APF common to both arms; no production changes.
  Forecast/path differences still need an executed economic benefit.
- `dh_feedback_replay.py --resume-from DIR` carries each arm's battery SoC, accepted DH
  trajectories/helpers/offset, parent revision and last activated command/effective plant.
  No reset to recorded SoC or a new archived parent at batch boundaries.
- `feedback_checkpoint.verified_checkpoint`: reproduce preceding requests, events, economics
  and checkpoint using exact saved core results before accepting continuation. Verify bundle
  digest and experiment/configuration/calibration/capacity contract. Older saved runs can derive
  a checkpoint only after reproduction. No new solve required for this verification.
- Gap before next decision: previous own command + explicit raw held telemetry, maximum120s;
  no interpolation or renewed source receipt. Second batch carries 18.730847s before its first
  MPC origin. DH updates during that gap also participate in the event timeline.
- `summarize_feedback_chain.py`: verify all checkpoints and preceding-report hashes; sum flows,
  retain first initial SoC and last ending inventory. Reject a missing/reordered predecessor.
- Tests: split/unbroken request identities, final state and flows; ignore replacement archived
  SoC/parent/command; explicit gap holds; reject incompatible/future/long-gap checkpoints,
  altered saved state and report lineage; derive older verified checkpoints.

## Execution evidence

- Export run Sep30 22:00–22:40 UTC: 96 pinned core solves across three batches; correction
  affects DH/MPC paths but **cashflow difference $0, ending inventory difference0 kWh**.
  This reaches past both first projected MPC divergence (22:20) and original DH divergence
  (22:30); replanning still removes the current-command difference. Third batch admits two DH
  refreshes and rejects one rollover update, retaining each arm's accepted parent.
- Both arms: variable export credit $0.66939740, exported2.50894411 kWh,
  imported0.00004805 kWh, DC throughput2.70137792 kWh; SoC78.18%→71.409120%,
  ending inventory28.77787544 kWh. Final target offset baseline+0.89%, calibrated+4.55%.
- These are conditional ideal-executor results, excluding wear, fixed charges and terminal
  inventory value. HWC remains exogenous; measured delivered PV is a supply lower bound.
- Attempted22:30–22:45 batch rejected before solver: published MPC curve22:41:27.337814,
  latest recorded solve anchor22:40:18.754043 (68.584s old, versus15s admission bound).
  Do not increase the bound or invent a decision. Need distinguish genuine solves, republished
  plans and an unchanged helper value that did not generate a stored sensor event.
- Extended frozen source archive: `dh_source_history_20261004_export_extended/`,
  Sep30 22:10–23:30 UTC. Read-only export; no device/API writes.

## Export-delivery diagnostic

First pilot22:00–22:15, identical raw held telemetry to the replay. Battery sign: positive charge;
DC discharge shown positive below. Net grid export = export minus import.

| Quantity | kWh |
|---|---:|
| Observed DC discharge | 1.88964816 |
| Simulated DC discharge | 2.01114744 |
| Observed net grid export | 1.72249127 |
| Simulated net grid export | 1.85954368 |
| Export gap | 0.13705241 |

Energy identity for this discharge window, using model conversion efficiency eta=.95:

- Tracking contribution: eta × (model DC discharge − observed DC discharge) = +0.11542432 kWh.
- Model PV curtailment: −eta ×0.02298005 = −0.02183105 kWh.
- Conversion/asynchronous residual: eta × (observed PV + observed DC discharge)
  − (observed site load + observed net export) = +0.04345914 kWh.
- Sum = export gap. Observed PV0.04601002, site load0.07292485 kWh;
  aggregate AC/DC ratio0.927548 versus model0.95. Residual averages~174W over15min.
  This is an accounting decomposition, not causal allocation: different sensor clocks,
  conversion/fixed losses, delivered-versus-available PV and curtailment assumptions remain.
  Do not infer a new efficiency or recoverable savings from one quarter-hour.

Confirmed control configuration, read-only inspection Oct4:

- Live `/opt/dockerfiles/hass/config/automations.yaml` MPC automation explicitly triggers battery
  controller after publication; timer skips minute%5==0. Battery controller also has5min fallback.
  Minute-level actuation exists; claiming it runs only every5min would be incorrect.
- Controller selects latest curve row date≤utcnow. At fallback ticks it can select a *future row
  from the preceding minute's plan*, until another MPC plan arrives. Partial-export branch calls
  `configure_sigen_ems_state` with grid export limit derived from P_grid, not a direct P_batt target.
- Nominal battery row at22:05:315.79W; next MPC publication22:05:21.655533.
  Observed DC discharge averaged1,280.92W before that publication,7,579.42W during next20s.
- At22:10:179.16W; next publication22:10:25.857036.
  Observed discharge2,161.14W before,7,482.20W during next20s.
- Timing consistent with temporary export reduction and recovery. No archived service-call/
  device-setpoint traces establish causality; grid limit, SOC guards, mode-single execution,
  device ramp/readbacks and sensor timing must be included before proposing a controller change.

## Reproduction and next work

```bash
./.venv/bin/python eval/dh_feedback_replay.py \
  --history data/energy_replay/dh_source_history_20261004_export \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --start 2026-09-30T22:15:00Z --end 2026-09-30T22:30:00Z \
  --experiment load_calibration \
  --calibration data/energy_replay/measured_week_calibration_20261004 \
  --dataset data/energy_replay/measured_week_20261004 \
  --resume-from data/energy_replay/dh_load_feedback_20261004_export_15m \
  --output data/energy_replay/NEW_CONTINUATION
./.venv/bin/python eval/summarize_feedback_chain.py \
  data/energy_replay/dh_load_feedback_20261004_export_15m \
  data/energy_replay/NEW_CONTINUATION \
  --output data/energy_replay/NEW_CHAIN.json
```

- Ignored evidence: `dh_load_feedback_20261004_export_30m_chunk2/`,
  `dh_load_feedback_20261004_export_40m_chunk3/`,
  `dh_load_feedback_20261004_export_{30m,40m}_chain.json`,
  `export_delivery_{decomposition,boundary_diagnostic}_20261004.json`.
  Raw energy diagnostic integrates both before/after-activation segments by duration;
  boundary diagnostic selects prior published curve via `parse_curve/asof`, then cuts raw holds
  at5min tick, next publication and20s later. Outputs are diagnostics, not solver inputs.
- Next priority: replay actual controller grid-cap/mode/fallback semantics and validate against
  measured dispatch; establish solve/republish clocks; continue inventory-constrained economic
  tests. Keep calibrated p65 a challenger and preserve its mixed quantile-loss finding.
- Validation:338 affected tests pass outside sandbox;9 checkpoint/chain tests rerun after
  tightening complete saved-solve reproduction. Sandbox run stalled/timed out in unchanged
  thread-to-async listener/source tests; same cases pass outside sandbox. Existing Influx
  datetime deprecation warning only. Verified all three real saved batches again.
