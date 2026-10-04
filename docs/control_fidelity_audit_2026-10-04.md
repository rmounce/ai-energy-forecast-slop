# Control fidelity audit — 2026-10-04

- Decision: improve replay cadence and DH/HWC input revisions before broader economic ranking.
  Three reconstructed live MPC checkpoints reproduce published commands within **0.005W**.
  No controller/model/policy deployment; no economic improvement established.
- Window: Oct 3 **02:00–03:00 UTC**, same hour as paired sequential pilot.
  Read-only raw archive includes 01:45–03:00 for causal lookbacks.
- Evidence: ignored `data/energy_replay/control_history_20261004/`,
  `data/energy_replay/control_fidelity_20261004_v5/`; v5 independently re-audits v4's exact
  six saved request/result identities. No repeat solver job for this re-audit.

## Confirmed cadence and feedback

- **60 MPC power-plan publications**, 60 distinct curves; spacing median59.89s,
  range49.84–70.82s. Publication alone does not prove a separate solve.
- **14 DH SoC**, **13 DH load**, **2 DH PV**, **14 DH HWC snapshot** publications.
  State-change recording can omit unchanged projections; counts are not solve counts.
- Pilot used 12 commands held5m, one rebuilt DH parent/HWC snapshot, previous-bin mean telemetry.
  Live controller used current telemetry and updated parent plans throughout the hour.
- Input time proxy: preceding `input_number.mpc_last_soc_init` publication within15s of power
  plan publication. Independently as-of recorded telemetry/parent used; no later input observations.
- All **60** input reconstructions admitted; no missing/stale/non-atomic-anchor exclusions here.
  Reconstructed `soc_init` MAE **0.00235 percentage points**, p95 **0.00451pp** against helper.
  Reconstructed terminal SoC matches all published MPC endpoints at numerical precision.
- First MPC endpoint recurrence from recorded starting helper and DC planned battery power:
  MAE **0.00345pp**, p95 **0.00846pp**. Rounding/time-proxy differences remain; endpoint labels
  describe interval ends, not measured SoC at publication.

## Parent/input differences

- Current telemetry/observed SoC held identical for fixed versus recorded-parent reconstructions.
  Same settings, APF receipts, current quotes and payload builder; only parent/HWC revisions differ.
- Recorded-parent PV energy over14h is **2.888kWh lower on average**, range2.696–2.992kWh lower.
  Recorded-parent load energy is **0.481kWh lower on average**, range−0.833 to+0.198kWh.
- Fixed-parent terminal target differs from published live endpoint by MAE **1.351pp**,
  p95 **2.944pp**; mean signed difference+1.135pp. This is a source/policy conditioning difference,
  not an independently measured forecast error.

Six pinned installed-core solves: three deterministic positions0/20/40 of admitted publications,
two matched-clock input variants. All Optimal; aggregate solver time **8.89s**.

| Input-time proxy UTC | Published DC battery W | Fixed parent, current telemetry W | Recorded parent, current telemetry W |
|---|---:|---:|---:|
| 02:00:16.712 | −3890.10 | −2942.00 | −3890.100 |
| 02:20:17.239 | −6049.76 | −4523.20 | −6049.764 |
| 02:40:18.678 | −2937.10 | −1989.00 | −2937.100 |

- PositiveW discharges; negativeW charges. Recorded-parent errors≤0.0043W, consistent with
  published rounding. Fixed-parent differences948–1527W under the same current inputs.
- Strong reproduction at these origins; does not establish every historical config/setting,
  full-horizon plan equivalence, or attribution of all one-hour trajectory differences.
- Observed SoC re-grounded at each checkpoint deliberately: these are input/solver diagnostics,
  not persistent counterfactual inventory or savings calculations.

## Timing and delivered energy

- Time-weighted raw published power reproduces the independent measured export exactly.
- Five-minute boundary-held live command versus within-bin live-plan mean: MAE **4803W**.
- Measured DC battery versus within-bin live plan: MAE **415W**, p95 **864W**.
- Pilot requested versus within-bin live plan: MAE **5303W**. Do not blame device actuation
  for differences already present in plans or cadence.

| Energy trace over hour | Net battery inventory increase |
|---|---:|
| Observed SoC ×40.3kWh capacity | 8.024kWh |
| Observed battery power ×configured efficiencies | 7.945kWh |
| Continuous published live-plan power ×configured efficiencies | 8.181kWh |
| Live plan sampled at each5m boundary and held | 8.639kWh |
| Pilot commands after ideal physical clipping | 9.340kWh |

- Holding boundary commands adds **0.458kWh** versus continuous live-plan trace; pilot ends
  **1.316kWh** above observed SoC. Descriptive comparisons, not additive causal attribution:
  execution clipping, endogenous SoC and parent feedback interact.
- Observed-power recurrence differs from measured SoC gain by **0.079kWh**. Sensor timing,
  capacity/efficiency assumptions and rounding unresolved; no inferred wear/loss calibration.
- Five-minute average recurrence cannot resolve within-bin sign changes/nonlinear clipping;
  no retrospective mean command used as a causal decision input.

## Tools / validation

- `eval/export_control_history.py`: fixed selected entities/fields, read-only Influx, ≤90m,
  ≤10000rows/source, no current-state fallback. Save raw rows/schema/query/hash manifest.
- `eval/audit_control_fidelity.py`: verifies archive/dataset/bundle identities; validates saved
  solver outputs; reconstructs historical inputs and reports exclusions separately.
- `--core-checkpoints 3`: six isolated pinned-image/core solves through shared runner;
  network disabled, read-only mounts, 1CPU/2GB, nice19, bounded solve/client timeouts.
- `--checkpoint-archive FILE`: exact request/image match and output revalidation required;
  no hidden fallback to new decisions. Auditor source hashed before audit starts.
- `snapshot_at` accepts subminute clocks: current quote/forecast is keyed by five-minute target,
  receipts still gated by actual origin. Boundary replay decisions remain unchanged.
- 288 affected tests pass; archive/cache mismatch, future-input exclusion, stale/nonfinite state,
  anchor-before-parent transition and plan integration mismatch covered.

```bash
./.venv/bin/python eval/export_control_history.py \
  --start 2026-10-03T01:45:00Z --end 2026-10-03T03:00:00Z \
  --output data/energy_replay/new_control_history

./.venv/bin/python eval/audit_control_fidelity.py \
  --history data/energy_replay/control_history_20261004 \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --dataset data/energy_replay/measured_week_20261004 \
  --output data/energy_replay/new_control_audit
```

Add `--core-checkpoints 3` for installed-core checks. Add `--checkpoint-archive
data/energy_replay/control_fidelity_20261004_v4/report.json` to reuse the original six solves offline.

## Next effort

1. [Minute-cadence replay completed](minute_economic_replay_2026-10-04.md):15m/30m pilots,
   raw subinterval execution, publication delay, endogenous inventory. Conditional parents retained;
   no meaningful terminal-policy gain. Representative high-value windows next.
2. Refresh each arm's DH/HWC parent and feedback state at historical input revisions. Recorded
   live DH plans can validate baseline inputs; injecting them into both arms does not reproduce
   each arm's counterfactual feedback. Label conditional-parent experiments explicitly.
3. Prove baseline plan/input reproduction across low-solar/full-battery/high-value export windows;
   then compare terminal/lock-in and net-energy challengers with inventory-value sensitivity.
4. Keep APF available. Cheap tail residual correction remains ahead of a complex replacement,
   subject to horizon-specific measured regret. Load calibration remains an accuracy challenger.
