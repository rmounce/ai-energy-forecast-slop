# Sequential economic replay — 2026-10-04

- Status: bounded one-hour paired MPC comparison completed; 24 installed core solves Optimal.
  Same measured load/delivered PV/observed rates; endogenous battery inventory per arm.
- Scope: fixed historical DH parents/HWC/risk weights, refreshed archived APF; ideal instantaneous
  physical execution. Not full production DH/MPC feedback or deployable savings evidence.
- Decision: load calibration remains a plausible cheap experiment, but the seven-day accuracy
  gain and single-DH forecast cashflow uplift do not establish financial benefit.

## Contract / tools

- `eval/sequential_core_replay.py`: verifies measured/quote/APF/calibration hashes; rebuilds exact
  recorded and calibrated DH parents using existing historical chain contract. Keeps source image,
  optimisation SHA, formatter evidence and parent request identities.
- Initial inventory: previous complete interval's bounded SoC endpoint observation. Thereafter
  each arm advances its OWN simulated SoC; no re-grounding from future observed SoC.
- Decisions: previous complete five-minute power/loss telemetry only. Source means assumed
  available at interval end. Current actual interval used only AFTER solve for physical execution
  and financial scoring; future target mutations tested not to alter current request.
- APF: latest independently received leg at/before origin, max age 15m. No future receipts.
  Exact current non-estimated quote used if received; otherwise explicitly identified already-issued
  `per_kwh` forecast. Raw API feed attributes negative earns; current feed state positive earns.
- Forecast source gate uses canonical start (`end − duration`) strictly AFTER current origin.
  Source `start_time` is +1s; blindly running at an exact boundary would duplicate the current
  slot after its separate quote and shift the curve. Replay normalises this gate; production
  templates unchanged. All 168 tariff/power inputs remain required.
- Same installed payload builder: smoothing, energy scaling, HWC addition, fixed losses, plant
  capacity/weights, MPC boundary SoC and positive-deviation terminal policy. Parents held fixed;
  this experiment includes load's influence through DH inventory, not just tactical load arrays.
- Physical executor: positive DC battery discharge; .99 battery efficiencies, .95 hybrid conversion,
  40.3kWh runtime capacity and configured power/grid/inverter limits. Clip commands to energy and
  AC/DC feasibility; curtail solar then limit excess discharge to prevent prohibited export.
  Current tariff guard allows export only when effective feed state >0. Infeasible site balance fails.
- Measured delivered DC PV is a COMMON SUPPLY LOWER BOUND, even during observed curtailment.
  Does not infer available solar or use the recent Solcast proxy as an actual target. PV/load/rate
  gaps fail the run rather than filling/zeroing/skipping divergent controller histories.
- Cashflow: raw observed quote import cost minus feed-state export revenue, same targets/rows
  for both arms. Throughput/end inventory/curtailment/clipped commands separate; no implicit salvage.
- One cap-free/read-only/network-disabled container; selected code files + credential-free bundle
  only, 1 CPU/2GB, nice 19, max 12 paired steps, 150s worker /180s client budget and named cleanup.
- Shared `scripts/emhass_solver_worker.py.solve_request`: isolated direct core, fresh temp paths
  per solve. EMHASS parser MUTATES config into Python time objects; copy before parsing to preserve
  request identity. Caller rejects request mutation. No HA/wrapper/shared-state writes.
- Saved solves can be replayed offline with `--solve-archive`; exact bundle/request identities must
  match. Cross-platform request mismatch fails; no fallback to new decisions hidden in that path.

## Oct 3 02:00–03:00 UTC result

- 12 five-minute steps per arm, 14h look-ahead each; initial measured SoC **52.86%**.
- Baseline/calibrated DH parents were solved from the Oct 3 01:51 snapshot, calibrated p65
  vintage 01:32; original DH source arrays not updated across this hour.
- 24 solves, **36.59s aggregate solver time**. Inputs, source, exact terminal endpoints and
  physical outputs validated both inside and outside the initial batch worker.

| Metric | Baseline | Calibrated load |
|---|---:|---:|
| Conditional variable cost | $0.020130 | $0.020913 |
| Grid import | 6.536 kWh | 6.548 kWh |
| Grid export | 0 | 0 |
| DC battery throughput | 9.434 kWh | 9.446 kWh |
| Ending inventory | 30.642 kWh | 30.654 kWh |
| Ending SoC | 76.036% | 76.065% |
| Physically clipped commands | 1 | 1 |

- Paired cashflow delta **−$0.000783** (less than 0.1 cent); ending inventory **+0.01154 kWh**.
  Economically neutral at this window's scale; no meaningful savings claim.
  Extra inventory breaks even at 6.78c/kWh, excluding wear and future uncertainty.
- Two executed commands differ >1W: 02:05 **−331 W**, 02:20 **+191 W** (positive discharge).
  Other ten intervals essentially identical. Both request −14.326kW at 02:35; low measured
  solar forces the same inverter-feasible clip to **−11.664kW**.
- Terminal targets differ as expected from DH parents + endogenous deviation: baseline
  **83.86–86.19%**, calibrated **86.34–88.81%**; initial command unchanged.
- Observed live trace for same hour: variable credit **$0.08454**, grid import **5.256kWh**,
  export **0.00191kWh**, ending measured SoC **72.77%**. Replay baseline ends ~1.316kWh higher.
  Material trajectory mismatch: this is not a validated reproduction of live execution.
- Live planned versus measured DC battery mean absolute error **415 W**; replay requested versus
  live planned mean **5303 W**. Largest discrepancy is already in planning/input/cadence; these
  data do not support blaming device execution. Production replans ~1m, while this replay holds
  its origin command five minutes and ignores within-bin quote/telemetry/plan updates.
- Prior single-DH **+$1.287** forecast cashflow change mainly changed assumed consumption.
  Scoring BOTH controllers against SAME measured load removes that artificial improvement.
- Private evidence: `data/energy_replay/sequential_load_replay_20261004_v3/` (offline exact-request
  re-score with corrected v6 rates); `amber_apf_sequential_20261004/` for APF window.
  First successful batch hashed host source after execution; those hashes can differ from staged
  code when editing during a run. New executions hash copied code before launch. Stored solver
  source/image/request identities remain pinned; all saved requests/results independently revalidated.

## Additional confirmed behaviour

- MQTT state/attribute rollover confirmed in frozen raw general stream Oct 3:
  `02:00:16.362091Z`: value0.0261, quoted end02:05, source update02:00:16.359908.
  `02:05:19.395265Z`: NEW value0.03 with OLD end02:05/source update02:00:16.359908;
  `02:05:19.396310Z`: value0.03 with NEW end02:10/source update02:05:19.394276.
  Earlier latest-per-interval logic misassigned the second price to the first interval.
- `transition_receipts`: same value, next interval end +5m, next receipt within100ms; changed
  source update marker for raw legs, matching raw_price for adjusted leg. Exclude identified
  transition rows before quote-revision selection; raw history preserved. Detection limited to
  this confirmed pair pattern; other unmatched/non-atomic failures remain possible.
- v6 week accounting excludes **1797 transitions per leg**, keeps2013 scored intervals;
  variable credit **$13.326** instead of v5's **$11.804**. Prior v5/v2 financial numbers superseded.
  CurrentInterval estimate=false remains valid before end; post-end-only filter now leaves2 bins.
- Sequential v3 re-score changes observed rates only; raw quote histories/physical inputs unchanged,
  all24 decision request identities and solver outputs reused exactly. Rate-rescoring gate rejects
  changed decision/physical inputs. No repeat solver job required for this accounting correction.

- Selected Oct 3 APF window: 25 revisions/leg; first 24 + final exhaust this small window.
  First payload has **193×5m intervals**, ends 18:00Z (~16.16h after receipt); final has **288×5m**,
  ends next day 03:00Z (~24.08h). Horizon varies across revisions; previous 18h pilot was not a
  universal horizon. Keep coverage-based admission, never infer fixed length from one sample.
- Influx event timestamp still assumed availability, not proven HA/application receipt. Separate
  general/feed events not atomic. Source weights/config/HWC held at capture revision explicitly.

## Next priority

1. [Control fidelity audit](control_fidelity_audit_2026-10-04.md) reproduces three live commands
   within0.005W using recorded parents/current inputs; 60 minute-cadence publications versus
   pilot12 held commands. [Minute-cadence pilots](minute_economic_replay_2026-10-04.md) now
   completed with conditional archived parents; each arm's own DH/HWC feedback remains open.
2. Cover later low-solar, full-battery and high-value export windows; same-row scoring, explicit
   available-PV uncertainty, persistent endogenous inventory, and matched final-inventory treatment.
3. Compare conditional terminal/lock-in policies separately from load calibration. Keep cheap tail
   price residual ahead of complex models unless measured horizon-specific regret supports more.
4. Production promotion only after multi-regime economic improvement with realistic execution;
   accuracy gains alone remain insufficient.

```bash
./.venv/bin/python eval/sequential_core_replay.py \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --config data/energy_replay/solver_config_20261004.json \
  --load-comparison data/energy_replay/load_solver_sensitivity_20261004 \
  --calibration data/energy_replay/measured_week_calibration_20261004 \
  --dataset data/energy_replay/measured_week_20261004 \
  --quotes data/energy_replay/amber_observed_week_20261004_v6 \
  --apf data/energy_replay/amber_apf_sequential_20261004 \
  --start 2026-10-03T02:00:00Z --steps 12 \
  --output data/energy_replay/new_sequential_replay
```

Add `--solve-archive data/energy_replay/sequential_load_replay_20261004` for offline exact-request
re-audit; new output path mandatory. All household evidence ignored; no generated targets committed.
Use `--rescore-rates` with a solve archive only for revised scoring rates; decision requests and
physical inputs must match exactly. Quote archive can be regenerated offline from preserved revisions.
