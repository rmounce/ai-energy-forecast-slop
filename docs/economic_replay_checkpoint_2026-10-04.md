# Economic replay checkpoint — 2026-10-04

Decision: reuse existing economic evaluation; add production solver/policy fidelity before
ranking terminal-policy, net-energy and APF-tail changes. No production change.

## Existing work retained

- `eval/rolling_mpc_eval.py`: rolling 14h dispatch, strategic handoff, quantile/policy experiments,
  tariffs, logged load substitutions, physical/regret diagnostics and saved comparisons.
- `eval/dispatch_simulator.py`: fast SciPy/HiGHS dispatch and realised interval accounting.
- Existing economic scripts are substantial. New historical worker checks their assumptions
  against installed EMHASS; it does not replace their experiment/reporting machinery.

## Confirmed fidelity gaps

| Area | Existing replay | Recorded production/core evidence | Required for economic attribution |
|---|---|---|---|
| Battery | 40 kWh, 10 kW, 0.95 charge/discharge efficiencies | Runtime 40.3 kWh; 0.99 battery efficiencies plus separate 0.95 inverter conversions; 9.98 kW AC inverter limits | Model DC/AC losses and limits consistently |
| Optimisation penalties | $0.05/kWh throughput default | Runtime discharge weight 0.04, charge 0; inverter stress enabled | Match objective; report cashflow and assumed wear separately |
| Strategic DH | Price-only 72h solve; optional exact terminal constraint already implemented | Load, PV, tariffs, hybrid inverter and recorded exact terminal target | Feed causal site forecasts into strategic solve |
| Forecast inputs | `netload_tariffed` defaults to realised load/PV; logged load alternatives available | Production uses forecast load/PV with HWC in load | Label oracle inputs; record forecast coverage and fallback use |
| State feedback | Re-solves from simulated current state | DH persisted anchor/offset, signed deviation; MPC positive lock-in and full-SoC guard | Reproduce current [SoC policy](production_soc_policy.md) over successive solves |
| Delivered action | Simulated first-step dispatch | Device limits, execution modes, curtailment/top balancing | Compare planned and delivered power before attributing forecast loss |
| Data window | Historical comparisons and exports predate current policy/post-PEC conditions | Fresh forecast logs exist; matched evaluation exports need refresh | Freeze causal origins, actuals, tariff/config lineage and missing-data exclusions |

Existing summaries also report different ending inventory. Cashflow differences alone do not
establish economic improvement. Compare matched initial/terminal energy or state an inventory
valuation and its sensitivity. Oracle substitutions diagnose headroom, not achievable savings.

## First implemented milestone

Separate historical core-solver runner; [isolation contract](energy_pipeline_solver_isolation.md).

- Frozen handoff from `/tmp/resident-handoff-20261003.sqlite`, latest record, Oct 3 01:51:09 UTC.
- Credential-free plant/optimisation config exported Oct 4; original capture-time equivalence
  unproven. Runtime overrides retained. Inputs and detailed outputs stay ignored/private.
- Immutable installed image: `sha256:e59cf736171831f58730adc8a4e0204c2fdb01cb6fc60351a70f72019f5791e8`.
- Installed `optimization.py` SHA-256: `cc709d26ad2b1e39fbace7f09f768592695c44c1ebbfc151afa76ec34c9ca624`.
- DH: 144 half-hour rows, Optimal, 2.73s; initial SoC 42.08%, exact final 83.71%.
- MPC: 168 five-minute rows, Optimal, 1.20s; initial SoC 48.44%, exact final 83.82%.
- Both pass input, tariff, power-balance, grid/inverter-limit and SoC/end-point checks.
  `SOC_opt` is end-of-interval, `P_batt` positive means DC discharge.
- Independent recorded payloads: MPC still uses its recorded DH parent. Chained new DH → MPC
  projection and production feedback are not yet implemented.
- No realised settlement comparison, loss ranking or savings claim yet. No live admission.
- Validation: 172 energy-pipeline tests pass outside sandbox; restricted sandbox stalls existing
  thread-to-asyncio wakeups. Replay/result-contract subset: 35 pass inside sandbox.

Example (paths are local private evidence, not bundled fixtures):

```bash
./.venv/bin/python scripts/replay_energy_solves.py \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --config data/energy_replay/solver_config_20261004.json \
  --image sha256:e59cf736171831f58730adc8a4e0204c2fdb01cb6fc60351a70f72019f5791e8 \
  --optimization-sha256 cc709d26ad2b1e39fbace7f09f768592695c44c1ebbfc151afa76ec34c9ca624 \
  --kind dh --output data/energy_replay/new_dh_solve.json
```

Use `--kind mpc` for recorded MPC; output must not already exist. Config requires only
`retrieve_hass_conf`, `optim_conf`, `plant_conf`; retrieval allowlist is in
`energy_pipeline/solver_replay.py`. Never pass production secrets/params pickle to the runner.
Output includes full request/result identities, physical outputs, forecast-only cashflow and
`publication_authorized: false`. It is not an `AcceptedDHSolve` or a control artefact.

## Next bounded work

1. Verify end-of-interval DH projection against HA, then chained DH/MPC anchor/offset/lock-in.
2. Freeze a recent matched event/quiet-day manifest: as-issued APF/tail/load/PV/HWC, current
   tariff and config, actual price/load/PV/SoC, delivered battery/grid power. Record gaps explicitly.
3. Reuse rolling evaluator/reporting; compare fast replay with core solver on a small set of origins
   before a multi-day run. Keep oracle inputs out of the baseline.
4. Attribute realised loss/headroom to execution, state policy, PV, load/HWC and price horizon.
5. Hold forecasts fixed for terminal/lock-in experiments; prioritise net-energy calibration or
   16.5–36h price residual correction according to measured headroom.
