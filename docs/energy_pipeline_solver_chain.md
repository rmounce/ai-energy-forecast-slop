# Historical DH → MPC solver chain

2026-10-04. Offline historical evidence only; no live admission/control.

- `energy_pipeline/solver_chain.py`: validated matching historical DH result → three consumed
  HA entities → MPC payload at the original capture clock. Owns copies; no helper/entity writes.
- Requires matching handoff/request identities, Optimal physical checks, original MPC coverage,
  current pure-builder parity with original recorded payloads, installed formatter evidence.
- CLI `scripts/replay_energy_solves.py --kind mpc --dh-result <DH artifact>` requires same pinned
  image, solver source and frozen config. Request retains DH result revision and price parent;
  full chained snapshot retained in ignored result for template replay. Chained record only supports MPC.
- Replace all three channels together: `sensor.dh_p_load_forecast`, `sensor.dh_p_pv_forecast`,
  `sensor.dh_soc_batt_forecast`. Original Amber prices, HWC and measured telemetry stay frozen.
- Installed `RetrieveHass.get_attr_data_dict` rounds schedule values to two decimals, serialises
  strings; SoC multiplied by 100. Dates remain interval starts; SoC values describe interval ends.
  HA interpolation adds 30 minutes and prepends the `dh_last_soc_init` anchor at plan start.
- Helper anchor uses `dh_soc(...).soc_init_pct`, not `100*payload.soc_init`; fractional payload
  rounding loses percent precision. Local reground helper updated only when wrapper would update it.
- Offset automation preview recorded separately; it affects a subsequent DH, not this MPC.
  No claim about observed helper event timing, HWC snapshot event or latency across boundaries.
- `publication_authorized=False`, all `solve_authorized=False`. Historical readiness never becomes
  fresh authority; no `AcceptedDHSolve` minted. Live result admission remains a separate gate.

## Evidence

- Same Oct 3 01:51:09 UTC capture and Oct 4 config as [core rehearsal](energy_pipeline_solver_isolation.md).
- Replayed DH Optimal 2.86s, 144 rows. Pure projection exactly matches installed static formatter.
  Formatter source SHA-256 `4c46ed61c15f611e9462f69c4b8c91b6c65d2f84b96cb25228423e92ec66e8b3`.
- Chained MPC Optimal 1.14s, 168 rows; tariff, AC/DC balance, grid/inverter and SoC checks pass.
- Recorded-parent/new-parent MPC: terminal SoC 83.82%/83.61%; initial 48.44% unchanged.
  Load/PV arrays unchanged; first battery action unchanged at −6172.3 W DC (charge).
  Different terminal energy prohibits treating forecast cashflow delta as savings.
- Installed HA Jinja oracle: ZERO DH/MPC payload mismatches for 17 recorded/synthetic cases,
  including weights, full/low SoC, reground, missing plan, sparse HWC, curtailment, boundaries/DST.
  Empty-power scenario has only two MPC slots: arithmetic parity is not coverage admission.
- 188 energy-pipeline tests pass outside sandbox; 51 focused chain/replay/payload tests inside.

Private artifacts: `data/energy_replay/dh_projected_20261004.json`,
`mpc_chained_20261004.json`, `chained_snapshot_20261004.json`; ignored, not distributed fixtures.

Re-run with the journal/config/image/source flags from the [economic checkpoint](economic_replay_checkpoint_2026-10-04.md),
adding `--kind mpc --dh-result data/energy_replay/dh_projected_20261004.json` and a new output path.
Older DH artifacts without `projected_dh_entities` must be regenerated to establish formatter parity.

Next: correct economic targets/PV provenance; freeze recent matched inputs/actuals. Multi-cycle state
feedback, execution model and fresh-parent admission still require separate evidence.
