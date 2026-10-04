# EMHASS solve isolation checkpoint

2026-10-04. Installed-source audit plus isolated historical DH/MPC rehearsals.

- Compose pins v0.17.9; installed Python source `/app/src/emhass`.
- `command_line.dayahead_forecast_optim`: `save_data_to_file=False` still writes
  shared `data_path/opt_res_latest.csv` unless `debug=True`. Flag selects filename, not disables saving.
- `entity_save=True` calls `publish_data(..., entity_save=True, dont_post=True)`:
  writes entity state even with HA posting disabled. Current DH payload includes `entity_save=True`.
- Web action also saves injection state and returns text acknowledgement, not dataframe.
  Separate prefix or disabling HA posts does NOT establish shadow isolation.
- `Optimization.perform_optimization` exposes dataframe-returning core interface. Class inspection
  found inverter pickle reads; this is not proof of zero side effects throughout dependencies/solver.
  Core solves now run only in disposable containers with private temporary paths.
- Selected boundary: separate worker/private workspace, frozen plant/solver configuration,
  no HA credentials/network, direct optimisation API, explicit output result. Do not invoke live
  action endpoint or production command-line wrappers for shadow solves.
- `energy_pipeline/solver_result.py` admits only Optimal, complete aware 144×30m output matching
  still-current price parent; finite nonnegative load/PV, fraction SoC, owned copy and content identity.
  Eight deterministic tests pass. Historical solver results use a separate validation path; acceptance is an internal
  result contract, not permission to publish/control.
- Source and real outputs verify `SOC_opt` is end-of-interval; positive `P_batt` is DC discharge.
  Terminal SoC is enforced by an exact energy equality, not a soft objective.
  [Historical projection/new DH → MPC](energy_pipeline_solver_chain.md) now verified for one frozen
  ordered cycle; multi-cycle/fresh-parent admission remains unverified.

## Historical worker implemented

- `energy_pipeline/solver_replay.py`: frozen recorded payload/config identity; coverage and supported
  field checks; Optimal-only result validation. Audits input powers/tariffs, target grid, grid signs,
  exclusive import/export, grid/inverter limits, curtailment, AC/DC balance, SoC transitions/endpoints.
- `scripts/emhass_solver_worker.py`: direct core API; no web/CLI publication wrappers.
- `scripts/replay_energy_solves.py`: immutable local image, source SHA-256 check, two read-only file
  mounts; no production mounts, credentials or network; read-only root filesystem and private `/tmp`.
  Cap-free root required by image's root-owned uv interpreter. One CPU/thread, 2 GiB memory,
  45-second maximum solver budget, 90-second client timeout with named-container stop.
- Capture readiness is historical evidence; it does not renew freshness or authorise operations.
  Resource overrides recorded. Deferrable solver loads unsupported: HWC must already be in load.
- Runtime capacity/minimum SoC/weights override static configuration, as production does.
  Endpoint requiring clamping is rejected; unsupported payload fields are rejected.
- Oct 3 01:51:09 UTC capture: DH 144×30m Optimal in 2.73s; MPC 168×5m Optimal in 1.20s.
  Physical/result checks passed; core temporary workspaces contained no generated files.
  MPC used its recorded existing DH parent, not the newly replayed DH result.
- Config exported Oct 4, inputs captured Oct 3: original-time config equivalence is unproven.
  Forecast cashflow is not realised savings. No live result admission, publishing or control.

Reproduction and economic gaps: [economic replay checkpoint](economic_replay_checkpoint_2026-10-04.md).
Next: correct PV target provenance, reproduce successive policy against matched historical inputs/actuals,
then economic loss attribution. Production ownership unchanged.
