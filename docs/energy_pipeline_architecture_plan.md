# Energy pipeline consolidation

Status: runtime routing baseline checked 2026-10-01; pure payload extraction and resident price shadow implemented offline.
Economic replay [control fidelity](control_fidelity_audit_2026-10-04.md): three live MPC commands
reproduced with recorded parents/current telemetry; cadence/parent input differences identified.
Minute cadence/raw execution pilots now [completed](minute_economic_replay_2026-10-04.md);
each arm's own battery DH feedback now implemented; HWC feedback and broader economic proof remain gates.
[Three stress windows/source admission](economic_regimes_and_dh_admission_2026-10-04.md) frozen;
four reconstructed origins have load one half-hour behind price/PV.
[Own-battery DH feedback](dh_feedback_economic_replay_2026-10-04.md) now passes two bounded
core pilots with aligned admission/coherent acceptance. HWC stays exogenous; next causal net-energy
calibration and export execution tracking. Full-battery and seasonal economic proof remain open.
[Causal load feedback](causal_load_feedback_results_2026-10-04.md) now passes two 36-solve pilots;
forecast/path changes, zero short-window cash/inventory gain, mixed quantile loss. Next state-carrying
bounded chunks beyond projected divergence; inventory constraints and device delivery still gates.
Replay evidence: [payload extraction](energy_pipeline_payload_replay.md),
[resident price shadow](energy_pipeline_resident_price.md),
[source-cache/memory checkpoint](energy_pipeline_source_cache.md). Production unchanged.
[Shadow acceptance](energy_pipeline_price_acceptance.md) rejects obsolete results;
[frozen tariffs](energy_pipeline_tariff_snapshot.md) implemented, publication pending.
Evidence and dependency map: [runtime inventory](energy_pipeline_runtime_inventory.md).
Scope: Amber → forecasts → EMHASS DH/MPC → battery/HWC/dump-load control; include HA,
MQTT bridges, InfluxDB, sibling `../hwc`, and system/user systemd units.

## Evidence and limits

- Read repository units, live HA `automations.yaml` and packages under
  `/opt/dockerfiles/hass/config/`, EMHASS/Amber Compose image declarations, and HWC daemon code.
- Live `packages/emhass.yaml` matches this repository's copy.
- Running EMHASS container uses `v0.17.9`.
- HA API confirms DH/MPC, battery policy, dump-load and Sigenergy controllers are enabled;
  the old direct Amber battery controller is disabled. See the runtime inventory for limits.
- No memory files found in repository `.agents`; historical plans are subordinate to current
  production routing and deployed configuration.
- Existing critical-path latency estimates (~7–8 seconds) predate September's observed
  ~25–30-second price runs; benchmark again before setting migration targets.

## Current ownership

| Component | Responsibility | Coordination today |
|---|---|---|
| amber2mqtt | Amber acquisition, MQTT publication | HA entities; Compose-managed bridge |
| HA Amber entities | Prices/APF exposed to consumers | Attribute/state events; updates are separate |
| Price listener | Debounce APF; run price extrapolation | HA WebSocket → subprocess; model reload each run |
| Load/STPASA/AEMO units | Load forecast, covariate refresh, archives | Independent timers and shared files |
| InfluxDB | Historical measurements and aggregation | Continuous-query timing; load timer offsets |
| HA EMHASS templates | Align arrays, quantile weights, tariffs/losses, HWC load | Large Jinja payloads assembled from mutable entity state |
| HA DH automation/script | Trigger solve, SoC feedback, HWC snapshot, publish | Forecast entity events; queued max 2; fixed delays |
| HA MPC automation/script | Raw Amber solve; SoC interpolation; publish; apply control | Price events + per-minute fallback; single mode drops overlaps |
| EMHASS | DH/MPC numerical optimisation and result publication | Separate solve/publish HTTP requests; shared saved state |
| HWC daemon | Thermal planning and execution | HA WebSocket; confirmed-price → MPC-entity publication handshake |
| HA battery policy | Select battery action, curtailment policy | EMHASS entities; periodic + explicit automation trigger |
| HA Sigenergy scripts | Ordered register changes, limits, ramping | Verified device-specific sequences and readbacks |

HWC affects battery load forecasts; battery MPC prices/curtailment affect HWC decisions.
This is a feedback loop. A simple acyclic dependency graph cannot describe the whole controller.

## Target boundaries

- One resident Python coordinator owns input snapshots, dependency scheduling, forecasting,
  optimisation requests, plan acceptance, retries, and pipeline health.
- `asyncio` handles HA/MQTT events and network clients; bounded worker threads handle blocking
  transforms/inference. One worker per stateful forecast family; models stay loaded.
- Keep EMHASS as the solver initially. Migration of orchestration does not require replacement
  of the optimisation algorithm or importing EMHASS into the coordinator process.
- Keep HWC planner/executor as a domain service initially. Replace its entity-publication
  handshake with explicit plan/input revisions before considering process consolidation.
- HA owns dashboards, user settings/overrides, device integration, and local control safeguards.
  Move optimisation payload calculations and solve ordering into tested Python modules.
- Retain the Sigenergy command sequencer until device behaviour is reproduced and verified.
- Keep archives, backfills, training, and maintenance independently schedulable; they must not
  exhaust the real-time worker pool or delay control.
- Keep one Amber acquisition owner. Start with amber2mqtt; inspect MQTT payload completeness
  and revision metadata before selecting direct MQTT vs HA ingress or retiring the bridge.
- Use systemd for supervision/restart; periodic reconciliation belongs inside the coordinator
  for migrated tasks. External archives/training may retain timers.

## Language decision (2026-10-02)

- Keep Python for coordinator and model/data work. Existing models/payload extraction are Python;
  observed issue was large archive expansion, substantially reduced by target filtering.
- Async I/O + bounded worker threads fit current network coordination. Pure Python CPU work is
  GIL-limited; many NumPy operations release it. Keep mutable frames/models privately owned.
  References: [Python threading with asyncio](https://docs.python.org/3.13/library/asyncio-task.html#asyncio.to_thread),
  [NumPy thread safety](https://numpy.org/doc/2.3/reference/thread_safety.html).
- Go/Rust dispatcher could improve runtime footprint, static contracts and CPU concurrency;
  numerical Python stack would still need a bridge/service or substantial replacement/equivalence work.
  Current measured coordination cost does not justify that migration.
- Revisit for measured CPU-bound coordination, required hard timing limits or native failure isolation.
  Prefer replace/profile a specific bottleneck first; process isolation is separate from language choice.

## Data and task contract

- Normalised input: source, received time, issued time when available, content digest/revision,
  target timestamps, units, quality/freshness. Distinguish receipt age from source-data age.
- Forecast family: complete p30/p50/p70 revision accepted as one bundle. Separate HA entity
  writes are projections of that bundle, not internal completion signals.
- Task reads an immutable snapshot and records exact input revisions. A newer input while
  running schedules one follow-up; no unbounded replay of intermediate updates.
- Explicit results: `price_forecast_ready`, `load_forecast_ready`, `dh_plan_ready`,
  `mpc_plan_ready`; include parent revisions, horizon and acceptance status.
- Dependency readiness uses usable coverage and freshness. An APF update need not fetch every
  covariate again; use validated cached inputs with independent refresh schedules.
- Separate data-refresh triggers from time/control triggers. MPC must reconcile SoC/load and
  apply interval commands even when no market event arrives.
- Reject late/outdated plans before actuation. Keep the last usable plan only within an explicit
  validity window; define behaviour when no plan remains usable.
- Persist accepted inputs/plans and feedback state (e.g. SQLite); resume by reconciling source
  state, not replaying old device commands. Bound retention and retain run provenance.
- Treat thread timeouts correctly: cancelling an await cannot kill a running thread. Prevent
  timed-out work from publishing; watchdog/process restart handles a genuinely stuck worker.
- Give each EMHASS request/publish transaction explicit identity and ordering. Inspect deployed
  solver state before deciding whether DH/MPC can safely run concurrently.

## Feedback ordering

- A planning cycle consumes the last accepted HWC plan as a frozen load input.
- Battery solve completion may request a subsequent HWC replan using that accepted revision.
- HWC publication must not recursively force unlimited battery solves.
- Choose and validate a policy: lagged plans with bounded reconciliation, or bounded alternating
  solves with a convergence rule. Preserve current behaviour first; change policy separately.
- Snapshot SoC, forecast arrays, user settings and HWC load at each solve. Commit feedback anchors
  only according to a documented accepted-plan policy; avoid a failed solve advancing state.

## Migration sequence

1. **Runtime inventory and baseline.** Verify running units/containers, HA enabled automations,
   MQTT contracts, input consumers, actuator writers, failure recovery and measured latency.
   Resolve stale docs; identify authoritative owners and historical paths.
2. **Extract pure calculations.** Python functions for time alignment, unit/sign conventions,
   quantile weighting, PV losses, HWC load reconstruction and SoC policy. Replay recorded inputs
   against current rendered HA payloads; investigate every material mismatch.
3. **Resident forecasting.** Load models once; snapshot/cache inputs; explicit bundle completion;
   replace the listener subprocess path. Retain existing published entities for consumers.
4. **Shadow coordination.** Produce DH/MPC payloads and trigger decisions from the same inputs;
   compare with live execution. No second actuator writer. Exercise disconnects/restarts/staleness.
5. **Move solve ownership.** Coordinator owns DH/MPC solve → result acceptance → publish ordering.
   Disable old HA solve triggers at cutover; verify one owner. Retain manual HA controls.
6. **Unify control intent.** Move battery decision policy after replay/shadow equivalence; send
   intent through existing Sigenergy sequencing. Replace HWC handshake with revision contracts;
   include dump-load/curtailment dependencies.
7. **Remove migrated paths.** Remove superseded units, HA payload templates/helpers, delays and
   triggers. Keep domain/device logic with clear owners; update recovery and operating docs.

Each phase: measurable acceptance gate, source of truth, rollback path, no simultaneous production
actuator writers. Freeze behaviour during extraction; tuning/model changes are separate work.

## Gates and monitoring

- Normal APF update → price bundle latency and memory before/after; model loading measured.
- One market revision yields bounded forecast/solve work; event bursts and slow workers covered.
- UTC interval alignment, DST transition, export sign, gross/base load and HWC tail preserved.
- DH/MPC payload equivalence includes SoC policy and user settings, not just forecast arrays.
- HA restart, MQTT disconnect, EMHASS failure, missing history and invalid forecast recovery.
- Pipeline health reports last usable input/forecast/plan/control confirmation and parent revisions.
  Retry errors remain diagnostic while usable output remains fresh; alert sustained loss,
  structural invalidity, missed control deadlines or missing device confirmation.
- Native crashes/worker hangs remain a single-process risk; watchdog and bounded task queues needed.

## Next concrete deliverable

Runtime routing inventory, dependency/feedback diagram and ownership/removal order are recorded.
Pure payload/time/SoC extraction passes one recorded snapshot and deterministic edge-case replay.
Resident price shadow now uses independent validated source caches. Warm inference ~4s; measured
memory reclamation keeps short runs below the unchanged 2 GiB guard. Current implementation/evidence
and resume gates: [source-cache checkpoint](energy_pipeline_source_cache.md).
Target-filtered STPASA archive reads substantially reduce multi-cycle memory/refresh cost with exact
feature parity; six accelerated refresh/inference cycles use ~1.2 GiB. Bounded HTTP/STPASA/HA freshness
evidence recorded; BOM provider freshness remains unknown. Next: validate long-run memory and expose
missing provider freshness metadata ([staged BOM patch and capture adapter](energy_pipeline_bom_freshness.md)).
Adapter checks observed marker stability with one retry; integration patch still unapplied and
collector/entity atomicity unverified. Tariff reads
now frozen per run. Shadow result acceptance now
rejects changed inputs, expiry and observed validation races. Opt-in [durable checkpoint](energy_pipeline_accepted_store.md)
stores accepted shadows; restart treats them as historical evidence. [Weather horizon admission](energy_pipeline_weather_coverage.md)
now preserves incumbent adjusted-tail fill within a one-hour cap. Live expired-checkpoint recovery and
fresh reconciliation passed; exact quantile reload parity retained.
[Local publication transaction](energy_pipeline_publication.md) now rehearses partial-write recovery,
current-parent checks and a verified completion marker; no HA output transport. [Bundle handoff](energy_pipeline_handoff.md)
now builds coherent DH prices and separately identifies MPC Amber/existing-DH parents; daytime HA
Jinja parity passed. Isolated recorded DH/MPC core solves and [one-cycle historical chain](energy_pipeline_solver_chain.md)
now pass. Next multi-cycle feedback and fresh-parent admission;
remote publication/consumer contract remains before single-owner cutover. Collect independent
DH/MPC daytime/curtailment/recovery snapshots alongside it. APF ingress remains HA WebSocket;
amber2mqtt remains acquisition owner. DH/MPC/control ownership remains in HA.
Production cutover remains gated on shadow equivalence and verified single ownership.

Solver isolation audit: [installed side effects/result contract](energy_pipeline_solver_isolation.md).
Live wrappers write shared state even with saving/posts disabled; direct-core disposable worker
implemented for historical evidence only. [Economic replay checkpoint](economic_replay_checkpoint_2026-10-04.md)
records policy/input/physical gaps before realised-loss attribution.
Separate [measured economic targets](measured_economic_actuals_2026-10-04.md) now exported for
seven completed days with causal forecast windows and an offline load-calibration challenger.
Longer-horizon p65 MAE improves 16–21%; short-horizon gain small. Interval price/revisions now
frozen and observed cashflow scored; fixed-endpoint load and fixed-forecast terminal core sensitivities
pass. Extended APF horizon varies by revision (inspected ~16–24h). Bounded one-hour sequential
MPC comparison now gives negligible load-calibration cashflow effect; baseline inventory/flow
mismatch needs diagnosis. Next baseline execution parity + complete origin manifests/repeated
DH/HWC feedback; verify settlement/PV lineage and inventory;
this evaluation work does not require orchestration cutover.
