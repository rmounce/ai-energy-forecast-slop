# Energy pipeline consolidation

Status: initial assessment 2026-10-01; implementation not started.
Scope: Amber → forecasts → EMHASS DH/MPC → battery/HWC/dump-load control; include HA,
MQTT bridges, InfluxDB, sibling `../hwc`, and system/user systemd units.

## Evidence and limits

- Read repository units, live HA `automations.yaml` and packages under
  `/opt/dockerfiles/hass/config/`, EMHASS/Amber Compose image declarations, and HWC daemon code.
- Live `packages/emhass.yaml` matches this repository's copy.
- Compose declares EMHASS `v0.17.9`; running container version not checked in this assessment.
- Automation definitions establish configured routing; enabled state and actual execution
  require HA API/traces. An old direct Amber battery controller remains defined; do not assume
  it is enabled or competing with EMHASS.
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

Complete runtime inventory and produce a current dependency/feedback diagram plus a component
ownership/deletion list. Use those to settle coordinator boundaries and the first migration phase.
This initial plan authorises no production cutover.
