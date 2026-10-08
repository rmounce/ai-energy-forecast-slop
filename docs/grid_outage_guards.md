# Grid outage guards

- 2026-10-08: dump-load and battery policy automations now trigger on
  `sensor.sigen_plant_grid_connection_status` changes.
- Dump loads: first branch turns all four switches off unless status is exactly `On Grid`.
  Applies to all triggers; negative prices cannot admit heaters while off-grid.
  `restart` mode interrupts an in-flight admission sequence.
- Battery policy: existing backup branch retained; queued mode preserves command ordering
  and processes a grid transition arriving during a policy execution. No new hardware mode.
- actrl: existing 10-second loop treats curtailment as zero unless grid status is `On Grid`.
  Retains normal integral decay, room-offset bounds and comfort control. No instant reset.
- Unknown/unavailable grid status uses the same fallback. On-grid restoration resumes normal
  policy/forecast admission; dump loads may reenter if their other conditions permit.
- Sources: `hass/automation-dump-loads.yaml`, `hass/automation-sigenergy-emhass.yaml`.
  Live actrl source is external to this repository; applied diff preserved in
  `hass/appdaemon/actrl_grid_status.patch` (paths relative to its app directory).
- Scope: software response after telemetry arrival; cannot protect before grid status updates,
  during HA/network failure, or when plug service calls fail. EMHASS remains grid-connected
  in its optimisation assumptions. Extended-outage comfort/HWC policy remains separate work.

## Deployment verification

- HA configuration check passed; automation domain hot reload succeeded; both target
  automations remained enabled. No HA restart.
- AppDaemon log confirmed actrl source reload and resumed control ticks.
- Focused offline checks: on-grid forecast still increases integral; off-grid, unknown and
  unavailable statuses skip forecast reads and decay a 3.0 integral to 2.95 in one
  ordinary-gain tick. Both YAML grid triggers and four-switch priority shedding verified.
- No live grid-status spoofing or intentional outage performed. Real outage latency still
  depends on telemetry, ongoing battery script execution and device service response.
