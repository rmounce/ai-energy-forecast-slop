# HWC Cycle Reporting

Per-cycle heat-pump run history surfaced to Home Assistant: a table of recent compressor
cycles (start, duration, ΔT, ambient/wet-bulb, elec kWh, thermal kWh, COP, clean flag) plus a
live row for the cycle currently in progress.

Status: **implemented** (2026-06-27), behind `hwc.reporting.enabled`. Publish-only and firewalled
from the planner/executor. Modules: counter-aware `hwc_cop_analysis.analyse`, pure
`hwc_cycle_reporter.py`, and the `cycle_reporter` task in `services/hwc_daemon.py`.

## Goal

- A HA card showing the last N heat-pump cycles with the key per-cycle statistics.
- A continuously-updating row for the **in-progress** cycle (elapsed, running kWh, current ΔT).
- No new timer/service: the existing HWC daemon owns it.
- Reuse the validated COP/thermal maths in `hwc_cop_analysis.py`; do not reimplement them.

## Why the daemon (not a separate timer job)

`hwc_cop_analysis.py` can only ever reconstruct **completed** cycles from InfluxDB after the
fact. The daemon (`services/hwc_daemon.py`) is a long-lived websocket process that already:

- sees the compressor **on/off edges live** (`_track_compressor_run_event`, `compressor_last_on_at`);
- watches the tank-temp and weather entities;
- owns a persistent state file with `_load_state`/`_save_state` (`data/hwc_daemon_state.json`);
- publishes to HA state (the SoC tracker publishes `sensor.hwc_soc_state`).

So it is the only component that can open a row on the start edge and keep a **current-cycle**
row updating in real time. Adding a ring buffer to the state file and a published sensor is
incremental.

## Energy source: use the Athom cumulative counter, not power integration

The Athom monitor exposes a cumulative kWh counter on the heat-pump channel:

- `sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_energy_2` — **cumulative kWh,
  monotonic** (on-device accumulation).
- vs `..._power_2` — instantaneous W, which `hwc_cop_analysis.py:252` Riemann-sums on a 30 s grid.

**As built**, the counter logic lives *inside* `hwc_cop_analysis.analyse()` (one code path, so the
batch tool benefits too), not in the daemon. For each compressor window `analyse` differences the
cumulative meter (`counter_cycle_kwh`) and records the chosen source in a new `elec_source` column
(`counter` | `power_integration`):

```
elec_kwh = energy_2(cycle_end) − energy_2(cycle_start)
```

This is more accurate than HA-side integration (no missed-sample or grid-quantisation error). The
daemon's **live** in-progress row computes the same way directly — `energy_2_now − energy_2_start`
— from the start snapshot it takes when the compressor turns on.

Cross-check (2026-06-18 cycle): counter Δ ≈ `6.957 − 5.235 = 1.72` kWh vs the integrated
`elec_kwh = 1.68` in `data/hwc_cop_cycles.csv` — agreement to ~2% (gap is standby draw the
counter captures + query-window slack). Counter reporting cadence is ~7 min when idle but
sub-second when power moves, so edge resolution is fine.

Two properties to handle:

- **Counter resets.** ESPHome totals can reset on reboot/firmware flash. A negative cycle delta
  means a mid-cycle reset → flag the row and fall back to the `hwc_cop_analysis.analyse()`
  power-integration value for that cycle.
- **Whole-channel measurement.** `energy_2` captures standby (~2 W, negligible) and, importantly,
  the resistive **element** if it fires. That is *correct* input energy for COP, but a row's COP
  legitimately drops on an element-assisted cycle — surface the existing `element_on` flag next to
  COP so a low value reads as expected, not as a bad measurement.

The power-integration path stays as the in-`analyse` reset/missing fallback; the counter is
preferred for completed (`elec_source` column) and live rows.

## Architecture

A new asyncio task, `cycle_reporter` (`services/hwc_daemon.py`), added to the daemon's `TaskGroup`
in `HwcDaemon.run`. Pure helpers live in `hwc_cycle_reporter.py`; the daemon owns the I/O.

Two halves, both on a single `poll_seconds` (60 s) tick:

**Completed rows — periodic backfill (authoritative).** Every `backfill_seconds` (900 s) and on
the first tick, the daemon runs `hwc_cop_analysis.analyse(since=now−lookback_hours, min_minutes=5)`
in a thread and merges the rows (`merge_records`, deduped by local start) into the ring buffer.
This is deliberately *not* per-cycle-on-close: a periodic re-scan **self-heals across restarts**
(cycles that closed while the daemon was down still get picked up) and keeps the thermal/clean
maths in exactly one place. `analyse` already prefers the counter, so the merged rows carry the
accurate `elec_kwh`.

**Live row — polled edges (lightweight).** Each tick the daemon polls compressor on/off, tank, and
the energy counter and advances a small state machine (`advance_live`):

- **off→on**: open a `running` row, snapshotting `start_ts`, `tank_start`, `energy_start`.
- **running**: refresh `tank_now`/`energy_now`; `live_view` renders elapsed, ΔT, and
  `energy_now − energy_start` kWh.
- **on→off**: mark `cooldown` (kept visible — `analyse` can't see the cycle until its post-window
  exists in InfluxDB).
- once a ring row within ~6 min of `start_ts` appears (`backfill_captured`), the live row clears.

A failed compressor read (`raw_on=None`) skips `advance_live`, so a flaky read never spuriously
closes a cycle.

### Persistence

Daemon state-file payload (`_save_state`/`_load_state`) gains:

```jsonc
{
  "last_reached_target_at": "...",
  "soc": { ... },
  "cycles": [ /* ring buffer, last N completed rows (records) */ ],
  "current_cycle": { "status": "running", "start_ts": 1750.0, "energy_start": 5.235, ... } // or absent
}
```

N configurable (`hwc.daemon.cycle_history_len`, default 20).

### HA publish

Publish `sensor.hwc_cycles` (via the existing `_ha_set_state` path):

- **state**: the **last completed cycle's COP** (the headline statistic; a single at-a-glance
  health number). Carry its real value even when element-assisted/unclean — the `element_on` and
  `clean` flags in the row explain a low reading rather than hiding it. `unknown` until the first
  cycle closes.
- **attributes.cycles**: the ring buffer (completed rows).
- **attributes.current**: the in-progress row, or null.
- **attributes.cycles_today** / **attributes.last_clean_cop**: cheap derived counters for the card.

Rendered with an apexcharts/markdown/flex-table card (same family as `sensor.hwc_soc_state`).

## Safety / isolation

Given the workstream's fragility history, the reporter must be **firewalled from the control
loop**:

- its own task, wrapped so exceptions log and continue — never propagate to planner/executor;
- it must **not** touch the `run_lock`-protected planner/executor state or the command path;
- publish-only telemetry: a failure here must never perturb actuation.

As-built: the reporter task is registered in `HwcDaemon.run` and returns immediately when
`hwc.reporting.enabled` is false (zero overhead when off).

## Config (`config.yaml hwc:`)

```yaml
  daemon:
    cycle_history_len: 10          # ring-buffer length = max rows in the table
  reporting:
    enabled: true                  # gates the cycle_reporter task
    cycles_entity: sensor.hwc_cycles
    energy_counter_entity: sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_energy_2
    poll_seconds: 60               # live-row refresh + compressor-edge poll
    backfill_seconds: 900          # re-run the COP analyser over recent history
    lookback_hours: 96             # window the analyser reconstructs cycles from (must cover cycle_history_len runs)
```

## What shipped (2026-06-27)

- `hwc_cop_analysis.py`: `counter_cycle_kwh` + `energy_2` series; `analyse` prefers the counter and
  emits `elec_source`. Benefits the batch CLI and CSV/markdown output too.
- `hwc_cycle_reporter.py`: pure `advance_live` / `live_view` / `records_from_analysis` /
  `merge_records` / `backfill_captured` / `build_payload`.
- `services/hwc_daemon.py`: `cycle_reporter` task (firewalled), `_reporter_tick`,
  `_reporter_backfill`, counter/compressor/tank reads; `cycles`/`current_cycle` in the state file.
- `tests/unit/test_hwc_cycle_reporter.py`: counter selection incl. reset/implausible fallback, live
  state machine, record projection/merge/capture, payload.

**Still to do (not code):** build the HA card against `sensor.hwc_cycles`
(`attributes.cycles` table + `attributes.current` live row).

## Relationship to auto-tuning

This is the **measurement/monitoring** half of the feedback loop discussed for the heat-rate and
compressor-power parameters (`config.yaml hwc.thermal.*`). It makes per-cycle drift visible without
closing an automatic tuning loop. A genuine auto-tune (periodic robust refit of the rate/power
slopes from accumulated clean cycles, written to an override the planner reads) is a separate,
later step and should land behind its own publish-only watch — not folded in here. See the
hand-anchored parameters in `config.yaml` and `docs/hwc_thermal_characterisation.md`.
