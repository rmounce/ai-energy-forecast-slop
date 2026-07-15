# HWC Cycle Reporting

Per-cycle heat-pump run history surfaced to Home Assistant: a table of recent compressor
cycles (start, duration, ΔT, ambient/wet-bulb, elec kWh, thermal kWh, COP, clean flag) plus a
live row for the cycle currently in progress.

Status: **implemented** (2026-06-27), behind `hwc.reporting.enabled`. Publish-only and firewalled
from the planner/executor. Modules: the SQLite system-of-record `hwc_cycle_store.py`, the shared
`hwc_cop_analysis.cycle_metrics`, pure `hwc_cycle_reporter.py`, and the `cycle_reporter` task in
`services/hwc_daemon.py`.

> **2026-06-30 rearchitecture.** The daemon now **records every cycle live to a local SQLite store**
> (`data/hwc_cycles.sqlite`) — a 30 s trace plus precise compressor-edge snapshots, finalised
> through `cycle_metrics` — instead of reconstructing completed cycles after the fact from InfluxDB.
> InfluxDB is no longer read for HWC. The store is the durable system-of-record (InfluxDB's `rp_raw`
> kept only 30 days and never downsampled HWC entities); see **`docs/hwc/local_store.md`** for the
> why and the schema. Sections below describe the as-built live-recording design.

## Goal

- A HA card showing the last N heat-pump cycles with the key per-cycle statistics.
- A continuously-updating row for the **in-progress** cycle (elapsed, running kWh, current ΔT).
- No new timer/service: the existing HWC daemon owns it.
- Reuse the validated COP/thermal maths in `hwc_cop_analysis.py`; do not reimplement them.

## Why the daemon records live

The daemon (`services/hwc_daemon.py`) is a long-lived websocket process that already sees every
HWC `state_changed` event. Rather than reconstruct completed cycles after the fact, it **observes
the run as it happens** and writes it to the store: it caches the latest value of each HWC sensor
from the websocket, opens a cycle on the compressor's off→on edge (snapshotting tank/energy at the
edge), samples the cache into a 30 s trace while running, and finalises on the on→off edge via the
shared `cycle_metrics`. This removes the whole after-the-fact machinery (settle delays, narrow
re-analyse windows, tolerant dedup of analyse-window wobble) — there is no reconstruction, so there
is nothing to wobble. It is also the only component that can keep a real-time **current-cycle** row.

## Energy source: use the Athom cumulative counter, not power integration

The Athom monitor exposes a cumulative kWh counter on the heat-pump channel:

- `sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_energy_2` — **cumulative kWh,
  monotonic** (on-device accumulation).
- vs `..._power_2` — instantaneous W, which `hwc_cop_analysis.py:252` Riemann-sums on a 30 s grid.

**As built**, the counter logic lives *inside* the shared `hwc_cop_analysis.cycle_metrics` (one code
path for both the daemon's live finalise and the offline `analyse`), not duplicated in the daemon.
It differences the cumulative meter (`counter_cycle_kwh`, or the precise edge snapshots the daemon
passes) and records the chosen source in the `elec_source` column (`counter` | `power_integration`):

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

The `cycle_reporter` task (`services/hwc_daemon.py`) is in the daemon's `TaskGroup`. Pure projection
helpers live in `hwc_cycle_reporter.py`; the store I/O in `hwc_cycle_store.py`; the shared
trace→summary maths in `hwc_cop_analysis.cycle_metrics`. The daemon owns the orchestration.

**Cache (websocket-fed).** `_reporter_observe`, called from `_read_events` for every event, keeps a
`report_cache` of the latest value of each configured HWC entity (`hwc.reporting.entities`), plus
humidity from a weather entity's attribute (`hwc.reporting.humidity_entity`). Near-zero cost; no
per-tick REST storm.

**Edges (compressor state-change).** On the compressor entity's event `_reporter_compressor_edge`
detects an off→on or on→off flip against the previous observed state:

- **off→on** (`_reporter_open`): open a `status='running'` row, snapshotting `tank_start` and
  `energy_start` from the cache *at the edge*; write the first trace sample.
- **on→off** (`_reporter_close`): snapshot `energy_end` and the compressor-off edge tank, append a
  boundary trace sample, and mark the cycle **settling** (`closed_at`) rather than finalising
  immediately. A run shorter than `min_cycle_seconds` (300 s, matching the analyser floor) is
  **discarded** here (a defrost flicker that briefly drops the compressor sensor), deleting its
  `running` row and samples.
- **finalise (`_reporter_finalise`, after the settle):** once `close_settle_seconds` (60 s) have
  elapsed — checked on the sampler tick — load the trace and finalise through
  `cycle_metrics(trace, edges=…)`. `cs`/`ce` and the counter elec (`energy_start`/`energy_end`) are
  edge-precise; `tank_start`/`tank_end` are instead the **min/max tank reading across the cycle
  window** (thermal inertia/stratification can dip the probe below its compressor-on reading well
  into a run, so an edge-only start understates the true delta-T and therm_kwh/COP). The edge tank
  snapshots are only a fallback for a trace with no tank readings. The summary upserts over the
  `running` row (same `start_ts` PK → `complete`).

  **Why the settle.** The tank probe keeps rising for a few seconds *after* the compressor stops
  (residual heat / probe lag) — e.g. on 2026-07-01 the compressor went off at `05:07:35` and the tank
  ticked 59→60 at `05:07:38`, a ~3 s lag that a same-instant snapshot missed (the row read 59, not
  60). The settle window extends the trace `tank_end` search past `ce` to capture that peak, and the
  sampler keeps appending during the settle so it lands in the stored trace for the offline
  recompute too. Elec/energy still end at the compressor-off edge (`energy_end` is frozen there);
  only the tank window extends past it.

**Sampler (`sample_seconds`, 30 s).** While a cycle is open *or settling*, `_reporter_sample_tick`
appends one trace sample from the cache (this is what pulls the settling probe's post-off peak into
the trace) and, once the settle window has elapsed, finalises the cycle; every tick (open or not)
republishes the card so the live row's elapsed time and running kWh stay current.

**Startup / restart (`_reporter_startup`).** Seeds the cache from current HA states, seeds the
previous-compressor-state from the live reading (so the first event is a real edge, not a phantom
open), and resolves a persisted cycle: **finalise** it if it was settling when we went down (the
probe has since settled); otherwise resume it if the compressor is still on, or drop it if the
compressor is now off (the run ended during downtime and can't be reconstructed precisely —
forward-only, gaps acceptable).

### Persistence

The store (`data/hwc_cycles.sqlite`) is the system-of-record: the card ring is `recent_cycles(N)`
and the in-progress row is the `status='running'` row. The daemon state file
(`_save_state`/`_load_state`) keeps only the open cycle's edge snapshot, so a restart can resume it
(or, if it was settling, finalise it):

```jsonc
{
  "last_reached_target_at": "...",
  "soc": { ... },
  // open cycle; once the compressor stops it also carries closed_at/end_ts/energy_end/tank_end_edge:
  "reporter_cycle": { "start_ts": 1750.0, "tank_start": 45.0, "energy_start": 5.235 } // or absent
}
```

Card length is `hwc.reporting.history_len` (default 20).

### HA publish

Publish `sensor.hwc_cycles` (via the existing `_ha_set_state` path):

- **state**: the most recent **computable** cycle COP (the headline statistic; a single
  at-a-glance health number). Carry its real value even when element-assisted/unclean — the
  `element_on` and `clean` flags in the row explain a low reading rather than hiding it. A run can
  land with a **null COP** (e.g. a sparse tank probe leaving the cycle-start temp as a leading-NaN
  interpolation when the finalise window starts in a data gap; this self-heals on the next
  re-analyse), so the headline skips null-COP rows rather than going `unknown`. `unknown` only
  until the first computable cycle closes. The card likewise renders `–`/`?` for any null field —
  one degenerate row must never blank the whole table.
- **attributes.cycles**: the ring buffer (completed rows).
- **attributes.current**: the in-progress row, or null.
- **attributes.next**: the next not-yet-started compressor-on block from the planner's own already-
  published schedule, or null — see "Next planned cycle" below.
- **attributes.cycles_today** / **attributes.last_clean_cop**: cheap derived counters for the card.

Rendered with an apexcharts/markdown/flex-table card (same family as `sensor.hwc_soc_state`).

### Next planned cycle

`hwc_cycle_reporter.next_planned_record` (`hwc_cycle_reporter.py`) projects the top row of the card
from the planner's own already-published output — `sensor.hwc_power_plan`'s `deferrables_schedule`
attribute (per-timestep planned watts) and `sensor.hwc_predicted_temp`'s `predicted_temperatures`
attribute (per-timestep predicted tank temp), both on the planner's grid (`optimization_time_step`
minutes, `horizon_steps` steps). It scans forward from "now" for the next run of nonzero-power
timesteps — skipping past the currently-running block first if the compressor is already on (that's
the `current` row) — and returns its predicted start, duration, tank temp range and elec_kwh, or
`None` if there's no schedule or no upcoming on-block in it.

This reads the planner's *externalised* state (an HA entity it already publishes), not its in-process
plan — no new REST polling either: the daemon watches `sensor.hwc_power_plan`/`sensor.hwc_predicted_temp`
via the same websocket `state_changed` stream it already needs for `plan_entities` (the executor
re-arm trigger), and caches their `deferrables_schedule`/`predicted_temperatures` attributes into
`report_cache` (`_reporter_observe`, seeded once at startup by `_reporter_seed_cache`). `next_planned_record`
is called fresh on every publish tick, purely from that cache, so **the "next" row jitters with every
replan by design** — it's a live forecast, not a commitment, and re-derives from scratch each time
rather than tracking a previous prediction across replans.

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
  reporting:
    enabled: true                  # gates the cycle_reporter task
    cycles_entity: sensor.hwc_cycles
    db_path: data/hwc_cycles.sqlite # durable system-of-record (docs/hwc/local_store.md)
    sample_seconds: 30             # trace-sampler cadence + live-row refresh
    history_len: 20                # completed rows shown in the card
    min_cycle_seconds: 300         # discard sub-floor runs (defrost flicker)
    close_settle_seconds: 60       # wait after compressor-off before finalising (capture the
                                   # tank probe's post-off peak as tank_end)
    humidity_entity: weather.woodville_west_hourly   # humidity (attribute) for wet-bulb
    entities:                      # live HWC sensor set captured into the trace
      compressor: binary_sensor.aquatech_compressor  # drives the on/off cycle edges
      tank: sensor.aquatech_current_temperature_local
      power: sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_power_2
      energy: sensor.athom_energy_monitor_02a3c8_athom_energy_monitor_02a3c8_energy_2
      ambient: sensor.aquatech_temperature
      element: binary_sensor.aquatech_element
      defrost: binary_sensor.aquatech_defrost
      four_way: binary_sensor.aquatech_four_way_valve
      exhaust: sensor.aquatech_exhaust_temperature
      coil: sensor.aquatech_coil_temperature
      return_air: sensor.aquatech_return_air_temperature
      fan: sensor.aquatech_flow      # enum sensor, "Off"/"Low"/"High" fan speed -> bool
```

`fan` is the one non-numeric, non-"on"/"off" entity in the set: it's an enum sensor (`options: [Off,
Low, High]`, not a `binary_sensor`), so `_reporter_observe`/`_reporter_seed_cache` coerce it to a bool
case-insensitively (`raw.lower() == "high"`) rather than `_coerce_float`. Per-cycle it's
classified the same way as `element_on`/`defrost_on`/`four_way_on` — `fan_high_on` is true if the fan was
ever "high" at any point in the cycle (`cycle_metrics`' `_any_on("fan")`). The in-progress row instead
carries the raw current reading as `fan_high` (no over-the-cycle classification exists yet for an open
cycle). Tracked to let COP analysis be split by fan speed when evaluating the fan-speed threshold setting.

**2026-07-04 09:55**: owner reverted the unit's fan-speed thresholds from the quiet settings
documented in the manufacturer's manual (F30 25→10, F35 55→30) back towards factory defaults,
expecting "high" to trigger more often (including on the typical mid-day run) — the intent is to
observe the COP effect using `fan_high_on` on cycles from this point on. Cycles before this timestamp reflect the old (more conservative) threshold, so don't pool them
with post-change cycles when comparing COP by fan speed.

**2026-07-12 first read: fan-high is worth ~+0.15 COP (~7%).** Every daytime cycle since the
2026-07-04 change has run fan-high; every one before it fan-low, giving a clean before/after split
(9 fan-high vs 20 fan-low clean, heat-pump-only daytime cycles finishing at ~60 °C since fan
tracking began 2026-06-02). COP is dominated by starting tank temperature (≈ −0.053 COP per °C
warmer start), so the comparison must control for it: a regression of COP on tank_start + ambient +
fan gives a fan-high coefficient of **+0.15 COP** (t ≈ 2.6); matched tank-start bands agree
(51.5–54 °C top-ups: 1.96 → 2.14; 44–50 °C reheats: 2.24 → 2.53). Mean compressor-period power is
only ~12 W higher on fan-high, so the gain is real extra heat delivery, not power-accounting drift.
Caveats: only 9 fan-high cycles, all in similar winter conditions — re-run at ~20 fan-high cycles
and before drawing conclusions for milder ambients.

*2026-07-15 refresh (12 fan-high cycles):* estimate softened to **+0.11 COP (~5%)**, t ≈ 1.95 —
one weak Jul-15 cycle (52→60 at COP 1.86) accounts for most of the drop; fan-low has similar
stragglers, so it reads as normal scatter. Matched 51.5–54 °C band is now 7 vs 7: 1.96 → 2.11.
Direction unchanged in both bands; still borderline significance — recheck at ~20 fan-high cycles
(~2026-07-22). The 2026-07-14 negative-price 75 °C override doesn't contaminate this: all cycles
compared ended at ~60 °C. *Post-mortem:* the Jul-15 cycle turned out to be a measurement artifact,
not a bad run — probe lag 37 min (vs the 13–21 min norm) from a below-probe cold slug after a
morning draw; the lag-phase electricity earns no probe-delta credit, understating COP. Its
`band_cop` (below) reads an on-trend 2.93.

### `band_cop`: fixed 54→60 °C band efficiency index (2026-07-15)

Full-cycle COP charges *all* cycle electricity but credits only the mid-tank probe's ΔT, so a
post-draw cycle (long `probe_lag_min` heating the below-probe cold slug) reads spuriously low.
`band_cop` (`hwc_cop_analysis.band_cop_from_trace`, stored per cycle, computed inside
`cycle_metrics`) instead measures elec only while the probe first traverses a **fixed 54→60 °C
band**, crediting the actual probe delta across the crossings plus standing loss. Every routine
cycle passes through the band from below (top-ups start ≤ ~54.6 °C), and entry happens after the
warm-up transient (the probe-lag phase precedes the first rise). The top is the full 60 °C target:
the compressor genuinely runs until 60 — when a run *appears* to stop at 59.9, or the final tick
lands after the off-edge, that's Local Tuya polling the values in an arbitrary order, not an early
stop — so the traversal reads the settle-window samples (where the delayed tick lands) and accepts
a reading within one probe tick of the top (`BAND_TOP_TOLERANCE_C = 0.1`, which also admits the
InfluxDB-seeded traces, grid-interpolated to just under the peak). NaN when a cycle doesn't
traverse the band from below, dips mid-band (a draw hit the probe → not comparable), or has no
measurable elec. It is a **relative A/B index,
not a true COP**: concurrent below-probe warming is ignored, identically for every cycle. Note it
is *not* free of tank-state dependence — cycles that started colder show higher `band_cop`
(≈ −0.06/°C) because the condenser sees the whole tank, and a cooler below-probe mass means a lower
condensing temperature during the band; compare within tank-start strata (the dependence is
convex, so a pooled linear control underfits).

*First read (19 fan-low vs 12 fan-high):* matched strata put fan-high ahead on `band_cop` —
top-ups (51.5–54 °C start) 2.30 → 2.51 (**+0.21, ~9%**, 7 v 7); reheats (44–50.5 °C) 2.85 → 2.99
(+0.14, 10 v 4). Consistent with (and cleaner than) the full-cycle read; per-cycle scatter is
higher (sd ~0.3) so keep accumulating before treating the magnitude as settled.

**Backfilling `fan_high_on` for pre-existing cycles (2026-07-04, one-off).** `fan` wasn't tracked
before this feature landed, so every already-stored cycle had a null `fan_high_on`. Rather than leave
it null, each cycle's window was checked against HA's own recorder history for
`sensor.aquatech_flow`, its 30 s trace samples' `fan` column patched from that history (forward-filled
segments), and the row recomputed through `cycle_metrics` — so the backfill survives any future
recompute instead of being silently overwritten back to null. Two sources were used:

- **Live HA recorder** (`/api/history/period`) for cycles within its purge window (~10 days).
- **ZFS snapshots** of the recorder's `home-assistant_v2.db` for older cycles: the recorder purges
  each snapshot's own history to the same ~10-day window relative to *when the snapshot was taken*,
  not relative to now, so one snapshot only covers ~10 days around its own date — 3 overlapping
  daily snapshots (roughly a week apart) were needed to cover a 3-week gap. The db is WAL-mode and
  the snapshot mount is read-only, so it can't be queried in place (SQLite needs a writable `-shm`
  file to coordinate against the `-wal` file; the `immutable=1` read-only mode that avoids that
  requirement also skips WAL recovery, silently missing anything not yet checkpointed into the main
  file). Each snapshot's `.db`/`-wal`/`-shm` trio was copied to local scratch (~11GB, ~10s), queried,
  then deleted.
- 6 of the very earliest stored cycles (the original CSV-seeded anchors, pre-dating this SQLite
  store) have no `end_ts` and no trace to patch, so they're unrecoverable and stay null.

(`sensor.aquatech_inlet_temperature` is in the InfluxDB-seeded history but no longer exists as a
live HA entity, so it isn't captured going forward; the store column stays nullable.)

## What shipped

- **2026-06-27** (original): counter-aware `analyse`, an InfluxDB after-the-fact reporter (live
  state machine + per-cycle re-analyse + cold-start backfill + JSON ring).
- **2026-06-30** (rearchitecture, `docs/hwc/local_store.md`):
  - `hwc_cycle_store.py`: SQLite system-of-record (`hwc_cycles` summary + `hwc_cycle_samples` 30 s
    trace), WAL writer / read-only reader.
  - `hwc_cop_analysis.cycle_metrics`: the shared trace→summary brain; `analyse` repointed to read the
    store; the InfluxDB COP extraction moved to the throwaway `scripts/seed_hwc_store.py`.
  - `hwc_cycle_reporter.py`: pure `record_from_store_row` / `records_from_store_rows` / `live_record`
    / `build_payload` (the InfluxDB-era `advance_live` / `merge_records` / cold-start / cooldown
    helpers are gone).
  - `services/hwc_daemon.py`: websocket cache + compressor-edge open/close + 30 s sampler + SQLite
    writes; publishes the card from the store; `reporter_cycle` (edge snapshot) in the state file.
  - Tests: store round-trip, `cycle_metrics` paths, reporter projection/live-row, and a
    daemon-level open→sample→close→finalise flow.

**HA card:** rendered against `sensor.hwc_cycles` (`attributes.cycles` table + `attributes.current`
live row) — `hass/lovelace-hwc-cycles.yaml`.

## Relationship to auto-tuning

This is the **measurement/monitoring** half of the feedback loop discussed for the heat-rate and
compressor-power parameters (`config.yaml hwc.thermal.*`). It makes per-cycle drift visible without
closing an automatic tuning loop. A genuine auto-tune (periodic robust refit of the rate/power
slopes from accumulated clean cycles, written to an override the planner reads) is a separate,
later step and should land behind its own publish-only watch — not folded in here. See the
hand-anchored parameters in `config.yaml` and `docs/hwc/thermal_characterisation.md`.
