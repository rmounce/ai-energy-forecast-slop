# HWC local store — drop InfluxDB for heat-pump hot-water cycle data

Status: **design agreed 2026-06-30, not yet implemented.** Spec for moving all HWC per-cycle data
(live "recent runs" reporting *and* offline COP characterisation) off InfluxDB onto a local SQLite
store owned by the HWC daemon.

## Why

InfluxDB is **not** the system-of-record for HWC, and quietly cannot be:

- Aquatech/Athom entities (tank `heat_pump_temperature`, `aquatech_compressor`, `power_2`,
  `energy_2`, element/defrost, exhaust/coil/return/inlet) land in the default RP **`rp_raw`, which
  retains only 720 h = 30 days**.
- The `rp_5m` (3 yr) / `rp_30m` (10 yr) downsampling is driven by *entity-specific* continuous
  queries covering only the forecasting signals (Sigen consumed power, PV, AEMO prices, Adelaide
  weather, dump/deferrable load). **No HWC entity is downsampled** — for HWC it is raw-or-nothing.
- So per-cycle history is a rolling 30-day window. Verified 2026-06-30: earliest retained tank /
  compressor data is **2026-05-31**; the install-period cycles (2026-05-28/30) in
  `docs/hwc/thermal_characterisation.md` are already gone from InfluxDB and survive only because
  they were snapshotted to `data/hwc_cop_cycles.csv`.

Reconstructing cycles after the fact from a 30-day-rolling time-series DB was never the right shape
for live telemetry and is unusable for long-term characterisation. This session's bugs (blank card
on a probe gap, today's run mis-flagged unclean, duplicate rows, the whole event-driven-finalise /
cold-start machinery existing only to dodge a 3-minute query) all trace to it.

InfluxDB stays exactly as-is for the **forecasting** workstream (load/price models, weather,
prices); this change touches HWC only.

## Decisions (locked)

1. **SQLite is the durable HWC system-of-record** — all cycles, forever, in-repo runtime data.
2. **Per-cycle trace table at 30 s resolution** — so every diagnostic `analyse` computes from raw
   traces survives past 30 days and can be recomputed if methodology changes. **Capture the full
   HWC sensor set** (tank, power, energy, ambient, element, defrost, four-way, exhaust, coil,
   return-air, inlet): when in doubt, gather everything — storage is trivial and we cannot recover
   a signal we didn't record once `rp_raw` ages out. Columns are nullable so a temporarily
   unavailable sensor never blocks a row.
3. **Precise edge snapshots** — `energy_2` and `tank` captured at the compressor on/off
   state-change (not a 60 s poll), removing the boundary lag discussed in
   [[project_heat_pump_hot_water]].
4. **`hwc_cop_analysis` is repointed to read SQLite** (no InfluxDB code left in the live tool).
5. **Forward-only** — cycles completing during daemon downtime are simply absent; acceptable, since
   InfluxDB was never a real safety net past 30 days. No steady-state InfluxDB backfill.

## Architecture

One pure function, two feeders — no divergence possible:

```
cycle_metrics(trace, edges) -> summary row     # thermal, counter elec, COP, cycle_is_clean,
                                               # probe-rise/p95 diagnostics — the shared brain
  ├─ daemon (live):  accumulates trace + edge snapshots, calls cycle_metrics, writes SQLite
  └─ hwc_cop_analysis (offline): loads traces from SQLite, calls the SAME cycle_metrics
```

`cycle_metrics` reuses the already-pure `cycle_is_clean` and the thermal/counter math (extract the
inline thermal formula from `analyse` into a shared helper). `analyse` keeps producing every column
it does today; it just sources rows from SQLite instead of InfluxDB.

### Store

`data/hwc_cycles.sqlite` (WAL, daemon is sole writer; analysis scripts open read-only). Gitignored
runtime data, like `data/hwc_daemon_state.json`.

**`hwc_cycles`** — one summary row per cycle:

```
start_ts REAL PK, start_local TEXT, end_ts REAL, dur_min INT,
tank_start REAL, tank_end REAL, ambient REAL, wet_bulb REAL,
elec_kwh REAL, elec_source TEXT, therm_kwh REAL, cop REAL NULL,
hp_mean_w REAL, hp_p95_w REAL,
probe_lag_min REAL NULL, probe_rise_10_min REAL NULL, probe_rise_50_min REAL NULL,
probe_rise_90_min REAL NULL,
exhaust_start REAL NULL, exhaust_max REAL NULL, exhaust_end REAL NULL,
coil_mean REAL NULL, return_air_mean REAL NULL, inlet_mean REAL NULL,
element_on INT, defrost_on INT, four_way_on INT, fan_high_on INT,
clean INT, status TEXT,            -- 'running' | 'complete'
updated_at REAL
```

The per-run thermal extras (exhaust/coil/return-air/inlet stats) are cached in the summary —
cheap at one row per run, and it spares every reader from re-deriving them off the trace — while
the raw signals still live in `hwc_cycle_samples` for any recompute. There is **no `baseline_w`**:
the dedicated meter makes the off-state baseline single-digit watts, so `cycle_metrics` drops the
baseline subtraction entirely (`hp_mean/p95` are raw cycle power; the `power_integration` fallback
integrates `power_w` directly; the baseline-drift term in `cycle_is_clean` goes inert).

The 20-row card ring is `SELECT … ORDER BY start_ts DESC LIMIT N`. The in-progress row is the
`status='running'` row (replaces the JSON ring in the state file and the `current_cycle` field).

**`hwc_cycle_samples`** — compact per-cycle trace (~60–240 rows/cycle, a few KB):

```
cycle_start_ts REAL,           -- FK -> hwc_cycles.start_ts
ts REAL,                       -- 30 s grid, epoch UTC
tank REAL, power_w REAL, energy_kwh REAL, ambient REAL, humidity REAL,
element INT, defrost INT, four_way INT, fan INT,
exhaust REAL NULL, coil REAL NULL, return_air REAL NULL, inlet REAL NULL,
PRIMARY KEY (cycle_start_ts, ts)
```

`four_way` is in the full sensor set (decision #2); `humidity` is stored alongside `ambient` so
`wet_bulb` is recomputable from the trace offline (not just a daemon-time scalar) — every column of
the `hwc_cycles` summary can then be regenerated from raw by `cycle_metrics`.

### Capture mechanism

- The daemon **subscribes** (websocket) to the HWC entity set and caches latest values; the trace
  sampler reads the **cache** every 30 s (near-zero cost, no per-tick REST storm). Edge snapshots
  read the same cache at the compressor state-change instant.
- **on-edge** (compressor off→on): open a `running` cycle; snapshot the edge tank, `energy_start`,
  `ts`. **off-edge**: snapshot the edge tank, `energy_end`, `ts`; finalise. The stored
  `tank_start`/`tank_end` are the min/max tank reading over the cycle trace (not the edge
  snapshots — see docs/hwc/cycle_reporting.md), since stratification can dip the probe below its
  on-edge reading well into a run.
- `elec_kwh = energy_end − energy_start` (counter), `power_integration` fallback on a counter
  reset/gap (computed from the trace `power_w`).
- Reconnect handling: a missed websocket reconnect just leaves a gap in the 30 s trace (and at
  worst drops a cycle); the control loop is untouched. Reporter stays publish-only / firewalled
  from `run_lock` and the command path.

## Migration / seeding (one-time, then InfluxDB is never read for HWC again)

A **separate throwaway seed script** (not part of the live tool):

1. Backfill `hwc_cycles` from `data/hwc_cop_cycles.csv` (pre-30-day history; summary only — no
   trace recoverable for aged cycles).
2. For the last ≤30 days still in `rp_raw`, run the existing InfluxDB `analyse` once to fill
   `hwc_cycles` **and** reconstruct 30 s traces into `hwc_cycle_samples` (so we don't start with an
   empty trace table).
3. After seeding, the daemon records forward and `hwc_cop_analysis.analyse` reads the store.

   *Scope note (decided in implementation):* the InfluxDB **access helpers** in `hwc_cop_analysis`
   (`_client`, `_series`, `_series_anchored`, `_interp_to_idx`, `_state_to_idx`, the entity
   constants) are **kept**, not deleted — `hwc_validate_cycle_traces.py` and `hwc_soc_extract.py`
   (separate offline thermal-characterisation tools, whose migration is deferred below) still read
   InfluxDB through them, and the seed reuses them. Only `analyse` itself is repointed to SQLite;
   the live per-cycle reporting path no longer touches InfluxDB, which is the actual goal.

## Work breakdown (post-compaction implementation)

- **`hwc_cycle_store.py`** (new): SQLite schema + read/write helpers (upsert cycle, append samples,
  recent-N, load-trace). Pure-ish, unit-testable with a temp DB.
- **`cycle_metrics`** (new shared fn, likely in `hwc_cop_analysis`): `trace -> summary`; extract the
  thermal/counter/diagnostic math out of `analyse`'s loop; reuse `cycle_is_clean`.
- **`services/hwc_daemon.py`**: replace the InfluxDB `analyse`-based reporter (cold-start +
  per-cycle finalise + JSON ring) with: subscribe/cache, 30 s trace sampler, edge snapshots,
  `cycle_metrics` on off-edge, SQLite writes, publish `sensor.hwc_cycles` from a `SELECT`.
- **`hwc_cop_analysis.py`**: repoint `analyse` to SQLite (load traces → `cycle_metrics`, stored
  summary as-is for trace-less CSV anchors). **Keep** the InfluxDB access helpers (shared by the
  seed + `hwc_validate_cycle_traces` + `hwc_soc_extract`; see scope note above). Keep CSV/markdown
  output.
- **`hwc_cycle_reporter.py`**: retire the InfluxDB-era helpers (`cold_start_since_ts`,
  `cooldown_settled/expired`, `backfill_captured`, `merge_records`); keep/adapt `advance_live`,
  `live_view`, `build_payload`.
- **`config.yaml` `hwc.reporting`**: drop `seed_lookback_hours` / `incremental_margin_hours` /
  `settle_seconds` / `finalize_*`; add `db_path`, `sample_seconds: 30`, HWC subscription entity
  list.
- **Seed script** (throwaway): CSV + final 30-day InfluxDB read.
- **Tests**: store round-trip, `cycle_metrics` parity vs a known InfluxDB-derived row, live state
  machine, edge-snapshot elec.

## Rollout

Shadow-run the recorder alongside the current path for a few days; diff its cycles against
`analyse`'s on COP/clean; then flip `sensor.hwc_cycles` to the SQLite source and remove the old
path.

## Open / deferred

- Eventual: point `docs/hwc/thermal_characterisation.md` refits at the SQLite store.
