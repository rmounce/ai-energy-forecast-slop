# Resident price worker: shadow implementation

Status: opt-in calculation-only runner; not installed/enabled; production listener unchanged.
Plan: [pipeline consolidation](energy_pipeline_architecture_plan.md).

## Contract

- `services/resident_price.py`: existing HA WebSocket listener ingress/debounce/retry policy;
  one dedicated executor thread; bounded pending trigger, no overlapping inference.
- Startup/reconnect triggers reconciliation from current HA state; no replay of old event payloads.
- `energy_pipeline/price_worker.py`: reload config/tariffs each run; one HA states response;
  retain only configured APF/Solcast inputs; all quantiles consume the same frozen APF.
- `forecast.prediction_resources`: worker-local inputs; missing entity cannot fetch newer live state.
  Legacy forecast globals still require single-thread serialization; no concurrent family inference.
- `energy_pipeline/model_cache.py`: load one complete quantile family; reuse unchanged artifacts;
  stage replacements on active-bundle/path/inode/size/mtime changes; failed load leaves prior cache
  intact but fails the requested run, never silently uses the wrong bundle.
- `forecast.run_predictions(..., calculation_only=True)`: validate full quantile family; return
  `PredictionOutcome`; skip HA publication, spot-history capture, prediction JSON and forecast CSV.
- Weather `get_forecasts` request reads forecast data; AEMO/history acquisition still runs per job.
- Completion records run ID, APF snapshot digest, capture time, model bundle, point counts,
  load time, total elapsed time and RSS. APF digest includes HA entity metadata; not an upstream
  atomic revision or a complete lineage identity for weather/AEMO/history.
- Shadow does not overwrite production health records. Errors retry on existing five-minute cadence;
  successful shadow generation advances only its own heartbeat.
- Worker deadline 180s: discard result, stop scheduler, exit process with failure for supervision.
  Async cancellation cannot kill a thread; never submit overlapping work after timeout.
- RSS >2048 MiB after a run: reject completion and exit with failure. Guard is measured after
  inference, not a hard OS peak-memory limit or a fix for memory growth.
- Shutdown cancels idle ingress/waiters and discards pending completion; CLI exits the process
  explicitly after flushing logs so Python cannot hang joining a stuck executor thread.
- No output/control publication option; production cutover needs separate acceptance/publication work.

## Run

From repo root, existing uv-created venv; HA/AEMO/Influx access required.

```bash
# Normal resident shadow (Ctrl-C stops it)
.venv/bin/python services/resident_price.py
# Finite cold/warm benchmark
.venv/bin/python services/resident_price.py --runs 4
# Compare cached/reloaded quantiles on identical frozen APF/history/covariates
.venv/bin/python services/resident_price.py --runs 2 --verify-reload
```

`--verify-reload` adds fresh deserialization/inference work; exclude it from latency comparisons.
It requires exact DataFrame equality before monotonic quantile rearrangement. Diagnostic stdout/
stderr only; household forecast files untouched. No systemd unit or enabled shadow daemon added.

## Evidence: 2026-10-01 Adelaide

- 55 focused tests: payload policy, cache reuse/replacement/failure, input isolation/no writes,
  worker-thread completion, event bursts, deadline discard, memory guard, reconnect reconciliation,
  idle shutdown; incumbent listener/contract/model-bundle regressions included.
- Live calculation-only benchmark: all three quantiles have 144 points; three model loads total
  across repeated runs. Cold family deserialization ~2.8–3.0s; warm family load 0s.
- Cached/reloaded exact equality for p30/p50/p70 on both cold and warm runs.
- First benchmark: 81.5s cold / 31.3s warm; repeated AEMO timeouts dominated cold run.
- Separate four-run check (no reference reload): 25.7 / 20.5 / 30.6 / 19.0s.
  Source conditions differ; these are observations, not a controlled speedup estimate.
- RSS after those runs: 1468 / 1725 / 1781 / 1847 MiB. Growth not yet bounded; no production
  residency approval implied. Parity runs reached ~1878 MiB with fresh reference models too.

## Next gates

- Profile repeated-run allocations/model mutation; distinguish retained objects from allocator
  high-water marks; reduce heavy runtime overhead where possible; demonstrate bounded RSS.
- Independent AEMO/weather/history refreshes with validated coverage, age budgets, provenance and
  last usable cache; APF update should not wait on a full upstream refresh every time.
- Shadow live event/trigger decisions over representative market and recovery conditions; collect
  independent daytime/curtailment/restart payload snapshots alongside the worker.
- Add publication acceptance and complete input lineage; reject obsolete/expired results before
  publishing; preserve current HA sensor contracts; only then replace the subprocess publisher.
- Later: shadow DH/MPC coordination and move solve ownership after its separate cutover gates.
