# Event-driven `predict-price` refresh

**Status:** implemented 2026-05-27; current behaviour checked 2026-08-10.

## Current Contract

- `services/ha_listener.py` subscribes to HA `state_changed` events over WebSocket.
- It filters for the configured Amber billing-interval APF entity.
- A matching event is debounced for 1 second.
- It runs:

```bash
./.venv/bin/python forecast.py predict-price --dynamic-handoff --publish-hass
```

- A 30-minute idle heartbeat triggers the same command if no run occurred.
- Runs are serialized; subprocess timeout is 180 seconds (changed 2026-09-27).
- WebSocket reconnect uses exponential backoff capped at 30 seconds.
- `systemd/ai-energy-predict.timer` is load-only and still runs at `:01,:31`.

The critical path is documented in
[../prod_pipeline_critical_path.md](../prod_pipeline_critical_path.md).

## Service

Tracked unit: `systemd/ai-energy-listener.service`.

```bash
systemctl --user enable --now ai-energy-listener.service
journalctl --user -u ai-energy-listener.service -f
```

Rollback to timer-driven price publication requires an explicit unit/code change; the current
`ai-energy-predict.service` invokes `predict-load`, not `predict-all`.

## Current Failure Semantics

- Listener process failure: systemd restarts it.
- HA/WebSocket failure: listener reconnects; heartbeat still runs inside the same process.
- Prediction timeout: child is killed and logged.
- Nonzero prediction exit: no healthcheck ping; a retry is scheduled on a five-minute cadence.
- `last_run_at` advances only after validated generation and successful publication.
- APF events received while a child runs coalesce into one follow-up run.
- Healthcheck failure: logged; forecast run remains successful.

Prediction validation now rejects missing/stale APF, incomplete or malformed quantile families,
and failed HA writes before the listener reports success.

## Non-Goals

- Do not event-drive load without a demonstrated freshness need.
- Do not revive archived tactical, PD-direct, TFT, or canonical AI publishers as part of listener
  hardening.
- Do not change EMHASS source selection in this track.

## Resident migration

Opt-in calculation-only successor: [resident price shadow](../energy_pipeline_resident_price.md).
Current production unit still runs the subprocess listener above. Model reuse passes exact parity;
independent source refreshes and memory reclamation now work in shadow. Current evidence and
remaining cutover gates: [checkpoint](../energy_pipeline_source_cache.md).
