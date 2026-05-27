# Event-driven `predict-price` refresh — design plan

Status: **draft, awaiting sign-off**
Date: 2026-05-27

## Motivation

During volatile periods (e.g. 2026-05-26) the strategic 30m/72h curve
served by `forecast.py predict-price` can lag the live MPC tier by up to
~30 minutes, because the timer that drives it (`ai-energy-predict.timer`)
fires only at `:01,:31`. The MPC tier sees fresh Amber APF each minute;
the strategic tier doesn't see it until the next 30-minute boundary.

Amber's APF itself refreshes every 5 minutes (NEM dispatch cadence), and
Home Assistant's Amber integration republishes the entity
`sensor.amber_billing_interval_forecasts_general_price` immediately on
each refresh. The proposal is to **trigger `predict-price` on that HA
state change** instead of polling on a fixed schedule.

## Approach: long-lived listener daemon

A single Python process subscribes to HA's WebSocket API and listens for
state changes on the Amber APF entity. On change (debounced), it
invokes `./forecast.py predict-price --dynamic-handoff --publish-hass`
as a subprocess. A 30-minute idle heartbeat re-runs the prediction even
if no event has fired (defensive: should never fire in normal
operation).

This replaces the price path of `ai-energy-predict.timer`. The timer
itself is retained at 30-min cadence but trimmed to load-only.

### Why a daemon, not a 1-min poll timer?

- Lower latency: seconds, not 1 min, between APF publish and
  republish.
- Fewer systemd units, not more: one new `Type=simple` service replaces
  zero things in the unit count (the existing timer stays, just trimmed
  to load).
- Cleanest mental model: "APF changed → strategic curve refreshes".

## Component breakdown

### `services/ha_listener.py` (new file, ~180 lines)

Async Python using the `websockets` package (new dependency, ~250 kB).

Responsibilities:

1. **Auth**: connect to `ws://<ha_host>:8123/api/websocket`; send
   long-lived token from `config.secrets.json` (same token used by the
   existing `get_entity_state()` REST calls).
2. **Subscribe**: `subscribe_events` with `event_type=state_changed`,
   filtered application-side to
   `entity_id == sensor.amber_billing_interval_forecasts_general_price`.
   (HA's subscribe_events doesn't accept an entity_id filter directly;
   we receive all state_changed events and ignore non-matches. Cost
   is trivial.)
3. **Debounce**: on matching event, set a "pending" flag and wait
   `DEBOUNCE_SECONDS` (default **1s**) before firing. If another
   matching event arrives during the wait, the timer is *not* reset —
   we fire after the original window so any near-simultaneous updates
   to the APF entity (which should arrive within milliseconds of each
   other) coalesce into one run. Tune up if we observe split runs.
4. **Subprocess**: shell out to
   `./forecast.py predict-price --dynamic-handoff --publish-hass` (via
   `asyncio.create_subprocess_exec`). Stream stdout/stderr through to
   journald so `journalctl --user -u ai-energy-listener -f` shows
   forecast progress live.
5. **Heartbeat**: a separate asyncio task sleeps `HEARTBEAT_SECONDS`
   (default **1800s = 30 min**); if no event-driven run has happened
   within that window, fires anyway. Resets on every successful run
   (event-driven or heartbeat-driven).
6. **Healthcheck ping**: on success, `requests.get(HC_PREDICT_URL,
   timeout=5)` — same env var the existing service uses.
7. **Reconnect**: on WebSocket close / network error, exponential
   backoff (1, 2, 4, 8, 16, 30s cap) and retry indefinitely. Log each
   reconnect.
8. **Concurrency guard**: an `asyncio.Lock` prevents overlapping
   `predict-price` runs. If a new event arrives mid-run, it's queued
   (deduplicated to one pending run).

No external state files. Restarts are clean — first successful
WebSocket subscription is the implicit "ready" state.

### `systemd/ai-energy-listener.service` (new file)

```ini
[Unit]
Description=Event-driven predict-price refresh on Amber APF state change
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/saltspork/src/ai-energy-forecast-slop
EnvironmentFile=/home/saltspork/src/ai-energy-forecast-slop/.env
ExecStart=/bin/bash -c 'source .venv/bin/activate && exec python services/ha_listener.py'
Restart=on-failure
RestartSec=10
StartLimitBurst=5
StartLimitIntervalSec=300

[Install]
WantedBy=default.target
```

No timer file — the service is a continuously-running daemon.

### `systemd/ai-energy-predict.service` (modified)

Change `ExecStart` to use `predict-load` instead of `predict-all`:

```ini
ExecStart=/bin/bash -c 'source .venv/bin/activate && ./forecast.py predict-load --publish-hass --publish-covariates && curl -fsS --retry 3 ${HC_PREDICT_URL}'
```

Timer cadence unchanged (`*:01,31:00`). The price path is fully owned
by the listener daemon now.

### `requirements.txt`

Add `websockets` (single line). No version pin — pull whatever uv
resolves. Sync-call to `requests` for the healthcheck reuses the
existing dep.

## Behaviour table

| Trigger | What runs | Cadence |
|---|---|---|
| HA APF state_changed (debounced 1s) | `predict-price` | ~every 5 min in normal operation |
| Listener 30-min idle heartbeat | `predict-price` | only if no event in 30 min |
| `ai-energy-predict.timer` at `:01,:31` | `predict-load` (+ covariates) | every 30 min |
| `ai-energy-p5min.timer` at `:02,:07,…` | `publish-tactical` | every 5 min (unchanged) |
| `ai-energy-predispatch.timer` at `:12,:42` | ingest + `publish-pd-direct` | every 30 min (unchanged) |

## Failure modes and mitigations

| Failure | What happens | Mitigation |
|---|---|---|
| Daemon crashes | systemd `Restart=on-failure` | `StartLimitBurst=5` prevents thrash |
| HA restart | WebSocket closes; daemon reconnects with backoff | Exponential backoff up to 30s |
| HA token expires/revoked | Auth failures repeatedly | Logged; user notices via healthcheck miss |
| Listener silently wedged (connected but receiving no events) | No predict-price runs at all | 30-min heartbeat covers this — daemon fires anyway |
| Subprocess hangs | Blocks the lock, no further runs | 2-min subprocess timeout (`asyncio.wait_for`) — kill + log on timeout. Tune after observing real run times in journald. |
| Healthcheck endpoint down | Forecast still publishes; HC misses | Failure here is non-fatal; daemon logs and continues |

The daemon being functionally degraded but not dead (e.g. socket
connected but no events delivered) is the most insidious case. The
30-min heartbeat is the explicit defence — at worst we degrade to the
pre-existing 30-min cadence.

## Migration path

1. Land daemon + new unit, **disabled**. Leave existing
   `ai-energy-predict.timer` untouched.
2. Start daemon manually, watch for 1 hour: confirm reconnect on HA
   restart, confirm debounce works, confirm subprocess streams to
   journal.
3. Enable + start the systemd unit. Trim `ai-energy-predict.service` to
   `predict-load`.
4. Monitor for 24-48 hours via the healthcheck dashboard and journald.
5. If anything goes wrong: stop the daemon, revert the service to
   `predict-all`. The rollback is one `git revert` + `systemctl daemon-reload`.

## Out of scope (deliberately)

- **Generalisation** to "subscribe to N entities → run M commands".
  YAGNI — one entity, one command is enough until a second use-case
  emerges.
- **Load-side event-driven refresh**. Load is driven by slower-moving
  signals (temperature forecast, day-of-week) — 30-min cadence is fine.
- **Covariate republishing on APF change**. Covariates aren't price-side;
  the existing 30-min predict-load run still publishes them.
- **EMHASS / dispatch trigger changes**. The HA template already
  handles sub-30min DH re-runs via SoC snapshot alignment per the
  user's note. No change there.

## Open questions

None blocking. Decisions baked in above:
- Debounce: 1s (APF entities update near-simultaneously)
- Heartbeat: 30 min
- Subprocess timeout: 2 min (revisit after first day of logs)
- Backoff cap: 30s
- WebSocket library: `websockets` (not `aiohttp`)

Happy to flip any of these in review.
