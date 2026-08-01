# HWC efficiency dashboard

First cut: a read-only `aiohttp` service serves `web/hwc/index.html` and JSON from the durable
SQLite cycle store. The HWC daemon remains the only writer and is not coupled to dashboard
availability.

## Local run

```bash
./.venv/bin/python services/hwc_web.py
```

Defaults:

- bind: `127.0.0.1:8765`
- database: `data/hwc_cycles.sqlite`
- static root: `web/hwc/`

Override with `--host`, `--port`, `--db`, or `--web-root`. Endpoints:

- `/api/health`
- `/api/cycles?limit=10000`
- `/api/cycles/{start_ts}/trace`

The frontend polls the cycle list once per minute. It currently plots COP or other cycle metrics
against wet bulb, ambient, or tank-start temperature, with filters for clean cycles, target
completion, date range, and fan regime.

## systemd and Traefik

Install `systemd/ai-energy-hwc-web.service` as a user service after verifying the local server.
The service binds localhost; Traefik can reverse-proxy it using a file-provider service pointing
to `http://127.0.0.1:8765`. Add the internal and Authelia-protected external routers to
`/opt/dockerfiles/traefik/config/dynamic.yml` as a host-level deployment step. This repository
does not modify that host configuration automatically.

## Future HWC split

This dashboard is intentionally HWC-local: it reads the HWC store and has no dependency on the
forecasting or battery-control code. When HWC moves to its own repository, move this service,
`web/hwc/`, this document, and the HWC systemd unit together. Keep the API boundary (`cycles`,
`trace`, and read-only SQLite ownership) stable during that migration.
