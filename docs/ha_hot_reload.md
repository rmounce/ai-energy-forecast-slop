# Home Assistant hot reload

## Supported path

- Credentials: `config.yaml` → `home_assistant.url` / `token`, or the `HA_TOKEN` environment
  variable. `HA_TOKEN` takes precedence. Never print or commit the token.
- Template package change: sync the YAML into HA, validate config, call `template.reload`.
- Storage-mode Lovelace change: use WebSocket `lovelace/config/save`; do not edit `.storage`
  while HA runs.
- Script: `scripts/ha_hot_reload.py`.

## Commands

Reload all YAML template entities without restarting HA:

```bash
./.venv/bin/python scripts/ha_hot_reload.py --reload-templates
```

Save a storage-mode dashboard from an HA storage wrapper and reload templates in one call:

```bash
./.venv/bin/python scripts/ha_hot_reload.py \
  --reload-templates \
  --lovelace-storage /tmp/lovelace.dashboard_battery.updated \
  --dashboard-url-path dashboard-battery \
  --dashboard-backup /tmp/dashboard-battery-before.json
```

The dashboard command first fetches the live config through WebSocket, writes the backup,
saves `data.config` from the supplied storage wrapper, then fetches and compares the result.
Treat it as a whole-dashboard replacement: inspect the diff immediately beforehand to avoid
overwriting concurrent UI edits.

## Validation

```bash
sudo docker exec hass python -m homeassistant --script check_config -c /config
```

Then verify the expected entity through REST or HA Developer Tools. A template reload creates
new state-based template entities and updates existing ones. Other non-reloadable package
domains still require a restart when their loaded configuration actually changes.

## 2026-08-20 status

- HA's configuration check passed with the adjusted-confirmed feed-in template.
- Live API reload was blocked because `config.yaml` had no HA token. Supply a current long-lived
  access token through `HA_TOKEN`, then rerun the combined command above.
- Expected API calls: `POST /api/services/template/reload`, then WebSocket
  `lovelace/config/save` with `url_path: dashboard-battery`.
- Direct `.storage` replacement alone was insufficient because HA retains dashboard state in
  memory. Use the WebSocket save path for a running instance.
