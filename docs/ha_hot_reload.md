# Home Assistant hot reload

## Supported path

- Credentials: `config_utils.load_config()` deep-merges ignored `config.secrets.yaml` into
  `config.yaml`. `HA_TOKEN` can override the merged token. Never print or commit the token.
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
- The helper uses the same `config.yaml` + ignored `config.secrets.yaml` merge as production.
- Confirmed: `POST /api/services/template/reload` created the new template entity without a
  restart.
- Confirmed: WebSocket `lovelace/config/save` with `url_path: dashboard-battery` updated the
  live dashboard and passed the helper's read-after-write comparison.
- Live values after reload: raw feed-in `0.0278`, adjusted-confirmed `0.0378`, effective
  `0.0378`; adjustment attribute `sapn_free_export_allowance: +0.01`.
- Direct `.storage` replacement alone was insufficient because HA retains dashboard state in
  memory. Use the WebSocket save path for a running instance.
