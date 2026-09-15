# Home Assistant hot reload

## Supported path

- Credentials: `config_utils.load_config()` deep-merges ignored `config.secrets.yaml` into
  `config.yaml`. `HA_TOKEN` can override the merged token. Never print or commit the token.
- Package change: sync YAML, validate, then reload every changed domain through the HA API.
  This installation exposes `rest.reload`, `input_datetime.reload`, `input_number.reload`,
  `template.reload`, `script.reload`, and `automation.reload`.
- Reload dependency providers before consumers. For `hass/packages/emhass.yaml`: helpers →
  templates → REST commands → scripts → automations.
- Storage-mode Lovelace change: use WebSocket `lovelace/config/save`; do not edit `.storage`
  while HA runs.
- Existing helper: `scripts/ha_hot_reload.py` currently covers templates and storage-mode
  Lovelace only; use the HA service API for the other reloadable domains.

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

## Restart recovery

- Avoid restarting HA for reloadable package changes. A restart removes EMHASS-published
  `dh_*` and `mpc_*` REST-created entities until their producer pipelines run again.
- Do not optimize while required forecast entities are absent. EMHASS can silently fall back
  to its configured internal forecast method instead of using the intended runtime arrays.
- Recovery order after an unavoidable HA restart:
  1. `systemctl --user start ai-energy-predict.service` — republishes the 72h load family.
  2. `systemctl --user restart ai-energy-listener.service` — reconnects the HA WebSocket
     listener and republishes price on its normal trigger/heartbeat path.
  3. Verify `sensor.ai_price_forecast` and `sensor.ai_load_forecast_high` exist.
  4. Let the forecast state changes trigger DH, then let the normal MPC trigger run.
  5. Verify `sensor.dh_optim_status` and `sensor.mpc_optim_status` are `Optimal`.
- Do not rely on `rest_command.emhass_publish_data_all` as restart recovery on EMHASS 0.17.9.
  See `docs/emhass_shared_state_race.md` for the confirmed mixed-resolution failure.

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

## 2026-09-15 status

- Confirmed the reload services listed above through HA `/api/services`.
- An unnecessary HA restart cleared the REST-created EMHASS plan entities.
- Retriggering `ai-energy-predict.service` restored 144-point load forecasts; the normal HA
  chain then produced fresh, optimal DH and MPC plans without another HA restart.
