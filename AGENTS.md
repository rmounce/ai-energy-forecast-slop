# Agent Notes

## Collaboration

- Keep worktree clean: check `git status` before finishing.
- Own your files: commit, ignore, or remove generated files.
- Inherited dirty/untracked state: ask before substantial work.
- Scratch/question files: do not commit unless explicitly requested.
- User preference change: offer to update `AGENTS.md`.
- Commit regularly; avoid noisy commits.
- Rename/refactor own code: migrate all call sites; no back-compat aliases. Don't accrue self-made tech debt.

## Project Documentation

- Update `docs/`, `README.md`, and `ARCHITECTURE.md` when needed to explain durable
  behaviour or correct misleading guidance. Not every change needs a documentation update.
- Public documentation describes software behaviour and reusable setup instructions.
  Keep private operational records, audit findings, credential administration, sensitive incident
  details and private recovery records in local-only notes unless publication is explicitly requested.
- Do not add personal OS usernames, machine hostnames or private infrastructure domains
  to tracked content. Prefer relative paths, `$HOME`, systemd `%h` or neutral examples.
- Caveman compression: short bullets, concrete facts, decisions, commands, paths, status.
- Maintain frequently referenced summaries when substantive behaviour changes.

## External Systems

- Record confirmed black-box behaviour when useful for future work; choose public
  documentation or private local notes according to the information involved.
- Cover device/API quirks, HA entity lifecycle, mode/setpoint semantics, operational limits.
- Record concrete facts: date/context, command/service, observed state, remaining uncertainty.
- Home Assistant: prefer domain hot reloads through the HA API whenever supported; restart
  HA only when the changed configuration cannot be reloaded. Follow `docs/ha_hot_reload.md`.

## Monitoring

- Alert on structural failures, stale data, or sustained loss of usable data.
- Keep transient forecast movements and threshold crossings as diagnostics; require persistence
  before paging on volatile signals.

## Plans And Memory

- Session start: check plan files against memory files.
- Conflict: memory wins.
- Update active plans when decisions change; keep routine session bookkeeping local.

## Infrastructure Notes

- InfluxDB data: `/opt/dockerfiles/influxdb/` (sudo required).
- InfluxDB Docker/config: `/opt/dockerfiles/`.

## Python Environment

- Package manager: `uv`, not raw `pip`.
- Venv: uv-created; see `.venv/pyvenv.cfg`.
- README may say `pip`; prefer `uv pip`.
- CPU-only torch install:

```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
