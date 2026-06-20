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

- Keep `docs/`, `README.md`, `ARCHITECTURE.md` current.
- Behaviour change affecting docs: update docs in same work.
- Caveman compression: short bullets, concrete facts, decisions, commands, paths, status.
- Frequently referenced docs: keep compressed summary current.

## External Systems

- Document confirmed black-box behaviour promptly.
- Cover device/API quirks, HA entity lifecycle, mode/setpoint semantics, operational limits.
- Record concrete facts: date/context, command/service, observed state, remaining uncertainty.

## Plans And Memory

- Session start: check plan files against memory files.
- Conflict: memory wins.
- Plan changed or rejected: update plan file immediately.

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
