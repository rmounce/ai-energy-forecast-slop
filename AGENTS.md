# Agent Notes

## Collaboration

- Keep the repository state clean. Before finishing a task, check `git status` and either commit, ignore, or deliberately remove new files created during the work.
- Do not commit local scratch/question files unless the user explicitly asks for them to become durable project documentation.
- When the user suggests that the agent should do something differently, offer to update this `AGENTS.md` file so the preference is remembered for future sessions.
- Commit changes regularly, but not excessively.

## Project Documentation

- Be aware of `docs/`, `README.md`, and `ARCHITECTURE.md`, updating them as you go when changes affect documented behaviour.
- For repository material likely to be referenced often, apply "Caveman compression": add or maintain a concise, low-prose summary with short bullets, concrete facts, decisions, commands, file paths, and current status. Prefer this as a companion summary or top section rather than deleting useful detail from the source document.
- When updating frequently referenced docs, keep the compressed summary current in the same change.

## Plans And Memory

- At the start of each session, if a plan file exists alongside memory files, explicitly check them for contradictions before acting.
- Memory takes precedence over plan files when they conflict. Update the plan file immediately to reflect the current decision.
- When the user overrides or rejects a planned approach mid-session, update the plan file in that same response. Do not defer it.

## Infrastructure Notes

- InfluxDB data directory: `/opt/dockerfiles/influxdb/` (requires sudo to inspect).
- InfluxDB runs as a Docker container; config is in `/opt/dockerfiles/`.

## Python Environment

- Use `uv` for package management rather than `pip`.
- The venv was created with `uv`; `.venv/pyvenv.cfg` contains `uv = ...`.
- `README.md` may still mention `pip`; prefer `uv pip` in practice.
- Install with CPU-only torch to avoid large CUDA wheels:

```bash
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
