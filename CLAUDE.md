- Be aware of `README.md` and `ARCHITECTURE.md`, updating them when making more substantial changes.
- Commit changes regularly (but not excessively), giving yourself credit.
- At the start of each session, if a plan file exists alongside memory files, explicitly check them for contradictions before acting. Memory takes precedence over plan files when they conflict — and update the plan file immediately to reflect the current decision.
- When the user overrides or rejects a planned approach mid-session, update the plan file in that same response. Do not defer it.

## Infrastructure notes
- InfluxDB data directory: `/opt/dockerfiles/influxdb/` (requires sudo to inspect)
- InfluxDB runs as a Docker container; config is in `/opt/dockerfiles/`
