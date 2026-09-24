# Repository Healthchecks

Configure the existing single-check ping URL as `HC_PREDICT_URL` in the ignored root `.env`.
The repository does not create extra Healthchecks records or need a project Ping Key.

Jobs write their latest result under ignored `data/healthcheck_status/`:

- `predict-load` — 30-minute load prediction service; 30-minute period plus 10-minute grace.
- `price-listener` — event-driven price prediction listener; 30-minute period plus 10-minute grace.
- `aemo-pec-mi-transition` — report capture and canary; 5-minute period plus 3-minute grace.

`ai-energy-healthcheck-aggregate.timer` evaluates those files every minute and is the only
component that contacts Healthchecks. It sends a success heartbeat while every job has a recent
success. If any job records a failure or exceeds its freshness window, it sends `/fail` and stops
sending success pings until all jobs recover. This prevents one job's success from masking
another job's failure. A failure remains pending through a quick recovery until the aggregate has
reported it.

On first installation, jobs without status get one period plus grace to produce their initial
success. After that, a missing or stale result fails the aggregate. The remote single check should
have a period of about two minutes and a short grace period so a stopped aggregator is detected;
job cadence and grace are enforced locally.

Run the aggregate manually with `.venv/bin/python healthchecks.py aggregate`. It prints the names
of failed jobs but never prints the configured URL. Job wrappers can record a result with
`.venv/bin/python healthchecks.py record JOB --exit-code CODE`. Use `aggregate --force` once to
resynchronize the remote check after removing an old direct pinger.

On 2026-09-25, two old direct pingers were found. The running user-level
`ai-energy-listener.service` still had the pre-aggregation code loaded from its 2026-09-18 start.
The `ai-energy-predict.service` file on disk had been updated, but the user systemd manager had
not reloaded it; its cached command still ran `curl` against `HC_PREDICT_URL`. The listener was
restarted and the user manager was reloaded. A forced aggregate ping then restored the current
failure state. Restart long-running services and reload their manager after deploying a change
that removes direct pings; restarting only the aggregator cannot prevent a cached unit or old
process from clearing its failure state.
