# Repository Healthchecks

## DST start audit — 2026-10-04

- Read-only journals/status audit through ~09:40 ACDT; no runtime changes.
- Window: 2026-10-03 14:00 UTC onward (~23:30 ACST); Adelaide skipped 02:00–02:59.
- Price: 110 successful publication summaries, each quantile 144 points;
  largest publication gap 371s. Across jump: 01:56:41 ACST → 03:00:33 ACDT
  (~232 real seconds); UTC target start advanced 16:00 → 16:30 normally.
- Load: 19 successful publication summaries, each quantile 144 points;
  largest gap ~1840s. Across jump: 01:31:56 ACST → 03:01:56 ACDT
  (~1800 real seconds). No skipped half-hour run observed.
- HWC journal received MPC cost updates at 01:55:24 ACST, 03:00:20 ACDT,
  03:05:22 ACDT. No HWC errors or DST timestamp exceptions observed.
- AEMO visualisations timeouts occurred before/after jump; some exhausted retries,
  but price/load still published. No other error/failure messages in inspected user units.
- Aggregate journal: 518 successful invocations, no failure messages in window.
  Status files: load has no recorded failure; price last failure remains Oct 3 13:26 ACST;
  aggregate currently success. Capture last failure ~Oct 3 23:32 ACST, subsequently recovered.
- Services/timers active as expected. Evidence supports smooth forecast/MPC-update continuity;
  physical actuator history and every tariff slot were not independently audited.

Configure the existing single-check ping URL as `HC_PREDICT_URL` in the ignored root `.env`.
The repository does not create extra Healthchecks records or need a project Ping Key.

Jobs write their latest result under ignored `data/healthcheck_status/`:

- `predict-load` — 30-minute load prediction service; 30-minute period plus 10-minute grace.
- `price-listener` — event-driven price prediction listener; 30-minute period plus 10-minute grace.
- `aemo-pec-mi-transition` — report capture and canary; two consecutive failed runs trigger an alert. A missing success becomes stale after 15 minutes (5-minute period plus 10-minute grace).

`ai-energy-healthcheck-aggregate.timer` evaluates those files every minute and is the only
component that contacts Healthchecks. It sends a success heartbeat while every job has a recent
success. If any job records a failure or exceeds its freshness window, it sends `/fail` and stops
sending success pings until all jobs recover. This prevents one job's success from masking
another job's failure. A failure remains pending through a quick recovery until the aggregate has
reported it.
Status writes and aggregate evaluation share a filesystem lock. A job result that arrives
during an aggregate pass waits for that pass to finish, then is evaluated on the next pass.

On first installation, jobs without status get one period plus grace to produce their initial
success. After that, a missing or stale result fails the aggregate. The remote single check should
have a period of about two minutes and a short grace period so a stopped aggregator is detected;
job cadence and grace are enforced locally. The AEMO capture job logs an isolated failed run,
but the shared check stays healthy if the next run succeeds within the freshness window.
Two failed capture runs remain pending until the aggregate reports them, even if a later run
succeeds before the next aggregate pass. Other monitored jobs still alert on one failure.
The capture canary treats a visualisations API transport outage as a diagnostic for 30 minutes;
NEMWeb report failures and API response/schema failures still affect the capture result.

On 2026-09-27 at 04:17 Adelaide time, the price listener's child published all three price
forecast sensors after AEMO visualisations API retries, but had not exited at the old 120-second
listener timeout. The listener killed it and recorded a failure; its 04:22 retry succeeded. The
child timeout is now 180 seconds to accommodate this observed slow path while still detecting a
stuck prediction process.

On 2026-10-03 at 13:26 Adelaide time, `price-listener` failed after reading the STPASA
parquet during `ai-energy-stpasa.service` refresh (13:25:42–13:25:58). The reader reported
missing Parquet footer magic bytes at 13:25:56; historical STPASA coverage fell to 0%,
and model generation rejected insufficient history. No new price forecast was published.
The aggregate reported failure at 13:26:36; the retry published all three sensors at
13:31:19 and the aggregate recovered at 13:31:47. The writer uses direct `to_parquet`
publication; concurrent read/write is the likely cause. Fixed the same day: the STPASA
writer now writes and flushes a unique temporary file in the destination directory, then
atomically replaces the archive. Readers retain the old complete file during refresh;
failed writes preserve it and clean up the temporary file. Existing permissions are preserved.
The next timer invocation loads this change; no service restart is required.

On 2026-09-25 at 23:13 Adelaide time, the AEMO visualisations `5MIN` API timed out after
three attempts; the 23:17 capture succeeded. This transient endpoint timeout prompted the
capture-specific persistence threshold.

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
