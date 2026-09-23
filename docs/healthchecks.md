# Repository healthchecks

Set one `HC_REPO_PING_KEY` in the ignored `.env` file. It is the Healthchecks.io **project Ping
Key**. Each monitored job uses a stable, unique slug, so check records have independent states
even though jobs share the same secret.

The shared helper is [healthchecks.py](../healthchecks.py). Current slugs are:

- `predict-load` — 30-minute load prediction service
- `price-listener` — event-driven price prediction listener
- `aemo-pec-mi-transition` — five-minute report capture and canary

Slug pings use `?create=1`, so Healthchecks creates each check on first ping. This avoids creating
and copying a UUID for every job. A shared slug or a shared UUID URL would let one job's success
clear another job's failure; give every independently monitored job its own slug.

Auto-created checks use Healthchecks defaults of a one-day period and one-hour grace. Set each
check's schedule or period and grace to match its job after the first ping. The current capture
needs a five-minute period; load prediction and the price listener need a 30-minute period. Use a
Management API key if check configuration should also be automated.

`HC_PREDICT_URL` remains a migration fallback for the load service and price listener while
`HC_REPO_PING_KEY` is unset. Those two jobs share that legacy check and can mask each other's
state until the project Ping Key is configured. The transition capture does not use this fallback.
