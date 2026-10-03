# Local price publication transaction rehearsal

2026-10-03 Adelaide. Opt-in SQLite-only shadow; no HA output writer or production cutover.
Parent: [accepted checkpoint](energy_pipeline_accepted_store.md).

## Usage / ownership

- `services/resident_price.py --runs 1 --state-file /tmp/accepted.json --publication-db /tmp/publication.sqlite`.
  Publication flag requires state file; omitted flags preserve existing shadow behaviour.
- `energy_pipeline/publication.py` has no HA write transport. Target entity IDs are labels in local
  SQLite rows. Inference acceptance revalidation reads HA; only incumbent publisher writes HA sensors.
- Separate process-lifetime flock prevents cooperating processes sharing the journal. SQLite FULL
  synchronous mode, separate committed sink/receipt operations, explicit completion-marker transaction.
  New database file mode 0600; private directory recommended. Lock held through late worker exit.
- At most one pending plan, ~10 retained jobs, three current sink rows and one marker. Latest fresh
  plan abandons prior pending jobs. Terminal history pruned on new plans; SQLite file may retain freed
  pages (not a database-file byte-size cap). Plans bounded to 512 KiB.

## Frozen plan / state machine

- Plan includes complete accepted bundle, target mapping, tariffed payloads, stable acceptance-time
  `last_updated`; SHA256 identity covers all plan fields. Payloads own JSON data, stable after recovery.
- Replay oracle: incumbent `_publish_lgbm_model_to_hass` + `publish_forecast_to_hass`, with HA API mocked.
  Exact state/attributes/records parity for missing and captured tariff profiles, including negative
  prices. Quantile order p30/p50/p70; no production helper refactor/behaviour change.
- `pending` recorded durably before writes. Revalidate current parents/source readiness/interval/age
  before start, each of three sink writes, and final commit. Shared 180s inference/acceptance/checkpoint/
  rehearsal deadline; observed HA/source generations checked around the work.
- Sink upsert and receipt commit are separate: crash after write/before receipt repeats IDENTICAL
  payload on explicit resume. Recorded receipt skips that write; final commit verifies actual sink
  against every receipt. Committed duplicate returns historical completion without replay/lease renewal.
- Full family and final validity →atomic local marker + `committed` status. Parent/expiry change
  →`abandoned`, no new marker; partial rows remain diagnostic. Storage/revalidation exception fatal,
  retained pending plan; no heartbeat advance. Configuration rejection preserves restart-only policy.
- `committed_family()` reads one database snapshot and rejects mixed sink job IDs/payloads even
  if old marker remains. This is a simulated consumer guard, not an HA automation integration.
- Startup abandons unfinished plans and reconciles fresh inputs; no automatic historical output replay.
  Explicit engine resume is tested with current-parent callback + expiry. Timeout may leave a late
  local commit; all recovered output remains historical evidence, never production authority.

## Evidence

- Focused suite: 177 passed; existing Influx deprecation warning only.
- Fault injection before/after receipt, invalid/current-parent changes between writes, expiry,
  old marker/mixed generation, stable duplicates, plan tampering, competing owner, bounded rows,
  listener heartbeat/config/fatal failure gates. Independent process exits after sink write/before
  receipt; restart resumes missing receipts and yields a verified committed family.
- Live calculation-only run 09:55–09:56 Adelaide, exit 0: run `20261003T002605.469393Z`,
  144 points per quantile, three model loads, generation 7.544s, RSS 1127.4 MiB. Local rehearsal
  committed; independent inspection verified three matching outputs/marker, zero pending jobs.
  No HA output requests, production health writes, unit changes or reloads. Shadow stopped.

## Remaining gates

- This verifies local protocol/storage, not remote HA idempotency or exactly-once notifications.
  Real REST writes cannot atomically update three entities; existing consumers do not read a bundle
  marker. Design a single accepted-bundle boundary/consumer handoff before production cutover.
- Revalidate at actual publication, verify exclusive production ownership, exercise remote timeout/
  unknown acknowledgement/restart/partial-write behaviour and explicit rollback. No HA transport added.
- Sustained memory/provider freshness/failure/DST gates remain open. BOM metadata patch unapplied.
- Next: design bundle-aware DH/MPC shadow handoff; gather independent daytime/curtailment/recovery
  replay evidence. Keep current HA solver/control ownership until separate acceptance gates pass.
