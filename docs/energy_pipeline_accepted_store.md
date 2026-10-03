# Accepted shadow checkpoint

Status: opt-in storage; no publication or production ownership changes. 2026-10-02 Adelaide.
Parent: [acceptance contract](energy_pipeline_price_acceptance.md).

- `services/resident_price.py --state-file /tmp/energy-shadow/accepted.json --runs 1` enables
  `energy_pipeline/accepted_store.py`. Omit flag →existing in-memory shadow behaviour.
  Use a private writable durable directory for sustained operation; `/tmp` is validation only.
- One latest complete accepted family; bounded 256 KiB JSON, schema 1, mode `shadow`, SHA256
  integrity envelope. Digest detects corruption, not authentication. No event history/publication ledger.
- Contains run/model/parent identities, 144-point quantile values/UTC targets, input/source revisions,
  source freshness evidence, captured tariff profile, capture/acceptance/start/expiry clocks.
  No HA tokens, raw configuration or household history stored.
- Validate finite complete aligned quantiles, timestamps/expiry, tariff identity and revision types
  before writing and on recovery. File mode 0600; separate advisory flock held for process lifetime
  prevents cooperating shadows sharing a path. Other paths remain separate owners; no production lease.
- Commit on inference thread after point-in-time acceptance: same-directory temporary file,
  flush + file fsync, atomic replace, directory fsync. At most one retained bundle; prior checkpoint
  survives failures before replace. Failure after replace →uncertain durability; stop and inspect/recover.
- Storage failure fatal; no heartbeat advance/five-minute retry. Aggregate inference/acceptance/storage
  deadline remains 180s. Threads cannot be killed: timed-out save may finish late. Ownership lock stays
  held until process exit; surviving checkpoint is always historical evidence.
- Post-save observed HA/source changes or time expiry reject live advancement and request reconciliation.
  The already-saved record remains historical, not an actuation lease. External tariff/model changes
  after acceptance still require fresh validation before any future publication.
- Startup validates checkpoint, logs recovered run/time-current status; NEVER restores `completed`,
  heartbeat, source/model caches or publication state. Startup/reconnect runs fresh reconciliation.
  Missing file normal; malformed/oversized/checksum/contract failure stops startup, retains file.
- Expiry = earlier of capture +180s and next UTC half-hour boundary. `time_current` checks that gate,
  current interval and nonnegative capture age; passing it does not validate current source parents.
- `.tmp` files left by a hard crash are ignored on recovery. No automatic deletion of forensic files.
  Lock file persists; kernel releases flock at exit. This is a checkpoint, not partial-publication recovery.

## Verification

- Focused regression suite: 159 passed, including abrupt process-exit recovery.
- Exact roundtrip/negative values, private file permissions, restart ownership release, expiry,
  corrupt/oversized/digest-valid invalid records, invalid family, file fsync/replace/directory fsync
  failure, temporary cleanup, abrupt child exit. Listener: rejected results never written, save before
  heartbeat, recovered data historical, storage failure/timeout, observed save races and expiry.
- Live attempt 2026-10-02 19:33 Adelaide failed existing weather coverage admission: 145 returned
  weather points did not cover current 72-hour window. No prediction/checkpoint/publication occurred.
  Coverage mismatch subsequently isolated and corrected with bounded incumbent-tail admission;
  [live save/expired-checkpoint restart reconciliation](energy_pipeline_weather_coverage.md) passed
  2026-10-03 09:34/09:41 Adelaide. Both processes exit 0, no publications.

Next: publication transaction
with current-parent revalidation, idempotency/partial-write handling and verified single ownership.
