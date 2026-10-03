# Resident price result acceptance

Status: calculation-only shadow; no publisher or ownership cutover. 2026-10-02 Adelaide.
Parent: [pipeline plan](energy_pipeline_architecture_plan.md),
[source/freshness/memory checkpoint](energy_pipeline_source_cache.md).

## Contract

- One frozen HA response feeds all inference quantiles. A second states read verifies parents
  after inference; no AEMO/weather/Influx acquisition on generation/acceptance path.
- `PriceWorker.evaluate_completion`: complete monotone finite 144-point family, current UTC
  half-hour start, aware capture age 0–180s. Recheck interval/age after acceptance network read.
- Cache age/coverage checked before and after that read. Compare source content revisions,
  HA APF/Solcast/poll digests, process config, effective tariffs, tariff-file bytes, active model
  pointer bytes and installed model artifact stat signatures.
- Active pointer/artifact check never loads/promotes models. Same stat signature limit as cache:
  not byte-content validation against an adversary preserving inode/size/mtime.
- HA consumed-entity events (including Solcast and poll marker) and reconnect increment ingress
  generation. Source content/readiness changes increment cache generation. Change during acceptance
  rejects even when the final response appeared to match. Unrelated HA entities do not trigger work.
- Inference + acceptance share one worker, one aggregate 180s deadline. Timeout stops scheduler;
  late thread result discarded. Cancellation cannot terminate native threads.
- Accepted completion alone advances shadow heartbeat. Rejection retains preceding completion and
  heartbeat; that object is historical evidence, not a lease to publish/actuate indefinitely.

## Decisions

| Condition | Decision / next work |
|---|---|
| Changed source/HA/tariff/model; interval rollover; observed change during acceptance | Reject; one coalesced reconciliation |
| Expired/missing source coverage | Reject; wait for source readiness/heartbeat, no immediate loop |
| Changed configuration | Reject; requires process restart |
| Invalid family or failed verification request | Reject/fail; bounded five-minute retry |
| Worker deadline / RSS guard | No acceptance; stop process |

- `energy_pipeline/acceptance.py` owns typed decisions; logs reasons without input values/secrets.
- Minimal incumbent listener event hook preserves its APF-only trigger. Resident overrides consumed
  entity routing; no production subprocess, retry, health or publication policy changed.
- `--runs` fails if a finite benchmark is superseded/not accepted; normal event loop reconciles.

## Limits before production

- This is point-in-time shadow admission. Remote HA updates and external model/tariff writes are
  not an atomic multi-source transaction. An opt-in [durable shadow checkpoint](energy_pipeline_accepted_store.md)
  now preserves accepted bundles as historical evidence. No publish transaction or partial-write
  recovery yet; publication must recheck validity and enforce single ownership.
- [Tariff snapshot](energy_pipeline_tariff_snapshot.md) freezes maps/loss/scaling from one read
  throughout generation; acceptance rejects persistent file changes. Completion retains the snapshot.
- Provider freshness unknown/transport-only evidence remains distinct from cache acquisition age.
  No new provider-age thresholds, health pages or upstream/HA integration changes.
- Multi-hour/full-day memory and live disconnect/failure/interval/DST gates remain open.

## Verification

- Tests: APF/PV/source/tariff/config/model pointer/artifact mutation; interval rollover before and
  during acceptance read; age expiry; invalid values; failed HA verification; source expiry/recovery;
  HA/source/reconnect races; acceptance-thread timeout; old completion retained; bounded replacement.
- Incumbent listener/forecast/model-bundle regressions included. Current counts/evidence in checkpoint.
- 105 focused tests pass. Six-minute live shadow 2026-10-02 14:03–14:09 Adelaide, exit 0:
  startup, real APF update, changed AEMO refresh all accepted after verification. Three completions,
  three model loads; generation 8.4s cold / 2.2–2.3s warm; RSS 1139 →1149 →1216 MiB.
  No publication or heap trim. Rejection/failure/reconnect/rollover injected in tests, not this
  live session. Final acceptance-duration logging added after that session.

Hourly [freshness capture adapter](energy_pipeline_bom_freshness.md) implemented; HA patch unapplied.
Live expired-checkpoint restart reconciliation verified; next explicit publication/rollback transaction.
Production price owner stays incumbent until gates pass.
