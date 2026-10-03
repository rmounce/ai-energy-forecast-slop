# Accepted bundle → DH/MPC shadow handoff

2026-10-03 Adelaide. Calculation/payload/journal only; HA still owns solves and control.
Parent: [local publication transaction](energy_pipeline_publication.md).

## Contract / usage

- `services/resident_price.py --runs 1 --state-file /tmp/accepted.json --publication-db /tmp/publication.sqlite --handoff-shadow`.
  Flag requires local publication. No solver requests, helper writes, HWC events or HA output writes.
- After local family commit: one selected HA states capture, bounded 30s request-start →completion;
  revalidate accepted price parents/time before recording handoff. Not an atomic HA/upstream snapshot.
- `energy_pipeline/handoff.py`: owned snapshot; replace ALL three DH AI price entities with the
  committed frozen tariffed bundle. Original live price triplet cannot leak into DH payload.
  Current entity mapping must match pure builders' policy; reject unexpected mapping.
- Pure builders preserve HA arithmetic, HWC precedence, SoC feedback and confirmed-price behavior.
  DH explicitly names new price bundle as parent. MPC keeps Amber five-minute prices and existing
  HA DH power/SoC plans. Their entity digests are separate parents; DH plan's price parent remains
  `unknown`. A new price bundle does not imply a new DH solve or a coherent new MPC lineage.
- Validate 144 DH/168 MPC vector lengths, finite values, nonnegative powers, SoC fractions, positive
  effective capacity and usable critical live states. Check positional DH load/Solcast targets,
  MPC DH power targets and next 167 five-minute prices. Matching lengths alone insufficient.
- `coverage_ready` is diagnostic structure/alignment, NOT freshness/provenance or solve permission.
  Both payloads ALWAYS `solve_authorized=False`. Default/fallback extraction preserved; critical
  unavailable live state cannot be mistaken for usable zero. No actuator admission defined here.
- `handoffs` table: one record per retained local publication job, selected snapshot, payloads,
  parent identities and readiness. Require matching committed local family before save; 2 MiB
  per-record limit; prune with job history. Household telemetry: private journal, never commit records.
- Price heartbeat can advance while handoff coverage is false; forecast acceptance and solver
  readiness separate. Superseded price/failed capture/storage cannot authorize a handoff solve.
- `scripts/replay_energy_payloads.py --handoff-db /tmp/publication.sqlite` reads journal readonly,
  renders exact stored snapshots with installed HA Jinja engine and checks stored/Python/oracle parity.
  Historical replay uses capture clock; it does not renew validity.

## Confirmed external quirk

- At 11:08 Adelaide, BOTH extended Amber price entities had `start_time` +1 second from UTC
  five-minute boundary, exactly 300s spacing, 196 future rows. First targets 01:40:01/01:45:01/01:50:01Z.
  Exact-boundary validation initially reported false despite complete correct-slot coverage.
- Alignment accepts only offsets 0..1s after intended boundary; preserves original timestamps and
  comparisons in payload extraction. +2s/negative offset/gaps/misordered slots still rejected.
  Source generating this offset not yet identified; no bridge/template changes.

## Evidence

- Focused suite: 196 passed; existing Influx deprecation warning only. Owned bundle substitution,
  expiry/capture budget/mapping rejection, distinct MPC lineage, partial/misaligned coverage,
  manifest covers all Jinja entity references, confirmed timestamp offset, unavailable telemetry,
  journal-parent match and listener capture/save-before-heartbeat checks.
- Two independent daytime captures, measured PV mode, 11:08 and 11:21 Adelaide. Accepted 144-point
  price families; local publication/handoff commit; both processes exit 0. Three model loads each;
  generation 12.626/10.018s; RSS 1126.1/1139.1 MiB. No real solves/outputs/health/unit/reload changes.
- Second startup recovered expired prior checkpoint as historical and reconciled fresh inputs.
  DH/MPC coverage ready after narrow offset correction; solve permission remains false.
- Both recorded bundle-overlaid DH/MPC payloads match installed HA Jinja exactly: ZERO mismatches
  at tolerance 1e-9. First record retains original failed timestamp diagnosis; re-evaluation with
  corrected check passes both. Last critical-unavailable-state checks added after live capture;
  both stored snapshots pass current validation offline. No independent curtailed/restart-solver case yet.

Next: isolated DH solve rehearsal, preserve result/price-parent identity, then feed THAT accepted DH
plan into MPC shadow. Existing EMHASS shares output state; audit installed behavior/isolation before
calling a shadow solver. See [shared-state history](emhass_shared_state_race.md). Current HA solve/control
ownership unchanged; remote publication, provider freshness and sustained-memory gates remain open.
