# EMHASS shared-state race — handoff brief (for the EMHASS source discussion)

**Discovered:** 2026-06-01, while bringing up the HWC planner. The HWC controller moved to
the sibling `../hwc` repository on 2026-08-10; this is retained as historical EMHASS context.
**Discovered on:** EMHASS v0.17.5. **Production:** official v0.17.9 since 2026-08-02.
**Scope of this brief:** the fix belongs upstream in EMHASS, not as a workaround in this
repo. This captures the problem + proposed fixes to seed that discussion.

## Status (updated 2026-08-02)

**Fix #1 (atomic metadata writes) is upstream and deployed in the official image.**

- The shared `entities/metadata.json` read-modify-write in `retrieve_hass.py:post_data` is
  now serialised by a process-wide `asyncio.Lock` and committed atomically via temp-file +
  `os.replace`; the racy recovery `os.rename` became a guarded `os.replace`. This closes the
  corruption / HTTP-500 race directly.
- Branch `fix/metadata-shared-state-race` on `rmounce/emhass` → **PR
  [davidusb-geek/emhass#919](https://github.com/davidusb-geek/emhass/pull/919)** (one squashed
  commit). 27 `test_retrieve_hass` tests pass incl. a concurrency regression test that
  reproduces the original `JSONDecodeError` / `FileNotFoundError` against pre-fix code.
- PR #919 merged and shipped in official v0.17.6. Production upgraded from local image
  `emhass:metadata-race-20260601` to `ghcr.io/davidusb-geek/emhass:v0.17.9` on 2026-08-02.
  First post-upgrade MPC solve: optimal, all 11 `mpc_*` entities published, no errors.
- Temporary file backup removed after validation; rollback remains available through ZFS snapshots.

**Not yet done:** fixes #2 (return result in HTTP response) and #3 (prefix-scoped state
files) below, and the "write once per publish" half of #1 (metadata is still written once per
entity, now each write atomic + lock-serialised).

## Symptom

Running a third frequent `entity_save` publisher (the HWC `naive-mpc-optim`, every 30 min)
alongside the battery DH and per-minute MPC optimisations produced:

```
ERROR retrieve_hass: Corrupted metadata file found at /data/entities/metadata.json. Creating a new one.
orjson.JSONDecodeError: unexpected content after document: line 187 column 4   # two concatenated JSON docs
FileNotFoundError: '/data/entities/metadata.json' -> '/data/entities/metadata_corrupt.json'   # recovery rename race
... and on the battery's own publish-data:
KeyError: 'sensor.mpc_p_load_forecast'   # metadata reset out from under a concurrent publish
```

Net effect: an HTTP 500 on the optim, and a corrupted/clobbered shared index that can make
a *concurrent battery publish* fail too.

## Root cause

`/data/entities/` holds a **single** `metadata.json` shared by every pipeline (`dh_*`,
`mpc_*`, `hwc_*`). In `retrieve_hass.py` `post_data` (~line 1380), for **each entity** in a
publish, EMHASS does: read `metadata.json` → set `metadata[entity_id]=…` → truncate and
rewrite the whole file. So one publish rewrites `metadata.json` ~N times (once per entity),
**non-atomically**. Two overlapping publishes (DH+MPC, or +HWC) interleave these
read-modify-writes → lost entries or concatenated-document corruption. The corruption
handler then does `os.rename(metadata.json, metadata_corrupt.json)`, which itself races
(file already moved) → `FileNotFoundError` → 500.

This is **latent in the stock battery pipeline too** (DH vs MPC share the same file); it is
rare only because DH runs on a slow day-ahead cadence while MPC runs ~every minute, so
overlap is infrequent and usually self-heals on the next clean publish. HWC just raised the
collision rate enough to surface it.

Relevant code: `retrieve_hass.py:post_data` (metadata read/modify/write + recovery);
`command_line.py:publish_data` / `_publish_from_saved_entities` / `_publish_standard_forecasts`;
`web_server.py:action_call`.

## Constraint worth noting

In v0.17.5 the optim action's HTTP response is **only a text ack** (`"EMHASS >> Action
naive-mpc-optim executed..."`) — no data. The optimisation result is written to the shared
`/data/opt_res_latest.csv` and to `/data/entities/*` (with `entity_save`). So a pure-network
client currently has **no way to retrieve results without `entity_save`** (hence the shared
store, hence the race). This is the gap proposed fix (2) closes.

## Proposed EMHASS improvements (any/all)

1. **Atomic metadata writes.** Write `metadata.json` via tmp-file + `os.replace`, and ideally
   **once per publish** rather than once per entity (accumulate then write). Optionally guard
   with a process lock. Fixes the corruption directly and the recovery-rename race.
2. **Publish without a follow-up HTTP request / return the result.** Have the optim action
   publish to HA itself (no separate `publish-data` call) and/or return the optimisation
   dataframe in the response body. Removes a round-trip, shrinks the race window, and lets a
   network-only client consume results without touching `entity_save` files at all.
3. **Prefix-scoped state files.** Namespace the saved state by `publish_prefix`
   (`metadata_<prefix>.json`, and/or per-prefix entity subdirs) so independent optimisations
   (`dh`, `mpc`, `hwc`) never share one `metadata.json`. Eliminates the cross-pipeline race on
   a single instance — isolation without separate containers.

(1)+(3) together remove both the intra-publish and cross-pipeline races; (2) is an
efficiency + decoupling improvement on top.

## Re-enabling HWC

The old EMHASS-backed HWC planner discussed here is retired. The live controller is the
separate sibling `../hwc` repository and does not use this `naive-mpc-optim` path. If that
path is ever reconsidered, confirm the deployed EMHASS image contains the upstream fix
before enabling any additional frequent publisher.

## Restart republish limitation — confirmed 2026-09-15

- Context: EMHASS `0.17.9`; HA restarted while EMHASS retained its data volume.
- HA lost the REST-created `dh_*` and `mpc_*` entities, as expected until republished.
- `POST /action/publish-data` with `publish_prefix: all` did not restore them.
- EMHASS reported no saved entity JSON files in `/data/entities`, fell back to the single
  `/data/opt_res_latest.csv`, inferred that result as 5-minute data, then tried to assign the
  configured 30-minute frequency. Pandas rejected the mismatch:
  `Inferred frequency 5min ... does not conform to passed frequency 30min`.
- A prefix-only DH republish hit the same fallback because the retained latest result was MPC.
- Operational conclusion: a shared EMHASS instance serving 30-minute DH and 5-minute MPC
  cannot use the generic latest-result fallback as reliable HA-restart recovery. Reseed the
  upstream HA forecasts and let each optimization republish its own prefix.
- A manual MPC trigger before reseeding `sensor.ai_load_forecast_high` omitted the intended
  runtime load array. EMHASS fell back to its internal ML forecaster, whose persisted
  `ForecasterRecursive` model was incompatible with the newer installed `skforecast`
  (`exog_dtypes_out_` missing). Preflight required runtime forecast entities first.
