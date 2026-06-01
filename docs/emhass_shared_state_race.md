# EMHASS shared-state race — handoff brief (for the EMHASS source discussion)

**Discovered:** 2026-06-01, while bringing up the HWC planner (`docs/hwc_emhass.md`).
**EMHASS version:** v0.17.5 (`ghcr.io/davidusb-geek/emhass`; custom images also built —
`emhass:rmounce`, `emhass:fix`).
**Scope of this brief:** the fix belongs upstream in EMHASS, not as a workaround in this
repo. This captures the problem + proposed fixes to seed that discussion.

## Status (updated 2026-06-01)

**Fix #1 (atomic metadata writes) is implemented, tested, and deployed locally.**

- The shared `entities/metadata.json` read-modify-write in `retrieve_hass.py:post_data` is
  now serialised by a process-wide `asyncio.Lock` and committed atomically via temp-file +
  `os.replace`; the racy recovery `os.rename` became a guarded `os.replace`. This closes the
  corruption / HTTP-500 race directly.
- Branch `fix/metadata-shared-state-race` on `rmounce/emhass` → **PR
  [davidusb-geek/emhass#919](https://github.com/davidusb-geek/emhass/pull/919)** (one squashed
  commit). 27 `test_retrieve_hass` tests pass incl. a concurrency regression test that
  reproduces the original `JSONDecodeError` / `FileNotFoundError` against pre-fix code.
- Built as the local image `emhass:metadata-race-20260601` (full EMHASS master + the fix) and
  **running on the production `emhass` container** (per `/opt/dockerfiles/emhass/docker-compose.yml`).

**Not yet done:** fixes #2 (return result in HTTP response) and #3 (prefix-scoped state
files) below, and the "write once per publish" half of #1 (metadata is still written once per
entity, now each write atomic + lock-serialised). PR #919 is awaiting upstream review/merge.

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

The HWC planner (`hwc_planner.py`) and its timer are **still disabled** in this repo. The
corruption race they would trigger is now fixed in the deployed build
(`emhass:metadata-race-20260601`, fix #1 above), so the original blocker is resolved on the
running instance. Before re-enabling per `docs/hwc_emhass.md`, confirm the EMHASS instance is
running a build that includes the fix (it is, as of 2026-06-01) — do **not** re-enable against
a stock image that predates it. Longer term, prefer the upstream release once PR #919 merges,
or move HWC to a dedicated instance.
