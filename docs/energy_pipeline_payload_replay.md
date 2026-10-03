# EMHASS payload extraction and replay

Status: offline extraction; production still uses HA scripts/templates.

## Files and boundaries

- `energy_pipeline/payloads.py`: pure frozen-input → DH/MPC payload functions; no I/O.
- `scripts/replay_energy_payloads.py`: selected HA entity capture; parity comparison.
- `scripts/ha_payload_oracle.py`: original repo Jinja rendered by the installed HA engine
  in `hass`, with frozen states/clock; evaluates script variables only; no services executed.
- `tests/energy_pipeline/test_payloads.py`: independent policy edge cases.
- Original payload/script source: `hass/packages/emhass.yaml`; deployed copy verified equal
  during the [runtime inventory](energy_pipeline_runtime_inventory.md).
- HA engine/environment changes may require adapting the oracle; it imports HA internals.

## Run

Use project venv. Docker access needed for the oracle; capture also needs HA API access.
Credentials loaded from existing config/secrets; never stored in replay snapshots.

```bash
.venv/bin/python -m pytest tests/energy_pipeline -q
.venv/bin/python scripts/replay_energy_payloads.py --capture data/energy_replay/baseline.json
.venv/bin/python scripts/replay_energy_payloads.py data/energy_replay/baseline.json --variants
```

- Snapshots are ignored JSON under `data/energy_replay/`; household telemetry, not public fixtures.
- Comparison checks every payload key and array element; numeric tolerance `1e-9`; mismatch exits 1.
- `--variants`: deterministic mutations, labelled separately from recorded observations.
- DST case shifts the whole input timeline, including JSON-encoded HWC dates; preserves coverage.
- Snapshot is one HA state response with capture completion time; not an atomic upstream revision.
- Frozen replay does not reproduce DH's helper writes, HWC snapshot event, one-second delay,
  changing inputs during render, solver execution or publication ordering. Coordinator must
  define accepted-plan feedback transactions and input readiness before taking ownership.

## Evidence, 2026-10-01

- One recorded nighttime snapshot: exact DH/MPC payload match; 144/168 load slots.
- 16 mutated cases: quantile weights -1/0/1/.371, free export allowance, full/low SoC,
  same DH reground block, missing prior SoC plan, sparse HWC, malformed saved HWC JSON fallback,
  curtailed PV reconstruction, empty/zero prior power, either side of a 30-minute boundary,
  populated horizon spanning Adelaide spring DST transition.
- Eight unit tests: sparse HWC averaging/tail repeat, SoC feedback/full guard, negative
  deviation policy, DH final endpoint, scaling residual/clipping, smoothing, DST allowance,
  target-offset tail selection.
- Replay found MPC floor filter precedence: HA rounds denominator `100`, then divides;
  floor itself is unrounded. Python preserves this. Capacity arithmetic grouping also preserved.
- Empty prior power currently yields just two live MPC power slots. Zero base residual and
  clipping can prevent energy conservation. Extraction preserves these quirks; coordinator
  must reject unusable coverage rather than assume payload generation implies readiness.

## Remaining gates

- Two independent daytime bundle-overlaid snapshots now pass exact HA Jinja parity:
  [handoff evidence/limits](energy_pipeline_handoff.md). Negative-price/curtailment/solver-restart
  independent snapshots remain open.
- Target-offset automation has unit coverage; full HA automation scheduling not replayed.
- No inference/model/solver/control migration yet; offline parity does not prove safe cutover.
- Resident price calculation-only worker implemented: [shadow evidence](energy_pipeline_resident_price.md).
  Memory stability and independent validated source refreshes remain gates before production cutover.
- Then shadow coordination; only move live solve ownership after replay/recovery gates pass.
