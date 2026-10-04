# Full-cycle PV support audit — 2026-10-05

- Follow-up [controlled cycle mechanisms](controlled_cycle_value_2026-10-05.md) completed:
  solar replacement/export opportunity cost and later import scarcity change sign; no historical
  savings claim. Equal ending stock reached physically; both-leg wear included.

- Offline; APF retained; no optimizer runs, HA/device writes, service changes or training.
- Follow-up to [inventory screening](inventory_cycle_screen_2026-10-05.md): selected20.583h
  full-to-full excursion Sep27 08:40–Sep28 05:15 UTC; minimum observed SoC41.34%.
- New fixed eight-source archive Sep27 08:35–Sep28 05:20 UTC: gross PV, two raw strings,
  battery DC power, inverter AC power, derived loss, derived SoC and last-full helper.
- `export_control_history.py --cycle-support-only`:≤24h,≤100,000 window rows/source.
  Cannot combine with full DH/EMS/energy flags. Existing default profile stays≤90min/10,000 rows.
  Read-only fixed fields/entities; no arbitrary-query surface or source-age relaxation.

## Recovery result

- 247 five-minute intervals: gross PV117 complete, each string117 complete; battery/inverter/loss247.
- Strings recover **zero** of130 missing gross-PV intervals. Unknown block09:15–20:05 UTC,
 10h50m; overnight plus dusk/dawn. Both underlying string histories stop supplying fresh readings.
- Where complete, gross minus summed strings max absolute difference0.008395W: effectively
  same derived measurement, not an independent meter.
- AC+battery+derived-loss residual over unknown intervals: mean0.832039W, max absolute22.315829W.
  Compatible with little PV, but loss is computed from the same powers; cannot validate missing
  PV independently or admit zero-filled night actuals. No recovered measured target written.
- `audit_cycle_support.py`: immutable archive hash,≤120s raw holds, contiguous missing runs,
  string recovery counts, algebraic residual labeled non-independent. No replay admission.

## Full-helper evidence

- Raw helper exists: receipt Sep27 05:35:04.858961 UTC, local state `2026-09-27 15:05:04`.
  Australia/Adelaide interpretation gives05:35:04 UTC. Initial8h holdoff active; at cycle end expired.
- Next receipt Sep28 05:19:21.858278 UTC, state `2026-09-28 14:49:21`; after cycle end.
  Future receipt excluded from endpoint interpretation. Counterfactual must own its last-full state.
- Repository [full timestamp automation](../hass/packages/emhass.yaml) triggers on numeric-state
  transition **above99.99%**; [controller](../hass/automation-sigenergy-emhass.yaml) full branches
  use **≥99.5%**, with8h holdoff. Updating last-full on first99.5% crossing would change semantics.
  Raw receipts confirm recorded helper values; do not prove historical YAML/config stability.
- Endpoint≥99.5% therefore neither guarantees a new full event nor identical modeled ending
  energy. Earlier nominal/BMS inventory limitations remain.

## Decision and next effort

- This cycle cannot yet support a fully recorded timing comparison under existing strict rules.
  No reason to launch thousands of minute solves before closing input gaps and controller coverage.
- Pause further expansion of historical timing reconstruction for now. Next: controlled paired
  cycle scenarios using existing physical executor, with equal initial energy/common inputs:
  (1) extra export subsequently replaced from capacity-limited solar; (2) extra energy needed later.
  Track cash, ending stock, incremental throughput/wear and binding-capacity/floor evidence.
- Mechanism tests, not historical savings or production proof. Any assumed overnight-zero curve
  explicit as scenario input, never relabeled measured PV. Retain APF for later forecast comparisons.
- 23 tests pass:5 archive-profile bounds,4 recovery/helper causality and14 existing measured-target
  integration tests. Synthetic balanced powers/stale zero strings cannot admit missing PV.
- Private ignored evidence: `data/energy_replay/cycle_support_20261005_full_cycle/`,
  `cycle_support_audit_20261005_full_cycle.json`. History SHA256
  `38b78f1d0db410ca47d58f92f9b55a24c05a83fa2a074b6e1e7ca2c8a6326b60`.

```bash
./.venv/bin/python eval/audit_cycle_support.py \
  --history data/energy_replay/cycle_support_20261005_full_cycle \
  --start 2026-09-27T08:40:00Z --end 2026-09-28T05:15:00Z \
  --local-time-zone Australia/Adelaide --output data/energy_replay/NEW_CYCLE_AUDIT.json
```
