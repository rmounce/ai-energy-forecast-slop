# PV controller branches and clock sensitivity — 2026-10-05

- Follow-up: [cycle coverage and low-stock replay](inventory_cycle_screen_2026-10-05.md).
 15min/36 requests near15% SoC: zero timing cash/inventory gain. Four observed full-to-full
  excursions have missing PV support; audit inputs before broad full-cycle execution.

- APF retained; offline evaluation only. No HA/service/device writes, reloads or training.
- `dh_feedback_replay.py --pv-export`: controller revision`pv_export_v2`, separate checkpoint
  contract from preceding export-only experiment. Same forecast/terminal policy in both arms;
  own physical battery, MPC plans, commands and DH helpers persist across continuation.
- PV-only export (`P_batt=0,P_PV>0,P_grid<0`): charging disabled; general price≤discharge weight
  also disables discharge, otherwise default24kW discharge allowed within hardware limits.
  General-price/weight threshold sampled at command activation/fallback, not continuously while
  holding a command. Current feed/flexible/on-grid guards still apply throughout execution.
- Surplus-PV charging (`P_batt<0,P_grid=0,P_PV>0`, no curtailment): self-consumption mode.
  From10:00 inclusive to16:00 exclusive local time, discharge disabled; otherwise default limits.
  Site timezone bound in execution contract, with explicit DST/boundary tests.
- PV-only full-SoC≥99.5% cases still fail: earlier YAML branch needs unavailable historical
  holdoff state. Grid charging, charging with simultaneous export, positive-grid partial battery
  support and curtailment branches remain unsupported. No general controller completeness claim.

## Inputs and execution contract

- New bounded read-only EMS archive: Sep30 21:55–23:25 UTC,90min. Includes curves/guards and
 36 raw effective-general-price receipts. No raw minimum-export-SoC/discharge-weight helper rows.
- Discharge-weight fallback0.04: Oct3 captured journal; last changed/updated/reported
  Sep18 08:23:57.973413 UTC. Helper/capture/record/source hashes explicit.5% export guard has
  analogous older unchanged capture. These are weaker than independent historical availability.
- `--timer-clock-seconds 25` or`25.1`: known helper clocks untouched; missing eligible timer
  clocks require both candidates' causal input consistency and later paired published evidence.
  Five-minute trigger origins, conflicting candidates and stale/missing sources reject.
  Retrospective clock sensitivity, not observed capture or a production admission rule.
- Command coverage≤120s, feed freshness through each held segment's end, general-price receipt
  age≤900s at branch selection, static weight≤31days. Incompatible clock/controller/timezone/
  loss assumptions cannot resume a checkpoint. Continuation requires same immutable guard archive.
- `--reuse-solves`: reproduce source reports first; exact request identity/equality and physical
  result validation required. Different inputs require fresh pinned core solve. Duplicate/mutated
  cache entries reject. Reused result origin/solve timing retained; reuse count separate from new.
- `--cached-only`: host execution from validated exact requests, fails if any new solve needed;
  no Docker/optimizer invocation. Backend/code/source lineage explicit. New requests still run in
  bounded read-only/network-none installed-core container. No archived SoC re-grounding.
- First30min rebuilt under expanded contract:72 exact requests reused,0 new solves, economics
  unchanged to floating precision. Intermediate35min extends through PV-only operation with
 12 new solves, zero extra cash/inventory difference in that additional5min.

## One-hour result

- Sep30 22:00:23.787881–23:00 UTC, actual duration3576.212119s. Four batches,144 validated
  DH/MPC requests per clock profile. Both25.0/25.1s profiles complete; only22:41 capture is inferred.
  Fixed140W DC-overhead ablation, nominal40.3kWh stock, common APF/forecast/terminal policy.

| Metric | Prior-plan fallback | Hold accepted command |
|---|---:|---:|
| Variable credit AUD | 0.81061754 | 0.87455800 |
| Grid export kWh | 3.07023553 | 3.32885470 |
| DC battery throughput kWh | 3.23075638 | 3.48169727 |
| Ending model inventory kWh | 28.26554300 | 27.99077646 |
| Ending model SoC% | 70.13782 | 69.45602 |
| Final DH offset percentage points | 2.09 | 2.77 |

- Extra credit6.3940c, ending inventory−0.274767kWh, extra DC throughput0.250941kWh.
  First30min:4.9632c/−0.198315kWh.45min:4.9733c/−0.198833kWh. The final15min adds
 1.4207c but increases inventory spent by0.075934kWh; cash alone overstates the benefit.
- Net AUD =0.06394046 −0.27476654×ending-energy value −0.25094089×DC wear cost.
  Ending-energy break-even23.271c/kWh before wear;19.618c/kWh with illustrative4c DC wear.
  Wear is a sensitivity, not independently measured degradation or configured discharge weight.
- At20c/kWh ending energy and4c wear: −0.1050c over the hour, versus+0.2116c at30min.
  At30c ending energy/4c wear: −2.8527c. No demonstrated deployment-worthy net gain.
  This is a change in modeled ranking under the same terminal-value assumption, not realized loss.
- Clock25.0/25.1 profiles produce identical cash and equivalent stock to floating precision;
  maximum per-arm ending-stock difference<1e−14kWh. Clock interval is robust in this one case,
  not proof of historical capture time or future scheduling/actuation latency.
- Profile25: first72 requests reused; third batch12 reused/24 new; fourth36 new. The12 reused
  third-batch requests originate from the valid35min intermediate. Alternative profile25.1
  needs only2 new requests at the inferred22:41 origin; all fourth-batch requests reuse exactly.
 144 requests/profile counts logical evaluations, not144 additional optimizer executions.
- Existing safe branch transitions now execute rather than stop: self-consumption, PV-only
  export with discharge permitted, surplus-PV charging and battery export. Clipped seconds include
  controller no-charge/no-discharge constraints; they do not prove firmware tracking failures.
- Next gate: a battery cycle with binding future inventory/capacity, including full-battery/
  holdoff and scarce-energy states. Retain APF and prioritize execution/value fidelity before
  training a more complex model or promoting the command-hold policy.

## Validation and saved evidence

- 141 affected tests pass: pure policy/physical modes, price causality, tariff/DST boundaries,
  exact-cache mutation/identity checks, incompatible controller/clock checkpoints, known-clock
  preservation and rejected ambiguous/stale inferred clocks. No source-age relaxation.
- Both one-hour chains reproduce exact saved requests/events/economics/checkpoints and validate
  predecessor report lineage. Independent fingerprints/backend/source receipts retained.
- Authoritative ignored outputs: `ems_feedback_20261005_pv2_clock25_{15m,30m_chunk2,45m_chunk3,60m_chunk4}/`
  and matching`pv2_clock251_*` folders; summaries`ems_feedback_20261005_pv2_clock{25,251}_60m_chain.json`.
  Transitional`pv_clock*` files remain separate sensitivities; do not mix checkpoint contracts.
- Retrospective stress selection; initial stock seeded from changing BMS ratio, not independent
  metrology. Delivered PV remains a lower bound; HWC thermal schedule exogenous; instantaneous
  modeled activation and fixed-efficiency stock; no ramp/transient PCS/device completion model.
  These uncertainties dominate the sub-cent terminal-value margins.

## Reproduction

```bash
./.venv/bin/python eval/dh_feedback_replay.py \
  --history data/energy_replay/dh_source_history_20261004_export \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --start 2026-09-30T22:00:00Z --end 2026-09-30T22:15:00Z \
  --experiment ems_timing \
  --ems-history data/energy_replay/ems_control_history_20261005_export_extended \
  --dc-fixed-loss-w 140 --pv-export --timer-clock-seconds 25 \
  --reuse-solves data/energy_replay/ems_feedback_20261005_export_15m_140w \
  --cached-only --output data/energy_replay/NEW_PV_CONTROLLER_15M
```

- Continue22:15–22:30 with verified own checkpoint and exact earlier solve cache. Later origins
  use `dh_source_history_20261004_export_extended`; new optimizer requests require container.
- All outputs ignored/private under `data/energy_replay/`; no raw data committed.
