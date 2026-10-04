# EMS energy reconciliation — 2026-10-05

- Follow-up: [own optimizer timing feedback](ems_timing_feedback_2026-10-05.md),30min/72 solves:
  extra4.96c spends0.198kWh; no demonstrated net gain. Independent stock uncertainty remains.

- Decision: carry DC energy as model stock; label `recorded SoC × rated capacity` an inventory
  proxy. Do not promote a timing/forecast policy on extra export credit alone. APF retained.
- Two completed paths: raw AC/DC/BMS reconciliation; [bounded fallback timing challenger](ems_fallback_challenger_2026-10-05.md).
  No production changes, model training, HA reload, service call or device write.
- `export_control_history.py --include-energy-balance`: inverter AC, PV1/2, available charging/
  discharging capacities, derived total capacity, BMS SoC, rated capacity and health.
  Fixed read-only raw queries;75min archive21:30–22:45 UTC Sep30. Existing EMS sources included.
- `reconcile_ems_energy.py`: exact-boundary power integrals, signed charge/discharge ledgers,
  ≤120s raw holds, source-specific endpoint clocks, capacity-ratio identity. Missing support
  rejects; no interpolation, later SoC reset or loss-parameter fitting.

## Confirmed installed SoC semantics

Read `/opt/dockerfiles/hass/config/configuration.yaml` Oct5, template lines120–125:

- Derived SoC =100 × available_max_discharging_capacity /
  (available_max_charging_capacity + available_max_discharging_capacity), rounded2 decimals.
- `Sigen Plant Battery Capactiy Derived` preserves the installed typo in entity ID; reports
  charging+discharging capacity, rounded2 decimals. These quantities are BMS estimates.
- EMHASS uses rated capacity × health: historical40.3kWh ×100%=40.3kWh. Its initial SoC is
  the *different* derived ratio above. No independently measured physical-inventory truth.
- Read-only installed formula is not proof of unchanged Sep30 configuration. Raw archived
  available-capacity ratios agree with observed SoC to its rounding/asynchronous receipt precision.

| Endpoint UTC Sep30 | Charging kWh | Discharging kWh | Sum kWh | Derived SoC% |
|---|---:|---:|---:|---:|
| 21:30 | 8.34 | 33.53 | 41.87 | 80.08 |
| 22:00:23.787881 | 9.14 | 32.74 | 41.88 | 78.18 |
| 22:15 | 11.00 | 30.21 | 41.21 | 73.31 |
| 22:40 | 11.85 | 30.10 | 41.95 | 71.75 |

- Sum moves0.67kWh down, then0.74kWh up while nominal rating stays40.3. Do not replace
  nominal capacity with a fitted constant41.x from one selected window.
- Identity D=C×s: ΔD = C_start×Δs + s_end×ΔC. At22:15, ΔD−2.53kWh decomposes into
  ratio−2.03884kWh and capacity-estimate−0.49116kWh. The latter is not battery discharge.
- Raw DC battery ledger with existing99% charge/discharge efficiencies changes−1.90874kWh
  in first pilot versus nominal SoC proxy−1.96261kWh: discrepancy+0.05387kWh.
  At22:40:−2.51556 versus−2.59129kWh, discrepancy+0.07573kWh.
  Available-discharge deltas differ still more (−2.53/−2.64kWh). None is certified metrology.

## AC/DC accounting and existing fixed-loss assumption

- Independent raw grid + inverter AC − site-load energy: +0.00080/+0.00200kWh in first/continued
  pilot. PV1+PV2 agrees with gross-PV integral to floating/time serialization precision.
- Raw DC input − inverter AC:0.13945/0.23733kWh. Derived clipped-loss sensor integrates
  0.14047/0.23834kWh; it uses the same powers, so agreement is not independent efficiency truth.
- Existing DH/MPC payload logic subtracts140W overhead from PV first, adds remainder to load.
  Physical replay previously omitted explicit DC overhead. Add optional `dc_fixed_loss_w` to
  executor, default0 preserving old saved-run reproduction; separate140W ablation consumes
  DC before conversion, updates battery stock and enforces inverter/grid/SOC bounds coherently.
  Overhead is not PV curtailment. No extra grid-credit creation from adding battery consumption.

| Raw-power AC prediction residual kWh | Preceding21:30–22:00 | First pilot | Continued pilot |
|---|---:|---:|---:|
| 95% × DC − observed AC | 0.08180 | 0.04266 | 0.10450 |
| 95% × (DC −140W×duration) − observed AC | 0.01530 | 0.01029 | 0.01671 |

- Existing parameters only; no fitting to these selected outcomes. The preceding30min is an
  adjacent check, not a seasonal holdout. Mixed operating levels and asynchronous powers remain.
- Adding140W to recorded mode/limit replay reduces40min inventory-proxy error +0.14812→+0.06160kWh;
  grid-export error +0.03835→+0.03201kWh. Model DC discharge2.50468 versus measured2.49105kWh.
  Recorded variable credit model$0.60524 versus observed$0.59621. Delivery/SoC uncertainties
  still matter; recorded-control trace remains exogenous and unsuitable for ranking forecasts.
- Old own-feedback replay requests, commands and zero-gain calibration results reproduce with
  default0. Existing core/price artifacts remain frozen; no retroactive economic reclassification.

## Economic implication and next gate

- Timing challenger already carries its own modeled SoC and emits its own mode/limit commands;
  frozen incumbent MPC plans remain common. It does **not** feed changed SoC into future solves.
- 30min extra credit4.9745c consumes0.19876kWh extra inventory. Break-even ending-energy value
  ~25.03c/kWh before wear, ~21.07c with illustrative4c/kWh DC wear. Fixed140W barely changes
  the comparison. Larger cash receipts alone do not show a profitable policy change.
- Next: integrate supported controller commands/activation timing and fixed-loss execution
  into own-DH/MPC feedback; retain causal price/flexible/SOC guards. Resolve solve/republish
  clocks, support later PV-only branches, then extend to scarce-inventory and negative-price cases.
  Compare cash, DC throughput and ending model stock over full opportunities; report uncertainty
  against raw DC ledger and BMS proxies separately. Keep calibration offline meanwhile.

## Reproduction and validation

```bash
./.venv/bin/python eval/export_control_history.py \
  --start 2026-09-30T21:30:00Z --end 2026-09-30T22:45:00Z \
  --include-ems-inputs --include-energy-balance \
  --output data/energy_replay/NEW_ENERGY_ARCHIVE
./.venv/bin/python eval/reconcile_ems_energy.py \
  --history data/energy_replay/ems_energy_history_20261004_export \
  --start 2026-09-30T22:00:23.787881Z --end 2026-09-30T22:40:00Z \
  --output data/energy_replay/NEW_ENERGY_RECONCILIATION.json
```

- Authoritative ignored outputs: `ems_energy_reconciliation_20261005_{preceding,15m,40m}_final.json`,
  `ems_delivery_audit_20261005_export_40m_140w_final.json`; timing outputs in linked challenger doc.
  Raw archive/manifest, matched receipts and code/dependency hashes retained; private telemetry uncommitted.
- 108 affected tests pass. Exact raw-boundary/expiry/NaN/future exclusion; separate capacity and
  DC ledgers; physical overhead conservation/grid/inverter/SOC bounds; static guard provenance;
  causal guarded timing policy; default saved feedback reproduction. No statistical-profit gate passed.
