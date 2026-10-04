# Inventory cycles and low-stock replay — 2026-10-05

- Offline; APF retained; no production/device/service changes or training.
- Next economic gate after one-hour timing result: complete battery excursions and scarce/full stock.
- `eval/screen_inventory_cycles.py`: hash-verified measured archive; full≥99.5%, material
  departure≤95%, scarce screening≤15%. Interval-start labels; SoC observations at interval end.
  Full-to-full windows start after last observed full sample before departure, end at first return.
  Missing SoC retained in coverage counts; missing timestamps/SoC split scarce episodes.
  Unsigned energy totals null if any interval missing; no night-zero filling.
- Retrospective selection; not achievable savings, equal counterfactual inventory or admission.

## Saved week: Sep27–Oct4 UTC

| Full-to-full observed excursion UTC | Duration h | Minimum observed SoC% | Finite PV intervals / expected |
|---|---:|---:|---:|
| Sep27 08:40–Sep28 05:15 | 20.583 | 41.34 | 117 / 247 |
| Sep28 06:40–Sep29 05:55 | 23.250 | 43.56 | 149 / 279 |
| Sep29 08:55–Sep30 03:35 | 18.667 | 79.85 | 92 / 224 |
| Sep30 08:15–Oct3 05:25 | 69.167 | 13.24 | 444 / 830 |

- All four incomplete delivered-PV coverage. Longest also lacks2 load/grid/battery intervals
  and30 SoC endpoints. No verified full-cycle timing economics from these records yet.
- No observed endpoint≤10% in week.10% reference base-config floor is not runtime constraint:
  payload raises floor with current SoC/buffer; do not equate15% screening with binding.
- Complete-power low-stock episodes include Oct1 22:50–23:15 UTC (13.24–13.40%) and
  Oct2 21:20–22:40 UTC (14.32–14.92%). Selection does not guarantee forecast/controller lineage.
- Full-battery controller needs `input_datetime.battery_last_reached_full` and8h holdoff;
  own counterfactual transitions must update helper, not reuse future site full events.
  Proactive-curtailment/charging branches also need explicit inputs before admission.

## New bounded archive and low-stock comparison

- Read-only Oct1 21:15–22:45 UTC archive,90min; DH/MPC/APF/controller and AC/DC/BMS inputs.
- 90 MPC publications:58 fresh helper clocks,32 missing; only4 missing supported by both
  retrospective25.0/25.1s candidates. No broad inference or source-age relaxation.
- Oct1 21:20:21.808729–21:35 UTC:15 MPC origins, all observed capture clocks; initial
  SoC15.06%, initial runtime floor13.06%.4 DH origins:3 aligned,1 rejected misaligned load.
- 36 fresh validated DH/MPC requests; common forecasts/terminal policy,140W DC overhead;
  own-state command-hold comparison. **Cashflow and ending inventory delta both zero.**
- Both arms: variable credit0.205151c, export0.021716kWh, DC throughput0.025392kWh,
  final SoC15.002234%, ending modeled stock6.045900kWh. Model remains above initial floor;
  this is low-stock evidence, not demonstrated binding-floor benefit. Distinct owned parent
  revisions can coexist with identical physical/economic outcomes. No promotion evidence.
- Raw75min reconciliation21:30–22:45: DC battery ledger−0.478666kWh versus nominal SoC
  proxy−0.499720kWh; difference0.021054kWh. Available-discharge change−0.530000kWh.
 140W/95% AC model residual−0.017054kWh versus+0.149196kWh without overhead.
  Separate low-stock consistency evidence; not parameter fitting or independent metrology.

## Validation, evidence and next gate

- 8 new screening tests pass: endpoint conventions, missing energy/timestamps/SoC,
  impossible physical values and ordering.7 existing energy reconciliation tests pass.
- Saved low-stock checkpoint reproduces all requests, events, cashflow and owned inventory;
  report SHA256 `d6403b4033fd59800868a0577d177f8693700dae3ce5a7d7f0867626ebdb8b7a`.
- Private ignored artifacts under `data/energy_replay/`:
  `inventory_cycle_screen_20261005_v2.json`, `constraint_history_20261005_scarce/`,
  `constraint_clock_audit_20261005_scarce.json`, `constraint_energy_20261005_scarce.json`,
  `ems_feedback_20261005_scarce_15m/`.
- Prefer20.6h full-to-full excursion for first cycle test: complete load/grid/battery support,
  substantial59 percentage-point dip, shorter than69h scarce cycle. Audit missing PV against raw
  strings/inverter support; retain daytime uncertainty, do not classify every missing night reading
  as zero. Audit full/holdoff branch inputs and clocks before admission.
- If historical inputs cannot support cycle, use explicit controlled constraint scenarios for
  mechanisms; separate from recorded-site savings claims. Thousands of minute solves cannot
  compensate for unavailable evidence. Full-to-full site endpoints do not guarantee equal
  counterfactual ending inventory; terminal-value sensitivity remains necessary.

```bash
./.venv/bin/python eval/screen_inventory_cycles.py \
  --dataset data/energy_replay/measured_week_20261004 \
  --output data/energy_replay/NEW_INVENTORY_SCREEN.json
```
