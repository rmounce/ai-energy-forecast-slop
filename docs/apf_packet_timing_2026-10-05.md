# APF packet timing diagnostic — 2026-10-05

- Follow-up [predeclared spaced causal correction](apf_causal_correction_2026-10-05.md):28
  receipts/146 paired cases, one changed choice, no broad benefit. Current APF retained; reserve-
  constrained empirical load-calibration comparison next.

- Parallel forecast effort, alongside [controlled cycle mechanisms](controlled_cycle_value_2026-10-05.md).
  APF retained; no optimizer, network, training, service or device calls.
- `eval/apf_packet_regret.py`: at each frozen APF feed receipt, choose one fully future export
  interval or keep packet. Compare predicted/conservative APF choices with hindsight over same slots.
- Hypothetical0.25kWh stored packet →0.2475kWh DC discharge →0.235125kWh AC export.
  Requires2.8215kW spare inverter/grid headroom for5min; bigger1kWh packet fails9.98kW bound.
  Site load/PV/constraints not reconstructed. Normalized cents/kWh cannot scale to arbitrary stock.
- Wear4c per discharged DC kWh; terminal stock value0 or20c per stored kWh. Withhold incremental
  value0; export value=AC×feed−DC throughput×wear−stored packet×terminal value.
- Raw API feed forecast sign negated into export revenue; pessimistic bound negates APF low.
  Confirmed quote manifest/sign/hash verified; raw canonical rates reproduce saved parquet exactly.
  Non-estimated CurrentInterval quotes are not invoice-reconciled settlement; allowance excluded.

## Coverage

- Existing initial APF archive plus three frozen DH/control histories; complete hash/source lineage.
 98 eligible receipts across four clustered windows: Sep27(12),Sep28(19),Sep30(30),Oct1(37).
- Each receipt evaluated6/12/14h:294 cases per terminal-value profile. Final Oct3 receipt
  excluded for all3 horizons because confirmed future quotes incomplete. No missing-label filling.
- Current/partial intervals excluded; all candidate intervals finish inside horizon. APF30min
  intervals can expand to5min price slots; these are not independent market samples.
- Overlapping receipts/horizons strongly dependent. Means below describe each cluster; **do not
  sum regrets or treat98 receipts as98 independent trading opportunities**. No weekly savings claim.

## Conditional timing regret

Mean14h regret, cents per hypothetical stored kWh;4c/DC-kWh wear common:

| Receipt cluster | Predicted APF, terminal0c | Conservative APF, terminal0c | Predicted APF, terminal20c | Conservative APF, terminal20c |
|---|---:|---:|---:|---:|
| Sep27 |0.9240 |1.7206 |0 |0 |
| Sep28 |0.9242 |0.9222 |0 |0 |
| Sep30 |12.9325 |12.4162 |2.6928 |1.9971 |
| Oct1 |2.8337 |3.7851 |0 |0 |

- Example Sep30 22:05:20 UTC,14h,terminal0: predicted and conservative choose Oct1 12:00 UTC;
  hindsight chooses Sep30 22:10 UTC. Packet value2.1442c vs6.0685c; regret3.9242c per0.25kWh
  packet,15.6969c/kWh. APF future timing/rank error conditional on free stock and spare headroom.
- Terminal20c changes decisions: all Sep27/Sep28/Oct1 cases withhold, as does hindsight.
  Sep30 predicted exports at2 of30 origins for12/14h; those exports lose value relative to keeping
  packet. Conservative withholds throughout, but can miss profitable near-term opportunities.
- Example Sep30 22:20:23 UTC,12h,terminal20: predicted chooses Oct1 10:00 UTC, incremental
  value−2.6912c/packet; hindsight Sep30 23:00 yields+0.7534c. Conservative withholds (0c).
  Conservative bound is not uniformly superior: Sep27/Oct1 terminal0 regret increases.
- Existing recorded export pilot often hits inverter output limit (10/15 initial decisions);
  this packet could lack spare headroom at site. Oracle gap is neither deployable savings nor
  full-system perfect-price headroom, and does not show incumbent DH/MPC made this packet choice.

## Decision and verification

- Prioritize APF-relative timing/spread errors and energy reserve valuation over a heavier model
  architecture search. First add predeclared spaced origins across more days/regimes; then compare
  a simple causal APF residual/ranking correction. No wholesale-label substitution with retail quotes.
- Conservative price/terminal choices interact; test separately, avoid counting same risk twice.
  Load calibration still needs inventory-sensitive controlled information arms;40min zero gain
  alone does not rule out longer-horizon value.
- 25 new tests pass: raw sign, future/stale/missing receipts, complete labels, partial intervals,
  power bound, withholding, DC wear versus stored value, canonical/source mutation and conflicts.
 27 existing parser/quote tests pass; combined new cycle/packet tests51 pass.
- Private authoritative outputs under `data/energy_replay/`:
  `apf_packet_regret_20261005_multi_zero_terminal/report.json` and
  `apf_packet_regret_20261005_multi_20c_terminal/report.json`.
  Older `apf_packet_regret_20261005_zero_terminal` predates corrected wear basis; superseded.
  Agent's corrected `multicluster_zero_terminal` matches root's `multi_zero_terminal` assumptions.

```bash
nice -n 19 ./.venv/bin/python eval/apf_packet_regret.py \
  --archive data/energy_replay/amber_apf_archive_20261004 \
  --quotes data/energy_replay/amber_observed_week_20261004_v6 \
  --control-history data/energy_replay/dh_source_history_20261004_near_full \
    data/energy_replay/dh_source_history_20261004_export_extended \
    data/energy_replay/constraint_history_20261005_scarce \
  --terminal-value-aud-per-dc-kwh .20 --output data/energy_replay/NEW_APF_PACKET
```
