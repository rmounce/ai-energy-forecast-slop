# Controlled cycle value — 2026-10-05

- Follow-up [controlled load-information value](load_information_value_2026-10-05.md): correcting
  synthetic demand matters at scarce reserves, not ample/replenished cases; empirical calibration
  savings remain unproven. Common realised inputs, own stock, adverse-bias/oracle controls.

- Parallel follow-up to [cycle-input audit](cycle_pv_support_2026-10-05.md); economic mechanism
  checks, not historical savings. APF/production unchanged; no optimizer, network or device calls.
- `eval/controlled_cycle_scenarios.py` reuses existing `execute_ems`/physical executor.
  Same initial stock, fixed shared PV/load/rate curves; prescribed extra2kW export for15min,
  then common self-consumption. Five-minute physical steps; no endpoint resets/re-grounding.
- Sixteen cases: six solar-recovery fractions×two export constraints, four early-feed prices
  in scarce later-demand case. Terminal values0/20/30/40c and wear0/4c are sensitivities.
- 40.3kWh capacity;99% battery efficiencies,95% inverter;140W DC overhead; synthetic15% floor.
  Solar cases start95% SoC; scarce case20%. These are explicit scenario choices, not site history.

## Default mechanism results

- Extra early export earns10c, spends0.531632kWh modeled inventory, adds0.526316kWh DC discharge.
  Illustrative20c/kWh ending energy and4c/DC-kWh wear below; both arms share exogenous inputs.

| Later outcome | Ending stock difference kWh | Extra DC throughput kWh | Net difference cents |
|---|---:|---:|---:|
| No replacement of extra spent energy | −0.531632 | +0.526316 | −2.7379 |
|50% replaced from otherwise-curtailed solar | −0.265816 | +0.794817 | +1.5044 |
|100% replaced from otherwise-curtailed solar | 0 | +1.063318 | +5.7467 |
|100% replaced from solar otherwise exported at5c/kWh | 0 | +1.063318 | +3.1960 |
| Needed later:20c early export,40c later import | 0 | ≈0 | −10.1362 |

- Free replacement requires **otherwise curtailed** solar: baseline reaches capacity and
  extra-export arm absorbs more of same PV. Both reach full through physical charging.
- Exportable solar sacrifices2.550760c of baseline later export; it is not free replacement.
  No charge/discharge wear omitted: full-refill extra throughput includes both legs.
- Scarce case: both physically reach15% floor. Early spending creates later imports. Each arm
  eventually discharges same stored energy, so incremental wear cancels; do not subtract early
  discharge wear again. Terminal value cancels because ending stock matches.
- Equal early/later prices give a small negative result with140W model overhead; zero-overhead
  oracle gives zero. Existing ideal executor routes negative DC power through inverter conversion
  at empty floor. Model sensitivity, not measured site standby/efficiency cost.

## Checks and interpretation

- 26 tests pass: analytic zero-overhead energy/cash identities, export opportunity cost,
  scarcity/throughput cancellation, capacity/floor binding, partial recovery and invalid assumptions.
- Independent accounting review: no blockers. Full-cycle stock closure is physical, not a reset.
- Net = cash difference + ending-stock difference×terminal value − throughput difference×wear.
  Valuations illustrative; no fitted degradation or measured opportunity value.
- The timing change has value only conditionally. Historical hour alone did not establish it;
  this suite establishes signs/mechanisms under controlled inputs, not a deployable policy.
- Next forecast effort: price timing regret with APF retained, then controlled load-information
  arms only where inventory/scarcity creates headroom. Stop expanding historical minute fidelity
  for its own sake. Require broad causal matched evidence before promotion.
- [APF packet diagnostic](apf_packet_timing_2026-10-05.md) now complete across four clustered
  windows; next spaced sampling and causal timing corrections before more complex price models.
- Private ignored result `data/energy_replay/controlled_cycle_scenarios_20261005.json`, including
  plant/phase choices, binding-step counts, both-arm flows and executor/code hashes.

```bash
nice -n 19 ./.venv/bin/python eval/controlled_cycle_scenarios.py \
  --output data/energy_replay/NEW_CONTROLLED_CYCLE.json
```
