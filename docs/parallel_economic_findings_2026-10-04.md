# Parallel economic evaluation — 2026-10-04

- Status: observed accounting + isolated load/terminal sensitivities + APF archive pilot.
  Production unchanged. Six bounded core solves; all Optimal, physical/result checks pass.
- Effort order: conditional inventory/terminal policy and longer-horizon load calibration;
  cheap incumbent tail residual next. No large-model training. Rankings remain hypotheses.
- [Model/path assessment](price_parallel_paths_2026-10-04.md);
  [measured targets/calibration](measured_economic_actuals_2026-10-04.md).
- [Sequential MPC pilot](sequential_economic_replay_2026-10-04.md) now completed: 24 solves,
  near-zero measured-target cashflow difference; baseline trajectory mismatch still open.

## Observed variable cashflow: UTC Sept 27–Oct 4

- `eval/export_amber_quote_actuals.py`: raw general/feed/adjusted-confirmed revision archive.
  `eval/amber_quote_actuals.py`: quoted interval end rather than recorder timestamp; unit,
  CurrentInterval/non-estimated status, five-minute convention, latest conflicting revisions,
  adjustment against contemporaneous and latest raw feed quote validated.
- 2014/2016 rate intervals per leg; 2013 complete energy/rate intersections (99.85%).
- Raw import cost **−$2.011**, net export cost **−$11.315**: variable credit **$13.326**.
  Excludes fixed charges, wear, gaps and inventory valuation; not invoice reconciliation,
  net system profit or counterfactual improvement.
- Gross export revenue **$11.321**; paid-export cost **$0.0056**, 0.410 kWh.
  Negative-export execution leakage is small here; no evidence to prioritise a curtailment
  rewrite over load/inventory experiments from this week.
- Recorded local allowance adjustment increases credit by **$0.213**, to **$13.540**.
  Scenario only; free-export threshold/billing semantics require reconciliation.
- CurrentInterval estimate=false retained before interval end. Optional post-end-only
  sensitivity leaves only two intervals after rollover filtering; not a validated truth filter.
  [Amber API staff semantics](https://github.com/amberelectric/public-api/discussions/214).
- Authoritative ignored evidence: `data/energy_replay/amber_observed_week_20261004_v6/`.
  Earlier v1–v4 accounting summaries used the wrong feed-state sign; superseded. Raw quote
  rows remain usable and hash-verified. v5 fixed sign but retained mixed state/attribute rollover
  rows; v6 excludes 1797 identified transitions per leg. Credit changes by **+$1.522**; no new
  network data. [Rollover evidence and sequential re-scoring](sequential_economic_replay_2026-10-04.md).

## Installed feed-state contract — verified Oct 4

- HA registry: current general/feed and forecast entities belong to MQTT.
- Installed `amber2mqtt` image:
  `sha256:5f29064be5fa497ac206587c15e0a91b50556777404ca2b6b76d70bed7bdaaf2`;
  registry digest `788413da1988ce954309e3cd03771ad63324e8d818de9d9143bbe941e39a9fb6`.
- Read-only `/app/mqttmessages.py`: current feed state converts `per_kwh * -1` cents→dollars
  (line 302); `Forecasts.per_kwh` retains API sign (lines 668, 741).
- **Current feed STATE positive earns; raw forecast ATTRIBUTE negative earns.** Unit alone
  does not identify the sign. Current MPC uses state directly, future tariffs negate attributes;
  export guard state <=0 clamp is consistent. No production sign bug established.
- Accounting draft conflated these signs; corrected before accepting results. Regression tests
  cover positive revenue and negative paid-export sensor states. Current publisher source does
  not independently prove historical configuration stability.

## Load-only core sensitivity: frozen Oct 3 01:51Z DH

- `eval/compare_load_solver_sensitivity.py`: baseline versus rebuilt p65 base-load challenger.
  Exact HA/log agreement on **44 overlapping targets**, one version/type; latest logged creation
  **01:32:11Z**. Version inferred from vector, not archived HA model metadata.
- Fitted at forecast creation with completed target +30-minute receipt lag; three-day history,
  48-target minimum. 143 released targets/band; latest target 00:30Z.
  Offsets **−11.9 / −124.1 / −134.3 / −266.2 W** across four horizon bands.
- Partial current interval unchanged. HWC, fixed losses, PV, prices, initial/exact terminal SoC,
  plant/source image fixed; synthetic provenance retained.
- Predicted 72h consumption **13.446 kWh lower**; import **23.696→19.601 kWh**,
  export **36.305→45.797 kWh**, forecast cashflow **$3.788→$5.075**.
- **+$1.287 is changed predicted consumption, not realised savings.** Ending inventory unchanged.
  Initial command unchanged (**7.771 kW charge**); later trajectory shifts by up to
  **7.77 SoC percentage points / 1.051 kW**.
- Interpretation: calibration influences later inventory/actions; value both controllers against
  the SAME measured load/PV/rates. Evidence: `load_solver_sensitivity_20261004/`.

## Fixed-forecast endpoint sensitivity

- `eval/compare_terminal_solver_sensitivity.py`: endpoint ±10 percentage points; all forecast
  arrays, initial state and constraints fixed. Ending energy changes **±4.03 kWh**.
- Break-even value: `cashflow_delta + value_per_kwh * ending_inventory_delta = 0`.
  Excludes wear/stress and horizon risk; cashflow alone is not a fair ranking.

| Layer / baseline endpoint | Perturbation | Forecast cashflow delta | Ending energy delta | Break-even inventory value |
|---|---:|---:|---:|---:|
| DH / 83.71% | −10pp | +$0.312 | −4.03 kWh | 7.74 c/kWh |
| DH / 83.71% | +10pp | −$0.397 | +4.03 kWh | 9.84 c/kWh |
| MPC / 83.82% | −10pp | +$0.395 | −4.03 kWh | 9.80 c/kWh |
| MPC / 83.82% | +10pp | −$1.025 | +4.03 kWh | 25.44 c/kWh |

- All first commands unchanged. MPC changes 52/56 later battery intervals; DH 10/13.
- Higher MPC endpoint requires extra forecast imports; retention cost is asymmetric.
  Supports regime-based terminal experiments; no justification to lower live endpoint yet.
- Evidence: ignored `dh_terminal_sensitivity_20261004/`, `mpc_terminal_sensitivity_20261004/`.

## Tactical APF archive availability

- `eval/audit_amber_forecast_archive.py`: first 12/final revisions per extended leg; ISO intervals,
  finite prices, sign-specific bounds, contiguous durations. As-of pairing rejects future/stale/
  conflicting receipts. This is an archive feasibility pilot, not a complete replay.
- Week: **3781 general / 3774 feed** revisions. Inspected first/final payloads each carry
  **216 five-minute intervals (18h)**; 23 bounded as-of pairs.
- Inspected revisions cover the MPC's 14h requirement. DH billing source/dynamic handoff cutoff
  need their own audit; 18h does not prove all-origin coverage or justify changing tail splice.
- Separate legs are not atomic. Influx event time is not proven HA availability; weights,
  allowance, accepted DH/PV/HWC/control state not reconstructed.
- Evidence: ignored `amber_apf_archive_20261004/` under `data/energy_replay/`.

## Next experiment / gates

1. Selected origins across import-heavy/export-heavy/cloudy days: full as-issued MPC APF,
   DH billing curve, PV/HWC, risk weights, tariff adjustment and endpoint policy state.
2. Matched sequential DH→MPC: baseline/calibrated load, then fixed-forecast terminal policy;
   identical actual targets, physical constraints, execution and evaluation inventory treatment.
3. Realised variable bill, wear/throughput, ending inventory, curtailment, day/event attribution.
   Delivered PV is not counterfactual available PV; gaps cannot become zero.
4. Rank deployable gains only then; investigate cheap causal tail residual before new models.

## Commands

```bash
./.venv/bin/python eval/export_amber_quote_actuals.py \
  --dataset data/energy_replay/measured_week_20261004 \
  --output data/energy_replay/new_observed_accounting
./.venv/bin/python eval/audit_amber_forecast_archive.py \
  --start 2026-09-27T00:00:00Z --end 2026-10-04T00:00:00Z \
  --max-revisions 12 --output data/energy_replay/new_apf_pilot
./.venv/bin/python eval/compare_terminal_solver_sensitivity.py \
  --baseline data/energy_replay/mpc_solve_20261004.json \
  --terminal-soc-pct 73.82 --terminal-soc-pct 93.82 \
  --output data/energy_replay/new_terminal_sensitivity
```

Load sensitivity arguments: recorded `--journal`, credential-free `--config`, prior
`--calibration` directory, immutable `--image`, pinned `--optimization-sha256`, new `--output`.
All outputs ignored/private, no overwrite; bounded Docker core uses no network/live mounts.
