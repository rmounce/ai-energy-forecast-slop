# Causal load calibration through own DH feedback — 2026-10-04

- Decision: retain rolling p65 residual calibration as an offline challenger. Better load MAE
  does not establish economic gain; these two 15-minute pilots have effectively zero cash/inventory
  difference. Continued40min export replay also has zero gain; next controller fidelity/coherent
  solve clocks and inventory-constrained execution, with quantile loss.
- `dh_feedback_replay.py --experiment load_calibration`: both arms retain incumbent terminal
  policy. Challenger changes admitted DH **base load only**, then regenerates its own DH/MPC
  trajectory/anchor/reground/offset/inventory. HWC, PV source, prices, settings and scoring targets common.
- Common initial archived DH parent; correction begins at first admitted DH refresh. Rejected
  rollover update retains each arm's accepted parent. This is not an already-calibrated warm start.
- `prepare_load_feedback.py` reuses `compare_load_solver_sensitivity.calibrated_handoff`;
  no duplicate fitting method. Exact overlapping p65 vector identifies August 10 model/version,
  `simple` prediction type. Fit at logged forecast creation, hold constant for that issued vintage.
- Defaults unchanged: 3-day trailing release window, ≥48 independent targets/band, residual p65,
  complete 30-minute base-load target plus assumed 30-minute availability lag. Sparse band → zero
  correction. This assumption is not historical measurement-receipt evidence.

## Source-clock evidence

- Frozen source events vs logged creation labels: Sep 30 22:02:05.842513 / .864411 UTC
  (**21.898 ms**); Sep 28 04:31:31.453072 / .595947 (**142.875 ms**);
  Sep 28 05:02:07.304491 / .327079 (**22.588 ms**).
- Arrays agree within float serialization precision; 142–143 causal overlapping targets/vintage.
  Logger labels can follow Influx sensor-event timestamps. These clocks are not proven atomic
  generation/receipt times. Exact match alone does not identify model version in HA attributes.
- Admission bounds logged creation by **both DH origin and receipt +2 s**. Matching future-DH
  vectors remain excluded. Recorded gap/tolerance accompany every correction; no receipt renewal.
- Past training labels are selected at logged creation. Partial current interval excluded from
  fitting/scoring; its base prediction stays unchanged. No future target labels in solver inputs.

## Core and projected-plan evidence

Same two stress subwindows/pinned core as [own-feedback pilots](dh_feedback_economic_replay_2026-10-04.md).
36 core solves/window, one rejected DH origin and three accepted DH updates per arm. 72 solves total.

| Metric | Export Sep 30 22:00–22:15 UTC | Near-full approach Sep 28 04:50–05:05 UTC |
|---|---|---|
| Training targets/band | 143 | 56 then 57 |
| Next-14h base forecast energy change | −0.48215 kWh | −0.70456 then −0.62937 kWh |
| Largest projected DH SoC path change | 4.87 percentage points | 4.79 percentage points |
| Largest projected battery path change | 1,050.53 W | 1,050.53 W |
| Largest MPC terminal target change | 0.38 percentage points | 2.55 percentage points |
| First battery commands changed | 0 / 15 | 0 / 15 (floating precision only) |
| Cashflow / ending inventory difference | $0 / 0 kWh | ≈$0 / 0 kWh |
| Baseline first inverter at AC output limit | 10 / 15 | 0 / 15 |
| Baseline first planned net grid flow zero | 0 / 15 | 15 / 15 |
| Final target offset baseline / challenger | −2.67% / +0.76% | −22.63% / −26.41% |
| Summed core solve time | 32.25 s | 41.76 s |

- Export first accepted DH battery-path divergence starts at 22:30 UTC; its MPC projected
  divergence can start at 22:20. Both extend beyond this execution window. Near-full DH projected
  divergence starts at 07:30, 08:00, then 07:00 as plans refresh.
- Projected divergence is a follow-up horizon diagnostic, not a guaranteed later action change:
  MPC can replan away a projected difference. Near-full MPC projected differences inside the
  pilot also disappear at subsequent current decisions.
- Interpretation: current dispatch is insensitive to these corrections in these samples; ten
  export decisions already hit the inverter limit and near-full decisions balance live supply/load.
  Does not show that load forecasts have no value when inventory/reserve becomes binding.
- Near-full first-half window does not reach 100%; available-PV reconstruction/full curtailment
  still unproven. Export model-vs-device gap remains as documented in preceding pilot.

## Posthoc forecast skill

`audit_feedback_sensitivity.py --dataset` scores each received forecast **once**, complete
future measured base-load half-hours only. Scoring never enters correction fitting or dispatch.

| Issued vector | Horizon | Paired targets | MAE baseline → calibrated W | p65 pinball baseline → calibrated W |
|---|---|---|---|---|
| Export 22:02 | 0–6h | 12 | 163.68 → 159.40 | 58.60 → 57.23 |
| Export 22:02 | 6–16.5h | 21 | 230.70 → 170.53 | 80.75 → 59.68 |
| Export 22:02 | 16.5–36h | 39 | 229.68 → 189.78 | 80.98 → 69.02 |
| Export 22:02 | 36–72h | 71 | 184.15 → 145.97 | 71.34 → 66.34 |
| Near-full 04:31 | 6–16.5h | 21 | 124.17 → 83.47 | 43.78 → 35.10 |
| Near-full 04:31 | 16.5–36h | 39 | 334.99 → 325.29 | **172.72 → 179.73** |
| Near-full 05:02 | 6–16.5h | 21 | 134.04 → 86.39 | 47.46 → 34.00 |
| Near-full 05:02 | 16.5–36h | 39 | 347.47 → 337.62 | **175.17 → 181.85** |

- Forecast MAE mostly improves; p65 loss worsens at 16.5–36h in both near-full vintages.
  MAE alone is insufficient to choose a risk quantile. Near-full fit has only ~one day of released
  labels; do not tune a new shrinkage/guard using these already-inspected future targets.
- Overlapping horizons/vintages dependent; retrospective stress selection, not seasonal holdout.
  This work calibrates one DH p65 input, not a deployable coherent multi-quantile bundle.

## Reproduction, validation and next step

```bash
./.venv/bin/python eval/dh_feedback_replay.py \
  --history data/energy_replay/dh_source_history_20261004_export \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --start 2026-09-30T22:00:00Z --end 2026-09-30T22:15:00Z \
  --experiment load_calibration \
  --calibration data/energy_replay/measured_week_calibration_20261004 \
  --dataset data/energy_replay/measured_week_20261004 \
  --output data/energy_replay/NEW_LOAD_FEEDBACK
./.venv/bin/python eval/audit_feedback_sensitivity.py \
  --replay data/energy_replay/dh_load_feedback_20261004_export_15m \
  --dataset data/energy_replay/measured_week_20261004 \
  --output data/energy_replay/NEW_LOAD_AUDIT
```

- Ignored authoritative outputs: `dh_load_feedback_20261004_{export,near_full}_15m/`;
  `dh_load_sensitivity_20261004_{export,near_full}_15m_scored/`. Earlier unscored audit retained
  only as intermediate. All under `data/energy_replay/`; private data never committed.
- Verify calibration artifact, measured manifest/parquet and original source/replay hashes;
  exact requests/results, staged code hashes, receipts and fit metadata saved.
- 330 affected tests +7 sensitivity/scoring tests pass (337 distinct); existing Influx datetime
  deprecation warning only. Tested receipt/creation cutoffs, exact vector lineage, held per-vintage
  correction, future-target exclusion, unchanged terminal policy and isolated own-DH feedback;
  paired request/result checks, target completeness and no repeated-vintage score inflation.
- [Continuation completed](continued_feedback_and_delivery_2026-10-04.md): own state/plan/command
  carried through40min/96 export solves, past original projected divergences; no cash/inventory gain.
  [Recorded controller audit](ems_delivery_fidelity_2026-10-04.md) improves grid fit; inventory
  remains a proxy. [Energy/timing pilot completed](ems_energy_reconciliation_2026-10-05.md);
  next own-state solver/controller integration and coherent clocks, then inventory-constrained periods.
  Diagnose requested-vs-delivered DC power in parallel with longer replay. Compare cashflow,
  throughput, curtailment and ending inventory; keep APF unchanged for this load experiment.
- No production changes, publication, device writes or new model training.
