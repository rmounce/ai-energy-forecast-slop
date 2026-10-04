# Own-battery DH feedback replay — 2026-10-04

- `eval/dh_feedback_replay.py`: bounded paired DH/MPC event replay. Each arm owns battery SoC,
  DH load/PV/SoC trajectory, DH anchor, reground block and target offset after one initial seed.
- Exact source-grid admission rejects rollover load lags. Rejection retains the accepted parent
  and helpers; fresh aligned source triggers the next archived DH decision proxy.
- Chronological physics between every decision and activation; prior command persists through
  delay. DH/MPC activation uses archived publication proxies. Future measured targets score
  execution only; own SoC replaces historical SoC in every request.
- DH result/formatter validated before coherent parent/helper acceptance. Offset update modeled
  at acceptance; later DH uses that arm's offset. This is a modeled admission/transaction policy,
  not an exact reproduction of non-atomic HA helper writes.
- HWC schedule/thermal state remain **exogenous**. Full HWC planner feedback is not implemented.
- APF revisions parsed from source archive; quote history reused from verified v6 replay bundle.
  Settings restored as-of each origin, with explicit older unchanged-state capture fallback.
  MPC receipt diagnostics exclude unused archived DH parents; own request ID supplies lineage.

## Bounded core evidence

Same pinned installed EMHASS image/optimizer/formatter as preceding replay. Two 15-minute
subwindows of retrospective stress cases; 15 MPC origins and 3 accepted DH origins per arm,
36 core solves/window. Each window rejects one misaligned DH origin, then admits its fresh successor.

| Metric | High-price export | Near-full approach |
|---|---|---|
| UTC selected start/end | Sep 30 22:00–22:15 | Sep 28 04:50–05:05 |
| Baseline export kWh | 1.85954 | ≈0 |
| Baseline import kWh | 0 | 0.002698 |
| Baseline variable cash credit AUD | 0.504014 | −0.000116 |
| Baseline final SoC | 73.13915% | 98.49629% |
| Measured final SoC | 73.31% | 98.48% |
| Ending inventory minus measured | −0.06885 kWh | +0.00657 kWh |
| Baseline requested-vs-published battery MAE | 0.00341 W | 0.00214 W |
| Terminal targets changed without positive lock-in | 0 / 15 | 1 / 15; max 0.03 percentage points |
| Challenger cashflow / inventory difference | $0 / 0 kWh | $0 / 0 kWh |
| Summed core solve time | 31.67 s | 42.44 s |

- Near-full pilot approaches ceiling; the first 15 minutes do **not** reach 100%. Full-battery
  curtailment/recovery evidence remains a separate gate. No counterfactual available-PV reconstruction.
- Export observed cash credit $0.466001, export 1.72261 kWh. Ideal execution exports 0.13694 kWh
  more. Model DC discharge 2.01115 kWh vs raw measured net DC discharge 1.88965 kWh.
  Nearly identical planned commands do not prove identical device delivery; this gap is not savings.
- Both pilots show no economic reason to remove positive terminal lock-in. These are short
  retrospective cases, not a general proof that terminal policy cannot matter.
- Output: ignored `data/energy_replay/dh_feedback_replay_20261004_export_15m_v2/` and
  `dh_feedback_replay_20261004_near_full_15m/`; bundle, exact requests/results, staged code hashes.
  Earlier export `...export_15m` superseded by v2 receipt/seed provenance; economics identical.

## Confirmed capacity contract

- Oct 4 frozen base EMHASS config reports **420,000 Wh**; runtime HA capacity/health override is
  **40,300 Wh**. Base config is not an execution capacity. Initial executor freezes runtime capacity
  and MPC floor before first activation; later MPC execution uses validated effective request plant.
- Fail if capacity changes during the bounded replay. Initial pilot stopped at this guard before
  emitting results; corrected completed reports use 40.3 kWh throughout. No live config change.

## Verification and commands

```bash
./.venv/bin/python eval/dh_feedback_replay.py \
  --history data/energy_replay/dh_source_history_20261004_export \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --start 2026-09-30T22:00:00Z --end 2026-09-30T22:15:00Z \
  --output data/energy_replay/NEW_FEEDBACK_REPLAY
```

- Network-free read-only disposable container: 1 CPU, 2 GiB, 150 s worker / 180 s client budget.
  ≤15 MPC origins, ≤8 DH origins; new output directory required. No HA/control/publication writes.
- Prior minute replay exact saved-request re-audit still passes after source override support:
  `minute_policy_replay_20261004_30m_v4`; economics unchanged, no extra core solves.
- 321 affected tests passed; final 8 feedback tests pass after adding explicit diverging-arm
  coverage (322 distinct tests total). Existing Influx datetime deprecation warning only.
- Tests cover retained parent, delayed coherent acceptance, own inventory/parent isolation,
  historical SoC exclusion, future price exclusion, changed admitted sources, exact timeline
  splits, runtime capacity vs placeholder config, and diverging-arm DH feedback.

## Next effort

1. Replay the cheap causal p65 load/net-energy calibration through own DH/MPC feedback, paired
   with unchanged forecasts/settings and frozen targets; score cash and terminal inventory separately.
2. Diagnose export requested-vs-delivered DC power and grid balance; extend full-battery window
   only after confirming measured-PV/device-mode interpretation. Do not promote modeled cash gap.
3. APF tail/residual price corrections after matching source availability and net-energy accuracy;
   cheap baselines before a new complex price model. HWC thermal feedback needs its own state model.
