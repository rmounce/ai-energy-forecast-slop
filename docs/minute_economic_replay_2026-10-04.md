# Minute economic replay — 2026-10-04

- Status: paired 15m then30m MPC policy pilots completed; **90 installed core solves Optimal**.
  Same raw execution targets, receipt-gated APF/current prices, dynamic archived DH/HWC parents.
- Decision: no promotion of terminal-lock-in change. Short-window financial effect negligible;
  forecast/policy ranking still needs longer high-value windows and each arm's own DH feedback.
- Scope: recorded parent plans condition both arms; parent generation/HWC thermal feedback is
  not simulated. Tactical inventory evolves independently; no observed SoC re-grounding.

## Timing / physical contract

- `eval/minute_core_replay.py`: actual historical MPC helper events proxy decision clocks;
  power-plan publications proxy activation clocks. Previous simulated command remains active
  during recorded solve/publication delay. Wall-clock solver runtime does not alter replay time.
- Initial SoC: latest bounded observation at first decision. Seed preceding live power command
  once; initial curtailment assumed0, prior export cap from recorded mode context.
- Both policies get identical as-of telemetry, DH PV/load/SoC parent, anchor, HWC snapshot,
  APF revisions and current quotes. Each subsequent payload substitutes only its OWN simulated SoC.
  Parent anchors come from live history; this deliberately remains conditional, not own-DH feedback.
- Baseline: existing boundary-backdated MPC starting SoC + positive-deviation terminal lock-in.
  Challenger: same starting policy, forecasts, physics and objective; exact terminal target uses
  archived DH future SoC without positive-deviation addition. Endpoint clamps0–100% retained.
- Required measured-PV mode at every decision; binding/transition modes fail this pilot rather
  than infer available solar. Raw delivered DC PV remains a common supply lower bound.
- Execution: joint event grid of raw PV/load/grid/battery samples, hold expiries, decision,
  activation and tariff boundaries. Held targets expire120s; incomplete/negative/nonfinite
  physical targets fail. No interpolation or averaging of import/export sign changes.
- Existing DC/AC physical projection now accepts explicit subinterval seconds; SoC/power limits,
  clipping, curtailment and configured efficiencies retained. Limits enforced per segment.
- Current price guard carried with each activated command. Recorded raw grid/battery streams
  score live trace; raw confirmed quote rates score both simulated arms on the same segments.
- Scoring uses full frozen quote history, including later corrections; decision quote receipts
  remain gated at origin. Full-history versus truncated-window rates identical in this window.
- Selected code + credential-free bundle staged/hash-pinned before launch; shared `run_batch`,
  one disposable network-disabled/read-only container, 1CPU/2GB, nice19, 150s worker/180s client.
  Max30 origins/60 paired solves per job; no live HA/wrapper/control state touched.
- `--solve-archive`: exact bundle/request identity required; stored results revalidated. No hidden
  fallback to new decisions. Full30m run independently repeated offline with the same60 results.

## 15-minute pilot

- Oct3 **02:00:16.712–02:15:00 UTC**, 15 origins/30 solves;37.91s aggregate solver time.
- Initial SoC52.96%; final simulated57.956%, measured58.11%; inventory gap **−0.062kWh**.
- Baseline requested/live published command MAE **0.00098W** across15 origins, despite
  endogenous inventory. All first commands identical between policy arms.
- Five terminal targets changed (max2.15pp); ending inventory/cashflow exactly unchanged.
- Variable credit simulation$0.01648, observed$0.02103; grid import1.287 versus1.302kWh.
  Model/actuation/telemetry differences remain; comparison is not a saving against live operation.
- Evidence ignored: `data/energy_replay/minute_policy_replay_20261004/`.

## 30-minute paired result

- Oct3 **02:00:16.712–02:30:00 UTC**, 30 origins/60 solves;73.29s aggregate solver time.
- Same initial inventory52.96%; no resets or observed SoC injections after start.

| Metric | Baseline | Without positive lock-in | Observed live trace |
|---|---:|---:|---:|
| Variable credit | $0.086679 | $0.086590 | $0.094789 |
| Grid import | 3.693kWh | 3.710kWh | 3.756kWh |
| Grid export | ~0 | ~0 | 0 |
| DC battery throughput | 4.723kWh | 4.740kWh | — |
| Ending SoC | 64.563% | 64.603% | 64.830% |
| Ending inventory | 26.019kWh | 26.035kWh | 26.126kWh |

- Baseline command MAE **19.28W** against30 live publications; **29/30** differ<1W.
  Only02:20 differs materially: baseline charges578W less than published plan. Own SoC differs
  from live by then; exact command reproduction at all origins is not claimed.
- Baseline ending inventory gap **−0.108kWh**; variable credit gap **−$0.00811** (<1cent).
  Useful bounded fidelity evidence; not comparable to the earlier one-hour endpoint gap without
  matching clocks, initial state, duration and conditions.
- Terminal targets differ at15/30 origins, max2.45pp. Only one requested command differs>1W:
  at02:20 challenger charges **872W more** than baseline.
- Challenger conditional cashflow delta **−$0.00008924** (<0.01cent), ending inventory
  **+0.01614kWh**, extra DC throughput **0.01630kWh**. No meaningful economic improvement.
- Inventory break-even value **0.553c/kWh**, excluding wear/uncertainty. At5c/kWh salvage,
  inventory-adjusted difference is+$0.000718 (<0.1cent). Tiny result under either valuation;
  no universal recommendation to remove lock-in.
- Both arms physically clipped for470.54s; extra curtailed common supply0.06847kWh.
  Delivered-PV bound, ideal instantaneous commands, device modes/ramps and fixed losses beyond
  .95 conversion/.99 battery efficiencies limit extrapolation to counterfactual operation.
- Authoritative ignored evidence: `data/energy_replay/minute_policy_replay_20261004_30m_v3/`
  (exact60 saved request/result re-audit); original60 solves in`..._30m/`.
  Raw archive`data/energy_replay/control_history_20261004_v2/` adds independent grid stream.

## Validation / command

- **299 affected tests pass**. Own-inventory persistence, old-command latency, no future actual
  leakage, no observed-SoC re-grounding, unchanged non-terminal payload fields, subinterval energy
  equivalence, sign preservation, gap/duration rejection, request mutation and mount-path checks.
- Existing five-minute physical semantics unchanged when duration defaults300s; existing
  sequential batch now uses shared runner, with source/request/result contracts retained.
- Both15m and30m real workers passed all host-side physical/result checks. No production changes.

```bash
./.venv/bin/python eval/minute_core_replay.py \
  --history data/energy_replay/control_history_20261004_v2 \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --start 2026-10-03T02:00:00Z --end 2026-10-03T02:30:00Z \
  --output data/energy_replay/new_minute_policy_replay
```

Add`--solve-archive data/energy_replay/minute_policy_replay_20261004_30m` for exact offline re-audit.
Choose a new output path; household evidence stays ignored.

## Next effort / economic priorities

1. Freeze representative low-solar, full-battery and positive high-value export windows, with
   complete raw targets and as-issued DH price/PV/load/HWC lineage. This midday pilot has no
   exports and does not test missed profitable discharge/export opportunities.
2. Regenerate each arm's DH trajectory/anchors at historical source updates; preserve its own
   feedback and document whether HWC remains exogenous. Conditional live parents can validate
   tactical fidelity; do not use them to claim a full forecast/policy counterfactual.
3. Compare forecast net-energy calibration and cheap APF-tail price residual corrections on
   common targets; matched inventory valuations, throughput, curtailment and structural exclusions.
4. Repeat terminal/lock-in experiments where endpoints actually constrain valuable dispatch.
   Neither this pilot nor the earlier fixed-endpoint sensitivity supports a production change.
