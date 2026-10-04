# MPC formatter and missing-anchor audit — 2026-10-05

- Decision: numerical MPC projection gate passed on the saved timing corpus. Publication
  atomicity, actual script/device latency and missing input-capture clocks remain separate gates.
  APF retained; no production change or additional optimizer solves.

## Installed formatter parity

- `eval/audit_mpc_formatter.py`: verify preceding feedback chain and each saved core result;
  run only `RetrieveHass.get_attr_data_dict` inside the immutable installed container.
  No HA client instantiation, posting, network or optimizer calls.
- All60 historical MPC cases from the30min/72-solve own-feedback experiment pass. Six consumed
  fields: battery, signed net grid, load, PV, hybrid inverter and PV curtailment power.
- Synthetic numerical fixture covers rounding ties, positive/negative/near-zero values and
  UTC targets spanning Adelaide DST transition.60,528 numeric power points checked total;
  selected current state and interval-start labels agree. Numeric consumed semantics, not
  byte-for-byte JSON formatting or proof of atomic HA publication.
- Installed `RetrieveHass` source SHA256:
  `4c46ed61c15f611e9462f69c4b8c91b6c65d2f84b96cb25228423e92ec66e8b3`.
  Container image/solver pins unchanged. Source mismatch, altered identity/state/sign/coverage/
  target labels or modeled projection mismatch fail before writing a successful report.
- Ignored report: `data/energy_replay/mpc_formatter_parity_20261005.json`; full installed
  formatted output, feedback lineage, worker/projection hashes retained. Six new verifier tests.
- Validation:44 affected formatter/minute/core-request tests pass. Resource bounds remain
  read-only/network-none,1CPU/2GiB,180s client; CPU work only formatting saved arrays.
- Receipt-settling diagnostic against `ems_control_history_20261004_export`: all30 historical
  MPC activation clocks match one complete six-curve receipt event within2s. Latest changed-field
  receipt trails battery receipt by at most0.099375s; sum0.916651s. Selected target clocks coherent.
  These are archive receipt offsets, not device activation delays. Own-feedback execution still
  uses battery publication as its modeled common activation clock; no revised economic claim.

```bash
./.venv/bin/python eval/audit_mpc_formatter.py \
  --replay data/energy_replay/ems_feedback_20261005_export_15m_140w \
    data/energy_replay/ems_feedback_20261005_export_30m_chunk2_140w \
  --formatter-sha256 4c46ed61c15f611e9462f69c4b8c91b6c65d2f84b96cb25228423e92ec66e8b3 \
  --output data/energy_replay/NEW_MPC_FORMATTER_PARITY.json
```

## Unchanged helper versus missing solve

- Sep30 22:41:27.337814 UTC published MPC curve is distinct from prior curve. Last raw
  `mpc_last_soc_init` receipt remains22:40:18.754043 at71.75%; observed derived SoC and
  reconstructed initial MPC SoC near22:41:25 also remain71.75%.
- Known eligible minute anchors nearby:22:34:25.060102,22:36:25.083538,
  22:37:25.088514,22:38:25.069678,22:39:25.062230. Subsequent22:42:25.062899 and
  22:43:25.072464 corroborate cadence retrospectively; future clocks are not causal inputs.
- Five-minute price-trigger origins differ, e.g.22:35:18.674037 and22:40:18.754043.
  Do not apply a second25 timer assumption to those minutes.
- At22:41:25: measured load267W, gross PV227.247W, conversion loss180W clamped140W;
  reconstructed first-slot MPC load267W/PV87W match published curves. New telemetry near
  22:41:26.29 changes load/PV. Reconstructing at publication27.337s would consume those newer
  values and incorrectly infer load298W/PV≈89W.
- Evidence supports unchanged-helper event suppression; it does not reveal exact capture time.
  Current scheduler YAML corroborates second25 timer, but no immutable Sep30 scheduler
  configuration was found. Preserve observed origins; proposed25s/25.1s reconstructions must
  remain labeled clock sensitivities with causal telemetry/parent and rounding checks.
- Strict15s helper-age guard remains unchanged. No automatic admission to economic replay.

## Bounded clock diagnostic

- `eval/audit_mpc_clock.py`: hash-verified historical/control archives; reference runtime plant,
  nominal capacity/health with explicit raw receipts or older unchanged capture evidence;
  modeled candidate inputs selected as-of25s/25.1s. Published curves are subsequent consistency
  evidence only, never candidate-time inputs.
- Extended source archive Sep30 22:10–23:30:80 publications,72 fresh observed helper clocks
  preserved,8 lacking a fresh helper. Exactly22:41 supports both timer-clock candidates.
- Other cases reject distinctly:23:00 is a five-minute price-trigger minute; later cases
  lack paired control curves beyond22:45 or have stale telemetry under the existing120s rule.
  No inferred origin is admitted; no successful economic report produced from incomplete support.
- Twelve focused tests cover preserved clocks, unchanged helpers, candidate source changes,
  future observation exclusion, endpoint/helper/receipt consistency, missing paired curves,
  preceding-clock evidence, five-minute exclusion and causal capacity fallback.
- Authoritative ignored clock report: `data/energy_replay/mpc_clock_audit_20261005_extended_v2.json`.
  Frozen archive/config/capture/dependency hashes, per-candidate source ages and receipt clocks
  retained. Supported22:41 capacity/health evidence is raw archived; no static fallback needed.
  Efficiencies use frozen reference configuration; historical equivalence remains unproven.
- Final combined formatter/clock/core/feedback/control suite86 tests passes. Next allow a
  separately labeled supported clock sensitivity only after controller branch/source coverage
  permits extending the own-feedback replay; do not globally loosen source freshness.

```bash
./.venv/bin/python eval/audit_mpc_clock.py \
  --history data/energy_replay/dh_source_history_20261004_export_extended \
  --control-history data/energy_replay/ems_control_history_20261004_export \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --replay data/energy_replay/sequential_load_replay_20261004_v3 \
  --output data/energy_replay/NEW_MPC_CLOCK_AUDIT.json
```
