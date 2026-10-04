# Economic regimes and DH source admission — 2026-10-04

- Decision: test input alignment before ranking new forecasting techniques. Four reconstructed
  DH origins combine current price/PV targets with a load array starting one half-hour earlier.
- Existing payload policy consumes load positionally. All arrays have 144 rows; length checks
  miss this defect. Exact target-grid checks reject it. No production changes in this work.
- Next: enforce this admission in bounded replay; hold the previous accepted DH/HWC plan until
  fresh aligned inputs arrive; regenerate each arm's DH/HWC from its own simulated inventory.
  Missing final load target cannot be repaired by shifting and filling with future information.
- Update: [own-battery DH feedback pilots](dh_feedback_economic_replay_2026-10-04.md) now implement
  admission/retained parent and per-arm battery feedback; HWC remains exogenous. Next calibration.

## Frozen diagnostic windows

Complete five-minute measured targets and frozen v6 observed Amber rates; six-bin windows,
initial measured inventory, deterministic ranking, fixed regime priority, no overlap.
Selection uses realised covariates: diagnostic stress cases, **not out-of-sample savings evidence**.

| Regime | UTC window | Mean delivered PV | Mean export price | Initial/final SoC |
|---|---|---|---|---|
| High-value export | Sep 30 22:00–22:30 | 214 W | 24.23 c/kWh | 78.18% / 71.78% |
| Low-solar export | Sep 30 21:30–22:00 | 80 W | 17.96 c/kWh | 80.08% / 78.18% |
| Near-full PV | Sep 28 04:50–05:20 | 5,386 W | −2.80 c/kWh | 96.46% / 100% |

- Candidate counts: high-value 36; low-solar 85; near-full negative-buy 0; near-full PV 3.
- 1,287 candidate windows lack complete required targets. No filling night PV recording gaps.
- Near-full PV is a separate relaxed category: mean import price +5.85 c/kWh. Do not describe
  it as a negative-import-price case. Delivered PV is not counterfactual available supply.
- High-value window observed export credit $0.59625; this is observed cashflow, not headroom.
- Ignored report: `data/energy_replay/selected_regimes_20261004_v3/report.json`.
- Actuals: `measured_week_20261004`; quotes: `amber_observed_week_20261004_v6`.
  Scripts verify dataset/rate hashes; private telemetry and forecasts stay uncommitted.

## As-of source admission

`export_control_history.py --include-dh-inputs` adds production DH prices/quantiles, load, four
Solcast days, APF legs, settings, reground state and plant capacity/health. Raw measurement discovery
must identify one matching sensor measurement; ambiguous/missing sources remain explicit.
Bounds: ≤90 minutes, ≤10,000 window rows/source plus one latest prior record. Admission applies
age limits to that prior record. Recorded helper/plan/source entities are not an atomic input capture.

| Frozen source archive suffix | UTC archive window | Aligned / audited DH origins |
|---|---|---|
| `midday` | Oct 3 01:45–02:30 | 9 / 10 |
| `export` | Sep 30 21:15–22:30 | 15 / 17 |
| `near_full` | Sep 28 04:35–05:20 | 9 / 10 |

- Paths: `data/energy_replay/dh_source_history_20261004_<suffix>/`.
- Authoritative admission/score reports: `dh_source_admission_20261004_<suffix>_scored/`.
- Midday archive predates addition of APF legs; use verified prior APF archive or refresh before
  feeding this archive into a complete DH→MPC replay. Other two include APF.
- Clock: DH initial-SoC helper-write timestamp minus 1 µs; restore previous parent/anchor.
  Reconstructed SoC can differ at non-atomic reground transitions. This is source alignment
  evidence, not proof of the exact historical HTTP request or solver lineage.
- Require price ≤15 min, load ≤60 min, Solcast ≤6 h; all 144 half-hour targets match the current
  UTC boundary, including across Adelaide daylight saving. Quantile receipts ≤2 s apart;
  receipt proximity does not prove common model-run identity.
- Six static settings have no Influx records in these exports. Explicit fallback uses the Oct 3
  captured state with `last_updated` Sep 18, before every origin. This supports unchanged-state
  reconstruction, **not independent historical availability**. Variable settings use recorded rows.
- Solcast `detailedForecast_str` is Python repr with `datetime.datetime` / `zoneinfo.ZoneInfo`.
  Parser accepts a bounded constructor grammar, never `eval`; unknown calls/timezones rejected.
  Mixed +09:30/+10:30 offsets normalise to UTC before target comparisons.

## Load timestamp diagnostic

Same issued load vintage: compare erroneous positional values against correctly timestamped
values on the 143 common targets. Complete measured base-load half-hours only; missing final
prediction omitted. Targets are used for scoring, never for reconstructing decision inputs.

| Reconstructed origin UTC | Complete paired targets | Positional / aligned MAE | Aligned − positional next-14h energy |
|---|---|---|---|
| Sep 28 05:00:43.576625 | 142 | 289.37 / 285.39 W | −0.34112 kWh |
| Sep 30 21:31:26.192262 | 143 | 202.81 / 199.02 W | +0.21518 kWh |
| Sep 30 22:01:32.217196 | 143 | 202.73 / 201.01 W | +0.20226 kWh |
| Oct 3 02:00:40.434122 | 44 | 177.83 / 174.41 W | −0.11377 kWh |

Small MAE changes do not establish economic improvement. Paired forecast changes average
40–53 W; optimizer actions, ending inventory and available-PV/controller limits still need replay.

## Commands and validation

```bash
./.venv/bin/python eval/select_economic_windows.py \
  --dataset data/energy_replay/measured_week_20261004 \
  --quotes data/energy_replay/amber_observed_week_20261004_v6 \
  --output data/energy_replay/NEW_SELECTION
./.venv/bin/python eval/export_control_history.py \
  --start 2026-09-28T04:35:00Z --end 2026-09-28T05:20:00Z \
  --include-dh-inputs --output data/energy_replay/NEW_ARCHIVE
./.venv/bin/python eval/audit_dh_source_history.py \
  --history data/energy_replay/dh_source_history_20261004_near_full \
  --journal /tmp/resident-handoff-20261003.sqlite \
  --dataset data/energy_replay/measured_week_20261004 \
  --output data/energy_replay/NEW_ADMISSION
```

- Output directories must be new. `--dataset` optional: adds measured alignment scoring.
- Regression coverage: equal-length shifted load; mixed-offset DST horizon; constructor grammar
  rejection; complete same-vintage pairing/no tail filling; deterministic non-overlapping regimes;
  missing targets/export energy; physical bounds and target-grid boundaries.
- Validation: 312 affected-suite checks passed; final 15 admission/regime checks passed after
  adding export-energy/boundary and future-capture checks (314 distinct tests total).
  Existing InfluxDB datetime deprecation warning only; three read-only exports/audits completed.
