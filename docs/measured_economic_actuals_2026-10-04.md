# Measured economic targets — 2026-10-04

Status: seven-day telemetry export + causal walk-forward load challenger. No production change,
settlement comparison or counterfactual savings claim. Existing replay remains the experiment scaffold.

## Tools / contract

- `eval/measured_actuals.py`: explicit entity/unit/role catalog; UTC five-minute windows, max seven
  days; finite sample-and-hold integration, bounded age, gap coverage. Conflicting duplicate timestamps
  rejected. Future observations never backfill earlier time; NaN terminates prior support.
- `eval/export_measured_actuals.py`: read-only HA metadata and InfluxDB queries; prefers raw
  `sensor__<device_class>` / `number__<device_class>` over CQ measurements with the same entity tag.
  Current unit must match declared unit. Existing output paths rejected; dataset/manifest ignored.
- Numeric powers: 120-second hold default, configurable up to 300. Full 300-second support required
  for primary interval mean; partial means/coverage/sample counts separate. Missing is not zero.
  SoC additionally carries latest past end-point observation, bounded age, no interpolation.
- Limits/mode: bounded one-day hold for state-change history. Stable PV-limit number had no recent
  raw events; fallback reads recorded `emhass_current_pv_input_mode.pv_limit_kw`, never today's state.
  Solcast-now proxy: 15-minute hold; comparison-only role, never measured actuals.
- Grid import/export and battery charge/discharge integrated separately BEFORE interval averaging.
  Net mean can be zero despite nonzero energy in both directions.
- `eval/audit_measured_actuals.py`: verifies complete manifest/parquet hash; coverage, balance
  consistency, plan tracking, current-PV proxy diagnostic. Streams only prediction/vintage/covariate
  columns from forecast logs; ignores their `actual` and `power_pv_actual` columns.
- Forecast rows: target inside measured window, creation at/before target, max 72h lead. Excludes
  partial interval/negative lead and all duplicate-key rows. Checks file inode/size/mtime stability;
  changed source excludes its diagnostic. Selected rows frozen with hashes in private parquet.
- Load diagnostic: six complete measured five-minute base-load bins per 30m target. All causal
  vintages reported by horizon/model; this is not an independent matched walk-forward experiment.
- Output manifest includes exact queries, units/roles, timestamps, coverage, source-code/data hashes
  and `publication_authorized=False`. No HA writes, controls, shared CQ/model/export changes.

```bash
./.venv/bin/python eval/export_measured_actuals.py \
  --start 2026-10-01T00:00:00Z --end 2026-10-03T00:00:00Z \
  --output data/energy_replay/new_measured_actuals
./.venv/bin/python eval/audit_measured_actuals.py \
  --dataset data/energy_replay/new_measured_actuals \
  --load-log load_forecast_log.csv --price-log price_forecast_log.csv \
  --output data/energy_replay/new_measured_audit
```

`--inventory-only` inspects selected metadata/schema without raw export. Commands run at nice 19;
bounded queries, no long dispatch/training job. Private evidence: `measured_actuals_20261004_v2/`
and `measured_audit_20261004_v2/` under `data/energy_replay/`; separate stable price refresh
`measured_price_audit_20261004/`. Initial pilots/inventory retained privately.

## Confirmed external definitions / conventions

Read installed `/opt/dockerfiles/hass/config/configuration.yaml` and current HA units on Oct 4:

- `sensor.sigen_power_pv_gross`: W; `max(PV1+PV2,0)`. Measured delivered DC from configured strings
  1/2, not Solcast or counterfactual available solar. Other strings/system topology still need checking.
- `sensor.sigen_inverter_conversion_loss`: W; clipped rounded `PV1+PV2−battery−inverter_AC`.
  Derived residual is not an independent efficiency measurement.
- `sensor.power_load_without_losses`: W; clipped `grid_active+inverter_active`.
- `sensor.power_consumed_without_deferrable_loads`: W; clipped site consumed power minus
  `deferrable_load_power`. Installed deferrable expression includes an Athom meter plus estimated
  dump loads. Separate compressor/element attribution not yet established by this export.
- Units: measured PV, inverter, battery, grid/site/base loads W; SoC percent; applied limits kW.
- Observed signs agree with formulas/plan tracking: grid positive import; inverter positive AC output;
  inverter/plant battery positive charge. EMHASS planned `P_batt` is positive discharge.
- Historical unit/config stability not asserted by today's metadata. State changes and independent
  sensors are asynchronous; interval balance agreement is a consistency check, not calibrated metrology.

## Seven-day challenger (UTC Sept 27–Oct 4)

- Private datasets: `measured_week_20261004/`, `measured_week_audit_20261004/`,
  `measured_week_calibration_20261004/` under ignored `data/energy_replay/`.
- 2016 five-minute intervals; grid/base load/battery 2014 complete, planned battery 2009,
  delivered PV 1116. Two incomplete load bins exclude one half-hour target.
- 144,144 causal forecast rows; 336 half-hour targets, one August 10 version per model.
  143,715 rows match complete measured targets; 91,839 pass calibration warm-up.
- `eval/calibrate_measured_load.py`: three-day rolling residual quantile by model/version/type/
  horizon band; latest causal vintage per past target, each training target once. Minimum 48
  past targets. Measurement available only at target interval end + 30 minutes assumed receipt lag.
  No future labels; unchanged zero correction until warm-up; corrected load clipped at zero.
- Paired eligible forecasts only; metrics average vintages within target, then give targets equal
  weights. Daily scores retained. Days/vintages overlap; no independent significance claim.

| p65 horizon | Paired targets | MAE baseline → corrected (W) | Pinball baseline → corrected (W) | Actual ≤ p65 baseline → corrected |
|---|---:|---:|---:|---:|
| 0–6h | 285 | 188 → 182 | 81.3 → 80.1 | 80.4% → 78.8% |
| 6–16.5h | 273 | 221 → 176 | 88.3 → 78.5 | 88.0% → 67.0% |
| 16.5–36h | 252 | 254 → 214 | 100.6 → 95.3 | 86.0% → 64.0% |
| 36–72h | 213 | 266 → 210 | 95.9 → 82.7 | 91.7% → 67.0% |

- p50 MAE improvements: ~3%, 17%, 12%, 21% respectively. Short-horizon p50 coverage still
  69.4%; coarse correction insufficient for full conditional calibration.
- Week includes substantial import days: Sept 27 **26.77 kWh**, Oct 3 **22.37 kWh**;
  other days 0.09–0.77 kWh. Not classified as high-price events without interval rates.
- Actual-minus-planned battery charge absolute p95 **411 W**, AC balance p95 **11.7 W**.
  Good coarse consistency does not establish price-weighted execution opportunity cost.
- PV current-proxy diagnostic: 391 strict measured-mode daytime bins; bias **−208 W**,
  MAE **799 W**, estimate/delivered ratio **0.937**. Two-day ratio 0.904 is not stable enough
  to justify a fixed multiplicative solar correction; still no future-vintage accuracy test.
- Decision: prioritise longer-horizon load calibration as a fixed-price/PV/terminal economic
  replay challenger. These are forecast-score gains, not dollar savings or permission to deploy.
  Independent quantile corrections can cross; coherent bundles and regime validation still required.

```bash
./.venv/bin/python eval/calibrate_measured_load.py \
  --dataset data/energy_replay/measured_week_20261004 \
  --audit data/energy_replay/measured_week_audit_20261004 \
  --output data/energy_replay/new_load_challenger
```

## Price archive contract — inspected live Oct 4

- Raw current general/feed sources: `sensor.amber_5min_current_{general,feed_in}_price`,
  `rp_raw.sensor__monetary`. Unit $/kWh; historical sample contains `type_str=CurrentInterval`,
  `estimate=0`, `duration=5`, `start_time_str`, `end_time_str`. Candidate observed-rate evidence;
  API confirmation is not invoice settlement reconciliation.
- Influx numeric `start_time`/`end_time` are date-like numbers (~2.026e17), not epoch timestamps.
  Use ISO `*_str` attributes. Observed interval start is end minus five minutes **plus one second**;
  validate duration/end and explicit source convention before joining five-minute energy bins.
- First Sept 27 raw sample arrived 00:00:23Z but quoted interval ended 00:00:00Z. State receipt
  time is not the priced interval; never align realised rates by observation-time resampling.
- Adjusted-confirmed feed source archives `raw_price`, `export_allowance_adjustment`,
  `confirmed_end_time_str`. Template adds $0.01/kWh at local 10:00–16:00 while configured allowance
  is positive. Preserve the recorded adjustment/config; verify billing meaning separately.
- Effective general/feed templates can fall back to forecasts; historical records lack interval/
  estimate/type metadata. Do not treat them as unconditional realised settlement targets.
- Amber feed sign: negative earns export revenue; internal solver positive export value requires
  negation at the boundary. Keep raw prices and adjustments separately identifiable.
- Next: freeze full raw interval quote/revision history, reject estimated/ambiguous intervals,
  match general/feed/adjustment by quoted end, retain receipt times. Invoice/rate reconciliation and
  complete as-issued MPC APF/PV/HWC lineage remain financial-ranking gates.

## Two-day evidence (UTC Oct 1–3, 576 intervals)

| Target/context | Complete intervals | Interpretation |
|---|---:|---|
| Grid, site/base load, inverter/battery power, planned battery power | 576 each | Usable observed-flow/plan-tracking pilot |
| Delivered PV | 320 | Conservative hold expires during unchanged zero states; retain missing periods |
| Mean SoC / end-point SoC | 517 / 561 | End-point observation preferable for inventory |
| Solcast-now comparison proxy | 305 | Source cadence differs from measured powers |
| Applied limits and curtailment/transition context | 576 | PV cap recovered from recorded mode attributes |
| Direct deferrable-power channel | 34 | Sparse state changes; cannot assume absent samples mean no HWC/dump load |

- AC consistency: site minus grid/inverter absolute error p95 **9.5 W** (576 rows).
- DC consistency with derived loss: absolute error p95 **30.1 W** (320 rows); not independent proof.
- Actual versus planned battery charge: absolute deviation p95 **411 W**, median **−7 W**;
  no five-minute mean deviations above 2 kW. Coarse means can conceal short transients, and no
  price-weighted opportunity loss calculated. These two days do not establish a general execution ranking.
- Reconstruction/transition flag occupied **6.25%** of recorded time. Many bins have brief flags;
  no bin above 50% flag occupancy here. Flagged time is not quantified lost solar energy.
- Strict PV-now diagnostic: context always `measured`, delivered PV ≥500 W, complete proxy/target:
  **108 bins**, bias **−257 W**, MAE **922 W**, estimate/delivered energy ratio **0.904**.
  This compares a recorded current estimate, not future forecast vintages; no correction validated.

## Fresh load / price evidence

- Initial paired audit: both CSVs stable during read. Final version-preserving load refresh stable;
  concurrent append caused its price refresh to be excluded and retried separately.
  Filtered load: **41,184 rows**, `load`, `load_p65`, `load_p75`,
  238 creation times/model, 96 distinct measured half-hour targets. 288 partial-interval rows excluded.
- Filtered price: **82,139 rows**, `price` / `dynamic_handoff`, 1426 creation times, 96 targets.
  576 partial-interval rows excluded. Only combined logged incumbent curve; no full MPC APF lineage
  or settlement/rate comparison established by this audit.

| Horizon | p50 bias / MAE (W) | p65 bias / MAE (W) | Actual ≤ p65 |
|---|---:|---:|---:|
| 0–6h | +53 / 95 | +117 / 141 | 87.7% |
| 6–16.5h | +109 / 134 | +182 / 197 | 93.9% |
| 16.5–36h | +122 / 158 | +230 / 249 | 93.9% |
| 36–72h | +166 / 198 | +289 / 304 | 94.7% |

Each band contains the same 96 distinct targets with multiple prior vintages. Rows are not independent;
these are short-window origin-weighted diagnostics. p65 coverage materially above nominal 65% supports
testing bias/quantile calibration; it does not justify a live quantile switch or establish savings.

## Consequence / next experiment

Evidence favours testing simple net-energy calibration before replacing model architectures:
base-load forecasts conservative here; current PV estimate low on a strict sample. Joint economic
effect still unmeasured. Hold price and terminal policy fixed when comparing p50/calibrated p65.

Next: extend a frozen manifest across recent event/quiet periods; verify settlement rates and full
as-issued MPC APF/tariff lineage. Use recorded PV mode to exclude curtailed samples from calibration;
bound available-PV headroom separately. Audit stable zero-state semantics rather than fill gaps blindly.
Then compare production-core dispatch alternatives on matched origins and equal ending inventory.

Validation: 19 new integration/audit tests; full energy-pipeline plus new tests: 207 pass.
