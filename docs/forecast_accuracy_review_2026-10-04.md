# Price and load accuracy review — 2026-10-04

## Decision summary

- User requested fresh technique assessment, including `purcell-lab/nem_pd7day`.
- Completed: source review, literature/repository comparison, local evidence/data inventory.
- No new accuracy benchmark, training, HA write, service change, or model promotion.
- Best first price experiments: rolling calibration; AEMO residual models; short/long-memory ensembles.
- Best first load experiments: seasonal baseline + recent-error adaptation; pooled horizon-aware LGBM; GAM.
- Evaluate APF-assisted and APF-free price separately. Load normally needs neither APF nor wholesale price.
- First dependency: refresh evaluation exports and establish input/label availability at each origin.
- Production remains `amber_apf_lgbm` and LGBM `load_p65`; implementation of APF-free candidates remains paused.
- This review replaces the July technique ranking for future experiment design; historical results remain evidence, not current accuracy claims.

## Current approach and evidence quality

Canonical routing: [production critical path](prod_pipeline_critical_path.md).

- MPC: 14h × 5-min, Amber directly. Improving only the 72h tail principally changes day-ahead inventory planning.
- Day-ahead: 72h × 30-min. APF covers approximately 16.5h; LGBM supplies the tail.
- `_predict_with_dynamic_handoff` in `forecast.py` appends APF predictions to observed history, then forecasts beyond that pseudo-history. Model training uses actual target history. Hypothesis: explicit APF features and residual targets may transfer better than treating predicted values as observations.
- `train_single_model` learns on realised weather/PV/demand; runtime uses forecasts. Candidate reports explicitly identify this causal limitation. The TFT load dataset likewise uses future realised weather/PV proxies.
- Existing price residual audit: interchange and renewable forecast errors explain useful structure. Realised errors are diagnostics; future errors cannot be model features.
- STPASA is already part of current production price covariates. The proposal is better causal use and model comparison, not simply adding a new feed.
- June STPASA results vary materially: a short tail validation improved `$31.16 → $20.97/MWh`; the larger validation improved `$37.02 → $36.19/MWh` beyond baseline correction. Those are different windows, not interchangeable estimates of expected gain. See [renewable discovery](archive/price_forecast_2026/aemo_renewable_availability_discovery_2026-06-14.md) and [residual audit](archive/price_forecast_2026/lgbm_residual_driver_audit_2026-06-14.md).
- `eval/ablate_stpasa_tail_features.py` splits by creation time without excluding training labels targeting beyond the cutoff. Adjacent 72h forecasts can share target outcomes across train/validation; reproduce with label-availability filtering before relying on effect sizes.
- `data/build_load_dataset.py` fits scalers on the full input before splitting; target scaling also sees validation. Training origins immediately before the cutoff have decoder targets extending into validation. Fix both before a new neural comparison. This does not quantify how much historical scores were affected.
- TFT load had encouraging q50 MAE, but q50 must be compared with q50, and operational tests must use production p65. May dispatch work also identified load-blind strategic targets and terminal-SoC confounds. See [load findings](load_source_dispatch_findings_2026-05-13.md).
- July review's claim that APF independence was demonstrated is too strong: historical dispatch wins are configuration/window-specific, and do not establish present forecast superiority or post-PEC transfer.

### Local inventory, read-only scan on 2026-10-04

Counts are CSV model-labelled rows with a nonblank/non-NaN actual field. They are not deduplicated samples, matched comparisons, validated actuals, or independent observations. Blank model-name legacy rows are excluded below and need quarantine.

| Surface | Rows | Actual-populated rows | Last creation, UTC |
|---|---:|---:|---|
| Price `price` | 7,254,721 | 7,049,606 | 2026-10-03 23:10 |
| Load `load` q50 | 2,774,160 | 2,761,416 | 2026-10-03 23:01 |
| Load `load_p65` | 996,624 | 983,880 | 2026-10-03 23:01 |
| TFT load q50 | 252,144 | 252,144 | 2026-06-14 22:31 |
| PD-direct | 1,135,008 | 0 | 2026-06-14 22:41 |

- TFT q50 creation window: May 9–June 14. q10/q90: May 12–June 14, 229,824 actual-populated rows each. Enough for a historical matched analysis; no October TFT evidence.
- p65 creation window: May 12–October 3. The earlier lack of production-surface logging is no longer a blocker.
- PD-direct actual fields are empty; join independently verified realised RRP, rather than treating the log as already scoreable.

| Local parquet | Latest run/actual, UTC | Consequence |
|---|---|---|
| `aemo_pd7day_sa1.parquet` | Run May 12 21:08 | Recent PD7DAY experiments require export refresh |
| `aemo_predispatch_sa1.parquet` | Run May 12 22:30 | Recent PD experiments require export refresh |
| `aemo_stpasa_regionsolution_sa1.parquet` | Run October 3 22:00 | Current tail covariates available; field coverage still needs audit |
| `actuals_sa1.parquet` | Actual May 28 22:30 | Not usable as October ground truth without refresh |
| `load_actuals_tft.parquet` | Actual April 16 13:30 | Retained TFT training export is stale |

Source and target timestamp ranges were read from parquet columns; future target endpoints do not establish fresh runs. Live InfluxDB may contain newer data; this review did not query or refresh it.

## Purcell repository: useful design, important qualifications

Inspected public source at commit `07dccf06352732c34863a212e072619a34679755`, release v3.19.2, September 30. [Pinned source](https://github.com/purcell-lab/nem_pd7day/tree/07dccf06352732c34863a212e072619a34679755).

The integration learns a monotone PD7DAY-to-actual mapping and adds STPASA correction for 22–120h. Worth adapting as a lightweight tail challenger. It complements Amber Express; that product/window is not our 16.5h APF contract. Its calibration inputs use AEMO rather than requiring APF. [README](https://github.com/purcell-lab/nem_pd7day/blob/07dccf06352732c34863a212e072619a34679755/README.md).

### Borrow

- Preserve forecast-run history and pair actuals with each original forecast and its STPASA snapshot. Avoid reconstructing training features from today's forecast.
- Rolling monotone calibration offers a cheap, inspectable baseline.
- Distinguish near-term, daily, and multi-day horizons; expose correction source and fallback reason.
- Share feature transforms between training and inference.
- Gate unsupported extrapolation using coefficient-weighted feature excursions relative to residual spread; compare with smooth shrinkage rather than copying constants.
- Preserve negative prices and calibrate uncertainty beside the served point forecast.

Sources: [actual recording](https://github.com/purcell-lab/nem_pd7day/blob/07dccf06352732c34863a212e072619a34679755/custom_components/nem_pd7day/actual_recorder.py), [serving](https://github.com/purcell-lab/nem_pd7day/blob/07dccf06352732c34863a212e072619a34679755/custom_components/nem_pd7day/serving.py).

### Do not copy unchecked

Source-level findings, not observed live failures; no upstream code was executed.

| Finding | Evaluation consequence |
|---|---|
| README says 60-day history; executable stage 1 uses 90 days and exponential decay with ~21-day half-life | Pin implementation; test window/decay rather than assuming README settings |
| Stage 1 trains with solar-elevation buckets, but serving lookup uses fixed NEM clock-hour buckets; source comments track #208 | Use one routing function everywhere; use Adelaide local calendar and solar geometry deliberately |
| Full buckets keep the first 5,000 pairs from oldest-first input; comments track #209 | Select newest rows or weighted samples; ensure adaptation survives caps |
| Stage 2 does not apply stage 1's explicit rolling-window/decay filter; comments track #210 | Give both stages explicit causal histories and independently tune adaptation |
| Stage 2 fits absolute actual price from nine features, including isotonic output; wind is mentioned in the README but absent from the executable feature vector | Compare OLS/ridge absolute fit against additive residual fit; include wind explicitly in ablation |
| Stage 1/2 exclude observations if actual or forecast reaches `$3,000/MWh`; high serving inputs can be clipped to the isotonic domain | Score all outcomes, retain separate spike modelling and recall; good ordinary MAE is insufficient |
| Stage 2 can reject sign flips relative to stage 1 | Test this policy: an accurate correction may legitimately cross zero |
| Stage 2 uses leave-one-out OLS residual quantiles; stage 1 is fitted from the same observations | LOO bands are not a temporal out-of-sample guarantee; fit stacked stages and calibrators from rolling out-of-fold predictions |

Sources: [fitting.py](https://github.com/purcell-lab/nem_pd7day/blob/07dccf06352732c34863a212e072619a34679755/custom_components/nem_pd7day/fitting.py), [calibration_engine.py](https://github.com/purcell-lab/nem_pd7day/blob/07dccf06352732c34863a212e072619a34679755/custom_components/nem_pd7day/calibration_engine.py), [serving.py](https://github.com/purcell-lab/nem_pd7day/blob/07dccf06352732c34863a212e072619a34679755/custom_components/nem_pd7day/serving.py).

README reports `$16.06 → $14.65/MWh` for STPASA versus isotonic alone. Treat as author-reported evidence: no reproduced SA1 holdout/our horizon/our dispatch comparison here. Isotonic squared-error fitting estimates a monotone conditional mean; naming a separately fitted quantile p50 does not make the isotonic point a median. Compare corresponding estimands.

## Other repositories and research

| Primary source | Inspiration | Fit here |
|---|---|---|
| [epftoolbox](https://github.com/jeslago/epftoolbox), [LEAR source](https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/models/_lear.py) | Sparse ARX/LASSO, repeated recalibration, reproducible baseline comparisons | Cheap linear challenger on PD/APF residuals; adapt hourly day-ahead assumptions to rolling NEM intervals |
| [SA NEM probabilistic forecasting paper](https://arxiv.org/html/2311.07289v2) | Quantile forecast averaging, different history lengths, recent-price autoregressive correction | Stronger domain match than European benchmark rankings; test multi-memory ensemble before larger networks |
| [Adaptive probabilistic load paper](https://arxiv.org/abs/2301.10090) | Kalman adaptation and online residual quantile updates | Cheap load drift correction; regional/city evidence does not establish household gains |
| [pyGAM](https://github.com/dswah/pyGAM) | Penalised smooth nonlinear effects | Interpretable household seasonality/temperature challenger; expectiles are not quantiles, so use quantile/residual calibration separately |
| [MLForecast](https://github.com/Nixtla/mlforecast), [StatsForecast](https://github.com/Nixtla/statsforecast) | Lag/rolling feature tools and statistical baselines | Useful implementation reference; no need to migrate the current stack |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE), [ACI paper](https://arxiv.org/abs/2106.00170) | Sequential uncertainty calibration | Test horizon-specific calibration with delayed labels; coverage changes do not inherently improve point MAE |
| [NeuralForecast NHITS](https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nhits.py) | Direct multi-horizon model with future covariates | One small load challenger after baseline/data fixes; price residuals optional later |
| [Chronos](https://github.com/amazon-science/chronos-forecasting) | Pretrained probabilistic forecasting; Chronos-2 supports covariates | Limited load zero-shot probe; benchmark CPU latency/memory. Generic benchmark wins are not household or SA1 evidence |
| [NEMSEER](https://github.com/UNSW-CEEM/NEMSEER) | Historical forecast runs for AEMO products | Data/replay tooling; not a forecasting algorithm |

## Ranked price experiments

### P1 — Rolling correction of a strong existing curve

Hypothesis: small corrections learned from settled forecast errors improve bias and pinball loss with less regime risk than relearning absolute price.

- APF-assisted: baseline is the complete current published wholesale curve. Correct the >APF tail first; separately test correction inside APF coverage.
- APF-free: baseline is P5MIN near term, PREDISPATCH where available, PD7DAY tail; retain seasonal fallback with explicit missing/stale flags.
- Compare raw baseline, horizon×local-time rolling bias, shrinkage isotonic, ridge residual, LGBM residual.
- Target convention: `residual = actual - baseline`; candidate = baseline + correction.
- Residual framing alone does not guarantee a safe fallback or prevent compression. Explicitly shrink to zero when data/domain support is weak.
- Use rolling out-of-fold residuals for calibration and stacked training. Test raw/asinh price representation; signed prices must remain valid. Quantile transforms are monotone, while inverse-transforming an estimated mean does not preserve the mean.

### P2 — Market fundamentals and explicit APF inputs

Hypothesis: forecast net supply, interchange, and revisions explain errors missed by local weather/site PV.

- PD/PDPASA: aggregate and split renewable availability, regional demand uncertainty, net-load proxy.
- Interconnectors: forecast flows, directional limits/headroom, binding constraint information where available. Include SA–NSW PEC, not only legacy SA–VIC routes.
- STPASA: wind/solar, surplus/availability, demand spread, age and source coverage.
- Forecast revisions: same target's price/UIGF/interchange changes across past runs; recent settled baseline errors. Everything available by origin only.
- APF-assisted challenger: train with APF level, shape, age, cutoff and overlap with PD/PD7DAY as explicit features; forecast the tail without inserting APF as observed target history.
- Optional later: price-sensitivity curves and state-wide rooftop PV. Availability/schema/publication-lag audit precedes feature claims.
- One feature family at a time. Missing forecast fields are missing, not zero generation or zero capacity.

### P3 — Multiple memories and conditional combinations

Hypothesis: stable seasonal patterns and rapid regime changes need different history lengths.

- Predeclare short/medium/long price histories, e.g. 30/90/365 days where source coverage permits.
- Start equal-weight or regularised nonnegative combinations; learn weights only from earlier out-of-fold losses.
- Test APF versus independently trained AEMO forecasts by horizon/availability. APF-free training must not inherit APF-seeded histories or features.
- Consider QRA after point/quantile candidates are independently useful; validate quantile crossing handling and calibration.

### P4 — Separate ordinary and scarcity regimes

Hypothesis: rare extreme outcomes require explicit occurrence probability and conditional severity.

- Two-part spike model, negative-price probability, and regime-dependent residual distributions.
- Scarcity inputs: supply margin, demand uncertainty, interconnector limits, notices available at origin.
- Preserve rare events in final scoring; never infer calibrated tail risk from spike-filtered training alone.
- Report threshold precision/recall, false spike duration, spread/rank and window timing alongside MAE. Scenario paths are preferable to independent interval draws for inventory decisions.
- Lower priority: too few post-PEC events to support a complex new scarcity model today.

## Ranked load experiments

### L1 — Seasonal profile plus online adaptation

Hypothesis: household schedule and recent level shifts are easier to learn than a complete 72h curve from scratch.

- Baselines: previous day/week; rolling median for each half-hour × weekday/weekend; weather-matched similar days.
- Correct recent settled errors using EWMA or Kalman level adjustment, decaying with horizon; compare with refit-only baseline.
- Separate base load from scheduled HWC/EV/dump loads. Existing code already subtracts deferrable/dump load; audit consistency of training/log actuals and reconstructed total rather than subtracting twice.
- Use occupancy/holiday/HVAC information only when known and recorded at origin. Temperature lags/degree-hours address thermal inertia.
- APF normally adds no causal information to uncontrollable base load. Test it only for an identified remaining price-responsive component, holding scheduled controls fixed.

### L2 — Pooled horizon-aware LGBM and GAM

Hypothesis: sharing examples across related horizons improves stability; smooth calendar/temperature structure complements trees.

- LGBM row = origin × target, with horizon, local target calendar, target-aligned seasonal lags, shifted rolling statistics, recent profile/energy, and forecast-vintage weather.
- Benchmark pooled horizons versus current multi-output formulation; sharing is not automatically superior.
- GAM: cyclic half-hour/week/year effects, temperature response and recent-level offset. Residual quantiles learned/calibrated separately.
- Test native W target versus current logarithmic transformation using MAE, original-unit pinball and energy bias. No assumption that removing the transform wins.
- Score q50 for median accuracy and q65 for operational comparison. A calibrated q65 should have about 65% of observations below it in aggregate; its intended positive bias is not inherently a defect.

### L3 — Small neural/foundation-model challenger

- Score retained TFT on matched May–June rows first. A favourable historical result earns a new causal-data trial, not promotion.
- Prefer one small NHITS trial or one Chronos-2 load probe to another broad model search.
- Use equal/controlled horizon weighting; the prior TFT's tail weight cliff must not recur unnoticed.
- Assess morning-ramp errors across weather/day types; do not impose universally increasing 03:00–07:00 load as physical law.
- Ensemble only if out-of-fold errors are complementary. APF-free price TFT failure does not establish that all neural household-load models fail.

## Fixed evaluation protocol for follow-on work

1. **Freeze rows and provenance.** UTC origin/target keys, interval-start/end convention, W versus kW and $/MWh versus $/kWh. Record forecast publication/first-seen time, artifact hash, source age, tariff and target definition. Publication or actual-availability time must be <= origin; `run_time <= origin` alone can admit reports not yet received.
2. **Refresh stale exports.** Independent realised wholesale/base-load series, PD/PD7DAY runs, weather/PV vintages and APF snapshots. Audit duplicates, legacy blank model rows and known device/control changes. Verify logs' actual columns against independent targets on a sample.
3. **Separate tracks.** APF-assisted price; APF-free price; APF outage fallback; household base load. APF-free means no APF-dependent training labels/features/pseudo-history. Price ground truth is realised wholesale RRP, not APF. Score retail outcomes after common tariff conversion.
4. **Match comparisons.** Exact origin/target rows where possible. Event-triggered APF and fixed AEMO updates require an as-of common origin schedule; report freshness and missingness, not just the intersection that flatters a source. Small run-time offsets between load quantiles need a documented run-id/tolerance rule with no future origin selection.
5. **Causal walk-forward.** Expanding/rolling fit, separate calibration and untouched test. At every fit, all target intervals/labels must have settled and been available. Purge overlapping 72h decoder labels across fold boundaries; fit scalers, transforms, bucket routing, windows and ensemble weights on training only.
6. **Seasonal/regime windows.** Use eligible 2025–2026 winter/summer/shoulder periods; PD7DAY comparisons only where its historical runs exist. Hold October 1 onward as a separately reported PEC regime, including a frozen pre-change model for transfer diagnosis. Approximately three elapsed days is an early diagnostic, not a post-PEC verdict. Short-history candidates may be unscoreable on early seasonal folds.
7. **Primary accuracy metrics.** Price: q50 MAE/bias, original-unit pinball/WIS, quantile coverage/width, negative/spike precision-recall, daily rank and cheap/expensive window timing. Load: W MAE/bias, pinball q50/q65, underprediction magnitude/duration and 4h/24h/72h cumulative kWh error. Avoid MAPE near zero/negative prices and near-zero loads.
8. **Horizon panels.** Price: 0–1h, 1h–APF cutoff, cutoff–28h, 28–48h, 48–72h. APF-free: additionally 1–6/6–12/12–28h. Load: 0–6/6–24/24–48/48–72h. Report Adelaide time-of-day and actual regime strata; keep these diagnostic labels out of future model inputs.
9. **Paired uncertainty.** Aggregate losses by day and use paired multi-day block bootstrap; overlapping 72h predictions are dependent. Compare longer block lengths. Fix the candidate roster/window choices before the test and retain failures/missing rows in the manifest.
10. **Economics after accuracy.** Price swap with fixed load/PV, then load swap with fixed price/PV, then combined winner. Existing `rolling_mpc_eval.py` and `netload_tariffed` are starting points. Verify present control, load-aware strategic target, battery/terminal-value assumptions and realistic observed PV; historical oracle inputs cannot establish live savings.
11. **Availability/safety comparison.** Fixed latency/resource budget, complete monotone quantile family, no unsupported extrapolation, explicit stale/missing-source fallbacks. Apply existing bundle eligibility/promotion gates after a shadow period; no automatic promotion from retraining or isolated MAE.

### Initial roster and stopping rule

| Track | Initial candidates | Primary question |
|---|---|---|
| APF-assisted | Incumbent; rolling tail bias; ridge tail residual; LGBM tail residual | Can simple corrections beat current tail on causal matched rows? |
| APF-free | Raw AEMO stack; rolling bias/isotonic; ridge residual; LGBM residual | Does independent correction improve raw AEMO and approach incumbent quality? |
| Load | Current q50/q65; seasonal profile; profile+EWMA; pooled LGBM; GAM | Does adaptation/shared structure improve median accuracy and p65 under-preparation? |

- Same feature rows, folds and original-unit scores within each track. Sparse source coverage is reported explicitly.
- Development screen: proposed >=5% improvement in primary MAE or pinball, without material complementary-metric deterioration. This is a search-budget rule, not proof or a promotion threshold.
- If simple candidates fail across folds, investigate residuals/source quality before expanding architecture search.
- Add multi-memory ensembles next; then only one neural load candidate if remaining errors suggest temporal structure worth the cost.
- Final accuracy claim requires consistent paired evidence on untouched windows; economic deployment additionally needs no material operational regression under the existing gates.

## Recommended execution order

1. Refresh/export audit and frozen row manifest. Score current p65 calibration and matched historical TFT. Establish post-PEC baseline diagnostics.
2. Run P1 and L1 cheap challengers; compare ridge with LGBM residuals on identical causal rows.
3. Add P2 fundamentals/revisions and L2 pooled horizon/GAM one family at a time.
4. Test P3 multiple memories; assess uncertainty calibration separately from point accuracy.
5. Reserve scarcity/neural/scenario work for demonstrated residual gaps; run controlled dispatch only on shortlisted candidates.

No expected percentage gain is asserted. Prior results and external models support these hypotheses; only the new matched walk-forward experiment can establish improvements here.

## Economic priorities with APF retained — user follow-up

User expects APF to remain available. Working assumption: improve the APF-backed system's
economic outcome; APF-free research becomes a resilience option with lower priority.
This changes the execution order above when choosing work for financial return rather than
forecast accuracy. These rankings are hypotheses, not measured savings.

1. **Establish trustworthy economic replay and loss attribution.** Finish the narrow isolated
   solver boundary already selected in [solver checkpoint](energy_pipeline_solver_isolation.md);
   avoid live EMHASS action wrappers, which write shared state even with posting disabled.
   Replay the real DH/MPC SoC feedback, current tariff, HWC inputs, export limits and battery
   losses. Compare planned with delivered action. Report net cost/export revenue, ending stored
   energy, throughput, clipping and unmet-load risk. Preserve the current production policy as
   baseline. Perfect price/PV/load substitutions separately diagnose headroom; oracle results
   require consistent endpoint/constraint treatment and are not achievable savings estimates.
2. **Test decision policies with forecasts held fixed.** Compare current terminal SoC/lock-in
   policy and buy/sell quantile weights with small bounded alternatives. Target repeated
   deferral to cheap windows that recede, premature depletion before expensive periods and
   unnecessary grid precharge before solar. Start from the existing
   [charge-lever hypothesis](charge_lever_controller_plan_2026-06-22.md), but test across event
   and quiet periods; the June episode alone does not validate it. Raising a quantile blend
   does not guarantee earlier charging: only relative price reshaping and resulting dispatch
   establish the mechanism. Test terminal policy and uncertainty weights separately before
   combining them; avoid counting one risk twice. Retain a route toward soft terminal energy
   value/scenario optimisation if simple controls show economic headroom.
3. **Improve usable net-energy forecasts and flexible-load coordination.** Attribute PV
   forecast errors, clipping/curtailment, load bias and HWC timing to actual costs. Test rolling
   PV/base-load calibration and p50 versus calibrated p65 under the same price/terminal policy.
   Verify HWC is represented once and that the battery and thermal plans share consistent
   accepted inputs. Give this effort priority over price-tail work if PV/load oracle substitution
   shows greater opportunity. Household-load model replacement alone has weak historical
   economic evidence; joint net-energy and controllable-load timing may be more consequential.
4. **Improve the APF tail where it changes inventory decisions.** Benchmark rolling bias,
   ridge residual and LGBM residual against the current >16.5h curve, emphasising 16.5–36h
   and the next evening/solar transition. Preserve APF near-term as baseline. Add forecast
   fundamentals/revisions and multiple memories only after cheap corrections. Evaluate the
   full DH→MPC chain: MPC already receives APF directly, so a better tail matters through
   DH energy posture, not direct replacement of the next battery setpoint. Post-PEC results
   need their own window and cannot be inferred from May performance.
5. **Reserve complex/new price models for demonstrated gaps.** APF-free replacements,
   foundation models and multi-day spike predictors are lower priority under this objective.
   Scarcity preparedness can still matter economically, but measure missed-event energy and
   opportunity cost before committing to a new predictor.

First concrete deliverable: one production-faithful replay manifest and a ranked table of
economic loss/headroom by policy, execution, PV, load/HWC and price horizon. Next deliverable:
a fixed-forecast experiment on terminal policy and conditional uncertainty weights, alongside
cheap net-energy calibration. Architecture consolidation proceeds for reliability on its own
merits; faster forecasting is not assumed to improve revenue unless stale/missed actions are
observed. No training or production-policy change was launched by this follow-up.

Execution checkpoint: [existing replay audit and isolated DH/MPC core solves](economic_replay_checkpoint_2026-10-04.md).
Both recorded horizons solved Optimal and passed physical checks. This establishes solver mechanics;
one-cycle historical DH→MPC handoff is now verified; multi-cycle replay and realised loss attribution
remain outstanding. Live PV source audit found recent `power_pv` actuals-export values are Solcast
estimates. Establish independent measured PV targets before assessing PV accuracy/economic headroom.

[Measured target pilot](measured_economic_actuals_2026-10-04.md) now supplies a separate two-day
dataset and causal forecast-window snapshots. Short-window p65 coverage is 88–95%, current-PV
estimate/delivered ratio 0.904 on strict measured-mode samples. Supports cheap calibration trials;
larger matched windows and dispatch/settlement evidence still needed before ranking realised gains.

Seven-day extension: causal residual correction improves p65 MAE **16–21% beyond six hours**,
about **3% below six hours**; corrected longer-horizon coverage near nominal 65%. Prioritise this
cheap challenger in matched economic replay. PV-now ratio moves from 0.904 to 0.937; fixed solar
rescaling is premature. Raw Amber archive preserves interval/estimate flags and ISO timestamps;
effective fallback prices unsuitable as unconditional realised rates. Details in measured-target doc.
