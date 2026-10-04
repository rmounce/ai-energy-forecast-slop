# Parallel price-path assessment — 2026-10-04

- Scope: independent source/code review; APF retained. No training, production write or savings claim.
- Recommendation: finish economic attribution, then cheap tail residual correction; uncertainty/policy
  calibration before another large price model. Existing APF-free paths remain research references.
- Reviewed production `forecast.py`, price-source contracts, June residual/dispatch evidence,
  October measured-target evidence and the new causal load-calibration implementation.

## What matters economically

- MPC already receives Amber directly over its 14h horizon. A better >16.5h wholesale tail changes
  dispatch through the DH inventory trajectory and inherited MPC endpoint. It does not directly
  replace the next MPC tariff. Report the complete DH→MPC effect, including unchanged first actions.
- June STPASA residual replay reported only **+$0.018** cashflow over its available window;
  candidate ended **0.689 kWh lower** than baseline. This is not an inventory-normalised gain.
  Better MAE and changed strategic targets therefore do not establish economic value.
  [Historical evidence](archive/price_forecast_2026/lgbm_residual_driver_audit_2026-06-14.md).
- Attribute opportunity to horizons/events before spending effort: price perfect-information
  substitutions separately for APF prefix, 16.5–36h, and 36–72h; preserve load/PV, constraints,
  execution rules and endpoint treatment. Oracle improvement is headroom, not achievable savings.
- Optimiser cashflow uses retailer import/export prices; price-model accuracy uses the model's
  wholesale target. Keep both labels explicit. Invoice reconciliation remains separate from
  non-estimated archived Amber interval quotes.

## Ranked experiments

| Rank | Candidate | Why plausible here | Minimal paired experiment |
|---|---|---|---|
| 1 | Terminal/uncertainty policy with fixed incumbent forecasts | Repeated DH/MPC replanning can defer charging or retain unnecessary energy; exact endpoint equality materially shapes the whole trajectory | Bounded endpoint/lock-in variants separately from buy/sell weights; show trajectory, first action, realised bill, throughput and ending energy. Match common ending inventory for forecast comparisons; for endpoint-policy experiments also report raw cashflow and an independently specified terminal-energy value/sensitivity |
| 2 | Tail-only rolling residual correction | Cheapest correction to the actual incumbent; already demonstrated substantial but window-dependent price bias | Baseline versus additive rolling median/bias, then regularised ridge; 16.5–36h primary, 36–72h secondary; no APF prefix mutation; bounded corrections and unchanged fallback |
| 3 | Causal fundamentals/revision residual model | June diagnostics implicated interchange and renewable error, while STPASA already exists locally | Ridge versus small LGBM on identical features: incumbent level, horizon/calendar, STPASA wind/solar/net-load, as-issued revision and recent completed residuals; ablate one family at a time |
| 4 | Conditional buy/sell risk calibration | Nominal low/high bands may not provide useful protection in the relevant regime; risk weights currently enter actual tariffs | Calibrate delayed completed residuals by horizon and coarse price/solar regime; score pinball/coverage and decision regret. Hold terminal policy fixed; do not combine multiple conservative adjustments initially |
| 5 | Multi-memory ensemble | Short history adapts, long history stabilises rare regimes | Equal-weight short/long residual experts first; adaptive weights only if equal averaging helps. Update weights after labels arrive, with one representative vintage per target/horizon rather than hundreds of correlated rows |
| 6 | Explicit scarcity-event readiness | Rare missed exports can dominate annual value, but APF already supplies near-term market information | First label missed high-value intervals and energy available to export; test whether prior APF/AEMO revisions predict residual event risk. Track false charging/carrying cost as well as recall |
| 7 | Neural/foundation/APF-free replacement | Higher cost and weaker immediate pathway to gains with APF retained | Small offline probes only after cheaper residual/policy experiments leave identifiable gaps; no broad model sweep yet |

These are effort rankings, not measured financial rankings. Load-calibration economic replay proceeds
ahead of a new price architecture because current October evidence supports that cheap challenger.

## Concrete price-model design

- Current `_predict_with_dynamic_handoff` appends APF values to logged-price history and uses them
  as synthetic target observations. Training predominantly sees realised targets. Explicit APF
  summary/revision features plus an incumbent residual target are an inspectable alternative to
  retraining a recursive extrapolator; benefit is a hypothesis, not demonstrated error reduction.
- Use `actual_wholesale - as_issued_incumbent_wholesale` as additive residual target. Fit median
  residual for MAE or mean residual for mean-loss objectives deliberately; do not call one the other.
- First residual features need no new network: horizon, Adelaide solar/calendar category, incumbent
  level, recent *completed* residuals. Then add AEMO as-issued fundamentals and revision magnitudes.
- Retain negative prices. Report large-event outcomes separately; do not silently remove >$3,000/MWh
  labels or isotonic-clip away scarcity forecasts. Purcell implementation exclusions make its ordinary
  calibration score insufficient for this system's tail-risk objective.
- Do not calculate future forecast-error features from realised demand/interchange/PV. Those remain
  diagnostic attribution channels. An as-issued forecast level/revision is a causal candidate.
- Event preparation can use a price-curve fragility feature rather than a second broad point model.
  AEMO documents demand-offset scenarios, including SA1 ±50, ±100 and ±200 MW. Their scenario prices
  measure specified perturbations; they are **not calibrated probability quantiles**. Local ingestion,
  current report availability and receipt timing need verification before an experiment.
  [AEMO scenario definitions](https://www.aemo.com.au/-/media/files/electricity/nem/security_and_reliability/dispatch/policy_and_process/pre-dispatch-sensitivities.pdf),
  [MMS table](https://visualisations.aemo.com.au/aemo/nemweb/mmsdatamodelreport/electricity/mms%20data%20model%20report_files/MMS_277.htm).

## Other primary sources worth borrowing from

- **LEAR / epftoolbox**: recalibrated sparse autoregression is a useful cheap challenger and benchmark
  pattern. Its hourly next-day interface needs adapting to rolling half-hourly horizons; adding its
  dependency or porting its whole pipeline is unnecessary.
  [Executable implementation](https://github.com/jeslago/epftoolbox/blob/master/epftoolbox/models/_lear.py)
  (mutable master inspected Oct 4; pin a commit if code is reused).
- **Cornell et al., SA NEM Q-QRA**, arXiv `2311.07289v2`: combines probabilistic forecasts and different
  training memories, with an explicit household battery economic case. Strong domain inspiration,
  but its daily origin/24h study is different from current 5m APF and rolling DH/MPC control.
  [Versioned paper](https://arxiv.org/html/2311.07289v2).
- **El Mahtout & Ziel**, arXiv `2601.02856v3`, Mar 27 2026: partial online updates and Bernstein Online
  Aggregation of linear/nonlinear experts; evaluated Germany-Luxembourg and Spain. Borrow adaptive
  weighting only after cheap equal-weight experts prove useful locally; European hourly accuracy
  results are not evidence of South Australian household savings.
  [Versioned paper](https://arxiv.org/html/2601.02856v3).
- **O'Connor et al.**, arXiv `2502.04935v1`: time-series conformal/quantile ensembles include simulated
  battery financial comparisons in Irish markets. Importantly, standalone split conformal underperformed
  economically in their comparisons; nominal coverage alone is insufficient. Battery rules, costs,
  markets and constraints differ from this installation. Borrow paired decision scoring and delayed
  residual calibration, not reported profits or guaranteed coverage under arbitrary drift.
  [Versioned paper, §§5.3–5.4](https://arxiv.org/html/2502.04935v1).
- **BTU-EnerEcon fundamental-input repo**: econometric models consume a structural model's clearing
  price and are evaluated in storage dispatch. Here AEMO already provides a structural-price/fundamental
  feed; borrow the combination/ablation concept instead of building a second NEM clearing simulator.
  German day-ahead results do not establish local improvement.
  [Author repository](https://github.com/BTU-EnerEcon/Electricity_price_forecasting_with_fundamental_model_input)
  (README inspected Oct 4; no code imported or pinned performance reproduced).
- **Purcell** remains a lightweight calibration reference, pinned `07dccf06352732c34863a212e072619a34679755`.
  Existing [review](forecast_accuracy_review_2026-10-04.md) records routing/window/cap/spike caveats;
  this pass does not repeat or revalidate upstream implementation.

## Data/evaluation gates and immediate work

- Immediately testable offline: original-bundle policy sensitivities; causal rolling tail residual
  corrections once fresh independent wholesale targets are frozen; current load challenger score
  and dispatch comparisons on matching origins.
- Available: active combined `price_forecast_log.csv`, archived production quantile/model versions,
  current STPASA parquet, measured seven-day grid/load/battery targets, isolated DH→MPC core replay.
- Missing/unproven: recent independently verified wholesale labels; complete as-issued 5m APF low/
  median/high and tariff transformations; multi-cycle historical accepted PV/HWC/control-state lineage;
  AEMO actual receipt timestamps; annual event/seasonal coverage and invoice reconciliation.
- `run_time <= origin` proves model-run ordering, not publication or local ingestion availability.
  Existing STPASA backfill exports run/target timestamps without a first-seen timestamp. Use recorded
  receipt where possible; otherwise explicit delay and sensitivity, clearly labelled. AEMO describes
  hourly half-hourly STPASA publication, but that is not evidence of our feed's historical latency.
  [Official PASA outputs](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/market-management-system-mms-data/projected-assessment-of-system-adequacy-pasa).
- Existing `eval/ablate_stpasa_tail_features.py` creation-only split allows training labels beyond the
  validation-origin cutoff. Purge labels by target interval completion + receipt lag; fit scaler,
  residual corrector and ensemble weights inside each walk-forward history. Historical MAE results
  motivate replication, not adoption. Preserve one original forecast vintage as training identity.
- Score paired windows, day/event contributions, cost and ending inventory. For uncertainty/scenarios,
  preserve whole-trajectory residual dependence; independent interval quantiles are not price paths.

## Independent review: October load challenger

- Inspected `eval/calibrate_measured_load.py` and seven focused tests. **No temporal leakage found**
  in latest-vintage training selection: selected vintage creation ≤ its target; a training target
  releases only after 30m completion plus configured lag. If released before scored creation, its
  selected vintage must also be earlier than scored creation.
- Version/type/horizon groups separate; each training target contributes once; complete six-bin
  measured targets required. Future-label mutation test verifies earlier corrections unchanged.
- Caveats: receipt lag assumed; horizon-band conditional distributions can differ; independent quantile
  corrections can cross. Repeated origin/day scores are dependent. Target-balanced marginal accuracy
  is appropriate for diagnostics but does not substitute for one whole-plan economic outcome per origin.
- Short-horizon corrected p65 still has ~79% coverage in seven-day evidence. Do not infer a universally
  calibrated p65 or switch deployment quantile from longer-horizon results alone.
