# Evaluation Scripts

## Existing scripts

| Script | Purpose |
|--------|---------|
| `dispatch_simulator.py` | Rolling MPC LP backtester (scipy HiGHS, 40 kWh/10 kW battery). Phase 3: price-only 5-min. Phase 6: tariffed site-flow mode with net load or split load/PV actuals. |
| `build_holistic_eval_set.py` | Build stratified eval index (spike/low/normal) from InfluxDB + forecast log. |
| `holistic_eval.py` | Run holistic dispatch simulation: oracle / amber_apf_lgbm / p5min_naive. `--ai-source`: add TFT q50. `--hybrid-source`: add Tier 1+TFT hybrid. |
| `retro_tft_inference.py` | Retrospective TFT Tier 2 batch inference → `retro_tft_forecasts.pkl` ({ts → ndarray(144,6)}). |
| `retro_tier1_inference.py` | Retrospective Tier 1 LGBM inference → `retro_tier1_forecasts.pkl` ({ts → ndarray(2,)}). Uses parquet P5MIN + actuals; InfluxDB for PV only. |
| `eval_tier1_accuracy.py` | **Pass A tactical eval**: MAE per horizon h0-h11 for Tier 1 vs p5min_direct/naive/oracle. Outputs `tier1_accuracy_by_horizon.csv` + `tier1_accuracy_summary.csv`. |
| `rolling_mpc_eval.py` | **Track A rolling MPC eval**: contiguous 5-min, SoC-carrying backtest scaffold for the execution-focused near horizon. Supports legacy `price_only` mode and newer `netload_tariffed` mode (actual load/PV plus separate import/feed-in economics), with regime, spike-band, coverage, and behavior diagnostics. For detached long runs, prefer `--workers 1` unless a short pilot has already validated a multi-worker setup; `--mp-start-method auto` now prefers `fork` on Linux. |
| `compare_rolling_mpc_raw.py` | Compare two Track A raw parquet outputs and report whether dispatch actually changed (`charge_kw`, `discharge_kw`, `soc_kwh`, terminal-contract columns). Useful as a cheap preflight before committing to long reruns. |
| `analyze_rolling_mpc_tariffed.py` | Source/day diagnostics for `rolling_mpc_eval.py --economic-mode netload_tariffed`: import/export energy, tariffed pnl decomposition, SoC posture, top charge/discharge events, a same-strategic-target/different-tactical daily delta table, and a missed-export interval report. |
| `build_tactical_action_regret_dataset.py` | Rebuild a per-step oracle-action / action-regret dataset from raw rolling-eval parquet using actual future tariffed prices plus the logged SoC and terminal constraints. Includes both first-action deltas and full-horizon forced-first-action regret fields. For long runs, use `--progress-every-rows` so logs show row-count progress and ETA. |
| `analyze_tactical_action_regret.py` | Summarize oracle-action datasets by bucket (`FIT >= 300/500`, negative net load, oracle exporting/charging) and report whether Hybrid or Amber is actually closer to the oracle first action, plus mean full-horizon first-action regret where available. |
| `analyze_dispatch_physical_feasibility.py` | Audit rolling raw parquet outputs for simple physical artifacts: simultaneous charge/discharge, simultaneous import/export, grid-balance residuals, SoC transition drift, curtailing more PV than available, and power/SoC bound violations. |
| `build_state_transition_label_dataset.py` | Build the first state-value / inventory-discipline label dataset from rolling raw parquet. Solves the realized-future tariffed oracle, then compares oracle/target/comparator 30-60 minute path metrics. Optional `--soc-finite-diff-kwh` adds a finite-difference marginal initial-SoC value label, at the cost of one extra LP solve per row. |
| `analyze_state_transition_labels.py` | Summarize state-transition label datasets by horizon: SoC movement, throughput/churn, import/export energy, prefix PnL, and direction rates for oracle/comparator relative to the target source. |
| `train_state_transition_value_model.py` | Train a small diagnostic LightGBM model on state-transition labels using production-side/current-time features. Reports whether oracle-vs-target path labels are learnable before any control integration. |
| `compare_tft_dispatch.py` | TFT vs LightGBM dispatch comparison on 130 overlapping 30-min boundary runs (Phase 3). |
| `compare_load_forecast.py` | TFT vs LightGBM load forecast comparison. |
| `eval_load_overnight.py` | Load TFT overnight ramp diagnostics. |

---

## Rolling MPC Run Hygiene

For `rolling_mpc_eval.py`, treat anything beyond a tiny smoke test as a managed background job:

- launch via detached `tmux`, not an assistant-attached foreground session
- use `PYTHONUNBUFFERED=1` so progress appears in the log promptly
- use `nice -n19` for long or parallel experiments
- write stdout/stderr to an `eval/results/*.log` file
- write the process exit code to a matching `eval/results/*.exitcode` file
- when using a shell wrapper, prefer `status=${PIPESTATUS[0]}; printf '%s\n' "$status" > ...`
  so the exit-code file contains a plain integer and not a malformed string
- for arbitrary detached runs, prefer `eval/run_rolling_mpc_managed.sh` over ad hoc nested shell
  snippets; it bakes in `nice`, `PYTHONUNBUFFERED`, log capture, and exit-code capture
- `eval/run_track10a_long.sh` supports `LOG_PATH=...` and `EXITCODE_PATH=...` for this pattern
- run a `1-3 day` pilot before promoting a new run shape to a full `6-week` window

Parallelism notes:

- `--workers` parallelizes across forecast sources, not timesteps
- common two-source runs can only usefully consume about two worker processes
- `--mp-start-method auto` prefers `fork` on Linux and has completed a real `2-day` pilot with
  `--workers 2`
- `--economic-mode netload_tariffed` is the first production-fidelity upgrade path for Track A:
  it uses actual 30-minute load/PV expanded to 5-minute site-flow inputs plus tariffed
  import/feed-in price curves for both the tactical solve and realized PnL
- site-flow mode supports explicit PV curtailment via `curtail_kw`. With split load/PV inputs,
  curtailment is bounded by available PV and can represent full PV turn-down while site load is
  positive; with net-load-only inputs, it falls back to surplus-only curtailment bounded by
  `max(0, -net_load_kw)`.
- `--progress-every-steps` controls how often the script prints elapsed/ETA progress lines and
  updates `eval/results/<output_prefix>_<source>.progress.json` checkpoint files
- still validate any new multi-worker run shape on a short pilot before leaving it unattended
- bridge-contract runs now support `--dynamic-bridge-terminal-scope extra_band`, which applies
  dynamic terminal value only to the terminal energy above the q50 floor inside band mode
- tactical sell-deferral probes can now use the `--sell-urgency-*` flags in
  `netload_tariffed` mode to leave step 0 unchanged while discounting only future export prices
  for selected tactical sources during already-high feed-in intervals; this is intended as a
  small export-monetization stress test rather than a general forecast-model change
- crossed tactical/strategic counterfactuals are supported via built-in source aliases:
  - `hybrid_tactical_amber_strategic`
  - `amber_tactical_hybrid_strategic`
  - or the generic form `cf:<label>:<tactical_source>:<strategic_source>`
  This keeps the tactical forecast curve and strategic handoff source separable inside the same
  rolling-eval harness
- tactical model candidates can now be evaluated side-by-side with the baseline by pointing
  `rolling_mpc_eval.py` at an alternate Tier 1 artifact directory via `--tactical-model-dir`;
  the same pattern is available in `retro_tier1_inference.py`, `eval_tier1_accuracy.py`,
  `eval_tier1_dispatch.py`, and `train_lgbm_tactical.py --model-dir`

Before promoting a control variant to a long run, compare raw pilot outputs with
`compare_rolling_mpc_raw.py`. If `charge_kw`, `discharge_kw`, and `soc_kwh` are unchanged,
the variant is probably not worth a long rerun unless the goal is purely diagnostic.

After a `netload_tariffed` pilot, use `analyze_rolling_mpc_tariffed.py` to answer the next
question: where did the winning source buy lower, sell higher, or hold a different SoC posture?

When the open question is no longer “which source won?” but “what action error was made?”,
switch to the oracle-action flow:

1. build a per-step oracle-action dataset with `build_tactical_action_regret_dataset.py`
2. summarize Hybrid-vs-Amber oracle closeness with `analyze_tactical_action_regret.py`
3. only then decide whether the next label should be first-action correction, multi-step regret,
   or a richer action-ranking target

For long oracle builds, add `--progress-every-rows 100` (or similar) so the detached log
shows elapsed time and rough ETA instead of staying silent for hours.

For the state-transition label branch, start with `build_state_transition_label_dataset.py`
on a small `--max-rows` smoke. Full Window B runs are LP-heavy, and finite-difference SoC labels
roughly double the solve count, so use detached `tmux`, `.venv`, logs, and `.exitcode` files
before running them unattended.

The current target bucket from the forced-prefix attribution is:

```bash
./.venv/bin/python eval/build_state_transition_label_dataset.py \
  --raw rolling_mpc_eval_counterfactual_windowb_7day_netload_011b_curtail_20260501_raw.parquet \
  --horizons 6,12 \
  --feed-in-max-mwh 300 \
  --net-load-max-kw 0 \
  --output-prefix state_transition_wb7_fitlt300_negload_curtail_20260501
```

Use `--max-rows` for smoke tests. Add `--soc-finite-diff-kwh 1.0` only after the cheaper
path-label distribution is worth expanding.

Then run the first diagnostic model pass:

```bash
./.venv/bin/python eval/train_state_transition_value_model.py \
  --labels state_transition_wb7_fitlt300_negload_curtail_20260501_state_transition_labels.parquet \
  --output-prefix state_transition_wb7_fitlt300_negload_curtail_20260501
```

Treat this as a signal test, not a deployable controller. The model uses current-time/site/forecast
summary features and predicts oracle-vs-target path labels such as prefix PnL, SoC delta, churn,
import/export, and curtailment deltas.

---

## Phase 6 — Holistic Dispatch Simulation

**Goal:** Produce a $/day simulated profit comparison across forecast sources on a stratified
eval set. This is the financial baseline that gates further pipeline evolution.

### Forecast sources

| Source | Identifier | How obtained |
|--------|-----------|-------------|
| Amber APF + LGBM extrapolation | `amber_apf_lgbm` | `price_forecast_log.csv` — Amber APF seeds first ~14-28h; LightGBM model extends to 72h |
| TFT Tier 2 q50 dispatch | `tft_tier2_q50` | Retrospective batch inference: `eval/retro_tft_inference.py` |
| Tier 1 + TFT Tier 2 hybrid | `tier1_tier2_hybrid` | Tier 1 LGBM q50 steps 0–1, TFT q50 steps 2–143. Run `retro_tier1_inference.py` + `retro_tft_inference.py` first. |
| Oracle | `oracle` | AEMO dispatch actuals — perfect foresight upper bound |
| P5MIN naive | `p5min_naive` | Window-start price held constant for all 144 steps |

**Source terminology:**
- `amber_apf_lgbm` is the as-run *production* system, not a pure LightGBM model.
  Amber's commercial APF drives the short-horizon signal; LGBM extrapolates beyond Amber's range.
  The logged predictions cannot be decomposed back into Amber-only vs LGBM-only components.
- `tft_tier2_q50` evaluates TFT as a standalone dispatch signal (q50 only). This is *not* how
  TFT is intended to be used in production.
- `tier1_tier2_hybrid` is the intended production architecture: Tier 1 LGBM handles the first
  2 × 30min steps (0–60 min) using P5MIN as primary signal, TFT handles 1h–72h. This is the
  Amber-independent target architecture.

**Data availability notes:**
- `rp_5m.aemo_p5min_forecast` in InfluxDB: only retained from **April 12, 2026** onwards
  (rp_5m retention = 3 years, but ingest of P5MIN to InfluxDB started April 2026).
  Retrospective P5MIN data for the eval period comes from `data/parquet/aemo_p5min_sa1.parquet`
  (March 2024 – March 2026). Use this parquet for any retrospective inference, not InfluxDB.
- `rp_5m.aemo_dispatch_sa1_5m` (5min actuals): parquet covers March 2024 – April 2026.
- `rp_5m.power_pv_5m`: InfluxDB covers **July 2, 2025 onwards** (eval period starts July 21, 2025).

**Key constraint:** Historical Amber APF forecasts are not stored — only the combined predictions
are available in `price_forecast_log.csv`. Comparison window: **July 2025 onwards**.

**P5MIN naive in Phase 6:** P5MIN only covers ~1h ahead. For 72h windows at 30-min resolution,
the naive baseline uses the price at window-start held constant for all 144 steps (persistence
forecast). This is a conservative baseline — a real P5MIN-only system would do better short-term.

### Eval set construction (`build_holistic_eval_set.py`)

New stratified sample from the overlapping data window. Not the existing 900-sample TFT
eval set — that predates P5MIN/Tier 1 availability and has no matched load/PV data.

**Stratification** (by actual SA1 dispatch price within each 72h window):
- **Spike:** ≥1 interval with RRP ≥ $300/MWh
- **Low/negative:** ≥1 interval with RRP ≤ −$50/MWh (genuine curtailment)
  - Note: SA midday prices are routinely $0–$50/MWh due to solar; $0 threshold left only
    9 normal windows out of 853. −$50 captures genuine negative-price curtailment events.
- **Normal:** all other windows

Actual index (July 2025 – April 2026): 300 spike, 300 low, 211 normal (811 total).
Output: `eval/results/holistic_eval_index.parquet`.

### Simulator enhancement (`dispatch_simulator.py`)

Extend `simulate_mpc()` to accept `net_load_actuals`:

```python
def simulate_mpc(price_forecast, actual_prices, net_load_actuals=None,
                 battery_kwh=40, max_kw=10, degradation_per_kwh=0.05, ...):
    # net_load_actuals[t] = power_load_30m[t] - power_pv_30m[t]  (kW)
    # positive = drawing from grid, negative = exporting to grid
    # LP objective: minimise sum_t(
    #     max(0, net_load[t] + charge[t] - discharge[t]) * actual_price[t]   # import cost
    #     - max(0, -(net_load[t] + charge[t] - discharge[t])) * actual_price[t]  # export revenue
    #     + degradation_per_kwh * (charge[t] + discharge[t])                 # degradation
    # )
```

Load and PV are **fixed actuals** (not forecast inputs) — we are testing price forecast
quality. Pull from `rp_30m.power_load_30m` and `rp_30m.power_pv_30m` in InfluxDB.

Degradation: linear $0.05/kWh throughput (total charge + discharge). No quadratic penalty.

If `net_load_actuals=None`, simulator falls back to price-only mode (backwards compatible).

### Reporting (`holistic_eval.py`)

For each eval window × forecast source:
1. Query actual load/PV from InfluxDB for the window
2. Load price forecast (from log CSV or retrospective inference)
3. Run `simulate_mpc()` → simulated profit ($/period)
4. Aggregate to $/day by stratum

Output table (printed + saved to `eval/results/holistic_eval_results.csv`):

**Results** (811 windows, July 2025–March 2026, price-only LP MPC):

| Source | Mean $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $6.00 | $11.97 | $2.77 | $2.12 |
| **Amber APF + LGBM (baseline)** | **$2.99** | **$6.82** | **$0.89** | **$0.52** |
| **Tier 1 + TFT hybrid (spike classifier, threshold=0.65)** | **$3.28 (+9.7%)** | **$7.31 (+7.2%)** | **$1.18 (+32.6%)** | **$0.52 (+0.4%)** |
| TFT Tier 2 q50 (standalone, archived) | $3.18 (+6.6%) | $7.22 (+5.8%) | $1.10 (+23.6%) | $0.41 (−21.1%) |
| P5MIN naive | $0.09 | $0.17 | −$0.01 | $0.13 |

*TFT Tier 2 q50 standalone archived in `holistic_eval_raw_{stratum}_ai.parquet`.
Current CSV (`holistic_eval_results.csv`) contains `tier1_tier2_hybrid` as the primary AI source.*

*Debiaser configuration: OOF debiased `pd_rrp` at steps 0–55; upstream LightGBM spike
classifier (`models/spike_classifier/lgbm_spike_clf.pkl`, threshold=0.65) routes each
run_time to debiaser or raw PREDISPATCH passthrough. See `docs/review_debiaser_spike_guard.md`.*

*Reproducibility: results use `eval/results/holistic_eval_actuals.parquet` (frozen from
InfluxDB 2026-04-19). Run `eval/export_holistic_actuals.py` to refresh the snapshot.*

**Companion net-load run** (same 811 windows, frozen actuals, load+PV included):

| Source | Mean $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $4.73 | $10.36 | $1.27 | $1.65 |
| **Amber APF + LGBM (baseline)** | **$1.72** | **$5.21** | **−$0.61** | **$0.05** |
| **Tier 1 + TFT hybrid** | **$2.01 (+16.9%)** | **$5.70 (+9.5%)** | **−$0.32 (+$0.29 abs)** | **$0.05 (+4.1%)** |

*Net-load results in `eval/results/holistic_eval_results_netload.csv` and `holistic_eval_raw_netload.parquet`.
Lower absolute values vs price-only because net load cost/revenue replaces pure price arbitrage.
Gate thresholds and primary results remain price-only (matching the established baseline).*

**Phase 8 financial gate thresholds** (vs Amber APF + LGBM baseline):
- Overall: ≥ $2.54/day (−15% tolerance)
- Spike: ≥ $5.46/day (−20% tolerance)
- Low: ≥ $0.87/day (−2%)
- Normal: ≥ $0.51/day (−2%)

**Gate status (2026-04-19): ALL GATES PASS ✅.** Phase 5 remainder unblocked.

Previous results (scalar 1000 $/MWh spike guard): normal −21.7% ❌. Root cause and
resolution documented in `docs/review_debiaser_spike_guard.md`.

---

### LightGBM Strategic Model (exploration track, 2026-04-19)

**Goal:** Compare a pure LightGBM 30-min/72-hour model against TFT to diagnose whether
TFT's long-horizon shape anomalies reflect a structural problem or are incidental.

**Architecture:** Single LightGBM quantile model (q5/q50/q95) trained in long format —
one row per `(run_time, step_idx)`. Features: `step_idx`, OOF-debiased PREDISPATCH for
steps 0–55 (28h), time+lag features for all 144 steps. Training script:
`train/train_lgbm_strategic.py`. Val MAE: $39.87/MWh (all steps, last 60 days).

**Pass 1 results — no spike routing** (OOF-debiased PREDISPATCH applied uniformly):

| Source | Mean $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $6.00 | $11.97 | $2.77 | $2.12 |
| Amber APF + LGBM (baseline) | $2.99 | $6.82 | $0.89 | $0.52 |
| Tier 1 + TFT hybrid | $3.28 (+9.7%) | $7.31 (+7.2%) | $1.18 (+32.6%) | $0.52 (+0.4%) |
| **lgbm_strategic (no routing)** | **$2.17 (−27.5%)** | **$4.66 (−31.7%)** | **$0.78 (−12.2%)** | **$0.59 (+13.4%)** |

*Results in `eval/results/holistic_eval_results_lgbm_strategic.csv`.*

**Key finding:** The spike failure is the same root cause as TFT before spike classifier
routing — the OOF debiaser suppresses genuine high PREDISPATCH prices during spike windows.
Normal stratum is notably strong (+13.4% vs amber), matching the expectation that a
PREDISPATCH-grounded LightGBM corrects normal-period overestimates cleanly.

**Pass 2 results — consistent spike routing** (routing applied in both training and inference;
14,603/56,434 run_times bypass debiaser at threshold=0.65):

| Source | Mean $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $6.00 | $11.97 | $2.77 | $2.12 |
| Amber APF + LGBM (baseline) | $2.99 | $6.82 | $0.89 | $0.52 |
| **lgbm_strategic (routed)** | **$2.02 (−32.4%)** | **$4.26 (−37.5%)** | **$0.79 (−11.7%)** | **$0.59 (+13.6%)** |

*Results in `eval/results/holistic_eval_results_lgbm_strategic_routed.csv`.*

**Key finding (Pass 2):** Consistent routing made spike performance *worse* (−37.5% vs −31.7%
without routing). Normal stratum held steady (+13.6%). This indicates the problem is not
train/inference distribution mismatch — the LightGBM strategic model has a structural
limitation for spike events. Likely causes: (1) no attention/memory mechanism to propagate
spike information beyond the 28h PREDISPATCH coverage window; (2) training the model on a
bimodal PREDISPATCH distribution (OOF-debiased vs raw 5000 $/MWh) creates hard-to-separate
tree splits. The TFT hybrid remains the better architecture for spike handling.

**Eval statistics caveat:** Windows are drawn from an every-6h grid; 72h windows overlap
by ~66h. Results are directionally robust but not 811 independent trials. Tight per-stratum
thresholds (−2%) should not be over-interpreted without block-bootstrap confidence intervals.

**Performance notes:**
- `holistic_eval.py --fast` (50/stratum, LP): ~3 min with 12 workers
- Full run (811 windows, LP): ~12 min with 12 workers (`nice -n 19 --workers 12`)
- `--dispatch greedy` available for development (~100× faster, O(N log N), not for baselines)

---

## Tactical Eval (Pass A) — 5-min Forecast Accuracy

**Goal:** Validate Tier 1 LGBM at 5-min/1h resolution, required prerequisite for Amber APF
replacement alongside Phase 6.

**Script:** `eval/eval_tier1_accuracy.py`

**Data:** `data/parquet/aemo_p5min_sa1.parquet` + `data/parquet/actuals_sa1_5m.parquet`
(no InfluxDB required; price-only first pass, PV excluded).

**Sources compared:**

| Source | Description |
|--------|-------------|
| `tier1_q50` | Tier 1 LGBM q50 corrections applied to P5MIN |
| `p5min_direct` | Raw P5MIN rrp unchanged |
| `p5min_naive` | Persistence: h0 price held constant for all 12 horizons |
| `oracle` | Actual 5-min rrp at each interval |

**Stratum classification:** based on actual rrp over the 12-step window.
Same thresholds as Phase 6: spike ≥ $300/MWh, low ≤ −$50/MWh.

**Actuals coverage note:** `actuals_sa1_5m.parquet` has ~88% per-interval coverage in the
Jul 2025–Mar 2026 eval period (clustered gaps from ingest outages). MAE is computed per
horizon step using only windows where that step's actual is available.

**Results** (Jul 2025–Mar 2026, ~77k windows, per-step NaN excluded):

| Source | All MAE | Spike MAE | Low MAE | Normal MAE | Skill vs naive |
|--------|---------|-----------|---------|------------|----------------|
| Oracle | $0.0 | $0.0 | $0.0 | $0.0 | +100% |
| **tier1_q50** | **$32.3** | **$356.5** | **$34.8** | **$15.5** | **+24.4%** |
| p5min_naive | $42.8 | $441.4 | $45.2 | $22.1 | — |
| p5min_direct | $46.2 | $517.1 | $46.0 | $22.1 | −8.0% |

**Per-horizon MAE (stratum=all):**

| Horizon | tier1_q50 | p5min_direct | p5min_naive |
|---------|-----------|--------------|-------------|
| h0 (0 min) | $25.9 | $33.7 | $33.7 |
| h1 (5 min) | $26.8 | $37.9 | $34.8 |
| h3 (15 min) | $29.1 | $40.4 | $37.8 |
| h5 (25 min) | $31.5 | $43.7 | $41.9 |
| h7 (35 min) | $34.0 | $46.9 | $45.6 |
| h11 (55 min) | $39.2 | $62.7 | $51.9 |

**Gate:** Pass A criteria — Tier 1 MAE < p5min_naive at h1–h5. **PASSES ✅ on all 12
horizons, all strata.** Tier 1 beats naive by 19–30% depending on stratum; p5min_direct
is worse than naive from h1 onwards.

**Output files:** `eval/results/tier1_accuracy_by_horizon.csv`,
`eval/results/tier1_accuracy_summary.csv`

---

## Tactical Eval (Pass B) — 5-min Dispatch Value

**Script:** `eval/eval_tier1_dispatch.py`

**Method:** 300 windows/stratum stratified sample, rolling LP MPC (`dispatch_simulator.simulate_mpc`),
revenue booked against actual prices. NaN actuals filled with P5MIN fallback.
$/day normalised (window = 60 min = 1/24 day).

**Results** (900 windows, Jul 2025–Mar 2026, price-only LP MPC, seed=42):

| Source | All $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $40.30 | $106.58 | $7.21 | $7.12 |
| **tier1_q50** | **$38.71 (+1.4%)** | **$104.98 (+0.7%)** | **$4.48 (+15.7%)** | **$6.67 (+4.2%)** |
| p5min_naive | $38.18 | $104.26 | $3.88 | $6.40 |
| p5min_direct | $38.11 (−0.2%) | $104.33 (+0.1%) | $3.51 (−9.5%) | $6.51 (+1.6%) |

**Gate:** Pass B criteria — Tier 1 dispatch revenue ≥ p5min_naive (no regression).
**PASSES ✅ all three strata.** Low stratum shows largest gain (+15.7%) — Tier 1
correctly avoids discharging into negative-price windows. p5min_direct regresses on
low (−9.5%) because raw P5MIN noise misleads the LP at negative-price horizons.

**Output files:** `eval/results/tier1_dispatch_results.csv`,
`eval/results/tier1_dispatch_summary.csv`

**Tactical eval complete.** Both Pass A (accuracy) and Pass B (dispatch value) pass.
Tier 1 is validated for the 5-min/1h tactical role. Combined with Phase 6 (30-min/72h
strategic), the dual prerequisite for Amber APF replacement is met from an eval standpoint.

### Simulator validation (optional)

HA's history database has recent actual battery dispatch (SOC, charge/discharge power).
Cross-check simulator output against 1–2 weeks of real HA history to validate battery
model fidelity before relying on holistic eval numbers.

---

## Rolling MPC Eval (planned)

The next evaluation layer is a contiguous rolling MPC backtest that carries SoC forward and
operates on the same `14h × 5-min` decision horizon as production. This is intentionally split
into two tracks.

`rolling_mpc_eval.py` also supports source-level parallelism via `--workers`, but only across
forecast sources. Timesteps within a source remain serial because SoC is path-dependent.
Startup is relatively expensive because each worker loads its own parquet/model state, so
`--workers > 1` is mainly useful for longer comparative runs rather than short smoke tests.

Current outputs:
- `{prefix}_raw.parquet`: one row per executed 5-minute step and source
- `{prefix}_summary.csv`: whole-window totals by source
- `{prefix}_summary_vs_baseline.csv`: whole-window totals with delta/ratio columns vs the chosen baseline
- `{prefix}_daily_summary.csv`: per-day PnL / SoC / average dispatch by source
- `{prefix}_daily_summary_vs_baseline.csv`: per-day deltas vs the chosen baseline source
- `{prefix}_daily_regimes.csv`: realized daily price regime labels (`spike` / `low` / `normal`)
- `{prefix}_regime_summary.csv`: aggregate results by source and realized regime
- `{prefix}_regime_summary_vs_baseline.csv`: regime-level deltas vs the chosen baseline
- `{prefix}_spike_band_summary.csv`: aggregate results by source and spike severity bucket
- `{prefix}_spike_band_summary_vs_baseline.csv`: spike-severity deltas vs the chosen baseline
- `{prefix}_behavior_summary.csv`: by-regime charge/discharge/SoC posture summary
- `{prefix}_behavior_summary_vs_baseline.csv`: by-regime behavior deltas vs the chosen baseline,
  including realized average charge/discharge prices
- `{prefix}_coverage.csv`: expected vs executed steps by source, useful for spotting missing
  forecast coverage. Includes skip counters for missing actuals, missing forecast curves, invalid
  forecast curves, and repaired-curve counts when present.

### Track A — Model A / execution track

- Goal: evaluate the execution-relevant near horizon with the longest available history
- Cadence: `5-minute` stepping, continuous SoC carryover
- Forecast semantics: current interval price treated as known; first hour from Tier 1 tactical
  forecast; remaining horizon supplied by a near-horizon strategic extension and repeated into
  5-minute slots where needed
- History: use the dense-history PREDISPATCH/P5MIN/actuals window rather than waiting on PD7Day

### Track B — full Phase 7 / planning track

- Goal: evaluate the intended stitched Tier1+Tier2 architecture under rolling MPC
- Cadence: `5-minute` stepping, continuous SoC carryover
- Forecast semantics: Tier 1 supplies the first 12 five-minute steps; Tier 2 30-minute steps
  are repeated across 6 × 5-minute slots for the rest of the 14h horizon
- History: starts at `2026-02-09` because PD7Day is required for the current Phase 7 Tier 2

Recommended build order: **Track A first**, then Track B once the core rolling machinery and
reporting format are stable.

**Observed Track A results so far:**
- **Window A** (`2025-07-21 → 2025-09-01`):
  - `amber_apf_lgbm`: **$2.523/day**
  - `model_a_hybrid`: **$2.585/day** (**+2.4%** vs amber)
- **Window B** (`2025-09-01 → 2025-10-13`):
  - `amber_apf_lgbm`: **$2.406/day**
  - `model_a_hybrid`: **$2.134/day** (**−11.3%** vs amber)

Window A regime view from `rolling_mpc_eval_tracka_6week_compare_regime_summary_vs_baseline.csv`:
- `spike`: hybrid better than amber (**$3.601/day** vs **$3.312/day**, +8.7%)
- `low`: near tie with slight hybrid edge (**$1.576/day** vs **$1.562/day**, +0.9%)
- `normal`: hybrid worse than amber (**$1.556/day** vs **$1.963/day**, −20.8%)

Window B regime view from `rolling_mpc_eval_tracka_followup_6week_regime_summary_vs_baseline.csv`:
- `low`: hybrid worse than amber (**$1.446/day** vs **$1.521/day**, −4.9%)
- `normal`: hybrid worse than amber (**$1.446/day** vs **$1.704/day**, −15.2%)
- `spike`: hybrid worse than amber (**$4.073/day** vs **$4.617/day**, −11.8%)

Window B spike-band view from `rolling_mpc_eval_tracka_followup_6week_spikebands_spike_band_summary_vs_baseline.csv`:
- `spike_moderate`: hybrid worse than amber (**$4.361/day** vs **$4.701/day**, −7.2%)
- `spike_extreme`: hybrid worse than amber on the single extreme day (**$1.195/day** vs **$3.782/day**)

**Current reading:** Track 10A is useful but mixed. The hybrid does not yet show a robust,
window-stable edge over Amber on the execution track.

Behavioral diagnosis from `rolling_mpc_eval_tracka_followup_6week_behavior_prices_behavior_summary_vs_baseline.csv`:
- `low`: hybrid charged less than amber and built less end-of-day energy (`soc_delta`
  **7.50 kWh** vs **10.78 kWh**); average charge price was slightly better than amber, so the
  loss looks more like weaker inventory build than obviously worse entry timing
- `normal`: hybrid charged slightly more but discharged less, started with lower SoC, and
  depleted less over the day (`soc_delta` **−2.64 kWh** vs **−5.17 kWh**); average discharge
  price was also worse than amber, consistent with weaker monetisation
- `spike`: hybrid was more active than amber (higher charge, discharge, dispatch intensity, and
  SoC posture) yet still earned less; average charge price was less negative and average
  discharge price was lower than amber, pointing to worse spread capture

Working hypothesis:
- the hybrid is not simply "doing less"
- on `low` days it appears to under-accumulate energy
- on `normal` days it appears to monetise stored energy less effectively and maintain a weaker
  SoC posture
- on `spike` days it appears active enough, but mistimed relative to amber

Potential next experiment:
- test an execution-layer opportunity-cost bias using the SoC shadow price / LP dual, so the
  controller becomes less willing to discharge when the marginal future value of stored energy
  is high
- `rolling_mpc_eval.py` now includes an experimental `--terminal-energy-value-mwh` flag, which
  adds a simple end-of-horizon salvage value for stored energy. This is not yet a dual-driven
  policy, but it is a useful first ablation hook for opportunity-cost-aware dispatch.

Window B salvage-value sweep (`rolling_mpc_eval_tracka_followup_6week_te{0,50,100,150,200}_summary_vs_baseline.csv`):
- `0 $/MWh`: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- `50 $/MWh`: hybrid **$2.254/day** vs amber **$2.393/day** (**−5.8%**)
- `100 $/MWh`: hybrid **$2.306/day** vs amber **$2.385/day** (**−3.3%**)
- `150 $/MWh`: hybrid **$2.287/day** vs amber **$2.383/day** (**−4.0%**)
- `200 $/MWh`: hybrid **$2.249/day** vs amber **$2.355/day** (**−4.5%**)

Interpretation:
- a simple inventory-value bias recovers a large fraction of the hybrid's loss on the bad
  follow-up window
- the benefit peaks around **`100 $/MWh`** in this coarse sweep
- that is strong enough evidence to move from fixed salvage-value sweeps to a more principled
  dual-driven opportunity-cost variant

Dual-driven follow-up:
- `rolling_mpc_eval.py` now also supports `--dual-terminal-scale`
- per MPC step, it first probes the LP with zero terminal value, reads the initial-SoC shadow
  price, and then re-solves with `terminal_energy_value = max(0, scale * shadow_price)`
- the raw parquet now records both `probe_initial_soc_shadow_price_per_kwh` and
  `control_initial_soc_shadow_price_per_kwh` so the policy can be inspected after a run

Window B dual sweep (`rolling_mpc_eval_tracka_followup_6week_dual{05,10,15,25,30}_summary_vs_baseline.csv`):
- `dual 0.5`: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- `dual 1.0`: hybrid **$2.140/day** vs amber **$2.411/day** (**−11.3%**)
- `dual 1.5`: hybrid **$2.256/day** vs amber **$2.364/day** (**−4.6%**)
- `dual 2.5`: hybrid **$2.236/day** vs amber **$2.360/day** (**−5.3%**)
- `dual 3.0`: hybrid **$2.245/day** vs amber **$2.363/day** (**−5.0%**)

Interpretation:
- the dual signal is not useless, but this first adaptive controller still underperforms the
  best fixed salvage-value proxy (`100 $/MWh`, **−3.3%**)
- the effective applied terminal values were too small at low scales and still not enough to
  beat the static proxy at higher scales
- treat the fixed terminal-value approach as a useful surrogate in Track 10A, not yet as the
  likely production design endpoint

Independent review checkpoint (2026-04-21):
- an external review, based on the standalone briefing in `docs/independent_review_brief_2026-04-21.md`,
  judged the static terminal-value win most likely to be an **eval surrogate** for the missing
  strategic `14h` SoC boundary condition
- on that reading, the next priority is to align the rolling eval with the described production
  SoC handoff before drawing stronger architectural conclusions from the terminal-value sweeps

Window B strategic-handoff rerun:
- pre-handoff: hybrid **$2.134/day** vs amber **$2.406/day** (**−11.3%**)
- handoff `exact`: hybrid **$2.271/day** vs amber **$2.451/day** (**−7.4%**)
- handoff `floor`: hybrid **$2.271/day** vs amber **$2.450/day** (**−7.3%**)

Interpretation:
- the reviewer checkpoint was directionally right: adding the strategic `14h` SoC handoff removes
  a meaningful part of the Window B deficit
- but it does not remove all of it, so the earlier terminal-value result was not purely an eval
  artifact
- the biggest improvement appears on `spike` days; `low` and `normal` remain materially weaker

Operational consequence:
- Track 10A should now treat the strategic `14h` SoC handoff as part of the aligned baseline
- any future Option A/B/C experiments should be judged against the **handoff-enabled** setup, not
  the older no-handoff variant

Option B implementation status:
- `rolling_mpc_eval.py` now includes fixed **upper-tail quantile blend** hooks for the
  handoff-enabled baseline:
  - `--tier1-quantile-blend`
  - `--tier2-quantile-blend`
  - `--tier2-upper-quantile {0.90,0.95}`
- current Stage B0 semantics:
  - first hour: tactical effective price = `q50 + w1 * (q95 - q50)`
  - strategic extension inside the 14h tactical solve =
    `q50 + w2 * (q_hi - q50)`
- the strategic `14h` SoC handoff remains unchanged during these sweeps; the blend only changes
  the price path presented to the 14h tactical LP
- recommended next step: run fixed-weight sweeps on the handoff-enabled Track 10A baseline before
  introducing any dynamic posture signal

Amber data-quality note:
- historical Amber forecasts in `price_forecast_log.csv` showed timestamp jitter and some
  partially invalid expanded curves
- `rolling_mpc_eval.py` now normalizes Amber target timestamps to the intended `30min` grid and
  repairs finite gaps before dispatch
- Window A achieved full coverage with **241 repaired curves** and **0 skipped steps**
- Window B achieved full coverage with **0 repaired curves** and **0 skipped steps**
