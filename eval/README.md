# Evaluation Scripts

## Existing scripts

| Script | Purpose |
|--------|---------|
| `dispatch_simulator.py` | Rolling MPC LP backtester (scipy HiGHS, 40 kWh/10 kW battery). Phase 3: price-only 5-min. Phase 6: 30-min with net_load_actuals. |
| `build_holistic_eval_set.py` | Build stratified eval index (spike/low/normal) from InfluxDB + forecast log. |
| `holistic_eval.py` | Run holistic dispatch simulation: oracle / amber_apf_lgbm / p5min_naive. `--ai-source`: add TFT q50. `--hybrid-source`: add Tier 1+TFT hybrid. |
| `retro_tft_inference.py` | Retrospective TFT Tier 2 batch inference → `retro_tft_forecasts.pkl` ({ts → ndarray(144,6)}). |
| `retro_tier1_inference.py` | Retrospective Tier 1 LGBM inference → `retro_tier1_forecasts.pkl` ({ts → ndarray(2,)}). Uses parquet P5MIN + actuals; InfluxDB for PV only. |
| `compare_tft_dispatch.py` | TFT vs LightGBM dispatch comparison on 130 overlapping 30-min boundary runs (Phase 3). |
| `compare_load_forecast.py` | TFT vs LightGBM load forecast comparison. |
| `eval_load_overnight.py` | Load TFT overnight ramp diagnostics. |

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
| Tier 1 + TFT hybrid | $3.15 (+5.5%) | $7.22 (+5.8%) | $1.04 (+17.1%) | $0.38 (−27.8%) |
| TFT Tier 2 q50 (standalone) | $3.18 (+6.6%) | $7.22 (+5.8%) | $1.10 (+23.6%) | $0.41 (−21.1%) |
| P5MIN naive | $0.09 | $0.17 | −$0.01 | $0.13 |

*TFT Tier 2 q50 standalone archived in `holistic_eval_raw_{stratum}_ai.parquet`.
Current CSV (`holistic_eval_results.csv`) contains `tier1_tier2_hybrid` as the primary AI source.*

**Phase 8 financial gate thresholds** (vs Amber APF + LGBM baseline):
- Overall: ≥ $2.99/day (no regression)
- Spike: ≥ $6.48/day (−5% tolerance)
- Low: ≥ $0.87/day (−2%)
- Normal: ≥ $0.51/day (−2%)

**Gate status:** `tier1_tier2_hybrid` passes overall/spike/low. Fails normal (−27.8%).
Likely root cause: **debiaser inference path mismatch** — TFT was trained with OOF-debiased
`pd_rrp` at decoder steps 0–55 (`data/build_training_dataset.py:291`), but both
`retro_tft_inference.py` and `forecast.py` feed raw PREDISPATCH at inference. Fix this first
before building `lgbm_strategic` or attempting q50 calibration. See `docs/roadmap.md`.

**Eval statistics caveat:** Windows are drawn from an every-6h grid; 72h windows overlap
by ~66h. Results are directionally robust but not 811 independent trials. Tight per-stratum
thresholds (−2%) should not be over-interpreted without block-bootstrap confidence intervals.

**Performance notes:**
- `holistic_eval.py --fast` (50/stratum, LP): ~3 min with 12 workers
- Full run (811 windows, LP): ~12 min with 12 workers (`nice -n 19 --workers 12`)
- `--dispatch greedy` available for development (~100× faster, O(N log N), not for baselines)

### Simulator validation (optional)

HA's history database has recent actual battery dispatch (SOC, charge/discharge power).
Cross-check simulator output against 1–2 weeks of real HA history to validate battery
model fidelity before relying on holistic eval numbers.
