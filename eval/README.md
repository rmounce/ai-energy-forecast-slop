# Evaluation Scripts

## Existing scripts

| Script | Purpose |
|--------|---------|
| `dispatch_simulator.py` | Rolling MPC LP backtester (scipy HiGHS, 40 kWh/10 kW battery). Phase 3: price-only 5-min. Phase 6: 30-min with net_load_actuals. |
| `build_holistic_eval_set.py` | Build stratified eval index (spike/low/normal) from InfluxDB + forecast log. |
| `holistic_eval.py` | Run holistic dispatch simulation: oracle / lgbm_legacy / p5min_naive. Reports $/day by stratum. |
| `compare_tft_dispatch.py` | TFT vs LightGBM dispatch comparison on 130 overlapping 30-min boundary runs (Phase 3). |
| `compare_load_forecast.py` | TFT vs LightGBM load forecast comparison. |
| `eval_load_overnight.py` | Load TFT overnight ramp diagnostics. |

---

## Phase 6 — Holistic Dispatch Simulation

**Goal:** Produce a $/day simulated profit comparison across forecast sources on a stratified
eval set. This is the financial baseline that gates further pipeline evolution.

### Forecast sources

| Source | How obtained |
|--------|-------------|
| Legacy LightGBM + Amber APF seed | Read from `price_forecast_log.csv` — Amber APF seed is embedded in logged predictions as they ran |
| Tier 1 + Tier 2 AI pipeline | Retrospective inference on InfluxDB data (PREDISPATCH + SDO from March 2025, P5MIN from April 2024) |
| Oracle | AEMO dispatch actuals from InfluxDB — perfect foresight upper bound |
| AEMO P5MIN naive | `rp_5m.aemo_p5min_forecast` — short-horizon naive baseline |

**Key constraint:** Historical Amber APF forecasts are not stored — only the already-seeded
LightGBM predictions are available via forecast logs. Comparison window: **July 2025 onwards**
(when the `price_forecast_log.csv` LightGBM log begins).

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

**Phase 6 baseline results** (811 windows, July 2025–March 2026, price-only LP MPC):

| Source | Mean $/day | Spike $/day | Low $/day | Normal $/day |
|--------|-----------|------------|----------|-------------|
| Oracle | $6.00 | $11.97 | $2.77 | $2.12 |
| **LightGBM (baseline)** | **$2.99** | **$6.82** | **$0.89** | **$0.52** |
| P5MIN naive | $0.09 | $0.17 | −$0.01 | $0.13 |

Note: Tier 1+2 AI source not yet in holistic_eval — TFT log only has 6 days of
data (Apr 2026). Retrospective TFT batch inference planned for Phase 6 extension.

**Phase 8 financial gate thresholds** (from this baseline):
- AI pipeline overall: ≥ $2.99/day (no worse than lgbm_legacy)
- Spike: ≥ $6.48/day (−5% tolerance)
- Low: ≥ $0.87/day (−2%)
- Normal: ≥ $0.51/day (−2%)

**Performance notes:**
- `holistic_eval.py --fast` (50/stratum, LP): ~3 min with 12 workers
- Full run (811 windows, LP): ~9 min with 12 workers (`nice -n 19 --workers 12`)
- `--dispatch greedy` available for development (~100× faster, O(N log N), not for baselines)

### Simulator validation (optional)

HA's history database has recent actual battery dispatch (SOC, charge/discharge power).
Cross-check simulator output against 1–2 weeks of real HA history to validate battery
model fidelity before relying on holistic eval numbers.
