# Evaluation Scripts

## Existing scripts

| Script | Purpose |
|--------|---------|
| `dispatch_simulator.py` | Rolling MPC LP backtester (scipy HiGHS, 40 kWh/10 kW battery). Currently price-only. |
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
LightGBM predictions are available via forecast logs. Comparison window: March 2025 onwards.

### Eval set construction (`build_holistic_eval_set.py`)

New stratified sample from the overlapping data window. Not the existing 900-sample TFT
eval set — that predates P5MIN/Tier 1 availability and has no matched load/PV data.

**Stratification** (by actual SA1 dispatch price within each 72h window):
- **Spike:** ≥1 interval with RRP ≥ $300/MWh
- **Low/negative:** ≥1 interval with RRP ≤ $0/MWh
- **Normal:** all other windows

Target ~300 windows per stratum. Output: index of (start_time, stratum) tuples saved to
`eval/results/holistic_eval_index.parquet`.

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

| Source | Mean $/day | Spike $/day | Low $/day | Normal $/day | vs Baseline |
|--------|-----------|------------|----------|-------------|-------------|
| Oracle | ... | ... | ... | ... | +X% |
| Tier 1+2 AI | ... | ... | ... | ... | +X% |
| Legacy LightGBM | ... | ... | ... | ... | baseline |
| P5MIN naive | ... | ... | ... | ... | −X% |

**This table sets the financial thresholds for the Phase 8 eval gate.**

### Simulator validation (optional)

HA's history database has recent actual battery dispatch (SOC, charge/discharge power).
Cross-check simulator output against 1–2 weeks of real HA history to validate battery
model fidelity before relying on holistic eval numbers.
