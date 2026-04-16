#!/usr/bin/env python3
"""
Phase 3 offline LP dispatch simulator.

Evaluates forecast quality via financial regret on the stratified eval set.
Compares three dispatch strategies at every run in the 1,600-sample eval set:

  oracle   — perfect foresight (actual prices used as forecast)
  p5min    — raw AEMO P5MIN forecast (X[:, 0:12], already in the dataset)
  lgbm_q50 — Tier 1 LightGBM q50 predictions

Battery model: rolling MPC — at each 5-min step, solve LP over the remaining
horizon, execute the first action, advance SoC, repeat.

Financial regret = oracle_revenue – strategy_revenue (lower is better).

Outputs:
  eval/results/dispatch_sim_run001.json — per-stratum and aggregate results
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
MODEL_DIR   = ROOT / "models" / "lgbm_tactical"
RESULTS_DIR = ROOT / "eval" / "results"

# ── Battery parameters ───────────────────────────────────────────────────────
# Sigenergy home battery, conservative round-trip efficiency.
CAPACITY_KWH   = 40.0    # usable capacity
MAX_POWER_KW   = 10.0    # max charge or discharge rate
EFF_C          = 0.95    # charge efficiency (fraction of drawn energy stored)
EFF_D          = 0.95    # discharge efficiency (fraction of stored energy exported)
DEG_PER_KWH    = 0.05    # cycle degradation cost $/kWh throughput
SOC_INIT_KWH   = 20.0    # fixed starting SoC for each run (50% capacity)
INTERVAL_H     = 5 / 60  # 5 minutes in hours

# ── Spike/low thresholds (mirrors build_stratified_eval_tactical.py) ─────────
SPIKE_THRESH   = 300.0   # $/MWh
LOW_THRESH     = 0.0     # $/MWh


# ── LP dispatch ──────────────────────────────────────────────────────────────

def lp_dispatch(prices_mwh: np.ndarray, soc_init: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimal battery dispatch over a price horizon via LP.

    Variables: x = [c_0..c_{n-1}, d_0..d_{n-1}]
      c_h — grid import rate (kW) at step h
      d_h — battery export rate (kW) at step h

    SoC model:
      SoC[h+1] = soc_init + Σ_{k≤h} (c_k × EFF_C − d_k) × INTERVAL_H

    Objective (minimise −revenue):
      revenue = Σ_h [(d_h × EFF_D − c_h) × price_kwh_h
                     − DEG × (c_h × EFF_C + d_h)] × INTERVAL_H

    Returns (c_kw, d_kw) arrays of length n.
    Infeasible solves return zero arrays (do nothing).
    """
    n = len(prices_mwh)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    p = prices_mwh / 1000.0  # $/kWh

    # Objective: minimise (charge_cost − discharge_revenue + degradation)
    c_obj = np.concatenate([
        (p + DEG_PER_KWH * EFF_C) * INTERVAL_H,   # charge: pay price + degrade stored
        (-p * EFF_D + DEG_PER_KWH) * INTERVAL_H,  # discharge: get revenue − degrade drawn
    ])

    # SoC constraints via lower-triangular cumsum matrix
    L = np.tril(np.ones((n, n)))

    # SoC[h+1] >= 0:  -EFF_C × Σ_{k≤h} c_k  +  Σ_{k≤h} d_k  ≤  soc_init / INTERVAL_H
    # (multiplied through by INTERVAL_H already baked in)
    A_lower = np.hstack([-EFF_C * INTERVAL_H * L,  INTERVAL_H * L])
    b_lower = np.full(n, soc_init)

    # SoC[h+1] ≤ CAPACITY:  EFF_C × Σ c_k − Σ d_k ≤ (CAPACITY − soc_init) / INTERVAL_H
    A_upper = np.hstack([ EFF_C * INTERVAL_H * L, -INTERVAL_H * L])
    b_upper = np.full(n, CAPACITY_KWH - soc_init)

    A_ub = np.vstack([A_lower, A_upper])
    b_ub  = np.concatenate([b_lower, b_upper])
    bounds = [(0.0, MAX_POWER_KW)] * (2 * n)

    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs",
                     options={"disp": False})

    if result.status != 0:
        return np.zeros(n), np.zeros(n)

    return result.x[:n], result.x[n:]


# ── MPC simulation ───────────────────────────────────────────────────────────

def simulate_mpc(forecast_prices: np.ndarray, actual_prices: np.ndarray,
                 soc_init: float = SOC_INIT_KWH) -> dict:
    """
    Rolling-horizon MPC dispatch simulation.

    At each 5-min step h:
      1. Solve LP with forecast_prices[h:] as the forward price curve.
      2. Execute only the first action (c_0, d_0).
      3. Book revenue against actual_prices[h].
      4. Advance SoC.

    Returns dict with total revenue ($), final SoC, and per-step actions.
    """
    n = len(actual_prices)
    soc = float(soc_init)
    total_revenue = 0.0
    c_actions = np.zeros(n)
    d_actions = np.zeros(n)

    for h in range(n):
        c_plan, d_plan = lp_dispatch(forecast_prices[h:], soc)
        c0 = float(c_plan[0])
        d0 = float(d_plan[0])

        # Revenue at actual (not forecast) price
        p_kwh = actual_prices[h] / 1000.0
        revenue = (d0 * EFF_D * p_kwh - c0 * p_kwh
                   - DEG_PER_KWH * (c0 * EFF_C + d0)) * INTERVAL_H
        total_revenue += revenue

        soc = float(np.clip(soc + (c0 * EFF_C - d0) * INTERVAL_H,
                            0.0, CAPACITY_KWH))
        c_actions[h] = c0
        d_actions[h] = d0

    return {
        "revenue": total_revenue,
        "soc_final": soc,
        "c_kw": c_actions,
        "d_kw": d_actions,
    }


# ── LightGBM prediction ──────────────────────────────────────────────────────

def load_models() -> dict:
    """Load the three quantile models from disk."""
    models = {}
    for tag, q in [("q05", 0.05), ("q50", 0.50), ("q95", 0.95)]:
        path = MODEL_DIR / f"lgbm_{tag}.pkl"
        with open(path, "rb") as f:
            models[tag] = pickle.load(f)
        print(f"  Loaded {path.name}")
    return models


def predict_q50(models: dict, X_wide: np.ndarray, feat_names: list[str]) -> np.ndarray:
    """
    Generate q50 predictions for all runs in X_wide.

    Builds long-format (all 12 horizons per run), predicts, reshapes to [n, 12].
    Returns float32 array [n_runs, 12].
    """
    n_runs = X_wide.shape[0]
    n_h = 12

    # Repeat each row for each horizon, append horizon index
    X_rep = np.repeat(X_wide, n_h, axis=0).astype(np.float32)
    horizon_col = np.tile(np.arange(n_h, dtype=np.float32), n_runs).reshape(-1, 1)
    X_long = np.hstack([X_rep, horizon_col])

    feat_names_long = list(feat_names) + ["horizon"]
    X_df = pd.DataFrame(X_long, columns=feat_names_long)

    preds = models["q50"].predict(X_df).astype(np.float32)
    return preds.reshape(n_runs, n_h)


# ── Stratum labelling ────────────────────────────────────────────────────────

def label_strata(y: np.ndarray, y_mask: np.ndarray) -> np.ndarray:
    """
    Assign stratum label to each run.
    Returns string array: 'spike' | 'low' | 'normal'
    """
    y_masked = np.where(y_mask, y, np.nan)
    max_rrp  = np.nanmax(y_masked, axis=1)
    min_rrp  = np.nanmin(y_masked, axis=1)
    labels   = np.where(max_rrp >= SPIKE_THRESH, "spike",
               np.where(min_rrp <  LOW_THRESH,   "low", "normal"))
    return labels


# ── Main evaluation ──────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("Phase 3 — Dispatch Simulator (Run 001)")
    print("=" * 60)

    # ── Load arrays ──────────────────────────────────────────────────────────
    print("\nLoading data...")
    X         = np.load(PARQUET_DIR / "X_tactical.npy")
    y         = np.load(PARQUET_DIR / "y_tactical.npy")
    y_mask    = np.load(PARQUET_DIR / "y_tactical_mask.npy")
    run_times = np.load(PARQUET_DIR / "run_times_tactical.npy")
    strat_rt  = np.load(PARQUET_DIR / "stratified_eval_run_times_tactical.npy")

    with open(PARQUET_DIR / "tactical_meta.json") as f:
        meta = json.load(f)
    feat_names = meta["feature_names"]

    # Map stratified eval run_times → indices in full array
    rt_int64   = run_times.view(np.int64)
    strat_set  = set(strat_rt.view(np.int64).tolist())
    eval_idx   = np.array([i for i, rt in enumerate(rt_int64)
                           if int(rt) in strat_set], dtype=np.int64)
    print(f"  Stratified eval runs: {len(eval_idx):,}")

    # ── Load models & generate q50 predictions ────────────────────────────────
    print("\nLoading LightGBM models...")
    models = load_models()

    print("\nGenerating q50 predictions for eval set...")
    X_eval = X[eval_idx]
    lgbm_q50 = predict_q50(models, X_eval, feat_names)   # [n_eval, 12]

    # P5MIN forecast: columns 0..11 of X (p5min_rrp_h{0..11})
    p5min_fcst = X_eval[:, :12].astype(np.float64)       # [n_eval, 12]

    # Oracle: actual prices
    y_eval      = y[eval_idx]                             # [n_eval, 12]
    y_mask_eval = y_mask[eval_idx]                        # [n_eval, 12] bool
    oracle_fcst  = np.where(y_mask_eval, y_eval, p5min_fcst)  # fallback P5MIN for missing actuals

    # ── Stratum labels ────────────────────────────────────────────────────────
    strata = label_strata(y_eval, y_mask_eval)
    stratum_names = ["spike", "low", "normal", "all"]
    stratum_masks = {
        "spike":  strata == "spike",
        "low":    strata == "low",
        "normal": strata == "normal",
        "all":    np.ones(len(eval_idx), dtype=bool),
    }
    strata_counts = {k: int(v.sum()) for k, v in stratum_masks.items()}
    print(f"\n  Strata: spike={strata_counts['spike']}, "
          f"low={strata_counts['low']}, normal={strata_counts['normal']}")

    # ── Run MPC simulation ────────────────────────────────────────────────────
    n_eval = len(eval_idx)
    strategies = {
        "oracle":    oracle_fcst,
        "p5min":     p5min_fcst,
        "lgbm_q50":  lgbm_q50.astype(np.float64),
    }

    revenues = {s: np.zeros(n_eval) for s in strategies}

    print(f"\nRunning MPC simulation ({n_eval} runs × 3 strategies)...")
    t0 = time.time()
    for i in range(n_eval):
        actual = y_eval[i].astype(np.float64)
        # For steps where actual is masked, fall back to P5MIN
        actual_for_revenue = np.where(y_mask_eval[i], actual, p5min_fcst[i])

        for name, fcast_2d in strategies.items():
            result = simulate_mpc(fcast_2d[i], actual_for_revenue)
            revenues[name][i] = result["revenue"]

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_eval} runs  ({elapsed:.1f}s elapsed)")

    elapsed_total = time.time() - t0
    print(f"  Done in {elapsed_total:.1f}s")

    # ── Compute regret and summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    results = {}
    for sname in stratum_names:
        mask = stratum_masks[sname]
        n = int(mask.sum())
        if n == 0:
            continue

        oracle_rev  = revenues["oracle"][mask]
        p5min_rev   = revenues["p5min"][mask]
        lgbm_rev    = revenues["lgbm_q50"][mask]

        regret_p5min  = oracle_rev - p5min_rev
        regret_lgbm   = oracle_rev - lgbm_rev
        improvement   = lgbm_rev   - p5min_rev   # LightGBM vs P5MIN revenue gain

        def stats(arr):
            return {
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p25":    float(np.percentile(arr, 25)),
                "p75":    float(np.percentile(arr, 75)),
                "total":  float(np.sum(arr)),
            }

        results[sname] = {
            "n": n,
            "revenue": {
                "oracle":    stats(oracle_rev),
                "p5min":     stats(p5min_rev),
                "lgbm_q50":  stats(lgbm_rev),
            },
            "regret_vs_oracle": {
                "p5min":     stats(regret_p5min),
                "lgbm_q50":  stats(regret_lgbm),
            },
            "lgbm_improvement_vs_p5min": stats(improvement),
            "lgbm_regret_reduction_pct": float(
                (np.mean(regret_p5min) - np.mean(regret_lgbm))
                / max(abs(np.mean(regret_p5min)), 1e-9) * 100
            ),
        }

        # Print table row
        print(f"\n  [{sname.upper():7s}]  n={n}")
        print(f"    {'Strategy':<12}  {'Mean Rev ($)':<14}  {'Mean Regret ($)':<16}  {'Regret %'}")
        print(f"    {'-'*58}")
        for strat, rev_arr, reg_arr in [
            ("oracle",   oracle_rev, np.zeros(n)),
            ("p5min",    p5min_rev,  regret_p5min),
            ("lgbm_q50", lgbm_rev,   regret_lgbm),
        ]:
            mean_rev = np.mean(rev_arr)
            mean_reg = np.mean(reg_arr)
            oracle_mean = np.mean(oracle_rev)
            reg_pct = mean_reg / max(abs(oracle_mean), 1e-9) * 100
            print(f"    {strat:<12}  {mean_rev:>12.4f}    {mean_reg:>14.4f}    {reg_pct:>6.1f}%")

        lgbm_impr_pct = results[sname]["lgbm_regret_reduction_pct"]
        print(f"    → LightGBM regret reduction vs P5MIN: {lgbm_impr_pct:+.1f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "dispatch_sim_run001.json"
    full_results = {
        "run_id": "dispatch_sim_run001",
        "n_eval": n_eval,
        "battery": {
            "capacity_kwh":    CAPACITY_KWH,
            "max_power_kw":    MAX_POWER_KW,
            "eff_charge":      EFF_C,
            "eff_discharge":   EFF_D,
            "deg_per_kwh":     DEG_PER_KWH,
            "soc_init_kwh":    SOC_INIT_KWH,
            "interval_h":      INTERVAL_H,
        },
        "strata_counts":    strata_counts,
        "elapsed_seconds":  elapsed_total,
        "results_by_stratum": results,
    }
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")

    return full_results


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    run_evaluation()
