#!/usr/bin/env python3
"""
Phase 4 — Conditional conformal calibration for Tier 1 LightGBM.

Computes per-regime additive corrections (δ) for q05 and q95 using split
conformal prediction on the val set, then validates coverage on the
stratified eval set.

Background
----------
The trained q95 model under-covers spike intervals: 0.857 vs target 0.950.
This is structural — quantile regression on a dataset that is ~5% spikes
will minimise total pinball loss by trading off spike coverage for baseload.

Fix: conditional conformal prediction. Split the calibration data by price
regime and fit a separate additive correction δ within each regime:

  adjusted_q = raw_q + δ_regime

This guarantees (approximately) that coverage within each regime matches
the target level.

Regime labelling
----------------
At calibration time:   outcome-based (actual price at each step)
At inference time:     feature-based (P5MIN forecast for that horizon)

  spike  — actual ≥ $300/MWh (cal) | p5min_rrp_h{h} ≥ $300 (inference)
  low    — actual < $0/MWh    (cal) | p5min_rrp_h{h} < $0 or residual_demand_t1 < $0
  normal — everything else

Non-conformity scores (per step in long format)
-----------------------------------------------
  q95:  s = actual − pred_q95   → δ_q95[regime] = np.percentile(s, 95)
  q05:  s = pred_q05 − actual   → δ_q05[regime] = np.percentile(s, 95)

  At inference:
    adjusted_q95[h] = raw_q95[h] + δ_q95[inferred_regime[h]]
    adjusted_q05[h] = raw_q05[h] − δ_q05[inferred_regime[h]]

Outputs
-------
  models/lgbm_tactical/conformal_deltas.json   — δ table + coverage report
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
MODEL_DIR   = ROOT / "models" / "lgbm_tactical"

SPIKE_THRESH = 300.0   # $/MWh
LOW_THRESH   =   0.0   # $/MWh


# ── Prediction helpers ───────────────────────────────────────────────────────

def predict_quantile(model, X_wide: np.ndarray, feat_names: list[str],
                     horizon: int) -> np.ndarray:
    """
    Predict a single quantile for a single horizon across all runs.

    Builds a [N, 25] DataFrame for the given horizon only, returns [N] predictions.
    """
    n = X_wide.shape[0]
    horizon_col = np.full((n, 1), horizon, dtype=np.float32)
    X_long = np.hstack([X_wide.astype(np.float32), horizon_col])
    X_df = pd.DataFrame(X_long, columns=feat_names + ["horizon"])
    return model.predict(X_df)


def predict_all_horizons(model, X_wide: np.ndarray, feat_names: list[str],
                         n_horizons: int = 12) -> np.ndarray:
    """
    Predict a quantile for all horizons.

    Returns [N, n_horizons] float64 array.
    """
    n = X_wide.shape[0]
    X_rep = np.repeat(X_wide, n_horizons, axis=0).astype(np.float32)
    hor_col = np.tile(np.arange(n_horizons, dtype=np.float32), n).reshape(-1, 1)
    X_long = np.hstack([X_rep, hor_col])
    X_df = pd.DataFrame(X_long, columns=feat_names + ["horizon"])
    preds = model.predict(X_df)
    return preds.reshape(n, n_horizons)


# ── Regime labelling ─────────────────────────────────────────────────────────

def outcome_regime(actual: np.ndarray) -> np.ndarray:
    """
    Label each (run, horizon) step by the actual price outcome.

    actual: [N, H] float
    Returns [N, H] str array: 'spike' | 'low' | 'normal'
    """
    return np.where(actual >= SPIKE_THRESH, "spike",
           np.where(actual <  LOW_THRESH,   "low",   "normal"))


def inference_regime(X_wide: np.ndarray, feat_names: list[str],
                     n_horizons: int = 12) -> np.ndarray:
    """
    Label each (run, horizon) step from features (used at inference time).

    Spike: p5min_rrp_h{h} ≥ $300  (AEMO forecasts spike at this horizon)
           OR actual_rrp_t1 ≥ $300  (most recent actual was already a spike — likely ongoing)
    Low:   p5min_rrp_h{h} < $0  OR  residual_demand_t1 < 0  (oversupply signals)
    Normal: everything else

    Spike detects ~76% of actual spike steps at ~64% precision (F1=0.694).
    Remaining ~24% of undetectable spikes (genuine sudden events) receive
    normal correction — a fundamental limit, not a model quality issue.

    Returns [N, H] str array.
    """
    p5min_matrix  = X_wide[:, :n_horizons]                              # [N, H]
    rrp_t1        = X_wide[:, feat_names.index("actual_rrp_t1")]        # [N]
    res_demand    = X_wide[:, feat_names.index("residual_demand_t1")]   # [N]
    rrp_t1_2d     = rrp_t1[:, np.newaxis]                               # [N, 1]
    res_demand_2d = res_demand[:, np.newaxis]                            # [N, 1]

    spike = (p5min_matrix >= SPIKE_THRESH) | (rrp_t1_2d >= SPIKE_THRESH)
    low   = ((p5min_matrix < LOW_THRESH) | (res_demand_2d < 0)) & ~spike
    return np.where(spike, "spike", np.where(low, "low", "normal"))


# ── Coverage reporting ────────────────────────────────────────────────────────

def coverage_report(preds_q05: np.ndarray, preds_q95: np.ndarray,
                    actual: np.ndarray, mask: np.ndarray,
                    label: str = "") -> dict:
    """
    Report per-regime calibration coverage.

    preds_q05/q95: [N, H] float
    actual:        [N, H] float (raw prices)
    mask:          [N, H] bool
    Returns dict: {regime: {q05_coverage, q95_coverage, n}}
    """
    regimes = outcome_regime(actual)
    results = {}
    for r in ["spike", "low", "normal", "all"]:
        if r == "all":
            rm = mask
        else:
            rm = mask & (regimes == r)
        n = int(rm.sum())
        if n == 0:
            continue
        cov_q05 = float((actual[rm] >= preds_q05[rm]).mean())
        cov_q95 = float((actual[rm] <= preds_q95[rm]).mean())
        results[r] = {"n": n, "q05_coverage": cov_q05, "q95_coverage": cov_q95}

    if label:
        print(f"\n  [{label}]")
        print(f"    {'Regime':<8}  {'n':>7}  {'q05 cov':>9}  {'q95 cov':>9}  {'q05 status':>12}  {'q95 status'}")
        print(f"    {'-'*70}")
        for r, v in results.items():
            s05 = "✓" if abs(v["q05_coverage"] - 0.95) < 0.02 else "❌"
            s95 = "✓" if abs(v["q95_coverage"] - 0.95) < 0.02 else "❌"
            print(f"    {r:<8}  {v['n']:>7,}  {v['q05_coverage']:>9.3f}  "
                  f"{v['q95_coverage']:>9.3f}  {s05:>12}  {s95}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 4 — Conditional Conformal Calibration")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    X         = np.load(PARQUET_DIR / "X_tactical.npy")
    y         = np.load(PARQUET_DIR / "y_tactical.npy")
    y_mask    = np.load(PARQUET_DIR / "y_tactical_mask.npy")
    splits    = np.load(PARQUET_DIR / "split_indices_tactical.npz")
    strat_rt  = np.load(PARQUET_DIR / "stratified_eval_run_times_tactical.npy")
    run_times = np.load(PARQUET_DIR / "run_times_tactical.npy")

    with open(PARQUET_DIR / "tactical_meta.json") as f:
        meta = json.load(f)
    feat_names = meta["feature_names"]

    val_idx   = splits["val"]
    strat_set = set(strat_rt.view(np.int64).tolist())
    eval_idx  = np.array([i for i, rt in enumerate(run_times.view(np.int64))
                          if int(rt) in strat_set], dtype=np.int64)

    print(f"  Val:  {len(val_idx):,} runs   Stratified eval: {len(eval_idx):,} runs")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading q05 and q95 models...")
    with open(MODEL_DIR / "lgbm_q05.pkl", "rb") as f:
        m_q05 = pickle.load(f)
    with open(MODEL_DIR / "lgbm_q95.pkl", "rb") as f:
        m_q95 = pickle.load(f)

    # ── Generate predictions for val set ─────────────────────────────────────
    print("\nGenerating val set predictions (q05 + q95)...")
    X_val   = X[val_idx]
    y_val   = y[val_idx]
    ym_val  = y_mask[val_idx]

    pred_q05_val = predict_all_horizons(m_q05, X_val, feat_names)  # [N_val, 12]
    pred_q95_val = predict_all_horizons(m_q95, X_val, feat_names)  # [N_val, 12]

    # ── Baseline coverage (before calibration) ────────────────────────────────
    print("\n--- Baseline coverage (val set, before calibration) ---")
    before_val = coverage_report(pred_q05_val, pred_q95_val, y_val, ym_val, "val set")

    # ── Compute non-conformity scores ─────────────────────────────────────────
    # For each valid (run, horizon) step, compute:
    #   s_q95 = actual - pred_q95   (positive → actual exceeded prediction)
    #   s_q05 = pred_q05 - actual   (positive → prediction exceeded actual)
    # Then stratify by outcome regime (actual price at that step).

    actual_val  = np.where(ym_val, y_val, np.nan)
    regimes_val = outcome_regime(actual_val)   # [N_val, 12]

    s_q95 = actual_val - pred_q95_val          # [N_val, 12]
    s_q05 = pred_q05_val - actual_val          # [N_val, 12]

    # ── Compute δ per regime ──────────────────────────────────────────────────
    print("\n--- Non-conformity scores and δ values ---")
    deltas = {}
    TARGET_ALPHA = 0.95

    for r in ["spike", "low", "normal"]:
        r_mask = ym_val & (regimes_val == r)   # valid AND in this regime
        n_r    = int(r_mask.sum())

        if n_r < 30:
            print(f"  {r}: only {n_r} steps — skipping (too few samples)")
            deltas[r] = {"delta_q95": 0.0, "delta_q05": 0.0, "n_cal": n_r}
            continue

        scores_q95 = s_q95[r_mask]
        scores_q05 = s_q05[r_mask]

        # Standard split conformal: use ⌈(n+1)(1-α)⌉/n quantile
        # For practical purposes, np.percentile with linear interpolation is fine.
        alpha_level = TARGET_ALPHA * 100  # for np.percentile
        d_q95 = float(np.percentile(scores_q95, alpha_level))
        d_q05 = float(np.percentile(scores_q05, alpha_level))

        deltas[r] = {"delta_q95": d_q95, "delta_q05": d_q05, "n_cal": n_r}

        # Current raw coverage in this regime
        raw_cov_q95 = float((actual_val[r_mask] <= pred_q95_val[r_mask]).mean())
        raw_cov_q05 = float((actual_val[r_mask] >= pred_q05_val[r_mask]).mean())

        print(f"  {r:<7}  n={n_r:>7,}  "
              f"raw_q95={raw_cov_q95:.3f}  raw_q05={raw_cov_q05:.3f}  "
              f"δ_q95={d_q95:+.2f}  δ_q05={d_q05:+.2f}")

    # ── Apply calibration to val set and re-check ─────────────────────────────
    print("\n--- Coverage after calibration (val set) ---")
    adj_q95_val = pred_q95_val.copy()
    adj_q05_val = pred_q05_val.copy()
    for r, d in deltas.items():
        r_mask = regimes_val == r   # [N, H] bool (applied to all horizons)
        adj_q95_val[r_mask] += d["delta_q95"]
        adj_q05_val[r_mask] -= d["delta_q05"]

    after_val = coverage_report(adj_q05_val, adj_q95_val, y_val, ym_val, "val set (calibrated)")

    # ── Validate on stratified eval ───────────────────────────────────────────
    print("\n--- Stratified eval coverage (before calibration) ---")
    X_eval  = X[eval_idx]
    y_eval  = y[eval_idx]
    ym_eval = y_mask[eval_idx]

    pred_q05_eval = predict_all_horizons(m_q05, X_eval, feat_names)
    pred_q95_eval = predict_all_horizons(m_q95, X_eval, feat_names)

    before_eval = coverage_report(pred_q05_eval, pred_q95_eval, y_eval, ym_eval,
                                  "stratified eval (before)")

    print("\n--- Stratified eval coverage (after calibration, outcome-based regime) ---")
    actual_eval   = np.where(ym_eval, y_eval, np.nan)
    regimes_eval  = outcome_regime(actual_eval)
    adj_q95_eval  = pred_q95_eval.copy()
    adj_q05_eval  = pred_q05_eval.copy()
    for r, d in deltas.items():
        r_mask = regimes_eval == r
        adj_q95_eval[r_mask] += d["delta_q95"]
        adj_q05_eval[r_mask] -= d["delta_q05"]

    after_eval_outcome = coverage_report(adj_q05_eval, adj_q95_eval, y_eval, ym_eval,
                                         "stratified eval (outcome regime)")

    print("\n--- Stratified eval coverage (after calibration, inference-based regime) ---")
    # Use feature-based regime detection (what happens at actual inference time)
    inf_regimes_eval = inference_regime(X_eval, feat_names)
    adj_q95_eval_inf = pred_q95_eval.copy()
    adj_q05_eval_inf = pred_q05_eval.copy()
    for r, d in deltas.items():
        r_mask = inf_regimes_eval == r
        adj_q95_eval_inf[r_mask] += d["delta_q95"]
        adj_q05_eval_inf[r_mask] -= d["delta_q05"]

    after_eval_inf = coverage_report(adj_q05_eval_inf, adj_q95_eval_inf, y_eval, ym_eval,
                                     "stratified eval (inference regime)")

    # ── Inference-time regime match rate ──────────────────────────────────────
    valid_mask = ym_eval
    outcome_flat = regimes_eval[valid_mask]
    inf_flat     = inf_regimes_eval[valid_mask]
    match_rate   = float((outcome_flat == inf_flat).mean())
    print(f"\n  Inference regime match rate vs outcome regime: {match_rate:.3f}")

    # ── Save deltas ───────────────────────────────────────────────────────────
    output = {
        "description": (
            "Additive conformal corrections for LightGBM q05/q95. "
            "At inference: adjusted_q95 = raw_q95 + delta_q95[regime]; "
            "adjusted_q05 = raw_q05 - delta_q05[regime]. "
            "Regime = spike if p5min_rrp_h{h} >= 300, "
            "low if p5min_rrp_h{h} < 0 or residual_demand_t1 < 0, else normal."
        ),
        "spike_threshold":  SPIKE_THRESH,
        "low_threshold":    LOW_THRESH,
        "target_alpha":     TARGET_ALPHA,
        "deltas":           deltas,
        "coverage_before":  {
            "val":            before_val,
            "stratified_eval": before_eval,
        },
        "coverage_after": {
            "val":                      after_val,
            "stratified_eval_outcome":  after_eval_outcome,
            "stratified_eval_inference": after_eval_inf,
        },
        "inference_regime_match_rate": match_rate,
    }

    out_path = MODEL_DIR / "conformal_deltas.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Saved → {out_path.relative_to(ROOT)}")
    print("\n  Done.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
