#!/usr/bin/env python3
"""
Train Tier 1 tactical LightGBM models (0–60 min, 5-min resolution).

Architecture: 3 LightGBM quantile regressors (q5, q50, q95), each trained
on all 12 horizons simultaneously via a "long format" + horizon feature.

Long format construction:
  Base features X [N, 32] are repeated 12 times.
  A horizon feature (0..11) is appended → X_long [N×12, 33].
  Targets are flattened from y [N, 12] → y_long [N×12].
  Runs where y_mask is False are dropped entirely from training.

This allows the model to learn:
  - horizon-specific AEMO bias patterns (e.g. h0 is systematic, h11 less so)
  - which p5min_rrp_h{k} is most relevant at each horizon via the horizon feature
  - non-uniform correction at different forecast horizons

Quantiles trained: q5, q50, q95 (one model per quantile).
  q5/q95 → tail risk spread for HA automation
  q50    → central forecast passed to EMHASS

Early stopping: monitored on time-ordered val set (last 60 days).
Final evaluation: reported on stratified eval set (spike/low/normal strata).

Outputs (models/lgbm_tactical/):
  lgbm_q{5,50,95}.pkl     — trained LightGBM models
  training_meta.json       — hyperparams, split sizes, calibration summary
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
DEFAULT_MODEL_DIR = ROOT / "models" / "lgbm_tactical"
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-energy-forecast-slop-mplconfig")

OUTPUT_STEPS = 12
QUANTILES    = [0.05, 0.50, 0.95]

# ── Hyperparameters ──────────────────────────────────────────────────────────
# Conservative defaults: SA1 has ~2.2M valid training steps in long format.
# num_leaves=63 is moderately complex; LightGBM handles large datasets well.
LGBM_BASE_PARAMS = {
    "objective":        "quantile",
    "n_estimators":     2000,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples": 50,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}
EARLY_STOPPING_ROUNDS = 50


def load_arrays():
    """Load tactical dataset arrays and split indices."""
    print("Loading arrays...")
    X         = np.load(PARQUET_DIR / "X_tactical.npy")          # [N, 32]
    y         = np.load(PARQUET_DIR / "y_tactical.npy")          # [N, 12]
    y_mask    = np.load(PARQUET_DIR / "y_tactical_mask.npy")     # [N, 12] bool
    run_times = np.load(PARQUET_DIR / "run_times_tactical.npy")  # [N]
    splits    = np.load(PARQUET_DIR / "split_indices_tactical.npz")
    train_idx = splits["train"]
    val_idx   = splits["val"]
    test_idx  = splits.get("test", np.array([], dtype=np.int64))

    print(f"  Total samples:     {len(X):,}")
    print(f"  Train:             {len(train_idx):,}")
    print(f"  Val:               {len(val_idx):,}")
    print(f"  Stratified eval:   {len(test_idx):,}")
    return X, y, y_mask, run_times, train_idx, val_idx, test_idx


def build_long_format(X: np.ndarray,
                      y: np.ndarray,
                      y_mask: np.ndarray,
                      idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert [N, 32] wide format → long format [M, 33] x [M] y.

    For each of the 12 horizons h=0..11:
      - Repeat X rows, append horizon as scalar feature
      - Stack matching y column
    Drop rows where y_mask is False (no actual available).

    Returns (X_long, y_long) with only valid rows.
    """
    n = len(idx)
    X_sub  = X[idx]       # [n, 24]
    y_sub  = y[idx]       # [n, 12]
    m_sub  = y_mask[idx]  # [n, 12] bool

    # Pre-allocate long arrays (upper bound = n × 12)
    X_parts = []
    y_parts = []

    for h in range(OUTPUT_STEPS):
        valid = m_sub[:, h]  # [n] bool
        if valid.sum() == 0:
            continue
        horizon_col = np.full((valid.sum(), 1), h, dtype=np.float32)
        X_parts.append(np.hstack([X_sub[valid], horizon_col]))   # [m, 33]
        y_parts.append(y_sub[valid, h])                           # [m]

    X_long = np.vstack(X_parts)
    y_long = np.concatenate(y_parts)
    return X_long, y_long


# ── Feature names including horizon ─────────────────────────────────────────
def feature_names_long() -> list[str]:
    from data.build_tactical_dataset import FEATURE_NAMES
    return list(FEATURE_NAMES) + ["horizon"]


def train_quantile(q: float,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray,   y_val: np.ndarray,
                   feat_names: list[str]) -> lgb.LGBMRegressor:
    """Train one LightGBM quantile regressor with early stopping."""
    params = {**LGBM_BASE_PARAMS, "alpha": q}
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    best_iter = model.best_iteration_
    val_loss  = model.best_score_["valid_0"]["quantile"]
    print(f"  q{int(q*100):02d}: best_iter={best_iter}, val_pinball={val_loss:.4f}")
    return model


def evaluate(models: dict[float, lgb.LGBMRegressor],
             X: np.ndarray, y: np.ndarray, y_mask: np.ndarray,
             idx: np.ndarray, label: str) -> dict:
    """
    Evaluate all quantile models on a given split.

    Returns dict with calibration bias per quantile and MAE for q50.
    """
    print(f"\n{'─'*50}")
    print(f"Evaluation: {label}  ({len(idx):,} runs, "
          f"{y_mask[idx].sum():,} valid steps)")

    X_sub  = X[idx]
    y_sub  = y[idx]
    m_sub  = y_mask[idx]

    results = {}

    for q, model in sorted(models.items()):
        preds = np.zeros_like(y_sub)  # [n, 12]
        for h in range(OUTPUT_STEPS):
            horizon_col = np.full((len(idx), 1), h, dtype=np.float32)
            X_h = np.hstack([X_sub, horizon_col])
            preds[:, h] = model.predict(X_h)

        # Quantile calibration: fraction of actuals below predicted quantile
        valid = m_sub  # [n, 12]
        actual_below = (y_sub[valid] < preds[valid]).mean()
        bias = actual_below - q

        # Pinball loss
        resid      = y_sub[valid] - preds[valid]
        pinball    = np.where(resid >= 0, q * resid, (q - 1) * resid).mean()

        results[q] = {"coverage": float(actual_below), "bias": float(bias),
                      "pinball": float(pinball)}

        bias_icon = "✓" if abs(bias) < 0.03 else "⚠" if abs(bias) < 0.06 else "❌"
        print(f"  q{int(q*100):02d}: coverage={actual_below:.3f}  "
              f"bias={bias:+.3f} {bias_icon}  pinball={pinball:.2f}")

    # MAE and nMAPE for q50
    q50_preds = np.zeros_like(y_sub)
    model_q50 = models[0.50]
    for h in range(OUTPUT_STEPS):
        horizon_col = np.full((len(idx), 1), h, dtype=np.float32)
        q50_preds[:, h] = model_q50.predict(np.hstack([X_sub, horizon_col]))

    valid_act  = y_sub[m_sub]
    valid_pred = q50_preds[m_sub]
    mae  = float(np.abs(valid_act - valid_pred).mean())
    # P5MIN baseline MAE (using h-matched p5min forecasts as naive baseline)
    # X columns 0..11 are p5min_rrp_h0..h11
    p5min_baseline = np.zeros_like(y_sub)
    for h in range(OUTPUT_STEPS):
        p5min_baseline[:, h] = X_sub[:, h]   # p5min_rrp_h{h}
    baseline_mae = float(np.abs(y_sub[m_sub] - p5min_baseline[m_sub]).mean())
    nmape_q50 = mae / (np.abs(valid_act).mean() + 1e-6)

    print(f"\n  q50 MAE:       {mae:.2f} $/MWh  (P5MIN baseline: {baseline_mae:.2f})")
    print(f"  q50 nMAPE:     {nmape_q50:.4f}")
    print(f"  MAE reduction: {(1 - mae / baseline_mae) * 100:.1f}% vs raw P5MIN")

    results["mae_q50"]       = mae
    results["baseline_mae"]  = baseline_mae
    results["nmape_q50"]     = nmape_q50
    return results


def evaluate_by_stratum(models: dict[float, lgb.LGBMRegressor],
                        X: np.ndarray, y: np.ndarray, y_mask: np.ndarray,
                        idx: np.ndarray):
    """Per-stratum evaluation on stratified eval set (spike/low/normal)."""
    # Score each run by max and min actual RRP across its 12-step window
    y_sub   = y[idx]
    m_sub   = y_mask[idx]
    y_msk   = np.where(m_sub, y_sub, np.nan)
    max_rrp = np.nanmax(y_msk, axis=1)   # [n] — NaN where no valid steps
    min_rrp = np.nanmin(y_msk, axis=1)

    # Build stratum membership using local (subset) indices, then map back
    valid_local = ~np.isnan(max_rrp)
    spike_local  = valid_local & (max_rrp >= 300)
    low_local    = valid_local & (min_rrp < 0) & (max_rrp < 300)
    normal_local = valid_local & (min_rrp >= 0) & (max_rrp < 300)

    strata = {
        "spike":  (idx[spike_local],  max_rrp[spike_local]),
        "low":    (idx[low_local],    max_rrp[low_local]),
        "normal": (idx[normal_local], max_rrp[normal_local]),
    }

    print(f"\n{'─'*50}")
    print("Stratified eval (spike / low / normal):")
    for name, (stratum_idx, s_max_rrp) in strata.items():
        if len(stratum_idx) == 0:
            continue
        X_s = X[stratum_idx]
        y_s = y[stratum_idx]
        m_s = y_mask[stratum_idx]
        if m_s.sum() == 0:
            continue

        q50_model      = models[0.50]
        q50_preds      = np.zeros_like(y_s)
        p5min_baseline = np.zeros_like(y_s)
        for h in range(OUTPUT_STEPS):
            horizon_col = np.full((len(stratum_idx), 1), h, dtype=np.float32)
            q50_preds[:, h]      = q50_model.predict(np.hstack([X_s, horizon_col]))
            p5min_baseline[:, h] = X_s[:, h]   # p5min_rrp_h{h}

        valid_act  = y_s[m_s]
        valid_pred = q50_preds[m_s]
        valid_base = p5min_baseline[m_s]
        mae_model  = float(np.abs(valid_act - valid_pred).mean())
        mae_base   = float(np.abs(valid_act - valid_base).mean())
        improvement = (1 - mae_model / mae_base) * 100

        # Tail calibration (q5/q95)
        cal_lines = []
        for q, model in sorted(models.items()):
            if q == 0.50:
                continue
            q_preds = np.zeros_like(y_s)
            for h in range(OUTPUT_STEPS):
                horizon_col = np.full((len(stratum_idx), 1), h, dtype=np.float32)
                q_preds[:, h] = model.predict(np.hstack([X_s, horizon_col]))
            cov  = float((y_s[m_s] < q_preds[m_s]).mean())
            bias = cov - q
            icon = "✓" if abs(bias) < 0.03 else "⚠" if abs(bias) < 0.06 else "❌"
            cal_lines.append(f"q{int(q*100):02d}: {cov:.3f} ({bias:+.3f}) {icon}")

        mean_max = float(np.mean(s_max_rrp))
        print(f"\n  {name.upper():6s} ({len(stratum_idx):,} runs, "
              f"mean max RRP {mean_max:.0f} $/MWh):")
        print(f"    q50 MAE: {mae_model:.1f}  baseline: {mae_base:.1f}  "
              f"improvement: {improvement:.1f}%")
        print(f"    " + "  ".join(cal_lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to write trained tactical models into.",
    )
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    print("=== Tier 1 Tactical LightGBM Training ===\n")
    X, y, y_mask, run_times, train_idx, val_idx, test_idx = load_arrays()

    # ── Build long-format arrays ──────────────────────────────────────────────
    print("\nBuilding long-format training arrays (N runs × 12 horizons)...")
    X_train, y_train = build_long_format(X, y, y_mask, train_idx)
    X_val,   y_val   = build_long_format(X, y, y_mask, val_idx)
    print(f"  Train: {len(X_train):,} rows  Val: {len(X_val):,} rows")

    feat_names = feature_names_long()
    print(f"  Features ({len(feat_names)}): {feat_names}")

    # ── Train quantile models ─────────────────────────────────────────────────
    print(f"\nTraining {len(QUANTILES)} quantile models "
          f"(early stopping after {EARLY_STOPPING_ROUNDS} rounds)...")
    models = {}
    for q in QUANTILES:
        print(f"\n  Fitting q{int(q*100):02d}...")
        sys.stdout.flush()
        models[q] = train_quantile(q, X_train, y_train, X_val, y_val, feat_names)

    # ── Evaluation on val set ─────────────────────────────────────────────────
    val_results = evaluate(models, X, y, y_mask, val_idx, "Val (last 60 days)")

    # ── Evaluation on stratified set ──────────────────────────────────────────
    strat_results = {}
    if len(test_idx) > 0:
        strat_results = evaluate(models, X, y, y_mask, test_idx, "Stratified eval")
        evaluate_by_stratum(models, X, y, y_mask, test_idx)
    else:
        print("\nNo stratified eval set found — run build_stratified_eval_tactical.py first.")

    # ── Save models ───────────────────────────────────────────────────────────
    print("\nSaving models...")
    for q, model in models.items():
        path = model_dir / f"lgbm_q{int(q*100):02d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved: {path}")

    # ── Save training metadata ────────────────────────────────────────────────
    meta = {
        "quantiles":           QUANTILES,
        "feature_names":       feat_names,
        "n_features":          len(feat_names),
        "output_steps":        OUTPUT_STEPS,
        "n_train_runs":        int(len(train_idx)),
        "n_val_runs":          int(len(val_idx)),
        "n_strat_eval_runs":   int(len(test_idx)),
        "n_train_rows":        int(len(X_train)),
        "n_val_rows":          int(len(X_val)),
        "lgbm_params":         LGBM_BASE_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iterations":     {f"q{int(q*100):02d}": int(m.best_iteration_)
                                for q, m in models.items()},
        "val_results":         val_results,
        "strat_results":       strat_results,
    }
    meta_path = model_dir / "training_meta.json"

    def _json_default(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    print(f"  Metadata: {meta_path}")

    print("\n=== Training complete ===")
    print(f"  Models saved to: {model_dir}/")
    for q, model in models.items():
        print(f"  q{int(q*100):02d}: {model.best_iteration_} trees, "
              f"val pinball={model.best_score_['valid_0']['quantile']:.4f}")


if __name__ == "__main__":
    main()
