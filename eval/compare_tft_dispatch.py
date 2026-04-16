#!/usr/bin/env python3
"""
Phase 3 — TFT vs LightGBM dispatch comparison at 0–60 min.

Identifies tactical stratified eval runs that fall on 30-min PREDISPATCH
boundaries and exist in both the tactical eval set and the TFT training
data's run_times. Runs TFT inference on that subset, step-holds the
30-min predictions to 5-min resolution, and compares dispatch regret
against LightGBM q50 and P5MIN on those same runs.

Note: TFT operates at 30-min resolution; we use steps 0–1 (first 60 min)
held constant for intervals 0–5 and 6–11 respectively. This approximation
slightly disadvantages TFT (loses within-interval price variation) but is
the fairest comparison possible without a 5-min TFT model.

Comparison set: ~130 runs (intersection of tactial stratified eval + TFT
run_times, filtered to 30-min boundaries).

Outputs:
  eval/results/tft_dispatch_comparison.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT        = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
MODEL_DIR   = ROOT / "models" / "lgbm_tactical"
TFT_DIR     = ROOT / "models" / "tft_price"
RESULTS_DIR = ROOT / "eval" / "results"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

from eval.dispatch_simulator import simulate_mpc, SPIKE_THRESH, LOW_THRESH  # noqa: E402
from train.train_tft_price import TFTPriceModel, PREDISPATCHDataset         # noqa: E402


# ── TFT inference (reused from evaluate_tft.py) ──────────────────────────────

def run_tft_inference_subset(model, X_enc, X_dec, y_norm, y_raw, y_mask,
                              scalers, batch_size=256):
    """
    Run TFT inference on a pre-indexed subset of arrays.
    Returns (pred_p50 [N, 144], y_raw [N, 144], mask [N, 144]).
    """
    ckpt_path = TFT_DIR / "checkpoint_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    target_scaling   = ckpt.get("meta", {}).get("target_scaling", "quantile")
    log_scale_factor = ckpt.get("meta", {}).get("log_scale_factor", 60.0)
    quantiles        = ckpt.get("quantiles", (0.3, 0.5, 0.7))

    if 0.5 in quantiles:
        p50_idx = list(quantiles).index(0.5)
    else:
        p50_idx = len(quantiles) // 2

    ds     = PREDISPATCHDataset(X_enc, X_dec, y_norm, y_raw, y_mask)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    qt = scalers["target_rrp"]
    all_pred, all_targ, all_mask = [], [], []

    model.eval()
    with torch.no_grad():
        for X_e, X_d, _yn, yr, m, _w in loader:
            preds_norm = model(X_e, X_d)
            preds_norm, _ = torch.sort(preds_norm, dim=-1)
            pnp = preds_norm.numpy()
            B, T, Q = pnp.shape

            if target_scaling == "log":
                preds_raw = log_scale_factor * (np.exp(pnp) - 1.0)
            else:
                preds_raw = qt.inverse_transform(pnp.reshape(-1, 1)).reshape(B, T, Q)

            all_pred.append(preds_raw[:, :, p50_idx])
            all_targ.append(yr.numpy())
            all_mask.append(m.numpy())

    return (
        np.concatenate(all_pred, axis=0),
        np.concatenate(all_targ, axis=0),
        np.concatenate(all_mask, axis=0),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 3 — TFT vs LightGBM Dispatch Comparison")
    print("=" * 60)

    # ── Find overlapping run times ────────────────────────────────────────────
    print("\nFinding overlap between TFT and tactical eval run times...")

    tft_rt_all  = np.load(PARQUET_DIR / "run_times.npy")          # TFT run_times [N_tft]
    tac_rt_strat = np.load(PARQUET_DIR / "stratified_eval_run_times_tactical.npy")
    tac_rt_all  = np.load(PARQUET_DIR / "run_times_tactical.npy")

    tft_dt     = pd.DatetimeIndex(tft_rt_all)
    tac_strat_dt = pd.DatetimeIndex(tac_rt_strat)

    # Tactical eval runs on 30-min boundaries
    tac_30min = tac_strat_dt[tac_strat_dt.minute.isin([0, 30]) & (tac_strat_dt.second == 0)]

    # Intersection with TFT run_times
    tft_set = set(tft_rt_all.view(np.int64).tolist())
    overlap_ts = [ts for ts in tac_30min if ts.value in tft_set]
    print(f"  Tactical stratified eval runs: {len(tac_strat_dt)}")
    print(f"  On 30-min boundary: {len(tac_30min)}")
    print(f"  Matched in TFT run_times: {len(overlap_ts)}")

    if len(overlap_ts) < 20:
        print("ERROR: too few overlapping runs for a meaningful comparison.")
        sys.exit(1)

    overlap_int64 = set(ts.value for ts in overlap_ts)

    # Map → TFT indices
    tft_eval_idx = np.array(
        [i for i, rt in enumerate(tft_rt_all.view(np.int64)) if int(rt) in overlap_int64],
        dtype=np.int64,
    )
    # Map → tactical indices
    tac_eval_idx = np.array(
        [i for i, rt in enumerate(tac_rt_all.view(np.int64)) if int(rt) in overlap_int64],
        dtype=np.int64,
    )
    assert len(tft_eval_idx) == len(tac_eval_idx) == len(overlap_ts), \
        "Index mismatch after filtering"
    print(f"  Using {len(overlap_ts)} runs for comparison")

    # ── Load TFT arrays ───────────────────────────────────────────────────────
    print("\nLoading TFT arrays...")
    X_enc_all  = np.load(PARQUET_DIR / "X_encoder.npy")
    X_dec_all  = np.load(PARQUET_DIR / "X_decoder.npy")
    y_norm_all = np.load(PARQUET_DIR / "y_targets.npy")
    y_raw_all  = np.load(PARQUET_DIR / "y_targets_raw.npy")
    y_mask_all = np.load(PARQUET_DIR / "y_mask.npy")

    X_enc  = X_enc_all[tft_eval_idx]
    X_dec  = X_dec_all[tft_eval_idx]
    y_norm = y_norm_all[tft_eval_idx]
    y_raw  = y_raw_all[tft_eval_idx]     # [N, 144] actual $/MWh at 30-min res
    y_mask = y_mask_all[tft_eval_idx]

    with open(PARQUET_DIR / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    # ── Load TFT model ────────────────────────────────────────────────────────
    print("Loading TFT checkpoint...")
    ckpt  = torch.load(TFT_DIR / "checkpoint_best.pt", map_location="cpu",
                       weights_only=False)
    cfg   = ckpt["model_config"]
    model = TFTPriceModel(**cfg)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

    # ── Run TFT inference ─────────────────────────────────────────────────────
    print("Running TFT inference...")
    tft_pred_30, _, tft_mask_30 = run_tft_inference_subset(
        model, X_enc, X_dec, y_norm, y_raw, y_mask, scalers
    )
    # tft_pred_30: [N, 144] p50 at 30-min resolution

    # Step-hold to 5-min: steps 0–1 → 12 × 5-min intervals
    #   step 0 (TFT 0–30 min) → 5-min intervals 0–5
    #   step 1 (TFT 30–60 min) → 5-min intervals 6–11
    tft_fcst_5min = np.stack([
        tft_pred_30[:, 0],  # TFT step 0 q50
        tft_pred_30[:, 1],  # TFT step 1 q50
    ], axis=1)
    # Expand: [N, 2] → [N, 12] by repeating each 30-min block 6 times
    tft_fcst_5min = np.repeat(tft_fcst_5min, 6, axis=1)   # [N, 12]

    # ── Load tactical arrays for the same runs ────────────────────────────────
    print("Loading tactical arrays for comparison runs...")
    X_tac  = np.load(PARQUET_DIR / "X_tactical.npy")[tac_eval_idx]
    y_tac  = np.load(PARQUET_DIR / "y_tactical.npy")[tac_eval_idx]
    ym_tac = np.load(PARQUET_DIR / "y_tactical_mask.npy")[tac_eval_idx]

    with open(PARQUET_DIR / "tactical_meta.json") as f:
        meta = json.load(f)
    feat_names = meta["feature_names"]

    # ── Load LightGBM q50 model ───────────────────────────────────────────────
    print("Loading LightGBM models...")
    with open(MODEL_DIR / "lgbm_q50.pkl", "rb") as f:
        lgbm_q50_model = pickle.load(f)

    # Build long-format predictions for the comparison runs
    n_runs = len(tac_eval_idx)
    n_h = 12
    X_rep = np.repeat(X_tac, n_h, axis=0).astype(np.float32)
    horizon_col = np.tile(np.arange(n_h, dtype=np.float32), n_runs).reshape(-1, 1)
    X_long = np.hstack([X_rep, horizon_col])
    X_long_df = pd.DataFrame(X_long, columns=feat_names + ["horizon"])
    lgbm_fcst = lgbm_q50_model.predict(X_long_df).reshape(n_runs, n_h).astype(np.float64)

    # ── Stratum labels ────────────────────────────────────────────────────────
    y_masked = np.where(ym_tac, y_tac, np.nan)
    max_rrp  = np.nanmax(y_masked, axis=1)
    min_rrp  = np.nanmin(y_masked, axis=1)
    strata = np.where(max_rrp >= SPIKE_THRESH, "spike",
             np.where(min_rrp <  LOW_THRESH,   "low", "normal"))

    stratum_masks = {
        "spike":  strata == "spike",
        "low":    strata == "low",
        "normal": strata == "normal",
        "all":    np.ones(n_runs, dtype=bool),
    }
    counts = {k: int(v.sum()) for k, v in stratum_masks.items()}
    print(f"\n  Comparison set strata: spike={counts['spike']}, "
          f"low={counts['low']}, normal={counts['normal']}")

    # ── Run MPC simulation ────────────────────────────────────────────────────
    p5min_fcst = X_tac[:, :12].astype(np.float64)
    oracle_fcst = np.where(ym_tac, y_tac, p5min_fcst).astype(np.float64)

    strategies = {
        "oracle":    oracle_fcst,
        "p5min":     p5min_fcst,
        "lgbm_q50":  lgbm_fcst,
        "tft_q50":   tft_fcst_5min,
    }
    revenues = {s: np.zeros(n_runs) for s in strategies}

    print(f"\nRunning MPC simulation ({n_runs} runs × 4 strategies)...")
    for i in range(n_runs):
        actual = np.where(ym_tac[i], y_tac[i], p5min_fcst[i]).astype(np.float64)
        for name, fcast_2d in strategies.items():
            revenues[name][i] = simulate_mpc(fcast_2d[i], actual)["revenue"]

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Results — TFT vs LightGBM Dispatch Comparison")
    print(f"(Note: TFT uses 30-min step-hold for 5-min intervals)")
    print("=" * 60)

    results = {}
    for sname in ["spike", "low", "normal", "all"]:
        mask = stratum_masks[sname]
        n = int(mask.sum())
        if n == 0:
            continue

        or_  = revenues["oracle"][mask]
        p5_  = revenues["p5min"][mask]
        lgb_ = revenues["lgbm_q50"][mask]
        tft_ = revenues["tft_q50"][mask]

        def reg(x): return float(np.mean(or_ - x))
        def reg_pct(x): return reg(x) / max(abs(float(np.mean(or_))), 1e-9) * 100

        print(f"\n  [{sname.upper():7s}]  n={n}")
        print(f"    {'Strategy':<12}  {'Mean Rev ($)':<14}  {'Regret ($)':<12}  {'Regret %'}")
        print(f"    {'-'*56}")
        for label, arr in [("oracle", or_), ("p5min", p5_), ("lgbm_q50", lgb_), ("tft_q50", tft_)]:
            print(f"    {label:<12}  {np.mean(arr):>12.4f}    {np.mean(or_-arr):>10.4f}    "
                  f"{reg_pct(arr):>6.1f}%")

        lgbm_vs_tft = float(np.mean(lgb_) - np.mean(tft_))
        lgbm_better = lgbm_vs_tft > 0
        print(f"    → LightGBM vs TFT revenue: {lgbm_vs_tft:+.4f} $ "
              f"({'LightGBM better' if lgbm_better else 'TFT better'})")

        results[sname] = {
            "n": n,
            "mean_revenue": {
                "oracle":   float(np.mean(or_)),
                "p5min":    float(np.mean(p5_)),
                "lgbm_q50": float(np.mean(lgb_)),
                "tft_q50":  float(np.mean(tft_)),
            },
            "mean_regret_vs_oracle": {
                "p5min":    reg(p5_),
                "lgbm_q50": reg(lgb_),
                "tft_q50":  reg(tft_),
            },
            "lgbm_vs_tft_revenue": lgbm_vs_tft,
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "tft_dispatch_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_comparison_runs": n_runs,
            "note": (
                "TFT uses step-hold: TFT 30-min step 0 q50 held for 5-min intervals 0-5, "
                "step 1 q50 for intervals 6-11. Comparison set = tactical stratified eval "
                "runs at 30-min boundaries that exist in TFT run_times."
            ),
            "strata_counts": counts,
            "results_by_stratum": results,
        }, f, indent=2)
    print(f"\n  Results saved → {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
