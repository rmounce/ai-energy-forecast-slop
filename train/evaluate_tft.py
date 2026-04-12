#!/usr/bin/env python3
"""
Rolling-origin evaluation: TFT vs LightGBM nMAPE comparison.

Loads the best TFT checkpoint and evaluates it on the pre-built validation
split (last ~30 days of PREDISPATCH runs). Compares against LightGBM forecasts
from price_forecast_log.csv over the same time window.

Horizon buckets: 1h, 2h, 4h, 8h, 16h, 28h
  Each bucket is cumulative (1-step to Nh), masked to valid steps only.
  28–72h skipped: ~11% coverage makes that bucket noisy before NEMSEER backfill.

Each bucket reports three nMAPE columns:
  all      — all valid steps (existing metric)
  base     — steps where actual RRP ≤ --spike-threshold (baseload)
  spike    — steps where actual RRP >  --spike-threshold (spikes/high price events)

Output:
  Printed comparison table + models/tft_price/evaluation_results.csv

Usage:
    python train/evaluate_tft.py --eval-set val
    python train/evaluate_tft.py --eval-set stratified
    python train/evaluate_tft.py --batch-size 512 --spike-threshold 150
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from train_tft_price import TFTPriceModel, PREDISPATCHDataset  # noqa: E402

PARQUET_DIR = ROOT / "data" / "parquet"
MODELS_DIR  = ROOT / "models" / "tft_price"

# (label, lo_step, hi_step) — Python slice [lo:hi], each step = 30 min
HORIZON_BUCKETS = [
    ("1h",   0,   2),
    ("2h",   0,   4),
    ("4h",   0,   8),
    ("8h",   0,  16),
    ("16h",  0,  32),
    ("28h",  0,  56),
]


# ─── TFT inference ────────────────────────────────────────────────────────────

def run_tft_inference(model, val_ds, scalers, batch_size=256,
                      target_scaling="quantile", log_scale_factor=60.0):
    """Run TFT inference on val set.

    Returns:
        pred_raw:  [N, 144]    — p50 in raw $/MWh (for nMAPE)
        targ_raw:  [N, 144]    — actuals in raw $/MWh
        mask:      [N, 144]    — bool valid steps
        preds_all: [N, 144, 3] — all quantiles (q30/q50/q70) in raw $/MWh
    """
    qt = scalers["target_rrp"]
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_pred, all_targ, all_mask, all_quants = [], [], [], []

    model.eval()
    with torch.no_grad():
        for X_enc, X_dec, _y_norm, y_raw, mask, _weights in loader:
            preds_norm = model(X_enc, X_dec)                     # [B, T, 3]
            preds_norm, _ = torch.sort(preds_norm, dim=-1)       # prevent quantile crossing
            preds_norm_np = preds_norm.numpy()                   # [B, T, 3]

            B, T, Q = preds_norm_np.shape
            if target_scaling == "log":
                preds_raw = log_scale_factor * (np.exp(preds_norm_np) - 1.0)
            else:
                preds_raw = qt.inverse_transform(
                    preds_norm_np.reshape(-1, 1)
                ).reshape(B, T, Q)

            all_pred.append(preds_raw[:, :, 1])   # p50
            all_targ.append(y_raw.numpy())
            all_mask.append(mask.numpy())
            all_quants.append(preds_raw)

    return (
        np.concatenate(all_pred,   axis=0),   # [N_val, 144]
        np.concatenate(all_targ,   axis=0),   # [N_val, 144]
        np.concatenate(all_mask,   axis=0),   # [N_val, 144] bool
        np.concatenate(all_quants, axis=0),   # [N_val, 144, 3]
    )


def quantile_calibration(preds_all, targ, mask, quantiles=(0.1, 0.5, 0.9)):
    """Compute empirical coverage rates for each quantile.

    For a well-calibrated model, P(actual <= pred_q) should equal q.
    Returns list of (q, empirical_coverage) over all valid steps.
    """
    flat_mask = mask.reshape(-1)
    flat_targ = targ.reshape(-1)[flat_mask]
    results = []
    for i, q in enumerate(quantiles):
        flat_pred_q = preds_all[:, :, i].reshape(-1)[flat_mask]
        coverage = (flat_targ <= flat_pred_q).mean()
        results.append((q, float(coverage)))
    return results


def bucket_nmape(pred, targ, mask, lo, hi, price_mask=None):
    """Global nMAPE for decoder steps [lo:hi], valid (masked) steps only.

    Uses sum(|e|)/sum(|y|) — scale-invariant, not distorted by near-zero prices.

    price_mask: optional bool array [N, 144] — further restricts to a price band.
    Returns (nmape_pct, n_valid_steps).
    """
    bm = mask[:, lo:hi]
    if price_mask is not None:
        bm = bm & price_mask[:, lo:hi]
    bp = pred[:, lo:hi][bm]
    bt = targ[:, lo:hi][bm]
    if len(bt) == 0:
        return float("nan"), 0
    denom = np.abs(bt).sum()
    if denom == 0:
        return float("nan"), 0
    return np.abs(bp - bt).sum() / denom * 100, len(bt)


# ─── LightGBM baseline ────────────────────────────────────────────────────────

def load_lgbm_log(path, val_start, val_end):
    """Load price_forecast_log.csv, filter to val window, compute horizon_h.

    val_start / val_end: timezone-aware UTC datetimes.
    Returns DataFrame with horizon_h column, or empty DataFrame if no overlap.
    """
    df = pd.read_csv(path, low_memory=False)

    # Parse datetime columns explicitly, normalise to UTC
    for col in ("forecast_creation_time", "forecast_target_time"):
        parsed = pd.to_datetime(df[col], utc=True, errors="coerce")
        df[col] = parsed

    # Filter to creation times within the TFT val window
    df = df[
        (df["forecast_creation_time"] >= val_start) &
        (df["forecast_creation_time"] <= val_end)
    ].copy()

    if df.empty:
        return df

    df["horizon_h"] = (
        (df["forecast_target_time"] - df["forecast_creation_time"])
        .dt.total_seconds() / 3600
    )

    return df


def lgbm_bucket_nmape(df, hi_h):
    """nMAPE from LightGBM log for forecasts with 0 < horizon_h <= hi_h.

    prediction/actual are linearly scaled — nMAPE is scale-invariant so the
    result is identical to computing in raw $/MWh.
    Returns (nmape_pct, n_rows).
    """
    sub = df[(df["horizon_h"] > 0) & (df["horizon_h"] <= hi_h)]
    if sub.empty:
        return float("nan"), 0
    p = sub["prediction"].values.astype(float)
    a = sub["actual"].values.astype(float)
    valid = np.isfinite(p) & np.isfinite(a) & (np.abs(a) > 0)
    if valid.sum() == 0:
        return float("nan"), 0
    return np.abs(p[valid] - a[valid]).sum() / np.abs(a[valid]).sum() * 100, valid.sum()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate TFT vs LightGBM on val set")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=str,
                        default=str(MODELS_DIR / "checkpoint_best.pt"))
    parser.add_argument("--log", type=str,
                        default=str(ROOT / "price_forecast_log.csv"))
    parser.add_argument("--eval-set", choices=["val", "stratified"], default="val",
                        help="Eval sample set: 'val' (default, last N days) or "
                             "'stratified' (fixed benchmark from build_stratified_eval.py)")
    parser.add_argument("--spike-threshold", type=float, default=150.0,
                        help="RRP threshold ($/MWh) splitting baseload vs spike bands "
                             "in the segmented nMAPE columns (default: 150)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    log_path  = Path(args.log)

    # ── Load checkpoint
    print("=== TFT Evaluation ===\n")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["model_config"]
    model = TFTPriceModel(**cfg)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Epoch {ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}  "
          f"nMAPE_all={ckpt['nmape_all']:.2f}%")
    print(f"  Config: {cfg}\n")

    # ── Load arrays
    print("Loading pre-built arrays...")

    if args.eval_set == "stratified":
        strat_path = PARQUET_DIR / "stratified_eval_run_times.npy"
        if not strat_path.exists():
            print("ERROR: stratified_eval_run_times.npy not found.")
            print("  Run: python data/build_stratified_eval.py")
            sys.exit(1)
        strat_rt = np.load(strat_path)                         # datetime64[ns]
        all_rt   = np.load(PARQUET_DIR / "run_times.npy")      # datetime64[ns]
        strat_int64 = set(strat_rt.view(np.int64).tolist())
        eval_idx = np.array(
            [i for i, rt in enumerate(all_rt.view(np.int64)) if rt in strat_int64],
            dtype=np.intp,
        )
        eval_label = "stratified"
        print(f"  Stratified eval: {len(eval_idx):,} samples matched "
              f"(requested {len(strat_rt):,})")
    else:
        split = np.load(PARQUET_DIR / "split_indices.npz")
        eval_idx = split["val"]
        eval_label = "val"
        print(f"  Val samples: {len(eval_idx):,}")

    X_enc  = np.load(PARQUET_DIR / "X_encoder.npy")[eval_idx]
    X_dec  = np.load(PARQUET_DIR / "X_decoder.npy")[eval_idx]
    y_norm = np.load(PARQUET_DIR / "y_targets.npy")[eval_idx]
    y_raw  = np.load(PARQUET_DIR / "y_targets_raw.npy")[eval_idx]
    y_mask = np.load(PARQUET_DIR / "y_mask.npy")[eval_idx]

    run_times_raw = np.load(PARQUET_DIR / "run_times.npy")[eval_idx]
    run_times = pd.DatetimeIndex(run_times_raw).tz_localize("UTC")

    with open(PARQUET_DIR / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    print(f"  Eval window:  {run_times.min()} → {run_times.max()}\n")

    # ── TFT inference
    val_ds = PREDISPATCHDataset(X_enc, X_dec, y_norm, y_raw, y_mask)
    print(f"Running TFT inference (batch_size={args.batch_size})...")    # ── Inference
    target_scaling = ckpt.get("meta", {}).get("target_scaling", "quantile")
    log_scale_factor = ckpt.get("meta", {}).get("log_scale_factor", 60.0)

    pred, targ, mask, preds_all = run_tft_inference(
        model, val_ds, scalers, args.batch_size,
        target_scaling=target_scaling, log_scale_factor=log_scale_factor
    )
    print(f"  Predictions: {pred.shape}  valid steps: {mask.sum():,}\n")

    # ── Price band masks for segmented nMAPE
    spike_thr = args.spike_threshold
    base_pmask  = targ <= spike_thr    # [N, 144] bool — baseload steps
    spike_pmask = targ >  spike_thr    # [N, 144] bool — spike steps
    n_spike_steps = (mask & spike_pmask).sum()
    n_base_steps  = (mask & base_pmask).sum()
    print(f"Price band split (threshold={spike_thr:.0f} $/MWh):")
    print(f"  Baseload steps (≤{spike_thr:.0f}): {n_base_steps:,}  "
          f"Spike steps (>{spike_thr:.0f}): {n_spike_steps:,}\n")

    # ── TFT nMAPE per bucket (all / baseload / spike)
    tft_results = {}
    for label, lo, hi in HORIZON_BUCKETS:
        nmape_all,   n_all   = bucket_nmape(pred, targ, mask, lo, hi)
        nmape_base,  n_base  = bucket_nmape(pred, targ, mask, lo, hi, base_pmask)
        nmape_spike, n_spike = bucket_nmape(pred, targ, mask, lo, hi, spike_pmask)
        tft_results[label] = (nmape_all, n_all, nmape_base, n_base, nmape_spike, n_spike)

    # ── LightGBM baseline
    lgbm_results = {}
    lgbm_available = False

    if log_path.exists():
        print(f"Loading LightGBM log: {log_path}")
        df_lgbm = load_lgbm_log(log_path, run_times.min(), run_times.max())
        if df_lgbm.empty:
            print(f"  WARNING: No LightGBM forecasts found in val window "
                  f"({run_times.min().date()} → {run_times.max().date()})")
            print(f"  LightGBM log may not cover this period — showing N/A\n")
        else:
            lgbm_available = True
            print(f"  LightGBM rows in window: {len(df_lgbm):,}  "
                  f"(horizon range: {df_lgbm['horizon_h'].min():.1f}h → "
                  f"{df_lgbm['horizon_h'].max():.1f}h)\n")
            for label, _lo, hi in HORIZON_BUCKETS:
                hi_h = hi * 0.5  # steps → hours
                nmape, n = lgbm_bucket_nmape(df_lgbm, hi_h)
                lgbm_results[label] = (nmape, n)
    else:
        print(f"  WARNING: LightGBM log not found at {log_path}\n")

    # ── Print comparison table
    # Columns: Horizon | TFT(all) | LGBM(all) | Delta | TFT(base) | TFT(spike)
    W = 100
    print("─" * W)
    print(f"  Eval set: {eval_label}  |  spike threshold: {spike_thr:.0f} $/MWh")
    print("─" * W)
    if lgbm_available:
        print(f"{'Horizon':>8}  {'TFT all':>8}  {'LGBM all':>9}  {'Delta':>7}  "
              f"{'TFT base':>9}  {'TFT spike':>10}  {'steps(all)':>11}")
        print("─" * W)
        for label, lo, hi in HORIZON_BUCKETS:
            tft_all, n_all, tft_base, _, tft_spike, n_spike_b = tft_results[label]
            lgbm_v, lgbm_n = lgbm_results.get(label, (float("nan"), 0))
            delta = tft_all - lgbm_v if not (np.isnan(tft_all) or np.isnan(lgbm_v)) else float("nan")

            def fmt(v): return f"{v:.1f}%" if not np.isnan(v) else "  N/A"
            def fmtd(v): return f"{v:+.1f}%" if not np.isnan(v) else "   N/A"
            print(f"{label:>8}  {fmt(tft_all):>8}  {fmt(lgbm_v):>9}  {fmtd(delta):>7}  "
                  f"{fmt(tft_base):>9}  {fmt(tft_spike):>10}  {n_all:>11,}")
    else:
        print(f"{'Horizon':>8}  {'TFT all':>8}  {'TFT base':>9}  {'TFT spike':>10}  {'steps(all)':>11}")
        print("─" * W)
        for label, lo, hi in HORIZON_BUCKETS:
            tft_all, n_all, tft_base, _, tft_spike, _ = tft_results[label]
            def fmt(v): return f"{v:.1f}%" if not np.isnan(v) else "  N/A"
            print(f"{label:>8}  {fmt(tft_all):>8}  {fmt(tft_base):>9}  {fmt(tft_spike):>10}  {n_all:>11,}")
    print("─" * W)

    # ── Note on buckets
    print("\nNote: all buckets are cumulative (1-step to Nh), valid steps only.")
    print("      28–72h omitted: 11% coverage pre-NEMSEER backfill.")
    print(f"      base = actual RRP ≤ {spike_thr:.0f}, spike = actual RRP > {spike_thr:.0f} $/MWh")

    # ── Quantile calibration
    target_quants = ckpt.get("quantiles", (0.3, 0.5, 0.7))
    cal = quantile_calibration(preds_all, targ, mask, target_quants)
    print("\n── Quantile calibration (all valid steps) ──")
    print(f"  {'Quantile':>10}  {'Expected':>10}  {'Actual':>10}  {'Bias':>8}")
    print(f"  {'-'*44}")
    for q, cov in cal:
        bias = cov - q
        bias_s = f"{bias:+.3f}"
        flag = "  ✓" if abs(bias) < 0.03 else ("  ↑ over-covers" if bias > 0 else "  ↓ under-covers")
        print(f"  {f'q{int(q*100):02d}':>10}  {q:>10.3f}  {cov:>10.3f}  {bias_s:>8}{flag}")
    print(f"  {'-'*44}")
    print("  Target: |bias| < 0.03 for reliable dispatch thresholds.")

    # ── Save CSV
    out_path = MODELS_DIR / "evaluation_results.csv"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, lo, hi in HORIZON_BUCKETS:
        tft_all, n_all, tft_base, n_base, tft_spike, n_spike_b = tft_results[label]
        lgbm_v, lgbm_n = lgbm_results.get(label, (float("nan"), 0))
        delta = tft_all - lgbm_v if not (np.isnan(tft_all) or np.isnan(lgbm_v)) else None
        def r(v): return round(float(v), 4) if not np.isnan(v) else None
        rows.append({
            "horizon": label,
            "eval_set": eval_label,
            "spike_threshold": spike_thr,
            "tft_nmape_all": r(tft_all),
            "tft_nmape_base": r(tft_base),
            "tft_nmape_spike": r(tft_spike),
            "lgbm_nmape": r(lgbm_v),
            "delta": round(delta, 4) if delta is not None else None,
            "tft_n_valid_steps": n_all,
            "tft_n_base_steps": n_base,
            "tft_n_spike_steps": n_spike_b,
            "lgbm_n_rows": lgbm_n,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
