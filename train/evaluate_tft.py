#!/usr/bin/env python3
"""
Rolling-origin evaluation: TFT vs LightGBM nMAPE comparison.

Loads the best TFT checkpoint and evaluates it on the pre-built validation
split (last ~30 days of PREDISPATCH runs). Compares against LightGBM forecasts
from price_forecast_log.csv over the same time window.

Horizon buckets: 2h, 4h, 8h, 16h, 28h
  Each bucket is cumulative (1-step to Nh), masked to valid steps only.
  28–72h skipped: ~11% coverage makes that bucket noisy before NEMSEER backfill.

Output:
  Printed comparison table + models/tft_price/evaluation_results.csv

Usage:
    python train/evaluate_tft.py
    python train/evaluate_tft.py --batch-size 512
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
    ("2h",   0,   4),
    ("4h",   0,   8),
    ("8h",   0,  16),
    ("16h",  0,  32),
    ("28h",  0,  56),
]


# ─── TFT inference ────────────────────────────────────────────────────────────

def run_tft_inference(model, val_ds, scalers, batch_size=256):
    """Run TFT inference on val set. Returns pred_raw, targ_raw, mask arrays."""
    qt = scalers["target_rrp"]
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_pred, all_targ, all_mask = [], [], []

    model.eval()
    with torch.no_grad():
        for X_enc, X_dec, _y_norm, y_raw, mask in loader:
            preds_norm = model(X_enc, X_dec)                     # [B, T, 3]
            preds_norm, _ = torch.sort(preds_norm, dim=-1)       # prevent quantile crossing
            p50_norm = preds_norm[:, :, 1].numpy()               # median

            B, T = p50_norm.shape
            p50_raw = qt.inverse_transform(
                p50_norm.reshape(-1, 1)
            ).reshape(B, T)

            all_pred.append(p50_raw)
            all_targ.append(y_raw.numpy())
            all_mask.append(mask.numpy())

    return (
        np.concatenate(all_pred, axis=0),   # [N_val, 144]
        np.concatenate(all_targ, axis=0),   # [N_val, 144]
        np.concatenate(all_mask, axis=0),   # [N_val, 144] bool
    )


def bucket_nmape(pred, targ, mask, lo, hi):
    """Global nMAPE for decoder steps [lo:hi], valid (masked) steps only.

    Uses sum(|e|)/sum(|y|) — scale-invariant, not distorted by near-zero prices.
    Returns (nmape_pct, n_valid_steps).
    """
    bm = mask[:, lo:hi]
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
    split = np.load(PARQUET_DIR / "split_indices.npz")
    val_idx = split["val"]

    X_enc  = np.load(PARQUET_DIR / "X_encoder.npy")[val_idx]
    X_dec  = np.load(PARQUET_DIR / "X_decoder.npy")[val_idx]
    y_norm = np.load(PARQUET_DIR / "y_targets.npy")[val_idx]
    y_raw  = np.load(PARQUET_DIR / "y_targets_raw.npy")[val_idx]
    y_mask = np.load(PARQUET_DIR / "y_mask.npy")[val_idx]

    run_times_raw = np.load(PARQUET_DIR / "run_times.npy")[val_idx]
    run_times = pd.DatetimeIndex(run_times_raw).tz_localize("UTC")

    with open(PARQUET_DIR / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    print(f"  Val samples: {len(val_idx)}")
    print(f"  Val window:  {run_times.min()} → {run_times.max()}\n")

    # ── TFT inference
    val_ds = PREDISPATCHDataset(X_enc, X_dec, y_norm, y_raw, y_mask)
    print(f"Running TFT inference (batch_size={args.batch_size})...")
    pred, targ, mask = run_tft_inference(model, val_ds, scalers, args.batch_size)
    print(f"  Predictions: {pred.shape}  valid steps: {mask.sum():,}\n")

    # ── TFT nMAPE per bucket
    tft_results = {}
    for label, lo, hi in HORIZON_BUCKETS:
        nmape, n = bucket_nmape(pred, targ, mask, lo, hi)
        tft_results[label] = (nmape, n)

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
    print("─" * 68)
    if lgbm_available:
        print(f"{'Horizon':>8}  {'TFT nMAPE':>10}  {'LGBM nMAPE':>11}  "
              f"{'Delta':>7}  {'TFT steps':>10}  {'LGBM rows':>10}")
        print("─" * 68)
        for label, lo, hi in HORIZON_BUCKETS:
            tft_n, lgbm_n = tft_results[label][1], lgbm_results.get(label, (float("nan"), 0))[1]
            tft_v  = tft_results[label][0]
            lgbm_v = lgbm_results.get(label, (float("nan"), 0))[0]
            delta  = tft_v - lgbm_v if not (np.isnan(tft_v) or np.isnan(lgbm_v)) else float("nan")
            delta_s  = f"{delta:+.1f}%" if not np.isnan(delta) else "   N/A"
            tft_s    = f"{tft_v:.1f}%"  if not np.isnan(tft_v)  else "  N/A"
            lgbm_s   = f"{lgbm_v:.1f}%" if not np.isnan(lgbm_v) else "  N/A"
            print(f"{label:>8}  {tft_s:>10}  {lgbm_s:>11}  "
                  f"{delta_s:>7}  {tft_n:>10,}  {lgbm_n:>10,}")
    else:
        print(f"{'Horizon':>8}  {'TFT nMAPE':>10}  {'Valid steps':>12}")
        print("─" * 36)
        for label, lo, hi in HORIZON_BUCKETS:
            tft_v, tft_n = tft_results[label]
            tft_s = f"{tft_v:.1f}%" if not np.isnan(tft_v) else "  N/A"
            print(f"{label:>8}  {tft_s:>10}  {tft_n:>12,}")
    print("─" * 68)

    # ── Note on buckets
    print("\nNote: all buckets are cumulative (1-step to Nh), valid steps only.")
    print("      28–72h omitted: 11% coverage pre-NEMSEER backfill.")

    # ── Save CSV
    out_path = MODELS_DIR / "evaluation_results.csv"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, lo, hi in HORIZON_BUCKETS:
        tft_v, tft_n = tft_results[label]
        lgbm_v, lgbm_n = lgbm_results.get(label, (float("nan"), 0))
        delta = tft_v - lgbm_v if not (np.isnan(tft_v) or np.isnan(lgbm_v)) else None
        rows.append({
            "horizon": label,
            "tft_nmape": round(tft_v, 4) if not np.isnan(tft_v) else None,
            "lgbm_nmape": round(lgbm_v, 4) if not np.isnan(lgbm_v) else None,
            "delta": round(delta, 4) if delta is not None else None,
            "tft_n_valid_steps": tft_n,
            "lgbm_n_rows": lgbm_n,
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
