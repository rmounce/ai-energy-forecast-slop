#!/usr/bin/env python3
"""
Overnight-stratified evaluation of the TFT load model.

Standard MAE metrics average over all times of day, masking systematic overnight
bias. This script stratifies val-set predictions by local hour and decoder horizon
window (+24h / +48h / +72h overnight), computing bias, MAE, and q10/q90 coverage
specifically for the overnight hours (3–6am Adelaide time).

Usage:
  python eval/eval_load_overnight.py
  python eval/eval_load_overnight.py --checkpoint models/tft_load/checkpoint_best.pt

Output: eval/results/load_overnight_eval.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
import torch

ROOT    = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "parquet"
RESULTS = ROOT / "eval" / "results"

DEFAULT_CHECKPOINT = ROOT / "models" / "tft_load" / "checkpoint_best.pt"
SCALERS_PATH       = PARQUET / "load_scalers.pkl"
ADL_TZ             = pytz.timezone("Australia/Adelaide")

OVERNIGHT_HOURS = {3, 4, 5, 6}   # local Adelaide hours considered "overnight"

# Horizon windows: (label, decoder_step_range) — three overnight periods in 72h
HORIZON_WINDOWS = [
    ("overnight_24h",  range(24, 72)),   # ~12–36h ahead: covers one overnight period
    ("overnight_48h",  range(72, 120)),  # ~36–60h ahead: covers second overnight
    ("overnight_72h",  range(120, 144)), # ~60–72h ahead: tail of third overnight
]


def load_model(checkpoint_path):
    sys.path.insert(0, str(ROOT / "train"))
    from train_tft_load import TFTLoadModel

    ckpt  = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = ckpt["model_config"]
    model = TFTLoadModel(
        n_enc=cfg["n_enc"], n_dec=cfg["n_dec"],
        n_quantiles=cfg.get("n_quantiles", 3),
        d_model=cfg.get("d_model", 64),
        n_heads=cfg.get("n_heads", 4),
        n_lstm_layers=cfg.get("n_lstm_layers", 2),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def run_inference(model, X_enc, X_dec, batch=256):
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_enc), batch):
            t_enc = torch.tensor(X_enc[i:i+batch])
            t_dec = torch.tensor(X_dec[i:i+batch])
            p = model(t_enc, t_dec)
            p = torch.sort(p, dim=-1).values
            all_preds.append(p.numpy())
    return np.concatenate(all_preds, axis=0)   # [N, 144, 3]


def decoder_local_hours(run_times, dec_steps=144):
    """
    For each val sample, compute the local Adelaide hour for each decoder step.
    run_times: [N] datetime64[ns] — the start of the decoder window (UTC).
    Returns [N, dec_steps] int8 array of local hours.
    """
    hours = np.empty((len(run_times), dec_steps), dtype=np.int8)
    bases = pd.to_datetime(run_times, utc=True).tz_convert(ADL_TZ)
    for s in range(dec_steps):
        hours[:, s] = (bases + pd.Timedelta(minutes=30 * s)).hour
    return hours


def compute_overnight_metrics(preds_W, y_raw, y_mask, overnight_mask, label, step_range):
    """
    Compute bias, MAE, and q10/q90 coverage for decoder steps in step_range
    that correspond to overnight local hours.

    overnight_mask: [N, dec_steps] bool — precomputed from decoder_local_hours.
    Returns dict with keys: n_steps, bias_W, mae_W, coverage, q50_mean, actual_mean.
    """
    steps = list(step_range)
    valid_mask = y_mask[:, steps] & overnight_mask[:, steps]

    if not valid_mask.any():
        return {"label": label, "n_steps": 0, "bias_W": None, "mae_W": None,
                "coverage": None, "q50_mean": None, "actual_mean": None}

    actuals = y_raw[:, steps][valid_mask]
    q10_v   = preds_W[:, steps, 0][valid_mask]
    q50_v   = preds_W[:, steps, 1][valid_mask]
    q90_v   = preds_W[:, steps, 2][valid_mask]

    bias     = float((q50_v - actuals).mean())
    mae      = float(np.abs(q50_v - actuals).mean())
    coverage = float(((actuals >= q10_v) & (actuals <= q90_v)).mean())

    print(f"  {label:<18}  n={valid_mask.sum():>6,}  "
          f"actual_mean={actuals.mean():.0f}W  q50_mean={q50_v.mean():.0f}W  "
          f"bias={bias:+.0f}W  MAE={mae:.0f}W  coverage={coverage:.3f}")

    return {
        "label":       label,
        "n_steps":     int(valid_mask.sum()),
        "bias_W":      round(bias, 1),
        "mae_W":       round(mae, 1),
        "coverage":    round(coverage, 4),
        "q50_mean":    round(float(q50_v.mean()), 1),
        "actual_mean": round(float(actuals.mean()), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print("=" * 60)
    print("Load TFT — Overnight-Stratified Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")

    model, ckpt = load_model(checkpoint_path)

    with open(SCALERS_PATH, "rb") as f:
        scalers = pickle.load(f)
    load_sc = scalers["power_load"]

    X_enc      = np.load(PARQUET / "X_enc_load.npy",     mmap_mode="r")
    X_dec      = np.load(PARQUET / "X_dec_load.npy",     mmap_mode="r")
    y_raw      = np.load(PARQUET / "y_load_raw.npy",     mmap_mode="r")
    y_mask     = np.load(PARQUET / "y_load_mask.npy",    mmap_mode="r")
    run_times  = np.load(PARQUET / "run_times_load.npy", mmap_mode="r")
    split      = np.load(PARQUET / "split_indices_load.npz")
    val_idx    = split["val"]

    X_enc_v   = X_enc[val_idx]
    X_dec_v   = X_dec[val_idx]
    y_raw_v   = y_raw[val_idx]
    y_mask_v  = y_mask[val_idx]
    rt_v      = run_times[val_idx]

    print(f"\nVal samples: {len(val_idx):,}  "
          f"({pd.Timestamp(rt_v.min()).date()} → {pd.Timestamp(rt_v.max()).date()})")

    print("\nRunning inference...")
    preds_norm = run_inference(model, X_enc_v, X_dec_v)
    preds_W    = load_sc.inverse_transform(
        preds_norm.reshape(-1, 1)
    ).reshape(preds_norm.shape).clip(min=0.0)
    del preds_norm

    print("Computing local hours for each decoder step...")
    local_hours    = decoder_local_hours(rt_v)
    overnight_mask = np.isin(local_hours, list(OVERNIGHT_HOURS))

    print(f"\nOvernight metrics (hours {sorted(OVERNIGHT_HOURS)}, Adelaide time):")
    print(f"  {'Horizon':<18}  {'n':>8}  {'actual_mean':>11}  {'q50_mean':>8}  "
          f"{'bias':>8}  {'MAE':>6}  {'cov':>6}")
    print("  " + "-" * 75)

    results = []
    for label, step_range in HORIZON_WINDOWS:
        r = compute_overnight_metrics(preds_W, y_raw_v, y_mask_v, overnight_mask, label, step_range)
        results.append(r)

    r_all = compute_overnight_metrics(preds_W, y_raw_v, y_mask_v, overnight_mask,
                                      "overnight_all", range(24, 144))
    results.append(r_all)

    RESULTS.mkdir(exist_ok=True)
    out = {
        "generated":  pd.Timestamp.now(tz="UTC").isoformat(),
        "checkpoint": str(checkpoint_path),
        "val_start":  str(pd.Timestamp(rt_v.min()).isoformat()),
        "val_end":    str(pd.Timestamp(rt_v.max()).isoformat()),
        "n_val":      int(len(val_idx)),
        "overnight_hours": sorted(OVERNIGHT_HOURS),
        "metrics":    results,
    }
    out_path = RESULTS / "load_overnight_eval.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
