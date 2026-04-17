#!/usr/bin/env python3
"""
Train a TFT-style household load forecast model.

Architecture: LSTM encoder-decoder with cross-attention (same as TFT price model).
  Encoder (96 steps = 48h): power_load, power_pv, weather, calendar features
  Decoder (144 steps = 72h): weather forecast, calendar features, horizon_norm
  Output: 3-quantile prediction [q10, q50, q90] in Watts

Loss: horizon-weighted quantile loss — weight_h = exp(-h/tau), default tau=24 (12h).
Short-horizon steps (0–24h) dominate gradients since EMHASS only uses 24-48h.

See data/build_load_dataset.py for feature definitions and array shapes.

Input data: pre-built numpy arrays from data/build_load_dataset.py
  data/parquet/X_enc_load.npy    [N, 96,  13]  normalised encoder features
  data/parquet/X_dec_load.npy    [N, 144, 13]  normalised decoder features
  data/parquet/y_load.npy        [N, 144]      normalised target
  data/parquet/y_load_raw.npy    [N, 144]      raw target in Watts
  data/parquet/y_load_mask.npy   [N, 144]      bool: valid step
  data/parquet/split_indices_load.npz          train/val split
  data/parquet/load_scalers.pkl               StandardScaler per feature

Output:
  models/tft_load/checkpoint_best.pt
  models/tft_load/training_log.csv

Usage:
  python train/train_tft_load.py
  python train/train_tft_load.py --epochs 100 --d-model 64
  python train/train_tft_load.py --dry-run
"""

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT       = Path(__file__).resolve().parent.parent
PARQUET    = ROOT / "data" / "parquet"
MODELS_DIR = ROOT / "models" / "tft_load"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.10, 0.50, 0.90]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class LoadDataset(Dataset):
    def __init__(self, X_enc, X_dec, y, y_raw, mask):
        self.X_enc = torch.tensor(X_enc, dtype=torch.float32)
        self.X_dec = torch.tensor(X_dec, dtype=torch.float32)
        self.y     = torch.tensor(y,     dtype=torch.float32)
        self.y_raw = torch.tensor(y_raw, dtype=torch.float32)
        self.mask  = torch.tensor(mask,  dtype=torch.bool)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_enc[i], self.X_dec[i], self.y[i], self.y_raw[i], self.mask[i]


# ─── Model ───────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1     = nn.Linear(d_model, d_model)
        self.fc2     = nn.Linear(d_model, d_model)
        self.gate    = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.elu     = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        g = self.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class TFTLoadModel(nn.Module):
    """
    Lightweight TFT-style encoder-decoder for load prediction.
    Same architecture as TFTPriceModel but with n_quantiles param.
    """
    def __init__(self, n_enc: int, n_dec: int, n_quantiles: int = 3,
                 d_model: int = 64, n_heads: int = 4,
                 n_lstm_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.enc_proj = nn.Sequential(
            nn.Linear(n_enc, d_model),
            GatedResidualNetwork(d_model, dropout),
        )
        self.dec_proj = nn.Sequential(
            nn.Linear(n_dec, d_model),
            GatedResidualNetwork(d_model, dropout),
        )
        self.encoder_lstm = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.decoder_lstm = nn.LSTM(
            d_model, d_model, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.attention   = nn.MultiheadAttention(d_model, n_heads,
                                                  batch_first=True, dropout=dropout)
        self.attn_grn    = GatedResidualNetwork(d_model, dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, n_quantiles)

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        enc_emb = self.enc_proj(enc)
        dec_emb = self.dec_proj(dec)
        enc_out, (h, c) = self.encoder_lstm(enc_emb)
        dec_out, _      = self.decoder_lstm(dec_emb, (h, c))
        attn_out, _     = self.attention(dec_out, enc_out, enc_out)
        dec_out         = self.output_norm(dec_out + attn_out)
        dec_out         = self.attn_grn(dec_out)
        return self.output_proj(dec_out)                 # [B, T_dec, Q]


# ─── Loss ────────────────────────────────────────────────────────────────────

class HorizonWeightedQuantileLoss(nn.Module):
    """
    Quantile loss with optional per-step horizon weights.
    mask [B, T] marks valid decoder steps.
    horizon_weights [T]: exp(-step/tau) — higher weight at short horizons.
    """
    def __init__(self, quantiles=QUANTILES, horizon_weights=None):
        super().__init__()
        self.quantiles = quantiles
        if horizon_weights is not None:
            self.register_buffer("horizon_weights", horizon_weights)
        else:
            self.horizon_weights = None

    def forward(self, preds, targets, mask):
        eff_w = self.horizon_weights.unsqueeze(0) * mask.float() \
            if self.horizon_weights is not None else mask.float()
        w_sum = eff_w.sum(dim=1).clamp(min=1e-8)

        loss = torch.tensor(0.0, device=preds.device)
        for i, q in enumerate(self.quantiles):
            e         = targets - preds[:, :, i]
            step_loss = torch.max(q * e, (q - 1) * e)
            loss      = loss + ((step_loss * eff_w).sum(dim=1) / w_sum).mean()
        return loss / len(self.quantiles)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_wmae(model, loader, scalers, horizon_weights_np):
    """Compute weighted MAE in Watts for the q50 (median) quantile."""
    model.eval()
    load_sc = scalers["power_load"]
    q50_idx = QUANTILES.index(0.5)

    all_pred, all_targ, all_mask = [], [], []
    with torch.no_grad():
        for X_e, X_d, _yn, yr, m in loader:
            preds = model(X_e, X_d)
            preds, _ = torch.sort(preds, dim=-1)
            p50_norm = preds[:, :, q50_idx].numpy()
            p50_raw  = load_sc.inverse_transform(
                p50_norm.reshape(-1, 1)
            ).reshape(p50_norm.shape)
            all_pred.append(p50_raw)
            all_targ.append(yr.numpy())
            all_mask.append(m.numpy())

    pred  = np.concatenate(all_pred,  axis=0)
    targ  = np.concatenate(all_targ,  axis=0)
    masks = np.concatenate(all_mask,  axis=0)

    # Unweighted MAE
    valid     = masks.astype(bool)
    mae_all   = np.abs(pred[valid] - targ[valid]).mean()

    # Horizon-weighted MAE (first 48h weighted more)
    hw        = horizon_weights_np[np.newaxis, :] if horizon_weights_np is not None else np.ones((1, pred.shape[1]))
    eff_w     = hw * masks
    w_numer   = (eff_w * np.abs(pred - targ)).sum()
    w_denom   = eff_w.sum()
    wmae      = w_numer / w_denom if w_denom > 0 else float("nan")

    # Bucket MAEs (W)
    def bucket(lo, hi):
        bm = masks[:, lo:hi].astype(bool)
        if bm.sum() == 0: return float("nan")
        return float(np.abs(pred[:, lo:hi][bm] - targ[:, lo:hi][bm]).mean())

    return mae_all, wmae, {
        "0-24h":  bucket(0,  48),
        "24-48h": bucket(48, 96),
        "48-72h": bucket(96, 144),
    }


def run_epoch(model, loader, criterion, optimiser=None):
    training = optimiser is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for X_e, X_d, y, _yr, m in loader:
            preds = model(X_e, X_d)
            loss  = criterion(preds, y, m)
            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
            total += loss.item()
            n     += 1
    return total / n if n else float("nan")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TFT load model")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch-size",     type=int,   default=256)
    parser.add_argument("--d-model",        type=int,   default=64)
    parser.add_argument("--n-heads",        type=int,   default=4)
    parser.add_argument("--n-layers",       type=int,   default=2)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--horizon-decay",  type=float, default=24.0,
                        help="Exp decay tau for horizon loss weighting (steps). "
                             "Default 24 (12h half-weight). 0 = uniform.")
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--dry-run",        action="store_true",
                        help="5 epochs, batch_size=64, validate shapes")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs     = 5
        args.batch_size = 64
        print("[dry-run] 5 epochs, batch_size=64")

    # ── Load data
    print("=== TFT Load Model Training ===")
    print("\nLoading pre-built arrays...")
    X_enc  = np.load(PARQUET / "X_enc_load.npy")
    X_dec  = np.load(PARQUET / "X_dec_load.npy")
    y      = np.load(PARQUET / "y_load.npy")
    y_raw  = np.load(PARQUET / "y_load_raw.npy")
    y_mask = np.load(PARQUET / "y_load_mask.npy")
    split  = np.load(PARQUET / "split_indices_load.npz")
    train_idx, val_idx = split["train"], split["val"]

    with open(PARQUET / "load_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    with open(PARQUET / "load_dataset_meta.json") as f:
        meta = json.load(f)

    print(f"  X_enc:  {X_enc.shape}")
    print(f"  X_dec:  {X_dec.shape}")
    print(f"  y:      {y.shape}")
    print(f"  mask:   {y_mask.shape}  valid={y_mask.sum():,}")
    print(f"  Train/val: {len(train_idx):,} / {len(val_idx):,}")

    # ── DataLoaders
    weights_path = PARQUET / "y_load_weights.npy"
    if weights_path.exists():
        all_weights  = np.load(weights_path)
        train_weights = all_weights[train_idx]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.tensor(train_weights, dtype=torch.float64),
            num_samples=len(train_idx),
            replacement=True,
        )
        print(f"  Sample weights loaded: tau decay active "
              f"(min={train_weights.min():.4f} max={train_weights.max():.4f})")
        use_sampler = True
    else:
        sampler     = None
        use_sampler = False
        print("  No sample weights found — uniform sampling")

    train_ds = LoadDataset(X_enc[train_idx], X_dec[train_idx],
                           y[train_idx],     y_raw[train_idx], y_mask[train_idx])
    val_ds   = LoadDataset(X_enc[val_idx],   X_dec[val_idx],
                           y[val_idx],       y_raw[val_idx],   y_mask[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, shuffle=(not use_sampler), num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_enc = X_enc.shape[2]
    n_dec = X_dec.shape[2]

    # ── Model
    model = TFTLoadModel(
        n_enc=n_enc, n_dec=n_dec, n_quantiles=len(QUANTILES),
        d_model=args.d_model, n_heads=args.n_heads,
        n_lstm_layers=args.n_layers, dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: d_model={args.d_model}  heads={args.n_heads}  layers={args.n_layers}")
    print(f"  Parameters: {n_params:,}")

    # ── Loss + optimiser
    if args.horizon_decay > 0:
        steps = torch.arange(meta["dec_steps"], dtype=torch.float32)
        hw    = torch.exp(-steps / args.horizon_decay)
    else:
        hw = None
    criterion = HorizonWeightedQuantileLoss(QUANTILES, horizon_weights=hw)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2)

    hw_numpy = hw.numpy() if hw is not None else None

    if args.horizon_decay > 0:
        w12h = float(torch.exp(torch.tensor(-24.0 / args.horizon_decay)))
        w24h = float(torch.exp(torch.tensor(-48.0 / args.horizon_decay)))
        w48h = float(torch.exp(torch.tensor(-96.0 / args.horizon_decay)))
        print(f"  Horizon weights: tau={args.horizon_decay:.0f}  "
              f"12h={w12h:.2f}  24h={w24h:.2f}  48h={w48h:.2f}")
    else:
        print("  Horizon weights: uniform (disabled)")

    # ── Training loop
    best_wmae     = float("inf")
    patience_cnt  = 0
    log_rows      = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'wMAE W':>9}  "
          f"{'LR':>10}  {'0-24h':>7}  {'24-48h':>7}  {'Time':>6}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        prev_lr    = optimiser.param_groups[0]["lr"]
        train_loss = run_epoch(model, train_loader, criterion, optimiser)
        val_loss   = run_epoch(model, val_loader,   criterion)
        mae_all, wmae, buckets = evaluate_wmae(model, val_loader, scalers, hw_numpy)
        elapsed    = time.time() - t0

        target_metric = wmae if not np.isnan(wmae) else val_loss
        scheduler.step(target_metric)
        lr_now = optimiser.param_groups[0]["lr"]

        wmae_s = f"{wmae:>9.1f}" if not np.isnan(wmae) else "      nan"
        b0  = f"{buckets['0-24h']:>7.1f}"  if not np.isnan(buckets['0-24h'])  else "    nan"
        b24 = f"{buckets['24-48h']:>7.1f}" if not np.isnan(buckets['24-48h']) else "    nan"
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  {wmae_s}  "
              f"{lr_now:>10.2e}  {b0}  {b24}  {elapsed:>5.1f}s")

        log_rows.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "wmae_W":     round(wmae,        2) if not np.isnan(wmae)              else None,
            "mae_all_W":  round(mae_all,     2) if not np.isnan(mae_all)           else None,
            "mae_0-24h":  round(buckets["0-24h"],  2) if not np.isnan(buckets["0-24h"])  else None,
            "mae_24-48h": round(buckets["24-48h"], 2) if not np.isnan(buckets["24-48h"]) else None,
            "mae_48-72h": round(buckets["48-72h"], 2) if not np.isnan(buckets["48-72h"]) else None,
            "lr":         lr_now,
            "time_s":     round(elapsed, 1),
        })

        if target_metric < best_wmae:
            best_wmae    = target_metric
            patience_cnt = 0
            ckpt = {
                "epoch":        epoch,
                "val_loss":     val_loss,
                "wmae_W":       wmae,
                "mae_buckets":  buckets,
                "model_state":  model.state_dict(),
                "model_config": {
                    "n_enc":        n_enc,
                    "n_dec":        n_dec,
                    "n_quantiles":  len(QUANTILES),
                    "d_model":      args.d_model,
                    "n_heads":      args.n_heads,
                    "n_lstm_layers": args.n_layers,
                    "dropout":      args.dropout,
                },
                "quantiles":    QUANTILES,
                "meta":         meta,
            }
            torch.save(ckpt, MODELS_DIR / "checkpoint_best.pt")
            import datetime as _dt
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(ckpt, MODELS_DIR / f"checkpoint_{_ts}_ep{epoch}.pt")
            print(f"         ↑ best wMAE={target_metric:.1f}W")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

        if lr_now < prev_lr:
            print(f"         ↓ LR {prev_lr:.2e} → {lr_now:.2e}")

    # ── Save training log
    log_path = MODELS_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log: {log_path}")

    def _wmae_key(r):
        v = r["wmae_W"]
        return v if (v is not None and not np.isnan(v)) else float("inf")

    best_row = min(log_rows, key=_wmae_key)
    if _wmae_key(best_row) == float("inf"):
        best_row = min(log_rows, key=lambda r: r["val_loss"])

    print(f"\n=== Training complete ===")
    print(f"  Best epoch:   {best_row['epoch']}")
    print(f"  Best val loss: {best_row['val_loss']:.4f}")
    if best_row.get("wmae_W") is not None:
        print(f"  Best wMAE:    {best_row['wmae_W']:.1f} W")
    if best_row.get("mae_0-24h") is not None:
        print(f"  MAE  0-24h:   {best_row['mae_0-24h']:.1f} W")
    if best_row.get("mae_24-48h") is not None:
        print(f"  MAE 24-48h:   {best_row['mae_24-48h']:.1f} W")


if __name__ == "__main__":
    main()
