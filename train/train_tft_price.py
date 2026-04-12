#!/usr/bin/env python3
"""
Train a TFT-style price forecast model for SA1 using AEMO PREDISPATCH + PD7Day data.

Architecture: LSTM encoder-decoder with cross-attention (simplified TFT).
  Encoder (96 steps = 2 days): actual historical RRP, demand, interchange,
                                local load/PV/weather, cyclic time encodings
  Decoder (144 steps = 72h):   PREDISPATCH forecasts (h=1–56) + PD7Day (h=57–144)
                                + cyclic time encodings + horizon_norm
  Output:                      3-quantile prediction [q10, q50, q90] in $/MWh

Masked loss: y_mask [N, 144] indicates which decoder steps have valid covariates
  AND actual targets. Loss is averaged only over valid steps.
  Steps 1–32  (16h): ~15K samples  (97.9% of PREDISPATCH runs)
  Steps 1–56  (28h): ~8.5K samples (47.9% of PREDISPATCH runs)
  Steps 57–144 (72h): ~1K samples  (PD7Day backfill 2026-02-09 → 2026-04-11, growing)

See data/build_training_dataset.py and docs/tft_price_forecast.md for full design rationale.

Input data: pre-built numpy arrays from data/build_training_dataset.py
  data/parquet/X_encoder.npy     [N, 96,  14]  — normalised encoder features
  data/parquet/X_decoder.npy     [N, 144, 10]  — normalised decoder features
  data/parquet/y_targets.npy     [N, 144]      — normalised target RRP
  data/parquet/y_targets_raw.npy [N, 144]      — raw target RRP (evaluation)
  data/parquet/y_mask.npy        [N, 144]      — bool: valid covariate + target
  data/parquet/split_indices.npz              — train/val split indices
  data/parquet/scalers.pkl                    — QuantileTransformer scalers

Output:
  models/tft_price/checkpoint_best.pt         — best model state dict
  models/tft_price/training_log.csv           — epoch-level metrics

Usage:
  python train/train_tft_price.py                            # train with defaults (100 epochs)
  python train/train_tft_price.py --epochs 100 --d-model 128
  python train/train_tft_price.py --temporal-halflife 90     # down-weight old samples (90d half-life)
  python train/train_tft_price.py --dry-run                  # 5 epochs, small batch, check shapes

Optimizer: AdamW (lr=2e-4, weight_decay=1e-4) + ReduceLROnPlateau(factor=0.5, patience=2).
Loss: horizon-weighted quantile loss — weight_h = exp(-h/tau), default tau=14 (7h).
  Short-horizon steps dominate gradients; long-horizon steps still contribute but less.
Early stopping: monitors wMAPE (horizon-weighted nMAPE, same weights as loss);
  falls back to val_loss when NaN.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
PARQUET_DIR = ROOT / "data" / "parquet"
MODELS_DIR = ROOT / "models" / "tft_price"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.1, 0.5, 0.9]


# ─── Dataset ─────────────────────────────────────────────────────────────────

class PREDISPATCHDataset(Dataset):
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
    """GRN from Lim et al. 2021 (simplified, without context vector)."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        g = self.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class TFTPriceModel(nn.Module):
    """
    Lightweight TFT-style encoder-decoder.
    Encoder: GRN + LSTM over 96 historical steps
    Decoder: GRN + LSTM (init from encoder state) + cross-attention
    Output:  3-quantile projection [q10, q50, q90]
    """
    def __init__(self, n_enc: int, n_dec: int, d_model: int = 64,
                 n_heads: int = 4, n_lstm_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.enc_proj = nn.Sequential(
            nn.Linear(n_enc, d_model),
            GatedResidualNetwork(d_model, dropout),
        )
        self.dec_proj = nn.Sequential(
            nn.Linear(n_dec, d_model),
            GatedResidualNetwork(d_model, dropout),
        )
        self.encoder_lstm = nn.LSTM(d_model, d_model, n_lstm_layers,
                                    batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0.0)
        self.decoder_lstm = nn.LSTM(d_model, d_model, n_lstm_layers,
                                    batch_first=True, dropout=dropout if n_lstm_layers > 1 else 0.0)
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                               dropout=dropout)
        self.attn_grn = GatedResidualNetwork(d_model, dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, len(QUANTILES))

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        """
        enc: [B, T_enc, n_enc]
        dec: [B, T_dec, n_dec]
        returns: [B, T_dec, n_quantiles]
        """
        enc_emb = self.enc_proj(enc)                          # [B, 96, d]
        dec_emb = self.dec_proj(dec)                          # [B, 32, d]

        enc_out, (h, c) = self.encoder_lstm(enc_emb)          # enc_out: [B, 96, d]
        dec_out, _ = self.decoder_lstm(dec_emb, (h, c))       # dec_out: [B, 32, d]

        # Cross-attention: decoder queries attend to encoder keys/values
        attn_out, _ = self.attention(dec_out, enc_out, enc_out)
        dec_out = self.output_norm(dec_out + attn_out)        # residual + norm
        dec_out = self.attn_grn(dec_out)

        return self.output_proj(dec_out)                      # [B, 32, 3]


# ─── Loss ────────────────────────────────────────────────────────────────────

class MaskedQuantileLoss(nn.Module):
    """
    Quantile loss averaged over valid (unmasked) decoder steps.
    Handles variable coverage across decoder horizon:
      - Steps 1–32 (~15K samples): all unmasked
      - Steps 33–56 (~8.5K samples): PREDISPATCH coverage
      - Steps 57–144 (~1K samples): PD7Day coverage, growing over time

    Optional horizon_weights [T]: per-step multipliers applied before averaging.
    Normalisation uses the sum of effective weights (mask * horizon_weight) so
    the loss scale stays stable regardless of the weight profile.
    Use exponential decay (exp(-step / tau)) to upweight short-horizon steps.
    """
    def __init__(self, quantiles: list = QUANTILES,
                 horizon_weights: torch.Tensor = None):
        super().__init__()
        self.quantiles = quantiles
        if horizon_weights is not None:
            self.register_buffer("horizon_weights", horizon_weights)
        else:
            self.horizon_weights = None

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        preds:   [B, T, Q]
        targets: [B, T]
        mask:    [B, T] bool — 1 where loss should be computed
        """
        loss = torch.tensor(0.0, device=preds.device)

        # Effective per-step weight: horizon_weight * mask (zero where masked)
        if self.horizon_weights is not None:
            eff_w = self.horizon_weights.unsqueeze(0) * mask.float()  # [B, T]
        else:
            eff_w = mask.float()
        w_sum = eff_w.sum(dim=1).clamp(min=1e-8)  # [B]

        for i, q in enumerate(self.quantiles):
            e = targets - preds[:, :, i]
            step_loss = torch.max(q * e, (q - 1) * e)         # [B, T]
            seq_loss = (step_loss * eff_w).sum(dim=1) / w_sum
            loss = loss + seq_loss.mean()

        return loss / len(self.quantiles)


# ─── Training ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimiser=None):
    """One training or validation epoch. Returns mean masked loss."""
    training = optimiser is not None
    model.train(training)
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(training):
        for X_enc, X_dec, y, y_raw, mask in loader:
            preds = model(X_enc, X_dec)
            loss = criterion(preds, y, mask)

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches else float("nan")


def evaluate_nmape(model, loader, scalers, horizon_weights=None):
    """Compute nMAPE on raw (denormalised) predictions vs raw targets (valid steps only).

    Returns (nmape_global, buckets) where:
      nmape_global:        unweighted nMAPE over all valid steps (informative)
      buckets["weighted"]: horizon-weighted nMAPE — sum(w_h*|e_h|)/sum(w_h*|y_h|) across
                           valid steps using the same exp(-h/tau) weights as the training
                           loss. Primary early stopping metric when horizon_weights active.
                           Equals nmape_global when horizon_weights is None.
      buckets["1-4h/16h/28h/72h"]: unweighted per-bucket nMAPEs for monitoring.

    horizon_weights: optional numpy [T] array matching the training loss weights.
    """
    model.eval()
    qt = scalers["target_rrp"]

    all_pred_raw, all_targ_raw, all_masks = [], [], []

    with torch.no_grad():
        for X_enc, X_dec, y_norm, y_batch_raw, mask in loader:
            preds_norm = model(X_enc, X_dec)           # [B, T, 3]
            preds_norm, _ = torch.sort(preds_norm, dim=-1) # Prevent quantile crossing
            p50_norm   = preds_norm[:, :, 1].numpy()   # median quantile
            p50_raw = qt.inverse_transform(p50_norm.reshape(-1, 1)).reshape(p50_norm.shape)

            all_pred_raw.append(p50_raw)
            all_targ_raw.append(y_batch_raw.numpy())
            all_masks.append(mask.numpy())

    pred  = np.concatenate(all_pred_raw, axis=0)    # [N, T]
    targ  = np.concatenate(all_targ_raw, axis=0)
    masks = np.concatenate(all_masks,    axis=0)

    # Global unweighted nMAPE (all valid steps) — informative
    p_valid = pred[masks]
    t_valid = targ[masks]
    nmape_global = np.abs(p_valid - t_valid).sum() / np.abs(t_valid).sum() * 100

    # Horizon-weighted nMAPE: sum(w_h * |e_h|) / sum(w_h * |y_h|) across valid steps.
    # Consistent with training loss — short horizons dominate when tau is small.
    if horizon_weights is not None:
        hw = horizon_weights[np.newaxis, :]          # [1, T]
        w_eff   = hw * masks                         # [N, T]  zero where invalid
        w_numer = (w_eff * np.abs(pred - targ)).sum()
        w_denom = (w_eff * np.abs(targ)).sum()
        nmape_weighted = w_numer / w_denom * 100 if w_denom > 0 else float("nan")
    else:
        nmape_weighted = nmape_global

    # Unweighted per-bucket nMAPEs (monitoring only)
    def bucket_nmape(lo, hi):
        bm = masks[:, lo:hi]
        bp = pred[:, lo:hi][bm]
        bt = targ[:, lo:hi][bm]
        if len(bt) == 0:
            return float("nan")
        return np.abs(bp - bt).sum() / np.abs(bt).sum() * 100

    return nmape_global, {
        "weighted": nmape_weighted,
        "1-4h":     bucket_nmape(0,   8),
        "1-16h":    bucket_nmape(0,  32),
        "1-28h":    bucket_nmape(0,  56),
        "28-72h":   bucket_nmape(56, 144),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train TFT price model")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max training epochs (default 100; ReduceLROnPlateau + early stopping will terminate earlier)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=64,
                        help="LSTM/attention hidden size (64 for CPU; 128+ for GPU)")
    parser.add_argument("--n-heads", type=int, default=4,
                        help="Attention heads (must divide d-model)")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--horizon-decay", type=float, default=14.0,
                        help="Exponential decay tau for horizon loss weighting (steps). "
                             "weight_h = exp(-h/tau). Default 14 (7h half-weight at step 14). "
                             "Set to 0 to disable (uniform loss).")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay (default 1e-4)")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (default 7)")
    parser.add_argument("--temporal-halflife", type=float, default=0.0,
                        help="Temporal sample weighting half-life in days (0 = disabled). "
                             "Weights training samples by exp(-age_days / (halflife/ln2)). "
                             "90–120 days recommended for SA1 evolving market dynamics.")
    parser.add_argument("--dry-run", action="store_true",
                        help="5 epochs, batch_size=64, validate shapes only")
    args = parser.parse_args()

    if args.dry_run:
        args.epochs = 5
        args.batch_size = 64
        print("[dry-run] 5 epochs, batch_size=64")

    # ── Load data
    print("=== TFT Price Model Training ===")
    print("\nLoading pre-built arrays...")
    X_enc  = np.load(PARQUET_DIR / "X_encoder.npy")
    X_dec  = np.load(PARQUET_DIR / "X_decoder.npy")
    y      = np.load(PARQUET_DIR / "y_targets.npy")
    y_raw  = np.load(PARQUET_DIR / "y_targets_raw.npy")
    y_mask = np.load(PARQUET_DIR / "y_mask.npy")
    split  = np.load(PARQUET_DIR / "split_indices.npz")
    train_idx = split["train"]
    val_idx   = split["val"]

    import pickle
    with open(PARQUET_DIR / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    with open(PARQUET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    cov = meta.get("mask_coverage", {})
    print(f"  X_encoder: {X_enc.shape}  ({meta['input_length']} steps, {meta['n_enc_features']} enc features)")
    print(f"  X_decoder: {X_dec.shape}  ({meta['output_length']} steps, {meta['n_dec_features']} dec features)")
    print(f"  y_targets: {y.shape}")
    print(f"  y_mask:    {y_mask.shape}  valid steps: {y_mask.sum():,}")
    if cov:
        print(f"  Coverage:  1–16h {cov.get('steps_1_32', 0):.1%}, "
              f"1–28h {cov.get('steps_1_56', 0):.1%}, "
              f"28–72h {cov.get('steps_57_144', 0):.1%}")
    print(f"  Train/val: {len(train_idx):,} / {len(val_idx):,} samples")

    # ── DataLoaders
    train_ds = PREDISPATCHDataset(X_enc[train_idx], X_dec[train_idx],
                                  y[train_idx],     y_raw[train_idx], y_mask[train_idx])
    val_ds   = PREDISPATCHDataset(X_enc[val_idx],   X_dec[val_idx],
                                  y[val_idx],       y_raw[val_idx],   y_mask[val_idx])

    if args.temporal_halflife > 0:
        run_times = np.load(PARQUET_DIR / "run_times.npy", allow_pickle=True)
        train_times = run_times[train_idx].astype("datetime64[s]").astype(np.int64)
        age_days = (train_times.max() - train_times) / 86400.0
        decay_rate = np.log(2) / args.temporal_halflife
        sample_weights = np.exp(-decay_rate * age_days).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=0)
        print(f"  Temporal weighting: half-life={args.temporal_halflife:.0f}d  "
              f"oldest weight={sample_weights.min():.4f}  newest=1.0000")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_enc = X_enc.shape[2]
    n_dec = X_dec.shape[2]

    # ── Model
    model = TFTPriceModel(
        n_enc=n_enc, n_dec=n_dec,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_lstm_layers=args.n_layers,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: d_model={args.d_model}, heads={args.n_heads}, layers={args.n_layers}")
    print(f"  Parameters: {n_params:,}")

    if args.horizon_decay > 0:
        steps = torch.arange(meta["output_length"], dtype=torch.float32)
        horizon_weights = torch.exp(-steps / args.horizon_decay)
    else:
        horizon_weights = None
    criterion = MaskedQuantileLoss(QUANTILES, horizon_weights=horizon_weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2)

    # ── Training loop
    # Primary metric: nMAPE_28h (lower is better); falls back to val_loss when NaN
    best_target_metric = float("inf")
    patience_counter = 0
    log_rows = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...")
    print(f"  Optimizer: AdamW  lr={args.lr:.1e}  weight_decay={args.weight_decay:.1e}")
    print(f"  Scheduler: ReduceLROnPlateau  factor=0.5  patience=2")
    if args.horizon_decay > 0:
        w4h = float(torch.exp(torch.tensor(-8.0 / args.horizon_decay)))
        w16h = float(torch.exp(torch.tensor(-32.0 / args.horizon_decay)))
        w28h = float(torch.exp(torch.tensor(-56.0 / args.horizon_decay)))
        print(f"  Horizon weights: tau={args.horizon_decay:.0f}steps  "
              f"4h={w4h:.2f}  16h={w16h:.2f}  28h={w28h:.2f}")
    else:
        print(f"  Horizon weights: uniform (disabled)")
    print(f"  Early stopping on: wMAPE (horizon-weighted, fallback: val_loss when NaN)")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'wMAPE%':>8}  {'LR':>10}  {'Time':>6}")
    print("-" * 60)

    # numpy version of horizon_weights for evaluate_nmape
    hw_numpy = horizon_weights.numpy() if horizon_weights is not None else None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        prev_lr = optimiser.param_groups[0]["lr"]
        train_loss = run_epoch(model, train_loader, criterion, optimiser)
        val_loss   = run_epoch(model, val_loader,   criterion)
        nmape_global, buckets = evaluate_nmape(model, val_loader, scalers, hw_numpy)
        elapsed    = time.time() - t0

        # Primary metric: horizon-weighted nMAPE; fall back to val_loss if NaN
        nmape_weighted = buckets["weighted"]
        target_metric = nmape_weighted if not np.isnan(nmape_weighted) else val_loss

        scheduler.step(target_metric)
        lr_now = optimiser.param_groups[0]["lr"]

        bucket_str = (f"  unw={nmape_global:.1f}%"
                      f"  4h={buckets['1-4h']:.1f}%"
                      f"  16h={buckets['1-16h']:.1f}%"
                      f"  28h={buckets['1-28h']:.1f}%"
                      + (f"  72h={buckets['28-72h']:.1f}%" if not np.isnan(buckets['28-72h']) else ""))
        wmape_s = f"{nmape_weighted:.2f}" if not np.isnan(nmape_weighted) else "   nan"
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{wmape_s:>8}  {lr_now:>10.2e}  {elapsed:>5.1f}s"
              f"  [{bucket_str.strip()}]")

        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "nmape_weighted": round(nmape_weighted, 4) if not np.isnan(nmape_weighted) else None,
            "nmape_all":  round(nmape_global, 4),
            "nmape_4h":   round(buckets["1-4h"],   4),
            "nmape_16h":  round(buckets["1-16h"],  4),
            "nmape_28h":  round(buckets["1-28h"],  4),
            "nmape_72h":  round(buckets["28-72h"], 4) if not np.isnan(buckets["28-72h"]) else None,
            "lr": lr_now, "time_s": round(elapsed, 1),
        })

        # Save best
        if target_metric < best_target_metric:
            best_target_metric = target_metric
            patience_counter = 0
            ckpt = {
                "epoch": epoch,
                "val_loss": val_loss,
                "nmape_all": nmape_global,
                "nmape_weighted": nmape_weighted,
                "nmape_buckets": buckets,
                "primary_metric": target_metric,
                "model_state": model.state_dict(),
                "model_config": {
                    "n_enc": n_enc, "n_dec": n_dec,
                    "d_model": args.d_model, "n_heads": args.n_heads,
                    "n_lstm_layers": args.n_layers, "dropout": args.dropout,
                },
                "meta": meta,
                "quantiles": QUANTILES,
            }
            torch.save(ckpt, MODELS_DIR / "checkpoint_best.pt")
            # Also save a timestamped copy so previous bests are never overwritten
            import datetime as _dt
            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(ckpt, MODELS_DIR / f"checkpoint_{_ts}_ep{epoch}.pt")
            metric_label = (f"wMAPE={target_metric:.2f}%"
                            if not np.isnan(nmape_weighted)
                            else f"val_loss={target_metric:.4f}")
            print(f"         ↑ best {metric_label}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience} exhausted)")
                break

        if lr_now < prev_lr:
            print(f"         ↓ LR reduced: {prev_lr:.2e} → {lr_now:.2e}")

    # ── Save training log
    log_path = MODELS_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nTraining log: {log_path}")

    # ── Final summary — pick best epoch by wMAPE (NaN-safe), fallback to val_loss
    def _wmape_key(r):
        v = r["nmape_weighted"]
        return v if (v is not None and not np.isnan(v)) else float("inf")

    best_row = min(log_rows, key=_wmape_key)
    if _wmape_key(best_row) == float("inf"):  # all NaN — fallback
        best_row = min(log_rows, key=lambda r: r["val_loss"])
    print(f"\n=== Training complete ===")
    print(f"  Best epoch:    {best_row['epoch']}")
    print(f"  Best val loss: {best_row['val_loss']:.4f}")
    if best_row.get("nmape_weighted") is not None:
        print(f"  wMAPE (weighted): {best_row['nmape_weighted']:.2f}%")
    print(f"  nMAPE (all):   {best_row['nmape_all']:.2f}%")
    print(f"  nMAPE (4h):    {best_row['nmape_4h']:.2f}%")
    print(f"  nMAPE (16h):   {best_row['nmape_16h']:.2f}%")
    print(f"  nMAPE (28h):   {best_row['nmape_28h']:.2f}%")
    if best_row.get("nmape_72h") is not None:
        print(f"  nMAPE (72h):   {best_row['nmape_72h']:.2f}%")
    print(f"  Checkpoint: {MODELS_DIR}/checkpoint_best.pt")

    if not args.dry_run:
        print(f"\n  Next: run rolling-origin evaluation:")
        print(f"    python train/evaluate_tft.py --weeks 12")


if __name__ == "__main__":
    main()
