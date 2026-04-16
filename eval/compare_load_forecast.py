#!/usr/bin/env python3
"""
Compare TFT load forecast vs existing LightGBM load forecast.

TFT: offline inference on the pre-built val set (data/parquet/split_indices_load.npz).
LightGBM: production forecasts from load_forecast_log.csv filtered to the same
          calendar window as the TFT val set.

Metrics:
  - q50 MAE by horizon bucket (0-24h, 24-48h, 48-72h)
  - Overall q50 MAE
  - TFT q10/q90 interval coverage (fraction of actuals within [q10, q90])

Output: eval/results/load_forecast_comparison.json
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT   = Path(__file__).resolve().parent.parent
PARQUET = ROOT / "data" / "parquet"
RESULTS = ROOT / "eval" / "results"

CHECKPOINT  = ROOT / "models" / "tft_load" / "checkpoint_best.pt"
SCALERS     = PARQUET / "load_scalers.pkl"
LOG_FILE    = ROOT / "load_forecast_log.csv"

BUCKET_EDGES = [0, 48, 96, 144]   # in 30-min steps
BUCKET_NAMES = ["0-24h", "24-48h", "48-72h"]


# ── helpers ──────────────────────────────────────────────────────────────────

def mae_by_bucket(preds_W, actuals_W, mask):
    """
    preds_W:   [N, 144]  float32  predicted Watts
    actuals_W: [N, 144]  float32  actual Watts
    mask:      [N, 144]  bool     True where actual is valid
    Returns dict {bucket_name: mae_W}
    """
    results = {}
    for name, lo, hi in zip(BUCKET_NAMES, BUCKET_EDGES, BUCKET_EDGES[1:]):
        m = mask[:, lo:hi]
        err = np.abs(preds_W[:, lo:hi] - actuals_W[:, lo:hi])
        valid = err[m]
        results[name] = float(valid.mean()) if len(valid) > 0 else float("nan")
    # overall
    err_all = np.abs(preds_W - actuals_W)
    results["overall"] = float(err_all[mask].mean())
    return results


def coverage_by_bucket(lo_W, hi_W, actuals_W, mask):
    """Fraction of valid actual steps that fall within [q10, q90]."""
    results = {}
    inside = (actuals_W >= lo_W) & (actuals_W <= hi_W)
    for name, lo, hi in zip(BUCKET_NAMES, BUCKET_EDGES, BUCKET_EDGES[1:]):
        m = mask[:, lo:hi]
        results[name] = float(inside[:, lo:hi][m].mean()) if m.any() else float("nan")
    results["overall"] = float(inside[mask].mean())
    return results


# ── TFT evaluation ────────────────────────────────────────────────────────────

def evaluate_tft():
    print("── TFT evaluation ──────────────────────────────────────────")

    sys.path.insert(0, str(ROOT / "train"))
    from train_tft_load import TFTLoadModel

    ckpt   = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    m_cfg  = ckpt["model_config"]
    model  = TFTLoadModel(
        n_enc=m_cfg["n_enc"], n_dec=m_cfg["n_dec"],
        n_quantiles=m_cfg.get("n_quantiles", 3),
        d_model=m_cfg.get("d_model", 64),
        n_heads=m_cfg.get("n_heads", 4),
        n_lstm_layers=m_cfg.get("n_lstm_layers", 2),
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(SCALERS, "rb") as f:
        scalers = pickle.load(f)
    load_sc = scalers["power_load"]

    X_enc    = np.load(PARQUET / "X_enc_load.npy")
    X_dec    = np.load(PARQUET / "X_dec_load.npy")
    y_raw    = np.load(PARQUET / "y_load_raw.npy")
    y_mask   = np.load(PARQUET / "y_load_mask.npy")
    run_times = np.load(PARQUET / "run_times_load.npy")
    split    = np.load(PARQUET / "split_indices_load.npz")
    val_idx  = split["val"]

    X_enc_v   = X_enc[val_idx]
    X_dec_v   = X_dec[val_idx]
    y_raw_v   = y_raw[val_idx]
    y_mask_v  = y_mask[val_idx]
    run_times_v = run_times[val_idx]

    print(f"  Val samples: {len(val_idx):,}")
    print(f"  Window: {run_times_v.min()} → {run_times_v.max()}")

    # Batch inference
    BATCH = 256
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(val_idx), BATCH):
            t_enc  = torch.tensor(X_enc_v[i:i+BATCH])
            t_dec  = torch.tensor(X_dec_v[i:i+BATCH])
            preds  = model(t_enc, t_dec)                         # [B, 144, 3]
            preds  = torch.sort(preds, dim=-1).values            # monotonicity
            all_preds.append(preds.numpy())
            if (i // BATCH) % 2 == 0:
                print(f"  {min(i + BATCH, len(val_idx)):,}/{len(val_idx):,} samples inferred...", end="\r")
    print()

    preds_norm = np.concatenate(all_preds, axis=0)               # [N, 144, 3]
    preds_W    = load_sc.inverse_transform(
        preds_norm.reshape(-1, 1)
    ).reshape(preds_norm.shape).clip(min=0.0)                    # [N, 144, 3]

    q10_W = preds_W[:, :, 0]
    q50_W = preds_W[:, :, 1]
    q90_W = preds_W[:, :, 2]

    mae   = mae_by_bucket(q50_W, y_raw_v, y_mask_v)
    cov   = coverage_by_bucket(q10_W, q90_W, y_raw_v, y_mask_v)

    print(f"  q50 MAE — 0-24h: {mae['0-24h']:.1f}W  24-48h: {mae['24-48h']:.1f}W  "
          f"48-72h: {mae['48-72h']:.1f}W  overall: {mae['overall']:.1f}W")
    print(f"  q10/q90 coverage — 0-24h: {cov['0-24h']:.3f}  24-48h: {cov['24-48h']:.3f}  "
          f"48-72h: {cov['48-72h']:.3f}  overall: {cov['overall']:.3f}")

    return {
        "n_samples": int(len(val_idx)),
        "val_start": str(pd.Timestamp(run_times_v.min()).isoformat()),
        "val_end":   str(pd.Timestamp(run_times_v.max()).isoformat()),
        "checkpoint": str(CHECKPOINT),
        "q50_mae":  mae,
        "q10_q90_coverage": cov,
    }


# ── LightGBM evaluation ───────────────────────────────────────────────────────

def evaluate_lgbm(val_start: pd.Timestamp, val_end: pd.Timestamp):
    print("── LightGBM evaluation ─────────────────────────────────────")

    if not LOG_FILE.exists():
        print(f"  ERROR: {LOG_FILE} not found")
        return None

    print(f"  Loading {LOG_FILE.name}...")
    df = pd.read_csv(
        LOG_FILE, low_memory=False,
        usecols=["forecast_target_time", "forecast_creation_time",
                 "model_name", "prediction", "actual"],
    )
    df = df[df["model_name"] == "load"].copy()
    df["forecast_target_time"]   = pd.to_datetime(df["forecast_target_time"],   utc=True, format="mixed")
    df["forecast_creation_time"] = pd.to_datetime(df["forecast_creation_time"], utc=True, format="mixed")

    # Filter to val calendar window
    df = df[(df["forecast_creation_time"] >= val_start) &
            (df["forecast_creation_time"] <= val_end)]
    print(f"  Rows in val window: {len(df):,}")

    if df.empty:
        print("  No LightGBM rows found in val window")
        return None

    # Drop rows with missing actuals
    df = df.dropna(subset=["actual", "prediction"])

    # For each forecast run, compute step index from the run's earliest target time
    df = df.sort_values(["forecast_creation_time", "forecast_target_time"])
    df["run_start"] = df.groupby("forecast_creation_time")["forecast_target_time"].transform("min")
    df["step"] = ((df["forecast_target_time"] - df["run_start"]).dt.total_seconds() / 1800).round().astype(int)

    # Keep only steps 0–143 (72h horizon)
    df = df[(df["step"] >= 0) & (df["step"] < 144)]

    n_runs = df["forecast_creation_time"].nunique()
    print(f"  Unique forecast runs: {n_runs:,}")

    # MAE by bucket
    df["mae"] = (df["prediction"] - df["actual"]).abs()
    df["bucket"] = pd.cut(
        df["step"],
        bins=BUCKET_EDGES + [999],
        labels=BUCKET_NAMES + ["72h+"],
        right=False,
    )

    mae_rows = (
        df[df["bucket"].isin(BUCKET_NAMES)]
        .groupby("bucket", observed=True)["mae"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mae_W", "count": "n"})
    )
    overall_mae = float(df[df["bucket"].isin(BUCKET_NAMES)]["mae"].mean())

    mae_dict = {row: float(mae_rows.loc[row, "mae_W"]) for row in BUCKET_NAMES if row in mae_rows.index}
    mae_dict["overall"] = overall_mae

    print(f"  q50 MAE — 0-24h: {mae_dict.get('0-24h', float('nan')):.1f}W  "
          f"24-48h: {mae_dict.get('24-48h', float('nan')):.1f}W  "
          f"48-72h: {mae_dict.get('48-72h', float('nan')):.1f}W  "
          f"overall: {mae_dict['overall']:.1f}W")
    print("  (LightGBM is point forecast only — no quantile coverage available)")

    return {
        "n_runs": n_runs,
        "n_rows": int(len(df)),
        "val_start": val_start.isoformat(),
        "val_end":   val_end.isoformat(),
        "source": str(LOG_FILE.name),
        "q50_mae": mae_dict,
        "q10_q90_coverage": None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Load Forecast Comparison: TFT vs LightGBM")
    print("=" * 60)
    print()

    tft = evaluate_tft()
    print()

    val_start = pd.Timestamp(tft["val_start"], tz="UTC")
    val_end   = pd.Timestamp(tft["val_end"],   tz="UTC") + pd.Timedelta(hours=72)
    lgbm = evaluate_lgbm(val_start, val_end)
    print()

    # Summary table
    print("=" * 60)
    print("Summary — q50 MAE (Watts)")
    print(f"  {'Bucket':<12}  {'TFT':>10}  {'LightGBM':>10}  {'Δ (TFT-LGBM)':>14}")
    print("  " + "-" * 52)
    for b in BUCKET_NAMES + ["overall"]:
        t_val = tft["q50_mae"].get(b, float("nan"))
        l_val = lgbm["q50_mae"].get(b, float("nan")) if lgbm else float("nan")
        delta = t_val - l_val
        print(f"  {b:<12}  {t_val:>10.1f}  {l_val:>10.1f}  {delta:>+14.1f}")
    print()
    print("TFT q10/q90 coverage (target ≈ 0.80):")
    for b in BUCKET_NAMES + ["overall"]:
        c = tft["q10_q90_coverage"].get(b, float("nan"))
        print(f"  {b:<12}  {c:.3f}")

    # Save
    RESULTS.mkdir(exist_ok=True)
    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "tft":  tft,
        "lgbm": lgbm,
    }
    out_path = RESULTS / "load_forecast_comparison.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
