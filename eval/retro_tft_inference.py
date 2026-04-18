#!/usr/bin/env python3
"""
Retrospective TFT price inference for holistic eval.

Runs TFT Tier 2 batch inference for all eval windows in holistic_eval_index.parquet,
using historical InfluxDB data as encoder input and stored PREDISPATCH/SDO as
decoder covariates.

Output: eval/results/retro_tft_forecasts.pkl
  dict with keys:
    'forecasts': {UTC Timestamp -> np.ndarray shape (144, 6)} in $/MWh
    'quantiles': [0.05, 0.10, 0.50, 0.90, 0.95, 0.99]
    'q50_idx':   2  (index of q50 in the quantile axis)

Decoder simplifications vs production:
  - PREDISPATCH: uses stored last() per 30m bucket (no per-window publication-time filtering)
  - Weather decoder: uses historical actuals (not BOM forecasts) — slightly advantages TFT
  - pd_demand/pd_net_interchange: from PREDISPATCH measurement (same as production)

Usage:
    source .venv/bin/activate
    nice -n 19 python eval/retro_tft_inference.py [--batch-size N] [--overwrite]
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

RESULTS_DIR  = ROOT / "eval" / "results"
INDEX_FILE   = RESULTS_DIR / "holistic_eval_index.parquet"
OUT_FILE     = RESULTS_DIR / "retro_tft_forecasts.pkl"

WINDOW_STEPS = 144
ENC_STEPS    = 96
BATCH_SIZE   = 32


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


# ── InfluxDB bulk fetch ───────────────────────────────────────────────────────

def _q30(client, meas, field, start_iso, end_iso,
         tag_key=None, tag_val=None, agg="mean") -> pd.Series:
    tag_clause = f" AND \"{tag_key}\"='{tag_val}'" if tag_key else ""
    q = (
        f'SELECT {agg}("{field}") AS val'
        f' FROM "rp_30m"."{meas}"'
        f' WHERE time >= \'{start_iso}\' AND time < \'{end_iso}\'{tag_clause}'
        f' GROUP BY time(30m) fill(none)'
    )
    result = client.query(q)
    rows = list(result.get_points())
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")["val"].sort_index()


def _q5m(client, start_iso, end_iso) -> pd.Series:
    q = (
        f'SELECT "price" AS val'
        f' FROM "rp_5m"."aemo_dispatch_sa1_5m"'
        f' WHERE time >= \'{start_iso}\' AND time < \'{end_iso}\''
    )
    result = client.query(q)
    rows = list(result.get_points())
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")["val"].sort_index()


def fetch_bulk(client, start_iso: str, end_iso: str) -> dict:
    """
    Fetch all encoder + decoder data series in one InfluxDB session.
    Covers [start_iso, end_iso) — set start = eval_start - 48h, end = eval_end + 72h.
    """
    print(f"Bulk fetching {start_iso} → {end_iso} ...")
    t0 = time.time()

    def q(meas, field, tag_key=None, tag_val=None, agg="mean"):
        return _q30(client, meas, field, start_iso, end_iso,
                    tag_key=tag_key, tag_val=tag_val, agg=agg)

    # Encoder: 30m actuals
    print("  fetching encoder actuals (price/demand/load/PV/weather)...")
    rrp_raw = q("aemo_dispatch_sa1_30m", "price")
    enc = {
        "rrp":             rrp_raw * 1000.0,   # $/kWh → $/MWh
        "total_demand":    q("aemo_dispatch_sa1_30m",  "total_demand"),
        "net_interchange": q("aemo_dispatch_sa1_30m",  "net_interchange"),
        "power_load":      q("power_load_30m",         "mean_value") / 1000.0,  # W → kW
        "power_pv":        q("power_pv_30m",           "mean_value") / 1000.0,
        "temp":            q("temperature_adelaide",   "mean_value"),
        "humidity":        q("humidity_adelaide",      "mean_value"),
        "wind_speed":      q("wind_speed_adelaide",    "mean_value"),
    }
    # Subtract dump load — matches get_historical_data() in forecast.py
    dump = q("power_dump_load_30m", "mean_value") / 1000.0
    if not dump.empty:
        enc["power_load"] = (
            enc["power_load"].sub(dump, fill_value=0.0).clip(lower=0)
        )

    # Encoder: 5m prices for rrp_5m_* features
    print("  fetching 5m prices...")
    prices_5m = _q5m(client, start_iso, end_iso)

    # Decoder: PREDISPATCH + PD7Day (target-time indexed, last() per 30m bucket)
    print("  fetching PREDISPATCH (SA1/VIC1/NSW1) + PD7Day ...")
    pd_sa1  = q("aemo_predispatch_forecast", "rrp",
                tag_key="region", tag_val="SA1",  agg="last")
    pd_vic1 = q("aemo_predispatch_forecast", "rrp",
                tag_key="region", tag_val="VIC1", agg="last")
    pd_nsw1 = q("aemo_predispatch_forecast", "rrp",
                tag_key="region", tag_val="NSW1", agg="last")
    pd7     = q("aemo_pd7day_forecast",      "rrp",
                tag_key="region", tag_val="SA1",  agg="last")
    pd_dem  = q("aemo_predispatch_forecast", "total_demand",
                tag_key="region", tag_val="SA1",  agg="last")
    pd_ni   = q("aemo_predispatch_forecast", "net_interchange",
                tag_key="region", tag_val="SA1",  agg="last")

    # Decoder: SevenDayOutlook
    print("  fetching SevenDayOutlook...")
    sdo_dem = q("aemo_sevendayoutlook", "scheduled_demand",
                tag_key="region", tag_val="SA1", agg="last")
    sdo_ni  = q("aemo_sevendayoutlook", "net_interchange",
                tag_key="region", tag_val="SA1", agg="last")

    # pd_rrp: PREDISPATCH primary, PD7Day fills gaps beyond 28h
    pd_rrp = pd_sa1.combine_first(pd7)

    print(f"  done in {time.time() - t0:.1f}s")
    return {
        "enc":               enc,
        "prices_5m":         prices_5m,
        "pd_rrp":            pd_rrp,
        "vic1_pd_rrp":       pd_vic1,
        "nsw1_pd_rrp":       pd_nsw1,
        "pd_demand":         pd_dem,
        "pd_net_interchange": pd_ni,
        "sd_demand":         sdo_dem,
        "sd_net_interchange": sdo_ni,
    }


# ── Derived feature construction ──────────────────────────────────────────────

def build_5m_features(prices_5m: pd.Series) -> pd.DataFrame:
    """
    Pre-compute rrp_5m_* encoder features at 30m resolution from the bulk 5m price series.
    Uses a 30-min rolling window (ending at each 5m step), then resamples to 30m.
    Matches the per-step get_agg() logic in _execute_tft_prediction().
    """
    if prices_5m.empty:
        return pd.DataFrame()
    p5 = prices_5m.sort_index()
    max30  = p5.rolling("30min", min_periods=1).max()
    std30  = p5.rolling("30min", min_periods=2).std().fillna(0.0)
    pers30 = (p5 > 150).astype(float).rolling("30min", min_periods=1).sum()
    return pd.DataFrame({
        "rrp_5m_max":         max30.resample("30min").last(),
        "rrp_5m_std":         std30.resample("30min").last(),
        "rrp_persistence":    pers30.resample("30min").last(),
        "rrp_volatility_30m": std30.resample("30min").last(),  # same as std (matches forecast.py)
    })


def time_sin_cos(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features in Brisbane timezone — matches forecast.py time_sin_cos()."""
    t = idx.tz_convert("Australia/Brisbane")
    return pd.DataFrame({
        "hour_sin":  np.sin(2 * np.pi * t.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * t.hour / 24),
        "dow_sin":   np.sin(2 * np.pi * t.dayofweek / 7),
        "dow_cos":   np.cos(2 * np.pi * t.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * (t.month - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (t.month - 1) / 12),
    }, index=idx)


# ── Feature order must match checkpoint meta exactly ──────────────────────────

ENC_CONT = [
    "rrp", "total_demand", "net_interchange", "power_load", "power_pv",
    "temp", "humidity", "wind_speed",
    "rrp_5m_max", "rrp_5m_std", "rrp_persistence", "rrp_volatility_30m",
    "rrp_log_momentum",
]
TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
ENC_FEATURES = ENC_CONT + TIME_COLS + ["rrp_5m_missing"]   # 20 features

DEC_CONT = [
    "pd_rrp", "pd_demand", "pd_net_interchange",
    "vic1_pd_rrp", "nsw1_pd_rrp",
    "sd_demand", "sd_net_interchange",
]
DEC_FEATURES = DEC_CONT + TIME_COLS + ["horizon_norm", "covar_missing"]  # 15 features


def _transform(vals: np.ndarray, feat: str, scalers: dict, log_scale: float) -> np.ndarray:
    s = scalers.get(feat)
    if s is None:
        return vals
    if s == "log":
        return np.sign(vals) * np.log1p(np.abs(vals) / log_scale)
    return s.transform(vals.reshape(-1, 1)).flatten()


def build_window_tensors(
    start_ts: pd.Timestamp,
    bulk: dict,
    feats_5m: pd.DataFrame,
    scalers: dict,
    log_scale: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Slice enc (96 steps) and dec (144 steps) arrays for one window.
    Returns (X_enc [96, 20], X_dec [144, 15]) or None if data is insufficient.
    """
    enc_idx = pd.date_range(
        start_ts - pd.Timedelta(hours=ENC_STEPS * 0.5),
        periods=ENC_STEPS, freq="30min", tz="UTC",
    )
    dec_idx = pd.date_range(start_ts, periods=WINDOW_STEPS, freq="30min", tz="UTC")

    # ── Encoder ───────────────────────────────────────────────────────────────
    enc_cols = {}
    for feat, series in bulk["enc"].items():
        enc_cols[feat] = series.reindex(enc_idx).ffill().bfill().fillna(0.0).values

    # Require ≥50% non-NaN raw rrp before fill (check the raw series)
    rrp_series = bulk["enc"]["rrp"].reindex(enc_idx)
    if rrp_series.isna().sum() > ENC_STEPS * 0.5:
        return None

    # 5m-derived features
    if not feats_5m.empty:
        for col in ["rrp_5m_max", "rrp_5m_std", "rrp_persistence", "rrp_volatility_30m"]:
            enc_cols[col] = feats_5m[col].reindex(enc_idx).ffill().bfill().fillna(0.0).values
        enc_cols["rrp_5m_missing"] = np.zeros(ENC_STEPS, dtype=np.float32)
    else:
        for col in ["rrp_5m_max", "rrp_5m_std", "rrp_persistence", "rrp_volatility_30m"]:
            enc_cols[col] = np.zeros(ENC_STEPS)
        enc_cols["rrp_5m_missing"] = np.ones(ENC_STEPS, dtype=np.float32)

    # rrp_log_momentum: 4-step diff of log-rrp (matches add_tft_regime_features)
    rrp_raw = enc_cols["rrp"]
    log_rrp = np.sign(rrp_raw) * np.log1p(np.abs(rrp_raw) / log_scale)
    log_mom = np.zeros_like(log_rrp)
    log_mom[4:] = (log_rrp[4:] - log_rrp[:-4]) / 4.0
    enc_cols["rrp_log_momentum"] = log_mom

    # Scale encoder continuous features
    for feat in ENC_CONT:
        enc_cols[feat] = _transform(enc_cols[feat], feat, scalers, log_scale)

    # Time features
    tc = time_sin_cos(enc_idx)
    for col in TIME_COLS:
        enc_cols[col] = tc[col].values

    X_enc = np.stack([enc_cols[f] for f in ENC_FEATURES], axis=1).astype(np.float32)

    # ── Decoder ───────────────────────────────────────────────────────────────
    dec_cols = {}
    for feat in DEC_CONT:
        series = bulk.get(feat, pd.Series(dtype=float))
        dec_cols[feat] = series.reindex(dec_idx).ffill().bfill().fillna(0.0).values

    # Scale decoder continuous features
    for feat in DEC_CONT:
        dec_cols[feat] = _transform(dec_cols[feat], feat, scalers, log_scale)

    # Time + positional
    tc_dec = time_sin_cos(dec_idx)
    for col in TIME_COLS:
        dec_cols[col] = tc_dec[col].values
    dec_cols["horizon_norm"] = np.arange(WINDOW_STEPS, dtype=np.float32) / WINDOW_STEPS
    dec_cols["covar_missing"] = np.zeros(WINDOW_STEPS, dtype=np.float32)

    X_dec = np.stack([dec_cols[f] for f in DEC_FEATURES], axis=1).astype(np.float32)

    return X_enc, X_dec


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Windows per forward pass (default {BATCH_SIZE})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output file")
    args = parser.parse_args()

    if OUT_FILE.exists() and not args.overwrite:
        print(f"Output exists: {OUT_FILE.relative_to(ROOT)}")
        print("  Use --overwrite to regenerate.")
        return

    if not INDEX_FILE.exists():
        print("ERROR: eval index not found. Run: python eval/build_holistic_eval_set.py")
        sys.exit(1)

    # ── Load model + scalers ──────────────────────────────────────────────────
    print("Loading TFT model and scalers...")
    import torch
    from train_tft_price import TFTPriceModel

    config  = load_config()
    paths   = config["paths"]
    ckpt    = torch.load(ROOT / paths["tft_price_model"], map_location="cpu", weights_only=False)
    meta    = ckpt.get("meta", {})
    m_cfg   = ckpt.get("model_config", {})
    log_scale = meta.get("log_scale_factor", 60.0)
    quantiles = meta.get("quantiles", [0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    q50_idx   = quantiles.index(0.5) if 0.5 in quantiles else 2

    model = TFTPriceModel(
        n_enc=m_cfg["n_enc"], n_dec=m_cfg["n_dec"],
        d_model=m_cfg["d_model"], n_heads=m_cfg["n_heads"],
        n_lstm_layers=m_cfg.get("n_lstm_layers", m_cfg.get("n_layers", 2)),
        dropout=m_cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(ROOT / paths["tft_price_scalers"], "rb") as f:
        scalers = pickle.load(f)
    s_targ = scalers.get("target_rrp", "log")
    print(f"  Model: n_enc={m_cfg['n_enc']}, n_dec={m_cfg['n_dec']}, quantiles={quantiles}")

    # ── Load eval index ───────────────────────────────────────────────────────
    df_index = pd.read_parquet(INDEX_FILE)
    df_index["start_time"] = pd.to_datetime(df_index["start_time"], utc=True)
    print(f"Eval index: {len(df_index)} windows "
          f"({df_index['start_time'].min().date()} → {df_index['start_time'].max().date()})")

    # Date ranges for bulk fetch
    eval_start = df_index["start_time"].min()
    eval_end   = df_index["start_time"].max()
    fetch_start = eval_start - pd.Timedelta(hours=ENC_STEPS * 0.5)    # 48h before first enc
    fetch_end   = eval_end + pd.Timedelta(hours=WINDOW_STEPS * 0.5)   # 72h after last window start

    start_iso = fetch_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso   = fetch_end.strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Bulk InfluxDB fetch ───────────────────────────────────────────────────
    client = InfluxDBClient(**config["influxdb"])
    try:
        bulk = fetch_bulk(client, start_iso, end_iso)
    finally:
        client.close()

    print("Pre-computing 5m aggregate features...")
    feats_5m = build_5m_features(bulk["prices_5m"])
    have_5m = not feats_5m.empty
    print(f"  5m coverage: {len(bulk['prices_5m'])} points, "
          f"{'computed' if have_5m else 'MISSING — rrp_5m_missing=1 fallback'}")

    # ── Build tensors for all windows ─────────────────────────────────────────
    print(f"Building encoder/decoder tensors for {len(df_index)} windows...")
    t0 = time.time()
    valid: list[tuple] = []
    skipped = 0
    for row in df_index.itertuples():
        result = build_window_tensors(row.start_time, bulk, feats_5m, scalers, log_scale)
        if result is None:
            skipped += 1
        else:
            valid.append((row.start_time, result[0], result[1]))
    print(f"  {len(valid)} valid, {skipped} skipped ({time.time() - t0:.1f}s)")

    if not valid:
        print("ERROR: no valid windows built. Check InfluxDB data coverage.")
        sys.exit(1)

    # ── Batched TFT inference ─────────────────────────────────────────────────
    batch_size = args.batch_size
    n_batches  = (len(valid) + batch_size - 1) // batch_size
    print(f"Running batched inference: {len(valid)} windows, "
          f"batch_size={batch_size}, {n_batches} batches...")
    t0 = time.time()
    forecasts: dict[pd.Timestamp, np.ndarray] = {}

    for bi in range(n_batches):
        batch = valid[bi * batch_size : (bi + 1) * batch_size]
        ts_list     = [w[0] for w in batch]
        X_enc_batch = np.stack([w[1] for w in batch])   # (B, 96, 20)
        X_dec_batch = np.stack([w[2] for w in batch])   # (B, 144, 15)

        with torch.no_grad():
            preds_norm = model(
                torch.tensor(X_enc_batch),
                torch.tensor(X_dec_batch),
            ).numpy()                                    # (B, 144, 6)

        preds_norm = np.sort(preds_norm, axis=-1)

        # Inverse-scale target (log-scale for Run 011b)
        if s_targ == "log":
            preds_raw = np.sign(preds_norm) * log_scale * (np.exp(np.abs(preds_norm)) - 1.0)
        else:
            B, T, Q = preds_norm.shape
            preds_raw = s_targ.inverse_transform(
                preds_norm.reshape(-1, Q)
            ).reshape(B, T, Q)

        for i, ts in enumerate(ts_list):
            forecasts[ts] = preds_raw[i]                # (144, 6) $/MWh

        done = min((bi + 1) * batch_size, len(valid))
        if done % 100 == 0 or done == len(valid):
            elapsed = time.time() - t0
            eta = (len(valid) - done) / (done / elapsed) if done else 0
            print(f"  {done}/{len(valid)} windows  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"Inference done: {len(forecasts)} windows in {elapsed:.1f}s "
          f"({len(forecasts) / elapsed:.1f} wins/s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"forecasts": forecasts, "quantiles": quantiles, "q50_idx": q50_idx}
    with open(OUT_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved {len(forecasts)} forecasts → {OUT_FILE.relative_to(ROOT)}")
    print(f"Quantiles: {quantiles}  (q50 at index {q50_idx})")

    # Quick sanity check
    sample_ts = list(forecasts.keys())[0]
    sample    = forecasts[sample_ts]
    print(f"Sample [{sample_ts.isoformat()}]  "
          f"q50 step0={sample[0, q50_idx]:.1f} $/MWh  "
          f"q50 step143={sample[143, q50_idx]:.1f} $/MWh")


if __name__ == "__main__":
    main()
