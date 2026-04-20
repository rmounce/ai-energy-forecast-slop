#!/usr/bin/env python3
"""
Rolling MPC eval — Track A / execution-focused backtest.

This script backtests a 14h × 5-min rolling MPC controller over a contiguous period,
carrying battery SoC forward between steps. It is designed as the first rolling eval
track: focused on the execution-relevant near horizon, with dense historical coverage.

Current scope (v1):
  - Price-only dispatch objective
  - 5-minute stepping
  - Current interval price treated as known
  - Tier 1 tactical q50 for the first hour
  - Tier 2 TFT q50 repeated from 30-min steps into 5-min slots for the remaining horizon
  - Optional Amber APF + LGBM baseline when forecast log coverage exists

This is intentionally narrower than the final full Phase 7 planning-track backtest.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytz
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "train"))

from dispatch_simulator import (  # noqa: E402
    CAPACITY_KWH,
    DEG_PER_KWH,
    EFF_C,
    EFF_D,
    INTERVAL_H,
    MAX_POWER_KW,
    SOC_INIT_KWH,
    lp_dispatch,
)
from retro_tft_inference import (  # noqa: E402
    build_5m_features,
    build_window_tensors,
    load_decoder_sources,
)
from retro_tier1_inference import build_features as build_tier1_features  # noqa: E402
from train_tft_price import TFTPriceModel  # noqa: E402

PARQUET_DIR = ROOT / "data" / "parquet"
RESULTS_DIR = ROOT / "eval" / "results"
PRICE_FORECAST_LOG = ROOT / "price_forecast_log.csv"
OOF_FILE = PARQUET_DIR / "debiased_pd_rrp_oof.parquet"

ACTUALS_5M_FILE = PARQUET_DIR / "actuals_sa1_5m.parquet"
ACTUALS_30M_FILE = PARQUET_DIR / "actuals_sa1.parquet"
P5MIN_FILE = PARQUET_DIR / "aemo_p5min_sa1.parquet"
TACTICAL_MODEL_DIR = ROOT / "models" / "lgbm_tactical"

HORIZON_5M_STEPS = 14 * 12  # 14h × 12 steps/hour = 168
TACTICAL_STEPS = 12         # 0–60 min at 5-min resolution


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


def _load_actuals_5m() -> pd.DataFrame:
    df = pd.read_parquet(ACTUALS_5M_FILE)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _load_actuals_30m() -> pd.DataFrame:
    df = pd.read_parquet(ACTUALS_30M_FILE)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def _load_pv_5m_from_30m(actuals_30m: pd.DataFrame) -> pd.Series:
    if "power_pv" not in actuals_30m.columns:
        return pd.Series(dtype=float)
    pv = actuals_30m["power_pv"].copy()
    pv_5m = pv.resample("5min").ffill()
    pv_5m.name = "power_pv"
    return pv_5m


def _load_p5min_runs() -> tuple[dict[pd.Timestamp, list[float]], list[pd.Timestamp], np.ndarray]:
    df = pd.read_parquet(P5MIN_FILE, columns=["run_time", "interval_dt", "rrp"])
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    df = df.sort_values(["run_time", "interval_dt"])

    runs: dict[pd.Timestamp, list[float]] = {}
    for rt, grp in df.groupby("run_time"):
        if len(grp) >= TACTICAL_STEPS:
            runs[rt] = grp["rrp"].values[:TACTICAL_STEPS].tolist()

    run_times_sorted = sorted(runs.keys())
    run_times_ns = np.array([rt.value for rt in run_times_sorted], dtype=np.int64)
    return runs, run_times_sorted, run_times_ns


def _load_amber_log_30m() -> tuple[dict[pd.Timestamp, pd.Series], list[pd.Timestamp], np.ndarray]:
    if not PRICE_FORECAST_LOG.exists():
        return {}, [], np.array([], dtype=np.int64)

    df = pd.read_csv(
        PRICE_FORECAST_LOG,
        usecols=["forecast_creation_time", "forecast_target_time", "prediction"],
        dtype_backend="pyarrow",
    )
    df["forecast_creation_time"] = pd.to_datetime(df["forecast_creation_time"], utc=True, format="mixed")
    df["forecast_target_time"] = pd.to_datetime(df["forecast_target_time"], utc=True, format="mixed")

    grouped: dict[pd.Timestamp, pd.Series] = {}
    for creation_time, grp in df.groupby("forecast_creation_time"):
        grp = grp.sort_values("forecast_target_time")
        s = pd.Series(
            grp["prediction"].values.astype(np.float64) * 1000.0,
            index=grp["forecast_target_time"],
        )
        grouped[creation_time] = s

    creation_sorted = sorted(grouped.keys())
    creation_ns = np.array([ts.value for ts in creation_sorted], dtype=np.int64)
    return grouped, creation_sorted, creation_ns


@dataclass
class TFTContext:
    model: TFTPriceModel
    scalers: dict
    log_scale: float
    s_targ: object
    dec_feature_names: list[str] | None
    bulk: dict
    feats_5m: pd.DataFrame
    decoder_sources: dict
    oof_by_run: dict | None
    q50_idx: int


def _load_tft_context(ckpt_path: Path, scalers_path: Path) -> TFTContext:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("meta", {})
    m_cfg = ckpt.get("model_config", {})
    quantiles = ckpt.get("quantiles", meta.get("quantiles", [0.05, 0.10, 0.50, 0.90, 0.95, 0.99]))
    q50_idx = quantiles.index(0.5) if 0.5 in quantiles else 2

    model = TFTPriceModel(
        n_enc=m_cfg["n_enc"],
        n_dec=m_cfg["n_dec"],
        d_model=m_cfg["d_model"],
        n_heads=m_cfg["n_heads"],
        n_lstm_layers=m_cfg.get("n_lstm_layers", m_cfg.get("n_layers", 2)),
        dropout=m_cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    actuals_30m = _load_actuals_30m()
    actuals_5m = _load_actuals_5m()
    bulk = {
        "enc": {
            "rrp": actuals_30m["rrp"],
            "total_demand": actuals_30m["total_demand"],
            "net_interchange": actuals_30m["net_interchange"],
            "power_load": actuals_30m["power_load"],
            "power_pv": actuals_30m["power_pv"],
            "temp": actuals_30m["temp"],
            "humidity": actuals_30m["humidity"],
            "wind_speed": actuals_30m["wind_speed"],
        },
        "prices_5m": actuals_5m["rrp"],
    }
    feats_5m = build_5m_features(actuals_5m["rrp"])
    decoder_sources = load_decoder_sources()

    oof_by_run = None
    if OOF_FILE.exists():
        oof_df = pd.read_parquet(OOF_FILE)
        oof_df["run_time"] = pd.to_datetime(oof_df["run_time"], utc=True).dt.as_unit("ns")
        oof_df["interval_dt"] = pd.to_datetime(oof_df["interval_dt"], utc=True).dt.as_unit("ns")
        oof_by_run = {
            run_t: grp.set_index("interval_dt")["oof_debiased_rrp"]
            for run_t, grp in oof_df.groupby("run_time")
        }

    return TFTContext(
        model=model,
        scalers=scalers,
        log_scale=meta.get("log_scale_factor", 60.0),
        s_targ=scalers.get("target_rrp", "log"),
        dec_feature_names=meta.get("dec_features") or meta.get("dec_feature_names"),
        bulk=bulk,
        feats_5m=feats_5m,
        decoder_sources=decoder_sources,
        oof_by_run=oof_by_run,
        q50_idx=q50_idx,
    )


def _infer_tft_q50_30m(start_ts: pd.Timestamp, ctx: TFTContext) -> np.ndarray | None:
    tensors = build_window_tensors(
        start_ts=start_ts,
        bulk=ctx.bulk,
        decoder_sources=ctx.decoder_sources,
        feats_5m=ctx.feats_5m,
        scalers=ctx.scalers,
        log_scale=ctx.log_scale,
        dec_feature_names=ctx.dec_feature_names,
        oof_by_run=ctx.oof_by_run,
    )
    if tensors is None:
        return None
    X_enc, X_dec = tensors
    with torch.no_grad():
        preds_norm = ctx.model(
            torch.tensor(X_enc).unsqueeze(0),
            torch.tensor(X_dec).unsqueeze(0),
        ).numpy()[0]
    preds_norm = np.sort(preds_norm, axis=-1)
    if ctx.s_targ == "log":
        preds_raw = np.sign(preds_norm) * ctx.log_scale * (np.exp(np.abs(preds_norm)) - 1.0)
    else:
        _, q = preds_norm.shape
        preds_raw = ctx.s_targ.inverse_transform(preds_norm.reshape(-1, q)).reshape(preds_norm.shape)
    return preds_raw[:, ctx.q50_idx].astype(np.float64)


class ForecastProviders:
    def __init__(self, args):
        self.actuals_5m = _load_actuals_5m()
        self.actuals_30m = _load_actuals_30m()
        self.pv_5m = _load_pv_5m_from_30m(self.actuals_30m)
        self.p5min_runs, self.p5_run_times_sorted, self.p5_run_times_ns = _load_p5min_runs()
        self.q50_model = joblib.load(TACTICAL_MODEL_DIR / "lgbm_q50.pkl")

        self.amber_runs, self.amber_creation_sorted, self.amber_creation_ns = _load_amber_log_30m()

        self.tft_ctx = None
        if args.tft_checkpoint and args.tft_scalers:
            self.tft_ctx = _load_tft_context(Path(args.tft_checkpoint), Path(args.tft_scalers))

        self._tier1_cache: dict[pd.Timestamp, pd.Series] = {}
        self._tft_cache: dict[pd.Timestamp, pd.Series] = {}
        self._amber_cache: dict[pd.Timestamp, pd.Series] = {}

    def current_actual_price(self, ts: pd.Timestamp) -> float:
        try:
            return float(self.actuals_5m["rrp"].asof(ts))
        except Exception:
            return float("nan")

    def tier1_q50(self, ts: pd.Timestamp) -> pd.Series | None:
        pos = int(np.searchsorted(self.p5_run_times_ns, ts.value, side="right"))
        if pos == 0:
            return None
        run_time = self.p5_run_times_sorted[pos - 1]
        if run_time in self._tier1_cache:
            return self._tier1_cache[run_time]

        p5min_rrp = self.p5min_runs[run_time]
        prev_rt = run_time - pd.Timedelta(minutes=5)
        prev_p5min_h0 = self.p5min_runs[prev_rt][0] if prev_rt in self.p5min_runs else float("nan")
        feats = build_tier1_features(run_time, p5min_rrp, prev_p5min_h0, self.actuals_5m, self.pv_5m)
        X_long = np.column_stack([
            np.tile(feats, (TACTICAL_STEPS, 1)),
            np.arange(TACTICAL_STEPS, dtype=np.float32).reshape(-1, 1),
        ])
        q50_raw = self.q50_model.predict(X_long).astype(np.float64)
        idx = pd.date_range(start=run_time, periods=TACTICAL_STEPS, freq="5min", tz="UTC")
        series = pd.Series(q50_raw, index=idx)
        self._tier1_cache[run_time] = series
        return series

    def tft_q50_expanded(self, ts: pd.Timestamp) -> pd.Series | None:
        if self.tft_ctx is None:
            return None
        anchor = ts.floor("30min")
        if anchor in self._tft_cache:
            return self._tft_cache[anchor]
        q50_30m = _infer_tft_q50_30m(anchor, self.tft_ctx)
        if q50_30m is None:
            return None
        idx_30m = pd.date_range(start=anchor, periods=len(q50_30m), freq="30min", tz="UTC")
        idx_5m = pd.date_range(start=anchor, periods=len(q50_30m) * 6, freq="5min", tz="UTC")
        series = pd.Series(np.repeat(q50_30m, 6), index=idx_5m)
        self._tft_cache[anchor] = series
        return series

    def amber_expanded(self, ts: pd.Timestamp) -> pd.Series | None:
        if len(self.amber_creation_ns) == 0:
            return None
        pos = int(np.searchsorted(self.amber_creation_ns, ts.value, side="right"))
        if pos == 0:
            return None
        creation = self.amber_creation_sorted[pos - 1]
        if creation in self._amber_cache:
            return self._amber_cache[creation]
        s30 = self.amber_runs[creation]
        s30 = s30[~s30.index.duplicated(keep="first")]
        idx5 = pd.date_range(start=s30.index.min(), end=s30.index.max() + pd.Timedelta(minutes=25), freq="5min", tz="UTC")
        expanded = pd.Series(index=idx5, dtype=np.float64)
        for t30, val in s30.items():
            block = pd.date_range(start=t30, periods=6, freq="5min", tz="UTC")
            expanded.loc[block] = float(val)
        expanded = expanded.ffill()
        self._amber_cache[creation] = expanded
        return expanded

    def build_forecast_curve(self, source: str, ts: pd.Timestamp) -> np.ndarray | None:
        idx = pd.date_range(start=ts, periods=HORIZON_5M_STEPS, freq="5min", tz="UTC")
        current_actual = self.current_actual_price(ts)
        if np.isnan(current_actual):
            return None

        if source == "p5min_naive":
            out = np.full(HORIZON_5M_STEPS, current_actual, dtype=np.float64)
            return out

        if source == "amber_apf_lgbm":
            amber = self.amber_expanded(ts)
            if amber is None:
                return None
            out = amber.reindex(idx).ffill().bfill().values.astype(np.float64)
            out[0] = current_actual
            return out

        if source == "model_a_hybrid":
            tier1 = self.tier1_q50(ts)
            tft = self.tft_q50_expanded(ts)
            if tier1 is None or tft is None:
                return None
            out = pd.Series(index=idx, dtype=np.float64)
            out.loc[idx[:TACTICAL_STEPS]] = tier1.reindex(idx[:TACTICAL_STEPS]).ffill().bfill().values
            later_idx = idx[TACTICAL_STEPS:]
            out.loc[later_idx] = tft.reindex(later_idx).ffill().bfill().values
            out.iloc[0] = current_actual
            return out.values.astype(np.float64)

        raise ValueError(f"Unknown source: {source}")


def simulate_stepwise(
    timestamps: pd.DatetimeIndex,
    actual_prices: pd.Series,
    providers: ForecastProviders,
    sources: list[str],
    soc_init: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = {src: float(soc_init) for src in sources}
    raw_rows = []
    summary = []

    pnl_totals = {src: 0.0 for src in sources}
    executed_steps = {src: 0 for src in sources}

    for i, ts in enumerate(timestamps):
        actual = float(actual_prices.asof(ts))
        if np.isnan(actual):
            continue

        for src in sources:
            curve = providers.build_forecast_curve(src, ts)
            if curve is None:
                continue
            c_plan, d_plan = lp_dispatch(curve, state[src])
            c0 = float(c_plan[0]) if len(c_plan) else 0.0
            d0 = float(d_plan[0]) if len(d_plan) else 0.0
            p_kwh = actual / 1000.0
            pnl = (d0 * EFF_D * p_kwh - c0 * p_kwh - DEG_PER_KWH * (c0 * EFF_C + d0)) * INTERVAL_H
            state[src] = float(np.clip(state[src] + (c0 * EFF_C - d0) * INTERVAL_H, 0.0, CAPACITY_KWH))
            pnl_totals[src] += pnl
            executed_steps[src] += 1
            raw_rows.append({
                "time": ts,
                "source": src,
                "actual_price_mwh": actual,
                "forecast_step0_mwh": float(curve[0]),
                "charge_kw": c0,
                "discharge_kw": d0,
                "soc_kwh": state[src],
                "step_pnl": pnl,
            })

        if (i + 1) % 288 == 0:
            print(f"  {i+1}/{len(timestamps)} steps ({timestamps[i].date()})")

    n_days = max((timestamps.max() - timestamps.min()).total_seconds() / 86400.0, 1e-9)
    for src in sources:
        summary.append({
            "source": src,
            "steps": executed_steps[src],
            "total_pnl": pnl_totals[src],
            "mean_per_day": pnl_totals[src] / n_days,
            "soc_final_kwh": state[src],
        })

    return pd.DataFrame(raw_rows), pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC start timestamp, e.g. 2025-07-21T00:00:00Z")
    parser.add_argument("--end", required=True, help="UTC end timestamp (exclusive)")
    parser.add_argument(
        "--sources",
        default="p5min_naive,model_a_hybrid",
        help="Comma-separated sources: p5min_naive, model_a_hybrid, amber_apf_lgbm",
    )
    parser.add_argument("--tft-checkpoint", default="", help="Path to TFT checkpoint for model_a_hybrid")
    parser.add_argument("--tft-scalers", default="", help="Path to matching TFT scalers for model_a_hybrid")
    parser.add_argument("--soc-init-kwh", type=float, default=SOC_INIT_KWH)
    parser.add_argument("--output-prefix", default="rolling_mpc_eval_model_a")
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    if "model_a_hybrid" in sources and (not args.tft_checkpoint or not args.tft_scalers):
        parser.error("model_a_hybrid requires --tft-checkpoint and --tft-scalers")

    providers = ForecastProviders(args)
    actuals_5m = providers.actuals_5m
    idx = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
    actual_prices = actuals_5m["rrp"].reindex(idx)

    print(f"Rolling MPC eval — Model A track")
    print(f"  Window: {start} → {end}")
    print(f"  Steps:  {len(idx):,}")
    print(f"  Sources: {sources}")

    t0 = time.time()
    raw_df, summary_df = simulate_stepwise(idx, actual_prices, providers, sources, args.soc_init_kwh)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / f"{args.output_prefix}_raw.parquet"
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.csv"
    raw_df.to_parquet(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved raw → {raw_path.relative_to(ROOT)}")
    print(f"Saved summary → {summary_path.relative_to(ROOT)}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
