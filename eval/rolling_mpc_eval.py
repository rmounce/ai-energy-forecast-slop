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
import concurrent.futures
import json
import multiprocessing as mp
import os
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
    solve_lp_dispatch,
)
from retro_tft_inference import (  # noqa: E402
    build_5m_features,
    build_window_tensors,
    load_decoder_sources,
)
from retro_tier1_inference import build_features as build_tier1_features  # noqa: E402
from train_tft_price import TFTPriceModel  # noqa: E402
from data.build_tactical_dataset import FEATURE_NAMES as TACTICAL_FEATURE_NAMES  # noqa: E402

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
    # Some logged Amber targets carry stray seconds/microseconds; normalize them onto the
    # intended 30-minute grid before later 5-minute expansion.
    df["forecast_target_time"] = df["forecast_target_time"].dt.floor("30min")
    df = df.sort_values(["forecast_creation_time", "forecast_target_time"])
    df = df.drop_duplicates(
        subset=["forecast_creation_time", "forecast_target_time"],
        keep="last",
    )

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
        self._last_curve_repaired: dict[str, bool] = {}

    def curve_was_repaired(self, source: str) -> bool:
        return self._last_curve_repaired.get(source, False)

    def _finalize_curve(self, source: str, values: np.ndarray, current_actual: float) -> np.ndarray | None:
        out = np.asarray(values, dtype=np.float64).copy()
        self._last_curve_repaired[source] = False
        if len(out) == 0:
            return None
        out[0] = current_actual
        if out.shape[0] != HORIZON_5M_STEPS:
            return out
        if np.isfinite(out).all():
            return out
        repaired = pd.Series(out).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
        if np.isfinite(repaired).all():
            repaired[0] = current_actual
            self._last_curve_repaired[source] = True
            return repaired
        return repaired

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
        X_long_df = pd.DataFrame(
            X_long,
            columns=list(TACTICAL_FEATURE_NAMES) + ["horizon"],
        )
        q50_raw = self.q50_model.predict(X_long_df).astype(np.float64)
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
            return self._finalize_curve(source, out, current_actual)

        if source == "amber_apf_lgbm":
            amber = self.amber_expanded(ts)
            if amber is None:
                return None
            out = amber.reindex(idx).ffill().bfill().values.astype(np.float64)
            return self._finalize_curve(source, out, current_actual)

        if source == "model_a_hybrid":
            tier1 = self.tier1_q50(ts)
            tft = self.tft_q50_expanded(ts)
            if tier1 is None or tft is None:
                return None
            out = pd.Series(index=idx, dtype=np.float64)
            out.loc[idx[:TACTICAL_STEPS]] = tier1.reindex(idx[:TACTICAL_STEPS]).ffill().bfill().values
            later_idx = idx[TACTICAL_STEPS:]
            out.loc[later_idx] = tft.reindex(later_idx).ffill().bfill().values
            return self._finalize_curve(source, out.values.astype(np.float64), current_actual)

        raise ValueError(f"Unknown source: {source}")


def simulate_stepwise(
    timestamps: pd.DatetimeIndex,
    actual_prices: pd.Series,
    providers: ForecastProviders,
    sources: list[str],
    soc_init: float,
    terminal_energy_value_per_kwh: float = 0.0,
    dual_terminal_scale: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = {src: float(soc_init) for src in sources}
    raw_rows = []
    summary = []

    pnl_totals = {src: 0.0 for src in sources}
    executed_steps = {src: 0 for src in sources}
    skipped_missing_actual = {src: 0 for src in sources}
    skipped_missing_curve = {src: 0 for src in sources}
    skipped_invalid_curve = {src: 0 for src in sources}
    repaired_invalid_curve = {src: 0 for src in sources}
    invalid_curve_logged = {src: 0 for src in sources}

    for i, ts in enumerate(timestamps):
        actual = float(actual_prices.asof(ts))
        if np.isnan(actual):
            for src in sources:
                skipped_missing_actual[src] += 1
            continue

        for src in sources:
            curve = providers.build_forecast_curve(src, ts)
            if curve is None:
                skipped_missing_curve[src] += 1
                continue
            if providers.curve_was_repaired(src):
                repaired_invalid_curve[src] += 1
            curve = np.asarray(curve, dtype=np.float64)
            if curve.shape[0] != HORIZON_5M_STEPS or not np.isfinite(curve).all():
                skipped_invalid_curve[src] += 1
                if invalid_curve_logged[src] < 3:
                    bad_positions = np.where(~np.isfinite(curve))[0].tolist()[:8]
                    print(
                        f"  Skipping invalid curve for {src} at {ts}: "
                        f"len={len(curve)} bad_positions={bad_positions}"
                    )
                    invalid_curve_logged[src] += 1
                continue
            soc_prev = state[src]
            probe_shadow_price_per_kwh = float("nan")
            applied_terminal_energy_value_per_kwh = terminal_energy_value_per_kwh
            if dual_terminal_scale > 0.0:
                probe = solve_lp_dispatch(curve, state[src], terminal_energy_value_per_kwh=0.0)
                probe_shadow_price_per_kwh = probe["initial_soc_shadow_price_per_kwh"]
                if np.isfinite(probe_shadow_price_per_kwh):
                    applied_terminal_energy_value_per_kwh = max(
                        0.0,
                        dual_terminal_scale * probe_shadow_price_per_kwh,
                    )
                else:
                    applied_terminal_energy_value_per_kwh = 0.0
            solve = solve_lp_dispatch(
                curve,
                state[src],
                terminal_energy_value_per_kwh=applied_terminal_energy_value_per_kwh,
            )
            c_plan = solve["charge_kw"]
            d_plan = solve["discharge_kw"]
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
                "soc_prev_kwh": soc_prev,
                "soc_kwh": state[src],
                "step_pnl": pnl,
                "terminal_energy_value_per_kwh": applied_terminal_energy_value_per_kwh,
                "probe_initial_soc_shadow_price_per_kwh": probe_shadow_price_per_kwh,
                "control_initial_soc_shadow_price_per_kwh": solve["initial_soc_shadow_price_per_kwh"],
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
            "terminal_energy_value_per_kwh": terminal_energy_value_per_kwh,
            "dual_terminal_scale": dual_terminal_scale,
            "skipped_missing_actual": skipped_missing_actual[src],
            "skipped_missing_curve": skipped_missing_curve[src],
            "skipped_invalid_curve": skipped_invalid_curve[src],
            "repaired_invalid_curve": repaired_invalid_curve[src],
        })

    return pd.DataFrame(raw_rows), pd.DataFrame(summary)


def build_daily_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "date",
                "steps",
                "total_pnl",
                "mean_price_mwh",
                "mean_abs_dispatch_kw",
                "charge_energy_kwh",
                "discharge_energy_kwh",
                "avg_charge_price_mwh",
                "avg_discharge_price_mwh",
                "soc_open_kwh",
                "soc_close_kwh",
                "soc_delta_kwh",
            ]
        )

    daily = raw_df.copy()
    daily["date"] = pd.to_datetime(daily["time"], utc=True).dt.date
    daily["abs_dispatch_kw"] = daily["charge_kw"].abs() + daily["discharge_kw"].abs()
    daily["charge_energy_kwh"] = daily["charge_kw"] * INTERVAL_H
    daily["discharge_energy_kwh"] = daily["discharge_kw"] * INTERVAL_H
    daily["charge_price_weight"] = daily["actual_price_mwh"] * daily["charge_energy_kwh"]
    daily["discharge_price_weight"] = daily["actual_price_mwh"] * daily["discharge_energy_kwh"]

    out = (
        daily.groupby(["source", "date"], as_index=False)
        .agg(
            steps=("time", "size"),
            total_pnl=("step_pnl", "sum"),
            mean_price_mwh=("actual_price_mwh", "mean"),
            mean_abs_dispatch_kw=("abs_dispatch_kw", "mean"),
            charge_energy_kwh=("charge_energy_kwh", "sum"),
            discharge_energy_kwh=("discharge_energy_kwh", "sum"),
            charge_price_weight=("charge_price_weight", "sum"),
            discharge_price_weight=("discharge_price_weight", "sum"),
            soc_open_kwh=("soc_prev_kwh", "first"),
            soc_close_kwh=("soc_kwh", "last"),
        )
        .sort_values(["source", "date"], kind="stable")
        .reset_index(drop=True)
    )
    out["avg_charge_price_mwh"] = np.where(
        out["charge_energy_kwh"] > 0,
        out["charge_price_weight"] / out["charge_energy_kwh"],
        np.nan,
    )
    out["avg_discharge_price_mwh"] = np.where(
        out["discharge_energy_kwh"] > 0,
        out["discharge_price_weight"] / out["discharge_energy_kwh"],
        np.nan,
    )
    out["soc_delta_kwh"] = out["soc_close_kwh"] - out["soc_open_kwh"]
    return out.drop(columns=["charge_price_weight", "discharge_price_weight"])


def classify_price_regime(prices: pd.Series) -> str:
    if len(prices) == 0:
        return "unknown"
    if float(prices.max()) >= 300.0:
        return "spike"
    if float(prices.min()) <= -50.0:
        return "low"
    return "normal"


def classify_spike_band(max_price_mwh: float, regime: str) -> str:
    if regime != "spike":
        return regime
    if max_price_mwh >= 1000.0:
        return "spike_extreme"
    return "spike_moderate"


def build_daily_regime_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "regime",
                "max_price_mwh",
                "min_price_mwh",
                "mean_price_mwh",
                "steps",
            ]
        )

    prices = raw_df[["time", "actual_price_mwh"]].drop_duplicates().copy()
    prices["date"] = pd.to_datetime(prices["time"], utc=True).dt.date
    out = (
        prices.groupby("date", as_index=False)
        .agg(
            steps=("time", "size"),
            max_price_mwh=("actual_price_mwh", "max"),
            min_price_mwh=("actual_price_mwh", "min"),
            mean_price_mwh=("actual_price_mwh", "mean"),
        )
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    out["regime"] = out.apply(
        lambda row: classify_price_regime(pd.Series([row["min_price_mwh"], row["max_price_mwh"]])),
        axis=1,
    )
    out["spike_band"] = out.apply(
        lambda row: classify_spike_band(float(row["max_price_mwh"]), str(row["regime"])),
        axis=1,
    )
    return out[["date", "regime", "spike_band", "max_price_mwh", "min_price_mwh", "mean_price_mwh", "steps"]]


def add_daily_regime_to_summary(
    daily_summary_df: pd.DataFrame,
    daily_regime_df: pd.DataFrame,
) -> pd.DataFrame:
    if daily_summary_df.empty:
        return daily_summary_df.copy()
    regime_cols = ["date", "regime", "spike_band", "max_price_mwh", "min_price_mwh"]
    return daily_summary_df.merge(daily_regime_df[regime_cols], on="date", how="left")


def build_coverage_summary(
    raw_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    sources: list[str],
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    expected_steps = int(len(pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")))
    actual_steps = raw_df.groupby("source").size().reindex(sources, fill_value=0)
    coverage = pd.DataFrame(
        {
            "source": sources,
            "expected_steps": expected_steps,
            "executed_steps": actual_steps.values.astype(int),
        }
    )
    coverage["missing_steps"] = coverage["expected_steps"] - coverage["executed_steps"]
    coverage["coverage_ratio"] = np.where(
        coverage["expected_steps"] > 0,
        coverage["executed_steps"] / coverage["expected_steps"],
        0.0,
    )
    if summary_df is not None and not summary_df.empty:
        extra_cols = [
            "source",
            "skipped_missing_actual",
            "skipped_missing_curve",
            "skipped_invalid_curve",
            "repaired_invalid_curve",
        ]
        present_cols = [col for col in extra_cols if col in summary_df.columns]
        if "source" in present_cols and len(present_cols) > 1:
            coverage = coverage.merge(summary_df[present_cols], on="source", how="left")
    return coverage.sort_values("source", kind="stable").reset_index(drop=True)


def add_baseline_deltas(
    df: pd.DataFrame,
    baseline_source: str,
    value_cols: list[str],
    join_cols: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    join_cols = join_cols or []
    base = df[df["source"] == baseline_source].copy()
    if base.empty:
        raise ValueError(f"Baseline source '{baseline_source}' not present in dataframe")

    if not join_cols:
        if len(base) != 1:
            raise ValueError(
                f"Baseline source '{baseline_source}' must have exactly one row when join_cols is empty"
            )
        out = df.copy()
        for col in value_cols:
            out[f"{col}_baseline"] = base.iloc[0][col]
    else:
        rename_map = {col: f"{col}_baseline" for col in value_cols}
        base = base[join_cols + value_cols].rename(columns=rename_map)
        out = df.merge(base, on=join_cols, how="left")
    for col in value_cols:
        base_col = f"{col}_baseline"
        delta_col = f"{col}_delta_vs_{baseline_source}"
        ratio_col = f"{col}_ratio_vs_{baseline_source}"
        out[delta_col] = out[col] - out[base_col]
        out[ratio_col] = np.where(out[base_col] != 0, out[col] / out[base_col], np.nan)
    return out


def build_regime_summary(daily_summary_df: pd.DataFrame) -> pd.DataFrame:
    if daily_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "regime",
                "n_days",
                "total_pnl",
                "mean_daily_pnl",
                "mean_abs_dispatch_kw",
                "mean_soc_close_kwh",
            ]
        )

    out = (
        daily_summary_df.groupby(["source", "regime"], as_index=False)
        .agg(
            n_days=("date", "nunique"),
            total_pnl=("total_pnl", "sum"),
            mean_daily_pnl=("total_pnl", "mean"),
            mean_abs_dispatch_kw=("mean_abs_dispatch_kw", "mean"),
            mean_soc_close_kwh=("soc_close_kwh", "mean"),
        )
        .sort_values(["regime", "source"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def build_spike_band_summary(daily_summary_df: pd.DataFrame) -> pd.DataFrame:
    if daily_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "spike_band",
                "n_days",
                "total_pnl",
                "mean_daily_pnl",
                "mean_abs_dispatch_kw",
                "mean_soc_close_kwh",
            ]
        )

    out = (
        daily_summary_df.groupby(["source", "spike_band"], as_index=False)
        .agg(
            n_days=("date", "nunique"),
            total_pnl=("total_pnl", "sum"),
            mean_daily_pnl=("total_pnl", "mean"),
            mean_abs_dispatch_kw=("mean_abs_dispatch_kw", "mean"),
            mean_soc_close_kwh=("soc_close_kwh", "mean"),
        )
        .sort_values(["spike_band", "source"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def build_behavior_summary(daily_summary_df: pd.DataFrame) -> pd.DataFrame:
    if daily_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "regime",
                "n_days",
                "mean_daily_pnl",
                "mean_charge_energy_kwh",
                "mean_discharge_energy_kwh",
                "mean_avg_charge_price_mwh",
                "mean_avg_discharge_price_mwh",
                "mean_abs_dispatch_kw",
                "mean_soc_open_kwh",
                "mean_soc_close_kwh",
                "mean_soc_delta_kwh",
            ]
        )

    out = (
        daily_summary_df.groupby(["source", "regime"], as_index=False)
        .agg(
            n_days=("date", "nunique"),
            mean_daily_pnl=("total_pnl", "mean"),
            mean_charge_energy_kwh=("charge_energy_kwh", "mean"),
            mean_discharge_energy_kwh=("discharge_energy_kwh", "mean"),
            mean_avg_charge_price_mwh=("avg_charge_price_mwh", "mean"),
            mean_avg_discharge_price_mwh=("avg_discharge_price_mwh", "mean"),
            mean_abs_dispatch_kw=("mean_abs_dispatch_kw", "mean"),
            mean_soc_open_kwh=("soc_open_kwh", "mean"),
            mean_soc_close_kwh=("soc_close_kwh", "mean"),
            mean_soc_delta_kwh=("soc_delta_kwh", "mean"),
        )
        .sort_values(["regime", "source"], kind="stable")
        .reset_index(drop=True)
    )
    return out


def _simulate_single_source(
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    soc_init: float,
    args_dict: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
    actual_prices = _load_actuals_5m()["rrp"].reindex(timestamps)
    providers = ForecastProviders(argparse.Namespace(**args_dict))
    return simulate_stepwise(
        timestamps=timestamps,
        actual_prices=actual_prices,
        providers=providers,
        sources=[source],
        soc_init=soc_init,
        terminal_energy_value_per_kwh=args_dict["terminal_energy_value_per_kwh"],
        dual_terminal_scale=args_dict["dual_terminal_scale"],
    )


def simulate_stepwise_parallel(
    start: pd.Timestamp,
    end: pd.Timestamp,
    sources: list[str],
    soc_init: float,
    args_dict: dict,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if workers <= 1 or len(sources) <= 1:
        providers = ForecastProviders(argparse.Namespace(**args_dict))
        timestamps = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
        actual_prices = providers.actuals_5m["rrp"].reindex(timestamps)
        return simulate_stepwise(
            timestamps,
            actual_prices,
            providers,
            sources,
            soc_init,
            terminal_energy_value_per_kwh=args_dict["terminal_energy_value_per_kwh"],
            dual_terminal_scale=args_dict["dual_terminal_scale"],
        )

    raw_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as ex:
        futures = {
            ex.submit(
                _simulate_single_source,
                source,
                start,
                end,
                soc_init,
                args_dict,
            ): source
            for source in sources
        }
        for fut in concurrent.futures.as_completed(futures):
            source = futures[fut]
            raw_df, summary_df = fut.result()
            print(f"  Completed source: {source}")
            raw_frames.append(raw_df)
            summary_frames.append(summary_df)

    raw_out = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    summary_out = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if not raw_out.empty:
        raw_out = raw_out.sort_values(["time", "source"], kind="stable").reset_index(drop=True)
    if not summary_out.empty:
        summary_out = summary_out.sort_values("source", kind="stable").reset_index(drop=True)
    return raw_out, summary_out


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
    parser.add_argument(
        "--terminal-energy-value-mwh",
        type=float,
        default=0.0,
        help=(
            "Optional end-of-horizon salvage value for stored energy, in $/MWh. "
            "Positive values bias the LP toward preserving charge."
        ),
    )
    parser.add_argument(
        "--dual-terminal-scale",
        type=float,
        default=0.0,
        help=(
            "If > 0, probe the LP shadow price of initial SoC each step and set the control "
            "solve terminal-energy value to max(0, scale * shadow_price)."
        ),
    )
    parser.add_argument("--output-prefix", default="rolling_mpc_eval_model_a")
    parser.add_argument(
        "--baseline-source",
        default="",
        help="Optional baseline source for delta reporting. Defaults to the first listed source.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes across sources. Default: min(number of sources, CPU count).",
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    baseline_source = args.baseline_source.strip() or sources[0]
    args.terminal_energy_value_per_kwh = args.terminal_energy_value_mwh / 1000.0
    if args.dual_terminal_scale < 0:
        parser.error("--dual-terminal-scale must be >= 0")
    if args.dual_terminal_scale > 0 and args.terminal_energy_value_mwh != 0.0:
        parser.error("Use either --terminal-energy-value-mwh or --dual-terminal-scale, not both")

    if "model_a_hybrid" in sources and (not args.tft_checkpoint or not args.tft_scalers):
        parser.error("model_a_hybrid requires --tft-checkpoint and --tft-scalers")
    if baseline_source not in sources:
        parser.error("--baseline-source must be one of --sources")

    idx = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
    workers = args.workers or min(len(sources), max(1, (os.cpu_count() or 1)))
    workers = max(1, min(workers, len(sources)))
    args_dict = vars(args).copy()

    print(f"Rolling MPC eval — Model A track")
    print(f"  Window: {start} → {end}")
    print(f"  Steps:  {len(idx):,}")
    print(f"  Sources: {sources}")
    print(f"  Baseline: {baseline_source}")
    print(f"  Terminal energy value: {args.terminal_energy_value_mwh:.3f} $/MWh")
    print(f"  Dual terminal scale: {args.dual_terminal_scale:.3f}")
    print(f"  Workers: {workers}")

    t0 = time.time()
    raw_df, summary_df = simulate_stepwise_parallel(
        start=start,
        end=end,
        sources=sources,
        soc_init=args.soc_init_kwh,
        args_dict=args_dict,
        workers=workers,
    )
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / f"{args.output_prefix}_raw.parquet"
    summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.csv"
    daily_summary_path = RESULTS_DIR / f"{args.output_prefix}_daily_summary.csv"
    daily_regime_path = RESULTS_DIR / f"{args.output_prefix}_daily_regimes.csv"
    coverage_path = RESULTS_DIR / f"{args.output_prefix}_coverage.csv"
    summary_delta_path = RESULTS_DIR / f"{args.output_prefix}_summary_vs_baseline.csv"
    daily_delta_path = RESULTS_DIR / f"{args.output_prefix}_daily_summary_vs_baseline.csv"
    regime_summary_path = RESULTS_DIR / f"{args.output_prefix}_regime_summary.csv"
    regime_delta_path = RESULTS_DIR / f"{args.output_prefix}_regime_summary_vs_baseline.csv"
    spike_band_summary_path = RESULTS_DIR / f"{args.output_prefix}_spike_band_summary.csv"
    spike_band_delta_path = RESULTS_DIR / f"{args.output_prefix}_spike_band_summary_vs_baseline.csv"
    behavior_summary_path = RESULTS_DIR / f"{args.output_prefix}_behavior_summary.csv"
    behavior_delta_path = RESULTS_DIR / f"{args.output_prefix}_behavior_summary_vs_baseline.csv"
    daily_summary_df = build_daily_summary(raw_df)
    daily_regime_df = build_daily_regime_summary(raw_df)
    daily_summary_df = add_daily_regime_to_summary(daily_summary_df, daily_regime_df)
    coverage_df = build_coverage_summary(raw_df, start, end, sources, summary_df=summary_df)
    regime_summary_df = build_regime_summary(daily_summary_df)
    spike_band_summary_df = build_spike_band_summary(daily_summary_df)
    behavior_summary_df = build_behavior_summary(daily_summary_df)
    summary_with_deltas_df = add_baseline_deltas(
        summary_df,
        baseline_source=baseline_source,
        value_cols=["total_pnl", "mean_per_day", "soc_final_kwh"],
    )
    daily_summary_with_deltas_df = add_baseline_deltas(
        daily_summary_df,
        baseline_source=baseline_source,
        value_cols=["total_pnl", "mean_abs_dispatch_kw", "soc_close_kwh"],
        join_cols=["date"],
    )
    regime_summary_with_deltas_df = add_baseline_deltas(
        regime_summary_df,
        baseline_source=baseline_source,
        value_cols=["total_pnl", "mean_daily_pnl", "mean_abs_dispatch_kw", "mean_soc_close_kwh"],
        join_cols=["regime"],
    )
    spike_band_summary_with_deltas_df = add_baseline_deltas(
        spike_band_summary_df,
        baseline_source=baseline_source,
        value_cols=["total_pnl", "mean_daily_pnl", "mean_abs_dispatch_kw", "mean_soc_close_kwh"],
        join_cols=["spike_band"],
    )
    behavior_summary_with_deltas_df = add_baseline_deltas(
        behavior_summary_df,
        baseline_source=baseline_source,
        value_cols=[
            "mean_daily_pnl",
            "mean_charge_energy_kwh",
            "mean_discharge_energy_kwh",
            "mean_avg_charge_price_mwh",
            "mean_avg_discharge_price_mwh",
            "mean_abs_dispatch_kw",
            "mean_soc_open_kwh",
            "mean_soc_close_kwh",
            "mean_soc_delta_kwh",
        ],
        join_cols=["regime"],
    )
    raw_df.to_parquet(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    daily_summary_df.to_csv(daily_summary_path, index=False)
    daily_regime_df.to_csv(daily_regime_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    summary_with_deltas_df.to_csv(summary_delta_path, index=False)
    daily_summary_with_deltas_df.to_csv(daily_delta_path, index=False)
    regime_summary_df.to_csv(regime_summary_path, index=False)
    regime_summary_with_deltas_df.to_csv(regime_delta_path, index=False)
    spike_band_summary_df.to_csv(spike_band_summary_path, index=False)
    spike_band_summary_with_deltas_df.to_csv(spike_band_delta_path, index=False)
    behavior_summary_df.to_csv(behavior_summary_path, index=False)
    behavior_summary_with_deltas_df.to_csv(behavior_delta_path, index=False)
    print(f"Saved raw → {raw_path.relative_to(ROOT)}")
    print(f"Saved summary → {summary_path.relative_to(ROOT)}")
    print(f"Saved daily summary → {daily_summary_path.relative_to(ROOT)}")
    print(f"Saved daily regimes → {daily_regime_path.relative_to(ROOT)}")
    print(f"Saved coverage → {coverage_path.relative_to(ROOT)}")
    print(f"Saved summary vs baseline → {summary_delta_path.relative_to(ROOT)}")
    print(f"Saved daily summary vs baseline → {daily_delta_path.relative_to(ROOT)}")
    print(f"Saved regime summary → {regime_summary_path.relative_to(ROOT)}")
    print(f"Saved regime summary vs baseline → {regime_delta_path.relative_to(ROOT)}")
    print(f"Saved spike-band summary → {spike_band_summary_path.relative_to(ROOT)}")
    print(f"Saved spike-band summary vs baseline → {spike_band_delta_path.relative_to(ROOT)}")
    print(f"Saved behavior summary → {behavior_summary_path.relative_to(ROOT)}")
    print(f"Saved behavior summary vs baseline → {behavior_delta_path.relative_to(ROOT)}")
    print(summary_with_deltas_df.to_string(index=False))
    print(coverage_df.to_string(index=False))


if __name__ == "__main__":
    main()
