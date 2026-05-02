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

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ai-energy-forecast-slop-mplconfig")

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
from retro_tier1_inference import (  # noqa: E402
    build_feature_dict as build_tier1_feature_dict,
    build_long_matrix_for_model as build_tier1_long_matrix,
)
from train_tft_price import TFTPriceModel  # noqa: E402
PARQUET_DIR = ROOT / "data" / "parquet"
RESULTS_DIR = ROOT / "eval" / "results"
PRICE_FORECAST_LOG = ROOT / "price_forecast_log.csv"
OOF_FILE = PARQUET_DIR / "debiased_pd_rrp_oof.parquet"

ACTUALS_5M_FILE = PARQUET_DIR / "actuals_sa1_5m.parquet"
ACTUALS_30M_FILE = PARQUET_DIR / "actuals_sa1.parquet"
P5MIN_FILE = PARQUET_DIR / "aemo_p5min_sa1.parquet"
DEFAULT_TACTICAL_MODEL_DIR = ROOT / "models" / "lgbm_tactical"

HORIZON_5M_STEPS = 14 * 12  # 14h × 12 steps/hour = 168
TACTICAL_STEPS = 12         # 0–60 min at 5-min resolution
STRATEGIC_HORIZON_5M_STEPS = 72 * 12  # 72h × 12 steps/hour = 864


def load_config() -> dict:
    with open(ROOT / "config.json") as f:
        return json.load(f)


CONFIG = load_config()
LOCAL_TZ = pytz.timezone(CONFIG["timezone"])


def _load_tariff_profile() -> tuple[dict[str, float], dict[str, float], float]:
    tariff_path = ROOT / CONFIG["paths"]["tariff_file"]
    try:
        with open(tariff_path) as f:
            tariffs = json.load(f)
    except FileNotFoundError:
        return {}, {}, 1.05
    return (
        tariffs.get("general_tariff", {}),
        tariffs.get("feed_in_tariff", {}),
        float(tariffs.get("network_loss_factor", 1.05)),
    )


GENERAL_TARIFF_MAP, FEED_IN_TARIFF_MAP, NETWORK_LOSS_FACTOR = _load_tariff_profile()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _curve_window_mean(curve: np.ndarray, steps: int) -> float:
    steps = max(1, min(int(steps), len(curve)))
    return float(np.mean(curve[:steps]))


def _parse_csv_set(text: str) -> set[str]:
    return {item.strip() for item in str(text).split(",") if item.strip()}


def _resolve_results_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find path: {path_arg}")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_grid_exchange_reduction_signal(
    path_arg: str,
    *,
    label: str,
    score_col: str,
    min_score: float,
    horizon_steps: int | None = None,
    split: str = "",
) -> dict[pd.Timestamp, float]:
    if not path_arg:
        return {}
    path = _resolve_results_path(path_arg)
    df = _read_table(path)
    required = {"time", score_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Grid-exchange signal file missing columns: {sorted(missing)}")
    if "label" in df.columns:
        df = df[df["label"] == label].copy()
    if split and "split" in df.columns:
        df = df[df["split"] == split].copy()
    if horizon_steps is not None and horizon_steps > 0 and "horizon_steps" in df.columns:
        df = df[pd.to_numeric(df["horizon_steps"], errors="coerce") == int(horizon_steps)].copy()
    if df.empty:
        return {}
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[np.isfinite(df[score_col]) & (df[score_col] >= float(min_score))].copy()
    if df.empty:
        return {}
    grouped = df.groupby("time", sort=True)[score_col].max()
    return {pd.Timestamp(ts): float(score) for ts, score in grouped.items()}


def _progress_path(output_prefix: str, source: str | None = None) -> Path:
    suffix = f"_{source}" if source else ""
    return RESULTS_DIR / f"{output_prefix}{suffix}.progress.json"


def _write_progress_checkpoint(
    path: Path,
    *,
    source: str | None,
    steps_completed: int,
    steps_total: int,
    elapsed_seconds: float,
    current_sim_time: pd.Timestamp,
    completed: bool,
) -> None:
    percent_complete = (100.0 * steps_completed / steps_total) if steps_total else 100.0
    rate_steps_per_second = (steps_completed / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    remaining_steps = max(0, steps_total - steps_completed)
    estimated_remaining_seconds = (
        remaining_steps / rate_steps_per_second if rate_steps_per_second > 0 else None
    )
    eta_timestamp = (
        pd.Timestamp.now(tz="UTC") + pd.to_timedelta(estimated_remaining_seconds, unit="s")
        if estimated_remaining_seconds is not None
        else None
    )
    payload = {
        "source": source,
        "steps_completed": int(steps_completed),
        "steps_total": int(steps_total),
        "percent_complete": float(percent_complete),
        "elapsed_seconds": float(elapsed_seconds),
        "rate_steps_per_second": float(rate_steps_per_second),
        "estimated_remaining_seconds": (
            float(estimated_remaining_seconds) if estimated_remaining_seconds is not None else None
        ),
        "eta_timestamp": eta_timestamp.isoformat() if eta_timestamp is not None else None,
        "current_sim_time": current_sim_time.isoformat(),
        "completed": bool(completed),
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _should_apply_sell_urgency(
    *,
    tactical_source: str,
    allowed_tactical_sources: set[str],
    trigger_feed_in_price_mwh: float,
    current_feed_in_price_mwh: float,
    max_strategic_target_kwh: float,
    strategic_soc_target_kwh: float,
    discount: float,
    horizon_steps: int,
) -> bool:
    return not (
        not allowed_tactical_sources
        or tactical_source not in allowed_tactical_sources
        or discount <= 0.0
        or horizon_steps <= 0
        or not np.isfinite(current_feed_in_price_mwh)
        or current_feed_in_price_mwh < trigger_feed_in_price_mwh
        or not np.isfinite(strategic_soc_target_kwh)
        or strategic_soc_target_kwh > max_strategic_target_kwh
    )


def _apply_sell_urgency_transform(
    export_prices_mwh: np.ndarray,
    *,
    discount: float,
    horizon_steps: int,
) -> tuple[np.ndarray, bool]:
    shaped = np.asarray(export_prices_mwh, dtype=np.float64).copy()
    if discount <= 0.0 or horizon_steps <= 0:
        return shaped, False
    stop = min(len(shaped), 1 + int(horizon_steps))
    if stop <= 1:
        return shaped, False
    shaped[1:stop] *= max(0.0, 1.0 - discount)
    return shaped, True


def _should_apply_inventory_discipline(
    *,
    source: str,
    allowed_sources: set[str],
    cycle_cost_adder_per_kwh: float,
    current_feed_in_price_mwh: float,
    current_net_load_kw: float,
    feed_in_max_mwh: float,
    net_load_max_kw: float,
) -> bool:
    return not (
        not allowed_sources
        or source not in allowed_sources
        or cycle_cost_adder_per_kwh <= 0.0
        or not np.isfinite(current_feed_in_price_mwh)
        or not np.isfinite(current_net_load_kw)
        or current_feed_in_price_mwh >= feed_in_max_mwh
        or current_net_load_kw >= net_load_max_kw
    )


def _should_apply_grid_exchange_reduction(
    *,
    source: str,
    ts: pd.Timestamp,
    allowed_sources: set[str],
    cycle_cost_adder_per_kwh: float,
    flow_cost_adder_per_kwh: float = 0.0,
    scores_by_time: dict[pd.Timestamp, float],
) -> bool:
    return not (
        not allowed_sources
        or source not in allowed_sources
        or (cycle_cost_adder_per_kwh <= 0.0 and flow_cost_adder_per_kwh <= 0.0)
        or not scores_by_time
        or pd.Timestamp(ts) not in scores_by_time
    )


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


def _load_load_5m_from_30m(actuals_30m: pd.DataFrame) -> pd.Series:
    load_5m = _expand_actual_30m_power_to_5m(actuals_30m, "power_load")
    if load_5m.empty:
        return load_5m
    load_kw = load_5m / 1000.0
    load_kw.name = "load_kw"
    return load_kw.astype(np.float64)


def _load_pv_kw_5m_from_30m(actuals_30m: pd.DataFrame) -> pd.Series:
    pv_5m = _expand_actual_30m_power_to_5m(actuals_30m, "power_pv")
    if pv_5m.empty:
        return pv_5m
    pv_kw = pv_5m / 1000.0
    pv_kw.name = "pv_kw"
    return pv_kw.astype(np.float64)


def _expand_actual_30m_power_to_5m(actuals_30m: pd.DataFrame, column: str) -> pd.Series:
    if column not in actuals_30m.columns:
        return pd.Series(dtype=np.float64)
    series = pd.to_numeric(actuals_30m[column], errors="coerce").astype(np.float64)
    expanded = series.resample("5min").ffill()
    expanded.name = column
    return expanded


def _load_net_load_5m_from_30m(actuals_30m: pd.DataFrame) -> pd.Series:
    load_5m = _expand_actual_30m_power_to_5m(actuals_30m, "power_load")
    pv_5m = _expand_actual_30m_power_to_5m(actuals_30m, "power_pv")
    if load_5m.empty and pv_5m.empty:
        return pd.Series(dtype=np.float64)
    idx = load_5m.index.union(pv_5m.index)
    net_load_kw = (
        load_5m.reindex(idx).ffill().fillna(0.0)
        - pv_5m.reindex(idx).ffill().fillna(0.0)
    ) / 1000.0
    net_load_kw.name = "net_load_kw"
    return net_load_kw.astype(np.float64)


def _tariffed_price_frame_from_wholesale_mwh(wholesale_prices_mwh: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=wholesale_prices_mwh.index.copy())
    frame["wholesale_price"] = wholesale_prices_mwh.astype(np.float64) / 1000.0
    local_time = pd.Series(
        frame.index.tz_convert(LOCAL_TZ).floor("30min").time.astype(str),
        index=frame.index,
        dtype="string",
    )
    general_tariff = local_time.map(GENERAL_TARIFF_MAP).fillna(0.0).astype(np.float64)
    feed_in_tariff = local_time.map(FEED_IN_TARIFF_MAP).fillna(0.0).astype(np.float64)

    general_price_ex_gst = frame["wholesale_price"] * NETWORK_LOSS_FACTOR + general_tariff
    feed_in_price_ex_gst = frame["wholesale_price"] * NETWORK_LOSS_FACTOR + feed_in_tariff

    frame["general_price"] = np.where(
        general_price_ex_gst > 0,
        general_price_ex_gst * CONFIG["gst_rate"],
        general_price_ex_gst,
    )
    frame["feed_in_price"] = np.where(
        feed_in_price_ex_gst < 0,
        feed_in_price_ex_gst * CONFIG["gst_rate"],
        feed_in_price_ex_gst,
    )
    frame["general_price_mwh"] = frame["general_price"] * 1000.0
    frame["feed_in_price_mwh"] = frame["feed_in_price"] * 1000.0
    return frame


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
    quantiles: list[float]
    q50_idx: int


@dataclass
class StrategicSolveSummary:
    target_kwh: float
    initial_shadow_price_per_kwh: float


@dataclass(frozen=True)
class SourceContract:
    name: str
    tactical_source: str
    strategic_source: str


VALID_BASE_SOURCES = {"p5min_naive", "amber_apf_lgbm", "model_a_hybrid"}

COUNTERFACTUAL_SOURCE_ALIASES = {
    "hybrid_tactical_amber_strategic": SourceContract(
        name="hybrid_tactical_amber_strategic",
        tactical_source="model_a_hybrid",
        strategic_source="amber_apf_lgbm",
    ),
    "amber_tactical_hybrid_strategic": SourceContract(
        name="amber_tactical_hybrid_strategic",
        tactical_source="amber_apf_lgbm",
        strategic_source="model_a_hybrid",
    ),
    "model_a_hybrid_inventory_bias": SourceContract(
        name="model_a_hybrid_inventory_bias",
        tactical_source="model_a_hybrid",
        strategic_source="model_a_hybrid",
    ),
    "model_a_hybrid_grid_exchange_gate": SourceContract(
        name="model_a_hybrid_grid_exchange_gate",
        tactical_source="model_a_hybrid",
        strategic_source="model_a_hybrid",
    ),
}


def _parse_source_contract(spec: str) -> SourceContract:
    token = spec.strip()
    if not token:
        raise ValueError("Empty source spec")
    if token in COUNTERFACTUAL_SOURCE_ALIASES:
        return COUNTERFACTUAL_SOURCE_ALIASES[token]
    if token in VALID_BASE_SOURCES:
        return SourceContract(name=token, tactical_source=token, strategic_source=token)
    if token.startswith("cf:"):
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                "Counterfactual source specs must look like "
                "'cf:<label>:<tactical_source>:<strategic_source>'"
            )
        _, label, tactical_source, strategic_source = parts
        if tactical_source not in VALID_BASE_SOURCES:
            raise ValueError(f"Unsupported tactical source '{tactical_source}' in '{token}'")
        if strategic_source not in VALID_BASE_SOURCES:
            raise ValueError(f"Unsupported strategic source '{strategic_source}' in '{token}'")
        return SourceContract(
            name=label,
            tactical_source=tactical_source,
            strategic_source=strategic_source,
        )
    raise ValueError(
        f"Unsupported source '{token}'. Use one of {sorted(VALID_BASE_SOURCES | set(COUNTERFACTUAL_SOURCE_ALIASES))} "
        "or cf:<label>:<tactical_source>:<strategic_source>."
    )


def _parse_source_contracts(specs: list[str]) -> list[SourceContract]:
    contracts = [_parse_source_contract(spec) for spec in specs]
    names = [contract.name for contract in contracts]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate source labels are not allowed: {names}")
    return contracts


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
        quantiles=list(quantiles),
        q50_idx=q50_idx,
    )


def _find_quantile_index(quantiles: list[float], target: float) -> int:
    for i, q in enumerate(quantiles):
        if abs(float(q) - float(target)) < 1e-9:
            return i
    raise ValueError(f"Quantile {target} not present in {quantiles}")


def _infer_tft_quantiles_30m(start_ts: pd.Timestamp, ctx: TFTContext) -> np.ndarray | None:
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
    return preds_raw.astype(np.float64)


class ForecastProviders:
    def __init__(self, args):
        self.actuals_5m = _load_actuals_5m()
        self.actuals_30m = _load_actuals_30m()
        self.pv_5m = _load_pv_5m_from_30m(self.actuals_30m)
        self.load_kw_5m = _load_load_5m_from_30m(self.actuals_30m)
        self.pv_kw_5m = _load_pv_kw_5m_from_30m(self.actuals_30m)
        self.net_load_5m = _load_net_load_5m_from_30m(self.actuals_30m)
        self.p5min_runs, self.p5_run_times_sorted, self.p5_run_times_ns = _load_p5min_runs()
        tactical_model_dir = Path(args.tactical_model_dir)
        self.tactical_models = {
            0.05: joblib.load(tactical_model_dir / "lgbm_q05.pkl"),
            0.50: joblib.load(tactical_model_dir / "lgbm_q50.pkl"),
            0.95: joblib.load(tactical_model_dir / "lgbm_q95.pkl"),
        }
        self.args = args

        self.amber_runs, self.amber_creation_sorted, self.amber_creation_ns = _load_amber_log_30m()

        self.tft_ctx = None
        if args.tft_checkpoint and args.tft_scalers:
            self.tft_ctx = _load_tft_context(Path(args.tft_checkpoint), Path(args.tft_scalers))

        self._tier1_cache: dict[tuple[pd.Timestamp, float], pd.Series] = {}
        self._tft_cache: dict[tuple[pd.Timestamp, float], pd.Series] = {}
        self._amber_cache: dict[pd.Timestamp, pd.Series] = {}
        self._strategic_solve_cache: dict[tuple[str, pd.Timestamp, float, float], StrategicSolveSummary | None] = {}
        self._last_curve_repaired: dict[str, bool] = {}

    def curve_was_repaired(self, source: str) -> bool:
        return self._last_curve_repaired.get(source, False)

    def _finalize_curve(self, source: str, values: np.ndarray, current_actual: float,
                        expected_steps: int = HORIZON_5M_STEPS) -> np.ndarray | None:
        out = np.asarray(values, dtype=np.float64).copy()
        self._last_curve_repaired[source] = False
        if len(out) == 0:
            return None
        out[0] = current_actual
        if out.shape[0] != expected_steps:
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

    def _blend_series(self, base: pd.Series, upper: pd.Series, weight: float) -> pd.Series:
        return base + float(weight) * (upper - base)

    def _build_model_a_curve(
        self,
        ts: pd.Timestamp,
        *,
        tier1_signed_blend: float,
        tier2_signed_blend: float,
    ) -> np.ndarray | None:
        idx = pd.date_range(start=ts, periods=HORIZON_5M_STEPS, freq="5min", tz="UTC")
        current_actual = self.current_actual_price(ts)
        if np.isnan(current_actual):
            return None
        tier1_q50 = self.tier1_quantile(ts, 0.50)
        tier1_q95 = self.tier1_quantile(ts, 0.95)
        tft_q50 = self.tft_quantile_expanded(ts, 0.50)
        tft_qhi = self.tft_quantile_expanded(ts, self.args.tier2_upper_quantile)
        if tier1_q50 is None or tier1_q95 is None or tft_q50 is None or tft_qhi is None:
            return None
        tier1 = self._blend_series(tier1_q50, tier1_q95, tier1_signed_blend)
        tft = self._blend_series(tft_q50, tft_qhi, tier2_signed_blend)
        out = pd.Series(index=idx, dtype=np.float64)
        out.loc[idx[:TACTICAL_STEPS]] = tier1.reindex(idx[:TACTICAL_STEPS]).ffill().bfill().values
        later_idx = idx[TACTICAL_STEPS:]
        out.loc[later_idx] = tft.reindex(later_idx).ffill().bfill().values
        return self._finalize_curve("model_a_hybrid", out.values.astype(np.float64), current_actual)

    def tier1_quantile(self, ts: pd.Timestamp, quantile: float) -> pd.Series | None:
        if quantile not in self.tactical_models:
            raise ValueError(f"Unsupported tactical quantile: {quantile}")
        pos = int(np.searchsorted(self.p5_run_times_ns, ts.value, side="right"))
        if pos == 0:
            return None
        run_time = self.p5_run_times_sorted[pos - 1]
        cache_key = (run_time, float(quantile))
        if cache_key in self._tier1_cache:
            return self._tier1_cache[cache_key]

        p5min_rrp = self.p5min_runs[run_time]
        prev_rt = run_time - pd.Timedelta(minutes=5)
        prev_p5min_h0 = self.p5min_runs[prev_rt][0] if prev_rt in self.p5min_runs else float("nan")
        feature_dict = build_tier1_feature_dict(
            run_time, p5min_rrp, prev_p5min_h0, self.actuals_5m, self.pv_5m
        )
        X_long = build_tier1_long_matrix(self.tactical_models[quantile], feature_dict, TACTICAL_STEPS)
        preds = self.tactical_models[quantile].predict(X_long).astype(np.float64)
        idx = pd.date_range(start=run_time, periods=TACTICAL_STEPS, freq="5min", tz="UTC")
        series = pd.Series(preds, index=idx)
        self._tier1_cache[cache_key] = series
        return series

    def tft_quantile_expanded(self, ts: pd.Timestamp, quantile: float) -> pd.Series | None:
        if self.tft_ctx is None:
            return None
        anchor = ts.floor("30min")
        cache_key = (anchor, float(quantile))
        if cache_key in self._tft_cache:
            return self._tft_cache[cache_key]
        preds_30m = _infer_tft_quantiles_30m(anchor, self.tft_ctx)
        if preds_30m is None:
            return None
        q_idx = _find_quantile_index(self.tft_ctx.quantiles, quantile)
        pred_30m = preds_30m[:, q_idx]
        idx_5m = pd.date_range(start=anchor, periods=len(pred_30m) * 6, freq="5min", tz="UTC")
        series = pd.Series(np.repeat(pred_30m, 6), index=idx_5m)
        self._tft_cache[cache_key] = series
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

    def build_strategic_curve(self, source: str, ts: pd.Timestamp, quantile: float = 0.50) -> np.ndarray | None:
        idx = pd.date_range(start=ts, periods=STRATEGIC_HORIZON_5M_STEPS, freq="5min", tz="UTC")
        current_actual = self.current_actual_price(ts)
        if np.isnan(current_actual):
            return None

        if source == "amber_apf_lgbm":
            amber = self.amber_expanded(ts)
            if amber is None:
                return None
            out = amber.reindex(idx).ffill().bfill().values.astype(np.float64)
            return self._finalize_curve(source, out, current_actual, expected_steps=STRATEGIC_HORIZON_5M_STEPS)

        if source == "model_a_hybrid":
            tft = self.tft_quantile_expanded(ts, quantile)
            if tft is None:
                return None
            out = tft.reindex(idx).ffill().bfill().values.astype(np.float64)
            return self._finalize_curve(source, out, current_actual, expected_steps=STRATEGIC_HORIZON_5M_STEPS)

        return None

    def strategic_solve_summary(
        self,
        source: str,
        ts: pd.Timestamp,
        soc_init_kwh: float,
        quantile: float = 0.50,
    ) -> StrategicSolveSummary | None:
        cache_key = (source, ts, round(float(soc_init_kwh), 6), round(float(quantile), 4))
        if cache_key in self._strategic_solve_cache:
            return self._strategic_solve_cache[cache_key]

        curve = self.build_strategic_curve(source, ts, quantile=quantile)
        if curve is None:
            self._strategic_solve_cache[cache_key] = None
            return None

        strategic = solve_lp_dispatch(curve, soc_init_kwh)
        soc_path = strategic.get("soc_trajectory_kwh")
        if soc_path is None or len(soc_path) < HORIZON_5M_STEPS:
            self._strategic_solve_cache[cache_key] = None
            return None

        target = float(np.clip(soc_path[HORIZON_5M_STEPS - 1], 0.0, CAPACITY_KWH))
        summary = StrategicSolveSummary(
            target_kwh=target,
            initial_shadow_price_per_kwh=float(strategic.get("initial_soc_shadow_price_per_kwh", float("nan"))),
        )
        self._strategic_solve_cache[cache_key] = summary
        return summary

    def strategic_soc_target(
        self,
        source: str,
        ts: pd.Timestamp,
        soc_init_kwh: float,
        quantile: float = 0.50,
    ) -> float | None:
        summary = self.strategic_solve_summary(source, ts, soc_init_kwh, quantile=quantile)
        if summary is None:
            return None
        return summary.target_kwh

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
            return self._build_model_a_curve(
                ts,
                tier1_signed_blend=self.args.tier1_quantile_blend,
                tier2_signed_blend=self.args.tier2_quantile_blend,
            )

        raise ValueError(f"Unknown source: {source}")

    def build_effective_netload_curves(
        self,
        source: str,
        ts: pd.Timestamp,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        base_curve = self.build_forecast_curve(source, ts)
        if base_curve is None:
            return None
        if (
            source != "model_a_hybrid"
            or source not in self.args.split_curve_tactical_sources
            or (
                self.args.buy_curve_quantile_blend == 0.0
                and self.args.sell_curve_quantile_blend == 0.0
            )
        ):
            return base_curve, base_curve.copy(), base_curve.copy()
        buy_curve = self._build_model_a_curve(
            ts,
            tier1_signed_blend=-self.args.buy_curve_quantile_blend,
            tier2_signed_blend=-self.args.buy_curve_quantile_blend,
        )
        sell_curve = self._build_model_a_curve(
            ts,
            tier1_signed_blend=self.args.sell_curve_quantile_blend,
            tier2_signed_blend=self.args.sell_curve_quantile_blend,
        )
        if buy_curve is None or sell_curve is None:
            return None
        return base_curve, buy_curve, sell_curve


def _resolve_mp_start_method(requested: str) -> str:
    available = set(mp.get_all_start_methods())
    if requested == "auto":
        for candidate in ("fork", "forkserver", "spawn"):
            if candidate in available:
                return candidate
        raise RuntimeError(f"No supported multiprocessing start method available from {sorted(available)}")
    if requested not in available:
        raise ValueError(
            f"Requested multiprocessing start method '{requested}' is unavailable; "
            f"available={sorted(available)}"
        )
    return requested


def simulate_stepwise(
    timestamps: pd.DatetimeIndex,
    actual_prices: pd.Series,
    providers: ForecastProviders,
    source_contracts: list[SourceContract],
    soc_init: float,
    economic_mode: str = "price_only",
    output_prefix: str = "",
    progress_every_steps: int = 72,
    terminal_energy_value_per_kwh: float = 0.0,
    dual_terminal_scale: float = 0.0,
    strategic_soc_handoff: bool = False,
    strategic_target_mode: str = "exact",
    dynamic_bridge_upper_quantile: float = 0.90,
    dynamic_bridge_target_scale: float = 0.0,
    dynamic_bridge_terminal_scale: float = 0.0,
    dynamic_bridge_terminal_scope: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_names = [contract.name for contract in source_contracts]
    state = {src: float(soc_init) for src in source_names}
    raw_rows = []
    summary = []
    started_at = time.time()
    steps_total = len(timestamps)
    source_progress_paths = {
        src: (_progress_path(output_prefix, src) if output_prefix else None) for src in source_names
    }

    pnl_totals = {src: 0.0 for src in source_names}
    executed_steps = {src: 0 for src in source_names}
    skipped_missing_actual = {src: 0 for src in source_names}
    skipped_missing_curve = {src: 0 for src in source_names}
    skipped_invalid_curve = {src: 0 for src in source_names}
    repaired_invalid_curve = {src: 0 for src in source_names}
    invalid_curve_logged = {src: 0 for src in source_names}

    if economic_mode == "netload_tariffed":
        tariffed_actuals = _tariffed_price_frame_from_wholesale_mwh(actual_prices.reindex(timestamps).ffill().bfill())
        actual_general_prices = tariffed_actuals["general_price_mwh"]
        actual_feed_in_prices = tariffed_actuals["feed_in_price_mwh"]
        actual_net_load_kw = providers.net_load_5m.reindex(timestamps).ffill().bfill()
        actual_load_kw = providers.load_kw_5m.reindex(timestamps).ffill().bfill()
        actual_pv_kw = providers.pv_kw_5m.reindex(timestamps).ffill().bfill()
        use_split_load_pv = (
            not actual_load_kw.empty
            and not actual_pv_kw.empty
            and actual_load_kw.notna().any()
            and actual_pv_kw.notna().any()
        )
    elif economic_mode == "price_only":
        actual_general_prices = None
        actual_feed_in_prices = None
        actual_net_load_kw = None
        actual_load_kw = None
        actual_pv_kw = None
        use_split_load_pv = False
    else:
        raise ValueError(f"Unsupported economic_mode: {economic_mode}")

    for i, ts in enumerate(timestamps):
        actual = float(actual_prices.asof(ts))
        if np.isnan(actual):
            for src in source_names:
                skipped_missing_actual[src] += 1
            continue
        actual_general_price_mwh = (
            float(actual_general_prices.asof(ts)) if actual_general_prices is not None else float("nan")
        )
        actual_feed_in_price_mwh = (
            float(actual_feed_in_prices.asof(ts)) if actual_feed_in_prices is not None else float("nan")
        )
        actual_net_load_step_kw = (
            float(actual_net_load_kw.asof(ts)) if actual_net_load_kw is not None else float("nan")
        )
        actual_load_step_kw = (
            float(actual_load_kw.asof(ts)) if actual_load_kw is not None else float("nan")
        )
        actual_pv_step_kw = (
            float(actual_pv_kw.asof(ts)) if actual_pv_kw is not None else float("nan")
        )

        for contract in source_contracts:
            src = contract.name
            effective_curves = (
                providers.build_effective_netload_curves(contract.tactical_source, ts)
                if economic_mode == "netload_tariffed"
                else None
            )
            if effective_curves is not None:
                curve, buy_curve, sell_curve = effective_curves
            else:
                curve = providers.build_forecast_curve(contract.tactical_source, ts)
                buy_curve = curve
                sell_curve = curve
            if curve is None or buy_curve is None or sell_curve is None:
                skipped_missing_curve[src] += 1
                continue
            if providers.curve_was_repaired(contract.tactical_source):
                repaired_invalid_curve[src] += 1
            curve = np.asarray(curve, dtype=np.float64)
            buy_curve = np.asarray(buy_curve, dtype=np.float64)
            sell_curve = np.asarray(sell_curve, dtype=np.float64)
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
            if (
                buy_curve.shape[0] != HORIZON_5M_STEPS
                or sell_curve.shape[0] != HORIZON_5M_STEPS
                or not np.isfinite(buy_curve).all()
                or not np.isfinite(sell_curve).all()
            ):
                skipped_invalid_curve[src] += 1
                continue
            soc_prev = state[src]
            forecast_mean_next_1h_mwh = _curve_window_mean(curve, 12)
            forecast_mean_next_4h_mwh = _curve_window_mean(curve, 48)
            forecast_mean_next_14h_mwh = _curve_window_mean(curve, HORIZON_5M_STEPS)
            forecast_buy_mean_next_1h_mwh = _curve_window_mean(buy_curve, 12)
            forecast_buy_mean_next_4h_mwh = _curve_window_mean(buy_curve, 48)
            forecast_buy_mean_next_14h_mwh = _curve_window_mean(buy_curve, HORIZON_5M_STEPS)
            forecast_sell_mean_next_1h_mwh = _curve_window_mean(sell_curve, 12)
            forecast_sell_mean_next_4h_mwh = _curve_window_mean(sell_curve, 48)
            forecast_sell_mean_next_14h_mwh = _curve_window_mean(sell_curve, HORIZON_5M_STEPS)
            probe_shadow_price_per_kwh = float("nan")
            applied_terminal_energy_value_per_kwh = terminal_energy_value_per_kwh
            strategic_soc_target_kwh = float("nan")
            strategic_soc_target_qhi_kwh = float("nan")
            strategic_target_gap_kwh = 0.0
            dynamic_target_uplift_kwh = 0.0
            strategic_shadow_q50_per_kwh = float("nan")
            strategic_shadow_qhi_per_kwh = float("nan")
            strategic_shadow_gap_per_kwh = 0.0
            dynamic_terminal_adder_per_kwh = 0.0
            extra_terminal_energy_floor_kwh = None
            extra_terminal_energy_cap_kwh = None
            extra_terminal_energy_value_per_kwh = 0.0
            min_terminal_soc_kwh = None
            max_terminal_soc_kwh = None
            inventory_discipline_applied = False
            inventory_discipline_cycle_cost_adder_per_kwh = 0.0
            grid_exchange_reduction_applied = False
            grid_exchange_reduction_cycle_cost_adder_per_kwh = 0.0
            grid_exchange_reduction_flow_cost_adder_per_kwh = 0.0
            grid_exchange_reduction_score = float("nan")
            if strategic_soc_handoff:
                strategic_q50 = providers.strategic_solve_summary(
                    contract.strategic_source,
                    ts,
                    soc_prev,
                    quantile=0.50,
                )
                if strategic_q50 is not None and np.isfinite(strategic_q50.target_kwh):
                    strategic_soc_target_kwh = float(strategic_q50.target_kwh)
                    strategic_shadow_q50_per_kwh = float(strategic_q50.initial_shadow_price_per_kwh)

                    strategic_qhi = None
                    if (
                        strategic_target_mode == "band"
                        or dynamic_bridge_target_scale > 0.0
                        or dynamic_bridge_terminal_scale > 0.0
                    ):
                        strategic_qhi = providers.strategic_solve_summary(
                            contract.strategic_source,
                            ts,
                            soc_prev,
                            quantile=dynamic_bridge_upper_quantile,
                        )
                    if strategic_qhi is not None and np.isfinite(strategic_qhi.target_kwh):
                        strategic_soc_target_qhi_kwh = float(strategic_qhi.target_kwh)
                        strategic_target_gap_kwh = max(
                            0.0,
                            strategic_soc_target_qhi_kwh - strategic_soc_target_kwh,
                        )
                        dynamic_target_uplift_kwh = max(
                            0.0,
                            dynamic_bridge_target_scale * strategic_target_gap_kwh,
                        )
                        strategic_shadow_qhi_per_kwh = float(strategic_qhi.initial_shadow_price_per_kwh)
                        if np.isfinite(strategic_shadow_q50_per_kwh) and np.isfinite(strategic_shadow_qhi_per_kwh):
                            strategic_shadow_gap_per_kwh = max(
                                0.0,
                                strategic_shadow_qhi_per_kwh - strategic_shadow_q50_per_kwh,
                            )
                            dynamic_terminal_adder_per_kwh = max(
                                0.0,
                                dynamic_bridge_terminal_scale * strategic_shadow_gap_per_kwh,
                            )

                    if strategic_target_mode == "floor":
                        min_terminal_soc_kwh = min(
                            CAPACITY_KWH,
                            strategic_soc_target_kwh + dynamic_target_uplift_kwh,
                        )
                    elif strategic_target_mode == "exact":
                        min_terminal_soc_kwh = strategic_soc_target_kwh
                        max_terminal_soc_kwh = strategic_soc_target_kwh
                    elif strategic_target_mode == "band":
                        min_terminal_soc_kwh = strategic_soc_target_kwh
                        max_terminal_soc_kwh = min(
                            CAPACITY_KWH,
                            strategic_soc_target_kwh + dynamic_target_uplift_kwh,
                        )
                        if max_terminal_soc_kwh < min_terminal_soc_kwh:
                            max_terminal_soc_kwh = min_terminal_soc_kwh
                    else:
                        raise ValueError(f"Unsupported strategic_target_mode: {strategic_target_mode}")
                    if dynamic_bridge_terminal_scope == "all":
                        applied_terminal_energy_value_per_kwh += dynamic_terminal_adder_per_kwh
                    elif dynamic_bridge_terminal_scope == "extra_band":
                        if (
                            strategic_target_mode == "band"
                            and dynamic_terminal_adder_per_kwh > 0.0
                            and dynamic_target_uplift_kwh > 0.0
                            and np.isfinite(strategic_soc_target_kwh)
                        ):
                            extra_terminal_energy_floor_kwh = strategic_soc_target_kwh
                            extra_terminal_energy_cap_kwh = dynamic_target_uplift_kwh
                            extra_terminal_energy_value_per_kwh = dynamic_terminal_adder_per_kwh
                    else:
                        raise ValueError(
                            f"Unsupported dynamic_bridge_terminal_scope: {dynamic_bridge_terminal_scope}"
                        )
            forecast_general_price_mwh = None
            forecast_feed_in_price_mwh = None
            forecast_net_load_kw = None
            forecast_load_kw = None
            forecast_pv_kw = None
            sell_urgency_applied = False
            forecast_feed_in_mean_next_1h_mwh = float("nan")
            forecast_feed_in_mean_next_4h_mwh = float("nan")
            forecast_feed_in_mean_next_14h_mwh = float("nan")
            if economic_mode == "netload_tariffed":
                curve_idx = pd.date_range(start=ts, periods=HORIZON_5M_STEPS, freq="5min", tz="UTC")
                tariffed_base_curve = _tariffed_price_frame_from_wholesale_mwh(
                    pd.Series(curve, index=curve_idx, dtype=np.float64)
                )
                tariffed_buy_curve = _tariffed_price_frame_from_wholesale_mwh(
                    pd.Series(buy_curve, index=curve_idx, dtype=np.float64)
                )
                tariffed_sell_curve = _tariffed_price_frame_from_wholesale_mwh(
                    pd.Series(sell_curve, index=curve_idx, dtype=np.float64)
                )
                forecast_general_price_mwh = tariffed_buy_curve["general_price_mwh"].to_numpy(dtype=np.float64, copy=True)
                urgency_triggered = _should_apply_sell_urgency(
                    tactical_source=contract.tactical_source,
                    allowed_tactical_sources=providers.args.sell_urgency_tactical_sources,
                    trigger_feed_in_price_mwh=providers.args.sell_urgency_trigger_feed_in_price_mwh,
                    current_feed_in_price_mwh=actual_feed_in_price_mwh,
                    max_strategic_target_kwh=providers.args.sell_urgency_max_strategic_target_kwh,
                    strategic_soc_target_kwh=strategic_soc_target_kwh,
                    discount=providers.args.sell_urgency_discount,
                    horizon_steps=providers.args.sell_urgency_horizon_steps,
                )
                if providers.args.sell_urgency_trigger_only_shaping:
                    forecast_feed_in_price_mwh = tariffed_base_curve["feed_in_price_mwh"].to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                    if urgency_triggered:
                        shaped_feed_in = tariffed_sell_curve["feed_in_price_mwh"].to_numpy(
                            dtype=np.float64,
                            copy=True,
                        )
                        forecast_feed_in_price_mwh, sell_urgency_applied = _apply_sell_urgency_transform(
                            shaped_feed_in,
                            discount=providers.args.sell_urgency_discount,
                            horizon_steps=providers.args.sell_urgency_horizon_steps,
                        )
                else:
                    forecast_feed_in_price_mwh = tariffed_sell_curve["feed_in_price_mwh"].to_numpy(
                        dtype=np.float64,
                        copy=True,
                    )
                    if urgency_triggered:
                        forecast_feed_in_price_mwh, sell_urgency_applied = _apply_sell_urgency_transform(
                            forecast_feed_in_price_mwh,
                            discount=providers.args.sell_urgency_discount,
                            horizon_steps=providers.args.sell_urgency_horizon_steps,
                        )
                forecast_feed_in_mean_next_1h_mwh = _curve_window_mean(forecast_feed_in_price_mwh, 12)
                forecast_feed_in_mean_next_4h_mwh = _curve_window_mean(forecast_feed_in_price_mwh, 48)
                forecast_feed_in_mean_next_14h_mwh = _curve_window_mean(
                    forecast_feed_in_price_mwh,
                    HORIZON_5M_STEPS,
                )
                forecast_net_load_kw = (
                    providers.net_load_5m.reindex(curve_idx).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
                )
                if use_split_load_pv:
                    maybe_load_kw = (
                        providers.load_kw_5m.reindex(curve_idx).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
                    )
                    maybe_pv_kw = (
                        providers.pv_kw_5m.reindex(curve_idx).ffill().bfill().to_numpy(dtype=np.float64, copy=True)
                    )
                    if np.isfinite(maybe_load_kw).all() and np.isfinite(maybe_pv_kw).all():
                        forecast_load_kw = maybe_load_kw
                        forecast_pv_kw = maybe_pv_kw
                inventory_discipline_applied = _should_apply_inventory_discipline(
                    source=src,
                    allowed_sources=providers.args.inventory_discipline_sources,
                    cycle_cost_adder_per_kwh=providers.args.inventory_discipline_cycle_cost_per_kwh,
                    current_feed_in_price_mwh=actual_feed_in_price_mwh,
                    current_net_load_kw=actual_net_load_step_kw,
                    feed_in_max_mwh=providers.args.inventory_discipline_feed_in_max_mwh,
                    net_load_max_kw=providers.args.inventory_discipline_net_load_max_kw,
                )
                if inventory_discipline_applied:
                    inventory_discipline_cycle_cost_adder_per_kwh = (
                        providers.args.inventory_discipline_cycle_cost_per_kwh
                    )
                grid_exchange_reduction_applied = _should_apply_grid_exchange_reduction(
                    source=src,
                    ts=ts,
                    allowed_sources=providers.args.grid_exchange_reduction_sources,
                    cycle_cost_adder_per_kwh=providers.args.grid_exchange_reduction_cycle_cost_per_kwh,
                    flow_cost_adder_per_kwh=providers.args.grid_exchange_reduction_flow_cost_per_kwh,
                    scores_by_time=providers.args.grid_exchange_reduction_scores_by_time,
                )
                if grid_exchange_reduction_applied:
                    grid_exchange_reduction_cycle_cost_adder_per_kwh = (
                        providers.args.grid_exchange_reduction_cycle_cost_per_kwh
                    )
                    grid_exchange_reduction_flow_cost_adder_per_kwh = (
                        providers.args.grid_exchange_reduction_flow_cost_per_kwh
                    )
                    grid_exchange_reduction_score = providers.args.grid_exchange_reduction_scores_by_time[
                        pd.Timestamp(ts)
                    ]
            if dual_terminal_scale > 0.0:
                probe = solve_lp_dispatch(
                    curve,
                    state[src],
                    import_prices_mwh=None if economic_mode == "price_only" else forecast_general_price_mwh,
                    export_prices_mwh=None if economic_mode == "price_only" else forecast_feed_in_price_mwh,
                    net_load_forecast_kw=None if economic_mode == "price_only" else forecast_net_load_kw,
                    load_forecast_kw=None if economic_mode == "price_only" else forecast_load_kw,
                    pv_forecast_kw=None if economic_mode == "price_only" else forecast_pv_kw,
                    terminal_energy_value_per_kwh=0.0,
                    min_terminal_soc_kwh=min_terminal_soc_kwh,
                    max_terminal_soc_kwh=max_terminal_soc_kwh,
                )
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
                import_prices_mwh=forecast_general_price_mwh,
                export_prices_mwh=forecast_feed_in_price_mwh,
                net_load_forecast_kw=forecast_net_load_kw,
                load_forecast_kw=forecast_load_kw,
                pv_forecast_kw=forecast_pv_kw,
                terminal_energy_value_per_kwh=applied_terminal_energy_value_per_kwh,
                extra_terminal_energy_value_per_kwh=extra_terminal_energy_value_per_kwh,
                extra_terminal_energy_floor_kwh=extra_terminal_energy_floor_kwh,
                extra_terminal_energy_cap_kwh=extra_terminal_energy_cap_kwh,
                min_terminal_soc_kwh=min_terminal_soc_kwh,
                max_terminal_soc_kwh=max_terminal_soc_kwh,
                throughput_cost_adder_per_kwh=(
                    inventory_discipline_cycle_cost_adder_per_kwh
                    + grid_exchange_reduction_cycle_cost_adder_per_kwh
                ),
                grid_exchange_cost_adder_per_kwh=grid_exchange_reduction_flow_cost_adder_per_kwh,
            )
            c_plan = solve["charge_kw"]
            d_plan = solve["discharge_kw"]
            c0 = float(c_plan[0]) if len(c_plan) else 0.0
            d0 = float(d_plan[0]) if len(d_plan) else 0.0
            grid_import0 = float(solve["grid_import_kw"][0]) if len(solve["grid_import_kw"]) else 0.0
            grid_export0 = float(solve["grid_export_kw"][0]) if len(solve["grid_export_kw"]) else 0.0
            curtail0 = float(solve.get("curtail_kw", [0.0])[0]) if len(solve.get("curtail_kw", [])) else 0.0
            if economic_mode == "netload_tariffed":
                if np.isfinite(actual_load_step_kw) and np.isfinite(actual_pv_step_kw):
                    grid_kw = actual_load_step_kw - actual_pv_step_kw + c0 - d0 * EFF_D + curtail0
                else:
                    grid_kw = actual_net_load_step_kw + c0 - d0 * EFF_D + curtail0
                realized_grid_import_kw = max(grid_kw, 0.0)
                realized_grid_export_kw = max(-grid_kw, 0.0)
                pnl = (
                    realized_grid_export_kw * (actual_feed_in_price_mwh / 1000.0)
                    - realized_grid_import_kw * (actual_general_price_mwh / 1000.0)
                    - DEG_PER_KWH * (c0 * EFF_C + d0)
                ) * INTERVAL_H
            else:
                p_kwh = actual / 1000.0
                realized_grid_import_kw = 0.0
                realized_grid_export_kw = 0.0
                pnl = (d0 * EFF_D * p_kwh - c0 * p_kwh - DEG_PER_KWH * (c0 * EFF_C + d0)) * INTERVAL_H
            state[src] = float(np.clip(state[src] + (c0 * EFF_C - d0) * INTERVAL_H, 0.0, CAPACITY_KWH))
            pnl_totals[src] += pnl
            executed_steps[src] += 1
            raw_rows.append({
                "time": ts,
                "source": src,
                "tactical_source": contract.tactical_source,
                "strategic_source": contract.strategic_source,
                "economic_mode": economic_mode,
                "actual_price_mwh": actual,
                "actual_general_price_mwh": actual_general_price_mwh,
                "actual_feed_in_price_mwh": actual_feed_in_price_mwh,
                "actual_net_load_kw": actual_net_load_step_kw,
                "actual_load_kw": actual_load_step_kw,
                "actual_pv_kw": actual_pv_step_kw,
                "forecast_step0_mwh": float(curve[0]),
                "forecast_mean_next_1h_mwh": forecast_mean_next_1h_mwh,
                "forecast_mean_next_4h_mwh": forecast_mean_next_4h_mwh,
                "forecast_mean_next_14h_mwh": forecast_mean_next_14h_mwh,
                "forecast_buy_step0_mwh": float(buy_curve[0]),
                "forecast_buy_mean_next_1h_mwh": forecast_buy_mean_next_1h_mwh,
                "forecast_buy_mean_next_4h_mwh": forecast_buy_mean_next_4h_mwh,
                "forecast_buy_mean_next_14h_mwh": forecast_buy_mean_next_14h_mwh,
                "forecast_sell_step0_mwh": float(sell_curve[0]),
                "forecast_sell_mean_next_1h_mwh": forecast_sell_mean_next_1h_mwh,
                "forecast_sell_mean_next_4h_mwh": forecast_sell_mean_next_4h_mwh,
                "forecast_sell_mean_next_14h_mwh": forecast_sell_mean_next_14h_mwh,
                "forecast_general_step0_mwh": (
                    float(forecast_general_price_mwh[0]) if forecast_general_price_mwh is not None else float("nan")
                ),
                "forecast_feed_in_step0_mwh": (
                    float(forecast_feed_in_price_mwh[0]) if forecast_feed_in_price_mwh is not None else float("nan")
                ),
                "forecast_feed_in_mean_next_1h_mwh": forecast_feed_in_mean_next_1h_mwh,
                "forecast_feed_in_mean_next_4h_mwh": forecast_feed_in_mean_next_4h_mwh,
                "forecast_feed_in_mean_next_14h_mwh": forecast_feed_in_mean_next_14h_mwh,
                "charge_kw": c0,
                "discharge_kw": d0,
                "grid_import_kw": grid_import0,
                "grid_export_kw": grid_export0,
                "curtail_kw": curtail0,
                "realized_grid_import_kw": realized_grid_import_kw,
                "realized_grid_export_kw": realized_grid_export_kw,
                "soc_prev_kwh": soc_prev,
                "soc_kwh": state[src],
                "step_pnl": pnl,
                "terminal_energy_value_per_kwh": applied_terminal_energy_value_per_kwh,
                "strategic_soc_target_kwh": strategic_soc_target_kwh,
                "strategic_soc_target_qhi_kwh": strategic_soc_target_qhi_kwh,
                "strategic_target_gap_kwh": strategic_target_gap_kwh,
                "dynamic_target_uplift_kwh": dynamic_target_uplift_kwh,
                "strategic_shadow_q50_per_kwh": strategic_shadow_q50_per_kwh,
                "strategic_shadow_qhi_per_kwh": strategic_shadow_qhi_per_kwh,
                "strategic_shadow_gap_per_kwh": strategic_shadow_gap_per_kwh,
                "dynamic_terminal_adder_per_kwh": dynamic_terminal_adder_per_kwh,
                "extra_terminal_energy_value_per_kwh": extra_terminal_energy_value_per_kwh,
                "extra_terminal_energy_floor_kwh": extra_terminal_energy_floor_kwh,
                "extra_terminal_energy_cap_kwh": extra_terminal_energy_cap_kwh,
                "extra_terminal_energy_kwh": float(solve.get("extra_terminal_energy_kwh", 0.0)),
                "min_terminal_soc_kwh": min_terminal_soc_kwh,
                "max_terminal_soc_kwh": max_terminal_soc_kwh,
                "tier1_quantile_blend": providers.args.tier1_quantile_blend,
                "tier2_quantile_blend": providers.args.tier2_quantile_blend,
                "buy_curve_quantile_blend": providers.args.buy_curve_quantile_blend,
                "sell_curve_quantile_blend": providers.args.sell_curve_quantile_blend,
                "sell_urgency_applied": sell_urgency_applied,
                "sell_urgency_trigger_feed_in_price_mwh": providers.args.sell_urgency_trigger_feed_in_price_mwh,
                "sell_urgency_discount": providers.args.sell_urgency_discount,
                "sell_urgency_horizon_steps": providers.args.sell_urgency_horizon_steps,
                "sell_urgency_max_strategic_target_kwh": providers.args.sell_urgency_max_strategic_target_kwh,
                "sell_urgency_trigger_only_shaping": providers.args.sell_urgency_trigger_only_shaping,
                "inventory_discipline_applied": inventory_discipline_applied,
                "inventory_discipline_cycle_cost_adder_per_kwh": inventory_discipline_cycle_cost_adder_per_kwh,
                "inventory_discipline_feed_in_max_mwh": providers.args.inventory_discipline_feed_in_max_mwh,
                "inventory_discipline_net_load_max_kw": providers.args.inventory_discipline_net_load_max_kw,
                "grid_exchange_reduction_applied": grid_exchange_reduction_applied,
                "grid_exchange_reduction_score": grid_exchange_reduction_score,
                "grid_exchange_reduction_min_score": providers.args.grid_exchange_reduction_min_score,
                "grid_exchange_reduction_cycle_cost_adder_per_kwh": (
                    grid_exchange_reduction_cycle_cost_adder_per_kwh
                ),
                "grid_exchange_reduction_flow_cost_adder_per_kwh": (
                    grid_exchange_reduction_flow_cost_adder_per_kwh
                ),
                "grid_exchange_reduction_horizon_steps": providers.args.grid_exchange_reduction_horizon_steps,
                "probe_initial_soc_shadow_price_per_kwh": probe_shadow_price_per_kwh,
                "control_initial_soc_shadow_price_per_kwh": solve["initial_soc_shadow_price_per_kwh"],
            })

        if progress_every_steps > 0 and ((i + 1) % progress_every_steps == 0 or (i + 1) == steps_total):
            elapsed_seconds = time.time() - started_at
            remaining_steps = max(0, steps_total - (i + 1))
            seconds_per_step = elapsed_seconds / max(i + 1, 1)
            eta_seconds = remaining_steps * seconds_per_step
            eta_ts = pd.Timestamp.now(tz="UTC") + pd.to_timedelta(eta_seconds, unit="s")
            print(
                f"  {i+1}/{steps_total} steps ({100.0 * (i+1) / max(steps_total, 1):.1f}%) "
                f"sim={timestamps[i]} elapsed={_format_duration(elapsed_seconds)} "
                f"eta={_format_duration(eta_seconds)} finish={eta_ts.isoformat()}",
                flush=True,
            )
            if len(source_contracts) == 1:
                src = source_contracts[0].name
                progress_path = source_progress_paths[src]
                if progress_path is not None:
                    _write_progress_checkpoint(
                        progress_path,
                        source=src,
                        steps_completed=i + 1,
                        steps_total=steps_total,
                        elapsed_seconds=elapsed_seconds,
                        current_sim_time=timestamps[i],
                        completed=(i + 1) == steps_total,
                    )

    n_days = max((timestamps.max() - timestamps.min()).total_seconds() / 86400.0, 1e-9)
    for contract in source_contracts:
        src = contract.name
        src_rows = [row for row in raw_rows if row["source"] == src]
        src_df = pd.DataFrame(src_rows)
        mean_dynamic_target_uplift_kwh = float(src_df["dynamic_target_uplift_kwh"].mean()) if not src_df.empty else 0.0
        max_dynamic_target_uplift_kwh = float(src_df["dynamic_target_uplift_kwh"].max()) if not src_df.empty else 0.0
        mean_dynamic_terminal_adder_per_kwh = float(src_df["dynamic_terminal_adder_per_kwh"].mean()) if not src_df.empty else 0.0
        max_dynamic_terminal_adder_per_kwh = float(src_df["dynamic_terminal_adder_per_kwh"].max()) if not src_df.empty else 0.0
        mean_strategic_target_gap_kwh = float(src_df["strategic_target_gap_kwh"].mean()) if not src_df.empty else 0.0
        max_strategic_target_gap_kwh = float(src_df["strategic_target_gap_kwh"].max()) if not src_df.empty else 0.0
        mean_strategic_shadow_gap_per_kwh = float(src_df["strategic_shadow_gap_per_kwh"].mean()) if not src_df.empty else 0.0
        max_strategic_shadow_gap_per_kwh = float(src_df["strategic_shadow_gap_per_kwh"].max()) if not src_df.empty else 0.0
        positive_target_uplift_steps = int((src_df["dynamic_target_uplift_kwh"] > 0).sum()) if not src_df.empty else 0
        positive_terminal_adder_steps = int((src_df["dynamic_terminal_adder_per_kwh"] > 0).sum()) if not src_df.empty else 0
        inventory_discipline_steps = (
            int(src_df["inventory_discipline_applied"].sum())
            if not src_df.empty and "inventory_discipline_applied" in src_df
            else 0
        )
        mean_inventory_discipline_cycle_cost_adder_per_kwh = (
            float(src_df["inventory_discipline_cycle_cost_adder_per_kwh"].mean())
            if not src_df.empty and "inventory_discipline_cycle_cost_adder_per_kwh" in src_df
            else 0.0
        )
        max_inventory_discipline_cycle_cost_adder_per_kwh = (
            float(src_df["inventory_discipline_cycle_cost_adder_per_kwh"].max())
            if not src_df.empty and "inventory_discipline_cycle_cost_adder_per_kwh" in src_df
            else 0.0
        )
        grid_exchange_reduction_steps = (
            int(src_df["grid_exchange_reduction_applied"].sum())
            if not src_df.empty and "grid_exchange_reduction_applied" in src_df
            else 0
        )
        mean_grid_exchange_reduction_cycle_cost_adder_per_kwh = (
            float(src_df["grid_exchange_reduction_cycle_cost_adder_per_kwh"].mean())
            if not src_df.empty and "grid_exchange_reduction_cycle_cost_adder_per_kwh" in src_df
            else 0.0
        )
        max_grid_exchange_reduction_cycle_cost_adder_per_kwh = (
            float(src_df["grid_exchange_reduction_cycle_cost_adder_per_kwh"].max())
            if not src_df.empty and "grid_exchange_reduction_cycle_cost_adder_per_kwh" in src_df
            else 0.0
        )
        mean_grid_exchange_reduction_flow_cost_adder_per_kwh = (
            float(src_df["grid_exchange_reduction_flow_cost_adder_per_kwh"].mean())
            if not src_df.empty and "grid_exchange_reduction_flow_cost_adder_per_kwh" in src_df
            else 0.0
        )
        max_grid_exchange_reduction_flow_cost_adder_per_kwh = (
            float(src_df["grid_exchange_reduction_flow_cost_adder_per_kwh"].max())
            if not src_df.empty and "grid_exchange_reduction_flow_cost_adder_per_kwh" in src_df
            else 0.0
        )
        mean_grid_exchange_reduction_score = (
            float(src_df.loc[src_df["grid_exchange_reduction_applied"], "grid_exchange_reduction_score"].mean())
            if (
                not src_df.empty
                and "grid_exchange_reduction_applied" in src_df
                and bool(src_df["grid_exchange_reduction_applied"].any())
            )
            else float("nan")
        )
        summary.append({
            "source": src,
            "tactical_source": contract.tactical_source,
            "strategic_source": contract.strategic_source,
            "steps": executed_steps[src],
            "total_pnl": pnl_totals[src],
            "mean_per_day": pnl_totals[src] / n_days,
            "soc_final_kwh": state[src],
            "economic_mode": economic_mode,
            "terminal_energy_value_per_kwh": terminal_energy_value_per_kwh,
            "dual_terminal_scale": dual_terminal_scale,
            "strategic_soc_handoff": strategic_soc_handoff,
            "strategic_target_mode": strategic_target_mode if strategic_soc_handoff else "",
            "dynamic_bridge_upper_quantile": dynamic_bridge_upper_quantile,
            "dynamic_bridge_target_scale": dynamic_bridge_target_scale,
            "dynamic_bridge_terminal_scale": dynamic_bridge_terminal_scale,
            "dynamic_bridge_terminal_scope": dynamic_bridge_terminal_scope,
            "mean_dynamic_target_uplift_kwh": mean_dynamic_target_uplift_kwh,
            "max_dynamic_target_uplift_kwh": max_dynamic_target_uplift_kwh,
            "mean_dynamic_terminal_adder_per_kwh": mean_dynamic_terminal_adder_per_kwh,
            "max_dynamic_terminal_adder_per_kwh": max_dynamic_terminal_adder_per_kwh,
            "mean_strategic_target_gap_kwh": mean_strategic_target_gap_kwh,
            "max_strategic_target_gap_kwh": max_strategic_target_gap_kwh,
            "mean_strategic_shadow_gap_per_kwh": mean_strategic_shadow_gap_per_kwh,
            "max_strategic_shadow_gap_per_kwh": max_strategic_shadow_gap_per_kwh,
            "positive_target_uplift_steps": positive_target_uplift_steps,
            "positive_terminal_adder_steps": positive_terminal_adder_steps,
            "inventory_discipline_steps": inventory_discipline_steps,
            "mean_inventory_discipline_cycle_cost_adder_per_kwh": mean_inventory_discipline_cycle_cost_adder_per_kwh,
            "max_inventory_discipline_cycle_cost_adder_per_kwh": max_inventory_discipline_cycle_cost_adder_per_kwh,
            "grid_exchange_reduction_steps": grid_exchange_reduction_steps,
            "mean_grid_exchange_reduction_score": mean_grid_exchange_reduction_score,
            "mean_grid_exchange_reduction_cycle_cost_adder_per_kwh": (
                mean_grid_exchange_reduction_cycle_cost_adder_per_kwh
            ),
            "max_grid_exchange_reduction_cycle_cost_adder_per_kwh": (
                max_grid_exchange_reduction_cycle_cost_adder_per_kwh
            ),
            "mean_grid_exchange_reduction_flow_cost_adder_per_kwh": (
                mean_grid_exchange_reduction_flow_cost_adder_per_kwh
            ),
            "max_grid_exchange_reduction_flow_cost_adder_per_kwh": (
                max_grid_exchange_reduction_flow_cost_adder_per_kwh
            ),
            "tier1_quantile_blend": providers.args.tier1_quantile_blend,
            "tier2_quantile_blend": providers.args.tier2_quantile_blend,
            "tier2_upper_quantile": providers.args.tier2_upper_quantile,
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
    source_contract: SourceContract,
    start: pd.Timestamp,
    end: pd.Timestamp,
    soc_init: float,
    args_dict: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    worker_label = mp.current_process().name
    print(
        f"  [{worker_label}] Starting source '{source_contract.name}' "
        f"(tactical={source_contract.tactical_source}, strategic={source_contract.strategic_source})",
        flush=True,
    )
    timestamps = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
    provider_t0 = time.time()
    providers = ForecastProviders(argparse.Namespace(**args_dict))
    provider_elapsed = time.time() - provider_t0
    actual_prices = providers.actuals_5m["rrp"].reindex(timestamps)
    print(
        f"  [{worker_label}] Provider init complete for '{source_contract.name}' in {provider_elapsed:.1f}s "
        f"({len(timestamps):,} steps)",
        flush=True,
    )
    return simulate_stepwise(
        timestamps=timestamps,
        actual_prices=actual_prices,
        providers=providers,
        source_contracts=[source_contract],
        soc_init=soc_init,
        economic_mode=args_dict["economic_mode"],
        output_prefix=args_dict["output_prefix"],
        progress_every_steps=args_dict["progress_every_steps"],
        terminal_energy_value_per_kwh=args_dict["terminal_energy_value_per_kwh"],
        dual_terminal_scale=args_dict["dual_terminal_scale"],
        strategic_soc_handoff=args_dict["strategic_soc_handoff"],
        strategic_target_mode=args_dict["strategic_target_mode"],
        dynamic_bridge_upper_quantile=args_dict["dynamic_bridge_upper_quantile"],
        dynamic_bridge_target_scale=args_dict["dynamic_bridge_target_scale"],
        dynamic_bridge_terminal_scale=args_dict["dynamic_bridge_terminal_scale"],
        dynamic_bridge_terminal_scope=args_dict["dynamic_bridge_terminal_scope"],
    )


def simulate_stepwise_parallel(
    start: pd.Timestamp,
    end: pd.Timestamp,
    source_contracts: list[SourceContract],
    soc_init: float,
    args_dict: dict,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if workers <= 1 or len(source_contracts) <= 1:
        providers = ForecastProviders(argparse.Namespace(**args_dict))
        timestamps = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
        actual_prices = providers.actuals_5m["rrp"].reindex(timestamps)
        return simulate_stepwise(
            timestamps,
            actual_prices,
            providers,
            source_contracts,
            soc_init,
            economic_mode=args_dict["economic_mode"],
            output_prefix=args_dict["output_prefix"],
            progress_every_steps=args_dict["progress_every_steps"],
            terminal_energy_value_per_kwh=args_dict["terminal_energy_value_per_kwh"],
            dual_terminal_scale=args_dict["dual_terminal_scale"],
            strategic_soc_handoff=args_dict["strategic_soc_handoff"],
            strategic_target_mode=args_dict["strategic_target_mode"],
            dynamic_bridge_upper_quantile=args_dict["dynamic_bridge_upper_quantile"],
            dynamic_bridge_target_scale=args_dict["dynamic_bridge_target_scale"],
            dynamic_bridge_terminal_scale=args_dict["dynamic_bridge_terminal_scale"],
            dynamic_bridge_terminal_scope=args_dict["dynamic_bridge_terminal_scope"],
        )

    raw_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    start_method = _resolve_mp_start_method(str(args_dict.get("mp_start_method", "auto")))
    print(
        f"  Parallel execution across {len(source_contracts)} source(s) with {workers} worker(s) "
        f"using start method '{start_method}'",
        flush=True,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context(start_method),
    ) as ex:
        futures = {
            ex.submit(
                _simulate_single_source,
                source_contract,
                start,
                end,
                soc_init,
                args_dict,
            ): source_contract
            for source_contract in source_contracts
        }
        for fut in concurrent.futures.as_completed(futures):
            source_contract = futures[fut]
            raw_df, summary_df = fut.result()
            print(f"  Completed source: {source_contract.name}")
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
        help=(
            "Comma-separated source labels. Built-ins: p5min_naive, model_a_hybrid, amber_apf_lgbm, "
            "hybrid_tactical_amber_strategic, amber_tactical_hybrid_strategic. "
            "Generic counterfactual form: cf:<label>:<tactical_source>:<strategic_source>."
        ),
    )
    parser.add_argument("--tft-checkpoint", default="", help="Path to TFT checkpoint for model_a_hybrid")
    parser.add_argument("--tft-scalers", default="", help="Path to matching TFT scalers for model_a_hybrid")
    parser.add_argument(
        "--tactical-model-dir",
        default=str(DEFAULT_TACTICAL_MODEL_DIR),
        help="Directory containing Tier 1 tactical LightGBM artifacts.",
    )
    parser.add_argument(
        "--economic-mode",
        choices=["price_only", "netload_tariffed"],
        default="price_only",
        help=(
            "Dispatch/PnL objective. 'price_only' keeps the legacy wholesale arbitrage eval; "
            "'netload_tariffed' adds actual load/PV plus separate import/feed-in economics."
        ),
    )
    parser.add_argument("--soc-init-kwh", type=float, default=SOC_INIT_KWH)
    parser.add_argument(
        "--tier1-quantile-blend",
        type=float,
        default=0.0,
        help=(
            "Blend weight for the first-hour tactical path: effective = q50 + w*(q95-q50). "
            "Range 0..1."
        ),
    )
    parser.add_argument(
        "--tier2-quantile-blend",
        type=float,
        default=0.0,
        help=(
            "Blend weight for the 1h..14h strategic extension in model_a_hybrid: effective = "
            "q50 + w*(q_hi-q50). Range 0..1."
        ),
    )
    parser.add_argument(
        "--tier2-upper-quantile",
        type=float,
        default=0.90,
        choices=[0.90, 0.95],
        help="Upper TFT quantile used for the Tier 2 strategic-extension blend.",
    )
    parser.add_argument(
        "--buy-curve-quantile-blend",
        type=float,
        default=0.0,
        help=(
            "Eval-only split-curve mode: symmetric lower-tail blend weight for import decisions. "
            "0 keeps the buy curve at q50; positive values move it lower using (q50 - qhi gap)."
        ),
    )
    parser.add_argument(
        "--sell-curve-quantile-blend",
        type=float,
        default=0.0,
        help=(
            "Eval-only split-curve mode: upper-tail blend weight for export decisions. "
            "0 keeps the sell curve at q50; positive values move it toward qhi."
        ),
    )
    parser.add_argument(
        "--split-curve-tactical-sources",
        default="",
        help=(
            "Comma-separated tactical source labels eligible for eval-only buy/sell split curves. "
            "Example: model_a_hybrid"
        ),
    )
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
    parser.add_argument(
        "--strategic-soc-handoff",
        action="store_true",
        help=(
            "Derive a 14h terminal SoC target from the source's longer strategic curve and "
            "pass it into the tactical 14h solve."
        ),
    )
    parser.add_argument(
        "--strategic-target-mode",
        choices=["exact", "floor", "band"],
        default="exact",
        help=(
            "How to enforce the derived 14h strategic SoC target: exact terminal target or "
            "minimum terminal floor. 'band' sets a minimum q50 target with optional dynamic "
            "upward headroom derived from the upper strategic quantile."
        ),
    )
    parser.add_argument(
        "--dynamic-bridge-upper-quantile",
        type=float,
        default=0.90,
        choices=[0.90, 0.95],
        help="Upper TFT quantile used to derive the dynamic bridge posture signal.",
    )
    parser.add_argument(
        "--dynamic-bridge-target-scale",
        type=float,
        default=0.0,
        help=(
            "Scale applied to max(0, target_qhi - target_q50) to widen the tactical terminal "
            "contract. For 'floor' mode this lifts the minimum terminal SoC; for 'band' mode "
            "it sets upward headroom above the q50 target."
        ),
    )
    parser.add_argument(
        "--dynamic-bridge-terminal-scale",
        type=float,
        default=0.0,
        help=(
            "Scale applied to max(0, strategic_shadow_qhi - strategic_shadow_q50) to add a "
            "bounded dynamic terminal-energy value during the tactical solve."
        ),
    )
    parser.add_argument(
        "--dynamic-bridge-terminal-scope",
        choices=["all", "extra_band"],
        default="all",
        help=(
            "How dynamic terminal value is applied. 'all' values all terminal energy; "
            "'extra_band' only values the energy above the q50 floor within band mode."
        ),
    )
    parser.add_argument(
        "--sell-urgency-trigger-feed-in-price-mwh",
        type=float,
        default=float("inf"),
        help=(
            "If finite, enable a tactical sell-urgency stress test in netload_tariffed mode: "
            "when the current actual feed-in price is at or above this threshold, selected "
            "tactical sources have their future export-price curve discounted after step 0."
        ),
    )
    parser.add_argument(
        "--sell-urgency-discount",
        type=float,
        default=0.0,
        help=(
            "Fractional discount applied to future export prices after step 0 when the "
            "sell-urgency trigger fires. Example: 0.25 discounts by 25%%."
        ),
    )
    parser.add_argument(
        "--sell-urgency-horizon-hours",
        type=float,
        default=0.0,
        help=(
            "How far after step 0 the sell-urgency discount applies, in hours. "
            "Typical probes are 1 or 4."
        ),
    )
    parser.add_argument(
        "--sell-urgency-max-strategic-target-kwh",
        type=float,
        default=float("inf"),
        help=(
            "Maximum strategic target SoC for the sell-urgency transform to apply. "
            "Use a low value (for example 0 or 2) to focus on low-reserve intervals."
        ),
    )
    parser.add_argument(
        "--sell-urgency-tactical-sources",
        default="",
        help=(
            "Comma-separated tactical source labels eligible for the sell-urgency transform. "
            "Example: model_a_hybrid"
        ),
    )
    parser.add_argument(
        "--sell-urgency-trigger-only-shaping",
        action="store_true",
        help=(
            "When set, keep the sell/export curve exactly at the unshaped tactical baseline "
            "outside triggered urgency intervals. During triggered intervals, apply the "
            "configured sell-side reshaping and urgency discount."
        ),
    )
    parser.add_argument(
        "--inventory-discipline-cycle-cost-mwh",
        type=float,
        default=0.0,
        help=(
            "Eval-only churn guard: add this $/MWh throughput friction to selected source "
            "LP control solves when the inventory-discipline regime gate is active. "
            "Realized P&L degradation accounting is unchanged."
        ),
    )
    parser.add_argument(
        "--inventory-discipline-sources",
        default="",
        help=(
            "Comma-separated source labels eligible for the inventory-discipline cycle-cost "
            "adder. Example: model_a_hybrid_inventory_bias"
        ),
    )
    parser.add_argument(
        "--inventory-discipline-feed-in-max-mwh",
        type=float,
        default=300.0,
        help="Inventory-discipline gate applies only when current actual feed-in price is below this $/MWh value.",
    )
    parser.add_argument(
        "--inventory-discipline-net-load-max-kw",
        type=float,
        default=0.0,
        help="Inventory-discipline gate applies only when current actual net load is below this kW value.",
    )
    parser.add_argument(
        "--grid-exchange-reduction-signal-file",
        default="",
        help=(
            "Eval-only event gate: CSV/parquet with time and score columns, typically from "
            "train_state_transition_direction_model.py predictions."
        ),
    )
    parser.add_argument(
        "--grid-exchange-reduction-sources",
        default="",
        help=(
            "Comma-separated source labels eligible for event-gated grid-exchange reduction. "
            "Example: model_a_hybrid_grid_exchange_gate"
        ),
    )
    parser.add_argument(
        "--grid-exchange-reduction-cycle-cost-mwh",
        type=float,
        default=0.0,
        help=(
            "Eval-only event-gated churn guard, in $/MWh, applied only when the external "
            "grid-exchange-reduction score is above the threshold."
        ),
    )
    parser.add_argument(
        "--grid-exchange-reduction-flow-cost-mwh",
        type=float,
        default=0.0,
        help=(
            "Eval-only event-gated import/export flow guard, in $/MWh, applied directly to "
            "grid import and grid export when the external grid-exchange-reduction score "
            "is above the threshold."
        ),
    )
    parser.add_argument(
        "--grid-exchange-reduction-min-score",
        type=float,
        default=0.8,
        help="Minimum external event score required for the grid-exchange-reduction gate.",
    )
    parser.add_argument(
        "--grid-exchange-reduction-label",
        default="grid_exchange_down",
        help="Label value to select from the signal file when a label column is present.",
    )
    parser.add_argument(
        "--grid-exchange-reduction-score-col",
        default="y_score",
        help="Score column in the signal file.",
    )
    parser.add_argument(
        "--grid-exchange-reduction-horizon-steps",
        type=int,
        default=12,
        help="Optional horizon_steps value to select from the signal file. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--grid-exchange-reduction-split",
        default="",
        help="Optional split value to select from the signal file when a split column is present.",
    )
    parser.add_argument("--output-prefix", default="rolling_mpc_eval_model_a")
    parser.add_argument(
        "--progress-every-steps",
        type=int,
        default=72,
        help=(
            "Progress logging/checkpoint cadence in simulated 5-minute steps. "
            "Use 0 to disable interim progress updates."
        ),
    )
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
    parser.add_argument(
        "--mp-start-method",
        choices=["auto", "fork", "forkserver", "spawn"],
        default="auto",
        help=(
            "Multiprocessing start method when --workers > 1. "
            "'auto' prefers fork, then forkserver, then spawn."
        ),
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    source_specs = [s.strip() for s in args.sources.split(",") if s.strip()]
    try:
        source_contracts = _parse_source_contracts(source_specs)
    except ValueError as exc:
        parser.error(str(exc))
    source_names = [contract.name for contract in source_contracts]
    baseline_source = args.baseline_source.strip() or source_names[0]
    args.terminal_energy_value_per_kwh = args.terminal_energy_value_mwh / 1000.0
    if not 0.0 <= args.tier1_quantile_blend <= 1.0:
        parser.error("--tier1-quantile-blend must be in [0, 1]")
    if not 0.0 <= args.tier2_quantile_blend <= 1.0:
        parser.error("--tier2-quantile-blend must be in [0, 1]")
    if args.buy_curve_quantile_blend < 0.0:
        parser.error("--buy-curve-quantile-blend must be >= 0")
    if args.sell_curve_quantile_blend < 0.0:
        parser.error("--sell-curve-quantile-blend must be >= 0")
    if args.dual_terminal_scale < 0:
        parser.error("--dual-terminal-scale must be >= 0")
    if not 0.0 <= args.sell_urgency_discount <= 1.0:
        parser.error("--sell-urgency-discount must be in [0, 1]")
    if args.sell_urgency_horizon_hours < 0.0:
        parser.error("--sell-urgency-horizon-hours must be >= 0")
    args.sell_urgency_horizon_steps = int(round(args.sell_urgency_horizon_hours * 12))
    args.sell_urgency_tactical_sources = _parse_csv_set(args.sell_urgency_tactical_sources)
    args.split_curve_tactical_sources = _parse_csv_set(args.split_curve_tactical_sources)
    args.inventory_discipline_sources = _parse_csv_set(args.inventory_discipline_sources)
    args.inventory_discipline_cycle_cost_per_kwh = args.inventory_discipline_cycle_cost_mwh / 1000.0
    args.grid_exchange_reduction_sources = _parse_csv_set(args.grid_exchange_reduction_sources)
    args.grid_exchange_reduction_cycle_cost_per_kwh = args.grid_exchange_reduction_cycle_cost_mwh / 1000.0
    args.grid_exchange_reduction_flow_cost_per_kwh = args.grid_exchange_reduction_flow_cost_mwh / 1000.0
    args.grid_exchange_reduction_horizon_steps = (
        None if args.grid_exchange_reduction_horizon_steps <= 0 else args.grid_exchange_reduction_horizon_steps
    )
    args.grid_exchange_reduction_scores_by_time = _load_grid_exchange_reduction_signal(
        args.grid_exchange_reduction_signal_file,
        label=args.grid_exchange_reduction_label,
        score_col=args.grid_exchange_reduction_score_col,
        min_score=args.grid_exchange_reduction_min_score,
        horizon_steps=args.grid_exchange_reduction_horizon_steps,
        split=args.grid_exchange_reduction_split,
    )
    if args.inventory_discipline_cycle_cost_mwh < 0.0:
        parser.error("--inventory-discipline-cycle-cost-mwh must be >= 0")
    if args.inventory_discipline_sources and args.economic_mode != "netload_tariffed":
        parser.error("--inventory-discipline-sources currently requires --economic-mode netload_tariffed")
    if args.grid_exchange_reduction_cycle_cost_mwh < 0.0:
        parser.error("--grid-exchange-reduction-cycle-cost-mwh must be >= 0")
    if args.grid_exchange_reduction_flow_cost_mwh < 0.0:
        parser.error("--grid-exchange-reduction-flow-cost-mwh must be >= 0")
    if not 0.0 <= args.grid_exchange_reduction_min_score <= 1.0:
        parser.error("--grid-exchange-reduction-min-score must be in [0, 1]")
    if args.grid_exchange_reduction_sources and args.economic_mode != "netload_tariffed":
        parser.error("--grid-exchange-reduction-sources currently requires --economic-mode netload_tariffed")
    if args.grid_exchange_reduction_sources and not args.grid_exchange_reduction_signal_file:
        parser.error("--grid-exchange-reduction-sources requires --grid-exchange-reduction-signal-file")
    if (
        args.grid_exchange_reduction_sources
        and args.grid_exchange_reduction_cycle_cost_mwh <= 0.0
        and args.grid_exchange_reduction_flow_cost_mwh <= 0.0
    ):
        parser.error(
            "--grid-exchange-reduction-sources requires either "
            "--grid-exchange-reduction-cycle-cost-mwh > 0 or "
            "--grid-exchange-reduction-flow-cost-mwh > 0"
        )
    if args.dual_terminal_scale > 0 and args.terminal_energy_value_mwh != 0.0:
        parser.error("Use either --terminal-energy-value-mwh or --dual-terminal-scale, not both")
    if args.dynamic_bridge_target_scale < 0:
        parser.error("--dynamic-bridge-target-scale must be >= 0")
    if args.dynamic_bridge_terminal_scale < 0:
        parser.error("--dynamic-bridge-terminal-scale must be >= 0")
    if args.dynamic_bridge_terminal_scope == "extra_band" and args.strategic_target_mode != "band":
        parser.error("--dynamic-bridge-terminal-scope=extra_band requires --strategic-target-mode band")
    if args.dual_terminal_scale > 0 and args.dynamic_bridge_terminal_scale > 0:
        parser.error("Use either --dual-terminal-scale or --dynamic-bridge-terminal-scale, not both")
    if args.terminal_energy_value_mwh != 0.0 and args.dynamic_bridge_terminal_scale > 0:
        parser.error("Use either --terminal-energy-value-mwh or --dynamic-bridge-terminal-scale, not both")
    if args.strategic_target_mode and not args.strategic_soc_handoff and args.strategic_target_mode != "exact":
        parser.error("--strategic-target-mode only applies with --strategic-soc-handoff")
    if (
        (args.dynamic_bridge_target_scale > 0 or args.dynamic_bridge_terminal_scale > 0)
        and not args.strategic_soc_handoff
    ):
        parser.error("Dynamic bridge options require --strategic-soc-handoff")

    if any(
        "model_a_hybrid" in (contract.tactical_source, contract.strategic_source)
        for contract in source_contracts
    ) and (not args.tft_checkpoint or not args.tft_scalers):
        parser.error("model_a_hybrid requires --tft-checkpoint and --tft-scalers")
    if baseline_source not in source_names:
        parser.error("--baseline-source must be one of --sources")

    idx = pd.date_range(start=start, end=end - pd.Timedelta(minutes=5), freq="5min", tz="UTC")
    workers = args.workers or min(len(source_contracts), max(1, (os.cpu_count() or 1)))
    workers = max(1, min(workers, len(source_contracts)))
    args_dict = vars(args).copy()

    print(f"Rolling MPC eval — Model A track")
    print(f"  Window: {start} → {end}")
    print(f"  Steps:  {len(idx):,}")
    print(f"  Sources: {source_names}")
    for contract in source_contracts:
        if contract.tactical_source != contract.strategic_source:
            print(
                f"    {contract.name}: tactical={contract.tactical_source}, "
                f"strategic={contract.strategic_source}"
            )
    print(f"  Baseline: {baseline_source}")
    print(f"  Economic mode: {args.economic_mode}")
    print(f"  Terminal energy value: {args.terminal_energy_value_mwh:.3f} $/MWh")
    print(f"  Dual terminal scale: {args.dual_terminal_scale:.3f}")
    print(f"  Strategic SoC handoff: {args.strategic_soc_handoff} ({args.strategic_target_mode})")
    print(
        f"  Dynamic bridge: qhi=q{int(args.dynamic_bridge_upper_quantile * 100):02d}, "
        f"target_scale={args.dynamic_bridge_target_scale:.3f}, "
        f"terminal_scale={args.dynamic_bridge_terminal_scale:.3f}, "
        f"terminal_scope={args.dynamic_bridge_terminal_scope}"
    )
    print(
        f"  Quantile blend: tier1={args.tier1_quantile_blend:.2f}, "
        f"tier2={args.tier2_quantile_blend:.2f}, tier2_qhi=q{int(args.tier2_upper_quantile * 100):02d}"
    )
    print(
        f"  Inventory discipline: sources={sorted(args.inventory_discipline_sources)}, "
        f"cycle_cost={args.inventory_discipline_cycle_cost_mwh:.3f} $/MWh, "
        f"feed_in_max={args.inventory_discipline_feed_in_max_mwh:.3f} $/MWh, "
        f"net_load_max={args.inventory_discipline_net_load_max_kw:.3f} kW"
    )
    print(
        f"  Grid-exchange reduction: sources={sorted(args.grid_exchange_reduction_sources)}, "
        f"cycle_cost={args.grid_exchange_reduction_cycle_cost_mwh:.3f} $/MWh, "
        f"flow_cost={args.grid_exchange_reduction_flow_cost_mwh:.3f} $/MWh, "
        f"min_score={args.grid_exchange_reduction_min_score:.3f}, "
        f"horizon={args.grid_exchange_reduction_horizon_steps}, "
        f"signals={len(args.grid_exchange_reduction_scores_by_time)}"
    )
    print(f"  Workers: {workers}")
    if workers > 1:
        print(f"  MP start method: {_resolve_mp_start_method(args.mp_start_method)}")

    t0 = time.time()
    raw_df, summary_df = simulate_stepwise_parallel(
        start=start,
        end=end,
        source_contracts=source_contracts,
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
    coverage_df = build_coverage_summary(raw_df, start, end, source_names, summary_df=summary_df)
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
