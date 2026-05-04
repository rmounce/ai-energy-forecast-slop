#!/usr/bin/env python3
"""
Phase 6 — Holistic dispatch simulation.

Runs rolling-MPC LP dispatch on a stratified eval set, comparing:
  oracle       — perfect foresight (actual prices as forecast)
  amber_apf_lgbm — as-run predictions from price_forecast_log.csv
                   (Amber APF seeds first ~14-28h, LightGBM extrapolates to 72h)
  p5min_naive  — window-start price held constant (naive persistence baseline)

Actual household load and PV are used as fixed inputs so that only price-forecast
quality drives differences between sources.

Performance design:
  - Bulk-fetch all price/load/PV data in 3 queries (not one per window)
  - Parallel LP solving via ProcessPoolExecutor
  - Per-stratum checkpointing: re-run skips already-completed strata

Outputs:
  eval/results/holistic_eval_raw_{stratum}.parquet — per-stratum checkpoints
  eval/results/holistic_eval_raw.parquet           — merged results
  eval/results/holistic_eval_results.csv           — $/day summary table

Usage:
    source .venv/bin/activate
    python eval/holistic_eval.py [--price-only] [--workers N] [--strata spike,low,normal]
"""

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR    = ROOT / "eval" / "results"
FORECAST_LOG   = ROOT / "price_forecast_log.csv"
INDEX_FILE     = RESULTS_DIR / "holistic_eval_index.parquet"
TFT_FCST_FILE      = RESULTS_DIR / "retro_tft_forecasts.pkl"
TIER1_FCST_FILE    = RESULTS_DIR / "retro_tier1_forecasts.pkl"
LGBM_STRAT_FILE    = RESULTS_DIR / "retro_lgbm_strategic_forecasts.pkl"
ACTUALS_FILE   = RESULTS_DIR / "holistic_eval_actuals.parquet"

WINDOW_STEPS = 144        # 72h at 30-min resolution
INTERVAL_H   = 30 / 60   # 30-min steps

SPIKE_THRESH = 300.0
LOW_THRESH   = -50.0


from config_utils import load_config


# ── Bulk InfluxDB fetch ───────────────────────────────────────────────────────

def query_bulk_30m(client, measurement, field, start_iso, end_iso) -> pd.Series:
    """Query a 30-min field for the full eval period. Returns UTC-indexed float64 Series."""
    q = (
        f'SELECT mean("{field}") AS val'
        f' FROM "rp_30m"."{measurement}"'
        f' WHERE time >= \'{start_iso}\' AND time < \'{end_iso}\''
        f' GROUP BY time(30m) fill(none)'
    )
    result = client.query(q)
    rows = list(result.get_points())
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")["val"].sort_index()


def load_frozen_actuals() -> dict | None:
    """
    Load frozen actuals snapshot from holistic_eval_actuals.parquet if it exists.
    Returns the same dict format as fetch_bulk_data(), or None if not found.
    Run eval/export_holistic_actuals.py to create the snapshot.
    """
    if not ACTUALS_FILE.exists():
        return None
    print(f"Loading frozen actuals from {ACTUALS_FILE.name} ...")
    t0 = time.time()
    df = pd.read_parquet(ACTUALS_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    idx = df.set_index("time")
    print(f"  {len(df):,} rows ({df['time'].min().date()} → {df['time'].max().date()}, {time.time()-t0:.1f}s)")
    return {
        "prices":   idx["price_mwh"],
        "load_kw":  idx["load_kw"],
        "pv_kw":    idx["pv_kw"],
    }


def fetch_bulk_data(client, start_iso: str, end_iso: str) -> dict:
    """Fetch price, load, and PV for the entire eval range in 3 queries."""
    print(f"Bulk fetching price/load/PV for {start_iso} → {end_iso}...")
    t0 = time.time()
    prices = query_bulk_30m(client, "aemo_dispatch_sa1_30m", "price", start_iso, end_iso)
    load   = query_bulk_30m(client, "power_load_30m",  "mean_value", start_iso, end_iso)
    pv     = query_bulk_30m(client, "power_pv_30m",    "mean_value", start_iso, end_iso)
    print(f"  Fetched {len(prices):,} price, {len(load):,} load, {len(pv):,} PV points in {time.time()-t0:.1f}s")
    return {"prices": prices, "load_kw": load / 1000.0, "pv_kw": pv / 1000.0}


def slice_window(bulk: dict, start_ts: pd.Timestamp) -> dict | None:
    """Slice a 72h window from pre-fetched bulk data. Returns None if insufficient."""
    idx = pd.date_range(start_ts, periods=WINDOW_STEPS, freq="30min", tz="UTC")

    prices  = bulk["prices"].reindex(idx).ffill().bfill()
    load_kw = bulk["load_kw"].reindex(idx).ffill().bfill().fillna(0.0)
    pv_kw   = bulk["pv_kw"].reindex(idx).ffill().bfill().fillna(0.0)

    # Require ≥80% of price steps to be non-NaN
    if prices.isna().sum() > WINDOW_STEPS * 0.20:
        return None

    prices_arr = prices.ffill().bfill().values.astype(np.float64)
    net_load   = (load_kw - pv_kw).values.astype(np.float64)

    return {"actual_prices_mwh": prices_arr, "net_load_kw": net_load}


# ── LightGBM forecast loading ─────────────────────────────────────────────────

def load_amber_lgbm_forecasts() -> dict:
    """
    Load as-run Amber APF + LGBM extrapolation forecasts from price_forecast_log.csv.
    These are the production predictions that ran live: Amber APF seeds the first ~14-28h,
    LightGBM extrapolates the remaining steps to 72h.
    Returns dict: floor-30min Timestamp → np.ndarray shape (144,) in $/MWh.
    """
    print("Loading Amber APF + LGBM forecast log...")
    t0 = time.time()
    df = pd.read_csv(
        FORECAST_LOG,
        usecols=["forecast_creation_time", "forecast_target_time", "prediction"],
        dtype_backend="pyarrow",
    )
    df["forecast_target_time"] = pd.to_datetime(
        df["forecast_target_time"], utc=True, format="mixed"
    )
    grouped = df.groupby("forecast_creation_time")
    forecasts = {}
    for creation_time, grp in grouped:
        grp = grp.sort_values("forecast_target_time")
        start_ts = grp["forecast_target_time"].iloc[0].floor("30min")
        if len(grp) < WINDOW_STEPS:
            continue
        preds_mwh = grp["prediction"].values[:WINDOW_STEPS].astype(np.float64) * 1000.0
        forecasts[start_ts] = preds_mwh
    print(f"  Loaded {len(forecasts):,} LightGBM runs in {time.time()-t0:.1f}s")
    return forecasts


def load_tft_forecasts() -> dict:
    """
    Load retrospective TFT forecasts from retro_tft_forecasts.pkl.
    Returns dict: UTC Timestamp → np.ndarray shape (144,) q50 in $/MWh.
    Returns empty dict if file not found.
    """
    if not TFT_FCST_FILE.exists():
        return {}
    print("Loading TFT forecast pickle...")
    t0 = time.time()
    with open(TFT_FCST_FILE, "rb") as f:
        data = pickle.load(f)
    q50_idx = data.get("q50_idx", 2)
    forecasts = {ts: arr[:, q50_idx] for ts, arr in data["forecasts"].items()}
    print(f"  Loaded {len(forecasts):,} TFT windows in {time.time()-t0:.1f}s  "
          f"(q50 index={q50_idx}, quantiles={data.get('quantiles')})")
    return forecasts


def load_lgbm_strategic_forecasts() -> dict:
    """
    Load retrospective LightGBM strategic forecasts from retro_lgbm_strategic_forecasts.pkl.
    Returns dict: UTC Timestamp → np.ndarray shape (144,) q50 in $/MWh.
    Returns empty dict if file not found.
    """
    if not LGBM_STRAT_FILE.exists():
        return {}
    print("Loading LightGBM strategic forecast pickle...")
    t0 = time.time()
    with open(LGBM_STRAT_FILE, "rb") as f:
        data = pickle.load(f)
    q50_col = data.get("quantiles", [0.05, 0.50, 0.95]).index(0.50)
    forecasts = {ts: arr[:, q50_col] for ts, arr in data["forecasts"].items()}
    print(f"  Loaded {len(forecasts):,} windows in {time.time()-t0:.1f}s")
    return forecasts


def load_tier1_forecasts() -> dict:
    """
    Load retrospective Tier 1 LGBM forecasts from retro_tier1_forecasts.pkl.
    Returns dict: UTC Timestamp → np.ndarray shape (2,) in $/MWh.
      [0] = mean h0..h5 (0–30 min),  [1] = mean h6..h11 (30–60 min)
    Returns empty dict if file not found.
    """
    if not TIER1_FCST_FILE.exists():
        return {}
    print("Loading Tier 1 forecast pickle...")
    t0 = time.time()
    with open(TIER1_FCST_FILE, "rb") as f:
        data = pickle.load(f)
    forecasts = data["forecasts"]
    print(f"  Loaded {len(forecasts):,} Tier 1 windows in {time.time()-t0:.1f}s")
    return forecasts


def build_hybrid_forecast(tier1: np.ndarray, tft: np.ndarray) -> np.ndarray:
    """
    Combine Tier 1 (first 2 steps) and TFT q50 (steps 2–143) into a 144-step forecast.
    tier1: shape (2,) $/MWh — 0–30 min and 30–60 min averaged
    tft:   shape (144,) $/MWh — full 72h TFT q50
    """
    return np.concatenate([tier1, tft[2:]])


# ── Greedy dispatch (fast alternative to rolling LP MPC) ──────────────────────

def greedy_dispatch(forecast_prices_mwh: np.ndarray, actual_prices_mwh: np.ndarray,
                    net_load_kw: np.ndarray | None = None,
                    capacity_kwh: float = 40.0, max_kw: float = 10.0,
                    eff_c: float = 0.95, eff_d: float = 0.95,
                    deg: float = 0.05, soc_init: float = 20.0) -> float:
    """
    O(N log N) offline greedy: sort forecast prices, charge at cheapest steps,
    discharge at most expensive steps. Much faster than rolling LP MPC and
    correctly captures spike opportunities.

    Returns total P&L ($) for the window.
    """
    n = len(actual_prices_mwh)
    ih = INTERVAL_H

    # How many steps needed to charge/discharge the battery
    kWh_per_step_charge    = max_kw * eff_c * ih
    kWh_per_step_discharge = max_kw * ih

    n_charge    = max(1, int(np.ceil((capacity_kwh - soc_init) / kWh_per_step_charge)))
    n_discharge = max(1, int(np.ceil(capacity_kwh / kWh_per_step_discharge)))

    # Schedule: charge at n_charge cheapest forecast steps,
    #           discharge at n_discharge most expensive forecast steps
    sorted_idx = np.argsort(forecast_prices_mwh)
    charge_set    = set(sorted_idx[:n_charge].tolist())
    discharge_set = set(sorted_idx[-n_discharge:].tolist())

    soc = float(soc_init)
    total_pnl = 0.0

    for h in range(n):
        ap = actual_prices_mwh[h] / 1000.0  # $/kWh

        if h in charge_set and soc < capacity_kwh - 1e-9:
            c = min(max_kw, (capacity_kwh - soc) / (eff_c * ih))
            soc = min(capacity_kwh, soc + c * eff_c * ih)
            if net_load_kw is None:
                total_pnl += (-c * ap - deg * c * eff_c) * ih
            else:
                grid = net_load_kw[h] + c
                total_pnl += (-grid * ap - deg * c * eff_c) * ih
        elif h in discharge_set and soc > 1e-9:
            d = min(max_kw, soc / ih)
            soc = max(0.0, soc - d * ih)
            if net_load_kw is None:
                total_pnl += (d * eff_d * ap - deg * d) * ih
            else:
                grid = net_load_kw[h] - d * eff_d
                total_pnl += (-grid * ap - deg * d) * ih
        else:
            # Idle: no battery action
            if net_load_kw is not None:
                total_pnl += -net_load_kw[h] * ap * ih

    return total_pnl


# ── Per-window simulation (picklable for multiprocessing) ─────────────────────

def simulate_window(args: tuple) -> dict | None:
    """
    Worker function: simulate oracle/lgbm/p5min (and optionally ai) for one window.
    Designed to be called from ProcessPoolExecutor.

    args = (start_ts, stratum, actual_prices, lgbm_fcst, net_load_kw,
            price_only, dispatch_mode, tft_fcst)
    tft_fcst: ndarray(144,) in $/MWh or None to skip AI source.
    """
    start_ts, stratum, actual_prices, amber_fcst, net_load_kw, price_only, dispatch_mode, tft_fcst = args

    net_load = None if price_only else net_load_kw

    if dispatch_mode == "greedy":
        oracle_pnl = greedy_dispatch(actual_prices, actual_prices, net_load)
        amber_pnl  = greedy_dispatch(amber_fcst,    actual_prices, net_load)
        p5min_fcst = np.full(WINDOW_STEPS, actual_prices[0])
        p5min_pnl  = greedy_dispatch(p5min_fcst,    actual_prices, net_load)
        ai_pnl     = (greedy_dispatch(tft_fcst, actual_prices, net_load)
                      if tft_fcst is not None else None)
    else:
        # Import inside worker (subprocess needs its own imports)
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from eval.dispatch_simulator import simulate_mpc

        oracle_pnl = simulate_mpc(actual_prices, actual_prices, net_load_actuals=net_load,
                                   interval_h=INTERVAL_H)["total_pnl"]
        amber_pnl  = simulate_mpc(amber_fcst,    actual_prices, net_load_actuals=net_load,
                                   interval_h=INTERVAL_H)["total_pnl"]
        p5min_fcst = np.full(WINDOW_STEPS, actual_prices[0])
        p5min_pnl  = simulate_mpc(p5min_fcst,   actual_prices, net_load_actuals=net_load,
                                   interval_h=INTERVAL_H)["total_pnl"]
        ai_pnl     = (simulate_mpc(tft_fcst, actual_prices, net_load_actuals=net_load,
                                    interval_h=INTERVAL_H)["total_pnl"]
                      if tft_fcst is not None else None)

    return {
        "start_time": start_ts,
        "stratum":    stratum,
        "oracle_pnl": oracle_pnl,
        "amber_pnl":  amber_pnl,
        "p5min_pnl":  p5min_pnl,
        "ai_pnl":     ai_pnl,
    }


def classify_window(prices_mwh: np.ndarray) -> str:
    if np.any(prices_mwh >= SPIKE_THRESH):
        return "spike"
    if np.any(prices_mwh <= LOW_THRESH):
        return "low"
    return "normal"


# ── Summary reporting ─────────────────────────────────────────────────────────

def build_summary(df_raw: pd.DataFrame,
                  ai_label: str = "tft_tier2_q50") -> pd.DataFrame:
    """Build $/day per-stratum summary table from raw per-window results."""
    days_per_window = (WINDOW_STEPS * INTERVAL_H) / 24
    pnl_cols = ["oracle_pnl", "amber_pnl", "p5min_pnl"]
    if "ai_pnl" in df_raw.columns and df_raw["ai_pnl"].notna().any():
        pnl_cols.append("ai_pnl")
    for col in pnl_cols:
        df_raw[col + "_per_day"] = df_raw[col] / days_per_window

    sources = [("oracle",          "oracle_pnl_per_day"),
               ("amber_apf_lgbm",  "amber_pnl_per_day"),
               ("p5min_naive",     "p5min_pnl_per_day")]
    if "ai_pnl_per_day" in df_raw.columns:
        sources.append((ai_label, "ai_pnl_per_day"))

    rows = []
    strata = [s for s in ["spike", "low", "normal"] if s in df_raw["stratum"].values]
    for s in strata + ["all"]:
        mask = df_raw["stratum"] == s if s != "all" else pd.Series(True, index=df_raw.index)
        sub = df_raw[mask]
        if sub.empty:
            continue
        amber_mean = sub["amber_pnl_per_day"].mean()
        for source, col in sources:
            if col not in sub.columns:
                continue
            sub_valid = sub[sub[col].notna()] if source == "tft_tier2_q50" else sub
            if sub_valid.empty:
                continue
            mean_val = sub_valid[col].mean()
            vs_amber = (mean_val / max(abs(amber_mean), 1e-9) - 1) * 100
            rows.append({
                "stratum": s, "source": source, "n": len(sub_valid),
                "mean_per_day": round(mean_val, 4),
                "vs_amber_apf_pct": round(vs_amber, 1) if source != "amber_apf_lgbm" else 0.0,
            })
    return pd.DataFrame(rows)


def print_summary(df_summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("Holistic Dispatch Simulation — Results")
    print("=" * 70)
    strata = df_summary["stratum"].unique()
    for s in [x for x in ["spike", "low", "normal", "all"] if x in strata]:
        sub = df_summary[df_summary["stratum"] == s]
        print(f"\n  [{s.upper():8s}]  n={sub['n'].iloc[0]}")
        print(f"  {'Source':<16}  {'Mean $/day':>12}  {'vs Amber APF':>12}")
        print(f"  {'-'*42}")
        for _, row in sub.iterrows():
            vs = f"{row['vs_amber_apf_pct']:+.1f}%" if row["source"] != "amber_apf_lgbm" else "baseline"
            print(f"  {row['source']:<16}  {row['mean_per_day']:>12.4f}  {vs:>12}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-only", action="store_true",
                        help="Skip load/PV data (price arbitrage only)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--strata", type=str, default=None,
                        help="Comma-separated strata to run (e.g. 'low,normal'). "
                             "Skips strata with existing checkpoint files.")
    parser.add_argument("--fast", action="store_true",
                        help="Use 50-window fast subset (greedy dispatch, <2 min). "
                             "Saves to holistic_eval_raw_{stratum}_fast.parquet")
    parser.add_argument("--dispatch", choices=["lp", "greedy"], default="lp",
                        help="Dispatch algorithm: 'lp' (rolling MPC, accurate) or "
                             "'greedy' (O(N) heuristic, ~100x faster). "
                             "--fast implies --dispatch greedy.")
    parser.add_argument("--ai-source", action="store_true",
                        help="Include TFT Tier2 q50 as standalone dispatch signal "
                             "(retro_tft_forecasts.pkl). Saves *_ai checkpoints.")
    parser.add_argument("--hybrid-source", action="store_true",
                        help="Include Tier 1 + TFT Tier 2 hybrid source: Tier 1 LGBM q50 "
                             "for first 2 steps (0–60 min), TFT q50 for steps 2–143 (1h–72h). "
                             "Requires both retro_tier1_forecasts.pkl and retro_tft_forecasts.pkl. "
                             "Saves *_hybrid checkpoints.")
    parser.add_argument("--lgbm-strategic", action="store_true",
                        help="Include LightGBM strategic (30-min/72-hour) as AI source. "
                             "Requires retro_lgbm_strategic_forecasts.pkl. "
                             "Saves *_lgbm_strategic checkpoints.")
    args = parser.parse_args()

    n_ai_flags = sum([args.ai_source, args.hybrid_source, args.lgbm_strategic])
    if n_ai_flags > 1:
        print("ERROR: --ai-source, --hybrid-source, --lgbm-strategic are mutually exclusive.")
        sys.exit(1)

    dispatch_mode    = args.dispatch
    fast_mode        = args.fast
    ai_source        = args.ai_source
    hybrid_source    = args.hybrid_source
    lgbm_strategic   = args.lgbm_strategic
    any_ai_source    = ai_source or hybrid_source or lgbm_strategic

    if fast_mode:
        ckpt_suffix = "_fast"
    elif hybrid_source:
        ckpt_suffix = "_hybrid"
    elif ai_source:
        ckpt_suffix = "_ai"
    elif lgbm_strategic:
        ckpt_suffix = "_lgbm_strategic"
    else:
        ckpt_suffix = ""

    if hybrid_source:
        ai_source_label = "tier1_tier2_hybrid"
    elif lgbm_strategic:
        ai_source_label = "lgbm_strategic"
    else:
        ai_source_label = "tft_tier2_q50"
    fast_n        = 50  # windows per stratum in fast mode

    if not INDEX_FILE.exists():
        print(f"ERROR: eval index not found at {INDEX_FILE}")
        print("Run: python eval/build_holistic_eval_set.py")
        sys.exit(1)

    config = load_config()
    df_index = pd.read_parquet(INDEX_FILE)
    df_index["start_time"] = pd.to_datetime(df_index["start_time"], utc=True)

    if fast_mode:
        # Use a reproducible 50-window-per-stratum fast subset
        parts = []
        for s in ["spike", "low", "normal"]:
            sub = df_index[df_index["stratum"] == s]
            parts.append(sub.sample(n=min(fast_n, len(sub)), random_state=99))
        df_index = pd.concat(parts).sort_values("start_time").reset_index(drop=True)
        print(f"Fast mode: {len(df_index)} windows ({fast_n}/stratum), {dispatch_mode} dispatch")
    else:
        print(f"Full mode: {len(df_index)} windows, {dispatch_mode} dispatch")

    all_strata = ["spike", "low", "normal"]
    if args.strata:
        requested = [s.strip() for s in args.strata.split(",")]
    else:
        requested = all_strata

    # Determine which strata still need running (skip if checkpoint exists)
    strata_todo = []
    strata_done = []
    for s in requested:
        ckpt = RESULTS_DIR / f"holistic_eval_raw_{s}{ckpt_suffix}.parquet"
        if ckpt.exists():
            print(f"  {s:8s}: checkpoint found — skipping ({ckpt.name})")
            strata_done.append(s)
        else:
            strata_todo.append(s)

    if not strata_todo:
        print("All requested strata have checkpoints. Merging and reporting.")
    else:
        # ── Bulk data: frozen parquet (preferred) or InfluxDB ────────────────
        bulk = load_frozen_actuals()
        if bulk is None:
            print("WARNING: holistic_eval_actuals.parquet not found — querying InfluxDB.")
            print("         Results will not be reproducible. Run eval/export_holistic_actuals.py")
            print("         to freeze the data.")
            todo_windows = df_index[df_index["stratum"].isin(strata_todo)]
            bulk_start = todo_windows["start_time"].min().strftime("%Y-%m-%dT%H:%M:%SZ")
            bulk_end = (todo_windows["start_time"].max() +
                        pd.Timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")
            client = InfluxDBClient(**config["influxdb"])
            try:
                bulk = fetch_bulk_data(client, bulk_start, bulk_end)
            finally:
                client.close()

        # ── Build work queue ──────────────────────────────────────────────────
        amber_forecasts        = load_amber_lgbm_forecasts()
        tft_forecasts          = load_tft_forecasts() if (ai_source or hybrid_source) else {}
        tier1_forecasts        = load_tier1_forecasts() if hybrid_source else {}
        lgbm_strat_forecasts   = load_lgbm_strategic_forecasts() if lgbm_strategic else {}

        if hybrid_source and not tier1_forecasts:
            print("ERROR: --hybrid-source requires retro_tier1_forecasts.pkl")
            print("Run: nice -n 19 python eval/retro_tier1_inference.py")
            sys.exit(1)
        if hybrid_source and not tft_forecasts:
            print("ERROR: --hybrid-source requires retro_tft_forecasts.pkl")
            print("Run: nice -n 19 python eval/retro_tft_inference.py")
            sys.exit(1)
        if lgbm_strategic and not lgbm_strat_forecasts:
            print("ERROR: --lgbm-strategic requires retro_lgbm_strategic_forecasts.pkl")
            print("Run: nice -n 19 python eval/retro_lgbm_strategic_inference.py")
            sys.exit(1)

        work = []
        skipped = 0
        for stratum in strata_todo:
            subset = df_index[df_index["stratum"] == stratum]
            for row in subset.itertuples():
                start_ts = row.start_time
                window_data = slice_window(bulk, start_ts)
                if window_data is None:
                    skipped += 1
                    continue
                amber_fcst = amber_forecasts.get(start_ts.floor("30min"))
                if amber_fcst is None:
                    skipped += 1
                    continue

                if hybrid_source:
                    t1_fcst  = tier1_forecasts.get(start_ts)
                    tft_fcst = tft_forecasts.get(start_ts)
                    if t1_fcst is not None and tft_fcst is not None:
                        ai_fcst = build_hybrid_forecast(t1_fcst, tft_fcst)
                    else:
                        ai_fcst = None
                elif ai_source:
                    ai_fcst = tft_forecasts.get(start_ts)
                elif lgbm_strategic:
                    ai_fcst = lgbm_strat_forecasts.get(start_ts)
                else:
                    ai_fcst = None

                work.append((
                    start_ts,
                    stratum,
                    window_data["actual_prices_mwh"],
                    amber_fcst,
                    window_data["net_load_kw"],
                    args.price_only,
                    dispatch_mode,
                    ai_fcst,
                ))

        print(f"\nWork queue: {len(work)} windows ({skipped} skipped), "
              f"{args.workers} workers, price_only={args.price_only}")

        # ── Parallel LP solving ───────────────────────────────────────────────
        t0 = time.time()
        results_by_stratum = {s: [] for s in strata_todo}
        done = 0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(simulate_window, item): item[1] for item in work}
            for fut in as_completed(futures):
                stratum = futures[fut]
                result = fut.result()
                if result:
                    results_by_stratum[stratum].append(result)
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    eta = (len(work) - done) / rate
                    print(f"  {done}/{len(work)}  ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        elapsed_total = time.time() - t0
        print(f"Parallel LP done in {elapsed_total:.0f}s ({len(work)/elapsed_total:.1f} windows/s)")

        # ── Checkpoint each stratum ───────────────────────────────────────────
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for s, rows in results_by_stratum.items():
            if rows:
                df_s = pd.DataFrame(rows)
                ckpt = RESULTS_DIR / f"holistic_eval_raw_{s}{ckpt_suffix}.parquet"
                df_s.to_parquet(ckpt, index=False)
                print(f"  Saved {len(df_s)} {s} rows → {ckpt.name}")

    # ── Merge checkpoints ─────────────────────────────────────────────────────
    parts = []
    for s in all_strata:
        ckpt = RESULTS_DIR / f"holistic_eval_raw_{s}{ckpt_suffix}.parquet"
        if ckpt.exists():
            parts.append(pd.read_parquet(ckpt))
    if not parts:
        print("ERROR: no results to merge.")
        sys.exit(1)

    df_raw = pd.concat(parts, ignore_index=True)

    # Migrate old column name from checkpoints written before the rename
    if "lgbm_pnl" in df_raw.columns and "amber_pnl" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"lgbm_pnl": "amber_pnl"})

    # Ensure ai_pnl column exists for build_summary (will be skipped if all NaN)
    if "ai_pnl" not in df_raw.columns:
        df_raw["ai_pnl"] = np.nan

    df_raw.to_parquet(RESULTS_DIR / "holistic_eval_raw.parquet", index=False)

    df_summary = build_summary(df_raw, ai_label=ai_source_label)
    print_summary(df_summary)

    csv_out = RESULTS_DIR / "holistic_eval_results.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSaved → {csv_out.relative_to(ROOT)}")
    print(f"Saved → eval/results/holistic_eval_raw.parquet")


if __name__ == "__main__":
    main()
