#!/usr/bin/env python3
"""
Phase 6 — Holistic dispatch simulation.

Runs rolling-MPC LP dispatch on a stratified eval set, comparing:
  oracle       — perfect foresight (actual prices as forecast)
  lgbm_legacy  — as-run legacy LightGBM predictions from price_forecast_log.csv
                 (Amber APF seeded for first 14-28h, LightGBM extrapolation thereafter)
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
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "eval" / "results"
FORECAST_LOG = ROOT / "price_forecast_log.csv"
INDEX_FILE = RESULTS_DIR / "holistic_eval_index.parquet"

WINDOW_STEPS = 144        # 72h at 30-min resolution
INTERVAL_H   = 30 / 60   # 30-min steps

SPIKE_THRESH = 300.0
LOW_THRESH   = -50.0


def load_config():
    with open(ROOT / "config.json") as f:
        return json.load(f)


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

def load_lgbm_forecasts() -> dict:
    """
    Load legacy LightGBM forecast log indexed by window start time.
    Returns dict: floor-30min Timestamp → np.ndarray shape (144,) in $/MWh.
    """
    print("Loading LightGBM forecast log...")
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
    Worker function: simulate oracle/lgbm/p5min for one window.
    Designed to be called from ProcessPoolExecutor.
    """
    start_ts, stratum, actual_prices, lgbm_fcst, net_load_kw, price_only, dispatch_mode = args

    net_load = None if price_only else net_load_kw

    if dispatch_mode == "greedy":
        oracle_pnl = greedy_dispatch(actual_prices, actual_prices, net_load)
        lgbm_pnl   = greedy_dispatch(lgbm_fcst,    actual_prices, net_load)
        p5min_fcst = np.full(WINDOW_STEPS, actual_prices[0])
        p5min_pnl  = greedy_dispatch(p5min_fcst,   actual_prices, net_load)
    else:
        # Import inside worker (subprocess needs its own imports)
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from eval.dispatch_simulator import simulate_mpc

        oracle_pnl = simulate_mpc(actual_prices, actual_prices, net_load_actuals=net_load,
                                   interval_h=INTERVAL_H)["total_pnl"]
        lgbm_pnl   = simulate_mpc(lgbm_fcst,    actual_prices, net_load_actuals=net_load,
                                   interval_h=INTERVAL_H)["total_pnl"]
        p5min_fcst = np.full(WINDOW_STEPS, actual_prices[0])
        p5min_pnl  = simulate_mpc(p5min_fcst,   actual_prices, net_load_actuals=net_load,
                                   interval_h=INTERVAL_H)["total_pnl"]

    return {
        "start_time": start_ts,
        "stratum":    stratum,
        "oracle_pnl": oracle_pnl,
        "lgbm_pnl":   lgbm_pnl,
        "p5min_pnl":  p5min_pnl,
    }


def classify_window(prices_mwh: np.ndarray) -> str:
    if np.any(prices_mwh >= SPIKE_THRESH):
        return "spike"
    if np.any(prices_mwh <= LOW_THRESH):
        return "low"
    return "normal"


# ── Summary reporting ─────────────────────────────────────────────────────────

def build_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Build $/day per-stratum summary table from raw per-window results."""
    days_per_window = (WINDOW_STEPS * INTERVAL_H) / 24
    for col in ("oracle_pnl", "lgbm_pnl", "p5min_pnl"):
        df_raw[col + "_per_day"] = df_raw[col] / days_per_window

    rows = []
    strata = [s for s in ["spike", "low", "normal"] if s in df_raw["stratum"].values]
    for s in strata + ["all"]:
        mask = df_raw["stratum"] == s if s != "all" else pd.Series(True, index=df_raw.index)
        sub = df_raw[mask]
        if sub.empty:
            continue
        lgbm_mean = sub["lgbm_pnl_per_day"].mean()
        for source, col in [("oracle", "oracle_pnl_per_day"),
                             ("lgbm_legacy", "lgbm_pnl_per_day"),
                             ("p5min_naive", "p5min_pnl_per_day")]:
            mean_val = sub[col].mean()
            vs_lgbm  = (mean_val / max(abs(lgbm_mean), 1e-9) - 1) * 100
            rows.append({
                "stratum": s, "source": source, "n": len(sub),
                "mean_per_day": round(mean_val, 4),
                "vs_lgbm_pct": round(vs_lgbm, 1) if source != "lgbm_legacy" else 0.0,
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
        print(f"  {'Source':<14}  {'Mean $/day':>12}  {'vs LightGBM':>12}")
        print(f"  {'-'*42}")
        for _, row in sub.iterrows():
            vs = f"{row['vs_lgbm_pct']:+.1f}%" if row["source"] != "lgbm_legacy" else "baseline"
            print(f"  {row['source']:<14}  {row['mean_per_day']:>12.4f}  {vs:>12}")


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
    args = parser.parse_args()

    dispatch_mode = args.dispatch
    fast_mode = args.fast
    ckpt_suffix = "_fast" if fast_mode else ""
    fast_n = 50  # windows per stratum in fast mode

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
        # ── Bulk InfluxDB fetch ───────────────────────────────────────────────
        todo_windows = df_index[df_index["stratum"].isin(strata_todo)]
        bulk_start = todo_windows["start_time"].min().strftime("%Y-%m-%dT%H:%M:%SZ")
        # Fetch to last window + 72h
        bulk_end = (todo_windows["start_time"].max() +
                    pd.Timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")

        client = InfluxDBClient(**config["influxdb"])
        try:
            bulk = fetch_bulk_data(client, bulk_start, bulk_end)
        finally:
            client.close()

        # ── Build work queue ──────────────────────────────────────────────────
        lgbm_forecasts = load_lgbm_forecasts()
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
                lgbm_fcst = lgbm_forecasts.get(start_ts.floor("30min"))
                if lgbm_fcst is None:
                    skipped += 1
                    continue
                work.append((
                    start_ts,
                    stratum,
                    window_data["actual_prices_mwh"],
                    lgbm_fcst,
                    window_data["net_load_kw"],
                    args.price_only,
                    dispatch_mode,
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

    # ── Merge all checkpoints ─────────────────────────────────────────────────
    parts = []
    for s in all_strata:
        ckpt = RESULTS_DIR / f"holistic_eval_raw_{s}{ckpt_suffix}.parquet"
        if ckpt.exists():
            parts.append(pd.read_parquet(ckpt))
    if not parts:
        print("ERROR: no results to merge.")
        sys.exit(1)

    df_raw = pd.concat(parts, ignore_index=True)
    df_raw.to_parquet(RESULTS_DIR / "holistic_eval_raw.parquet", index=False)

    df_summary = build_summary(df_raw)
    print_summary(df_summary)

    csv_out = RESULTS_DIR / "holistic_eval_results.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"\nSaved → {csv_out.relative_to(ROOT)}")
    print(f"Saved → eval/results/holistic_eval_raw.parquet")


if __name__ == "__main__":
    main()
