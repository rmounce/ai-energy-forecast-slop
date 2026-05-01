#!/usr/bin/env python3
"""
Build forced-prefix regret diagnostics from rolling MPC raw parquet outputs.

For each start time and each requested source:
- take the realized closed-loop actions from that source for the next N steps
- pin that prefix in the tariffed LP
- solve the remaining horizon against actual future prices/net load
- compare the resulting full-horizon objective against the unconstrained oracle

This answers: when does Amber's advantage appear along the path?
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.dispatch_simulator import solve_lp_dispatch


def _resolve_path(path_arg: str) -> Path:
    p = Path(path_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / path_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find parquet: {path_arg}")


def _none_if_nan(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        if np.isfinite(value):
            return float(value)
    except TypeError:
        return None
    return None


def _forced_prefix_objective(
    *,
    import_prices_mwh: np.ndarray,
    export_prices_mwh: np.ndarray,
    net_load_forecast_kw: np.ndarray,
    load_forecast_kw: np.ndarray | None,
    pv_forecast_kw: np.ndarray | None,
    soc_prev_kwh: float,
    prefix_charge_kw: np.ndarray,
    prefix_discharge_kw: np.ndarray,
    terminal_energy_value_per_kwh: float,
    extra_terminal_energy_value_per_kwh: float,
    extra_terminal_energy_floor_kwh: float | None,
    extra_terminal_energy_cap_kwh: float | None,
    min_terminal_soc_kwh: float | None,
    max_terminal_soc_kwh: float | None,
) -> tuple[float, bool]:
    solve = solve_lp_dispatch(
        import_prices_mwh.copy(),
        soc_prev_kwh,
        import_prices_mwh=import_prices_mwh.copy(),
        export_prices_mwh=export_prices_mwh.copy(),
        net_load_forecast_kw=net_load_forecast_kw.copy(),
        load_forecast_kw=None if load_forecast_kw is None else load_forecast_kw.copy(),
        pv_forecast_kw=None if pv_forecast_kw is None else pv_forecast_kw.copy(),
        terminal_energy_value_per_kwh=terminal_energy_value_per_kwh,
        extra_terminal_energy_value_per_kwh=extra_terminal_energy_value_per_kwh,
        extra_terminal_energy_floor_kwh=extra_terminal_energy_floor_kwh,
        extra_terminal_energy_cap_kwh=extra_terminal_energy_cap_kwh,
        min_terminal_soc_kwh=min_terminal_soc_kwh,
        max_terminal_soc_kwh=max_terminal_soc_kwh,
        force_prefix_charge_kw=prefix_charge_kw,
        force_prefix_discharge_kw=prefix_discharge_kw,
    )
    if not solve["success"]:
        return float("nan"), False
    return float(solve["objective_value"]), True


def _split_load_pv_arrays_or_none(market_df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "actual_load_kw" not in market_df.columns or "actual_pv_kw" not in market_df.columns:
        return None, None
    load_kw = market_df["actual_load_kw"].to_numpy(dtype=np.float64, copy=True)
    pv_kw = market_df["actual_pv_kw"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(load_kw).all() or not np.isfinite(pv_kw).all():
        return None, None
    return load_kw, pv_kw


def build_dataset(
    raw_df: pd.DataFrame,
    *,
    source_a: str,
    source_b: str,
    prefix_steps: int,
    progress_every_rows: int = 0,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = raw_df.sort_values(["source", "time"], kind="stable").reset_index(drop=True)

    market_cols = ["time", "actual_general_price_mwh", "actual_feed_in_price_mwh", "actual_net_load_kw"]
    if "actual_load_kw" in raw_df.columns and "actual_pv_kw" in raw_df.columns:
        market_cols.extend(["actual_load_kw", "actual_pv_kw"])
    market_df = (
        raw_df[market_cols]
        .drop_duplicates("time")
        .sort_values("time", kind="stable")
        .set_index("time")
    )

    def src_df(source: str) -> pd.DataFrame:
        df = raw_df[raw_df["source"] == source].copy().sort_values("time", kind="stable").reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows found for source: {source}")
        return df

    a_df = src_df(source_a)
    b_df = src_df(source_b)

    # Shared start times only.
    common_times = sorted(set(a_df["time"]).intersection(set(b_df["time"])))
    a_by_time = a_df.set_index("time")
    b_by_time = b_df.set_index("time")

    rows: list[dict[str, float | str | int | bool | None]] = []
    start_t = time.perf_counter()
    total_rows = len(common_times)

    for i, ts in enumerate(common_times, start=1):
        a_row = a_by_time.loc[ts]
        if isinstance(a_row, pd.DataFrame):
            a_row = a_row.iloc[0]
        b_row = b_by_time.loc[ts]
        if isinstance(b_row, pd.DataFrame):
            b_row = b_row.iloc[0]

        future = market_df.loc[ts:]
        if future.empty:
            continue

        import_prices_mwh = future["actual_general_price_mwh"].to_numpy(dtype=np.float64, copy=True)
        export_prices_mwh = future["actual_feed_in_price_mwh"].to_numpy(dtype=np.float64, copy=True)
        net_load_forecast_kw = future["actual_net_load_kw"].to_numpy(dtype=np.float64, copy=True)
        load_forecast_kw, pv_forecast_kw = _split_load_pv_arrays_or_none(future)

        oracle = solve_lp_dispatch(
            import_prices_mwh.copy(),
            float(a_row["soc_prev_kwh"]),
            import_prices_mwh=import_prices_mwh.copy(),
            export_prices_mwh=export_prices_mwh.copy(),
            net_load_forecast_kw=net_load_forecast_kw.copy(),
            load_forecast_kw=None if load_forecast_kw is None else load_forecast_kw.copy(),
            pv_forecast_kw=None if pv_forecast_kw is None else pv_forecast_kw.copy(),
            terminal_energy_value_per_kwh=float(a_row["terminal_energy_value_per_kwh"]),
            extra_terminal_energy_value_per_kwh=float(a_row.get("extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
            extra_terminal_energy_floor_kwh=_none_if_nan(a_row.get("extra_terminal_energy_floor_kwh", None)),
            extra_terminal_energy_cap_kwh=_none_if_nan(a_row.get("extra_terminal_energy_cap_kwh", None)),
            min_terminal_soc_kwh=_none_if_nan(a_row.get("min_terminal_soc_kwh", None)),
            max_terminal_soc_kwh=_none_if_nan(a_row.get("max_terminal_soc_kwh", None)),
        )
        if not oracle["success"]:
            continue

        # Build realized prefixes from each source.
        a_prefix = a_df[a_df["time"] >= ts].head(prefix_steps)
        b_prefix = b_df[b_df["time"] >= ts].head(prefix_steps)
        horizon_available = int(len(future))
        actual_prefix_steps = min(prefix_steps, horizon_available, len(a_prefix), len(b_prefix))
        if actual_prefix_steps == 0:
            continue
        a_prefix = a_prefix.head(actual_prefix_steps)
        b_prefix = b_prefix.head(actual_prefix_steps)

        a_obj, a_ok = _forced_prefix_objective(
            import_prices_mwh=import_prices_mwh,
            export_prices_mwh=export_prices_mwh,
            net_load_forecast_kw=net_load_forecast_kw,
            load_forecast_kw=load_forecast_kw,
            pv_forecast_kw=pv_forecast_kw,
            soc_prev_kwh=float(a_row["soc_prev_kwh"]),
            prefix_charge_kw=a_prefix["charge_kw"].to_numpy(dtype=np.float64, copy=True),
            prefix_discharge_kw=a_prefix["discharge_kw"].to_numpy(dtype=np.float64, copy=True),
            terminal_energy_value_per_kwh=float(a_row["terminal_energy_value_per_kwh"]),
            extra_terminal_energy_value_per_kwh=float(a_row.get("extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
            extra_terminal_energy_floor_kwh=_none_if_nan(a_row.get("extra_terminal_energy_floor_kwh", None)),
            extra_terminal_energy_cap_kwh=_none_if_nan(a_row.get("extra_terminal_energy_cap_kwh", None)),
            min_terminal_soc_kwh=_none_if_nan(a_row.get("min_terminal_soc_kwh", None)),
            max_terminal_soc_kwh=_none_if_nan(a_row.get("max_terminal_soc_kwh", None)),
        )
        b_obj, b_ok = _forced_prefix_objective(
            import_prices_mwh=import_prices_mwh,
            export_prices_mwh=export_prices_mwh,
            net_load_forecast_kw=net_load_forecast_kw,
            load_forecast_kw=load_forecast_kw,
            pv_forecast_kw=pv_forecast_kw,
            soc_prev_kwh=float(a_row["soc_prev_kwh"]),
            prefix_charge_kw=b_prefix["charge_kw"].to_numpy(dtype=np.float64, copy=True),
            prefix_discharge_kw=b_prefix["discharge_kw"].to_numpy(dtype=np.float64, copy=True),
            terminal_energy_value_per_kwh=float(a_row["terminal_energy_value_per_kwh"]),
            extra_terminal_energy_value_per_kwh=float(a_row.get("extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
            extra_terminal_energy_floor_kwh=_none_if_nan(a_row.get("extra_terminal_energy_floor_kwh", None)),
            extra_terminal_energy_cap_kwh=_none_if_nan(a_row.get("extra_terminal_energy_cap_kwh", None)),
            min_terminal_soc_kwh=_none_if_nan(a_row.get("min_terminal_soc_kwh", None)),
            max_terminal_soc_kwh=_none_if_nan(a_row.get("max_terminal_soc_kwh", None)),
        )

        rows.append(
            {
                "time": ts,
                "prefix_steps_requested": int(prefix_steps),
                "prefix_steps_actual": int(actual_prefix_steps),
                "horizon_steps_available": horizon_available,
                "actual_feed_in_price_mwh": float(a_row["actual_feed_in_price_mwh"]),
                "actual_net_load_kw": float(a_row["actual_net_load_kw"]),
                "actual_load_kw": float(a_row.get("actual_load_kw", np.nan)),
                "actual_pv_kw": float(a_row.get("actual_pv_kw", np.nan)),
                "soc_prev_kwh": float(a_row["soc_prev_kwh"]),
                "oracle_objective_value": float(oracle["objective_value"]),
                "a_source": source_a,
                "b_source": source_b,
                "a_forced_prefix_objective_value": a_obj,
                "a_forced_prefix_success": bool(a_ok),
                "a_forced_prefix_regret": (a_obj - float(oracle["objective_value"])) if a_ok else float("nan"),
                "b_forced_prefix_objective_value": b_obj,
                "b_forced_prefix_success": bool(b_ok),
                "b_forced_prefix_regret": (b_obj - float(oracle["objective_value"])) if b_ok else float("nan"),
                "a_minus_b_forced_prefix_regret": (a_obj - b_obj) if (a_ok and b_ok) else float("nan"),
                "a_prefix_export_kw_sum": float(a_prefix["grid_export_kw"].sum()),
                "b_prefix_export_kw_sum": float(b_prefix["grid_export_kw"].sum()),
                "a_prefix_discharge_kw_sum": float(a_prefix["discharge_kw"].sum()),
                "b_prefix_discharge_kw_sum": float(b_prefix["discharge_kw"].sum()),
            }
        )

        if progress_every_rows > 0 and (i % progress_every_rows == 0 or i == total_rows):
            elapsed = time.perf_counter() - start_t
            rate = i / elapsed if elapsed > 0 else float("nan")
            remain = max(total_rows - i, 0)
            eta = remain / rate if np.isfinite(rate) and rate > 0 else float("nan")
            eta_str = f"{eta/60:.1f}m" if np.isfinite(eta) else "?"
            print(f"[progress] {i}/{total_rows} rows ({100*i/total_rows:.1f}%) elapsed={elapsed/60:.1f}m eta={eta_str}", flush=True)

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw parquet path or filename under eval/results")
    parser.add_argument("--source-a", default="amber_tactical_hybrid_strategic")
    parser.add_argument("--source-b", default="model_a_hybrid")
    parser.add_argument("--prefix-steps", type=int, required=True)
    parser.add_argument("--progress-every-rows", type=int, default=0)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    raw_df = pd.read_parquet(_resolve_path(args.raw))
    out = build_dataset(
        raw_df,
        source_a=args.source_a,
        source_b=args.source_b,
        prefix_steps=int(args.prefix_steps),
        progress_every_rows=max(0, int(args.progress_every_rows)),
    )
    parquet = RESULTS_DIR / f"{args.output_prefix}_forced_prefix_regret.parquet"
    csv_path = RESULTS_DIR / f"{args.output_prefix}_forced_prefix_regret.csv"
    out.to_parquet(parquet, index=False)
    out.to_csv(csv_path, index=False)
    print(f"[done] wrote {parquet}")
    print(f"[done] wrote {csv_path}")
    if not out.empty:
        print(
            out[
                [
                    "prefix_steps_requested",
                    "a_forced_prefix_regret",
                    "b_forced_prefix_regret",
                    "a_minus_b_forced_prefix_regret",
                ]
            ].describe().to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
