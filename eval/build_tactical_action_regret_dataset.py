#!/usr/bin/env python3
"""
Build an oracle-action / action-regret dataset from rolling MPC raw parquet outputs.

For each step of a target source, replay the tariffed LP using:
- actual future import/export prices
- actual future net load
- the same SoC and terminal constraints logged in the raw row

This produces a per-step dataset comparing:
- observed target-source first action
- optional comparator-source first action (e.g. Amber)
- oracle first action under actual future prices

The goal is to move tactical diagnosis away from price-MAE and toward action-value error.
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

from eval.dispatch_simulator import DEG_PER_KWH, EFF_C, EFF_D, INTERVAL_H, solve_lp_dispatch


def _resolve_path(raw_arg: str) -> Path:
    p = Path(raw_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / raw_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find parquet: {raw_arg}")


def _none_if_nan(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        if np.isfinite(value):
            return float(value)
    except TypeError:
        return None
    return None


def _step_components(
    *,
    charge_kw: float,
    discharge_kw: float,
    curtail_kw: float = 0.0,
    actual_general_price_mwh: float,
    actual_feed_in_price_mwh: float,
    actual_net_load_kw: float,
    actual_load_kw: float = float("nan"),
    actual_pv_kw: float = float("nan"),
) -> dict[str, float]:
    if np.isfinite(actual_load_kw) and np.isfinite(actual_pv_kw):
        base_grid_kw = float(actual_load_kw) - float(actual_pv_kw)
    else:
        base_grid_kw = float(actual_net_load_kw)
    grid_kw = base_grid_kw + float(charge_kw) - float(discharge_kw) * EFF_D + float(curtail_kw)
    realized_grid_import_kw = max(grid_kw, 0.0)
    realized_grid_export_kw = max(-grid_kw, 0.0)
    import_cost = realized_grid_import_kw * (actual_general_price_mwh / 1000.0) * INTERVAL_H
    export_revenue = realized_grid_export_kw * (actual_feed_in_price_mwh / 1000.0) * INTERVAL_H
    degradation_cost = DEG_PER_KWH * (charge_kw * EFF_C + discharge_kw) * INTERVAL_H
    step_pnl = export_revenue - import_cost - degradation_cost
    return {
        "grid_import_kw": realized_grid_import_kw,
        "grid_export_kw": realized_grid_export_kw,
        "import_cost": import_cost,
        "export_revenue": export_revenue,
        "degradation_cost": degradation_cost,
        "step_pnl": step_pnl,
    }


def _forced_first_action_total_objective(
    *,
    charge_kw: float,
    discharge_kw: float,
    soc_prev_kwh: float,
    import_prices_mwh: np.ndarray,
    export_prices_mwh: np.ndarray,
    net_load_forecast_kw: np.ndarray,
    load_forecast_kw: np.ndarray | None,
    pv_forecast_kw: np.ndarray | None,
    terminal_energy_value_per_kwh: float,
    extra_terminal_energy_value_per_kwh: float,
    extra_terminal_energy_floor_kwh: float | None,
    extra_terminal_energy_cap_kwh: float | None,
    min_terminal_soc_kwh: float | None,
    max_terminal_soc_kwh: float | None,
) -> tuple[float, bool]:
    forced = solve_lp_dispatch(
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
        force_first_charge_kw=charge_kw,
        force_first_discharge_kw=discharge_kw,
    )
    if not forced["success"]:
        return float("nan"), False
    return float(forced["objective_value"]), True


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
    target_source: str,
    comparator_source: str | None = None,
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
        .reset_index(drop=True)
    )
    market_df = market_df.set_index("time")

    target_df = (
        raw_df[raw_df["source"] == target_source]
        .copy()
        .sort_values("time", kind="stable")
        .reset_index(drop=True)
    )
    if target_df.empty:
        raise ValueError(f"No rows found for target source: {target_source}")

    comparator_by_time: pd.DataFrame | None = None
    if comparator_source:
        comparator_df = (
            raw_df[raw_df["source"] == comparator_source]
            .copy()
            .sort_values("time", kind="stable")
            .set_index("time")
        )
        if comparator_df.empty:
            raise ValueError(f"No rows found for comparator source: {comparator_source}")
        comparator_by_time = comparator_df

    rows: list[dict[str, float | str | int | bool | None]] = []
    total_rows = len(target_df)
    start_t = time.perf_counter()

    for i, row in enumerate(target_df.itertuples(index=False), start=1):
        ts = pd.Timestamp(row.time)
        future = market_df.loc[ts:]
        if future.empty:
            continue

        import_prices_mwh = future["actual_general_price_mwh"].to_numpy(dtype=np.float64, copy=True)
        export_prices_mwh = future["actual_feed_in_price_mwh"].to_numpy(dtype=np.float64, copy=True)
        net_load_forecast_kw = future["actual_net_load_kw"].to_numpy(dtype=np.float64, copy=True)
        load_forecast_kw, pv_forecast_kw = _split_load_pv_arrays_or_none(future)
        prices_mwh = import_prices_mwh.copy()

        solve = solve_lp_dispatch(
            prices_mwh,
            float(row.soc_prev_kwh),
            import_prices_mwh=import_prices_mwh,
            export_prices_mwh=export_prices_mwh,
            net_load_forecast_kw=net_load_forecast_kw,
            load_forecast_kw=load_forecast_kw,
            pv_forecast_kw=pv_forecast_kw,
            terminal_energy_value_per_kwh=float(row.terminal_energy_value_per_kwh),
            extra_terminal_energy_value_per_kwh=float(getattr(row, "extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
            extra_terminal_energy_floor_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_floor_kwh", None)),
            extra_terminal_energy_cap_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_cap_kwh", None)),
            min_terminal_soc_kwh=_none_if_nan(getattr(row, "min_terminal_soc_kwh", None)),
            max_terminal_soc_kwh=_none_if_nan(getattr(row, "max_terminal_soc_kwh", None)),
        )

        oracle_charge_kw = float(solve["charge_kw"][0]) if len(solve["charge_kw"]) else 0.0
        oracle_discharge_kw = float(solve["discharge_kw"][0]) if len(solve["discharge_kw"]) else 0.0
        oracle_curtail_kw = float(solve["curtail_kw"][0]) if len(solve.get("curtail_kw", [])) else 0.0

        oracle_step = _step_components(
            charge_kw=oracle_charge_kw,
            discharge_kw=oracle_discharge_kw,
            curtail_kw=oracle_curtail_kw,
            actual_general_price_mwh=float(row.actual_general_price_mwh),
            actual_feed_in_price_mwh=float(row.actual_feed_in_price_mwh),
            actual_net_load_kw=float(row.actual_net_load_kw),
            actual_load_kw=float(getattr(row, "actual_load_kw", np.nan)),
            actual_pv_kw=float(getattr(row, "actual_pv_kw", np.nan)),
        )
        observed_step = _step_components(
            charge_kw=float(row.charge_kw),
            discharge_kw=float(row.discharge_kw),
            curtail_kw=float(getattr(row, "curtail_kw", 0.0) or 0.0),
            actual_general_price_mwh=float(row.actual_general_price_mwh),
            actual_feed_in_price_mwh=float(row.actual_feed_in_price_mwh),
            actual_net_load_kw=float(row.actual_net_load_kw),
            actual_load_kw=float(getattr(row, "actual_load_kw", np.nan)),
            actual_pv_kw=float(getattr(row, "actual_pv_kw", np.nan)),
        )
        observed_total_objective, observed_total_success = _forced_first_action_total_objective(
            charge_kw=float(row.charge_kw),
            discharge_kw=float(row.discharge_kw),
            soc_prev_kwh=float(row.soc_prev_kwh),
            import_prices_mwh=import_prices_mwh,
            export_prices_mwh=export_prices_mwh,
            net_load_forecast_kw=net_load_forecast_kw,
            load_forecast_kw=load_forecast_kw,
            pv_forecast_kw=pv_forecast_kw,
            terminal_energy_value_per_kwh=float(row.terminal_energy_value_per_kwh),
            extra_terminal_energy_value_per_kwh=float(getattr(row, "extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
            extra_terminal_energy_floor_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_floor_kwh", None)),
            extra_terminal_energy_cap_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_cap_kwh", None)),
            min_terminal_soc_kwh=_none_if_nan(getattr(row, "min_terminal_soc_kwh", None)),
            max_terminal_soc_kwh=_none_if_nan(getattr(row, "max_terminal_soc_kwh", None)),
        )

        out: dict[str, float | str | int | bool | None] = {
            "time": ts,
            "target_source": target_source,
            "comparator_source": comparator_source or "",
            "horizon_steps_available": int(len(future)),
            "soc_prev_kwh": float(row.soc_prev_kwh),
            "actual_general_price_mwh": float(row.actual_general_price_mwh),
            "actual_feed_in_price_mwh": float(row.actual_feed_in_price_mwh),
            "actual_net_load_kw": float(row.actual_net_load_kw),
            "actual_load_kw": float(getattr(row, "actual_load_kw", np.nan)),
            "actual_pv_kw": float(getattr(row, "actual_pv_kw", np.nan)),
            "strategic_soc_target_kwh": _none_if_nan(getattr(row, "strategic_soc_target_kwh", None)),
            "terminal_energy_value_per_kwh": float(row.terminal_energy_value_per_kwh),
            "min_terminal_soc_kwh": _none_if_nan(getattr(row, "min_terminal_soc_kwh", None)),
            "max_terminal_soc_kwh": _none_if_nan(getattr(row, "max_terminal_soc_kwh", None)),
            "forecast_step0_mwh": float(getattr(row, "forecast_step0_mwh", np.nan)),
            "forecast_mean_next_1h_mwh": float(getattr(row, "forecast_mean_next_1h_mwh", np.nan)),
            "forecast_mean_next_4h_mwh": float(getattr(row, "forecast_mean_next_4h_mwh", np.nan)),
            "forecast_mean_next_14h_mwh": float(getattr(row, "forecast_mean_next_14h_mwh", np.nan)),
            "forecast_buy_mean_next_1h_mwh": float(getattr(row, "forecast_buy_mean_next_1h_mwh", np.nan)),
            "forecast_buy_mean_next_4h_mwh": float(getattr(row, "forecast_buy_mean_next_4h_mwh", np.nan)),
            "forecast_buy_mean_next_14h_mwh": float(getattr(row, "forecast_buy_mean_next_14h_mwh", np.nan)),
            "forecast_sell_mean_next_1h_mwh": float(getattr(row, "forecast_sell_mean_next_1h_mwh", np.nan)),
            "forecast_sell_mean_next_4h_mwh": float(getattr(row, "forecast_sell_mean_next_4h_mwh", np.nan)),
            "forecast_sell_mean_next_14h_mwh": float(getattr(row, "forecast_sell_mean_next_14h_mwh", np.nan)),
            "observed_charge_kw": float(row.charge_kw),
            "observed_discharge_kw": float(row.discharge_kw),
            "oracle_charge_kw": oracle_charge_kw,
            "oracle_discharge_kw": oracle_discharge_kw,
            "oracle_curtail_kw": oracle_curtail_kw,
            "oracle_success": bool(solve["success"]),
            "oracle_objective_value": float(solve["objective_value"]),
            "oracle_initial_soc_shadow_price_per_kwh": float(solve["initial_soc_shadow_price_per_kwh"]),
            "observed_forced_total_objective_value": observed_total_objective,
            "observed_forced_total_objective_success": bool(observed_total_success),
            "observed_forced_total_objective_regret": (
                observed_total_objective - float(solve["objective_value"])
                if observed_total_success
                else float("nan")
            ),
            "oracle_charge_delta_kw": oracle_charge_kw - float(row.charge_kw),
            "oracle_discharge_delta_kw": oracle_discharge_kw - float(row.discharge_kw),
            "oracle_step_pnl": oracle_step["step_pnl"],
            "observed_step_pnl_recomputed": observed_step["step_pnl"],
            "oracle_step_pnl_delta": oracle_step["step_pnl"] - observed_step["step_pnl"],
            "oracle_import_cost_delta": oracle_step["import_cost"] - observed_step["import_cost"],
            "oracle_export_revenue_delta": oracle_step["export_revenue"] - observed_step["export_revenue"],
            "oracle_degradation_cost_delta": oracle_step["degradation_cost"] - observed_step["degradation_cost"],
            "oracle_grid_import_delta_kw": oracle_step["grid_import_kw"] - observed_step["grid_import_kw"],
            "oracle_grid_export_delta_kw": oracle_step["grid_export_kw"] - observed_step["grid_export_kw"],
        }

        if comparator_by_time is not None and ts in comparator_by_time.index:
            comp_row = comparator_by_time.loc[ts]
            if isinstance(comp_row, pd.DataFrame):
                comp_row = comp_row.iloc[0]
            comp_step = _step_components(
                charge_kw=float(comp_row["charge_kw"]),
                discharge_kw=float(comp_row["discharge_kw"]),
                curtail_kw=float(comp_row.get("curtail_kw", 0.0) or 0.0),
                actual_general_price_mwh=float(row.actual_general_price_mwh),
                actual_feed_in_price_mwh=float(row.actual_feed_in_price_mwh),
                actual_net_load_kw=float(row.actual_net_load_kw),
                actual_load_kw=float(getattr(row, "actual_load_kw", np.nan)),
                actual_pv_kw=float(getattr(row, "actual_pv_kw", np.nan)),
            )
            comp_total_objective, comp_total_success = _forced_first_action_total_objective(
                charge_kw=float(comp_row["charge_kw"]),
                discharge_kw=float(comp_row["discharge_kw"]),
                soc_prev_kwh=float(row.soc_prev_kwh),
                import_prices_mwh=import_prices_mwh,
                export_prices_mwh=export_prices_mwh,
                net_load_forecast_kw=net_load_forecast_kw,
                load_forecast_kw=load_forecast_kw,
                pv_forecast_kw=pv_forecast_kw,
                terminal_energy_value_per_kwh=float(row.terminal_energy_value_per_kwh),
                extra_terminal_energy_value_per_kwh=float(getattr(row, "extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
                extra_terminal_energy_floor_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_floor_kwh", None)),
                extra_terminal_energy_cap_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_cap_kwh", None)),
                min_terminal_soc_kwh=_none_if_nan(getattr(row, "min_terminal_soc_kwh", None)),
                max_terminal_soc_kwh=_none_if_nan(getattr(row, "max_terminal_soc_kwh", None)),
            )
            out.update(
                {
                    "comparator_charge_kw": float(comp_row["charge_kw"]),
                    "comparator_discharge_kw": float(comp_row["discharge_kw"]),
                    "comparator_step_pnl_recomputed": comp_step["step_pnl"],
                    "comparator_forced_total_objective_value": comp_total_objective,
                    "comparator_forced_total_objective_success": bool(comp_total_success),
                    "comparator_forced_total_objective_regret": (
                        comp_total_objective - float(solve["objective_value"])
                        if comp_total_success
                        else float("nan")
                    ),
                    "oracle_vs_comparator_charge_delta_kw": oracle_charge_kw - float(comp_row["charge_kw"]),
                    "oracle_vs_comparator_discharge_delta_kw": oracle_discharge_kw - float(comp_row["discharge_kw"]),
                    "oracle_vs_comparator_step_pnl_delta": oracle_step["step_pnl"] - comp_step["step_pnl"],
                    "comparator_vs_observed_step_pnl_delta": comp_step["step_pnl"] - observed_step["step_pnl"],
                }
            )

        rows.append(out)

        if progress_every_rows > 0 and (i % progress_every_rows == 0 or i == total_rows):
            elapsed = time.perf_counter() - start_t
            rate = i / elapsed if elapsed > 0 else float("nan")
            remaining = max(total_rows - i, 0)
            eta_sec = remaining / rate if rate and np.isfinite(rate) and rate > 0 else float("nan")
            eta_str = f"{eta_sec/60:.1f}m" if np.isfinite(eta_sec) else "?"
            print(
                f"[progress] {i}/{total_rows} rows ({100.0 * i / total_rows:.1f}%) "
                f"elapsed={elapsed/60:.1f}m eta={eta_str}",
                flush=True,
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Raw parquet path or filename under eval/results")
    parser.add_argument("--target-source", default="model_a_hybrid")
    parser.add_argument("--comparator-source", default="amber_tactical_hybrid_strategic")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--progress-every-rows", type=int, default=0, help="Emit progress every N processed target rows")
    args = parser.parse_args()

    raw_path = _resolve_path(args.raw)
    raw_df = pd.read_parquet(raw_path)
    dataset = build_dataset(
        raw_df,
        target_source=args.target_source,
        comparator_source=args.comparator_source or None,
        progress_every_rows=max(0, int(args.progress_every_rows)),
    )

    out_parquet = RESULTS_DIR / f"{args.output_prefix}_oracle_action_regret.parquet"
    out_csv = RESULTS_DIR / f"{args.output_prefix}_oracle_action_regret.csv"
    dataset.to_parquet(out_parquet, index=False)
    dataset.to_csv(out_csv, index=False)

    print(f"Wrote {len(dataset)} rows")
    print(f"Parquet: {out_parquet}")
    print(f"CSV:     {out_csv}")
    if not dataset.empty:
        print(
            dataset[
                [
                    "oracle_step_pnl_delta",
                    "oracle_charge_delta_kw",
                    "oracle_discharge_delta_kw",
                ]
            ]
            .describe()
            .to_string()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
