#!/usr/bin/env python3
"""
Build short-horizon state-transition labels from rolling MPC raw outputs.

The forced-prefix diagnostics showed the remaining Amber edge is a 30-60 minute
inventory/path-quality effect, not a first-action imitation problem. This script
creates the first small dataset for that modeling branch:

- solve the realized-future tariffed oracle for each target row
- summarize oracle, target, and optional comparator paths over N-step prefixes
- emit state-transition labels such as oracle-vs-target SoC delta, churn, import,
  export, and prefix PnL differences

The output is intended as a label/search dataset. Columns prefixed with
`future_` or `oracle_` are not production features.
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

from eval.dispatch_simulator import (  # noqa: E402
    CAPACITY_KWH,
    DEG_PER_KWH,
    EFF_C,
    EFF_D,
    INTERVAL_H,
    compute_soc_trajectory,
    solve_lp_dispatch,
)


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


def _optional_float(row: object, name: str) -> float:
    value = getattr(row, name, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _path_metrics(
    *,
    charge_kw: np.ndarray,
    discharge_kw: np.ndarray,
    market_df: pd.DataFrame,
    soc_prev_kwh: float,
    steps: int,
) -> dict[str, float]:
    actual_steps = min(int(steps), len(charge_kw), len(discharge_kw), len(market_df))
    if actual_steps <= 0:
        return {
            "steps": 0.0,
            "charge_kwh": np.nan,
            "discharge_kwh": np.nan,
            "throughput_kwh": np.nan,
            "import_kwh": np.nan,
            "export_kwh": np.nan,
            "import_cost": np.nan,
            "export_revenue": np.nan,
            "degradation_cost": np.nan,
            "step_pnl": np.nan,
            "soc_delta_kwh": np.nan,
            "end_soc_kwh": np.nan,
        }

    c = np.asarray(charge_kw[:actual_steps], dtype=np.float64)
    d = np.asarray(discharge_kw[:actual_steps], dtype=np.float64)
    future = market_df.head(actual_steps)
    net_load = future["actual_net_load_kw"].to_numpy(dtype=np.float64)
    import_price = future["actual_general_price_mwh"].to_numpy(dtype=np.float64) / 1000.0
    export_price = future["actual_feed_in_price_mwh"].to_numpy(dtype=np.float64) / 1000.0

    grid_kw = net_load + c - d * EFF_D
    grid_import_kw = np.maximum(grid_kw, 0.0)
    grid_export_kw = np.maximum(-grid_kw, 0.0)
    import_cost = float(np.sum(grid_import_kw * import_price) * INTERVAL_H)
    export_revenue = float(np.sum(grid_export_kw * export_price) * INTERVAL_H)
    degradation_cost = float(np.sum(DEG_PER_KWH * (c * EFF_C + d)) * INTERVAL_H)
    soc = compute_soc_trajectory(c, d, float(soc_prev_kwh))
    end_soc = float(soc[-1]) if len(soc) else float(soc_prev_kwh)

    return {
        "steps": float(actual_steps),
        "charge_kwh": float(np.sum(c) * INTERVAL_H),
        "discharge_kwh": float(np.sum(d) * INTERVAL_H),
        "throughput_kwh": float(np.sum(c * EFF_C + d) * INTERVAL_H),
        "import_kwh": float(np.sum(grid_import_kw) * INTERVAL_H),
        "export_kwh": float(np.sum(grid_export_kw) * INTERVAL_H),
        "import_cost": import_cost,
        "export_revenue": export_revenue,
        "degradation_cost": degradation_cost,
        "step_pnl": export_revenue - import_cost - degradation_cost,
        "soc_delta_kwh": end_soc - float(soc_prev_kwh),
        "end_soc_kwh": end_soc,
    }


def _prefix_from_source(source_df: pd.DataFrame, ts: pd.Timestamp, steps: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = source_df[source_df["time"] >= ts].head(steps)
    return (
        prefix["charge_kw"].to_numpy(dtype=np.float64, copy=True),
        prefix["discharge_kw"].to_numpy(dtype=np.float64, copy=True),
    )


def _add_metrics(out: dict[str, float | str | int | bool | None], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        out[f"{prefix}_{key}"] = value


def _add_delta_metrics(
    out: dict[str, float | str | int | bool | None],
    *,
    prefix: str,
    left: dict[str, float],
    right: dict[str, float],
) -> None:
    for key, left_value in left.items():
        right_value = right.get(key, np.nan)
        out[f"{prefix}_{key}"] = float(left_value - right_value)


def build_dataset(
    raw_df: pd.DataFrame,
    *,
    target_source: str,
    comparator_source: str | None,
    horizons: list[int],
    max_rows: int | None = None,
    progress_every_rows: int = 0,
    soc_finite_diff_kwh: float = 0.0,
) -> pd.DataFrame:
    raw_df = raw_df.copy()
    raw_df["time"] = pd.to_datetime(raw_df["time"], utc=True)
    raw_df = raw_df.sort_values(["source", "time"], kind="stable").reset_index(drop=True)

    market_df = (
        raw_df[["time", "actual_general_price_mwh", "actual_feed_in_price_mwh", "actual_net_load_kw"]]
        .drop_duplicates("time")
        .sort_values("time", kind="stable")
        .set_index("time")
    )

    target_path_df = raw_df[raw_df["source"] == target_source].copy().sort_values("time", kind="stable").reset_index(drop=True)
    if target_path_df.empty:
        raise ValueError(f"No rows found for target source: {target_source}")
    target_df = target_path_df
    if max_rows is not None:
        target_df = target_df.head(max(0, int(max_rows)))

    comparator_df: pd.DataFrame | None = None
    if comparator_source:
        comparator_df = (
            raw_df[raw_df["source"] == comparator_source]
            .copy()
            .sort_values("time", kind="stable")
            .reset_index(drop=True)
        )
        if comparator_df.empty:
            raise ValueError(f"No rows found for comparator source: {comparator_source}")

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
        net_load_kw = future["actual_net_load_kw"].to_numpy(dtype=np.float64, copy=True)

        oracle = solve_lp_dispatch(
            import_prices_mwh.copy(),
            float(row.soc_prev_kwh),
            import_prices_mwh=import_prices_mwh.copy(),
            export_prices_mwh=export_prices_mwh.copy(),
            net_load_forecast_kw=net_load_kw.copy(),
            terminal_energy_value_per_kwh=float(row.terminal_energy_value_per_kwh),
            extra_terminal_energy_value_per_kwh=float(getattr(row, "extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
            extra_terminal_energy_floor_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_floor_kwh", None)),
            extra_terminal_energy_cap_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_cap_kwh", None)),
            min_terminal_soc_kwh=_none_if_nan(getattr(row, "min_terminal_soc_kwh", None)),
            max_terminal_soc_kwh=_none_if_nan(getattr(row, "max_terminal_soc_kwh", None)),
        )
        if not oracle["success"]:
            continue

        soc_finite_diff_value = float("nan")
        if soc_finite_diff_kwh > 0.0:
            soc_delta = min(float(soc_finite_diff_kwh), CAPACITY_KWH - float(row.soc_prev_kwh))
            if soc_delta > 1e-9:
                oracle_plus = solve_lp_dispatch(
                    import_prices_mwh.copy(),
                    float(row.soc_prev_kwh) + soc_delta,
                    import_prices_mwh=import_prices_mwh.copy(),
                    export_prices_mwh=export_prices_mwh.copy(),
                    net_load_forecast_kw=net_load_kw.copy(),
                    terminal_energy_value_per_kwh=float(row.terminal_energy_value_per_kwh),
                    extra_terminal_energy_value_per_kwh=float(getattr(row, "extra_terminal_energy_value_per_kwh", 0.0) or 0.0),
                    extra_terminal_energy_floor_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_floor_kwh", None)),
                    extra_terminal_energy_cap_kwh=_none_if_nan(getattr(row, "extra_terminal_energy_cap_kwh", None)),
                    min_terminal_soc_kwh=_none_if_nan(getattr(row, "min_terminal_soc_kwh", None)),
                    max_terminal_soc_kwh=_none_if_nan(getattr(row, "max_terminal_soc_kwh", None)),
                )
                if oracle_plus["success"]:
                    soc_finite_diff_value = (float(oracle["objective_value"]) - float(oracle_plus["objective_value"])) / soc_delta

        for horizon_steps in horizons:
            horizon_steps = int(horizon_steps)
            target_charge, target_discharge = _prefix_from_source(target_path_df, ts, horizon_steps)
            oracle_metrics = _path_metrics(
                charge_kw=oracle["charge_kw"],
                discharge_kw=oracle["discharge_kw"],
                market_df=future,
                soc_prev_kwh=float(row.soc_prev_kwh),
                steps=horizon_steps,
            )
            target_metrics = _path_metrics(
                charge_kw=target_charge,
                discharge_kw=target_discharge,
                market_df=future,
                soc_prev_kwh=float(row.soc_prev_kwh),
                steps=horizon_steps,
            )

            out: dict[str, float | str | int | bool | None] = {
                "time": ts,
                "target_source": target_source,
                "comparator_source": comparator_source or "",
                "horizon_steps": horizon_steps,
                "horizon_minutes": horizon_steps * 5,
                "horizon_steps_available": int(len(future)),
                "soc_prev_kwh": float(row.soc_prev_kwh),
                "actual_general_price_mwh": float(row.actual_general_price_mwh),
                "actual_feed_in_price_mwh": float(row.actual_feed_in_price_mwh),
                "actual_net_load_kw": float(row.actual_net_load_kw),
                "bucket_fit_lt_300_negload": bool(
                    float(row.actual_feed_in_price_mwh) < 300.0 and float(row.actual_net_load_kw) < 0.0
                ),
                "bucket_fit_lt_300_nonnegload": bool(
                    float(row.actual_feed_in_price_mwh) < 300.0 and float(row.actual_net_load_kw) >= 0.0
                ),
                "bucket_fit_gte_300": bool(float(row.actual_feed_in_price_mwh) >= 300.0),
                "forecast_feed_in_step0_mwh": _optional_float(row, "forecast_feed_in_step0_mwh"),
                "forecast_feed_in_mean_next_1h_mwh": _optional_float(row, "forecast_feed_in_mean_next_1h_mwh"),
                "forecast_feed_in_mean_next_4h_mwh": _optional_float(row, "forecast_feed_in_mean_next_4h_mwh"),
                "forecast_buy_mean_next_1h_mwh": _optional_float(row, "forecast_buy_mean_next_1h_mwh"),
                "forecast_buy_mean_next_4h_mwh": _optional_float(row, "forecast_buy_mean_next_4h_mwh"),
                "forecast_sell_mean_next_1h_mwh": _optional_float(row, "forecast_sell_mean_next_1h_mwh"),
                "forecast_sell_mean_next_4h_mwh": _optional_float(row, "forecast_sell_mean_next_4h_mwh"),
                "strategic_soc_target_kwh": _optional_float(row, "strategic_soc_target_kwh"),
                "terminal_energy_value_per_kwh": float(row.terminal_energy_value_per_kwh),
                "oracle_success": bool(oracle["success"]),
                "oracle_objective_value": float(oracle["objective_value"]),
                "oracle_initial_soc_shadow_price_per_kwh": float(oracle["initial_soc_shadow_price_per_kwh"]),
                "oracle_initial_soc_finite_diff_value_per_kwh": soc_finite_diff_value,
                "future_mean_feed_in_price_mwh": float(future.head(horizon_steps)["actual_feed_in_price_mwh"].mean()),
                "future_mean_general_price_mwh": float(future.head(horizon_steps)["actual_general_price_mwh"].mean()),
                "future_mean_net_load_kw": float(future.head(horizon_steps)["actual_net_load_kw"].mean()),
            }
            _add_metrics(out, "oracle", oracle_metrics)
            _add_metrics(out, "target", target_metrics)
            _add_delta_metrics(out, prefix="oracle_minus_target", left=oracle_metrics, right=target_metrics)

            if comparator_df is not None:
                comp_charge, comp_discharge = _prefix_from_source(comparator_df, ts, horizon_steps)
                comp_metrics = _path_metrics(
                    charge_kw=comp_charge,
                    discharge_kw=comp_discharge,
                    market_df=future,
                    soc_prev_kwh=float(row.soc_prev_kwh),
                    steps=horizon_steps,
                )
                _add_metrics(out, "comparator", comp_metrics)
                _add_delta_metrics(out, prefix="comparator_minus_target", left=comp_metrics, right=target_metrics)
                _add_delta_metrics(out, prefix="oracle_minus_comparator", left=oracle_metrics, right=comp_metrics)

            rows.append(out)

        if progress_every_rows > 0 and (i % progress_every_rows == 0 or i == total_rows):
            elapsed = time.perf_counter() - start_t
            rate = i / elapsed if elapsed > 0 else float("nan")
            remaining = max(total_rows - i, 0)
            eta_sec = remaining / rate if np.isfinite(rate) and rate > 0 else float("nan")
            eta_str = f"{eta_sec/60:.1f}m" if np.isfinite(eta_sec) else "?"
            print(
                f"[progress] {i}/{total_rows} rows ({100.0 * i / total_rows:.1f}%) "
                f"elapsed={elapsed/60:.1f}m eta={eta_str}",
                flush=True,
            )

    return pd.DataFrame(rows)


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not horizons:
        raise argparse.ArgumentTypeError("At least one horizon step is required")
    if any(step <= 0 for step in horizons):
        raise argparse.ArgumentTypeError("Horizon steps must be positive")
    return horizons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Rolling raw parquet path or filename under eval/results")
    parser.add_argument("--target-source", default="model_a_hybrid")
    parser.add_argument("--comparator-source", default="amber_tactical_hybrid_strategic")
    parser.add_argument("--horizons", type=_parse_horizons, default=_parse_horizons("6,12"), help="Comma-separated 5-min step counts")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit")
    parser.add_argument("--progress-every-rows", type=int, default=0)
    parser.add_argument(
        "--soc-finite-diff-kwh",
        type=float,
        default=0.0,
        help="Optional +kWh initial-SoC finite difference value label. Adds one LP solve per row when >0.",
    )
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    raw_df = pd.read_parquet(_resolve_path(args.raw))
    dataset = build_dataset(
        raw_df,
        target_source=args.target_source,
        comparator_source=args.comparator_source or None,
        horizons=args.horizons,
        max_rows=args.max_rows,
        progress_every_rows=max(0, int(args.progress_every_rows)),
        soc_finite_diff_kwh=max(0.0, float(args.soc_finite_diff_kwh)),
    )

    out_parquet = RESULTS_DIR / f"{args.output_prefix}_state_transition_labels.parquet"
    out_csv = RESULTS_DIR / f"{args.output_prefix}_state_transition_labels.csv"
    dataset.to_parquet(out_parquet, index=False)
    dataset.to_csv(out_csv, index=False)
    print(f"Wrote {len(dataset)} rows")
    print(f"Parquet: {out_parquet}")
    print(f"CSV:     {out_csv}")
    if not dataset.empty:
        cols = [
            "horizon_steps",
            "oracle_minus_target_soc_delta_kwh",
            "oracle_minus_target_throughput_kwh",
            "oracle_minus_target_export_kwh",
            "oracle_minus_target_step_pnl",
            "oracle_initial_soc_finite_diff_value_per_kwh",
            "oracle_initial_soc_shadow_price_per_kwh",
        ]
        print(dataset[cols].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
