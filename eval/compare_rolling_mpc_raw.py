#!/usr/bin/env python3
"""
Compare two rolling MPC raw parquet outputs and report whether dispatch changed.

This is meant as a cheap diagnostic before launching long follow-up runs:
if the proposed control variant does not materially change `charge_kw`,
`discharge_kw`, `soc_kwh`, or terminal contract columns on a short window,
it is unlikely to justify a long backtest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"

KEY_COLS = ["time", "source"]
DEFAULT_COMPARE_COLS = [
    "charge_kw",
    "discharge_kw",
    "soc_prev_kwh",
    "soc_kwh",
    "step_pnl",
    "terminal_energy_value_per_kwh",
    "strategic_soc_target_kwh",
    "strategic_soc_target_qhi_kwh",
    "strategic_target_gap_kwh",
    "dynamic_target_uplift_kwh",
    "dynamic_terminal_adder_per_kwh",
    "extra_terminal_energy_value_per_kwh",
    "extra_terminal_energy_floor_kwh",
    "extra_terminal_energy_cap_kwh",
    "extra_terminal_energy_kwh",
    "min_terminal_soc_kwh",
    "max_terminal_soc_kwh",
]


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values(KEY_COLS, kind="stable").reset_index(drop=True)


def _resolve_path(raw_arg: str) -> Path:
    p = Path(raw_arg)
    if p.exists():
        return p.resolve()
    p_results = RESULTS_DIR / raw_arg
    if p_results.exists():
        return p_results.resolve()
    raise FileNotFoundError(f"Could not find parquet: {raw_arg}")


def _series_max_abs(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if vals.size == 0:
        return 0.0
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.max(np.abs(vals)))


def _series_nonzero_count(s: pd.Series, tol: float) -> int:
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if vals.size == 0:
        return 0
    mask = np.isfinite(vals) & (np.abs(vals) > tol)
    return int(mask.sum())


def _series_mean_abs(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if vals.size == 0:
        return 0.0
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(vals)))


def compare_raw(
    lhs_path: Path,
    rhs_path: Path,
    compare_cols: list[str],
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lhs = _load_raw(lhs_path)
    rhs = _load_raw(rhs_path)

    compare_cols = [c for c in compare_cols if c in lhs.columns and c in rhs.columns]
    if not compare_cols:
        raise ValueError("No overlapping compare columns found in the two parquet files")

    merged = lhs[KEY_COLS + compare_cols].merge(
        rhs[KEY_COLS + compare_cols],
        on=KEY_COLS,
        how="inner",
        suffixes=("_lhs", "_rhs"),
    )
    if len(merged) != len(lhs) or len(merged) != len(rhs):
        raise ValueError(
            f"Raw files do not align cleanly on {KEY_COLS}: "
            f"lhs={len(lhs)} rows rhs={len(rhs)} rows merged={len(merged)} rows"
        )

    summary_rows: list[dict] = []
    for col in compare_cols:
        lhs_col = f"{col}_lhs"
        rhs_col = f"{col}_rhs"
        delta = pd.to_numeric(merged[lhs_col], errors="coerce") - pd.to_numeric(merged[rhs_col], errors="coerce")
        summary_rows.append(
            {
                "column": col,
                "max_abs_delta": _series_max_abs(delta),
                "mean_abs_delta": _series_mean_abs(delta),
                "changed_steps": _series_nonzero_count(delta, tolerance),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["changed_steps", "max_abs_delta", "column"], ascending=[False, False, True])

    delta_df = merged[KEY_COLS].copy()
    for col in compare_cols:
        delta_df[f"{col}_delta"] = pd.to_numeric(merged[f"{col}_lhs"], errors="coerce") - pd.to_numeric(
            merged[f"{col}_rhs"], errors="coerce"
        )

    return summary_df.reset_index(drop=True), delta_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs", required=True, help="Left raw parquet path or filename under eval/results")
    parser.add_argument("--rhs", required=True, help="Right raw parquet path or filename under eval/results")
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COMPARE_COLS),
        help="Comma-separated columns to compare",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute delta threshold used when counting changed steps",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix for saving summary/delta CSVs under eval/results",
    )
    args = parser.parse_args()

    lhs_path = _resolve_path(args.lhs)
    rhs_path = _resolve_path(args.rhs)
    compare_cols = [c.strip() for c in args.columns.split(",") if c.strip()]

    summary_df, delta_df = compare_raw(
        lhs_path=lhs_path,
        rhs_path=rhs_path,
        compare_cols=compare_cols,
        tolerance=args.tolerance,
    )

    print(f"Compare raw MPC outputs")
    print(f"  lhs: {lhs_path.relative_to(ROOT)}")
    print(f"  rhs: {rhs_path.relative_to(ROOT)}")
    print(f"  tolerance: {args.tolerance:g}")
    print(summary_df.to_string(index=False))

    if args.output_prefix:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = RESULTS_DIR / f"{args.output_prefix}_summary.csv"
        delta_path = RESULTS_DIR / f"{args.output_prefix}_deltas.csv"
        summary_df.to_csv(summary_path, index=False)
        delta_df.to_csv(delta_path, index=False)
        print(f"Saved summary -> {summary_path.relative_to(ROOT)}")
        print(f"Saved deltas -> {delta_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
