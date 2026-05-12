#!/usr/bin/env python3
"""Compare load-source rolling-MPC eval runs side-by-side.

Reads `*_summary.csv` files for a set of `--output-prefix` runs and prints a
tidy table indexed by (price_source, load_source). Headline columns: total
PnL, mean/day, terminal SoC, repaired-curve count, under-prep counts by
Adelaide bucket. Use this after a batch of load-forecast-source counterfactual
runs to read the verdict at a glance.

Example:
    ./.venv/bin/python eval/compare_loadsrc_runs.py \\
        --prefix loadsrc_A_3day_actual \\
        --prefix loadsrc_A_3day_lgbm_load_log \\
        --prefix loadsrc_A_3day_tft_load_log
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"

HEADLINE_COLS = [
    "source",
    "load_forecast_source",
    "steps",
    "total_pnl",
    "mean_per_day",
    "soc_final_kwh",
    "repaired_invalid_curve",
    "load_forecast_unavailable_steps",
    "load_under_prep_total_steps",
    "load_under_prep_mean_kw",
    "load_under_prep_p90_kw",
    "load_under_prep_steps_overnight",
    "load_under_prep_steps_morning",
    "load_under_prep_steps_solar",
    "load_under_prep_steps_evening",
    "load_under_prep_steps_late",
]


def _resolve_summary(prefix: str) -> Path:
    cand = RESULTS_DIR / f"{prefix}_summary.csv"
    if cand.exists():
        return cand
    cand_local = Path(f"{prefix}_summary.csv")
    if cand_local.exists():
        return cand_local
    raise FileNotFoundError(
        f"No summary CSV for prefix '{prefix}' (looked for "
        f"{cand} and {cand_local})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefix",
        action="append",
        required=True,
        help="Output prefix used at run time; can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, also write the comparison to this CSV path.",
    )
    args = parser.parse_args()

    frames = []
    for prefix in args.prefix:
        path = _resolve_summary(prefix)
        df = pd.read_csv(path)
        available_cols = [c for c in HEADLINE_COLS if c in df.columns]
        sub = df[available_cols].copy()
        sub.insert(0, "prefix", prefix)
        frames.append(sub)

    big = pd.concat(frames, ignore_index=True)
    big = big.sort_values(
        ["source", "load_forecast_source"], kind="stable"
    ).reset_index(drop=True)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print(big.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        big.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
