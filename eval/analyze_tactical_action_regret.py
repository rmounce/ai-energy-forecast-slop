#!/usr/bin/env python3
"""
Summarize oracle-action / action-regret datasets.

This is a lightweight companion to build_tactical_action_regret_dataset.py that focuses on
action-closeness patterns rather than immediate step PnL. In particular, it helps answer:

- How often do Hybrid and Amber exactly match the oracle first action?
- On high-FIT or export-heavy rows, which source is closer to the oracle?
- Are the main action errors under-discharge or over-charge?

It intentionally uses only the standard library so it can run in thin environments.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"


def _resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.exists():
        return path.resolve()
    candidate = RESULTS_DIR / path_arg
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not find action-regret dataset: {path_arg}")


def _f(value: str | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def _summarize_bucket(rows: list[dict[str, float]], *, bucket_name: str) -> dict[str, float | int | str]:
    n = len(rows)
    hybrid_closer = sum(1 for r in rows if r["hybrid_dist"] + 1e-9 < r["amber_dist"])
    amber_closer = sum(1 for r in rows if r["amber_dist"] + 1e-9 < r["hybrid_dist"])
    equal = n - hybrid_closer - amber_closer
    return {
        "bucket": bucket_name,
        "n_rows": n,
        "p_hybrid_closer": hybrid_closer / n if n else math.nan,
        "p_amber_closer": amber_closer / n if n else math.nan,
        "p_equal": equal / n if n else math.nan,
        "mean_hybrid_action_l1_to_oracle": _mean([r["hybrid_dist"] for r in rows]),
        "mean_amber_action_l1_to_oracle": _mean([r["amber_dist"] for r in rows]),
        "hybrid_under_discharge_rate": sum(1 for r in rows if r["oracle_discharge_delta_kw"] > 1e-6) / n if n else math.nan,
        "amber_under_discharge_rate": sum(1 for r in rows if r["oracle_vs_amber_discharge_delta_kw"] > 1e-6) / n if n else math.nan,
        "hybrid_over_charge_rate": sum(1 for r in rows if r["oracle_charge_delta_kw"] < -1e-6) / n if n else math.nan,
        "amber_over_charge_rate": sum(1 for r in rows if r["oracle_vs_amber_charge_delta_kw"] < -1e-6) / n if n else math.nan,
        "mean_oracle_discharge_delta_kw_vs_hybrid": _mean([r["oracle_discharge_delta_kw"] for r in rows]),
        "mean_oracle_discharge_delta_kw_vs_amber": _mean([r["oracle_vs_amber_discharge_delta_kw"] for r in rows]),
        "mean_oracle_charge_delta_kw_vs_hybrid": _mean([r["oracle_charge_delta_kw"] for r in rows]),
        "mean_oracle_charge_delta_kw_vs_amber": _mean([r["oracle_vs_amber_charge_delta_kw"] for r in rows]),
        "mean_oracle_export_revenue_delta_vs_hybrid": _mean([r["oracle_export_revenue_delta"] for r in rows]),
        "mean_oracle_step_pnl_delta_vs_hybrid": _mean([r["oracle_step_pnl_delta"] for r in rows]),
        "mean_observed_total_objective_regret": _mean([r["observed_total_objective_regret"] for r in rows]),
        "mean_amber_total_objective_regret": _mean([r["amber_total_objective_regret"] for r in rows]),
    }


def analyze(csv_path: Path) -> list[dict[str, float | int | str]]:
    parsed: list[dict[str, float]] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            oracle_charge = _f(row["oracle_charge_kw"])
            oracle_discharge = _f(row["oracle_discharge_kw"])
            observed_charge = _f(row["observed_charge_kw"])
            observed_discharge = _f(row["observed_discharge_kw"])
            amber_charge = _f(row.get("comparator_charge_kw"))
            amber_discharge = _f(row.get("comparator_discharge_kw"))
            parsed.append(
                {
                    "actual_feed_in_price_mwh": _f(row["actual_feed_in_price_mwh"]),
                    "actual_net_load_kw": _f(row["actual_net_load_kw"]),
                    "oracle_charge_delta_kw": _f(row["oracle_charge_delta_kw"]),
                    "oracle_discharge_delta_kw": _f(row["oracle_discharge_delta_kw"]),
                    "oracle_vs_amber_charge_delta_kw": _f(row.get("oracle_vs_comparator_charge_delta_kw")),
                    "oracle_vs_amber_discharge_delta_kw": _f(row.get("oracle_vs_comparator_discharge_delta_kw")),
                    "oracle_export_revenue_delta": _f(row["oracle_export_revenue_delta"]),
                    "oracle_step_pnl_delta": _f(row["oracle_step_pnl_delta"]),
                    "observed_total_objective_regret": _f(row.get("observed_forced_total_objective_regret")),
                    "amber_total_objective_regret": _f(row.get("comparator_forced_total_objective_regret")),
                    "hybrid_dist": abs(oracle_charge - observed_charge) + abs(oracle_discharge - observed_discharge),
                    "amber_dist": abs(oracle_charge - amber_charge) + abs(oracle_discharge - amber_discharge),
                    "oracle_exporting": 1.0 if oracle_discharge > 0 else 0.0,
                    "oracle_charging": 1.0 if oracle_charge > 0 else 0.0,
                }
            )

    buckets: list[tuple[str, list[dict[str, float]]]] = [
        ("all", parsed),
        ("fit_gte_300", [r for r in parsed if r["actual_feed_in_price_mwh"] >= 300.0]),
        ("fit_gte_500", [r for r in parsed if r["actual_feed_in_price_mwh"] >= 500.0]),
        ("negative_net_load", [r for r in parsed if r["actual_net_load_kw"] < 0.0]),
        ("oracle_exporting", [r for r in parsed if r["oracle_exporting"] > 0.0]),
        ("oracle_charging", [r for r in parsed if r["oracle_charging"] > 0.0]),
    ]
    return [_summarize_bucket(rows, bucket_name=name) for name, rows in buckets if rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV path or filename under eval/results")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    csv_path = _resolve_path(args.csv)
    summary_rows = analyze(csv_path)

    out_csv = RESULTS_DIR / f"{args.output_prefix}_oracle_action_summary.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] wrote {out_csv}")
    for row in summary_rows:
        print(
            f"{row['bucket']}: n={row['n_rows']} "
            f"hybrid_closer={row['p_hybrid_closer']:.3f} "
            f"amber_closer={row['p_amber_closer']:.3f} "
            f"equal={row['p_equal']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
