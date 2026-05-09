"""Analyze whether high/capped PD7Day prices materialise in actual RRP.

This is Phase alpha-prime Step 7's first diagnostic. It does not train a model.
It answers the practical question: when PD7Day emits a high tail value, how often
does the realised interval become an actual high-price interval?

Outputs are CSV summaries under ``eval/results`` by default:

- ``*_overall.csv``
- ``*_by_horizon_bucket.csv``
- ``*_by_target_hour_utc.csv``
- ``*_by_run_hour_utc.csv``
- ``*_by_cap_run_length_bucket.csv``
- ``*_by_cap_run_age_bucket.csv``
- ``*_forecast_thresholds.csv``
- ``*_joined_rows.parquet`` when ``--save-joined`` is set
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PD7DAY = REPO_ROOT / "data" / "parquet" / "aemo_pd7day_sa1.parquet"
DEFAULT_ACTUALS = REPO_ROOT / "data" / "parquet" / "actuals_sa1.parquet"
DEFAULT_OUTPUT_PREFIX = REPO_ROOT / "eval" / "results" / "pd7day_cap_materialisation"


HORIZON_BUCKETS = [
    0.0,
    6.0,
    24.0,
    48.0,
    72.0,
    120.0,
    np.inf,
]
HORIZON_LABELS = [
    "0-6h",
    "6-24h",
    "24-48h",
    "48-72h",
    "72-120h",
    "120h+",
]

RUN_LENGTH_BINS = [0, 1, 2, 4, 8, 16, 32, np.inf]
RUN_LENGTH_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]

RUN_AGE_BINS = [0, 1, 2, 4, 8, 16, 32, np.inf]
RUN_AGE_LABELS = ["first", "2nd", "3-4", "5-8", "9-16", "17-32", "33+"]


def _parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("At least one threshold is required")
    return out


def _load_joined(pd7day_path: Path, actuals_path: Path) -> pd.DataFrame:
    pd7 = pd.read_parquet(pd7day_path).copy()
    actuals = pd.read_parquet(actuals_path, columns=["time", "rrp"]).copy()

    pd7["interval_dt"] = pd.to_datetime(pd7["interval_dt"], utc=True)
    pd7["run_time"] = pd.to_datetime(pd7["run_time"], utc=True)
    actuals["interval_dt"] = pd.to_datetime(actuals["time"], utc=True)
    actuals = actuals.rename(columns={"rrp": "actual_rrp"})

    joined = pd7.merge(
        actuals[["interval_dt", "actual_rrp"]],
        on="interval_dt",
        how="inner",
    )
    joined = joined.rename(columns={"rrp": "pd7_rrp"})
    joined = joined.dropna(subset=["pd7_rrp", "actual_rrp"]).copy()
    joined["horizon_hours"] = (
        joined["interval_dt"] - joined["run_time"]
    ).dt.total_seconds() / 3600.0
    joined = joined[joined["horizon_hours"] >= 0.0].copy()
    joined["target_hour_utc"] = joined["interval_dt"].dt.hour
    joined["run_hour_utc"] = joined["run_time"].dt.hour
    joined["horizon_bucket"] = pd.cut(
        joined["horizon_hours"],
        bins=HORIZON_BUCKETS,
        labels=HORIZON_LABELS,
        right=False,
    )
    return joined.sort_values(["run_time", "interval_dt"]).reset_index(drop=True)


def _add_cap_run_features(df: pd.DataFrame, cap_threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["cap_flag"] = df["pd7_rrp"] >= float(cap_threshold)
    df["cap_run_id"] = -1
    df["cap_run_length"] = 0
    df["cap_run_pos"] = 0

    for _, idx in df.groupby("run_time", sort=False).groups.items():
        run_idx = np.asarray(idx)
        flags = df.loc[run_idx, "cap_flag"].to_numpy(dtype=bool)
        if len(flags) == 0:
            continue
        starts = np.r_[True, flags[1:] != flags[:-1]]
        group_ids = np.cumsum(starts) - 1
        for group_id in np.unique(group_ids[flags]):
            pos = np.where(group_ids == group_id)[0]
            pos = pos[flags[pos]]
            if len(pos) == 0:
                continue
            abs_idx = run_idx[pos]
            df.loc[abs_idx, "cap_run_id"] = int(group_id)
            df.loc[abs_idx, "cap_run_length"] = int(len(abs_idx))
            df.loc[abs_idx, "cap_run_pos"] = np.arange(1, len(abs_idx) + 1)

    df["cap_run_length_bucket"] = pd.cut(
        df["cap_run_length"],
        bins=RUN_LENGTH_BINS,
        labels=RUN_LENGTH_LABELS,
        right=True,
    )
    df["cap_run_age_bucket"] = pd.cut(
        df["cap_run_pos"],
        bins=RUN_AGE_BINS,
        labels=RUN_AGE_LABELS,
        right=True,
    )
    return df


def _summarise(
    df: pd.DataFrame,
    group_cols: list[str],
    actual_thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False, observed=False)
    else:
        grouped = [("all", df)]

    for key, grp in grouped:
        if len(grp) == 0:
            continue
        row: dict[str, object] = {}
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            for col, value in zip(group_cols, key):
                row[col] = value
        else:
            row["group"] = "all"

        row["rows"] = int(len(grp))
        row["runs"] = int(grp["run_time"].nunique())
        row["intervals"] = int(grp["interval_dt"].nunique())
        row["pd7_mean"] = float(grp["pd7_rrp"].mean())
        row["pd7_median"] = float(grp["pd7_rrp"].median())
        row["actual_mean"] = float(grp["actual_rrp"].mean())
        row["actual_median"] = float(grp["actual_rrp"].median())
        row["bias_mean"] = float((grp["pd7_rrp"] - grp["actual_rrp"]).mean())
        row["mae"] = float((grp["pd7_rrp"] - grp["actual_rrp"]).abs().mean())
        for threshold in actual_thresholds:
            name = str(threshold).replace(".", "p")
            row[f"actual_ge_{name}_rate"] = float((grp["actual_rrp"] >= threshold).mean())
            row[f"actual_ge_{name}_count"] = int((grp["actual_rrp"] >= threshold).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def _forecast_threshold_summary(
    df: pd.DataFrame,
    forecast_thresholds: list[float],
    actual_thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    for forecast_threshold in forecast_thresholds:
        flagged = df[df["pd7_rrp"] >= forecast_threshold]
        if flagged.empty:
            row: dict[str, object] = {
                "forecast_threshold": forecast_threshold,
                "rows": 0,
                "runs": 0,
                "intervals": 0,
            }
            for actual_threshold in actual_thresholds:
                name = str(actual_threshold).replace(".", "p")
                row[f"actual_ge_{name}_rate"] = np.nan
                row[f"actual_ge_{name}_count"] = 0
            rows.append(row)
            continue
        row = {
            "forecast_threshold": forecast_threshold,
            "rows": int(len(flagged)),
            "runs": int(flagged["run_time"].nunique()),
            "intervals": int(flagged["interval_dt"].nunique()),
            "pd7_mean": float(flagged["pd7_rrp"].mean()),
            "pd7_median": float(flagged["pd7_rrp"].median()),
            "actual_mean": float(flagged["actual_rrp"].mean()),
            "actual_median": float(flagged["actual_rrp"].median()),
            "bias_mean": float((flagged["pd7_rrp"] - flagged["actual_rrp"]).mean()),
            "mae": float((flagged["pd7_rrp"] - flagged["actual_rrp"]).abs().mean()),
        }
        for actual_threshold in actual_thresholds:
            name = str(actual_threshold).replace(".", "p")
            row[f"actual_ge_{name}_rate"] = float((flagged["actual_rrp"] >= actual_threshold).mean())
            row[f"actual_ge_{name}_count"] = int((flagged["actual_rrp"] >= actual_threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pd7day", type=Path, default=DEFAULT_PD7DAY)
    parser.add_argument("--actuals", type=Path, default=DEFAULT_ACTUALS)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument(
        "--cap-threshold",
        type=float,
        default=300.0,
        help="Forecast RRP threshold used to define cap/high-flagged rows for bucket summaries.",
    )
    parser.add_argument(
        "--forecast-thresholds",
        default="150,300,500,980.89,1000,20300",
        help="Comma-separated forecast thresholds for materialisation summary.",
    )
    parser.add_argument(
        "--actual-thresholds",
        default="150,300,500",
        help="Comma-separated actual RRP thresholds to report as materialised rates.",
    )
    parser.add_argument("--save-joined", action="store_true")
    args = parser.parse_args()

    forecast_thresholds = _parse_thresholds(args.forecast_thresholds)
    actual_thresholds = _parse_thresholds(args.actual_thresholds)

    joined = _load_joined(args.pd7day, args.actuals)
    joined = _add_cap_run_features(joined, args.cap_threshold)
    flagged = joined[joined["cap_flag"]].copy()

    print(
        f"Joined {len(joined):,} materialised PD7Day rows from "
        f"{joined['run_time'].nunique():,} runs; "
        f"{len(flagged):,} rows have PD7Day >= {args.cap_threshold:g} $/MWh."
    )
    print(
        f"Actual coverage: {joined['interval_dt'].min()} -> {joined['interval_dt'].max()}"
    )

    prefix: Path = args.output_prefix
    _write_csv(_summarise(flagged, [], actual_thresholds), prefix.with_name(prefix.name + "_overall.csv"))
    _write_csv(
        _summarise(flagged, ["horizon_bucket"], actual_thresholds),
        prefix.with_name(prefix.name + "_by_horizon_bucket.csv"),
    )
    _write_csv(
        _summarise(flagged, ["target_hour_utc"], actual_thresholds),
        prefix.with_name(prefix.name + "_by_target_hour_utc.csv"),
    )
    _write_csv(
        _summarise(flagged, ["run_hour_utc"], actual_thresholds),
        prefix.with_name(prefix.name + "_by_run_hour_utc.csv"),
    )
    _write_csv(
        _summarise(flagged, ["cap_run_length_bucket"], actual_thresholds),
        prefix.with_name(prefix.name + "_by_cap_run_length_bucket.csv"),
    )
    _write_csv(
        _summarise(flagged, ["cap_run_age_bucket"], actual_thresholds),
        prefix.with_name(prefix.name + "_by_cap_run_age_bucket.csv"),
    )
    _write_csv(
        _forecast_threshold_summary(joined, forecast_thresholds, actual_thresholds),
        prefix.with_name(prefix.name + "_forecast_thresholds.csv"),
    )

    if args.save_joined:
        path = prefix.with_name(prefix.name + "_joined_rows.parquet")
        joined.to_parquet(path, index=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
