#!/usr/bin/env python3
"""Build and validate PD-direct empirical residual bands.

The band table estimates residual quantiles for:

    actual_RRP - debiased_PD_q50

It is intentionally a small, leakage-safe calibration artifact for PD-direct, not a new
trained price model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pd_direct_baseline import (
    ACTUALS_PARQUET,
    DEBIASED_PARQUET,
    RESIDUAL_BAND_CAP_DEFAULT,
    RESIDUAL_BAND_MIN_SAMPLES_DEFAULT,
    RESIDUAL_BANDS_PARQUET,
    apply_pd_residual_bands,
    horizon_bucket_from_hours,
    pd_level_bucket,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "eval" / "results"


def _load_residual_frame(debiased_path: Path, actuals_path: Path,
                         actuals_shift_min: int = 0) -> pd.DataFrame:
    """Join OOF debiased PD with actuals to compute residuals.

    `actuals_shift_min` corrects the interval-end (forecast) vs interval-start
    (actuals, CQ-aggregated) mismatch documented in
    `docs/timestamp_convention_audit_2026-05-11.md`. Default 0 preserves the
    canonical band table's join convention. Set to 30 to align actuals to the
    forecast's interval-end convention — this is required when building bands
    from the aligned OOF parquet (output of `train_pd_debiaser.py
    --actuals-shift-min=30`), otherwise the residuals would be measured against
    the wrong half-hour.
    """
    deb = pd.read_parquet(debiased_path)
    actuals = pd.read_parquet(actuals_path)[["time", "rrp"]].rename(
        columns={"time": "interval_dt", "rrp": "actual_rrp"}
    )
    deb["run_time"] = pd.to_datetime(deb["run_time"], utc=True)
    deb["interval_dt"] = pd.to_datetime(deb["interval_dt"], utc=True)
    actuals["interval_dt"] = pd.to_datetime(actuals["interval_dt"], utc=True)
    if actuals_shift_min:
        actuals["interval_dt"] = actuals["interval_dt"] + pd.Timedelta(minutes=int(actuals_shift_min))
    df = deb.merge(actuals, on="interval_dt", how="inner")
    df = df.dropna(subset=["oof_debiased_rrp", "actual_rrp"])
    df["horizon_hours"] = (
        df["interval_dt"] - df["run_time"]
    ).dt.total_seconds() / 3600.0
    df = df[df["horizon_hours"] >= 0.0].copy()
    df["horizon_bucket"] = df["horizon_hours"].map(horizon_bucket_from_hours)
    df["hod_30m"] = df["interval_dt"].dt.hour * 2 + (df["interval_dt"].dt.minute // 30)
    df["pd_level_bucket"] = df["oof_debiased_rrp"].map(pd_level_bucket)
    df["day_type"] = np.where(df["interval_dt"].dt.weekday >= 5, "weekend", "weekday")
    df["residual"] = df["actual_rrp"].astype(float) - df["oof_debiased_rrp"].astype(float)
    return df


def _summarise_groups(
    df: pd.DataFrame,
    keys: list[str],
    granularity: str,
    *,
    residual_cap: float,
    min_samples: int,
) -> pd.DataFrame:
    clipped = df.assign(residual_clipped=df["residual"].clip(-residual_cap, residual_cap))
    rows = []
    for key_vals, grp in clipped.groupby(keys, sort=True, dropna=False):
        if len(grp) < min_samples:
            continue
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        row = dict(zip(keys, key_vals, strict=True))
        row.update(
            {
                "granularity": granularity,
                "n": int(len(grp)),
                "q10": float(grp["residual_clipped"].quantile(0.10)),
                "q20": float(grp["residual_clipped"].quantile(0.20)),
                "q50": float(grp["residual_clipped"].quantile(0.50)),
                "q80": float(grp["residual_clipped"].quantile(0.80)),
                "q90": float(grp["residual_clipped"].quantile(0.90)),
                "residual_cap": float(residual_cap),
                "min_samples": int(min_samples),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_residual_bands(
    residual_df: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    residual_cap: float = RESIDUAL_BAND_CAP_DEFAULT,
    min_samples: int = RESIDUAL_BAND_MIN_SAMPLES_DEFAULT,
    min_daytype_samples: int = 50,
) -> pd.DataFrame:
    train = residual_df[residual_df["interval_dt"] < train_end].copy()
    pieces = [
        _summarise_groups(
            train,
            ["horizon_bucket", "hod_30m", "pd_level_bucket", "day_type"],
            "horizon_hod_level_daytype",
            residual_cap=residual_cap,
            min_samples=min_daytype_samples,
        ),
        _summarise_groups(
            train,
            ["horizon_bucket", "hod_30m", "pd_level_bucket"],
            "horizon_hod_level",
            residual_cap=residual_cap,
            min_samples=min_samples,
        ),
        _summarise_groups(
            train,
            ["horizon_bucket", "hod_30m"],
            "horizon_hod",
            residual_cap=residual_cap,
            min_samples=min_samples,
        ),
        _summarise_groups(
            train,
            ["horizon_bucket", "pd_level_bucket"],
            "horizon_level",
            residual_cap=residual_cap,
            min_samples=min_samples,
        ),
        _summarise_groups(
            train,
            ["horizon_bucket"],
            "horizon",
            residual_cap=residual_cap,
            min_samples=min_samples,
        ),
    ]
    bands = pd.concat([p for p in pieces if not p.empty], ignore_index=True, sort=False)
    for col in ("hod_30m",):
        if col in bands.columns:
            bands[col] = bands[col].astype("Int64")
    bands["quantile_low"] = 0.20
    bands["quantile_high"] = 0.80
    bands["train_end_utc"] = train_end.isoformat()
    return bands


def validate_bands(
    residual_df: pd.DataFrame,
    bands: pd.DataFrame,
    *,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    lower_col: str = "q20",
    upper_col: str = "q80",
) -> pd.DataFrame:
    valid = residual_df[
        (residual_df["interval_dt"] >= validation_start)
        & (residual_df["interval_dt"] < validation_end)
    ].copy()
    rows = []
    for run_time, grp in valid.groupby("run_time", sort=True):
        q50 = grp.set_index("interval_dt")["oof_debiased_rrp"].sort_index()
        lower, upper = apply_pd_residual_bands(
            q50,
            pd.Timestamp(run_time),
            bands,
            lower_col=lower_col,
            upper_col=upper_col,
        )
        actual = grp.set_index("interval_dt")["actual_rrp"].reindex(q50.index)
        horizon_bucket = grp.set_index("interval_dt")["horizon_bucket"].reindex(q50.index)
        covered = (actual >= lower) & (actual <= upper)
        tmp = pd.DataFrame(
            {
                "horizon_bucket": horizon_bucket.values,
                "covered": covered.values,
                "width": (upper - lower).values,
            }
        )
        rows.append(tmp)
    if not rows:
        return pd.DataFrame()
    joined = pd.concat(rows, ignore_index=True)
    summary = joined.groupby("horizon_bucket", sort=True).agg(
        n=("covered", "size"),
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
        median_width=("width", "median"),
    )
    overall = pd.DataFrame(
        {
            "n": [int(len(joined))],
            "coverage": [float(joined["covered"].mean())],
            "mean_width": [float(joined["width"].mean())],
            "median_width": [float(joined["width"].median())],
        },
        index=["overall"],
    )
    out = pd.concat([overall, summary])
    out.index.name = "bucket"
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debiased-path", type=Path, default=DEBIASED_PARQUET)
    parser.add_argument("--actuals-path", type=Path, default=ACTUALS_PARQUET)
    parser.add_argument("--output", type=Path, default=RESIDUAL_BANDS_PARQUET)
    parser.add_argument("--validation-output", type=Path, default=None)
    parser.add_argument("--train-end", default="2025-07-01T00:00:00Z")
    parser.add_argument("--validation-start", default="2025-07-01T00:00:00Z")
    parser.add_argument("--validation-end", default="2026-01-01T00:00:00Z")
    parser.add_argument("--residual-cap", type=float, default=RESIDUAL_BAND_CAP_DEFAULT)
    parser.add_argument("--min-samples", type=int, default=RESIDUAL_BAND_MIN_SAMPLES_DEFAULT)
    parser.add_argument("--min-daytype-samples", type=int, default=50)
    parser.add_argument(
        "--actuals-shift-min", type=int, default=0,
        help=(
            "Shift actuals.interval_dt by this many minutes before merging with the "
            "debiased OOF parquet. Default 0 preserves canonical join convention. "
            "Set to 30 when consuming an aligned OOF (output of "
            "`train_pd_debiaser.py --actuals-shift-min=30`) so the residual is "
            "measured against the correctly-aligned half-hour."
        ),
    )
    args = parser.parse_args()

    train_end = pd.Timestamp(args.train_end).tz_convert("UTC")
    validation_start = pd.Timestamp(args.validation_start).tz_convert("UTC")
    validation_end = pd.Timestamp(args.validation_end).tz_convert("UTC")

    residual_df = _load_residual_frame(args.debiased_path, args.actuals_path,
                                       actuals_shift_min=args.actuals_shift_min)
    bands = build_residual_bands(
        residual_df,
        train_end=train_end,
        residual_cap=args.residual_cap,
        min_samples=args.min_samples,
        min_daytype_samples=args.min_daytype_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bands.to_parquet(args.output, index=False)

    if args.validation_output is None:
        stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        args.validation_output = RESULTS_DIR / f"pd_residual_bands_validation_{stamp}.csv"
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation = validate_bands(
        residual_df,
        bands,
        validation_start=validation_start,
        validation_end=validation_end,
    )
    validation.to_csv(args.validation_output, index=False)

    print(f"Wrote {len(bands)} band rows to {args.output}")
    print(f"Wrote validation summary to {args.validation_output}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
