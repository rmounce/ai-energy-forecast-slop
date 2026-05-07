"""PD-direct baseline forecast (Phase α, 2026-05-05 strategic pivot).

A no-ML strategic price forecast stitched from sources we already trust:

  - 0–60 min      : Tier 1 tactical LightGBM q50 (handled by caller; not in this module)
  - 60 min – ~30h : debiased PREDISPATCH q50 (look up OOF parquet by run_time)
  - ~30h – 7d     : PD7Day capped + hour-of-day seasonal mean for any remaining gaps

The OOF debiased parquet is keyed by (run_time, interval_dt) and contains the value the
debiaser would have published at run_time without label leakage — exactly what we want for
historical eval. Raw PD7Day values frequently sit at the soft cap ($980.89) as a
"spike-risk" indicator; we cap them for dispatch sanity. Hour-of-day mean from a trailing
28-day actuals window fills any remaining tail.

Used by `eval/rolling_mpc_eval.py` as a `pd_direct` source for Window A/B `netload_tariffed`
gates. Plan and decision rule live in `docs/roadmap.md` (top section).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEBIASED_PARQUET = REPO_ROOT / "data" / "parquet" / "debiased_pd_rrp_oof.parquet"
PD7DAY_PARQUET = REPO_ROOT / "data" / "parquet" / "aemo_pd7day_sa1.parquet"
ACTUALS_PARQUET = REPO_ROOT / "data" / "parquet" / "actuals_sa1.parquet"

# PD7Day reaches the soft cap ($980.89) as a categorical "spike-risk" indicator. Treat that
# as "above-baseload" but not as a literal price: cap to a value that lets the LP charge
# during cheap periods and discharge during PD7Day-flagged peaks without overcommitting.
PD7DAY_CAP_DEFAULT = 300.0

# 28-day trailing window for the hour-of-day seasonal fallback. Captures recent regime
# without smearing across season changes.
SEASONAL_WINDOW_DAYS_DEFAULT = 28


@dataclass
class PDDirectContext:
    """Pre-loaded lookup tables for fast inference across many eval anchors."""

    debiased_runs: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    debiased_run_times_sorted: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    debiased_run_times_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    pd7_runs: dict[pd.Timestamp, pd.Series] = field(default_factory=dict)
    pd7_run_times_sorted: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    pd7_run_times_ns: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # Hour-of-day (30-min slot, 0–47) -> mean RRP $/MWh for the seasonal fallback layer.
    seasonal_hod_table: pd.Series = field(default_factory=lambda: pd.Series(dtype=np.float64))

    pd7day_cap: float = PD7DAY_CAP_DEFAULT


def _index_runs(df: pd.DataFrame, value_col: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Group a (run_time, interval_dt, value) frame by run_time → Series indexed by interval_dt.

    Returns (runs_dict, sorted_run_times_array, sorted_run_times_ns_array).
    """
    if df.empty:
        return {}, np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.int64)
    df = df.copy()
    df["run_time"] = pd.to_datetime(df["run_time"], utc=True)
    df["interval_dt"] = pd.to_datetime(df["interval_dt"], utc=True)
    runs: dict[pd.Timestamp, pd.Series] = {}
    for rt, grp in df.groupby("run_time", sort=True):
        s = grp.set_index("interval_dt")[value_col].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        runs[rt] = s
    sorted_keys = sorted(runs.keys())
    sorted_rt = np.array(sorted_keys, dtype=object)
    sorted_ns = np.array([pd.Timestamp(rt).value for rt in sorted_keys], dtype=np.int64)
    return runs, sorted_rt, sorted_ns


def _build_seasonal_hod_table(
    actuals: pd.DataFrame,
    seasonal_end: pd.Timestamp | None,
    seasonal_window_days: int,
) -> pd.Series:
    """Return a Series indexed 0..47 (30-min hour-of-day slots) with mean RRP."""
    actuals = actuals.copy()
    actuals["time"] = pd.to_datetime(actuals["time"], utc=True)
    if seasonal_end is None:
        seasonal_end = actuals["time"].max()
    seasonal_end = pd.Timestamp(seasonal_end).tz_convert("UTC")
    seasonal_start = seasonal_end - pd.Timedelta(days=int(seasonal_window_days))
    mask = (actuals["time"] >= seasonal_start) & (actuals["time"] < seasonal_end)
    sl = actuals.loc[mask, ["time", "rrp"]].dropna()
    if sl.empty:
        return pd.Series([60.0] * 48, index=range(48), dtype=np.float64)
    sl["hod_30m"] = sl["time"].dt.hour * 2 + (sl["time"].dt.minute // 30)
    seasonal = sl.groupby("hod_30m")["rrp"].mean()
    seasonal = seasonal.reindex(range(48)).ffill().bfill()
    return seasonal.astype(np.float64)


def load_pd_direct_context(
    debiased_path: Path = DEBIASED_PARQUET,
    pd7_path: Path = PD7DAY_PARQUET,
    actuals_path: Path = ACTUALS_PARQUET,
    seasonal_end: pd.Timestamp | None = None,
    seasonal_window_days: int = SEASONAL_WINDOW_DAYS_DEFAULT,
    pd7day_cap: float = PD7DAY_CAP_DEFAULT,
) -> PDDirectContext:
    """Load and pre-index everything the curve builder needs.

    `seasonal_end` should be set to the eval anchor (start) to prevent look-ahead. If None,
    the most recent timestamp in the actuals parquet is used (fine for live inference).
    """
    deb = pd.read_parquet(debiased_path)
    debiased_runs, deb_rt_sorted, deb_rt_ns = _index_runs(
        deb[["run_time", "interval_dt", "oof_debiased_rrp"]],
        "oof_debiased_rrp",
    )

    pd7_runs: dict[pd.Timestamp, pd.Series] = {}
    pd7_rt_sorted = np.array([], dtype=object)
    pd7_rt_ns = np.array([], dtype=np.int64)
    if pd7_path.exists():
        pd7 = pd.read_parquet(pd7_path)
        if not pd7.empty:
            pd7_runs, pd7_rt_sorted, pd7_rt_ns = _index_runs(
                pd7[["run_time", "interval_dt", "rrp"]], "rrp"
            )

    actuals = pd.read_parquet(actuals_path)
    seasonal = _build_seasonal_hod_table(actuals, seasonal_end, seasonal_window_days)

    return PDDirectContext(
        debiased_runs=debiased_runs,
        debiased_run_times_sorted=deb_rt_sorted,
        debiased_run_times_ns=deb_rt_ns,
        pd7_runs=pd7_runs,
        pd7_run_times_sorted=pd7_rt_sorted,
        pd7_run_times_ns=pd7_rt_ns,
        seasonal_hod_table=seasonal,
        pd7day_cap=float(pd7day_cap),
    )


def _latest_run_at(run_times_ns: np.ndarray, run_times_sorted: np.ndarray, anchor_ns: int):
    if len(run_times_ns) == 0:
        return None
    pos = int(np.searchsorted(run_times_ns, anchor_ns, side="right"))
    if pos == 0:
        return None
    return pd.Timestamp(run_times_sorted[pos - 1])


def build_pd_direct_30m_curve(
    ctx: PDDirectContext,
    anchor_30m: pd.Timestamp,
    horizon_30m_steps: int,
) -> pd.Series:
    """Build a 30-min q50 strategic price curve from `anchor_30m` for `horizon_30m_steps`.

    Layered fallback:
      1. Debiased PREDISPATCH (run aligned to most-recent run_time ≤ anchor)
      2. PD7Day, capped at `ctx.pd7day_cap`, for steps still missing
      3. Hour-of-day seasonal mean for any remaining tail

    Returns a fully-populated Series indexed by 30-min UTC timestamps.
    """
    anchor_30m = pd.Timestamp(anchor_30m).tz_convert("UTC") if anchor_30m.tzinfo else anchor_30m
    idx = pd.date_range(start=anchor_30m, periods=horizon_30m_steps, freq="30min", tz="UTC")
    out = pd.Series(index=idx, dtype=np.float64)
    anchor_ns = int(idx[0].value)

    # Layer 1: debiased PREDISPATCH from the most recent run_time ≤ anchor.
    deb_rt = _latest_run_at(ctx.debiased_run_times_ns, ctx.debiased_run_times_sorted, anchor_ns)
    if deb_rt is not None:
        deb_series = ctx.debiased_runs[deb_rt]
        common = deb_series.index.intersection(idx)
        if len(common) > 0:
            out.loc[common] = deb_series.loc[common].values

    # Layer 2: PD7Day capped, fills any still-NaN slots.
    if out.isna().any():
        pd7_rt = _latest_run_at(ctx.pd7_run_times_ns, ctx.pd7_run_times_sorted, anchor_ns)
        if pd7_rt is not None:
            pd7_series = ctx.pd7_runs[pd7_rt]
            missing = out.index[out.isna()]
            common = pd7_series.index.intersection(missing)
            if len(common) > 0:
                capped = np.minimum(pd7_series.loc[common].values.astype(np.float64), ctx.pd7day_cap)
                out.loc[common] = capped

    # Layer 3: hour-of-day seasonal mean for any remaining gaps (PD7Day not available, e.g.
    # 2025-era eval slices, or slots beyond PD7Day horizon).
    if out.isna().any():
        hods = out.index.hour.values * 2 + (out.index.minute.values // 30)
        seasonal_vals = ctx.seasonal_hod_table.reindex(hods).to_numpy(dtype=np.float64)
        seasonal_series = pd.Series(seasonal_vals, index=out.index)
        out = out.where(~out.isna(), seasonal_series)

    # Final guard: any remaining NaN (should not happen after layer 3) gets a sensible
    # constant rather than propagating into the LP.
    if out.isna().any():
        out = out.fillna(60.0)

    return out


def expand_30m_to_5m(curve_30m: pd.Series, target_idx_5m: pd.DatetimeIndex) -> pd.Series:
    """Repeat each 30-min value across its six 5-min sub-slots, aligned to `target_idx_5m`.

    Pads with ffill/bfill if the target index extends slightly past the source range.
    """
    if len(curve_30m) == 0:
        return pd.Series(np.nan, index=target_idx_5m, dtype=np.float64)
    src_start = curve_30m.index.min()
    src_end = curve_30m.index.max() + pd.Timedelta(minutes=25)
    expanded_idx = pd.date_range(start=src_start, end=src_end, freq="5min", tz="UTC")
    expanded = pd.Series(index=expanded_idx, dtype=np.float64)
    for t30, val in curve_30m.items():
        block = pd.date_range(start=t30, periods=6, freq="5min", tz="UTC")
        expanded.loc[block] = float(val)
    return expanded.reindex(target_idx_5m).ffill().bfill()


def build_pd_direct_5m_curve(
    ctx: PDDirectContext,
    anchor_5m: pd.Timestamp,
    horizon_5m_steps: int,
) -> np.ndarray:
    """Convenience: build a 5-min curve covering `horizon_5m_steps` from `anchor_5m`.

    Internally builds a 30-min curve covering enough slots to span the requested horizon,
    then expands to 5-min and reindexes. Returns a plain ndarray (the caller is expected to
    apply `out[0] = current_actual` via the eval framework's `_finalize_curve`).
    """
    anchor_5m = pd.Timestamp(anchor_5m).tz_convert("UTC") if anchor_5m.tzinfo else anchor_5m
    idx_5m = pd.date_range(start=anchor_5m, periods=horizon_5m_steps, freq="5min", tz="UTC")
    anchor_30m = anchor_5m.floor("30min")
    horizon_30m = int(np.ceil(horizon_5m_steps / 6)) + 2  # +2 slots of safety margin
    curve_30m = build_pd_direct_30m_curve(ctx, anchor_30m, horizon_30m)
    expanded = expand_30m_to_5m(curve_30m, idx_5m)
    return expanded.to_numpy(dtype=np.float64)
