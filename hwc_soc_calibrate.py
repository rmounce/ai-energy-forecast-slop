#!/usr/bin/env python3
"""First-cut calibration/visualisation for the (V_hot, T_hot) stratified tank model.

Loads a window CSV from ``hwc_soc_extract.py`` and produces diagnostic plots:

  traces   probe / exhaust / power / compressor over time (any window) — shows the charge
           "flat-then-jump", the two-phase condensing rise, and draws.
  cop      phase-2 COP vs condensing (exhaust) temp, estimated from a clean reheat where the
           tank is destratified and the probe rate ≈ the bulk thermal rate.

Design context: docs/hwc_thermal_characterisation.md, docs/hwc_short_cycle_review_2026-06-26.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TANK_L = 222.0
CAP_KWH_PER_K = TANK_L * 0.997 * 4.186 / 3600.0  # ≈0.257 kWh/K


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df


def plot_traces(df: pd.DataFrame, out: Path, title: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(df.index, df["probe_ctrl"], label="probe (control)", lw=1.6)
    if "exhaust" in df:
        ax[0].plot(df.index, df["exhaust"], label="exhaust (condensing)", lw=1.0, alpha=0.8)
    if "coil" in df:
        ax[0].plot(df.index, df["coil"], label="coil (evap)", lw=0.8, alpha=0.6)
    ax[0].axhline(60, ls=":", c="r", lw=0.8)
    ax[0].axhline(53, ls=":", c="0.6", lw=0.8)
    ax[0].set_ylabel("°C"); ax[0].legend(loc="upper left", fontsize=8); ax[0].set_title(title)

    ax[1].plot(df.index, df["power_w"], c="tab:green", lw=1.0)
    ax[1].set_ylabel("compressor W")

    ax[2].fill_between(df.index, 0, df.get("compressor_on", 0), step="post", alpha=0.4, label="compressor")
    if "defrost_on" in df:
        ax[2].fill_between(df.index, 0, df["defrost_on"], step="post", alpha=0.4, color="tab:red", label="defrost")
    ax[2].set_ylabel("on/off"); ax[2].set_ylim(-0.1, 1.1); ax[2].legend(loc="upper left", fontsize=8)
    ax[2].set_xlabel("local time")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"wrote {out}")


def cop_curve(df: pd.DataFrame, out: Path, lo=53.0, hi=60.0, width=0.5) -> pd.DataFrame:
    """Robust phase-2 COP(T_hot): electrical energy to raise the probe through each band.

    In phase-2 the tank is destratified so probe ≈ bulk; raising the probe by ``width`` adds
    ``cap·width`` of thermal energy, so COP = cap·width / (∫P over the time the probe spends in
    that band). Binning by *probe* (monotone in phase-2) sidesteps the sparse-probe dProbe/dt
    noise. The probe can't give *phase-1* COP (it's flat while the hot zone grows) — that needs an
    exhaust/power thermal proxy, a separate step.
    """
    on = df.get("compressor_on", 0) > 0.5
    sub = df[on & df["exhaust"].notna() & (df["probe_ctrl"] >= lo)].copy()
    dt_h = sub.index.to_series().diff().dt.total_seconds() / 3600.0
    sub["elec_kwh"] = (sub["power_w"] / 1000.0) * dt_h
    bins = np.arange(lo, hi + width, width)
    sub["band"] = pd.cut(sub["probe_ctrl"], bins)
    g = sub.groupby("band", observed=True).agg(
        elec=("elec_kwh", "sum"), exhaust=("exhaust", "mean"), n=("elec_kwh", "size")
    )
    g = g[(g["elec"] > 0) & (g["n"] >= 2)]
    g["probe"] = [iv.mid for iv in g.index]
    g["cop"] = CAP_KWH_PER_K * width / g["elec"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(g["probe"], g["cop"], "o-")
    ax[0].set_xlabel("probe ≈ T_hot (°C)"); ax[0].set_ylabel("COP (phase-2, band-integrated)")
    ax[0].set_title("COP vs T_hot"); ax[0].grid(alpha=0.3)
    ax[1].plot(g["exhaust"], g["cop"], "o-", color="tab:orange")
    ax[1].set_xlabel("exhaust / condensing temp (°C)"); ax[1].set_title("COP vs condensing temp")
    ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"wrote {out}")
    print(g[["probe", "exhaust", "cop", "n"]].round(2).to_string(index=False))
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--mode", choices=("traces", "cop", "both"), default="traces")
    ap.add_argument("--out", type=Path, help="output PNG (traces); defaults next to csv")
    args = ap.parse_args()
    df = load(args.csv)
    stem = args.csv.with_suffix("")
    if args.mode in ("traces", "both"):
        plot_traces(df, args.out or Path(f"{stem}_traces.png"), args.csv.name)
    if args.mode in ("cop", "both"):
        cop_curve(df, Path(f"{stem}_cop.png"))


if __name__ == "__main__":
    main()
