#!/usr/bin/env python3
"""Fit simple HWC planner parameters from extracted compressor cycles.

This deliberately fits only low-dimensional parameters from
``data/hwc_cop_cycles.csv``. It does not change live config; use the output as a
review artifact before updating planner assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CSV = "data/hwc_cop_cycles.csv"
DEFAULT_TARGET_C = 60.0
DEFAULT_TARGET_TOLERANCE_C = 0.2
DEFAULT_TOP_UP_START_C = 53.0


def _round_to(value: float, step: float) -> float:
    return round(round(float(value) / step) * step, 3)


def load_cycles(path: str = DEFAULT_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"start", "dur_min", "tank_start", "tank_end", "hp_mean_w", "elec_kwh", "cop", "clean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required cycle columns: {sorted(missing)}")
    return df


def prepare_cycles(
    df: pd.DataFrame,
    *,
    target_c: float = DEFAULT_TARGET_C,
    target_tolerance_c: float = DEFAULT_TARGET_TOLERANCE_C,
    top_up_start_c: float = DEFAULT_TOP_UP_START_C,
) -> pd.DataFrame:
    out = df.copy()
    out["clean"] = out["clean"].astype(bool)
    for col in ("element_on", "defrost_on", "four_way_on"):
        if col not in out:
            out[col] = False
        out[col] = out[col].astype(bool)

    out["compressor_only"] = ~(out["element_on"] | out["defrost_on"] | out["four_way_on"])
    out["to_target"] = out["tank_end"] >= target_c - target_tolerance_c
    out["lift_c"] = out["tank_end"] - out["tank_start"]
    out["dur_h"] = out["dur_min"] / 60.0
    out["heat_rate_c_per_hour"] = out["lift_c"] / out["dur_h"]
    out["elec_kwh_per_c"] = out["elec_kwh"] / out["lift_c"]
    out["cycle_class"] = "partial"
    out.loc[out["to_target"] & (out["tank_start"] >= top_up_start_c), "cycle_class"] = "top_up"
    out.loc[out["to_target"] & (out["tank_start"] < top_up_start_c), "cycle_class"] = "full_reheat"
    return out


def _summary_row(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "class": label,
            "n": 0,
            "median_heat_rate_c_per_hour": None,
            "median_hp_mean_w": None,
            "mean_cop": None,
            "median_elec_kwh_per_c": None,
            "median_probe_lag_min": None,
        }
    return {
        "class": label,
        "n": int(len(df)),
        "median_heat_rate_c_per_hour": round(float(df["heat_rate_c_per_hour"].median()), 2),
        "median_hp_mean_w": round(float(df["hp_mean_w"].median())),
        "mean_cop": round(float(df["cop"].mean()), 2),
        "median_elec_kwh_per_c": round(float(df["elec_kwh_per_c"].median()), 3),
        "median_probe_lag_min": (
            round(float(df["probe_lag_min"].median()), 1)
            if "probe_lag_min" in df and df["probe_lag_min"].notna().any()
            else None
        ),
    }


def fit_parameters(
    df: pd.DataFrame,
    *,
    target_c: float = DEFAULT_TARGET_C,
    top_up_start_c: float = DEFAULT_TOP_UP_START_C,
) -> dict:
    prepared = prepare_cycles(df, target_c=target_c, top_up_start_c=top_up_start_c)
    usable = prepared[prepared["clean"] & prepared["compressor_only"] & prepared["to_target"]].copy()
    full = usable[usable["cycle_class"] == "full_reheat"]
    top_up = usable[usable["cycle_class"] == "top_up"]

    heat_source = full if not full.empty else usable
    if usable.empty:
        raise ValueError("no clean compressor-only cycles reaching target")

    suggestions = {
        "nominal_power_w": int(_round_to(usable["hp_mean_w"].median(), 10)),
        "heat_rate_c_per_hour": _round_to(heat_source["heat_rate_c_per_hour"].median(), 0.1),
        "mean_clean_cop": round(float(usable["cop"].mean()), 2),
    }
    if not top_up.empty:
        suggestions["top_up_start_temp_c"] = top_up_start_c
        suggestions["top_up_heat_rate_c_per_hour"] = _round_to(
            top_up["heat_rate_c_per_hour"].median(), 0.1
        )
        suggestions["top_up_cop"] = round(float(top_up["cop"].mean()), 2)

    return {
        "target_c": target_c,
        "top_up_start_c": top_up_start_c,
        "usable_cycle_count": int(len(usable)),
        "suggestions": suggestions,
        "summaries": [
            _summary_row(usable, "all_usable"),
            _summary_row(full, "full_reheat"),
            _summary_row(top_up, "top_up"),
        ],
    }


def _format_value(value) -> str:
    return "" if value is None else str(value)


def write_markdown(fit: dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suggestions = fit["suggestions"]
    lines = [
        "# HWC Thermal Model Fit",
        "",
        "Derived from clean, compressor-only cycles in `data/hwc_cop_cycles.csv`.",
        "",
        f"- Target temperature: `{fit['target_c']}` C",
        f"- Top-up class starts at: `tank_start >= {fit['top_up_start_c']}` C",
        f"- Usable cycles: `{fit['usable_cycle_count']}`",
        "",
        "## Suggested Existing Planner Parameters",
        "",
        "| parameter | suggested | note |",
        "| --- | --- | --- |",
        f"| `thermal.nominal_power_w` | `{suggestions['nominal_power_w']}` | median clean HP proxy watts, rounded to 10 W |",
        f"| `thermal.heat_rate_c_per_hour` | `{suggestions['heat_rate_c_per_hour']}` | median full-reheat probe lift rate where available |",
        "",
    ]
    if "top_up_heat_rate_c_per_hour" in suggestions:
        lines += [
            "## Optional Top-Up Model Parameters",
            "",
            "| parameter | suggested | note |",
            "| --- | --- | --- |",
            f"| `thermal.top_up_start_temp_c` | `{suggestions['top_up_start_temp_c']}` | use the top-up rate at or above this modelled tank temp |",
            f"| `thermal.top_up_heat_rate_c_per_hour` | `{suggestions['top_up_heat_rate_c_per_hour']}` | median near-target top-up probe lift rate |",
            "",
            "## Diagnostic Split",
            "",
            f"- Top-up median heat rate: `{suggestions['top_up_heat_rate_c_per_hour']}` C/h",
            f"- Top-up mean COP: `{suggestions['top_up_cop']}`",
            f"- Mean usable-cycle COP: `{suggestions['mean_clean_cop']}`",
            "",
        ]

    cols = [
        "class",
        "n",
        "median_heat_rate_c_per_hour",
        "median_hp_mean_w",
        "mean_cop",
        "median_elec_kwh_per_c",
        "median_probe_lag_min",
    ]
    lines += [
        "## Cycle Class Summary",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in fit["summaries"]:
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    lines.append("")
    lines.append(
        "Do not treat this as a stratified tank model yet; it is a parameter fit for the current single-node block planner."
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--target-c", type=float, default=DEFAULT_TARGET_C)
    ap.add_argument("--top-up-start-c", type=float, default=DEFAULT_TOP_UP_START_C)
    ap.add_argument("--summary-md", default="docs/hwc_model_fit.md")
    args = ap.parse_args()

    fit = fit_parameters(
        load_cycles(args.csv),
        target_c=args.target_c,
        top_up_start_c=args.top_up_start_c,
    )
    write_markdown(fit, args.summary_md)
    print(f"wrote {args.summary_md}")
    for key, value in fit["suggestions"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
