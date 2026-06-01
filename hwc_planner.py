#!/usr/bin/env python3
"""Heat-pump hot water (HWC) scheduling planner — v1 (modelling only).

Optimises a heat-pump hot water unit (Aquatech RAPID X6) as an EMHASS
``thermal_battery`` deferrable load, *separately* from the home battery, against the
import-price forecast and a wet-bulb outdoor-temperature forecast. It POSTs a
``naive-mpc-optim`` request to the existing EMHASS instance with the battery and PV
disabled, then asks EMHASS to publish the resulting plan (predicted tank temperature +
planned power) back to Home Assistant.

This is the *modelling* phase: it produces and publishes a plan only — it does NOT
actuate the unit. See ``docs/hwc_emhass.md`` for the full design, the calibration
anchors, and the v1/v2/v3 roadmap.

Pure helpers (``stull_wet_bulb``, ``interpolate_to_grid``, ``build_draw_off_profile``,
``build_payload``) are unit-tested in ``tests/unit/test_hwc_planner.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytz
import requests

from config_utils import load_config


# ── Pure helpers (unit-tested) ──────────────────────────────────────────────


def stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """Wet-bulb temperature (°C) from dry-bulb temp (°C) and relative humidity (%).

    Stull (2011) empirical approximation — valid for roughly RH 5–99 % and
    T −20…50 °C, which comfortably covers Adelaide conditions. The heat pump's COP is
    governed by evaporator (wet-bulb) conditions, so this is what we feed EMHASS as the
    outdoor-temperature forecast. See docs/hwc_emhass.md ("Wet-bulb vs dry-bulb").
    """
    rh = max(1.0, min(100.0, float(rh_pct)))
    t = float(temp_c)
    return (
        t * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )


def interpolate_to_grid(
    src_epoch: list[float], src_values: list[float], grid_epoch: list[float]
) -> list[float]:
    """Linearly interpolate ``src_values`` (sampled at ``src_epoch`` seconds) onto
    ``grid_epoch``. Ends are held flat (np.interp constant extrapolation), so a grid
    extending slightly past the weather horizon reuses the last forecast value."""
    if not src_epoch:
        raise ValueError("interpolate_to_grid: empty source series")
    order = np.argsort(src_epoch)
    xs = np.asarray(src_epoch, dtype=float)[order]
    ys = np.asarray(src_values, dtype=float)[order]
    return np.interp(np.asarray(grid_epoch, dtype=float), xs, ys).tolist()


def build_draw_off_profile(
    grid_times_utc: list[datetime],
    tz_name: str,
    window_start: str,
    window_end: str,
    total_kwh: float,
) -> list[float]:
    """Build a per-timestep hot-water draw-off profile (kWh) aligned to *clock time*.

    EMHASS tiles ``draw_off_demand`` from the start of the horizon (index 0 = "now"),
    NOT by time of day, so we must place the draw ourselves at the timesteps whose local
    clock time falls in [window_start, window_end). ``total_kwh`` is distributed evenly
    across the in-window slots, so each day in the horizon receives ``total_kwh``.
    """
    tz = pytz.timezone(tz_name)
    sh, sm = (int(x) for x in window_start.split(":"))
    eh, em = (int(x) for x in window_end.split(":"))
    start_min, end_min = sh * 60 + sm, eh * 60 + em
    slots_per_window = max(1, (end_min - start_min) // _step_minutes(grid_times_utc))
    per_slot = float(total_kwh) / slots_per_window

    profile = []
    for t in grid_times_utc:
        local = t.astimezone(tz)
        minute_of_day = local.hour * 60 + local.minute
        in_window = start_min <= minute_of_day < end_min
        profile.append(round(per_slot, 5) if in_window else 0.0)
    return profile


def _step_minutes(grid_times_utc: list[datetime]) -> int:
    """Infer the grid step in minutes (defaults to 30 if not inferable)."""
    if len(grid_times_utc) >= 2:
        return max(1, int((grid_times_utc[1] - grid_times_utc[0]).total_seconds() // 60))
    return 30


def build_payload(
    *,
    grid_times_utc: list[datetime],
    load_cost: list[float],
    wet_bulb: list[float],
    draw_off: list[float],
    start_temperature: float,
    cfg: dict,
) -> dict:
    """Assemble the EMHASS ``naive-mpc-optim`` payload for the HWC thermal load.

    Battery and PV are disabled (runtime-overridable via associations.csv), so the
    optimisation is HWC-only against the import-price forecast.
    """
    n = len(grid_times_utc)
    th = cfg["thermal"]
    return {
        "prediction_horizon": n,
        "optimization_time_step": cfg.get("optimization_time_step", 30),
        # HWC-only: disable battery + PV so this is fast and decoupled.
        "set_use_battery": False,
        "set_use_pv": False,
        "number_of_deferrable_loads": 1,
        "nominal_power_of_deferrable_loads": [th["nominal_power_w"]],
        "treat_deferrable_load_as_semi_cont": [False],
        "set_deferrable_load_single_constant": [False],
        "set_deferrable_startup_penalty": [0],
        # Import-price objective; export/PV/base-load irrelevant here.
        "load_cost_forecast": [round(float(c), 5) for c in load_cost],
        "prod_price_forecast": [0.0] * n,
        "load_power_forecast": [0] * n,
        # Wet-bulb drives the Carnot COP (the only place outdoor temp is used in tank mode).
        "outdoor_temperature_forecast": [round(float(w), 3) for w in wet_bulb],
        "def_load_config": [
            {
                "thermal_battery": {
                    "volume": th["volume_l"] / 1000.0,  # litres → m³
                    "density": th.get("density", 997),
                    "heat_capacity": th.get("heat_capacity", 4.184),
                    "supply_temperature": th["supply_temperature"],
                    "carnot_efficiency": th["carnot_efficiency"],
                    "thermal_loss": th["thermal_loss_kw"],
                    "start_temperature": round(float(start_temperature), 2),
                    "min_temperatures": [th["min_temp"]] * n,
                    "max_temperatures": [th["max_temp"]] * n,
                    "desired_temperatures": [th["desired_temp"]] * n,
                    "penalty_factor": th.get("penalty_factor", 15),
                    "thermal_inertia_time_constant": th.get(
                        "thermal_inertia_time_constant", 0.5
                    ),
                    "draw_off_demand": [round(float(d), 5) for d in draw_off],
                }
            }
        ],
        "custom_predicted_temperature_id": [
            {
                "entity_id": cfg["predicted_temp_entity"],
                "unit_of_measurement": "°C",
                "friendly_name": "HWC Predicted Tank Temp",
            }
        ],
        "custom_deferrable_forecast_id": [
            {
                "entity_id": cfg["power_plan_entity"],
                "unit_of_measurement": "W",
                "friendly_name": "HWC Planned Power",
            }
        ],
        "entity_save": True,
        "publish_prefix": cfg.get("publish_prefix", "hwc_"),
    }


# ── Home Assistant I/O ──────────────────────────────────────────────────────


def _ha_call(cfg: dict, method: str, endpoint: str, payload: dict | None = None):
    ha = cfg["home_assistant"]
    url = f"{ha['url']}/api/{endpoint}"
    headers = {"Authorization": f"Bearer {ha['token']}", "Content-Type": "application/json"}
    resp = (
        requests.post(url, headers=headers, json=payload, timeout=30)
        if method == "POST"
        else requests.get(url, headers=headers, timeout=30)
    )
    resp.raise_for_status()
    return resp.json()


def get_tank_temperature(cfg: dict) -> float:
    state = _ha_call(cfg, "GET", f"states/{cfg['hwc']['tank_temp_entity']}")
    return float(state["state"])


def get_import_price_grid(cfg: dict, horizon_steps: int):
    """Return (grid_times_utc, load_cost) from the published DH import-price forecast,
    starting at the current 30-min interval."""
    entity = cfg["hwc"]["import_price_entity"]
    state = _ha_call(cfg, "GET", f"states/{entity}")
    forecast = state.get("attributes", {}).get("forecast", []) or []
    if not forecast:
        raise RuntimeError(f"{entity} has no 'forecast' attribute")

    now = datetime.now(timezone.utc)
    step = cfg["hwc"].get("optimization_time_step", 30)
    now_floor = now.replace(
        minute=(now.minute // step) * step, second=0, microsecond=0
    )
    rows = []
    for item in forecast:
        dt = pd.to_datetime(item["datetime"], utc=True).to_pydatetime()
        if dt >= now_floor:
            rows.append((dt, float(item["native_value"])))
    rows.sort(key=lambda r: r[0])
    rows = rows[:horizon_steps]
    if not rows:
        raise RuntimeError(f"{entity}: no future forecast points")
    return [r[0] for r in rows], [r[1] for r in rows]


def get_weather_series(cfg: dict):
    """Return (epoch_seconds, temp_c, rh_pct) from the BOM hourly weather forecast."""
    ha = cfg["home_assistant"]
    resp = _ha_call(
        cfg,
        "POST",
        "services/weather/get_forecasts?return_response=true",
        {"type": "hourly", "entity_id": ha["weather_entity"]},
    )
    forecast = (
        resp.get("service_response", {}).get(ha["weather_entity"], {}).get("forecast", [])
    )
    if not forecast:
        raise RuntimeError("weather.get_forecasts returned no forecast")

    tz = pytz.timezone(cfg["timezone"])
    epochs, temps, rhs = [], [], []
    for f in forecast:
        dt = pd.to_datetime(f["datetime"])
        if dt.tzinfo is None:
            dt = tz.localize(dt.to_pydatetime(), is_dst=None)
        epochs.append(dt.timestamp())
        temps.append(float(f["temperature"]))
        rhs.append(float(f.get("humidity", 60.0)))
    return epochs, temps, rhs


def run(cfg: dict, horizon_steps: int, dry_run: bool) -> dict:
    grid_times, load_cost = get_import_price_grid(cfg, horizon_steps)
    grid_epoch = [t.timestamp() for t in grid_times]

    w_epoch, w_temp, w_rh = get_weather_series(cfg)
    temp_grid = interpolate_to_grid(w_epoch, w_temp, grid_epoch)
    rh_grid = interpolate_to_grid(w_epoch, w_rh, grid_epoch)
    wet_bulb = [stull_wet_bulb(t, rh) for t, rh in zip(temp_grid, rh_grid)]

    draw_off = build_draw_off_profile(
        grid_times,
        cfg["timezone"],
        cfg["hwc"]["draw_off"]["window_start"],
        cfg["hwc"]["draw_off"]["window_end"],
        cfg["hwc"]["draw_off"]["total_kwh"],
    )

    start_temp = get_tank_temperature(cfg)

    payload = build_payload(
        grid_times_utc=grid_times,
        load_cost=load_cost,
        wet_bulb=wet_bulb,
        draw_off=draw_off,
        start_temperature=start_temp,
        cfg=cfg["hwc"],
    )

    logging.info(
        "HWC plan: horizon=%d steps, start_temp=%.1f°C, wet-bulb %.1f→%.1f°C, "
        "import $%.3f→$%.3f/kWh, draw-off total=%.2f kWh",
        len(grid_times),
        start_temp,
        min(wet_bulb),
        max(wet_bulb),
        min(load_cost),
        max(load_cost),
        sum(draw_off),
    )

    if dry_run:
        logging.info("Dry run — not POSTing to EMHASS. Payload:\n%s", json.dumps(payload, indent=2))
        return payload

    base = cfg["hwc"]["emhass_base_url"].rstrip("/")
    r = requests.post(f"{base}/action/naive-mpc-optim", json=payload, timeout=300)
    r.raise_for_status()
    logging.info("naive-mpc-optim OK (%s)", r.status_code)

    # Ask EMHASS to publish the plan (predicted temp + planned power) to HA.
    publish_payload = {
        "publish_prefix": cfg["hwc"].get("publish_prefix", "hwc_"),
        "custom_predicted_temperature_id": payload["custom_predicted_temperature_id"],
        "custom_deferrable_forecast_id": payload["custom_deferrable_forecast_id"],
        "def_load_config": [{"thermal_battery": {}}],
    }
    rp = requests.post(f"{base}/action/publish-data", json=publish_payload, timeout=60)
    rp.raise_for_status()
    logging.info("publish-data OK (%s)", rp.status_code)
    return payload


def main():
    parser = argparse.ArgumentParser(description="HWC EMHASS planner (v1, modelling only)")
    parser.add_argument("--dry-run", action="store_true", help="Build + log the payload, don't POST")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon (timesteps)")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # load_config deep-merges the untracked config.secrets.json (HA token, EMHASS URL)
    # over config.json — same practice as forecast.py.
    cfg = load_config(args.config)
    horizon = args.horizon or cfg["hwc"].get("horizon_steps", 72)
    try:
        run(cfg, horizon, args.dry_run)
    except Exception:
        logging.exception("HWC planner failed")
        raise


if __name__ == "__main__":
    main()
