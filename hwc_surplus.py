"""Reactive otherwise-curtailed PV override for the HWC resistive element.

The current EMHASS curtailment value is an entry signal only. Once the 1800 W element
starts it consumes the surplus that caused entry, so physical grid/battery flows retain
or release the event without creating a self-cancelling loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import hwc_executor


@dataclass(frozen=True)
class OverrideState:
    active: bool = False
    entry_since: float | None = None
    adverse_since: float | None = None

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "entry_since": self.entry_since,
            "adverse_since": self.adverse_since,
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "OverrideState":
        raw = raw or {}
        return cls(
            active=bool(raw.get("active", False)),
            entry_since=_optional_float(raw.get("entry_since")),
            adverse_since=_optional_float(raw.get("adverse_since")),
        )


def _optional_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def config(cfg: dict) -> dict:
    return (cfg.get("hwc") or {}).get("surplus") or {}


def enabled(cfg: dict) -> bool:
    return bool(config(cfg).get("enabled", False))


def watched_entities(cfg: dict) -> set[str]:
    scfg = config(cfg)
    return {
        entity
        for entity in (
            scfg.get("curtailment_entity", "sensor.mpc_p_pv_curtailment"),
            scfg.get("grid_import_entity", "sensor.sigen_plant_grid_import_power"),
            scfg.get("battery_power_entity", "sensor.sigen_plant_battery_power"),
        )
        if entity
    }


def decide(
    cfg: dict,
    *,
    planned: hwc_executor.Decision,
    now_ts: float,
    tank_c: float | None,
    curtailment_w: float | None,
    grid_import_w: float | None,
    battery_power_w: float | None,
    element_on: bool | None,
    compressor_on: bool,
    state: OverrideState,
    permit: bool = True,
) -> tuple[hwc_executor.Decision | None, OverrideState]:
    """Return an element override, or ``None`` to restore ``planned``.

    Sigenergy battery power is positive when charging and negative when discharging.
    Missing temperature/grid/battery telemetry is adverse while active and ineligible for entry.
    ``permit=False`` clears the event when a higher-priority override is active.
    """
    del planned  # The caller restores it when this function returns no override.
    scfg = config(cfg)
    if not enabled(cfg) or not permit:
        return None, OverrideState()

    start_w = float(scfg.get("start_curtailment_w", 2200))
    start_s = float(scfg.get("start_seconds", 120))
    exit_w = float(scfg.get("exit_import_or_discharge_w", 400))
    exit_s = float(scfg.get("exit_seconds", 180))
    max_entry_c = float(scfg.get("max_entry_temp_c", 60))
    target_c = float(scfg.get("setpoint_c", 70))
    completion_c = float(scfg.get("completion_temp_c", target_c - 0.2))

    if state.active:
        if tank_c is not None and tank_c >= completion_c:
            return None, OverrideState()
        adverse = (
            tank_c is None
            or grid_import_w is None
            or battery_power_w is None
            or grid_import_w > exit_w
            or battery_power_w < -exit_w
        )
        adverse_since = state.adverse_since
        if adverse:
            adverse_since = adverse_since if adverse_since is not None else now_ts
            if now_ts - adverse_since >= exit_s:
                return None, OverrideState()
        else:
            adverse_since = None
        return _element_decision("otherwise-curtailed PV event latched", target_c), OverrideState(
            active=True, adverse_since=adverse_since
        )

    eligible = (
        tank_c is not None
        and tank_c <= max_entry_c
        and curtailment_w is not None
        and curtailment_w >= start_w
        and grid_import_w is not None
        and battery_power_w is not None
        and grid_import_w <= exit_w
        and battery_power_w >= -exit_w
        and element_on is False
        and not compressor_on
    )
    if not eligible:
        return None, OverrideState()

    entry_since = state.entry_since if state.entry_since is not None else now_ts
    if now_ts - entry_since < start_s:
        return None, OverrideState(entry_since=entry_since)
    return (
        _element_decision(
            f"otherwise-curtailed PV sustained at {curtailment_w:.0f} W", target_c
        ),
        OverrideState(active=True),
    )


def _element_decision(reason: str, target_c: float) -> hwc_executor.Decision:
    return hwc_executor.Decision(
        action="heat",
        reason=reason,
        setpoint_c=target_c,
        mode="electric",
        uses_compressor=False,
    )
