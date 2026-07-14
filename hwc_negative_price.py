#!/usr/bin/env python3
"""Negative-price override for the HWC.

When the live buy price is negative we are *paid* to import, so the HWC should dump as much
electrical power as it can — COP is irrelevant. This layer sits on top of the DP plan inside
``hwc_daemon`` (never as a separate HA automation: a second controller fighting for
``water_heater.aquatech`` is the one thing guaranteed to misbehave).

Full design + rationale: ``docs/hwc/surplus_negative_price.md``. In short:

- The unit is on a 10 A plug, so the heat pump (~700 W) and element (1800 W) are **mutually
  exclusive** and 1800 W is its hard maximum draw.
- **Compressor off** -> ``electric`` at 75 °C: nothing to interrupt, go straight to max draw.
- **Compressor running** -> ``performance`` at 75 °C: the compressor runs on uninterrupted to
  60 °C and the element then takes it 60->75 by itself. Interrupting a *running* compressor for
  the element buys no extra heat (1800 W element ~= 1750 W thermal from the heat pump) — only
  Δ ~1.1 kW of paid draw — while costing a restart.
- ...**unless the restart pays for itself**, i.e.
  ``|price| × Δ × gain_hours > transition_cost_aud``, in which case ``electric``, latched for
  the event.

``gain_hours = min(remaining_negative_window, time_to_60C)``: past 60 °C ``performance`` would
have escalated to the element anyway, so the switch buys nothing beyond that point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from dateutil import parser as date_parser

import hwc_executor

log = logging.getLogger(__name__)

# Mode names as accepted by water_heater.set_operation_mode on the Aquatech RAPID X6.
MODE_ELECTRIC = "electric"
MODE_PERFORMANCE = "performance"

DEFAULT_SETPOINT_C = 75.0
DEFAULT_ELEMENT_POWER_W = 1800.0
DEFAULT_HEAT_PUMP_POWER_W = 700.0
DEFAULT_CONFIRM_SECONDS = 45.0
# Element handover temperature: above this, `performance` is running the element anyway.
DEFAULT_ELEMENT_HANDOVER_C = 60.0


@dataclass(frozen=True)
class OverrideState:
    """Cross-tick state. Persisted by the daemon so a restart mid-event doesn't drop it."""

    latched_electric: bool = False
    # Wall-clock (epoch, UTC) of the first negative price read of the current event; used to
    # tell a *confirmed* negative price from the conservative estimate that precedes it.
    negative_since: float | None = None

    def to_dict(self) -> dict:
        return {"latched_electric": self.latched_electric, "negative_since": self.negative_since}

    @classmethod
    def from_dict(cls, raw: dict | None) -> "OverrideState":
        raw = raw or {}
        since = raw.get("negative_since")
        return cls(
            latched_electric=bool(raw.get("latched_electric", False)),
            negative_since=float(since) if since is not None else None,
        )


def config(cfg: dict) -> dict:
    return (cfg.get("hwc") or {}).get("negative_price") or {}


def enabled(cfg: dict) -> bool:
    return bool(config(cfg).get("enabled", False))


def price_entity(cfg: dict) -> str:
    return config(cfg).get("price_entity", "sensor.amber_effective_general_price")


def forecast_entity(cfg: dict) -> str:
    return config(cfg).get(
        "forecast_entity", "sensor.amber_billing_interval_forecasts_general_price"
    )


def _delta_kw(cfg: dict) -> float:
    """Extra draw bought by stopping the compressor and running the element instead."""
    ncfg = config(cfg)
    element_w = float(ncfg.get("element_power_w", DEFAULT_ELEMENT_POWER_W))
    heat_pump_w = float(ncfg.get("heat_pump_power_w", DEFAULT_HEAT_PUMP_POWER_W))
    return max(0.0, (element_w - heat_pump_w) / 1000.0)


def remaining_negative_window_hours(forecasts: list[dict], now: datetime, *, cfg: dict) -> float:
    """Hours of *contiguous* forecast-negative price from ``now`` (the leading negative run).

    Reads the Amber billing-interval ``Forecasts`` attribute (5-minute resolution). The run must
    start at the interval covering ``now`` — a negative patch an hour away is not a reason to
    stop the compressor now.

    Falls back to a single interval when the forecast is missing or unparseable: pessimistic, so
    the break-even is hard to clear and we won't interrupt the compressor on a blind guess.
    """
    ncfg = config(cfg)
    fallback_h = float(ncfg.get("fallback_window_minutes", 5.0)) / 60.0

    intervals: list[tuple[datetime, datetime, float]] = []
    for item in forecasts or []:
        try:
            start = date_parser.isoparse(str(item["start_time"]))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            duration_h = float(item.get("duration", 5)) / 60.0
            price = float(item["per_kwh"])
        except (KeyError, TypeError, ValueError):
            continue
        end = start + _timedelta_hours(duration_h)
        intervals.append((start, end, price))

    if not intervals:
        return fallback_h

    intervals.sort(key=lambda iv: iv[0])
    hours = 0.0
    started = False
    for start, end, price in intervals:
        if end <= now:
            continue
        if price >= 0:
            break  # the leading negative run ends here
        if not started and start > now:
            # The current interval isn't negative in the forecast (or isn't present); a run that
            # only begins later doesn't justify acting now.
            break
        started = True
        hours += (end - max(start, now)).total_seconds() / 3600.0

    if not started:
        return fallback_h
    max_h = float(ncfg.get("max_window_hours", 4.0))
    return min(hours, max_h)


def _timedelta_hours(hours: float):
    from datetime import timedelta

    return timedelta(hours=hours)


def time_to_element_handover_hours(tank_c: float, *, cfg: dict) -> float:
    """Hours for the heat pump to lift the tank to the 60 °C element-handover point.

    Uses the DP's *maximum* modelled heat rate, which makes this an under-estimate of the time
    and so an under-estimate of ``gain_hours`` — pessimistic in the right direction: it raises
    the bar for interrupting a running compressor.
    """
    ncfg = config(cfg)
    thermal = (cfg.get("hwc") or {}).get("thermal") or {}
    handover_c = float(ncfg.get("element_handover_c", DEFAULT_ELEMENT_HANDOVER_C))
    rate = float(
        thermal.get("heat_rate_max_c_per_hour", thermal.get("heat_rate_c_per_hour", 6.6))
    )
    if rate <= 0:
        return 0.0
    return max(0.0, (handover_c - tank_c) / rate)


def break_even_price_aud_per_kwh(*, transition_cost_aud: float, delta_kw: float, gain_hours: float):
    """Price at or below which stopping a running compressor for the element pays for itself.

    Returns a negative number, or ``None`` when no price could justify it (nothing to gain).
    """
    energy_kwh = delta_kw * gain_hours
    if energy_kwh <= 0:
        return None
    return -transition_cost_aud / energy_kwh


def decide(
    cfg: dict,
    *,
    planned: hwc_executor.Decision,
    price_aud_per_kwh: float | None,
    compressor_on: bool,
    tank_c: float | None,
    forecasts: list[dict],
    now: datetime,
    state: OverrideState,
) -> tuple[hwc_executor.Decision | None, OverrideState]:
    """Override ``planned`` while the buy price is negative.

    Returns ``(decision_or_None, new_state)``; ``None`` means "no override — use the DP plan".

    On exit (``price >= 0``) the caller must *actively re-assert* the plan, not merely stop
    overriding: ``performance``'s 60->75 element leg is ungated and would keep importing at
    1800 W to reach setpoint.
    """
    if not enabled(cfg) or price_aud_per_kwh is None or price_aud_per_kwh >= 0:
        return None, OverrideState()

    ncfg = config(cfg)
    setpoint_c = float(ncfg.get("setpoint_c", DEFAULT_SETPOINT_C))
    now_ts = now.timestamp()
    negative_since = state.negative_since if state.negative_since is not None else now_ts

    def _electric(reason: str) -> hwc_executor.Decision:
        return hwc_executor.Decision(
            action="heat",
            reason=reason,
            setpoint_c=setpoint_c,
            mode=MODE_ELECTRIC,
            compressor_on=planned.compressor_on,
            uses_compressor=False,
        )

    # Compressor already stopped: nothing to interrupt, go straight to maximum draw.
    if not compressor_on:
        return (
            _electric(f"negative price {price_aud_per_kwh:.3f} $/kWh; element (compressor idle)"),
            OverrideState(latched_electric=True, negative_since=negative_since),
        )

    # Latched for the event: never flip back to the compressor mid-event — that would pay the
    # very restart the latch exists to avoid *and* give up the dump.
    if state.latched_electric:
        return (
            _electric(f"negative price {price_aud_per_kwh:.3f} $/kWh; element (latched)"),
            OverrideState(latched_electric=True, negative_since=negative_since),
        )

    # The compressor is running. Interrupting it costs a restart, so that decision waits for the
    # *confirmed* price: the entity publishes a conservative estimate at the start of each 5-min
    # interval and the actual ~30 s later, with nothing to distinguish them, so we require the
    # price to have read negative across an update boundary.
    confirm_s = float(ncfg.get("confirm_seconds", DEFAULT_CONFIRM_SECONDS))
    confirmed = (now_ts - negative_since) >= confirm_s

    gain_hours = None
    threshold = None
    if confirmed and tank_c is not None:
        window_h = remaining_negative_window_hours(forecasts, now, cfg=cfg)
        to_handover_h = time_to_element_handover_hours(tank_c, cfg=cfg)
        gain_hours = min(window_h, to_handover_h)
        threshold = break_even_price_aud_per_kwh(
            transition_cost_aud=float((cfg.get("hwc") or {}).get("transition_cost_aud", 0.05)),
            delta_kw=_delta_kw(cfg),
            gain_hours=gain_hours,
        )

    if threshold is not None and price_aud_per_kwh < threshold:
        log.info(
            "HWC negative-price: price %.3f < break-even %.3f $/kWh (gain %.2f h); "
            "stopping compressor for the element",
            price_aud_per_kwh,
            threshold,
            gain_hours,
        )
        return (
            _electric(
                f"negative price {price_aud_per_kwh:.3f} $/kWh below break-even "
                f"{threshold:.3f}; element"
            ),
            OverrideState(latched_electric=True, negative_since=negative_since),
        )

    # Default while the compressor runs: keep it running to 60 °C, then `performance` hands over
    # to the element by itself. No restart, and the full dump on any event long enough to matter.
    return (
        hwc_executor.Decision(
            action="heat",
            reason=f"negative price {price_aud_per_kwh:.3f} $/kWh; performance (compressor running)",
            setpoint_c=setpoint_c,
            mode=MODE_PERFORMANCE,
            compressor_on=planned.compressor_on,
            uses_compressor=True,
        ),
        OverrideState(latched_electric=False, negative_since=negative_since),
    )
