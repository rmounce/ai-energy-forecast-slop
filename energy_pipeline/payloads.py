"""Reproduce the current HA DH/MPC payload policy from a frozen input snapshot.

No I/O, publication or helper writes. Compatibility with the deployed templates is
deliberate, including rounding, list truncation and clipping after energy scaling.
Admission/freshness validation belongs before these calculations in the coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


def timestamp(value: str | datetime) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("input timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def boundary(now: datetime, minutes: int) -> datetime:
    now = timestamp(now)
    return now.replace(minute=now.minute // minutes * minutes, second=0, microsecond=0)


def number(value: Any, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any) -> int:
    return int(number(value))


def blend(low: float, middle: float, high: float, weight: float) -> float:
    return middle + (high - middle) * weight if weight > 0 else middle - (middle - low) * abs(weight)


class Inputs:
    """Read-only view of the caller-owned, already frozen HA snapshot."""

    def __init__(self, states: Mapping[str, Mapping], now: datetime, timezone_name: str = "Australia/Adelaide"):
        self.states = states
        self.now = timestamp(now)
        self.local_tz = ZoneInfo(timezone_name)

    def state(self, entity: str) -> Any:
        return self.states.get(entity, {}).get("state", "unknown")

    def value(self, entity: str, default: float = 0) -> float:
        return number(self.state(entity), default)

    def attr(self, entity: str, key: str, default=None):
        return self.states.get(entity, {}).get("attributes", {}).get(key, default)

    def hwc(self) -> list:
        saved = self.attr("sensor.emhass_dh_hwc_power_plan_snapshot", "deferrables_schedule_json") or []
        if isinstance(saved, str):
            try:
                saved = json.loads(saved)
            except (TypeError, ValueError):
                saved = []
        return saved or self.attr("sensor.hwc_power_plan", "deferrables_schedule") or []

    def power(self, entity: str, key: str, start: datetime) -> list[int]:
        return [integer(row[key]) for row in self.attr(entity, "forecasts", []) or []
                if timestamp(row["date"]) >= start]


@dataclass(frozen=True)
class SocResult:
    soc_init_pct: float
    soc_final_pct: float
    deviation_pct: float
    reground_block: str | None = None
    should_reground: bool = False


def soc_points(schedule: Sequence[Mapping], anchor: float) -> list[tuple[datetime, float]]:
    if not schedule:
        return []
    return [(timestamp(schedule[0]["date"]), anchor)] + [
        (timestamp(row["date"]) + timedelta(minutes=30), float(row["dh_soc_batt_forecast"]))
        for row in schedule
    ]


def interpolate(points: Sequence[tuple[datetime, float]], at: datetime, default: float) -> float:
    """Preserve template order; clamp to available endpoints outside the plan."""
    left = right = None
    for point in points:
        if point[0] <= at:
            left = point
        elif right is None:
            right = point
    if left is not None and right is not None:
        span = (right[0] - left[0]).total_seconds()
        return left[1] if span <= 0 else left[1] * (1 - (at-left[0]).total_seconds()/span) + right[1] * ((at-left[0]).total_seconds()/span)
    return left[1] if left is not None else right[1] if right is not None else default


def dh_soc(inputs: Inputs) -> SocResult:
    now = inputs.now
    actual = inputs.value("sensor.sigen_plant_battery_state_of_charge_derived")
    block = boundary(now, 30).strftime("dh-%Y%m%dT%H%MZ")
    reground = inputs.state("input_text.dh_last_reground_block") != block
    last = inputs.value("input_number.dh_last_soc_init")
    anchor = actual if reground or last <= 0 else last
    schedule = inputs.attr("sensor.dh_soc_batt_forecast", "battery_scheduled_soc") or []
    deviation = 0.0
    if not reground and schedule:
        start = timestamp(schedule[0]["date"])
        end = timestamp(schedule[-1]["date"]) + timedelta(minutes=30)
        if start <= now <= end:
            # At the exact final endpoint the DH template has no right segment and uses zero.
            if now < end:
                deviation = round(actual - interpolate(soc_points(schedule, anchor), now, actual), 6)
    floor = inputs.value("input_number.battery_soc_min_target")
    clamp = lambda x: sorted([floor, x, 100])[1]
    return SocResult(round(clamp(anchor + deviation), 4),
                     round(clamp(anchor + inputs.value("input_number.emhass_target_soc_offset") + deviation), 4),
                     deviation, block, reground)


def mpc_soc(inputs: Inputs) -> SocResult:
    actual = inputs.value("sensor.sigen_plant_battery_state_of_charge_derived")
    last = inputs.value("input_number.dh_last_soc_init")
    pts = soc_points(inputs.attr("sensor.dh_soc_batt_forecast", "battery_scheduled_soc") or [], last if last > 0 else actual)
    start = boundary(inputs.now, 5)
    at_boundary = interpolate(pts, start, actual)
    at_now = interpolate(pts, inputs.now, actual)
    at_future = interpolate(pts, start + timedelta(hours=14), actual)
    deviation = round(actual-at_now, 6)
    positive = round(max(actual-at_now, 0), 6)
    clamp = lambda x: sorted([0, x, 100])[1]
    return SocResult(100 if actual >= 100 else round(clamp(at_boundary+deviation), 4),
                     round(clamp(at_future+positive), 4), deviation)


def target_soc_offset(inputs: Inputs) -> float:
    tail = [float(row["dh_soc_batt_forecast"]) for row in
            inputs.attr("sensor.dh_soc_batt_forecast", "battery_scheduled_soc") or []
            if timestamp(row["date"]) > inputs.now + timedelta(hours=47, minutes=30)]
    current = inputs.value("input_number.emhass_target_soc_offset")
    return round(current + (97.5-max(tail)) if tail else current, 4)


def hwc_averages(schedule: Sequence[Mapping], start: datetime, repeat_tail: bool = False) -> list[int]:
    result = []
    # Only present samples count in the average, matching HA even for sparse schedules.
    rows = [(timestamp(row["date"]), number(row.get("hwc_power_plan"))) for row in schedule]
    for i in range(144):
        begin = start + timedelta(minutes=30*i)
        values = [value for at, value in rows if begin <= at < begin + timedelta(minutes=30)]
        result.append(int(round(sum(values)/len(values))) if values else result[i-48] if repeat_tail and i >= 96 else 0)
    return result


def native_hwc(schedule: Sequence[Mapping], start: datetime) -> list[int]:
    rows = [(timestamp(row["date"]), integer(row.get("hwc_power_plan"))) for row in schedule]
    result = []
    for i in range(168):
        begin = start + timedelta(minutes=5*i)
        values = [value for at, value in rows if begin <= at < begin+timedelta(minutes=5)]
        result.append(values[-1] if values else 0)
    return result


def smooth_power(values: Sequence[int]) -> list[int]:
    result = []
    for i, value in enumerate(values):
        previous = values[i-1] if i else value
        following = values[i+1] if i+1 < len(values) else value
        start, end = (previous+value)/2, (value+following)/2
        block = [start + (j+0.5)/6*(end-start) for j in range(6)]
        error = value - sum(block)/6
        result.extend(max(0, int(round(x+error))) for x in block)
    return result


def flat_power(values: Sequence[int], periods_passed: int) -> list[int]:
    return [x for i, value in enumerate(values) for x in [value]*(6-periods_passed if i == 0 else 6)]


def scale_power(values: Sequence[float], target: float, fixed: Sequence[int] | None = None) -> list[float]:
    """Keep first two live slots; allocate error proportionally with rounding carry.

    Clipping and a zero base residual can prevent exact energy conservation; that
    is current HA behaviour, not an invariant this extraction silently changes.
    """
    extra = list(fixed[2:]) if fixed is not None else []
    tail = values[2:]
    total = sum(tail)
    error = target - sum(values) - sum(extra)
    carry = 0.0
    result = list(values[:2])
    for i, value in enumerate(tail):
        changed = value
        if total > 0:
            adjustment = error*(value/total) + carry
            rounded = int(round(adjustment))
            changed += rounded
            carry = adjustment-rounded
        result.append(max(0, int(changed)+(extra[i] if i < len(extra) else 0)))
    return result


def export_allowance(inputs: Inputs, at: str) -> float:
    return 0.01 if inputs.value("input_number.sapn_free_exports") > 0 and 10 <= timestamp(at).astimezone(inputs.local_tz).hour < 16 else 0


def common_payload(inputs: Inputs, soc: SocResult) -> dict:
    return {"entity_save": True,
            "soc_init": round(soc.soc_init_pct/100, 4), "soc_final": round(soc.soc_final_pct/100, 4),
            "battery_nominal_energy_capacity": int(1000*(inputs.value("sensor.sigen_plant_rated_energy_capacity")*(inputs.value("sensor.sigen_plant_battery_state_of_health")/100))),
            "weight_battery_discharge": inputs.value("input_number.emhass_weight_battery_discharge", 0.02),
            "weight_battery_charge": 0}


def build_dh_payload(inputs: Inputs) -> dict:
    result = common_payload(inputs, dh_soc(inputs))
    pv = []
    for day in ("today", "tomorrow", "day_3", "day_4"):
        for row in inputs.attr(f"sensor.solcast_pv_forecast_forecast_{day}", "detailedForecast") or []:
            if timestamp(row["period_start"]) > inputs.now-timedelta(minutes=30):
                pv.append(max(0, int(1000*blend(number(row.get("pv_estimate10")), number(row.get("pv_estimate")), number(row.get("pv_estimate90")), inputs.value("input_number.emhass_weight_pv_forecast")))))
    base = [round(float(row["power_load"]), 0) for row in inputs.attr("sensor.ai_load_forecast_high", "forecasts") or []]
    hwc = hwc_averages(inputs.hwc(), boundary(inputs.now, 30), repeat_tail=True)
    result["pv_power_forecast"] = [int(max(0, power-140)) for power in pv[:len(base)]]
    result["load_power_forecast"] = [int(load+max(0, 140-power)+(hwc[i] if i < len(hwc) else 0)) for i, (power, load) in enumerate(zip(pv, base))]
    median = inputs.attr("sensor.ai_price_forecast", "forecasts") or []
    low = inputs.attr("sensor.ai_price_forecast_low", "forecasts") or []
    high = inputs.attr("sensor.ai_price_forecast_high", "forecasts") or []
    result["load_cost_forecast"] = [round(blend(low[i]["general_price"], row["general_price"], high[i]["general_price"], inputs.value("input_number.emhass_weight_buy_forecast")), 4) for i, row in enumerate(median)]
    result["prod_price_forecast"] = [round(blend(low[i]["feed_in_price"], row["feed_in_price"], high[i]["feed_in_price"], inputs.value("input_number.emhass_weight_sell_forecast"))+export_allowance(inputs, row["timestamp"]), 4) for i, row in enumerate(median)]
    result.update(publish_prefix="dh_", delta_forecast_daily=3, optimization_time_step=30,
                  prediction_horizon=144, battery_minimum_state_of_charge=inputs.value("input_number.battery_soc_min_target")/100)
    return result


def build_mpc_payload(inputs: Inputs) -> dict:
    result = common_payload(inputs, mpc_soc(inputs))
    actual = inputs.value("sensor.sigen_plant_battery_state_of_charge_derived")
    floor = sorted([0, actual-inputs.value("input_number.battery_soc_min_buffer"), inputs.value("input_number.battery_soc_min_target")])[1]
    # HA filter precedence rounds the denominator 100, not the quotient.
    result.update(publish_prefix="mpc_", delta_forecast_daily=1, optimization_time_step=5,
                  prediction_horizon=168, battery_minimum_state_of_charge=floor/100)
    for channel, field, live in (("general", "load_cost_forecast", "sensor.amber_5min_current_general_price"),
                                 ("feed_in", "prod_price_forecast", "sensor.amber_adjusted_confirmed_feed_in_price")):
        values = [inputs.value(live)]
        sign = -1 if channel == "feed_in" else 1
        weight = inputs.value("input_number.emhass_weight_sell_forecast" if sign == -1 else "input_number.emhass_weight_buy_forecast")
        for row in inputs.attr(f"sensor.amber_5min_forecasts_extended_{channel}_price", "Forecasts") or []:
            at = timestamp(row["start_time"])
            if at <= inputs.now:
                continue
            value = blend(sign*row["advanced_price_low"], sign*row["advanced_price_predicted"], sign*row["advanced_price_high"], weight)
            if at-inputs.now < timedelta(minutes=5):
                value = min(value, -row["per_kwh"]) if sign == -1 else max(value, row["per_kwh"])
            values.append(round(value+(export_allowance(inputs, row["start_time"]) if sign == -1 else 0), 4))
        result[field] = values[:168]
    start = boundary(inputs.now, 30)
    passed = (inputs.now.minute % 30)//5
    schedule = inputs.hwc()
    dh_load = inputs.power("sensor.dh_p_load_forecast", "dh_p_load_forecast", start)
    dh_pv = inputs.power("sensor.dh_p_pv_forecast", "dh_p_pv_forecast", start)
    hwc = hwc_averages(schedule, start)
    base = [max(0, load-hwc[i]) for i, load in enumerate(dh_load[:len(hwc)])]
    base_future = smooth_power(base)[passed:][:168]
    pv_future = smooth_power(dh_pv)[passed:][:168]
    fixed_loss = max(0, min(integer(inputs.state("sensor.sigen_inverter_conversion_loss")), 140))
    gross_pv = integer(inputs.state("sensor.sigen_power_pv_gross"))
    pv_now = max(0, gross_pv-fixed_loss)
    load_now = integer(inputs.state("sensor.sigen_plant_consumed_power"))+max(0, fixed_loss-gross_pv)
    if inputs.state("sensor.emhass_current_pv_input_mode") in ("transition", "pv_limit", "export_limit"):
        forecast = blend(number(inputs.attr("sensor.solcast_pv_forecast_power_now", "estimate10")),
                         number(inputs.attr("sensor.solcast_pv_forecast_power_now", "estimate")),
                         number(inputs.attr("sensor.solcast_pv_forecast_power_now", "estimate90")),
                         inputs.value("input_number.emhass_weight_pv_forecast"))
        pv_now = max(pv_now, max(0, forecast-fixed_loss))
    result["load_power_forecast"] = scale_power([load_now]*2+base_future[2:], sum(flat_power(dh_load, passed)[:168]), native_hwc(schedule, boundary(inputs.now, 5)))
    result["pv_power_forecast"] = scale_power([pv_now]*2+pv_future[2:], sum(flat_power(dh_pv, passed)[:168]))
    return result
