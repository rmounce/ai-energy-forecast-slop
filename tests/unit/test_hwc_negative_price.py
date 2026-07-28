"""Negative-price override (docs/hwc/surplus_negative_price.md)."""

from datetime import datetime, timedelta, timezone

import hwc_executor as he
import hwc_negative_price as np


NOW = datetime(2026, 7, 14, 3, 0, tzinfo=timezone.utc)


def _cfg(**overrides):
    cfg = {
        "hwc": {
            "transition_cost_aud": 0.05,
            "thermal": {"heat_rate_max_c_per_hour": 7.6},
            "negative_price": {
                "enabled": True,
                "setpoint_c": 70,
                "element_power_w": 1800,
                "heat_pump_power_w": 700,
                "confirm_seconds": 45,
                "element_handover_c": 60,
                "fallback_window_minutes": 5,
                "max_window_hours": 4,
            },
        }
    }
    cfg["hwc"]["negative_price"].update(overrides)
    return cfg


def _forecasts(prices_per_5min, start=NOW):
    return [
        {
            "start_time": (start + timedelta(minutes=5 * i)).isoformat(),
            "duration": 5,
            "per_kwh": p,
        }
        for i, p in enumerate(prices_per_5min)
    ]


PLANNED = he.Decision(action="heat", reason="plan", setpoint_c=58.0)
PLANNED_OFF = he.Decision(action="off", reason="plan")


def _decide(price, compressor_on, tank_c=50.0, forecasts=None, state=None, now=NOW, cfg=None):
    return np.decide(
        cfg or _cfg(),
        planned=PLANNED,
        price_aud_per_kwh=price,
        compressor_on=compressor_on,
        tank_c=tank_c,
        forecasts=forecasts if forecasts is not None else [],
        now=now,
        state=state or np.OverrideState(),
    )


def test_no_override_when_price_non_negative_or_disabled():
    assert _decide(0.25, compressor_on=False)[0] is None
    assert _decide(0.0, compressor_on=False)[0] is None
    cfg = _cfg(enabled=False)
    assert _decide(-0.10, compressor_on=False, cfg=cfg)[0] is None


def test_compressor_idle_goes_straight_to_electric_at_70():
    decision, state = _decide(-0.10, compressor_on=False)
    assert decision.action == "heat"
    assert decision.mode == np.MODE_ELECTRIC
    assert decision.setpoint_c == 70
    assert decision.uses_compressor is False
    assert state.latched_electric


def test_compressor_idle_above_handover_keeps_electric_armed():
    """All element-capable modes appear to share the same re-trigger hysteresis. Keep the direct
    electric request armed above 60 C and let the controller start it at its threshold."""
    decision, state = _decide(-0.10, compressor_on=False, tank_c=61.0)

    assert decision.action == "heat"
    assert decision.mode == np.MODE_ELECTRIC
    assert decision.setpoint_c == 70
    assert decision.uses_compressor is False
    assert state.latched_electric


def test_running_compressor_gets_performance_not_electric():
    """Interrupting a running compressor buys no extra heat — only Δ1.1 kW of paid draw — and
    costs a restart, so the default is to let it run on and let `performance` hand over at 60."""
    decision, state = _decide(-0.10, compressor_on=True, forecasts=_forecasts([-0.10] * 6))
    assert decision.mode == np.MODE_PERFORMANCE
    assert decision.setpoint_c == 70
    assert decision.uses_compressor is True
    assert not state.latched_electric


def test_deep_negative_price_clears_break_even_and_stops_the_compressor():
    # 30 min of negative window, tank at 50 °C (10 °C below handover => ~1.3 h at 7.6 °C/h),
    # so gain_hours = 0.5 h => break-even = -0.05 / (1.1 × 0.5) = -0.0909 $/kWh.
    forecasts = _forecasts([-0.50] * 6 + [0.10])
    now = NOW + timedelta(seconds=60)  # past confirm_seconds
    seen = np.OverrideState(negative_since=NOW.timestamp())  # price already confirmed negative

    shallow, _ = _decide(-0.05, compressor_on=True, forecasts=forecasts, now=now, state=seen)
    assert shallow.mode == np.MODE_PERFORMANCE  # above break-even: not worth the restart

    deep, state = _decide(-0.50, compressor_on=True, forecasts=forecasts, now=now, state=seen)
    assert deep.mode == np.MODE_ELECTRIC
    assert deep.uses_compressor is False
    assert state.latched_electric


def test_break_even_waits_for_the_confirmed_price():
    """The interrupt is the only decision that costs a restart, so it must not act on the
    conservative estimate published at the start of each 5-minute interval."""
    forecasts = _forecasts([-0.50] * 6)

    # The very first negative read is never enough, however deep.
    first, first_state = _decide(-0.50, compressor_on=True, forecasts=forecasts, now=NOW)
    assert first.mode == np.MODE_PERFORMANCE
    assert first_state.negative_since == NOW.timestamp()  # the clock starts here

    state = np.OverrideState(negative_since=NOW.timestamp())

    early, _ = _decide(
        -0.50, compressor_on=True, forecasts=forecasts, state=state, now=NOW + timedelta(seconds=10)
    )
    assert early.mode == np.MODE_PERFORMANCE

    confirmed, _ = _decide(
        -0.50, compressor_on=True, forecasts=forecasts, state=state, now=NOW + timedelta(seconds=50)
    )
    assert confirmed.mode == np.MODE_ELECTRIC


def test_electric_is_latched_for_the_event():
    """Flipping back mid-event would pay the restart the latch exists to avoid *and* give up
    the dump."""
    latched = np.OverrideState(latched_electric=True, negative_since=NOW.timestamp())
    # Shallow price that would otherwise fail the break-even, compressor reported running.
    decision, state = _decide(-0.01, compressor_on=True, state=latched, now=NOW + timedelta(minutes=5))
    assert decision.mode == np.MODE_ELECTRIC
    assert state.latched_electric


def test_latched_element_remains_electric_after_tank_crosses_handover():
    latched = np.OverrideState(latched_electric=True, negative_since=NOW.timestamp())

    decision, state = _decide(
        -0.01,
        compressor_on=False,
        tank_c=61.0,
        state=latched,
        now=NOW + timedelta(minutes=5),
    )

    assert decision.mode == np.MODE_ELECTRIC
    assert decision.uses_compressor is False
    assert state.latched_electric


def test_latch_and_negative_since_clear_when_the_price_recovers():
    latched = np.OverrideState(latched_electric=True, negative_since=NOW.timestamp())
    decision, state = _decide(0.20, compressor_on=False, state=latched)
    assert decision is None  # caller re-asserts the DP plan
    assert not state.latched_electric
    assert state.negative_since is None


def test_gain_hours_is_capped_by_time_to_the_element_handover():
    """A nearly-hot tank reaches 60 °C in minutes, after which `performance` runs the element
    anyway — so a long negative window doesn't justify an interrupt."""
    forecasts = _forecasts([-0.15] * 24)  # 2 h of negative price
    now = NOW + timedelta(seconds=60)
    seen = np.OverrideState(negative_since=NOW.timestamp())

    # Tank at 59.5 °C => ~0.066 h to handover => break-even ≈ -0.69 $/kWh; -0.15 doesn't clear.
    hot, _ = _decide(-0.15, compressor_on=True, tank_c=59.5, forecasts=forecasts, now=now, state=seen)
    assert hot.mode == np.MODE_PERFORMANCE

    # Tank at 45 °C => ~1.97 h, capped by the 2 h window => break-even ≈ -0.023; -0.15 clears.
    cold, _ = _decide(-0.15, compressor_on=True, tank_c=45.0, forecasts=forecasts, now=now, state=seen)
    assert cold.mode == np.MODE_ELECTRIC


def test_missing_forecast_falls_back_to_one_interval():
    """Pessimistic: with no forward view, assume the shortest window so the break-even is hard
    to clear and we don't interrupt the compressor on a blind guess."""
    assert np.remaining_negative_window_hours([], NOW, cfg=_cfg()) == 5 / 60

    now = NOW + timedelta(seconds=60)
    # -0.30 $/kWh over a 5-min window: break-even = -0.05/(1.1 × 0.0833) = -0.545 => no switch.
    decision, _ = _decide(-0.30, compressor_on=True, forecasts=[], now=now)
    assert decision.mode == np.MODE_PERFORMANCE


def test_window_counts_only_the_leading_negative_run():
    cfg = _cfg()
    # Negative now for 15 min, then positive, then negative again: only the leading run counts.
    forecasts = _forecasts([-0.1, -0.1, -0.1, 0.2, -0.4, -0.4])
    assert np.remaining_negative_window_hours(forecasts, NOW, cfg=cfg) == 15 / 60

    # A run that only starts later is no reason to act now.
    later = _forecasts([0.2, 0.2, -0.4, -0.4])
    assert np.remaining_negative_window_hours(later, NOW, cfg=cfg) == 5 / 60

    # Partly-elapsed current interval counts only its remaining part.
    mid = NOW + timedelta(minutes=2)
    assert np.remaining_negative_window_hours(_forecasts([-0.1, 0.3]), mid, cfg=cfg) == 3 / 60


def test_break_even_price_matches_the_documented_table():
    # docs/hwc/surplus_negative_price.md: Δ 1.1 kW, transition cost 0.05.
    def be(hours):
        return np.break_even_price_aud_per_kwh(
            transition_cost_aud=0.05, delta_kw=1.1, gain_hours=hours
        )

    assert round(be(5 / 60), 2) == -0.55  # ~ -54 c/kWh
    assert round(be(15 / 60), 2) == -0.18
    assert round(be(30 / 60), 2) == -0.09
    assert round(be(60 / 60), 3) == -0.045
    assert be(0.0) is None  # nothing to gain => no price justifies the restart
