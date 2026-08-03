import hwc_executor
import hwc_surplus as hs


def _cfg():
    return {
        "hwc": {
            "surplus": {
                "enabled": True,
                "start_curtailment_w": 2200,
                "start_seconds": 120,
                "exit_import_or_discharge_w": 400,
                "exit_seconds": 180,
                "max_entry_temp_c": 60,
                "setpoint_c": 70,
                "completion_temp_c": 69.8,
            }
        }
    }


PLAN = hwc_executor.Decision(action="off", reason="plan")


def _decide(now, state=hs.OverrideState(), **overrides):
    inputs = {
        "tank_c": 59.0,
        "curtailment_w": 2300.0,
        "grid_import_w": 0.0,
        "battery_power_w": 0.0,
        "element_on": False,
        "compressor_on": False,
    }
    inputs.update(overrides)
    return hs.decide(_cfg(), planned=PLAN, now_ts=now, state=state, **inputs)


def test_requires_sustained_surplus_then_starts_electric_to_70():
    decision, state = _decide(1000)
    assert decision is None
    assert state.entry_since == 1000

    decision, state = _decide(1119, state)
    assert decision is None
    decision, state = _decide(1120, state)
    assert decision.mode == "electric"
    assert decision.setpoint_c == 70
    assert decision.uses_compressor is False
    assert state.active


def test_entry_timer_resets_and_entry_is_blocked_above_60_or_on_adverse_flow():
    _, state = _decide(1000)
    assert _decide(1060, state, curtailment_w=2199)[1] == hs.OverrideState()
    assert _decide(1000, tank_c=60.1)[1] == hs.OverrideState()
    assert _decide(1000, grid_import_w=401)[1] == hs.OverrideState()
    assert _decide(1000, battery_power_w=-401)[1] == hs.OverrideState()
    assert _decide(1000, compressor_on=True)[1] == hs.OverrideState()


def test_active_event_ignores_disappearing_curtailment_and_brief_adverse_flow():
    active = hs.OverrideState(active=True)
    decision, state = _decide(1000, active, curtailment_w=0, grid_import_w=600)
    assert decision.mode == "electric"
    assert state.adverse_since == 1000
    decision, state = _decide(1179, state, curtailment_w=0, grid_import_w=600)
    assert decision is not None
    decision, state = _decide(1180, state, curtailment_w=0, grid_import_w=0)
    assert decision is not None
    assert state.adverse_since is None


def test_active_event_exits_after_sustained_grid_import_or_battery_discharge():
    for field, value in (("grid_import_w", 401), ("battery_power_w", -401)):
        decision, state = _decide(1000, hs.OverrideState(active=True), **{field: value})
        assert decision is not None
        decision, state = _decide(1180, state, **{field: value})
        assert decision is None
        assert state == hs.OverrideState()


def test_completion_and_higher_priority_override_clear_event():
    active = hs.OverrideState(active=True)
    assert _decide(1000, active, tank_c=69.8) == (None, hs.OverrideState())
    decision, state = hs.decide(
        _cfg(),
        planned=PLAN,
        now_ts=1000,
        tank_c=59,
        curtailment_w=3000,
        grid_import_w=0,
        battery_power_w=0,
        element_on=False,
        compressor_on=False,
        state=active,
        permit=False,
    )
    assert decision is None
    assert state == hs.OverrideState()


def test_missing_power_telemetry_blocks_entry_and_is_adverse_when_active():
    assert _decide(1000, grid_import_w=None)[1] == hs.OverrideState()
    decision, state = _decide(1000, hs.OverrideState(active=True), grid_import_w=None)
    assert decision is not None
    assert state.adverse_since == 1000
