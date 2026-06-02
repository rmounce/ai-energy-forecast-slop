from datetime import datetime, timedelta, timezone

import hwc_executor as he


def _points():
    base = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    powers = [0, 0, 800, 800, 800, 0, 0]
    temps = [50, 50, 50, 52, 55, 58, 57]
    return [
        he.PlanPoint(at=base + timedelta(minutes=30 * idx), power_w=power, temp_c=temp)
        for idx, (power, temp) in enumerate(zip(powers, temps, strict=True))
    ]


def test_decide_heats_inside_block_to_block_end_setpoint():
    decision = he.decide(
        _points(),
        now=datetime(2026, 6, 1, 1, 10, tzinfo=timezone.utc),
        compressor_on=False,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
        post_block_grace=timedelta(minutes=90),
    )

    assert decision.action == "heat"
    assert decision.setpoint_c == 58


def test_decide_keeps_enabled_after_block_until_compressor_stops():
    decision = he.decide(
        _points(),
        now=datetime(2026, 6, 1, 2, 40, tzinfo=timezone.utc),
        compressor_on=True,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
        post_block_grace=timedelta(minutes=90),
    )

    assert decision.action == "heat"
    assert decision.reason == "compressor still running after planned block"
    assert decision.setpoint_c == 58


def test_decide_turns_off_outside_block_when_compressor_off():
    decision = he.decide(
        _points(),
        now=datetime(2026, 6, 1, 3, 0, tzinfo=timezone.utc),
        compressor_on=False,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
        post_block_grace=timedelta(minutes=90),
    )

    assert decision.action == "off"


def test_decide_caps_setpoint():
    points = _points()
    points[-2] = he.PlanPoint(points[-2].at, points[-2].power_w, 62)
    decision = he.decide(
        points,
        now=datetime(2026, 6, 1, 1, 10, tzinfo=timezone.utc),
        compressor_on=False,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
        post_block_grace=timedelta(minutes=90),
    )

    assert decision.setpoint_c == 60


def test_decide_current_suppresses_heat_when_tank_near_setpoint(monkeypatch):
    cfg = {
        "hwc": {
            "tank_temp_entity": "sensor.tank",
            "actuation": {
                "compressor_entity": "binary_sensor.compressor",
                "power_on_threshold_w": 100,
                "setpoint_min_c": 55,
                "setpoint_max_c": 60,
                "post_block_grace_minutes": 90,
                "min_heat_start_delta_c": 2.0,
            },
        },
    }

    monkeypatch.setattr(he, "load_plan", lambda _cfg: _points())

    def fake_state(_cfg, entity_id):
        if entity_id == "binary_sensor.compressor":
            return {"state": "off"}
        if entity_id == "sensor.tank":
            return {"state": "59.0"}
        raise AssertionError(entity_id)

    monkeypatch.setattr(he, "_entity_state", fake_state)
    monkeypatch.setattr(
        he,
        "decide",
        lambda *_args, **_kwargs: he.Decision(
            action="heat",
            reason="inside planned block",
            setpoint_c=60.0,
            block_start=datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc),
            block_end=datetime(2026, 6, 1, 1, 30, tzinfo=timezone.utc),
        ),
    )

    decision = he.decide_current(cfg)

    assert decision.action == "off"
    assert decision.reason.startswith("planned heat suppressed")
