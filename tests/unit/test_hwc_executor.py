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
    )

    assert decision.action == "heat"
    assert decision.setpoint_c == 58


def test_decide_turns_off_immediately_after_block():
    # No post-block grace: the executor follows the plan and stops at the block boundary; the
    # compressor stops promptly, so there is nothing to let "finish".
    decision = he.decide(
        _points(),
        now=datetime(2026, 6, 1, 2, 40, tzinfo=timezone.utc),
        compressor_on=True,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
    )

    assert decision.action == "off"
    assert decision.reason == "outside planned block; stopping running compressor"


def test_decide_turns_off_outside_block_when_compressor_off():
    decision = he.decide(
        _points(),
        now=datetime(2026, 6, 1, 3, 0, tzinfo=timezone.utc),
        compressor_on=False,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
    )

    assert decision.action == "off"


def test_decide_turns_off_running_compressor_outside_planned_block():
    points = _points()
    last = points[-1]
    points.extend(
        he.PlanPoint(at=last.at + timedelta(minutes=30 * idx), power_w=0, temp_c=56)
        for idx in range(1, 5)
    )
    decision = he.decide(
        points,
        now=datetime(2026, 6, 1, 4, 10, tzinfo=timezone.utc),
        compressor_on=True,
        threshold_w=100,
        setpoint_min=55,
        setpoint_max=60,
    )

    assert decision.action == "off"
    assert decision.reason == "outside planned block; stopping running compressor"


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
    )

    assert decision.setpoint_c == 60
