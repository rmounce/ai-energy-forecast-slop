"""Unit tests for the daemon-side (V_hot, T_hot) tracker (hwc_soc_tracker).

Design-faithful checks (docs/hwc_2state_soc_model.md "Production architecture"): the two watermark
resets are exact and dominate, the conservative draw prior only decrements, standing loss cools
T_hot, active heating advances the state, and everything clamps. No daemon / no I/O.
"""

import math

import pytest

from hwc_soc_model import SoCParams, probe_temp
from hwc_soc_tracker import TrackerState, advance, draw_kwh_per_s_to_dv, seed_state

P = SoCParams()
DESIRED = 60.0
CLIFF = (P.t_mains_c + DESIRED) / 2.0  # ~39 °C — the cold-slug cliff anchor


def _adv(state, dt_s, *, heating=False, probe=None, power=800.0, draw_kwh_per_s=0.0):
    return advance(state, dt_s, heating=heating, probe_c=probe, modelled_power_w=power,
                   draw_kwh_per_s=draw_kwh_per_s, p=P, desired_c=DESIRED, cliff_probe_c=CLIFF)


# ── seed ──────────────────────────────────────────────────────────────────────

def test_seed_is_conservative_and_floors_t_hot():
    s = seed_state(54.0, P)
    assert s.v_hot == 0.5
    assert s.t_hot == 54.0
    assert seed_state(10.0, P).t_hot >= P.t_mains_c  # floored above mains


@pytest.mark.xfail(strict=True, reason="seed-divergence (Option 2); see docs/hwc_2state_soc_model.md")
def test_seed_from_hot_probe_is_observation_consistent():
    # Live divergence on first enable (2026-06-27): with the probe reading ~57 °C the thermocline
    # is plainly above the 0.50 sensor, yet seed_state pins V_hot at the conservative 0.50 and
    # T_hot at the under-reading mid-probe. The model's own observation then reads
    # g(0.50, 57) = (T_mains + 57)/2 ≈ 37 °C — ~20 °C below the real probe — so every plan built
    # from that seed starts from a phantom-cold tank and diverges. A seed MUST be
    # observation-consistent: feeding it back through g must reproduce the probe it was seeded from.
    # Fix = Option 2 (seed T_hot at the delivery temp, V_hot0 = invert_g(probe, T_hot)).
    for probe in (53.0, 55.0, 57.0, 59.0):
        s = seed_state(probe, P)
        assert abs(probe_temp(s.v_hot, s.t_hot, P) - probe) <= 3.0, probe


# ── watermark resets dominate ──────────────────────────────────────────────────

def test_top_watermark_snaps_full_at_target():
    # Probe at/above target ⇒ tank full + hot, regardless of the carried estimate.
    s, why = _adv(TrackerState(0.3, 55.0), 60.0, heating=True, probe=60.0)
    assert s.v_hot == 1.0
    assert s.t_hot >= 60.0
    assert why == "watermark:full@target"


def test_cliff_watermark_snaps_to_sensor_height():
    # A deep draw drops the probe past the cliff ⇒ thermocline at the sensor ⇒ V_hot≈sensor_height.
    s, why = _adv(TrackerState(0.95, 58.0), 60.0, heating=False, probe=35.0)
    assert s.v_hot == P.sensor_height
    assert why == "watermark:cliff"


def test_cliff_does_not_raise_v_hot():
    # If already below the sensor, the cliff watermark must not bump V_hot up.
    s, _ = _adv(TrackerState(0.3, 55.0), 60.0, heating=False, probe=35.0)
    assert s.v_hot <= 0.3 + 1e-9


def test_mid_probe_does_not_trigger_either_watermark():
    # Probe between cliff and target ⇒ unobservable band ⇒ no snap, only dynamics.
    s, why = _adv(TrackerState(0.7, 55.0), 60.0, heating=False, probe=52.0)
    assert why == "coast"
    assert 0.0 <= s.v_hot <= 0.7


# ── dynamics between resets ────────────────────────────────────────────────────

def test_heating_advances_v_hot():
    s, why = _adv(TrackerState(0.5, 53.0), 120.0, heating=True, probe=53.0, power=800.0)
    assert s.v_hot > 0.5
    assert why == "heat"  # probe 53 < target, no watermark


def test_standing_loss_cools_t_hot_when_idle():
    s, _ = _adv(TrackerState(1.0, 60.0), 3600.0, heating=False, probe=None)
    assert s.t_hot < 60.0
    assert math.isclose(s.v_hot, 1.0)  # no draw, no loss on V_hot here


def test_conservative_draw_prior_only_decrements():
    # A draw rate during an idle interval reduces V_hot (pessimistic guard), T_hot unchanged-ish.
    rate = 0.5 / 3600.0  # 0.5 kWh/h of hot water drawn
    s, _ = _adv(TrackerState(0.9, 55.0), 3600.0, heating=False, probe=None, draw_kwh_per_s=rate)
    assert s.v_hot < 0.9
    # standing loss cools T_hot ~0.3 °C first, nudging the lift used in the conversion ~1%.
    expected_dv = draw_kwh_per_s_to_dv(rate, 55.0, P) * 3600.0
    assert math.isclose(0.9 - s.v_hot, expected_dv, rel_tol=2e-2)


def test_draw_prior_clamps_at_empty():
    rate = 50.0 / 3600.0  # absurd draw
    s, _ = _adv(TrackerState(0.2, 55.0), 3600.0, heating=False, probe=None, draw_kwh_per_s=rate)
    assert s.v_hot == 0.0


def test_full_cycle_sequence():
    # Draw → cliff snap → reheat advances V_hot → target snap to full. End-to-end sanity.
    s = seed_state(58.0, P)                                   # start mid
    s, _ = _adv(s, 60.0, heating=False, probe=34.0)           # deep draw
    assert s.v_hot == P.sensor_height
    for _ in range(40):                                       # reheat
        s, _ = _adv(s, 60.0, heating=True, probe=54.0, power=820.0)
    assert s.v_hot > P.sensor_height
    s, why = _adv(s, 60.0, heating=True, probe=60.0)          # reached target
    assert s.v_hot == 1.0 and why == "watermark:full@target"
