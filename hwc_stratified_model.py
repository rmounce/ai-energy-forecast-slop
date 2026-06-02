#!/usr/bin/env python3
"""Small stratified tank model for HWC calibration experiments.

This is intentionally separate from the live planner. It models the tank as a
hot layer above a cold layer, with the control probe seeing the thermocline only
after enough of the tank has been charged. The goal is to compare modelled probe
temperature against observed compressor cycles before promoting anything into
the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StratifiedTankParams:
    volume_l: float = 225.0
    density_kg_per_m3: float = 997.0
    heat_capacity_kj_per_kg_c: float = 4.184
    standing_loss_ua_kw_per_c: float = 0.0025
    hot_target_c: float = 60.0
    probe_height_fraction: float = 0.55
    thermocline_width_fraction: float = 0.18


@dataclass(frozen=True)
class StratifiedTankState:
    cold_temp_c: float
    hot_temp_c: float
    hot_fraction: float


def capacity_kwh_per_c(params: StratifiedTankParams) -> float:
    mass_kg = params.volume_l * params.density_kg_per_m3 / 1000.0
    return mass_kg * params.heat_capacity_kj_per_kg_c / 3600.0


def stored_energy_kwh(state: StratifiedTankState, params: StratifiedTankParams, reference_c: float = 0.0) -> float:
    cap = capacity_kwh_per_c(params)
    hot = max(0.0, min(1.0, state.hot_fraction))
    return cap * (hot * (state.hot_temp_c - reference_c) + (1.0 - hot) * (state.cold_temp_c - reference_c))


def mean_temp_c(state: StratifiedTankState) -> float:
    hot = max(0.0, min(1.0, state.hot_fraction))
    return hot * state.hot_temp_c + (1.0 - hot) * state.cold_temp_c


def probe_temp_c(state: StratifiedTankState, params: StratifiedTankParams) -> float:
    """Return the modelled control-probe temperature.

    ``probe_height_fraction`` is the hot-layer fraction where the thermocline is
    centred on the probe. ``thermocline_width_fraction`` smooths the transition
    from cold to hot so the model is differentiable enough for fitting.
    """
    hot = max(0.0, min(1.0, state.hot_fraction))
    width = max(1e-6, params.thermocline_width_fraction)
    lower = params.probe_height_fraction - width / 2.0
    progress = (hot - lower) / width
    progress = max(0.0, min(1.0, progress))
    return state.cold_temp_c + progress * (state.hot_temp_c - state.cold_temp_c)


def apply_idle_loss(
    state: StratifiedTankState,
    params: StratifiedTankParams,
    *,
    ambient_c: float,
    step_h: float,
) -> StratifiedTankState:
    cap = capacity_kwh_per_c(params)
    avg = mean_temp_c(state)
    loss_kwh = max(0.0, avg - ambient_c) * params.standing_loss_ua_kw_per_c * step_h
    delta_c = loss_kwh / cap
    return StratifiedTankState(
        cold_temp_c=state.cold_temp_c - delta_c,
        hot_temp_c=max(state.cold_temp_c - delta_c, state.hot_temp_c - delta_c),
        hot_fraction=state.hot_fraction,
    )


def apply_draw_off(
    state: StratifiedTankState,
    params: StratifiedTankParams,
    *,
    draw_kwh: float,
) -> StratifiedTankState:
    """Remove hot-layer useful energy first, approximating top draw + cold refill."""
    if draw_kwh <= 0:
        return state
    cap = capacity_kwh_per_c(params)
    lift = max(0.1, state.hot_temp_c - state.cold_temp_c)
    hot_energy_kwh = cap * state.hot_fraction * lift
    if draw_kwh <= hot_energy_kwh:
        return StratifiedTankState(
            cold_temp_c=state.cold_temp_c,
            hot_temp_c=state.hot_temp_c,
            hot_fraction=max(0.0, state.hot_fraction - draw_kwh / (cap * lift)),
        )

    remaining = draw_kwh - hot_energy_kwh
    return StratifiedTankState(
        cold_temp_c=state.cold_temp_c - remaining / cap,
        hot_temp_c=state.hot_temp_c,
        hot_fraction=0.0,
    )


def apply_heat(
    state: StratifiedTankState,
    params: StratifiedTankParams,
    *,
    heat_kwh: float,
) -> StratifiedTankState:
    """Add compressor heat by growing a hot layer above the cold layer."""
    if heat_kwh <= 0:
        return state
    cap = capacity_kwh_per_c(params)
    target = max(params.hot_target_c, state.hot_temp_c, state.cold_temp_c + 0.1)
    lift = max(0.1, target - state.cold_temp_c)
    new_hot = min(1.0, state.hot_fraction + heat_kwh / (cap * lift))
    leftover = max(0.0, heat_kwh - (new_hot - state.hot_fraction) * cap * lift)
    if leftover > 0:
        # Once the whole tank is hot, extra heat raises the bulk temperature.
        target += leftover / cap
    return StratifiedTankState(
        cold_temp_c=state.cold_temp_c,
        hot_temp_c=target,
        hot_fraction=new_hot,
    )


def simulate_probe_temperatures(
    *,
    schedule_heat_kwh: list[float],
    draw_off_kwh: list[float],
    ambient_c: list[float],
    initial_state: StratifiedTankState,
    params: StratifiedTankParams,
    step_h: float,
) -> tuple[list[float], list[StratifiedTankState]]:
    state = initial_state
    probes: list[float] = []
    states: list[StratifiedTankState] = []
    for heat_kwh, draw_kwh, amb in zip(schedule_heat_kwh, draw_off_kwh, ambient_c, strict=True):
        probes.append(round(probe_temp_c(state, params), 2))
        states.append(state)
        state = apply_idle_loss(state, params, ambient_c=amb, step_h=step_h)
        state = apply_draw_off(state, params, draw_kwh=draw_kwh)
        state = apply_heat(state, params, heat_kwh=heat_kwh)
    return probes, states
