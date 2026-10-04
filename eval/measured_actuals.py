"""Independent measured economic targets; bounded sample-and-hold, no gap filling."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Source:
    column: str
    entity: str
    unit: str | None
    role: str


SOURCES = (
    Source('pv_dc_w', 'sensor.sigen_power_pv_gross', 'W', 'delivered_dc_pv_strings_1_2'),
    Source('inverter_ac_w', 'sensor.sigen_inverter_active_power', 'W', 'positive_ac_output'),
    Source('battery_charge_w', 'sensor.sigen_inverter_battery_power', 'W', 'positive_dc_charge'),
    Source('plant_battery_charge_w', 'sensor.sigen_plant_battery_power', 'W', 'positive_charge_candidate'),
    Source('grid_import_w', 'sensor.sigen_plant_grid_active_power', 'W', 'positive_ac_import'),
    Source('load_site_w', 'sensor.sigen_plant_consumed_power', 'W', 'site_load_reported'),
    Source('load_ac_derived_w', 'sensor.power_load_without_losses', 'W', 'grid_plus_inverter_clipped'),
    Source('load_base_w', 'sensor.power_consumed_without_deferrable_loads', 'W', 'base_load_derived'),
    Source('flexible_load_w', 'sensor.deferrable_load_power', 'W', 'flexible_load_derived'),
    Source('conversion_loss_w', 'sensor.sigen_inverter_conversion_loss', 'W', 'clipped_balance_residual'),
    Source('soc_pct', 'sensor.sigen_plant_battery_state_of_charge_derived', '%', 'derived_soc'),
    Source('pv_proxy_w', 'sensor.solcast_pv_forecast_power_now', 'W', 'forecast_proxy_comparison_only'),
    Source('planned_battery_discharge_w', 'sensor.mpc_p_batt_forecast', 'W', 'planned_dc_discharge'),
    Source('pv_limit_kw', 'number.sigen_plant_pv_max_power_limit', 'kW', 'applied_limit'),
    Source('export_limit_kw', 'number.sigen_plant_grid_export_limitation', 'kW', 'applied_limit'),
    Source('charge_limit_kw', 'number.sigen_plant_ess_max_charging_limit', 'kW', 'applied_limit'),
    Source('curtailment_fraction', 'sensor.emhass_current_pv_input_mode', None, 'curtailment_context'),
)


def window(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError('window endpoints require timezone')
    start, end = start.tz_convert('UTC'), end.tz_convert('UTC')
    if start != start.floor('5min') or end != end.floor('5min'):
        raise ValueError('window endpoints require five-minute boundaries')
    if not pd.Timedelta(0) < end-start <= pd.Timedelta(days=7):
        raise ValueError('window must be positive and at most seven days')
    return start, end


def clean_samples(samples):
    """Reject conflicting duplicate timestamps rather than average sensor series."""
    if not isinstance(samples.index, pd.DatetimeIndex) or samples.index.tz is None:
        raise ValueError('samples require timezone-aware timestamps')
    samples = samples.copy()
    samples.index = samples.index.tz_convert('UTC')
    if samples.index.hasnans:
        raise ValueError('invalid sample timestamp')
    if samples.groupby(level=0).nunique(dropna=False).gt(1).any():
        raise ValueError('conflicting sample values at the same timestamp')
    return samples[~samples.index.duplicated()].sort_index()


def integrate(samples, start, end, *, max_hold_seconds=120):
    """Integrate held finite numeric observations into five-minute UTC bins.

    Each sample expires at the next observation or max_hold_seconds, whichever
    comes first. NaN observations terminate prior support. No future sample is
    used before its timestamp. Complete means all 300 seconds observed.
    """
    start, end = window(start, end)
    if not np.isfinite(max_hold_seconds) or not 0 < max_hold_seconds <= 86400:
        raise ValueError('hold bound must be positive and at most one day')
    samples = clean_samples(samples)
    numeric = pd.to_numeric(samples, errors='coerce').to_numpy(dtype=float)
    times = samples.index.as_unit('ns').asi8
    start_ns, end_ns = start.value, end.value
    step_ns = 300*10**9
    count = int((end_ns-start_ns)//step_ns)
    coverage = np.zeros(count)
    integral = np.zeros(count)
    if len(times):
        begins = np.maximum(times, start_ns)
        finishes = np.minimum(np.minimum(np.r_[times[1:], end_ns],
                                         times+int(max_hold_seconds*10**9)), end_ns)
        valid = (finishes > begins) & np.isfinite(numeric)
        begins, finishes, values = begins[valid], finishes[valid], numeric[valid]
        first = (begins-start_ns)//step_ns
        last = (finishes-1-start_ns)//step_ns
        same = first == last
        def add(bins, seconds, vals):
            np.add.at(coverage, bins, seconds)
            np.add.at(integral, bins, seconds*vals)
        add(first[same], (finishes[same]-begins[same])/1e9, values[same])
        cross = ~same
        a, b, vals = first[cross], last[cross], values[cross]
        add(a, (start_ns+(a+1)*step_ns-begins[cross])/1e9, vals)
        add(b, (finishes[cross]-(start_ns+b*step_ns))/1e9, vals)
        interior = b > a+1
        # Difference arrays account for long-held states without expanding raw
        # samples or repeatedly scanning the whole time series for each bin.
        for destination, weights in ((coverage, np.full(interior.sum(), 300.)),
                                     (integral, vals[interior]*300)):
            changes = np.zeros(count+1)
            np.add.at(changes, a[interior]+1, weights)
            np.add.at(changes, b[interior], -weights)
            destination += np.cumsum(changes[:-1])
    complete = np.isclose(coverage, 300, rtol=0, atol=1e-6)
    observed_mean = np.divide(integral, coverage, out=np.full(count, np.nan), where=coverage > 0)
    sample_counts = np.zeros(count, dtype=int)
    inside = (times >= start_ns) & (times < end_ns)
    np.add.at(sample_counts, (times[inside]-start_ns)//step_ns, 1)
    return pd.DataFrame({'mean': np.where(complete, integral/300, np.nan),
        'observed_mean': observed_mean, 'coverage': coverage/300,
        'sample_count': sample_counts}, index=pd.date_range(start, periods=count, freq='5min'))


def endpoint_samples(samples, targets, *, max_hold_seconds=120):
    """Latest observation at/before each endpoint, with a bounded age; no interpolation."""
    samples = clean_samples(samples)
    positions = samples.index.searchsorted(targets, side='right')-1
    result = np.full(len(targets), np.nan)
    if not len(samples):
        return result
    safe = np.maximum(positions, 0)
    ages = (targets.as_unit('ns').asi8-samples.index.as_unit('ns').asi8[safe])/1e9
    valid = (positions >= 0) & (ages >= 0) & (ages <= max_hold_seconds)
    result[valid] = pd.to_numeric(samples, errors='coerce').to_numpy()[safe[valid]]
    return result
