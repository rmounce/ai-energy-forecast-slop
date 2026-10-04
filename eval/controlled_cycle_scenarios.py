"""Paired synthetic battery-cycle mechanisms; not measured savings or forecast skill."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.audit_ems_delivery import execute_ems


PLANT = {
    'battery_nominal_energy_capacity': 40300., 'battery_minimum_state_of_charge': .15,
    'battery_maximum_state_of_charge': 1., 'battery_charge_efficiency': .99,
    'battery_discharge_efficiency': .99, 'inverter_efficiency_dc_ac': .95,
    'inverter_efficiency_ac_dc': .95, 'battery_charge_power_max': 20000.,
    'battery_discharge_power_max': 20000., 'inverter_ac_output_max': 9980.,
    'inverter_ac_input_max': 9980., 'maximum_power_from_grid': 15000.,
    'maximum_power_to_grid': 10000., 'inverter_is_hybrid': True,
}
ARMS = ('baseline', 'extra_early_export')


def controls(extra_export, cap_kw):
    return {'ems_mode': {'state': 'Command Discharging (PV First)' if extra_export
                         else 'Maximum Self Consumption'},
            'grid_export_limit': {'value': cap_kw}, 'pcs_export_limit': {'value': 100.},
            'discharge_limit': {'value': 24.}, 'charge_limit': {'value': 21.}}


def phase(name, minutes, *, pv_w=0., load_w=300., general_rate=.40, feed_rate=.20,
          export_cap_kw=10.):
    if minutes <= 0 or minutes % 5:
        raise ValueError('phase minutes must be a positive multiple of five')
    return {'name': name, 'minutes': minutes, 'pv_dc_w': pv_w, 'load_site_w': load_w,
            'general_rate': general_rate, 'feed_rate': feed_rate,
            'export_cap_kw': export_cap_kw}


def run_arm(phases, initial_soc, extra_export, *, plant=None, dc_fixed_loss_w=140.):
    plant = PLANT if plant is None else plant
    soc = initial_soc
    capacity_kwh = plant['battery_nominal_energy_capacity']/1000
    totals = dict.fromkeys(('variable_cost_aud', 'grid_import_kwh', 'grid_export_kwh',
                           'dc_throughput_kwh', 'curtailed_pv_dc_kwh'), 0.)
    logs = []
    for item in phases:
        flow = dict.fromkeys(totals, 0.)
        lower_hits = upper_hits = clipped = 0
        start_soc = soc
        for _ in range(item['minutes']//5):
            actual = item | {'duration_seconds': 300,
                             'controls': controls(extra_export and item['name'] == 'early_export',
                                                  item['export_cap_kw'])}
            out = execute_ems(plant, soc, actual, dc_fixed_loss_w=dc_fixed_loss_w)
            for key in ('grid_import_kwh', 'grid_export_kwh'):
                flow[key] += out[key]
            flow['variable_cost_aud'] += (out['grid_import_kwh']*item['general_rate']
                                         - out['grid_export_kwh']*item['feed_rate'])
            flow['dc_throughput_kwh'] += out['battery_throughput_kwh']
            flow['curtailed_pv_dc_kwh'] += out['curtailed_pv_w']/12000
            soc = out['end_soc']
            lower_hits += int(abs(soc-plant['battery_minimum_state_of_charge']) < 1e-10)
            upper_hits += int(abs(soc-plant['battery_maximum_state_of_charge']) < 1e-10)
            clipped += int(out['command_clipped'])
        for key, value in flow.items():
            totals[key] += value
        logs.append({'name': item['name'], 'start_soc': start_soc, 'end_soc': soc,
                     'floor_binding_steps': lower_hits, 'capacity_binding_steps': upper_hits,
                     'command_clipped_steps': clipped, **flow})
    return {**totals, 'initial_soc': initial_soc, 'final_soc': soc,
            'ending_inventory_kwh': soc*capacity_kwh, 'phases': logs}


def paired(phases, initial_soc, *, plant=None, dc_fixed_loss_w=140.):
    arms = {name: run_arm(phases, initial_soc, index == 1, plant=plant,
                          dc_fixed_loss_w=dc_fixed_loss_w) for index, name in enumerate(ARMS)}
    base, extra = (arms[name] for name in ARMS)
    cash = base['variable_cost_aud']-extra['variable_cost_aud']
    stock = extra['ending_inventory_kwh']-base['ending_inventory_kwh']
    throughput = extra['dc_throughput_kwh']-base['dc_throughput_kwh']
    return {'initial_soc': initial_soc, 'common_exogenous_phases': phases,
            'dc_fixed_loss_w': dc_fixed_loss_w, 'arms': arms,
            'comparison': {'variable_credit_gain_aud': cash,
                           'ending_inventory_delta_kwh': stock,
                           'dc_throughput_delta_kwh': throughput,
                           'net_value_formula': 'credit_gain + inventory_delta * terminal_value - throughput_delta * wear_cost'}}


def net_value(comparison, terminal_value, wear_cost):
    if not all(math.isfinite(x) and x >= 0 for x in (terminal_value, wear_cost)):
        raise ValueError('terminal value and wear cost must be finite and nonnegative')
    return (comparison['variable_credit_gain_aud']
            + comparison['ending_inventory_delta_kwh']*terminal_value
            - comparison['dc_throughput_delta_kwh']*wear_cost)


def solar_replacement(fraction, *, exportable=False, early_feed=.20, later_feed=.05,
                      dc_fixed_loss_w=140.):
    """Specify solar from common baseline headroom + fraction of initial stock gap."""
    if not math.isfinite(fraction) or not 0 <= fraction <= 1.25:
        raise ValueError('replacement fraction must be between zero and 1.25')
    initial_soc = .95
    early = phase('early_export', 15, export_cap_kw=2., feed_rate=early_feed)
    before = paired([early], initial_soc, dc_fixed_loss_w=dc_fixed_loss_w)
    base = before['arms']['baseline']
    gap = -before['comparison']['ending_inventory_delta_kwh']
    headroom = PLANT['battery_nominal_energy_capacity']/1000-base['ending_inventory_kwh']
    # One hour: PV covers common site load/overhead plus stipulated stored-energy input.
    pv_w = 300/PLANT['inverter_efficiency_dc_ac']+dc_fixed_loss_w
    pv_w += (headroom+fraction*gap)*1000/PLANT['battery_charge_efficiency']
    solar = phase('later_solar', 60, pv_w=pv_w,
                  export_cap_kw=10. if exportable else 0., feed_rate=later_feed if exportable else 0.)
    result = paired([early, solar], initial_soc, dc_fixed_loss_w=dc_fixed_loss_w)
    result['scenario'] = 'exportable_solar_replacement' if exportable else 'otherwise_curtailed_solar_replacement'
    result['construction'] = {'replacement_fraction': fraction, 'initial_extra_stock_spent_kwh': gap,
                              'baseline_solar_headroom_kwh': headroom,
                              'solar_curve_kind': 'synthetic_common_input_not_measured_or_forecast'}
    return result


def scarce_later_energy(*, early_feed=.20, later_general=.40, dc_fixed_loss_w=140.):
    phases = [phase('early_export', 15, export_cap_kw=2., feed_rate=early_feed),
              phase('later_demand', 60, load_w=2000., general_rate=later_general)]
    result = paired(phases, .20, dc_fixed_loss_w=dc_fixed_loss_w)
    result['scenario'] = 'scarce_later_energy_both_arms_reach_floor'
    return result


def suite():
    scenarios = [solar_replacement(fraction, exportable=exportable)
                 for exportable in (False, True) for fraction in (0., .25, .5, .75, 1., 1.25)]
    scenarios += [scarce_later_energy(early_feed=rate) for rate in (0., .20, .40, .60)]
    for row in scenarios:
        row['value_sensitivity'] = [
            {'terminal_value_aud_per_kwh': terminal, 'wear_aud_per_dc_kwh': wear,
             'net_value_aud': net_value(row['comparison'], terminal, wear)}
            for terminal in (0., .20, .30, .40) for wear in (0., .04)]
    return {'scope': 'controlled_paired_cycle_mechanisms_not_historical_savings',
            'plant': PLANT, 'scenarios': scenarios,
            'limitations': ['Synthetic common PV/load/rate curves; no measured history or APF skill evaluation.',
                            'Prescribed initial extra export; no optimiser feedback or deployable controller policy.',
                            'Fixed efficiencies, ideal equilibrium, nominal SoC stock and explicit 140W DC overhead.',
                            'Wear and terminal values are illustrative linear sensitivities, not measured costs.',
                            'Zero export cap represents otherwise-curtailed solar; exportable variant prices foregone export.',
                            'No device ramp, full-battery holdoff, temperature, tariff fixed charges or battery ageing model.'],
            'publication_authorized': False}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = suite()
    root = Path(__file__).resolve().parents[1]
    result['code_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('eval/controlled_cycle_scenarios.py', 'eval/audit_ems_delivery.py',
                                          'eval/sequential_core_replay.py')}
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'scenarios': len(result['scenarios']), 'scope': result['scope']}))


if __name__ == '__main__':
    main()
