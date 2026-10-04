"""Synthetic load-information value at a battery reserve boundary; no historical skill claim."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.controlled_cycle_scenarios import PLANT, phase, run_arm, net_value

ARMS = ('incumbent', 'modest_correction', 'worsened_bias', 'perfect_load_oracle')
EXPORT_CANDIDATES_KW = tuple(index/20 for index in range(41))


def objective(result, terminal_value=.20, wear_cost=.04):
    if not all(math.isfinite(x) and x >= 0 for x in (terminal_value, wear_cost)):
        raise ValueError('value assumptions must be finite and nonnegative')
    return (result['variable_cost_aud'] + result['dc_throughput_kwh']*wear_cost
            - result['ending_inventory_kwh']*terminal_value)


def select_export(forecast_phases, initial_soc, *, terminal_value=.20, wear_cost=.04, dc_fixed_loss_w=140.):
    """Enumerate one early export cap using forecast inputs and carried physical inventory."""
    scored = []
    for cap in EXPORT_CANDIDATES_KW:
        planned = deepcopy(forecast_phases)
        planned[0]['export_cap_kw'] = cap
        result = run_arm(planned, initial_soc, cap > 0, dc_fixed_loss_w=dc_fixed_loss_w)
        scored.append((objective(result, terminal_value, wear_cost), cap))
    score, cap = min(scored)  # Stable tie-break: lowest cap.
    return {'export_cap_kw': cap, 'forecast_objective_aud': score,
            'candidate_count': len(scored)}


def scenario(stock, demand, solar, *, dc_fixed_loss_w=140.):
    if stock not in ('scarce', 'ample') or demand not in ('low', 'high') or not isinstance(solar, bool):
        raise ValueError('unsupported scenario labels')
    actual_load_kwh = 1.2 if demand == 'low' else 2.4
    initial_soc = (.15 + 2.5/40.3) if stock == 'scarce' else .95
    common = [phase('early_export', 60, load_w=300., feed_rate=.30)]
    if solar:
        common.append(phase('later_solar', 60, pv_w=8000., load_w=300., export_cap_kw=0., feed_rate=0.))
    common.append(phase('later_demand', 60, load_w=actual_load_kwh*1000, general_rate=.50))
    # Deliberately stipulated errors, not estimated from actuals or fitted calibration.
    forecasts = {'incumbent': 1.8 if demand == 'low' else 1.3,
                 'modest_correction': 1.3 if demand == 'low' else 1.8,
                 'worsened_bias': 2.3 if demand == 'low' else .8,
                 'perfect_load_oracle': actual_load_kwh}
    arms = {}
    for name, load_kwh in forecasts.items():
        predicted = deepcopy(common)
        predicted[-1]['load_site_w'] = load_kwh*1000
        decision = select_export(predicted, initial_soc, dc_fixed_loss_w=dc_fixed_loss_w)
        realized = deepcopy(common)
        realized[0]['export_cap_kw'] = decision['export_cap_kw']
        result = run_arm(realized, initial_soc, decision['export_cap_kw'] > 0, dc_fixed_loss_w=dc_fixed_loss_w)
        arms[name] = result | {'decision': decision, 'assumed_future_load_kwh': load_kwh,
                             'absolute_future_load_error_kwh': abs(load_kwh-actual_load_kwh)}
    comparisons = {}
    base = arms['incumbent']
    for name in ARMS[1:]:
        other = arms[name]
        comparison = {'variable_credit_gain_aud': base['variable_cost_aud']-other['variable_cost_aud'],
                      'ending_inventory_delta_kwh': other['ending_inventory_kwh']-base['ending_inventory_kwh'],
                      'dc_throughput_delta_kwh': other['dc_throughput_kwh']-base['dc_throughput_kwh']}
        comparison['net_value_aud_at_decision_assumptions'] = net_value(comparison, .20, .04)
        comparison['value_sensitivity'] = [
            {'terminal_value_aud_per_kwh': terminal, 'wear_aud_per_dc_kwh': wear,
             'net_value_aud': net_value(comparison, terminal, wear)}
            for terminal in (0., .20, .30, .50) for wear in (0., .04)]
        comparisons[name] = comparison
    return {'stock': stock, 'future_demand': demand, 'otherwise_curtailed_later_solar': solar,
            'initial_soc': initial_soc, 'dc_fixed_loss_w': dc_fixed_loss_w,
            'common_realized_phases': common,
            'actual_future_load_kwh': actual_load_kwh, 'arms': arms, 'comparisons': comparisons}


def suite():
    return {'scope': 'synthetic_forecast_information_mechanism_not_historical_savings',
            'plant': PLANT, 'decision': {'terminal_value_aud_per_kwh': .20,
                                        'wear_aud_per_dc_kwh': .04,
                                        'early_export_candidates_kw': EXPORT_CANDIDATES_KW},
            'scenarios': [scenario(stock, demand, solar) for stock in ('scarce', 'ample')
                          for demand in ('low', 'high') for solar in (False, True)],
            'limitations': [
                'Load errors and corrections stipulated; corrected arm knows synthetic scenario label.',
                'Correction 0.5 kWh resembles magnitude of recorded 14h corrections (0.48-0.70 kWh), but is concentrated in one hour; not a calibration replay.',
                'All arms share realized load, PV, prices, initial inventory and physical executor; no state resets.',
                'Oracle is diagnostic upper bound within this restricted export-cap policy and stated value assumptions.',
                'Future solar and prices known equally to every arm; no APF changes or price skill claims.',
                'One early decision followed by live self-consumption; not installed optimizer or deployable controller.',
                'Nominal SoC stock, fixed efficiencies and 140W DC loss; no device ramp or full-battery holdoff.',
                'Wear and terminal values illustrative; sensitivity changes valuation, not the selected decisions.'],
            'publication_authorized': False}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = suite()
    root = Path(__file__).resolve().parents[1]
    result['code_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                             for name in ('eval/controlled_load_information.py',
                                          'eval/controlled_cycle_scenarios.py',
                                          'eval/audit_ems_delivery.py', 'eval/sequential_core_replay.py')}
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'scope': result['scope'], 'scenarios': len(result['scenarios'])}))


if __name__ == '__main__':
    main()
