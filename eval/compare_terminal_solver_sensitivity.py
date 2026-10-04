"""Fixed-forecast terminal-inventory sensitivity; isolated core, no savings claim."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.solver_replay import digest, validate_result, forecast_summary


def terminal_request(baseline, terminal_pct):
    if baseline.get('publication_authorized') is not False:
        raise ValueError('require unauthorised historical baseline')
    original = baseline['request']
    if original['request_id'] != digest({k: v for k, v in original.items() if k != 'request_id'}):
        raise ValueError('baseline request identity changed')
    validate_result(original, baseline['result'])
    if isinstance(terminal_pct, bool) or not isinstance(terminal_pct, (int, float)) or not np.isfinite(terminal_pct):
        raise ValueError('terminal percentage must be finite')
    plant = original['configuration']['plant_conf']
    endpoint = terminal_pct/100
    if not plant['battery_minimum_state_of_charge'] <= endpoint <= plant['battery_maximum_state_of_charge']:
        raise ValueError('terminal endpoint outside plant bounds')
    request = deepcopy(original)
    request.pop('request_id')
    request['payload']['soc_final'] = endpoint
    request['counterfactual'] = {'kind': 'terminal_soc_perturbation',
        'baseline_request_id': original['request_id'], 'baseline_terminal_soc': original['payload']['soc_final'],
        'terminal_soc': endpoint, 'publication_authorized': False}
    request['request_id'] = digest(request)
    return request


def compare(baseline, challenger):
    a, b = baseline['request'], challenger['request']
    if (challenger.get('publication_authorized') is not False or
            b['request_id'] != digest({k: v for k, v in b.items() if k != 'request_id'}) or
            b.get('counterfactual', {}).get('baseline_request_id') != a['request_id']):
        raise ValueError('challenger provenance changed')
    fa = validate_result(a, baseline['result'])
    fb = validate_result(b, challenger['result'])
    pa, pb = a['payload'], b['payload']
    # Different endpoints are intentional. Every other operational input is fixed.
    if ({k: v for k, v in pa.items() if k != 'soc_final'} !=
            {k: v for k, v in pb.items() if k != 'soc_final'} or
            a['configuration'] != b['configuration'] or a['forecast_start'] != b['forecast_start'] or
            a['kind'] != b['kind'] or a['handoff_revision'] != b['handoff_revision'] or
            a['optimization_sha256'] != b['optimization_sha256'] or baseline['image'] != challenger['image']):
        raise ValueError('comparison changed more than terminal endpoint')
    sa, sb = forecast_summary(a, fa), forecast_summary(b, fb)
    capacity = a['configuration']['plant_conf']['battery_nominal_energy_capacity']/1000
    energy = (sb['final_soc']-sa['final_soc'])*capacity
    cash = sb['cashflow_aud']-sa['cashflow_aud']
    step_h = pa['optimization_time_step']/60
    difference = fb.P_batt-fa.P_batt
    # Charge/discharge DC battery-energy throughput, not stored-energy delta.
    discharged_delta = float((fb.P_batt.clip(lower=0)-fa.P_batt.clip(lower=0)).sum()*step_h/1000)
    charged_delta = float((-fb.P_batt.clip(upper=0)+fa.P_batt.clip(upper=0)).sum()*step_h/1000)
    return {'scope': 'forecast_sensitivity_with_different_ending_inventory',
        'baseline_terminal_soc_pct': sa['final_soc']*100, 'terminal_soc_pct': sb['final_soc']*100,
        'ending_inventory_delta_kwh': energy, 'cashflow_delta_aud': cash,
        'inventory_value_break_even_aud_per_kwh': -cash/energy if abs(energy) > 1e-8 else None,
        'battery_throughput_delta_kwh': sb['battery_throughput_kwh']-sa['battery_throughput_kwh'],
        'battery_charge_delta_kwh': charged_delta, 'battery_discharge_delta_kwh': discharged_delta,
        'first_battery_discharge_baseline_w': float(fa.P_batt.iloc[0]),
        'first_battery_discharge_challenger_w': float(fb.P_batt.iloc[0]),
        'first_battery_discharge_delta_w': float(difference.iloc[0]),
        'changed_battery_intervals': int((difference.abs() > 1).sum()),
        'max_battery_discharge_delta_w': float(difference.abs().max()),
        'baseline': sa, 'challenger': sb}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terminal-soc-pct', type=float, action='append', required=True,
                        help='explicit endpoint; repeat for up to four bounded solves')
    args = parser.parse_args()
    if args.output.exists(): parser.error('output directory exists')
    if not 1 <= len(args.terminal_soc_pct) <= 4 or len(set(args.terminal_soc_pct)) != len(args.terminal_soc_pct):
        parser.error('require one to four distinct terminal endpoints')
    baseline = json.loads(args.baseline.read_text())
    requests = [terminal_request(baseline, value) for value in args.terminal_soc_pct]
    # Shared isolated worker runner; Docker network disabled, resources bounded.
    from scripts.replay_energy_solves import run_request
    results = []
    for request in requests:
        challenger = run_request(request, baseline['image'])
        results.append({'artifact': challenger, 'comparison': compare(baseline, challenger)})
        print(json.dumps(results[-1]['comparison'], allow_nan=False), flush=True)
    args.output.mkdir(parents=True)
    for number, result in enumerate(results):
        (args.output/f'challenger_{number}.json').write_text(json.dumps(result['artifact'], indent=2, allow_nan=False)+'\n')
    report = {'scope': 'terminal_inventory_forecast_sensitivity_not_realised_savings',
        'baseline_path': str(args.baseline), 'baseline_sha256': hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'comparisons': [result['comparison'] for result in results], 'publication_authorized': False,
        'limitations': ['different terminal inventory: cashflow alone is not a fair economic ranking',
            'break-even value excludes degradation/inverter stress and uncertainty beyond the horizon',
            'one frozen origin, no live feedback or changed future forecasts',
            'no tariff settlement or realised available-PV counterfactual included']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
