"""Offline frozen-plan timing sensitivity; policy owns modes, limits and physical SoC."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_pipeline.solver_replay import digest
from eval.audit_control_fidelity import asof
from eval.audit_ems_delivery import CURVES, STATE_MAX_AGE, execute_ems, selected_plan
from eval.minute_core_replay import between
from eval.summarize_feedback_chain import summarize


def static_guard(journal, start):
    """Capture-confirmed unchanged helper; explicitly weaker than historical receipts."""
    with sqlite3.connect('file:'+str(journal)+'?mode=ro', uri=True) as connection:
        records = [json.loads(row[0]) for row in connection.execute('SELECT record FROM handoffs')]
    record = max(records, key=lambda row: row['captured_at'])
    state = record['input_snapshot']['states']['input_number.battery_soc_min_export']
    clocks = [pd.Timestamp(state[key]) for key in ('last_changed', 'last_updated', 'last_reported')]
    value = float(state['state'])/100
    if not math.isfinite(value) or not 0 <= value <= 1 or max(clocks) > pd.Timestamp(start):
        raise ValueError('no capture-confirmed static historical export SOC guard')
    return value, {'kind': 'capture_confirmed_static_fallback_not_independent_historical_availability',
                   'journal_sha256': hashlib.sha256(journal.read_bytes()).hexdigest(),
                   'record_sha256': digest(record), 'capture': record['captured_at'],
                   'entity': state['entity_id'], 'state': state['state'],
                   'last_updated': state['last_updated'], 'last_reported': state['last_reported']}


def policy(plan, soc, minimum_export_soc, *, effective_general_price=None, discharge_weight=None,
           local_hour=None, allow_pv_charge=False):
    """Small verified YAML subset; unsupported choose branches stop the experiment."""
    if any(not math.isfinite(float(value)) for value in plan.values()):
        raise ValueError('nonfinite selected plan')
    if (plan['load'] <= 0 or plan['pv'] < 0
            or plan['hybrid'] < 0 or plan['curtailment'] != 0 or plan['grid'] > 0):
        raise ValueError('unsupported controller branch')
    if plan['battery'] < 0:
        if allow_pv_charge is not True or plan['grid'] != 0 or plan['pv'] <= 0:
            raise ValueError('unsupported controller charging branch')
        try:
            hour = float(local_hour)
        except (TypeError, ValueError):
            raise ValueError('missing causal local hour for PV charging branch') from None
        if not math.isfinite(hour) or not 0 <= hour < 24:
            raise ValueError('invalid causal local hour for PV charging branch')
        # This noncurtailing branch precedes the general zero-grid fallback.
        if 10 <= hour < 16:
            return {'mode': 'Maximum Self Consumption', 'export_kw': 10.,
                    'branch': 'pv_charge_prevent_discharge', 'discharge_limit_kw': 0.}
        return {'mode': 'Maximum Self Consumption', 'export_kw': 10.,
                'branch': 'pv_charge_self_consume'}
    # Zero-grid and PV-only price branches precede the export-SOC guard in YAML.
    if plan['grid'] == 0:
        return {'mode': 'Maximum Self Consumption', 'export_kw': 10., 'branch': 'self_consume'}
    if plan['battery'] == 0 and plan['pv'] > 0 and plan['grid'] < 0:
        # Earlier full-battery branch consumes a historical holdoff state that
        # this bounded policy does not possess; do not silently change its limits.
        if soc >= .995:
            raise ValueError('unsupported full-SOC PV-only export branch; missing holdoff input')
        try:
            general, weight = float(effective_general_price), float(discharge_weight)
        except (TypeError, ValueError):
            raise ValueError('missing causal PV-only general-price/discharge-weight inputs') from None
        if not math.isfinite(general) or not math.isfinite(weight):
            raise ValueError('nonfinite causal PV-only general-price/discharge-weight inputs')
        prevent = general <= weight
        return {'mode': 'Maximum Self Consumption', 'export_kw': 10.,
                'branch': 'pv_export_prevent_discharge' if prevent else 'pv_export_allow_discharge',
                'charge_limit_kw': 0., 'discharge_limit_kw': 0. if prevent else 24.}
    if plan['battery'] <= 0:
        raise ValueError('unsupported PV-only export branch')
    if soc < minimum_export_soc:
        return {'mode': 'Maximum Self Consumption', 'export_kw': 10., 'branch': 'low_soc_self_consume'}
    full = plan['hybrid'] >= 9979 or plan['grid'] <= -9999
    return {'mode': 'Command Discharging (PV First)',
            'export_kw': 10. if full else round(-plan['grid']/1000, 3),
            'branch': 'full_export' if full else 'partial_export'}


def guarded_controls(command, history, at, soc, minimum_export_soc):
    if asof(history['grid_status'], at, STATE_MAX_AGE)['state'] != 'On Grid':
        raise ValueError('unsupported off-grid operation')
    feed = float(asof(history['effective_feed'], at, 900)['value'])
    flexible = float(asof(history['flexible_export_limit'], at, STATE_MAX_AGE)['value'])
    if not math.isfinite(feed) or not math.isfinite(flexible) or flexible < 0:
        raise ValueError('invalid current export guard')
    mode = command['mode']
    if mode == 'Command Discharging (PV First)' and soc < minimum_export_soc:
        mode = 'Maximum Self Consumption'
    discharge, charge = command.get('discharge_limit_kw', 24.), command.get('charge_limit_kw', 21.)
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in (discharge, charge)):
        raise ValueError('invalid controller charge/discharge limit override')
    # Script defaults, not incumbent readbacks; caller guards plant hardware too.
    return {'ems_mode': {'state': mode}, 'grid_export_limit': {'value': min(command['export_kw'], flexible, 10. if feed > 0 else 0.)},
            'pcs_export_limit': {'value': 100.}, 'discharge_limit': {'value': discharge},
            'charge_limit': {'value': charge}}


def publication_events(history, start, end):
    """Freeze accepted current plans after changing entity receipts settle (<=2s)."""
    events = []
    for row in history['mpc_battery']:
        at = pd.Timestamp(row['time'])
        if not start < at < end:
            continue
        receipts = []
        # Unchanged curves may retain older receipts. Changed fields within the
        # bounded publication envelope determine activation; future data stays out.
        for name in CURVES:
            receipts.extend(pd.Timestamp(item['time']) for item in history['mpc_'+name]
                            if at-pd.Timedelta(seconds=2) <= pd.Timestamp(item['time']) <= at+pd.Timedelta(seconds=2))
        accepted = max([at]+receipts)
        if accepted >= end:
            continue
        plan, refs = selected_plan(history, accepted)
        if len({pd.Timestamp(ref['target']) for ref in refs.values()}) != 1:
            raise ValueError('mixed selected plan target clocks')
        events.append((accepted, plan, refs))
    return events


def replay(history, segments, plant, initial_soc, minimum_export_soc, *, dc_fixed_loss_w=0.):
    if dc_fixed_loss_w not in (0., 140.):
        raise ValueError('only explicit fixed DC loss ablations 0/140W supported')
    if minimum_export_soc > plant['battery_minimum_state_of_charge']:
        raise ValueError('unsupported export guard above physical SOC floor; intra-segment guard crossing required')
    start, end = pd.Timestamp(segments[0]['start']), pd.Timestamp(segments[-1]['end'])
    events = publication_events(history, start, end)
    by_time = {at: (plan, refs) for at, plan, refs in events}
    ticks = set(pd.date_range(start.ceil('5min'), end, freq='5min'))-{end}
    cuts = {start, end} | ticks | set(by_time)
    for key in ('effective_feed', 'flexible_export_limit', 'grid_status'):
        cuts.update(pd.Timestamp(row['time']) for row in history[key] if start < pd.Timestamp(row['time']) < end)
    initial_plan, initial_refs = selected_plan(history, start)
    accepted_clocks = [pd.Timestamp(initial_refs['battery']['publication'])]+[at for at, _, _ in events]+[end]
    if any(right-left > pd.Timedelta(seconds=120) for left, right in zip(accepted_clocks, accepted_clocks[1:])):
        raise ValueError('no bounded fresh MPC command coverage')
    results, logs = {}, []
    for arm in ('preceding_plan_fallback', 'hold_accepted_command'):
        soc = initial_soc
        command = policy(initial_plan, soc, minimum_export_soc)
        totals = dict.fromkeys(('variable_cost_aud', 'grid_import_kwh', 'grid_export_kwh', 'dc_throughput_kwh'), 0.)
        for left, right in zip(sorted(cuts), sorted(cuts)[1:]):
            if left in by_time:
                command = policy(by_time[left][0], soc, minimum_export_soc)
                logs.append({'arm': arm, 'at': left.isoformat(), 'event': 'current_plan_activation', 'command': command.copy(),
                             'receipts': by_time[left][1]})
            elif left in ticks and arm == 'preceding_plan_fallback':
                plan, refs = selected_plan(history, left)
                command = policy(plan, soc, minimum_export_soc)
                logs.append({'arm': arm, 'at': left.isoformat(), 'event': 'five_minute_fallback', 'command': command.copy(), 'receipts': refs})
            for actual in between(segments, left, right):
                controls = guarded_controls(command, history, pd.Timestamp(actual['start']), soc, minimum_export_soc)
                out = execute_ems(plant, soc, actual | {'controls': controls}, dc_fixed_loss_w=dc_fixed_loss_w)
                for key in ('grid_import_kwh', 'grid_export_kwh'):
                    totals[key] += out[key]
                totals['variable_cost_aud'] += out['grid_import_kwh']*actual['general_rate']-out['grid_export_kwh']*actual['feed_rate']
                totals['dc_throughput_kwh'] += abs(out['battery_discharge_w'])*actual['duration_seconds']/3_600_000
                soc = out['end_soc']
        results[arm] = totals | {'initial_soc': initial_soc, 'final_soc': soc,
                                'ending_inventory_kwh': soc*plant['battery_nominal_energy_capacity']/1000}
    base, challenger = results.values()
    credit = base['variable_cost_aud']-challenger['variable_cost_aud']
    inventory = challenger['ending_inventory_kwh']-base['ending_inventory_kwh']
    throughput = challenger['dc_throughput_kwh']-base['dc_throughput_kwh']
    return {'scope': 'frozen_incumbent_plan_conditioned_timing_sensitivity', 'start': start.isoformat(), 'end': end.isoformat(),
            'dc_fixed_loss_w': dc_fixed_loss_w,
            'summary': results, 'comparison': {'variable_credit_gain_aud': credit, 'ending_inventory_delta_kwh': inventory,
                'dc_throughput_delta_kwh': throughput,
                'break_even_inventory_value_aud_per_kwh_before_wear': -credit/inventory if inventory < -1e-12 else None,
                'net_value_formula': 'credit_gain + inventory_delta * inventory_value - throughput_delta * wear_cost'},
            'events': logs, 'limitations': ['frozen incumbent plans; no own-SOC solver feedback; no deployable savings estimate',
                'instant modeled command activation after bounded publication receipts; no script latency, physical ramp or transient PCS controller',
                'supported on-grid noncharging noncurtailing battery export/self-consume branches only; other branches fail',
                'own SoC and policy mode/limits; no recorded incumbent EMS mode/limit trace used',
                'fixed-efficiency ideal equilibrium, explicit fixed DC loss ablation and delivered PV lower bound; inventory fidelity unresolved',
                'static export SOC helper capture fallback is not independent historical availability',
                'export SOC guard must be at or below physical SOC floor; higher crossing cases fail',
                'observed Amber rates score outcomes; causal effective-feed receipts govern export permission'],
            'publication_authorized': False}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('history', 'journal', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--replay', type=Path, nargs='+', required=True)
    parser.add_argument('--dc-fixed-loss-w', type=float, choices=(0., 140.), default=0.)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output required')
    manifest = json.loads((args.history/'manifest.json').read_text())
    path = args.history/'history.json'
    if not manifest['export_complete'] or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['history_sha256']:
        raise ValueError('changed or incomplete history')
    chain = summarize(args.replay)
    bundles = [json.loads((folder/'bundle.json').read_text()) for folder in args.replay]
    segments = [actual for bundle in bundles for actual in bundle.get('prelude', [])+[
        row for step in bundle['steps'] for key in ('before_activation', 'after_activation') for row in step[key]]]
    guard, provenance = static_guard(args.journal, segments[0]['start'])
    if not pd.Timestamp(manifest['start']) <= pd.Timestamp(segments[0]['start']) < pd.Timestamp(segments[-1]['end']) <= pd.Timestamp(manifest['end']):
        raise ValueError('execution outside source archive')
    result = replay(json.loads(path.read_text()), segments, bundles[0]['execution_plant'], bundles[0]['initial_soc'], guard,
                    dc_fixed_loss_w=args.dc_fixed_loss_w)
    result['provenance'] = {'history_sha256': manifest['history_sha256'], 'static_export_guard': provenance,
                            'dc_fixed_loss_w': args.dc_fixed_loss_w,
                            'replay_lineage': chain['lineage'], 'bundle_sha256': [digest(bundle) for bundle in bundles],
                            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                              for name in ('eval/audit_control_fidelity.py', 'eval/measured_actuals.py', 'eval/sequential_core_replay.py', 'eval/audit_ems_delivery.py', 'eval/dh_feedback_replay.py', 'eval/feedback_checkpoint.py', 'eval/summarize_feedback_chain.py')},
                            'yaml_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                                           for name in ('hass/automation-sigenergy-emhass.yaml', 'hass/packages/sigenergy_ems.yaml')}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
