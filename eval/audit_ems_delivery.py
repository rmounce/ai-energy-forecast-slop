"""Read-only EMS mode/limit delivery audit; recorded controls are not a policy challenger."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from energy_pipeline.solver_replay import digest
from eval.audit_control_fidelity import asof, parse_curve
from eval.dh_feedback_replay import between
from eval.summarize_feedback_chain import summarize
from eval.sequential_core_replay import execute

CURVES = {'battery': ('battery_scheduled_power_str', 'mpc_p_batt_forecast'),
          'grid': ('forecasts_str', 'mpc_p_grid_forecast'),
          'load': ('forecasts_str', 'mpc_p_load_forecast'),
          'pv': ('forecasts_str', 'mpc_p_pv_forecast'),
          'hybrid': ('forecasts_str', 'mpc_p_hybrid_inverter'),
          'curtailment': ('forecasts_str', 'mpc_p_pv_curtailment')}
CONTROLS = ('ems_mode', 'grid_export_limit', 'pcs_export_limit', 'discharge_limit', 'charge_limit')
STATE_MAX_AGE = 31*86400  # Stable register/mode events, not periodic telemetry receipts.


def selected_plan(history, at):
    """HA uses each entity's last row <=now, even from a preceding plan."""
    at = pd.Timestamp(at)
    values, refs = {}, {}
    for name, (column, key) in CURVES.items():
        row = asof(history['mpc_'+name], at, STATE_MAX_AGE)
        curve = parse_curve(row[column], key)
        available = [point for point in curve if pd.Timestamp(point['date']) <= at]
        if not available:
            raise ValueError('no causal selected plan target: '+name)
        point = available[-1]
        values[name] = float(point[key])
        refs[name] = {'publication': row['time'], 'target': point['date']}
    return values, refs


def bounded_self_consume_branch(plan, grid_status):
    """Exact supported subset of YAML choose order; no guessed export-SOC guard."""
    if (grid_status == 'On Grid' and plan['grid'] == 0 and plan['battery'] >= 0 and plan['load'] > 0
            and plan['pv'] >= 0 and plan['hybrid'] >= 0 and plan['curtailment'] == 0):
        return 'Self-consume from battery'
    return None


def control_segments(history, segments):
    """Cut raw execution holds at every recorded mode/register change."""
    start, end = pd.Timestamp(segments[0]['start']), pd.Timestamp(segments[-1]['end'])
    streams = {key: history[key] for key in CONTROLS if key != 'charge_limit'}
    # The existing PV-mode sensor records charging-limit readback as an attribute.
    # Retain actual change clocks; do not renew stable-state receipts each telemetry tick.
    charging, previous = [], None
    for row in history['mode']:
        value = row.get('charge_limit_kw')
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError('missing/invalid charging-limit readback')
        if value != previous:
            charging.append({'time': row['time'], 'value': value})
            previous = value
    streams['charge_limit'] = charging
    cuts = {start, end}
    for key in CONTROLS:
        cuts.update(pd.Timestamp(row['time']) for row in streams[key]
                    if start < pd.Timestamp(row['time']) < end)
    result = []
    for left, right in zip(sorted(cuts), sorted(cuts)[1:]):
        states = {key: asof(streams[key], left, STATE_MAX_AGE) for key in CONTROLS}
        for actual in between(segments, left, right):
            result.append(actual | {'controls': states})
    return result


def execute_ems(plant, soc, actual, *, use_pcs_limit=True):
    """Ideal equilibrium behind recorded grid cap/mode, with physical energy limits."""
    states = actual['controls']
    mode = states['ems_mode']['state']
    if mode not in ('Maximum Self Consumption', 'Command Discharging (PV First)'):
        raise ValueError('unsupported recorded EMS mode: '+mode)
    limits = {key: float(states[key]['value']) for key in CONTROLS if key != 'ems_mode'}
    if any(not math.isfinite(value) or value < 0 for value in limits.values()):
        raise ValueError('invalid recorded power limit')
    effective = deepcopy(plant)
    effective['battery_charge_power_max'] = min(plant['battery_charge_power_max'], limits['charge_limit']*1000)
    effective['battery_discharge_power_max'] = min(plant['battery_discharge_power_max'], limits['discharge_limit']*1000)
    if use_pcs_limit:
        effective['inverter_ac_output_max'] = min(plant['inverter_ac_output_max'], limits['pcs_export_limit']*1000)
    cap = min(plant['maximum_power_to_grid'], limits['grid_export_limit']*1000)
    pv, load = actual['pv_dc_w'], actual['load_site_w']
    ac = load if mode == 'Maximum Self Consumption' else min(effective['inverter_ac_output_max'], load+cap)
    command = ac/plant['inverter_efficiency_dc_ac']-pv
    if mode == 'Command Discharging (PV First)':
        command = max(0., command)  # PV first; this mode does not command battery charging.
    return execute(effective, soc, command, 0., pv, load, cap,
                   duration_seconds=actual['duration_seconds'])


def integrate_observed(segments):
    out = {'duration_seconds': 0., 'grid_import_kwh': 0., 'grid_export_kwh': 0.,
           'dc_discharge_kwh': 0., 'variable_cost_aud': 0.}
    for row in segments:
        dt = row['duration_seconds']/3_600_000
        imported, exported = max(row['observed_grid_import_w'], 0)*dt, max(-row['observed_grid_import_w'], 0)*dt
        out['duration_seconds'] += row['duration_seconds']
        out['grid_import_kwh'] += imported
        out['grid_export_kwh'] += exported
        out['dc_discharge_kwh'] += max(-row['observed_battery_charge_w'], 0)*dt
        out['variable_cost_aud'] += imported*row['general_rate']-exported*row['feed_rate']
    return out


def audit(history, segments, plant, initial_soc, baseline_summary, observed_final_soc):
    start, end = pd.Timestamp(segments[0]['start']), pd.Timestamp(segments[-1]['end'])
    ticks = []
    for tick in pd.date_range(start.ceil('5min'), end, freq='5min'):
        if tick >= end:
            continue
        plan, refs = selected_plan(history, tick)
        future = [pd.Timestamp(row['time']) for row in history['mpc_battery'] if tick < pd.Timestamp(row['time']) < end]
        publication = min(future) if future else end
        if publication-tick > pd.Timedelta(seconds=120):
            raise ValueError('no bounded next MPC publication')
        interval = between(segments, tick, publication)
        action = asof(history['ems_action'], min(tick+pd.Timedelta(seconds=2), publication), STATE_MAX_AGE)
        # Labels are evidence of the selected branch, not script completion or successful register writes.
        changes = [row for row in history['ems_mode'] if tick <= pd.Timestamp(row['time']) < publication]
        ticks.append({'tick': tick.isoformat(), 'plan': plan, 'plan_receipts': refs,
                      'supported_expected_action': bounded_self_consume_branch(plan, asof(history['grid_status'], tick, STATE_MAX_AGE)['state']),
                      'observed_action': action['state'], 'action_receipt': action['time'],
                      'next_mpc_publication': publication.isoformat(), 'mode_changes_before_publication': changes,
                      'observed_before_publication': integrate_observed(interval)})
    controlled = control_segments(history, segments)
    physical = {}
    for arm, use_pcs in (('recorded_mode_grid_and_pcs', True), ('recorded_mode_and_grid_only', False)):
        soc, total = initial_soc, {'grid_import_kwh': 0., 'grid_export_kwh': 0.,
                                           'variable_cost_aud': 0., 'dc_discharge_kwh': 0.}
        for actual in controlled:
            out = execute_ems(plant, soc, actual, use_pcs_limit=use_pcs)
            for key in ('grid_import_kwh', 'grid_export_kwh'):
                total[key] += out[key]
            total['variable_cost_aud'] += out['grid_import_kwh']*actual['general_rate']-out['grid_export_kwh']*actual['feed_rate']
            total['dc_discharge_kwh'] += max(out['battery_discharge_w'], 0)*actual['duration_seconds']/3_600_000
            soc = out['end_soc']
        total.update(initial_soc=initial_soc, final_soc=soc,
                     ending_inventory_kwh=soc*plant['battery_nominal_energy_capacity']/1000)
        physical[arm] = total
    observed = integrate_observed(segments)
    original = baseline_summary
    net = lambda row: row['grid_export_kwh']-row['grid_import_kwh']
    return {'scope': 'recorded_control_conditioned_delivery_diagnostic', 'start': start.isoformat(),
            'end': end.isoformat(), 'ticks': ticks, 'observed': observed, 'physical': physical,
            'diagnostics': {'original_ideal_net_export_error_kwh': net(original)-net(observed),
                'recorded_control_net_export_error_kwh': net(physical['recorded_mode_grid_and_pcs'])-net(observed),
                'original_ideal_ending_inventory_error_kwh': baseline_summary['ending_inventory_kwh']-observed_final_soc*plant['battery_nominal_energy_capacity']/1000,
                'recorded_control_ending_inventory_error_kwh': (physical['recorded_mode_grid_and_pcs']['final_soc']-observed_final_soc)*plant['battery_nominal_energy_capacity']/1000,
                'supported_tick_count': sum(t['supported_expected_action'] is not None for t in ticks),
                'supported_tick_action_matches': sum(t['supported_expected_action'] == t['observed_action']
                    for t in ticks if t['supported_expected_action'] is not None)},
            'limitations': ['recorded mode/register timestamps are asynchronous readbacks, not command completion clocks',
                'existing YAML supported self-consume subset only; unsupported branches not guessed',
                'recorded control trace is exogenous and cannot rank counterfactual forecasts or establish savings',
                'ideal equilibrium; no physical ramp, firmware response, fixed-loss or optimistic-select model',
                'recorded PCS register already incorporates any transient cap; do not apply the cap helper twice',
                'delivered PV lower bound and one initial battery SoC; no later recorded SoC resets',
                'stable-state31day bound is not proof of continuous availability; plan receipt/target clocks explicit',
                'current repository YAML semantics; no independently captured historical configuration version'],
            'publication_authorized': False}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('history', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--replay', type=Path, nargs='+', required=True, help='ordered initial batch and optional verified continuations')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output file required')
    manifest = json.loads((args.history/'manifest.json').read_text())
    path = args.history/'history.json'
    if not manifest['export_complete'] or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['history_sha256']:
        raise ValueError('changed or incomplete control archive')
    chain = summarize(args.replay)
    bundles = [json.loads((folder/'bundle.json').read_text()) for folder in args.replay]
    reports = [json.loads((folder/'report.json').read_text()) for folder in args.replay]
    first = bundles[0]
    segments = [row for bundle in bundles for row in bundle.get('prelude',[])+[
        actual for step in bundle['steps'] for key in ('before_activation','after_activation') for actual in step[key]]]
    result = audit(json.loads(path.read_text()), segments, first['execution_plant'], first['initial_soc'],
                   chain['summary']['baseline'], reports[-1]['observed']['final_soc'])
    if not pd.Timestamp(manifest['start']) <= pd.Timestamp(result['start']) < pd.Timestamp(result['end']) <= pd.Timestamp(manifest['end']):
        raise ValueError('execution span outside archive')
    result['provenance'] = {'control_history_sha256': manifest['history_sha256'],
        'replay_bundle_sha256': [digest(bundle) for bundle in bundles], 'replay_lineage': chain['lineage'],
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'yaml_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                       for name in ('hass/automation-sigenergy-emhass.yaml','hass/packages/sigenergy_ems.yaml')}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'diagnostics': result['diagnostics'], 'physical': result['physical'], 'observed': result['observed']}))


if __name__ == '__main__':
    main()
