"""Minute-cadence MPC policy comparison conditioned on archived DH/HWC parents."""
import argparse
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.payloads import Inputs, build_mpc_payload, soc_points, interpolate, boundary
from energy_pipeline.solver_replay import digest, prepare_request, validate_result
from eval.audit_control_fidelity import asof, numeric_series, recorded_states
from eval.measured_actuals import clean_samples
from eval.amber_quote_actuals import canonical_rates
from eval.sequential_core_replay import snapshot_at, execute


def execution_segments(history, start, end, rates):
    """Raw sample-and-hold targets on their joint event grid; no averaging/filling."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not start < end <= start+pd.Timedelta(hours=1):
        raise ValueError('execution window must be positive and <=1h')
    streams = {key: clean_samples(numeric_series(history[key])) for key in ('pv', 'load', 'battery', 'grid')}
    cuts = {start, end}
    for samples in streams.values():
        for stamp in samples.index:
            if start < stamp < end: cuts.add(stamp)
            expiry = stamp+pd.Timedelta(seconds=120)
            if start < expiry < end: cuts.add(expiry)
    cuts.update(pd.date_range(start.ceil('5min'), end, freq='5min'))
    cuts = sorted(cuts)
    result = []
    for left, right in zip(cuts, cuts[1:]):
        values = {}
        for key, samples in streams.items():
            position = samples.index.searchsorted(left, side='right')-1
            if position < 0 or (right-samples.index[position]).total_seconds() > 120+1e-6:
                raise ValueError('incomplete held execution target: '+key)
            values[key] = float(samples.iloc[position])
        if not np.isfinite(list(values.values())).all() or min(values['pv'], values['load']) < 0:
            raise ValueError('invalid execution targets')
        target = left.floor('5min')
        result.append({'start': left.isoformat(), 'end': right.isoformat(), 'duration_seconds': (right-left).total_seconds(),
            'pv_dc_w': values['pv'], 'load_site_w': values['load'], 'observed_battery_charge_w': values['battery'],
            'observed_grid_import_w': values['grid'], 'general_rate': float(rates['general'].loc[target, 'rate']),
            'feed_rate': float(rates['feed'].loc[target, 'rate'])})
    return result


def policy_payload(states, origin, soc, arm):
    """Each arm's simulated current SoC; observed SoC is never re-grounded here."""
    states = deepcopy(states)
    states['sensor.sigen_plant_battery_state_of_charge_derived'] = {'state': str(soc*100)}
    inputs = Inputs(states, pd.Timestamp(origin).to_pydatetime())
    payload = build_mpc_payload(inputs)
    if arm == 'without_positive_lockin':
        anchor = inputs.value('input_number.dh_last_soc_init')
        points = soc_points(inputs.attr('sensor.dh_soc_batt_forecast', 'battery_scheduled_soc') or [], anchor if anchor > 0 else soc*100)
        target = interpolate(points, boundary(inputs.now, 5)+timedelta(hours=14), soc*100)
        payload['soc_final'] = round(min(100, max(0, target))/100, 4)
    elif arm != 'baseline':
        raise ValueError('unknown minute replay policy')
    return payload


def advance(plant, soc, command, segments):
    totals = {'variable_cost_aud': 0., 'grid_import_kwh': 0., 'grid_export_kwh': 0.,
        'battery_throughput_kwh': 0., 'curtailed_pv_kwh': 0., 'clipped_seconds': 0.,
        'executed_discharge_ws': 0.}
    for actual in segments:
        duration = actual['duration_seconds']
        result = execute(plant, soc, command['battery_w'], command['curtail_w'], actual['pv_dc_w'],
            actual['load_site_w'], command['export_limit_w'], duration_seconds=duration)
        for key in ('grid_import_kwh', 'grid_export_kwh', 'battery_throughput_kwh'):
            totals[key] += result[key]
        totals['variable_cost_aud'] += result['grid_import_kwh']*actual['general_rate']-result['grid_export_kwh']*actual['feed_rate']
        totals['curtailed_pv_kwh'] += result['curtailed_pv_w']*duration/3_600_000
        totals['clipped_seconds'] += duration*result['command_clipped']
        totals['executed_discharge_ws'] += result['battery_discharge_w']*duration
        soc = result['end_soc']
    return soc, totals


def simulate(bundle, solve):
    if not 1 <= len(bundle['steps']) <= 30: raise ValueError('require 1–30 paired origins')
    started = time.monotonic()
    arms = ('baseline', 'without_positive_lockin')
    inventory = {arm: bundle['initial_soc'] for arm in arms}
    pending = {arm: deepcopy(bundle['initial_command']) for arm in arms}
    totals = {arm: {} for arm in arms}
    rows, artifacts = [], []
    for step in bundle['steps']:
        if time.monotonic()-started > 150: raise TimeoutError('minute replay exceeded batch budget')
        for arm in arms:
            initial = inventory[arm]
            payload = policy_payload(step['states'], step['origin'], initial, arm)
            record = {'captured_at': step['origin'], 'publication_id': bundle['source_publication_id'],
                'mode': 'minute_conditioned_replay', 'payloads': {'mpc': payload},
                'readiness': {'mpc': {'coverage_ready': True, 'reasons': []}}}
            request = prepare_request(record, bundle['configuration'], kind='mpc', optimization_sha256=bundle['optimization_sha256'])
            request['counterfactual'] = {'scope': 'archived_parent_conditioned_policy_replay', 'arm': arm,
                'input_receipts': step['input_receipts'], 'activation': step['activation'],
                'publication_authorized': False}
            request.pop('request_id')
            request['request_id'] = digest(request)
            result = solve(request)
            if request['request_id'] != digest({key: value for key, value in request.items() if key != 'request_id'}):
                raise ValueError('solver mutated frozen request')
            frame = validate_result(request, result)
            plant = request['configuration']['plant_conf']
            command = {'battery_w': float(frame.P_batt.iloc[0]),
                'curtail_w': float(frame.get('P_PV_curtailment', pd.Series(0., index=frame.index)).iloc[0]),
                'export_limit_w': plant['maximum_power_to_grid'] if payload['prod_price_forecast'][0] > 0 else 0.}
            before_soc, before = advance(plant, initial, pending[arm], step['before_activation'])
            inventory[arm], after = advance(plant, before_soc, command, step['after_activation'])
            total = {key: before[key]+after[key] for key in before}
            for key, value in total.items(): totals[arm][key] = totals[arm].get(key, 0.)+value
            duration = (pd.Timestamp(step['end'])-pd.Timestamp(step['origin'])).total_seconds()
            rows.append({'origin': step['origin'], 'activation': step['activation'], 'end': step['end'],
                'arm': arm, 'initial_soc': initial, 'activation_soc': before_soc, 'end_soc': inventory[arm],
                'requested_battery_discharge_w': command['battery_w'], 'terminal_soc': payload['soc_final'],
                'executed_battery_discharge_mean_w': total['executed_discharge_ws']/duration,
                'published_battery_discharge_w': step['published_battery_discharge_w'],
                'request_id': request['request_id'], **total})
            pending[arm] = command
            artifacts.append({'request': request, 'result': result})
    capacity = artifacts[0]['request']['payload']['battery_nominal_energy_capacity']/1000
    for arm in arms:
        totals[arm].update(initial_soc=bundle['initial_soc'], final_soc=inventory[arm], ending_inventory_kwh=inventory[arm]*capacity)
    cash = totals['baseline']['variable_cost_aud']-totals['without_positive_lockin']['variable_cost_aud']
    energy = (inventory['without_positive_lockin']-inventory['baseline'])*capacity
    baseline = [row for row in rows if row['arm'] == 'baseline']
    return {'scope': 'minute_archived_parent_conditioned_not_full_dh_hwc_feedback', 'summary': totals,
        'comparison': {'cashflow_delta_aud': cash, 'ending_inventory_delta_kwh': energy,
            'inventory_value_break_even_aud_per_kwh': -cash/energy if abs(energy) > 1e-8 else None},
        'baseline_diagnostics': {'requested_vs_published_battery_mae_w': float(np.mean([
            abs(row['requested_battery_discharge_w']-row['published_battery_discharge_w']) for row in baseline])),
            'ending_inventory_minus_observed_kwh': (inventory['baseline']-bundle['observed']['final_soc'])*capacity},
        'observed': bundle['observed'], 'steps': rows, 'solves': artifacts, 'publication_authorized': False}


def build_bundle(args, *, apf_revisions=None, captured_states=None):
    read = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = read(args.history/'manifest.json')
    old, parent_bundle = read(args.replay/'report.json'), read(args.replay/'bundle.json')
    if (not manifest['export_complete'] or manifest['history_sha256'] != sha(args.history/'history.json')
            or old['bundle_sha256'] != digest(parent_bundle)):
        raise ValueError('changed evidence bundle')
    history = read(args.history/'history.json')
    if apf_revisions is not None:
        parent_bundle['apf_revisions'] = deepcopy(apf_revisions)
    if captured_states is not None:
        parent_bundle['parents']['baseline'] = deepcopy(captured_states)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if (start.tzinfo is None or end.tzinfo is None or not start < end <= start+pd.Timedelta(minutes=30)
            or start < pd.Timestamp(manifest['start']) or end > pd.Timestamp(manifest['end'])):
        raise ValueError('aware <=30m window inside frozen history required')
    publications = [row for row in history['mpc_battery'] if start <= pd.Timestamp(row['time']) < end]
    if not 1 <= len(publications) <= 30: raise ValueError('require 1–30 publications')
    timing = []
    for publication in publications:
        anchor = asof(history['mpc_anchor'], publication['time'], 15)
        timing.append((pd.Timestamp(anchor['time']), pd.Timestamp(publication['time']), publication['value']))
    if timing[0][0] < start or any(b[0] <= a[1] for a, b in zip(timing, timing[1:])):
        raise ValueError('overlapping/ambiguous decision-publication timings')
    start = timing[0][0]
    quotes = {key: [row for row in values if start-pd.Timedelta(minutes=15) <= pd.Timestamp(row['time']) < end]
        for key, values in parent_bundle['quote_rows'].items()}
    # Financial targets use the full frozen revision archive, including later
    # non-estimated corrections. Decision prices above remain receipt-gated.
    rates = {key: canonical_rates(parent_bundle['quote_rows'][key], start.floor('5min'), end.ceil('5min'))[0]
        for key in ('general', 'feed')}
    steps = []
    for i, (origin, activation, published) in enumerate(timing):
        finish = timing[i+1][0] if i+1 < len(timing) else end
        states, refs = recorded_states(history, parent_bundle['parents']['baseline'], origin, with_parent=True)
        if states['sensor.emhass_current_pv_input_mode']['state'] != 'measured':
            raise ValueError('minute pilot requires measured PV mode at every decision')
        soc = float(states['sensor.sigen_plant_battery_state_of_charge_derived']['state'])/100
        telemetry = {'pv_dc_w': float(states['sensor.sigen_power_pv_gross']['state']),
            'load_site_w': float(states['sensor.sigen_plant_consumed_power']['state']),
            'conversion_loss_w': float(states['sensor.sigen_inverter_conversion_loss']['state'])}
        frozen, apf = snapshot_at(states, soc, telemetry, origin, parent_bundle['apf_revisions'], quotes)
        frozen.pop('sensor.sigen_plant_battery_state_of_charge_derived')
        refs.pop('soc')
        steps.append({'origin': origin.isoformat(), 'activation': activation.isoformat(), 'end': finish.isoformat(),
            'states': frozen, 'input_receipts': {'telemetry_and_parent': refs, 'apf': apf},
            'published_battery_discharge_w': published,
            'before_activation': execution_segments(history, origin, activation, rates),
            'after_activation': execution_segments(history, activation, finish, rates)})
    initial_soc = float(asof(history['soc'], start, 120)['value'])/100
    actuals = execution_segments(history, start, end, rates)
    import_kwh = sum(max(row['observed_grid_import_w'], 0)*row['duration_seconds']/3_600_000 for row in actuals)
    export_kwh = sum(max(-row['observed_grid_import_w'], 0)*row['duration_seconds']/3_600_000 for row in actuals)
    observed = {'initial_soc': initial_soc, 'final_soc': float(asof(history['soc'], end, 120)['value'])/100,
        'grid_import_kwh': import_kwh, 'grid_export_kwh': export_kwh,
        'variable_cost_aud': sum((max(row['observed_grid_import_w'], 0)*row['general_rate']
            -max(-row['observed_grid_import_w'], 0)*row['feed_rate'])*row['duration_seconds']/3_600_000 for row in actuals)}
    for source, column in (('mpc_battery', 'value'), ('mode', 'export_limit_kw')):
        row = asof(history[source], start, 120)
        if not np.isfinite(float(row[column])): raise ValueError('invalid initial command')
    initial_command = {'battery_w': float(asof(history['mpc_battery'], start, 120)['value']),
        'curtail_w': 0., 'export_limit_w': min(parent_bundle['configuration']['plant_conf']['maximum_power_to_grid'],
            max(0, float(asof(history['mode'], start, 120)['export_limit_kw'])*1000))}
    provenance = {'history_sha256': manifest['history_sha256'], 'parent_bundle_sha256': digest(parent_bundle),
        'start': start.isoformat(), 'end': end.isoformat()}
    if apf_revisions is not None or captured_states is not None:
        provenance['source_overrides'] = {'apf': digest(apf_revisions) if apf_revisions is not None else None,
            'capture': digest(captured_states) if captured_states is not None else None}
    return {'steps': steps, 'initial_soc': initial_soc, 'initial_command': initial_command, 'observed': observed,
        'image': parent_bundle['image'], 'configuration': parent_bundle['configuration'],
        'optimization_sha256': parent_bundle['optimization_sha256'], 'source_publication_id': parent_bundle['source_publication_id'],
        'provenance': provenance}


def main():
    os.nice(19)
    if sys.argv[1:2] == ['--worker']:
        from scripts.emhass_solver_worker import solve_request
        print(json.dumps(simulate(json.loads(Path(sys.argv[2]).read_text()), solve_request), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('history', 'replay', 'output'): parser.add_argument('--'+flag, type=Path, required=True)
    for flag in ('start', 'end'): parser.add_argument('--'+flag, required=True)
    parser.add_argument('--solve-archive', type=Path)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    builder_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    bundle = build_bundle(args)
    if args.solve_archive:
        old = json.loads((args.solve_archive/'report.json').read_text())
        if old['bundle_sha256'] != digest(bundle): raise ValueError('saved minute bundle differs')
        saved = {row['request']['request_id']: row for row in old['solves']}
        if len(saved) != len(old['solves']): raise ValueError('duplicate saved solve identity')
        def solve(request):
            row = saved.get(request['request_id'])
            if row is None or row['request'] != request: raise ValueError('saved minute request differs')
            return deepcopy(row['result'])
        result = simulate(bundle, solve)
        result['solve_archive_sha256'] = hashlib.sha256((args.solve_archive/'report.json').read_bytes()).hexdigest()
    else:
        from scripts.replay_energy_solves import run_batch
        files = ['scripts/emhass_solver_worker.py', 'eval/minute_core_replay.py', 'eval/sequential_core_replay.py',
            'eval/audit_control_fidelity.py', 'eval/audit_amber_forecast_archive.py', 'eval/amber_quote_actuals.py',
            'eval/measured_actuals.py', 'energy_pipeline/payloads.py', 'energy_pipeline/solver_replay.py']
        result, hashes = run_batch(bundle, 'eval/minute_core_replay.py', files)
        result['code_sha256'] = hashes
    for artifact in result['solves']: validate_result(artifact['request'], artifact['result'])
    result.update(bundle_sha256=digest(bundle), provenance=bundle['provenance'], builder_sha256=builder_sha, limitations=[
        'archived DH/HWC parents are exogenous conditioning; counterfactual feedback not regenerated',
        'decision and activation clocks use historical helper/plan events, not proven capture/actuation timestamps',
        'own simulated SoC persists; initial prior live command seeded once, initial curtailment assumed zero',
        'PV supply uses delivered raw measurements, not reconstructed available solar',
        'ideal instantaneous DC/AC projection; device ramps and fixed losses beyond efficiencies omitted',
        'forecast settings/configuration held at original replay snapshot; no historical unit stability proof',
        'variable cashflow excludes wear, fixed charges, invoice reconciliation and ending inventory valuation'])
    args.output.mkdir(parents=True)
    (args.output/'bundle.json').write_text(json.dumps(bundle, indent=2, allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('scope', 'summary', 'comparison', 'baseline_diagnostics', 'observed')}))


if __name__ == '__main__': main()
