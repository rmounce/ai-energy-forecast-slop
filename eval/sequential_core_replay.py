"""Bounded MPC replay conditional on fixed DH parents and measured delivered PV."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.payloads import Inputs, build_mpc_payload
from energy_pipeline.solver_replay import digest, prepare_request, validate_result
from eval.audit_amber_forecast_archive import parse_revision, select_asof
from eval.amber_quote_actuals import canonical_rates, adjusted_rates


ENTITIES = ('sensor.sigen_plant_rated_energy_capacity', 'sensor.sigen_plant_battery_state_of_health',
    'input_number.emhass_weight_battery_discharge', 'input_number.battery_soc_min_buffer',
    'input_number.battery_soc_min_target', 'input_number.emhass_weight_sell_forecast',
    'input_number.emhass_weight_buy_forecast', 'input_number.emhass_weight_pv_forecast',
    'input_number.sapn_free_exports', 'input_number.dh_last_soc_init',
    'sensor.dh_soc_batt_forecast', 'sensor.dh_p_load_forecast', 'sensor.dh_p_pv_forecast',
    'sensor.emhass_dh_hwc_power_plan_snapshot', 'sensor.hwc_power_plan',
    'sensor.solcast_pv_forecast_power_now')


def execute(plant, soc, battery_w, curtail_w, pv_w, load_w, export_limit_w):
    """Ideal DC battery command; deterministic feasible AC/DC projection, no gap fill."""
    inputs = [soc, battery_w, curtail_w, pv_w, load_w, export_limit_w]
    if not np.isfinite(inputs).all() or min(pv_w, load_w, export_limit_w) < 0:
        raise ValueError('invalid execution input')
    if not plant['inverter_is_hybrid']:
        raise ValueError('require hybrid plant')
    dt = 5/60
    capacity = plant['battery_nominal_energy_capacity']
    minimum, maximum = plant['battery_minimum_state_of_charge'], plant['battery_maximum_state_of_charge']
    if not minimum-1e-8 <= soc <= maximum+1e-8:
        raise ValueError('initial execution SoC outside bounds')
    ec, ed = plant['battery_charge_efficiency'], plant['battery_discharge_efficiency']
    eta_out, eta_in = plant['inverter_efficiency_dc_ac'], plant['inverter_efficiency_ac_dc']
    low = -min(plant['battery_charge_power_max'], max(0, maximum-soc)*capacity/ec/dt)
    high = min(plant['battery_discharge_power_max'], max(0, soc-minimum)*capacity*ed/dt)
    batt = float(np.clip(battery_w, low, high))
    pv = pv_w-float(np.clip(curtail_w, 0, pv_w))
    # Respect AC inverter and grid limits. First curtail solar; reduce excess
    # discharge if curtailment cannot prevent prohibited export.
    max_ac = min(plant['inverter_ac_output_max'], load_w+export_limit_w,
                 load_w+plant['maximum_power_to_grid'])
    max_dc = max_ac/eta_out
    excess = max(0, pv+batt-max_dc)
    removed = min(pv, excess)
    pv -= removed
    batt -= excess-removed
    minimum_ac = max(-plant['inverter_ac_input_max'], load_w-plant['maximum_power_from_grid'])
    minimum_dc = minimum_ac*eta_in if minimum_ac < 0 else minimum_ac/eta_out
    batt = max(batt, minimum_dc-pv)
    if batt < low-1e-6 or batt > high+1e-6:
        raise ValueError('site cannot execute within grid/inverter/SoC bounds')
    dc = pv+batt
    ac = dc*eta_out if dc >= 0 else dc/eta_in
    grid = load_w-ac
    if grid > plant['maximum_power_from_grid']+1e-6 or grid < -export_limit_w-1e-6:
        raise ValueError('execution grid limit violated')
    end_soc = soc-batt*(1/ed if batt >= 0 else ec)*dt/capacity
    if not minimum-1e-8 <= end_soc <= maximum+1e-8:
        raise ValueError('execution SoC limit violated')
    return {'requested_battery_discharge_w': battery_w, 'battery_discharge_w': batt,
        'command_clipped': abs(batt-battery_w) > 1e-6, 'curtailed_pv_w': pv_w-pv,
        'grid_import_kwh': max(grid, 0)*dt/1000, 'grid_export_kwh': max(-grid, 0)*dt/1000,
        'battery_throughput_kwh': abs(batt)*dt/1000, 'end_soc': end_soc,
        'inverter_ac_w': ac}


def snapshot_at(parent, soc, past, origin, revisions, quote_rows):
    """Decisions see previous completed-bin telemetry and past quote receipts only."""
    origin = pd.Timestamp(origin)
    states = deepcopy(parent)
    def state(entity, value): states[entity] = {'state': str(value), 'attributes': {}}
    state('sensor.sigen_plant_battery_state_of_charge_derived', soc*100)
    state('sensor.sigen_power_pv_gross', past['pv_dc_w'])
    state('sensor.sigen_plant_consumed_power', past['load_site_w'])
    state('sensor.sigen_inverter_conversion_loss', past['conversion_loss_w'])
    # Using delivered-PV lower bound, never reconstruct additional available solar.
    state('sensor.emhass_current_pv_input_mode', 'measured')
    provenance = {}
    for leg in ('general', 'feed_in'):
        chosen = select_asof(revisions[leg], origin)
        if chosen is None:
            raise ValueError('missing causal APF revision')
        states['sensor.amber_5min_forecasts_extended_'+leg+'_price'] = {'attributes': {'Forecasts': chosen['rows']}}
        provenance[leg] = {'receipt': chosen['receipt'], 'payload_sha256': chosen['payload_sha256']}
    quotes = {leg: [row for row in rows if pd.Timestamp(row['time']) <= origin]
              for leg, rows in quote_rows.items()}
    start = origin.floor('5min')
    end = start+pd.Timedelta(minutes=5)
    general, _ = canonical_rates(quotes['general'], start, end)
    feed, _ = canonical_rates(quotes['feed'], start, end)
    adjusted, _ = adjusted_rates(quotes['adjusted_feed'], feed, quotes['feed'])
    # An exact-boundary current quote is often still the preceding interval.
    # Use the latest already-issued forecast for this interval, explicitly.
    for leg, entity in [('general', 'sensor.amber_5min_current_general_price'),
                        ('feed_in', 'sensor.amber_adjusted_confirmed_feed_in_price')]:
        table = general if leg == 'general' else adjusted
        if start in table.index:
            value = float(table.loc[start, 'rate'])
            provenance[leg]['current_source'] = 'non_estimated_received_quote'
        else:
            raw = states['sensor.amber_5min_forecasts_extended_'+leg+'_price']['attributes']['Forecasts']
            match = [row for row in raw if pd.Timestamp(row['end_time'])-pd.Timedelta(minutes=row['duration']) == start]
            if len(match) != 1 or match[0]['duration'] != 5:
                raise ValueError('no causal current-interval quote or forecast')
            row = match[0]
            sign = 1 if leg == 'general' else -1
            value = sign*float(row['per_kwh'])
            if leg == 'feed_in':
                from energy_pipeline.payloads import export_allowance
                value += export_allowance(Inputs(states, origin.to_pydatetime()), origin.isoformat())
            provenance[leg]['current_source'] = 'already_issued_per_kwh_forecast'
        state(entity, value)
        # Recorded start_time is canonical start +1s. At an exact replay
        # boundary the template would otherwise append the current interval
        # AGAIN after the separate current-price slot, shifting the whole curve.
        raw = states['sensor.amber_5min_forecasts_extended_'+leg+'_price']['attributes']['Forecasts']
        future = [row for row in raw if pd.Timestamp(row['end_time'])-pd.Timedelta(minutes=row['duration']) > origin]
        if any(row['duration'] != 5 for row in future):
            raise ValueError('MPC replay requires five-minute APF rows')
        states['sensor.amber_5min_forecasts_extended_'+leg+'_price']['attributes']['Forecasts'] = future
        provenance[leg]['future_interval_gate'] = 'canonical_start_strictly_after_origin'
    return states, provenance


def simulate(bundle, solve):
    started = time.monotonic()
    rows, requests = [], []
    state = {arm: bundle['initial_soc'] for arm in bundle['parents']}
    for step, actual in enumerate(bundle['actuals']):
        if time.monotonic()-started > 150:
            raise TimeoutError('batch replay exceeded bounded budget')
        origin = pd.Timestamp(actual['time'])
        past = bundle['past'] if step == 0 else bundle['actuals'][step-1]
        for arm, parent in bundle['parents'].items():
            states, provenance = snapshot_at(parent, state[arm], past, origin,
                bundle['apf_revisions'], bundle['quote_rows'])
            payload = build_mpc_payload(Inputs(states, origin.to_pydatetime()))
            record = {'captured_at': origin.isoformat(), 'publication_id': bundle['source_publication_id'],
                'mode': 'conditioned_sequential_mpc', 'payloads': {'mpc': payload},
                'readiness': {'mpc': {'coverage_ready': True, 'reasons': []}}}
            request = prepare_request(record, bundle['configuration'], kind='mpc',
                                      optimization_sha256=bundle['optimization_sha256'])
            request['counterfactual'] = {'scope': 'fixed_dh_parent_conditioned_sequential_replay',
                'arm': arm, 'dh_parent_request_id': bundle['parent_request_ids'][arm],
                'apf': provenance, 'telemetry_source': 'previous_completed_5m_bin',
                'publication_authorized': False}
            request.pop('request_id')
            request['request_id'] = digest(request)
            result = solve(request)
            if request['request_id'] != digest({key: value for key, value in request.items() if key != 'request_id'}):
                raise ValueError('solver mutated frozen request')
            frame = validate_result(request, result)
            plant = request['configuration']['plant_conf']
            # Same current-price guard for both arms; ideal instantaneous response.
            export_limit = plant['maximum_power_to_grid'] if payload['prod_price_forecast'][0] > 0 else 0.
            execution = execute(plant, state[arm], float(frame.P_batt.iloc[0]),
                float(frame.get('P_PV_curtailment', pd.Series(0., index=frame.index)).iloc[0]),
                actual['pv_dc_w'], actual['load_site_w'], export_limit)
            execution['variable_cost_aud'] = (execution['grid_import_kwh']*actual['general_rate']
                                              -execution['grid_export_kwh']*actual['feed_rate'])
            rows.append({'time': actual['time'], 'arm': arm, 'initial_soc': state[arm],
                'solver_seconds': result['solve_seconds'], 'request_id': request['request_id'],
                'terminal_soc': payload['soc_final'], **execution})
            state[arm] = execution['end_soc']
            requests.append({'request': request, 'result': result})
    # Runtime capacity overrides source configuration (which may have a stale capacity).
    capacity_kwh = requests[0]['request']['payload']['battery_nominal_energy_capacity']/1000
    summary = {}
    for arm in state:
        selected = [row for row in rows if row['arm'] == arm]
        summary[arm] = {field: sum(row[field] for row in selected) for field in (
            'variable_cost_aud', 'grid_import_kwh', 'grid_export_kwh', 'battery_throughput_kwh')}
        summary[arm].update(initial_soc=bundle['initial_soc'], final_soc=state[arm],
            ending_inventory_kwh=state[arm]*capacity_kwh,
            clipped_commands=sum(row['command_clipped'] for row in selected),
            curtailed_pv_kwh=sum(row['curtailed_pv_w'] for row in selected)*5/60000)
    cash = summary['baseline']['variable_cost_aud']-summary['calibrated_load']['variable_cost_aud']
    energy = summary['calibrated_load']['ending_inventory_kwh']-summary['baseline']['ending_inventory_kwh']
    first = [row for row in rows if row['arm'] == 'baseline']
    second = [row for row in rows if row['arm'] == 'calibrated_load']
    differences = [b['battery_discharge_w']-a['battery_discharge_w'] for a, b in zip(first, second)]
    return {'scope': 'conditioned_sequential_mpc_not_deployable_savings', 'summary': summary,
        'comparison': {'cashflow_delta_aud': cash, 'ending_inventory_delta_kwh': energy,
            'changed_executed_battery_steps': sum(abs(value) > 1 for value in differences),
            'max_executed_battery_difference_w': max(abs(value) for value in differences),
            'inventory_value_break_even_aud_per_kwh': -cash/energy if abs(energy) > 1e-8 else None},
        'steps': rows, 'solves': requests, 'publication_authorized': False}


def build_bundle(args):
    from energy_pipeline.solver_chain import build_chained_handoff
    from eval.compare_load_solver_sensitivity import calibrated_handoff
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    actual_manifest = json.loads((args.dataset/'manifest.json').read_text())
    actual_path = args.dataset/'actuals.parquet'
    if not actual_manifest['export_complete'] or sha(actual_path) != actual_manifest['parquet_sha256']:
        raise ValueError('measured dataset incomplete or changed')
    rates = json.loads((args.quotes/'manifest.json').read_text())
    if not rates['export_complete'] or rates['measured_actuals_sha256'] != sha(actual_path):
        raise ValueError('price archive does not match measured targets')
    quote_rows = {}
    for leg in ('general', 'feed', 'adjusted_feed'):
        path = args.quotes/(leg+'_revisions.json')
        if sha(path) != rates['files'][path.name]: raise ValueError('changed quote archive')
        quote_rows[leg] = json.loads(path.read_text())
    apf_manifest = json.loads((args.apf/'manifest.json').read_text())
    path = args.apf/'revisions.json'
    if sha(path) != apf_manifest['raw_revisions_sha256']: raise ValueError('changed APF archive')
    apf = json.loads(path.read_text())
    revisions = {leg: [parse_revision(row, leg) for row in data['first']+data['last']]
                 for leg, data in apf.items()}
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro', uri=True) as db:
        record = json.loads(db.execute('SELECT record FROM handoffs ORDER BY rowid DESC LIMIT 1').fetchone()[0])
    calibration = json.loads((args.calibration/'report.json').read_text())
    path = args.calibration/'challenger.parquet'
    if sha(path) != calibration['challenger_sha256']: raise ValueError('changed calibration archive')
    synthetic, _ = calibrated_handoff(record, pd.read_parquet(path), calibration['settings'])
    parents, parent_ids, image, source = {}, {}, None, None
    for arm, handoff, filename in [('baseline', record, 'baseline.json'),
            ('calibrated_load', synthetic, 'calibrated_load.json')]:
        artifact = json.loads((args.load_comparison/filename).read_text())
        artifact['mode'] = 'historical_solver_replay'
        chained = build_chained_handoff(handoff, artifact)
        states = chained['input_snapshot']['states']
        parents[arm] = {key: deepcopy(states[key]) for key in ENTITIES if key in states}
        parent_ids[arm] = artifact['request']['request_id']
        if image is not None and image != artifact['image']: raise ValueError('parent images differ')
        image, source = artifact['image'], artifact['request']['optimization_sha256']
    start = pd.Timestamp(args.start)
    if start.tzinfo is None or start != start.floor('5min') or start < pd.Timestamp(record['captured_at']):
        raise ValueError('start must be aware 5m boundary after historical capture')
    start = start.tz_convert('UTC')
    targets = pd.date_range(start, periods=args.steps, freq='5min')
    frame = pd.read_parquet(actual_path)
    actual = frame.reindex(targets.union(pd.DatetimeIndex([start-pd.Timedelta(minutes=5)])))
    required = ['pv_dc_w', 'load_site_w', 'conversion_loss_w']
    if actual[required].isna().any().any(): raise ValueError('incomplete observed execution/lagged telemetry window')
    general, _ = canonical_rates(quote_rows['general'], start, targets[-1]+pd.Timedelta(minutes=5))
    feed, _ = canonical_rates(quote_rows['feed'], start, targets[-1]+pd.Timedelta(minutes=5))
    if general.reindex(targets).rate.isna().any() or feed.reindex(targets).rate.isna().any():
        raise ValueError('incomplete observed scoring rate window')
    actual['general_rate'], actual['feed_rate'] = general.rate, feed.rate
    actual['time'] = actual.index.map(lambda stamp: stamp.isoformat())
    past = actual.loc[start-pd.Timedelta(minutes=5)]
    initial_soc = float(past.soc_pct_end)/100
    if not np.isfinite(initial_soc): raise ValueError('missing initial inventory observation')
    observations = actual.loc[targets, required+['time', 'general_rate', 'feed_rate']].to_dict('records')
    configuration = json.loads(args.config.read_text())
    for filename in ('baseline.json', 'calibrated_load.json'):
        artifact = json.loads((args.load_comparison/filename).read_text())
        if artifact['request']['config_revision'] != digest(configuration): raise ValueError('parent configuration mismatch')
    return {'parents': parents, 'parent_request_ids': parent_ids, 'image': image,
        'configuration': configuration, 'optimization_sha256': source,
        'source_publication_id': record['publication_id'], 'initial_soc': initial_soc,
        'actuals': observations, 'past': {key: float(past[key]) for key in required},
        'apf_revisions': revisions, 'quote_rows': quote_rows,
        'provenance': {'actuals_sha256': sha(actual_path),
            'quote_manifest_sha256': sha(args.quotes/'manifest.json'),
            'apf_manifest_sha256': sha(args.apf/'manifest.json'),
            'calibration_manifest_sha256': sha(args.calibration/'report.json')}}


def decision_bundle_identity(bundle):
    """Ignore only scoring-rate values and their archive manifest when rescoring."""
    content = deepcopy(bundle)
    for row in content['actuals']:
        row.pop('general_rate', None)
        row.pop('feed_rate', None)
    content.get('provenance', {}).pop('quote_manifest_sha256', None)
    return digest(content)


def main():
    os.nice(19)
    if sys.argv[1:2] == ['--worker']:
        from scripts.emhass_solver_worker import solve_request
        bundle = json.loads(Path(sys.argv[2]).read_text())
        print(json.dumps(simulate(bundle, solve_request), allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('journal', 'config', 'load-comparison', 'calibration', 'dataset', 'quotes', 'apf', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--solve-archive', type=Path,
                        help='replay exactly matching saved solve requests offline')
    parser.add_argument('--rescore-rates', action='store_true',
                        help='allow updated scoring rates; all decisions/physical inputs must still match archive')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.steps <= 12:
        parser.error('new output directory and 1–12 steps required')
    if args.rescore_rates and not args.solve_archive: parser.error('--rescore-rates requires --solve-archive')
    bundle = build_bundle(args)
    if args.solve_archive:
        original_path = args.solve_archive/'report.json'
        original = json.loads(original_path.read_text())
        if original['bundle_sha256'] != digest(bundle):
            archived_bundle = json.loads((args.solve_archive/'bundle.json').read_text())
            if (not args.rescore_rates or digest(archived_bundle) != original['bundle_sha256'] or
                    decision_bundle_identity(archived_bundle) != decision_bundle_identity(bundle)):
                raise ValueError('solve archive belongs to another decision/physical input bundle')
        saved = {artifact['request']['request_id']: artifact for artifact in original['solves']}
        if len(saved) != len(original['solves']): raise ValueError('duplicate archived solve identity')
        def archived_solve(request):
            artifact = saved[request['request_id']]
            if artifact['request'] != request: raise ValueError('archived request changed')
            return deepcopy(artifact['result'])
        result = simulate(bundle, archived_solve)
        result['solve_archive_sha256'] = hashlib.sha256(original_path.read_bytes()).hexdigest()
        result['rescored_rates'] = args.rescore_rates
        result['bundle_sha256'] = digest(bundle)
        result['provenance'] = bundle['provenance']
        result['limitations'] = original['limitations']
        result['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['original_worker_code_sha256'] = original.get('code_sha256')
        # Old orchestration versions hashed host files after execution; those
        # hashes need not describe the code copied into the running container.
        result['limitations'].append('original host-after-run code hashes may differ from staged worker; solve source pinned and requests revalidated')
        args.output.mkdir(parents=True)
        (args.output/'bundle.json').write_text(json.dumps(bundle, indent=2, allow_nan=False)+'\n')
        (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps({key: result[key] for key in ('scope', 'summary', 'comparison')}))
        return
    from scripts.replay_energy_solves import container_command
    files = ['scripts/emhass_solver_worker.py', 'eval/sequential_core_replay.py',
        'eval/audit_amber_forecast_archive.py', 'eval/amber_quote_actuals.py',
        'energy_pipeline/payloads.py', 'energy_pipeline/solver_replay.py']
    with tempfile.TemporaryDirectory(prefix='energy-sequential-') as folder:
        staging = Path(folder)
        for filename in files:
            path = staging/filename
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/filename, path)
            path.chmod(0o444)
        staged_hashes = {filename: hashlib.sha256((staging/filename).read_bytes()).hexdigest() for filename in files}
        (staging/'worker.py').write_text('# unused entrypoint placeholder\n')
        (staging/'request.json').write_text(json.dumps(bundle, allow_nan=False))
        for path in staging.iterdir():
            if path.is_dir(): path.chmod(0o755)
            else: path.chmod(0o444)
        command = container_command(bundle['image'], folder)
        for code_directory in ('scripts', 'eval', 'energy_pipeline'):
            command[command.index('--env'):command.index('--env')] = ['--mount', f'type=bind,src={folder}/{code_directory},dst=/work/{code_directory},readonly']
        command[-2:] = ['/work/eval/sequential_core_replay.py', '--worker', '/work/request.json']
        name = 'energy-sequential-'+uuid.uuid4().hex[:16]
        command[2:2] = ['--name', name]
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '1', name], capture_output=True, timeout=15)
            raise
        if completed.returncode: raise RuntimeError(completed.stderr[-4000:])
        result = json.loads(completed.stdout)
    # Revalidate every solve outside the worker, and retain all inputs/results.
    for artifact in result['solves']: validate_result(artifact['request'], artifact['result'])
    result.update({'bundle_sha256': digest(bundle), 'provenance': bundle['provenance'],
        'code_sha256': staged_hashes,
        'limitations': ['one-hour five-minute MPC only; DH parents/HWC/risk weights held fixed',
            'measured delivered PV treated as common available supply lower bound, not counterfactual solar',
            'previous complete 5m telemetry used with ideal instantaneous control/limits; device ramps omitted',
            'APF Influx event time assumed availability; legs independently received',
            'current interval may use a previously issued forecast when no causal current quote exists',
            'canonical interval gate prevents duplicate current APF slot at exact replay boundary',
            'new DH solves, offset updates, HWC re-scheduling and long-run feedback not simulated',
            'variable cashflow excludes wear, fixed charges, invoice reconciliation and final inventory value']})
    args.output.mkdir(parents=True)
    (args.output/'bundle.json').write_text(json.dumps(bundle, indent=2, allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('scope', 'summary', 'comparison')}))


if __name__ == '__main__': main()
