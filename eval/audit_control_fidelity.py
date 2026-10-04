"""Offline timing/input diagnostics for a frozen sequential replay; no new solves."""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.payloads import Inputs, build_mpc_payload
from energy_pipeline.solver_replay import digest, validate_result
from eval.measured_actuals import integrate, endpoint_samples
from eval.sequential_core_replay import snapshot_at


def parse_curve(raw, value_key):
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise ValueError('missing or oversized published curve')
    try:
        curve = json.loads(raw)
    except json.JSONDecodeError:
        curve = ast.literal_eval(raw)
    if not isinstance(curve, list) or not 1 <= len(curve) <= 1000:
        raise ValueError('invalid published curve size')
    dates = [pd.Timestamp(row['date']) for row in curve]
    if any(date.tzinfo is None or pd.isna(date) for date in dates):
        raise ValueError('invalid published curve dates')
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError('unordered published curve')
    if not all(np.isfinite(float(row[value_key])) for row in curve):
        raise ValueError('nonfinite published curve')
    return curve


def asof(rows, at, max_age_seconds):
    """Require an existing finite-age record; never use a future observation."""
    at = pd.Timestamp(at)
    available = [row for row in rows if pd.Timestamp(row['time']) <= at]
    if not available:
        raise ValueError('no causal control observation')
    latest = max(available, key=lambda row: pd.Timestamp(row['time']))
    if (at-pd.Timestamp(latest['time'])).total_seconds() > max_age_seconds:
        raise ValueError('stale control observation')
    return latest


def numeric_series(rows):
    return pd.Series([row.get('value') for row in rows],
        index=pd.to_datetime([row['time'] for row in rows], utc=True), dtype=float)


def recorded_states(history, parent, at, *, with_parent):
    states = deepcopy(parent)
    refs = {}
    sources = {
        'soc': ('sensor.sigen_plant_battery_state_of_charge_derived', 120),
        'pv': ('sensor.sigen_power_pv_gross', 120),
        'load': ('sensor.sigen_plant_consumed_power', 120),
        'loss': ('sensor.sigen_inverter_conversion_loss', 120),
    }
    for key, (entity, age) in sources.items():
        row = asof(history[key], at, age)
        if not np.isfinite(float(row['value'])):
            raise ValueError('nonfinite control telemetry: '+key)
        states[entity] = {'state': str(row['value']), 'attributes': {}}
        refs[key] = row['time']
    row = asof(history['mode'], at, 120)
    states['sensor.emhass_current_pv_input_mode'] = {'state': row['state'], 'attributes': {}}
    refs['mode'] = row['time']
    if with_parent:
        for key, entity, field, attribute, value_key in (
            ('dh_soc', 'sensor.dh_soc_batt_forecast', 'battery_scheduled_soc_str', 'battery_scheduled_soc', 'dh_soc_batt_forecast'),
            ('dh_load', 'sensor.dh_p_load_forecast', 'forecasts_str', 'forecasts', 'dh_p_load_forecast'),
            ('dh_pv', 'sensor.dh_p_pv_forecast', 'forecasts_str', 'forecasts', 'dh_p_pv_forecast'),
            ('hwc', 'sensor.emhass_dh_hwc_power_plan_snapshot', 'deferrables_schedule_json_str', 'deferrables_schedule_json', 'hwc_power_plan')):
            row = asof(history[key], at, 3600)
            states[entity] = {'attributes': {attribute: parse_curve(row[field], value_key)}}
            refs[key] = row['time']
        row = asof(history['dh_anchor'], at, 3600)
        if not np.isfinite(float(row['value'])):
            raise ValueError('nonfinite DH anchor')
        states['input_number.dh_last_soc_init'] = {'state': str(row['value'])}
        refs['dh_anchor'] = row['time']
        # A helper can move before its new DH plan is published. Do not pretend
        # that this non-atomic transient is a coherent accepted parent.
        if pd.Timestamp(refs['dh_anchor']) > pd.Timestamp(refs['dh_soc']):
            raise ValueError('DH anchor advanced before published DH parent')
    return states, refs


def error_summary(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('empty/nonfinite diagnostic comparison')
    return {'mean_signed': float(values.mean()), 'mae': float(abs(values).mean()),
        'absolute_p95': float(np.quantile(abs(values), .95)), 'count': len(values)}


def reconstruct_inputs(history, bundle, at, quote_rows):
    telemetry, refs = recorded_states(history, bundle['parents']['baseline'], at, with_parent=True)
    soc = float(telemetry['sensor.sigen_plant_battery_state_of_charge_derived']['state'])/100
    past = {'pv_dc_w': float(telemetry['sensor.sigen_power_pv_gross']['state']),
        'load_site_w': float(telemetry['sensor.sigen_plant_consumed_power']['state']),
        'conversion_loss_w': float(telemetry['sensor.sigen_inverter_conversion_loss']['state'])}
    states, _ = snapshot_at(bundle['parents']['baseline'], soc, past, at,
        bundle['apf_revisions'], quote_rows)
    fixed = build_mpc_payload(Inputs(states, at.to_pydatetime()))
    states.update(telemetry)
    return fixed, build_mpc_payload(Inputs(states, at.to_pydatetime())), refs


def audit(history, bundle, replay, actuals):
    origins = pd.DatetimeIndex([row['time'] for row in bundle['actuals']])
    start, end = origins[0], origins[-1]+pd.Timedelta(minutes=5)
    frame = actuals.reindex(origins)
    required = ['battery_charge_w', 'planned_battery_discharge_w', 'soc_pct_end']
    if frame[required].isna().any().any():
        raise ValueError('incomplete matched measured diagnostics')
    published = [row for row in history['mpc_battery'] if start <= pd.Timestamp(row['time']) < end]
    dates = pd.DatetimeIndex([row['time'] for row in published])
    cadence = np.diff(dates.as_unit('ns').asi8)/1e9
    if not len(cadence): raise ValueError('insufficient plan publication evidence')
    held = endpoint_samples(numeric_series(history['mpc_battery']), origins)
    if not np.isfinite(held).all(): raise ValueError('missing boundary-held plan')
    raw_mean = integrate(numeric_series(history['mpc_battery']), start, end)['mean']
    if raw_mean.isna().any(): raise ValueError('incomplete published plan support')
    if not np.allclose(raw_mean, frame.planned_battery_discharge_w, atol=1e-6, rtol=0):
        raise ValueError('frozen plan integration does not reproduce measured export')
    baseline = [row for row in replay['steps'] if row['arm'] == 'baseline']
    if [row['time'] for row in baseline] != list(origins.map(lambda stamp: stamp.isoformat())):
        raise ValueError('replay window differs from diagnostic targets')
    for artifact in replay['solves']: validate_result(artifact['request'], artifact['result'])
    requested = np.array([row['requested_battery_discharge_w'] for row in baseline])
    # Only near-window current quotes can affect these input reconstructions.
    # Keep all receipts in that window (including identified transitions), then
    # let snapshot_at enforce the per-origin causal gate.
    quote_rows = {key: [row for row in values if start-pd.Timedelta(minutes=15) <= pd.Timestamp(row['time']) < end]
        for key, values in bundle['quote_rows'].items()}
    rows, excluded = [], []
    for publication in published:
        try:
            anchor = asof(history['mpc_anchor'], publication['time'], 15)
            at = pd.Timestamp(anchor['time'])
            # Helper publication is a timestamp proxy, not atomic input capture.
            fixed, refreshed, refs = reconstruct_inputs(history, bundle, at, quote_rows)
            soc_publication = asof(history['mpc_soc'], pd.Timestamp(publication['time'])+pd.Timedelta(seconds=1), 2)
            if abs((pd.Timestamp(soc_publication['time'])-pd.Timestamp(publication['time'])).total_seconds()) > 1:
                raise ValueError('MPC SoC and power publication pair missing')
            power_curve = parse_curve(publication['battery_scheduled_power_str'], 'mpc_p_batt_forecast')
            soc_curve = parse_curve(soc_publication['battery_scheduled_soc_str'], 'mpc_soc_batt_forecast')
            if power_curve[0]['date'] != soc_curve[0]['date']:
                raise ValueError('MPC plan publication dates differ')
            if float(power_curve[0]['mpc_p_batt_forecast']) != publication['value']:
                raise ValueError('MPC published state/curve disagree')
            initial = float(anchor['value'])
            first_power = float(power_curve[0]['mpc_p_batt_forecast'])
            efficiency = bundle['configuration']['plant_conf']
            cap = replay['solves'][0]['request']['payload']['battery_nominal_energy_capacity']
            expected = initial-first_power*(1/efficiency['battery_discharge_efficiency'] if first_power >= 0
                else efficiency['battery_charge_efficiency'])*(5/60)/cap*100
            row = {'publication': publication['time'], 'input_time_proxy': at.isoformat(),
                'recorded_soc_init_pct': initial, 'published_first_battery_discharge_w': first_power,
                'refreshed_soc_init_pct': refreshed['soc_init']*100,
                'fixed_parent_terminal_pct': fixed['soc_final']*100,
                'refreshed_parent_terminal_pct': refreshed['soc_final']*100,
                'published_terminal_pct': float(soc_curve[-1]['mpc_soc_batt_forecast']),
                'first_endpoint_error_pct': float(soc_curve[0]['mpc_soc_batt_forecast'])-expected,
                'fixed_load_14h_kwh': sum(fixed['load_power_forecast'])*5/60000,
                'refreshed_load_14h_kwh': sum(refreshed['load_power_forecast'])*5/60000,
                'fixed_pv_14h_kwh': sum(fixed['pv_power_forecast'])*5/60000,
                'refreshed_pv_14h_kwh': sum(refreshed['pv_power_forecast'])*5/60000,
                'input_receipts': refs}
            rows.append(row)
        except (ValueError, KeyError, TypeError) as exc:
            excluded.append({'publication': publication['time'], 'reason': str(exc)})
    dt = 5/60/1000
    efficiency = bundle['configuration']['plant_conf']
    def energy_change(discharge):
        discharge = np.asarray(discharge)
        return float(np.where(discharge >= 0, -discharge/efficiency['battery_discharge_efficiency'],
            -discharge*efficiency['battery_charge_efficiency']).sum()*dt)
    capacity = replay['solves'][0]['request']['payload']['battery_nominal_energy_capacity']/1000
    measured_inventory = (float(frame.soc_pct_end.iloc[-1])/100-bundle['initial_soc'])*capacity
    effects = {
        'live_planned_minus_measured_discharge_w': error_summary(raw_mean+frame.battery_charge_w),
        'five_minute_boundary_hold_minus_live_plan_w': error_summary(held-raw_mean),
        'replay_requested_minus_boundary_live_plan_w': error_summary(requested-held),
        'replay_requested_minus_live_plan_w': error_summary(requested-raw_mean),
        'live_plan_net_inventory_change_kwh': energy_change(raw_mean),
        'boundary_held_plan_net_inventory_change_kwh': energy_change(held),
        'measured_power_net_inventory_change_kwh': energy_change(-frame.battery_charge_w),
        'measured_soc_inventory_change_kwh': measured_inventory,
        'replay_executed_inventory_change_kwh': replay['summary']['baseline']['ending_inventory_kwh']-bundle['initial_soc']*capacity,
    }
    if rows:
        effects.update({key: error_summary([row[a]-row[b] for row in rows]) for key, a, b in (
            ('refreshed_init_minus_recorded_init_pct', 'refreshed_soc_init_pct', 'recorded_soc_init_pct'),
            ('refreshed_terminal_minus_published_pct', 'refreshed_parent_terminal_pct', 'published_terminal_pct'),
            ('fixed_terminal_minus_published_pct', 'fixed_parent_terminal_pct', 'published_terminal_pct'))})
        effects['first_endpoint_recurrence_error_pct'] = error_summary([row['first_endpoint_error_pct'] for row in rows])
        for name in ('load', 'pv'):
            effects['refreshed_minus_fixed_'+name+'_14h_kwh'] = error_summary([
                row['refreshed_'+name+'_14h_kwh']-row['fixed_'+name+'_14h_kwh'] for row in rows])
    targets = [{'time': at.isoformat(), 'boundary_live_plan_w': float(held[i]),
        'within_bin_live_plan_mean_w': float(raw_mean.iloc[i]), 'replay_requested_w': float(requested[i]),
        'measured_battery_discharge_w': -float(frame.battery_charge_w.iloc[i])} for i, at in enumerate(origins)]
    return {'scope': 'retrospective_control_fidelity_not_savings_or_atomic_input_reconstruction',
        'plan_publications': len(published), 'distinct_mpc_power_curves': len({digest(parse_curve(row['battery_scheduled_power_str'], 'mpc_p_batt_forecast')) for row in published}),
        'publication_spacing_seconds': {'median': float(np.median(cadence)), 'min': float(cadence.min()), 'max': float(cadence.max())},
        'parent_publications': {key: sum(start <= pd.Timestamp(row['time']) < end for row in history[key])
            for key in ('dh_soc', 'dh_load', 'dh_pv', 'hwc')},
        'effects': effects, 'targets': targets, 'input_diagnostics': rows, 'excluded_inputs': excluded,
        'reconstructed_input_count': len(rows), 'excluded_input_count': len(excluded),
        'publication_authorized': False}


def archived_checkpoint_runner(archive):
    saved = {row['artifact']['request']['request_id']: row['artifact'] for row in archive['core_checkpoints']}
    if len(saved) != len(archive['core_checkpoints']): raise ValueError('duplicate archived checkpoint')
    def run(request, image):
        artifact = saved.get(request['request_id'])
        if artifact is None or artifact['request'] != request or artifact['image'] != image:
            raise ValueError('archived checkpoint request or image differs')
        return deepcopy(artifact)
    return run


def core_checkpoints(history, bundle, report, runner=None):
    """Six pinned core solves: three matched-clock fixed/refreshed input pairs."""
    from scripts.replay_energy_solves import run_request
    from energy_pipeline.solver_replay import prepare_request
    rows = report['input_diagnostics']
    if len(rows) < 3: raise ValueError('insufficient reconstructed input checkpoints')
    # Fixed deterministic positions; no selection based on resulting performance.
    selected = [rows[index] for index in (0, len(rows)//3, 2*len(rows)//3)]
    runner = runner or run_request
    checks = []
    for row in selected:
        at = pd.Timestamp(row['input_time_proxy'])
        start = at.floor('5min')
        quotes = {key: [value for value in values if start-pd.Timedelta(minutes=15) <= pd.Timestamp(value['time']) <= at]
            for key, values in bundle['quote_rows'].items()}
        fixed, refreshed, _ = reconstruct_inputs(history, bundle, at, quotes)
        for arm, payload in [('fixed_parent_current_telemetry', fixed), ('recorded_parent_current_telemetry', refreshed)]:
            record = {'captured_at': at.isoformat(), 'publication_id': bundle['source_publication_id'],
                'mode': 'reconstructed_live_control_diagnostic', 'payloads': {'mpc': payload},
                'readiness': {'mpc': {'coverage_ready': True, 'reasons': []}}}
            request = prepare_request(record, bundle['configuration'], kind='mpc',
                optimization_sha256=bundle['optimization_sha256'])
            request['counterfactual'] = {'scope': 'observed_soc_regrounded_checkpoint_not_sequential_savings',
                'arm': arm, 'history_sha256': report['provenance']['history_sha256'],
                'publication_authorized': False}
            request.pop('request_id')
            request['request_id'] = digest(request)
            artifact = runner(request, bundle['image'])
            frame = validate_result(request, artifact['result'])
            power = float(frame.P_batt.iloc[0])
            checks.append({'publication': row['publication'], 'input_time_proxy': at.isoformat(), 'arm': arm,
                'published_first_battery_discharge_w': row['published_first_battery_discharge_w'],
                'reconstructed_first_battery_discharge_w': power,
                'difference_from_published_w': power-row['published_first_battery_discharge_w'],
                'artifact': artifact})
    return checks


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('history', 'replay', 'dataset', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--core-checkpoints', type=int, choices=(0, 3), default=0,
        help='optionally run six isolated core solves at three deterministic diagnostic origins')
    parser.add_argument('--checkpoint-archive', type=Path,
        help='reuse only exact matching saved checkpoint requests/results; requires --core-checkpoints 3')
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    if args.checkpoint_archive and not args.core_checkpoints:
        parser.error('--checkpoint-archive requires --core-checkpoints 3')
    read = lambda path: json.loads(path.read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hm, am = read(args.history/'manifest.json'), read(args.dataset/'manifest.json')
    replay, bundle = read(args.replay/'report.json'), read(args.replay/'bundle.json')
    if (not hm['export_complete'] or hm['history_sha256'] != sha(args.history/'history.json') or
            not am['export_complete'] or am['parquet_sha256'] != sha(args.dataset/'actuals.parquet') or
            replay['bundle_sha256'] != digest(bundle) or bundle['provenance']['actuals_sha256'] != am['parquet_sha256']):
        raise ValueError('changed or incompatible evidence archive')
    auditor_sha = sha(Path(__file__))
    report = audit(read(args.history/'history.json'), bundle, replay, pd.read_parquet(args.dataset/'actuals.parquet'))
    report['provenance'] = {'history_sha256': hm['history_sha256'], 'actuals_sha256': am['parquet_sha256'],
        'replay_sha256': sha(args.replay/'report.json'), 'auditor_sha256': auditor_sha}
    report['limitations'] = hm['limitations']+[
        'settings/configuration retained from replay; original capture-time equivalence unproven',
        'within-bin mean and later paired plan publications are retrospective diagnostics only',
        'five-minute averaged energy recurrence does not resolve sign changes or physical clipping within bins',
        'input time uses recorded MPC anchor publication; exact request snapshot and script timing unobserved',
        'live SoC/parent reconstruction is not a counterfactual economic comparison']
    if args.core_checkpoints:
        runner = archived_checkpoint_runner(read(args.checkpoint_archive)) if args.checkpoint_archive else None
        report['core_checkpoints'] = core_checkpoints(read(args.history/'history.json'), bundle, report, runner)
        if args.checkpoint_archive: report['checkpoint_archive_sha256'] = sha(args.checkpoint_archive)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: report[key] for key in ('plan_publications', 'parent_publications',
        'reconstructed_input_count', 'excluded_input_count', 'effects')}))


if __name__ == '__main__':
    main()
