"""Supported EMS timing policy over each arm's own published MPC trajectory."""
from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
from energy_pipeline.solver_replay import digest

from eval.audit_control_fidelity import asof, parse_curve
from eval.audit_ems_delivery import CURVES, STATE_MAX_AGE, execute_ems
from eval.replay_ems_fallback import guarded_controls, policy, static_guard


def selected(plan, at):
    at = pd.Timestamp(at)
    if not pd.Timestamp(plan['accepted_at']) <= at <= pd.Timestamp(plan['accepted_at'])+pd.Timedelta(seconds=120):
        raise ValueError('no bounded fresh own MPC command coverage')
    points = [row for row in plan['points'] if pd.Timestamp(row['target']) <= at]
    if not points:
        raise ValueError('no causal own MPC target')
    return {key: value for key, value in points[-1].items() if key != 'target'}


def project(frame, accepted_at):
    """Mirror consumed two-decimal power fields, preserving target start labels."""
    columns = {'battery': frame.P_batt, 'grid': frame.P_grid_pos+frame.P_grid_neg,
               'load': frame.P_Load, 'pv': frame.P_PV, 'hybrid': frame.P_hybrid_inverter,
               'curtailment': frame.get('P_PV_curtailment', pd.Series(0., index=frame.index))}
    return {'accepted_at': accepted_at, 'points': [
        {'target': at.isoformat()} | {key: float(np.round(values.iloc[i], 2)) for key, values in columns.items()}
        for i, at in enumerate(frame.index)]}


def seed(history, at):
    curves, clocks = {}, []
    for name, (column, key) in CURVES.items():
        row = asof(history['mpc_'+name], at, STATE_MAX_AGE)
        curves[name] = {pd.Timestamp(point['date']).isoformat(): float(point[key])
                        for point in parse_curve(row[column], key)}
        clocks.append(pd.Timestamp(row['time']))
        if name == 'battery' and pd.Timestamp(at)-pd.Timestamp(row['time']) > pd.Timedelta(seconds=120):
            raise ValueError('no bounded fresh initial MPC battery publication')
    targets = set(curves['battery'])
    if any(set(curve) != targets for curve in curves.values()):
        raise ValueError('mixed initial MPC target grids')
    result = {'accepted_at': max(clocks).isoformat(), 'points': [
        {'target': target} | {key: curve[target] for key, curve in curves.items()}
        for target in sorted(targets)]}
    selected(result, at)
    return result


def configure(bundle, folder, journal, loss, *, pv_export=False):
    if loss not in (0., 140.):
        raise ValueError('only explicit DC loss ablations 0/140W supported')
    manifest = json.loads((folder/'manifest.json').read_text())
    path = folder/'history.json'
    if not manifest['export_complete'] or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['history_sha256']:
        raise ValueError('changed or incomplete EMS history')
    start = bundle.get('resume_checkpoint', {}).get('cursor', bundle['steps'][0]['origin'])
    end = bundle['steps'][-1]['end']
    if not pd.Timestamp(manifest['start']) <= pd.Timestamp(start) < pd.Timestamp(end) <= pd.Timestamp(manifest['end']):
        raise ValueError('execution outside EMS history')
    history = json.loads(path.read_text())
    guard, evidence = static_guard(journal, start)
    if guard > bundle['execution_plant']['battery_minimum_state_of_charge']:
        raise ValueError('unsupported export guard above physical SOC floor')
    bundle['ems_execution'] = {'schema': 1, 'dc_fixed_loss_w': loss, 'minimum_export_soc': guard,
        'guards': {key: history[key] for key in ('effective_feed', 'flexible_export_limit', 'grid_status')}}
    if pv_export:
        if not history.get('effective_general'):
            raise ValueError('PV export requires archived general price')
        weight_evidence = None
        weight_rows = history.get('controller_discharge_weight',[])
        if not weight_rows:
            with sqlite3.connect(journal.resolve().as_uri()+'?mode=ro',uri=True) as connection:
                record = json.loads(connection.execute('SELECT record FROM handoffs ORDER BY rowid DESC LIMIT 1').fetchone()[0])
            entity = 'input_number.emhass_weight_battery_discharge'
            state = record['input_snapshot']['states'][entity]
            clocks = [pd.Timestamp(state[key]) for key in ('last_changed','last_updated','last_reported')]
            value = float(state['state'])
            if not np.isfinite(value) or max(clocks) > pd.Timestamp(start):
                raise ValueError('no capture-confirmed older controller discharge weight')
            weight_rows = [{'time':max(clocks).isoformat(),'value':value}]
            weight_evidence = {'kind':'capture_confirmed_static_fallback_not_independent_historical_availability',
                'entity':entity,'state':state['state'],'capture':record['captured_at'],
                'record_sha256':digest(record),
                'clocks':{key:state[key] for key in ('last_changed','last_updated','last_reported')}}
        bundle['ems_execution']['controller_branches'] = 'pv_export_v2'
        bundle['ems_execution']['local_time_zone'] = bundle['configuration']['retrieve_hass_conf']['time_zone']
        bundle['ems_execution']['guards'].update(effective_general=history['effective_general'],controller_discharge_weight=weight_rows)
    if 'resume_checkpoint' not in bundle:
        bundle['initial_ems_plan'] = seed(history, start)
    bundle['provenance']['ems_execution'] = {'history_sha256': manifest['history_sha256'],
        'static_export_guard': evidence}
    if pv_export: bundle['provenance']['ems_execution']['controller_discharge_weight_fallback'] = weight_evidence
    return bundle


def validate_guard_archive(bundle, previous):
    """Current bounded continuation requires one immutable EMS guard archive."""
    if (bundle['provenance']['ems_execution']['history_sha256'] != previous['provenance']['ems_execution']['history_sha256']
            or bundle['ems_execution']['guards'] != previous['ems_execution']['guards']):
        raise ValueError('continuation EMS guard archive differs; historical overlap audit required')


def activate(plan, soc, settings, plant):
    validate(settings, plant)
    return {'ems_plan': deepcopy(plan), 'ems_command': select_command(plan,plan['accepted_at'],soc,settings),
            'plant': deepcopy(plant)}


def select_command(plan, at, soc, settings):
    selected_plan = selected(plan,at)
    kwargs = {}
    if settings.get('controller_branches') == 'pv_export_v2':
        local = pd.Timestamp(at).tz_convert(settings['local_time_zone'])
        kwargs.update(allow_pv_charge=True,local_hour=local.hour+local.minute/60+local.second/3600)
    if selected_plan['battery'] == 0 and selected_plan['grid'] < 0 and selected_plan['pv'] > 0:
        if settings.get('controller_branches') not in ('pv_export_v1','pv_export_v2'):
            raise ValueError('unsupported PV-only export branch')
        kwargs.update(effective_general_price=float(asof(settings['guards']['effective_general'],at,900)['value']),
                  discharge_weight=float(asof(settings['guards']['controller_discharge_weight'],at,STATE_MAX_AGE)['value']))
    try:
        return policy(selected_plan,soc,settings['minimum_export_soc'],**kwargs)
    except ValueError as exc:
        raise ValueError(f'{exc}; selected_at={at}; plan={selected_plan}') from exc


def fallback(command, at, soc, settings):
    command = deepcopy(command)
    command['ems_command'] = select_command(command['ems_plan'],at,soc,settings)
    return command


def advance(plant, soc, command, segments, settings):
    validate(settings, plant)
    totals = dict.fromkeys(('variable_cost_aud', 'grid_import_kwh', 'grid_export_kwh',
        'battery_throughput_kwh', 'curtailed_pv_kwh', 'clipped_seconds', 'executed_discharge_ws'), 0.)
    for actual in segments:
        # Also bound a held challenger command; a policy hold cannot excuse missing plans.
        selected(command['ems_plan'], actual['end'])
        feed = asof(settings['guards']['effective_feed'], actual['start'], 900)
        if pd.Timestamp(actual['end'])-pd.Timestamp(feed['time']) > pd.Timedelta(seconds=900):
            raise ValueError('effective feed stale before execution segment end')
        controls = guarded_controls(command['ems_command'], settings['guards'], actual['start'],
                                    soc, settings['minimum_export_soc'])
        result = execute_ems(plant, soc, actual | {'controls': controls},
                             dc_fixed_loss_w=settings['dc_fixed_loss_w'])
        for key in ('grid_import_kwh', 'grid_export_kwh', 'battery_throughput_kwh'):
            totals[key] += result[key]
        dt = actual['duration_seconds']
        totals['variable_cost_aud'] += result['grid_import_kwh']*actual['general_rate']-result['grid_export_kwh']*actual['feed_rate']
        totals['curtailed_pv_kwh'] += result['curtailed_pv_w']*dt/3_600_000
        totals['clipped_seconds'] += dt*result['command_clipped']
        totals['executed_discharge_ws'] += result['battery_discharge_w']*dt
        soc = result['end_soc']
    return soc, totals


def validate(settings, plant):
    if settings['schema'] != 1 or settings['dc_fixed_loss_w'] not in (0., 140.):
        raise ValueError('unsupported EMS execution contract')
    guard = settings['minimum_export_soc']
    if not np.isfinite(guard) or not 0 <= guard <= plant['battery_minimum_state_of_charge']:
        raise ValueError('unsupported export guard above physical SOC floor')
