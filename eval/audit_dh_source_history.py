"""Admit causal historical DH source arrays and audit clock/lineage gaps."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.payloads import Inputs, build_dh_payload, boundary
from energy_pipeline.solver_replay import digest
from eval.archived_forecasts import parse_rows
from eval.audit_control_fidelity import asof, recorded_states

SETTINGS = {
    'buy_weight': 'input_number.emhass_weight_buy_forecast', 'sell_weight': 'input_number.emhass_weight_sell_forecast',
    'pv_weight': 'input_number.emhass_weight_pv_forecast', 'discharge_weight': 'input_number.emhass_weight_battery_discharge',
    'minimum_soc': 'input_number.battery_soc_min_target', 'soc_buffer': 'input_number.battery_soc_min_buffer',
    'target_offset': 'input_number.emhass_target_soc_offset', 'export_allowance': 'input_number.sapn_free_exports',
    'rated_capacity': 'sensor.sigen_plant_rated_energy_capacity', 'battery_health': 'sensor.sigen_plant_battery_state_of_health',
}


def source_states(history, captured, at):
    """Use as-of recorded rows; missing static settings need older capture metadata."""
    at = pd.Timestamp(at)
    states, refs = recorded_states(history, captured, at, with_parent=True)
    inherited = []
    for key, entity in SETTINGS.items():
        rows = history.get(key, [])
        if rows:
            row = asof(rows, at, 31*86400)
            if not np.isfinite(float(row['value'])): raise ValueError('nonfinite setting: '+key)
            if key in ('rated_capacity', 'battery_health') and row.get('unit_of_measurement_str') != ('kWh' if key == 'rated_capacity' else '%'):
                raise ValueError('unsupported plant unit: '+key)
            states[entity] = {'state': str(row['value'])}
            refs[key] = row['time']
        else:
            old = captured.get(entity, {})
            updated = old.get('last_updated')
            if not updated or pd.Timestamp(updated) > at or not np.isfinite(float(old.get('state', 'nan'))):
                raise ValueError('no causal setting evidence: '+key)
            inherited.append({'entity': entity, 'last_updated': updated,
                'evidence': 'later capture indicates unchanged state since before origin; not independent archived availability'})
            refs[key] = updated
    row = asof(history['reground_block'], at, 31*86400)
    states['input_text.dh_last_reground_block'] = {'state': row['state']}
    refs['reground_block'] = row['time']
    for key, entity in [('dh_price','sensor.ai_price_forecast'), ('dh_price_low','sensor.ai_price_forecast_low'),
        ('dh_price_high','sensor.ai_price_forecast_high'), ('load_forecast','sensor.ai_load_forecast_high')]:
        row = asof(history[key], at, 900 if key.startswith('dh_price') else 3600)
        states[entity] = {'attributes': {'forecasts': parse_rows(row['forecasts_str'])}}
        refs[key] = row['time']
    if max(pd.Timestamp(refs[key]) for key in ('dh_price','dh_price_low','dh_price_high'))-min(
            pd.Timestamp(refs[key]) for key in ('dh_price','dh_price_low','dh_price_high')) > pd.Timedelta(seconds=2):
        raise ValueError('price quantile receipt spread exceeds2s')
    for day in ('today','tomorrow','day_3','day_4'):
        row = asof(history['solcast_'+day], at, 6*3600)
        states['sensor.solcast_pv_forecast_forecast_'+day] = {'attributes': {'detailedForecast': parse_rows(row['detailedForecast_str'])}}
        refs['solcast_'+day] = row['time']
    return states, refs, inherited


def source_coverage(states, at):
    """Report exact source target alignment separately from array-length parity."""
    at = pd.Timestamp(at)
    start = pd.Timestamp(boundary(at.to_pydatetime(),30))
    expected = pd.date_range(start, periods=144, freq='30min')
    targets = {}
    for key, entity, field in [('price','sensor.ai_price_forecast','timestamp'),
        ('price_low','sensor.ai_price_forecast_low','timestamp'), ('price_high','sensor.ai_price_forecast_high','timestamp'),
        ('load','sensor.ai_load_forecast_high','timestamp')]:
        targets[key] = [pd.Timestamp(row[field]) for row in states[entity]['attributes']['forecasts']]
    targets['pv'] = [pd.Timestamp(row['period_start']) for day in ('today','tomorrow','day_3','day_4')
        for row in states['sensor.solcast_pv_forecast_forecast_'+day]['attributes']['detailedForecast']
        if pd.Timestamp(row['period_start']) > at-pd.Timedelta(minutes=30)]
    reasons = []
    for key, values in targets.items():
        if any(value.tzinfo is None for value in values):
            raise ValueError('naive source target: '+key)
        if len(values) < 144 or pd.to_datetime(values[:144],utc=True).tolist() != expected.tz_convert('UTC').tolist():
            reasons.append(key+'_targets:misaligned_or_incomplete')
    payload = build_dh_payload(Inputs(states, at.to_pydatetime()))
    for key in ('pv_power_forecast','load_power_forecast','load_cost_forecast','prod_price_forecast'):
        if len(payload[key]) != 144 or not np.isfinite(payload[key]).all(): reasons.append(key+':invalid_array')
    return {'coverage_ready': not reasons, 'reasons': reasons, 'expected_start': start.isoformat(),
        'source_starts': {key: values[0].isoformat() if values else None for key,values in targets.items()},
        'source_counts': {key:len(values) for key,values in targets.items()}, 'payload_sha256':digest(payload)}


def load_alignment_score(rows, at, actuals):
    """Diagnostic only: same-vintage overlap, complete measured 30m targets."""
    start = pd.Timestamp(boundary(pd.Timestamp(at).to_pydatetime(),30))
    dates = pd.to_datetime([row['timestamp'] for row in rows], utc=True)
    expected = pd.date_range(start-pd.Timedelta(minutes=30), periods=144, freq='30min')
    if len(rows) != 144 or not dates.equals(expected):
        return None
    powers = np.array([float(row['power_load']) for row in rows])
    if not np.isfinite(powers).all(): raise ValueError('nonfinite archived load')
    # Last desired target has no prediction in this vintage: omit, never fill it.
    measured = actuals.load_base_w.resample('30min').agg(['mean','count'])
    measured = measured.reindex(pd.date_range(start, periods=143, freq='30min'))
    valid = measured['count'].eq(6) & np.isfinite(measured['mean'])
    truth = measured.loc[valid,'mean'].to_numpy()
    return {'paired_complete_targets':int(valid.sum()),
        'shifted_mae_w':float(np.abs(powers[:143][valid]-truth).mean()) if len(truth) else None,
        'aligned_mae_w':float(np.abs(powers[1:][valid]-truth).mean()) if len(truth) else None,
        'paired_forecast_change_w':float(np.abs(powers[1:][valid]-powers[:143][valid]).mean()) if len(truth) else None,
        'aligned_minus_shifted_14h_kwh':float((powers[1:29]-powers[:28]).sum()*.5/1000),
        'missing_tail_not_filled':True, 'mode':'timestamp_diagnostic_not_dispatch_savings'}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('history','journal','output'): parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--dataset',type=Path,help='optional frozen measured targets for alignment diagnostics')
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = json.loads((args.history/'manifest.json').read_text())
    if not manifest['export_complete'] or manifest['history_sha256'] != sha(args.history/'history.json'):
        raise ValueError('changed/incomplete source archive')
    history = json.loads((args.history/'history.json').read_text())
    actuals = None
    actual_sha = None
    if args.dataset:
        actual_sha = sha(args.dataset/'actuals.parquet')
        am = json.loads((args.dataset/'manifest.json').read_text())
        if not am['export_complete'] or actual_sha != am['parquet_sha256']:
            raise ValueError('changed/incomplete measured archive')
        actuals = pd.read_parquet(args.dataset/'actuals.parquet')
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro',uri=True) as db:
        record = json.loads(db.execute('SELECT record FROM handoffs ORDER BY rowid DESC LIMIT 1').fetchone()[0])
    rows = []
    for event in history['dh_anchor']:
        if pd.Timestamp(event['time']) < pd.Timestamp(manifest['start']): continue
        # Helper is written before solve; exclude its new value when restoring
        # previous-plan policy state at this input-time proxy.
        at = pd.Timestamp(event['time'])-pd.Timedelta(microseconds=1)
        try:
            states, refs, inherited = source_states(history,record['input_snapshot']['states'],at)
            row = {'origin_proxy':at.isoformat(), 'recorded_initial_soc_pct':event['value'],
                'reconstructed_initial_soc_pct':build_dh_payload(Inputs(states,at.to_pydatetime()))['soc_init']*100,
                'input_receipts':refs, 'capture_confirmed_settings':inherited, **source_coverage(states,at)}
            if actuals is not None:
                row['load_alignment_diagnostic'] = load_alignment_score(
                    states['sensor.ai_load_forecast_high']['attributes']['forecasts'],at,actuals)
        except (ValueError,KeyError,TypeError,SyntaxError,IndexError,OverflowError) as exc:
            row = {'origin_proxy':at.isoformat(), 'coverage_ready':False, 'reasons':[str(exc)]}
        rows.append(row)
    report = {'mode':'dh_source_admission_not_control_or_savings', 'origins':rows,
        'ready_count':sum(row['coverage_ready'] for row in rows), 'origin_count':len(rows),
        'provenance':{'history_sha256':manifest['history_sha256'],'captured_handoff_sha256':digest(record),
            'measured_actuals_sha256':actual_sha,
            'auditor_sha256':sha(Path(__file__)),'parser_sha256':sha(ROOT/'eval/archived_forecasts.py')},
        'publication_authorized':False,'limitations':['historical clock is helper-write proxy, not atomic input capture',
            'price quantile receipt proximity does not establish common model run identity',
            'missing static settings can use later capture with older unchanged-state metadata, explicitly identified',
            'alignment failures retained; do not shift/fill/renew stale source arrays']}
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'ready_count':report['ready_count'],'origin_count':len(rows),
        'origins':[{key:row.get(key) for key in ('origin_proxy','coverage_ready','reasons','source_starts')} for row in rows]}))


if __name__ == '__main__': main()
