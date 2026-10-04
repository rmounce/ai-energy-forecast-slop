"""Read-only missing MPC helper-clock diagnostics; never admit inferred replay origins."""
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
from energy_pipeline.payloads import Inputs, integer, mpc_soc
from energy_pipeline.solver_replay import digest
from eval.audit_control_fidelity import asof, parse_curve, recorded_states


def preceding_timer_evidence(history, at):
    at = pd.Timestamp(at)
    eligible = [row for row in history['mpc_anchor']
                if at-pd.Timedelta(minutes=15) <= pd.Timestamp(row['time']) < at
                and pd.Timestamp(row['time']).minute % 5 != 0
                and 25 <= (pd.Timestamp(row['time'])-pd.Timestamp(row['time']).floor('min')).total_seconds() <= 25.2]
    return eligible[-5:]


def paired_curve(rows, publication, column, key, first_target):
    at = pd.Timestamp(publication['time'])
    pairs = [row for row in rows if abs((pd.Timestamp(row['time'])-at).total_seconds()) <= 1]
    if len(pairs) != 1:
        raise ValueError('missing or ambiguous paired publication: '+key)
    row = pairs[0]
    curve = parse_curve(row[column], key)
    if pd.Timestamp(curve[0]['date']) != pd.Timestamp(first_target):
        raise ValueError('paired target start mismatch: '+key)
    return curve, row['time']


def capacity_evidence(history, captured, at):
    states, refs, inherited = {}, {}, []
    for key, entity, unit in [('rated_capacity', 'sensor.sigen_plant_rated_energy_capacity', 'kWh'),
                              ('battery_health', 'sensor.sigen_plant_battery_state_of_health', '%')]:
        if history.get(key):
            row = asof(history[key], at, 31*86400)
            if row.get('unit_of_measurement_str') != unit:
                raise ValueError('unsupported capacity/health unit')
            value = float(row['value'])
            refs[key] = row['time']
        else:
            row = captured.get(entity, {})
            if not row.get('last_updated') or pd.Timestamp(row['last_updated']) > pd.Timestamp(at):
                raise ValueError('missing causal capacity/health evidence')
            if row.get('attributes', {}).get('unit_of_measurement') != unit:
                raise ValueError('unsupported captured capacity/health unit')
            value = float(row['state'])
            refs[key] = row['last_updated']
            inherited.append({'entity': entity, 'last_updated': row['last_updated'],
                              'evidence': 'capture_confirmed_static_fallback_not_independent_historical_availability'})
        if not math.isfinite(value) or value <= 0:
            raise ValueError('invalid capacity/health')
        states[key] = value
    return int(1000*states['rated_capacity']*states['battery_health']/100), refs, inherited


def candidate(history, controls, captured, plant, publication, at):
    at, pub = pd.Timestamp(at), pd.Timestamp(publication['time'])
    result = {'modeled_origin': at.isoformat(), 'diagnostic_only': True, 'checks': {}, 'supported': False}
    try:
        if at.minute % 5 == 0 or not 0 < (pub-at).total_seconds() <= 15:
            raise ValueError('not a bounded eligible non-five-minute timer candidate')
        evidence = preceding_timer_evidence(history, at)
        result['preceding_timer_clock_evidence'] = evidence
        if len(evidence) < 3:
            raise ValueError('insufficient preceding observed timer-clock evidence')
        # Attribute an admission failure explicitly without renewing stable receipts.
        ages = {}
        for key in ('soc', 'pv', 'load', 'loss', 'mode', 'dh_soc', 'dh_load', 'dh_pv', 'dh_anchor', 'hwc'):
            row = asof(history[key], at, float('inf'))
            ages[key] = (at-pd.Timestamp(row['time'])).total_seconds()
        result['source_age_seconds'] = ages
        stale = [key for key, age in ages.items() if age > (120 if key in ('soc', 'pv', 'load', 'loss', 'mode') else 3600)]
        if stale:
            raise ValueError('stale control observation: '+','.join(stale))
        states, refs = recorded_states(history, captured, at, with_parent=True)
        if states['sensor.emhass_current_pv_input_mode']['state'] != 'measured':
            raise ValueError('unsupported nonmeasured PV source mode')
        energy, capacity_refs, inherited = capacity_evidence(history, captured, at)
        helper = asof(history['mpc_anchor'], at, 31*86400)
        soc = mpc_soc(Inputs(states, at.to_pydatetime()))
        fixed = max(0, min(integer(states['sensor.sigen_inverter_conversion_loss']['state']), 140))
        gross = integer(states['sensor.sigen_power_pv_gross']['state'])
        load = integer(states['sensor.sigen_plant_consumed_power']['state'])+max(0, fixed-gross)
        pv = max(0, gross-fixed)
        power = parse_curve(publication['battery_scheduled_power_str'], 'mpc_p_batt_forecast')
        target = power[0]['date']
        if pd.Timestamp(target) != at.floor('5min') or len(power) != 168:
            raise ValueError('unexpected published MPC power target grid')
        curves, publication_refs = {}, {}
        for name, column, key, rows in (
                ('soc', 'battery_scheduled_soc_str', 'mpc_soc_batt_forecast', history['mpc_soc']),
                ('load', 'forecasts_str', 'mpc_p_load_forecast', controls['mpc_load']),
                ('pv', 'forecasts_str', 'mpc_p_pv_forecast', controls['mpc_pv'])):
            curves[name], publication_refs[name] = paired_curve(rows, publication, column, key, target)
            if [pd.Timestamp(row['date']) for row in curves[name]] != [pd.Timestamp(row['date']) for row in power]:
                raise ValueError('paired full target grid mismatch: '+name)
        initial = round(soc.soc_init_pct/100, 4)*100
        terminal = round(soc.soc_final_pct/100, 4)*100
        p = float(power[0]['mpc_p_batt_forecast'])
        change = p*(1/plant['battery_discharge_efficiency'] if p >= 0 else plant['battery_charge_efficiency'])*5/60/energy*100
        expected_end = initial-change
        actual_end = float(curves['soc'][0]['mpc_soc_batt_forecast'])
        checks = {'unchanged_helper_value': abs(float(helper['value'])-soc.soc_init_pct) <= 0.00011,
                  'first_two_load_slots': all(float(row['mpc_p_load_forecast']) == load for row in curves['load'][:2]),
                  'first_two_pv_slots': all(float(row['mpc_p_pv_forecast']) == pv for row in curves['pv'][:2]),
                  'initial_to_first_endpoint_rounding': abs(actual_end-expected_end) <= 0.0051,
                  'terminal_rounding': abs(float(curves['soc'][-1]['mpc_soc_batt_forecast'])-terminal) <= 0.0051,
                  'power_state_curve_consistent': p == float(publication['value'])}
        previous = [row for row in history['mpc_battery'] if pd.Timestamp(row['time']) < pub]
        if not previous:
            raise ValueError('no preceding power curve')
        old = max(previous, key=lambda row: pd.Timestamp(row['time']))
        checks['distinct_power_trajectory'] = power != parse_curve(old['battery_scheduled_power_str'], 'mpc_p_batt_forecast')
        result.update(checks=checks, supported=all(checks.values()), helper_receipt=helper,
            input_receipts=refs | capacity_refs, capture_confirmed_settings=inherited,
            published_evidence_receipts=publication_refs,
            expected={'load_first_two_w': load, 'pv_first_two_w': pv, 'soc_init_pct': soc.soc_init_pct,
                      'solver_rounded_initial_pct': initial, 'solver_rounded_terminal_pct': terminal,
                      'first_endpoint_pct': expected_end, 'runtime_capacity_wh': energy},
            first_endpoint_error_pct=actual_end-expected_end)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        result['rejection'] = str(exc)
    return result


def audit(history, controls, captured, plant, start, end):
    rows = []
    for publication in history['mpc_battery']:
        pub = pd.Timestamp(publication['time'])
        if not pd.Timestamp(start) <= pub < pd.Timestamp(end):
            continue
        row = {'publication': publication['time'], 'candidate_origins_admitted': False}
        try:
            anchor = asof(history['mpc_anchor'], pub, 15)
            row.update(status='observed_fresh_helper_clock', observed_origin=anchor['time'], helper_value=anchor['value'])
        except ValueError as exc:
            row.update(status='missing_fresh_helper_clock', rejection=str(exc), candidates=[
                candidate(history, controls, captured, plant, publication,
                          pub.floor('min')+pd.Timedelta(seconds=offset)) for offset in (25., 25.1)])
        rows.append(row)
    return {'scope': 'retrospective_helper_event_clock_diagnostic_not_replay_admission', 'rows': rows,
            'summary': {'publications': len(rows), 'observed_fresh_helper_clocks': sum(row['status']=='observed_fresh_helper_clock' for row in rows),
                        'missing_fresh_helper_clocks': sum(row['status']=='missing_fresh_helper_clock' for row in rows),
                        'missing_clocks_with_both_supported_candidates': sum(len(row.get('candidates', [])) == 2 and all(c['supported'] for c in row['candidates']) for row in rows)},
            'limitations': ['known observed helper clocks preserved; no global relaxation of15s helper-publication freshness',
                'candidate timer25/25.1s is a modeled clock sensitivity corroborated by preceding receipts, not exact historical capture',
                'no immutable historical scheduler configuration; current config cannot establish historical schedule',
                'published future curves diagnose consistency only; all modeled inputs selected as-of candidate clock',
                'SoC recurrence uses nominal capacity/runtime health and published rounding; independent physical stock unavailable',
                'battery efficiencies from frozen reference solver configuration; independent historical settings equivalence unproven',
                'distinct curve is consistent with a fresh solve, but is not proof of its invocation or request snapshot',
                'static settings require explicit older capture metadata; no current-state substitution'],
            'publication_authorized': False, 'replay_admission_authorized': False}


def read_archive(folder):
    manifest = json.loads((folder/'manifest.json').read_text())
    path = folder/'history.json'
    if not manifest['export_complete'] or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['history_sha256']:
        raise ValueError('changed or incomplete clock archive')
    return manifest, json.loads(path.read_text())


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('history', 'control-history', 'journal', 'replay', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('new output file required')
    manifest, history = read_archive(args.history)
    controls_manifest, controls = read_archive(args.control_history)
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro', uri=True) as connection:
        records = [json.loads(row[0]) for row in connection.execute('SELECT record FROM handoffs')]
    record = max(records, key=lambda row: row['captured_at'])
    bundle = json.loads((args.replay/'bundle.json').read_text())
    report = json.loads((args.replay/'report.json').read_text())
    if digest(bundle) != report['bundle_sha256']:
        raise ValueError('changed reference solver bundle')
    result = audit(history, controls, record['input_snapshot']['states'], bundle['configuration']['plant_conf'], manifest['start'], manifest['end'])
    result['provenance'] = {'history_sha256': manifest['history_sha256'], 'control_history_sha256': controls_manifest['history_sha256'],
        'journal_sha256': hashlib.sha256(args.journal.read_bytes()).hexdigest(), 'captured_handoff_sha256': digest(record),
        'captured_at': record['captured_at'], 'reference_bundle_sha256': digest(bundle),
        'configuration_sha256': digest(bundle['configuration']),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                             for name in ('energy_pipeline/payloads.py', 'eval/audit_control_fidelity.py', 'energy_pipeline/solver_replay.py')}}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
