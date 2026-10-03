"""Capture selected HA inputs, then compare pure payloads with deployed HA Jinja.

Read-only: no solver requests, services, helper writes or forecast publications.
Snapshots contain household telemetry; keep them in ignored data/energy_replay/.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sqlite3
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from config_utils import load_config
from energy_pipeline.payloads import Inputs, boundary, build_dh_payload, build_mpc_payload


class PackageLoader(yaml.SafeLoader):
    pass


PackageLoader.add_constructor('!secret', lambda loader, node: loader.construct_scalar(node))


def templates():
    package = yaml.load((ROOT / 'hass/packages/emhass.yaml').read_text(), Loader=PackageLoader)
    return {kind: {'variables': [step['variables'] for step in package['script'][name]['sequence']
                                if 'variables' in step],
                   'payload': package['rest_command'][name]['payload']}
            for kind, name in [('dh', 'emhass_dayahead_optim'), ('mpc', 'emhass_mpc')]}


def capture():
    sources = templates()
    entities = set(re.findall(r'(?:sensor|input_number|input_text)\.[a-z0-9_]+', json.dumps(sources)))
    cfg = load_config(ROOT / 'config.yaml')['home_assistant']
    req = urllib.request.Request(cfg['url'].rstrip('/') + '/api/states',
                                 headers={'Authorization': 'Bearer ' + cfg['token']})
    with urllib.request.urlopen(req, timeout=30) as response:
        rows = json.load(response)
    return {'captured_at': datetime.now(timezone.utc).isoformat(), 'timezone': 'Australia/Adelaide',
            'states': {row['entity_id']: {'state': row['state'], 'attributes': row['attributes']}
                       for row in rows if row['entity_id'] in entities}}


def differences(expected, actual, path=''):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            if key not in expected or key not in actual:
                yield f'{path}.{key}: missing key'
            else:
                yield from differences(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f'{path}: lengths {len(expected)} != {len(actual)}'
        for i, (left, right) in enumerate(zip(expected, actual)):
            yield from differences(left, right, f'{path}[{i}]')
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not (math.isfinite(expected) and math.isfinite(actual)) or abs(expected-actual) > 1e-9:
            yield f'{path}: {expected} != {actual}'
    elif expected != actual:
        yield f'{path}: {expected!r} != {actual!r}'


def replay(snapshot, container='hass'):
    request = dict(snapshot, templates=templates())
    oracle = (ROOT / 'scripts/ha_payload_oracle.py').read_text()
    result = subprocess.run(['docker', 'exec', '-i', container, 'python', '-c', oracle],
                            input=json.dumps(request), text=True, capture_output=True, check=True, timeout=120)
    expected = json.loads(result.stdout)
    inputs = Inputs(snapshot['states'], datetime.fromisoformat(snapshot['captured_at']), snapshot['timezone'])
    actual = {'dh': build_dh_payload(inputs), 'mpc': build_mpc_payload(inputs)}
    return list(differences(expected, actual)), actual


def scenarios(snapshot):
    """Deterministic mutations of a recorded snapshot; never presented as observations."""
    yield 'recorded', snapshot
    for weight in (-1, 0, 1, .371):
        variant = copy.deepcopy(snapshot)
        for kind in ('pv', 'buy', 'sell'):
            variant['states'][f'input_number.emhass_weight_{kind}_forecast']['state'] = str(weight)
        variant['states']['input_number.sapn_free_exports']['state'] = '100'
        yield f'weights_{weight}', variant
    for name in ('full_soc', 'low_soc', 'same_block', 'missing_prior', 'sparse_hwc',
                 'bad_hwc_json', 'curtailment', 'empty_power', 'zero_power',
                 'boundary_before', 'boundary_after', 'dst'):
        variant = copy.deepcopy(snapshot)
        states = variant['states']
        now = datetime.fromisoformat(snapshot['captured_at'])
        if name in ('full_soc', 'low_soc'):
            states['sensor.sigen_plant_battery_state_of_charge_derived']['state'] = '100' if name == 'full_soc' else '3.123456'
        elif name == 'same_block':
            states['input_text.dh_last_reground_block']['state'] = boundary(now, 30).strftime('dh-%Y%m%dT%H%MZ')
        elif name == 'missing_prior':
            states['sensor.dh_soc_batt_forecast']['attributes']['battery_scheduled_soc'] = []
        elif name in ('sparse_hwc', 'bad_hwc_json'):
            attrs = states['sensor.emhass_dh_hwc_power_plan_snapshot']['attributes']
            rows = attrs.get('deferrables_schedule_json') or []
            rows = json.loads(rows) if isinstance(rows, str) else rows
            attrs['deferrables_schedule_json'] = rows[::7] if name == 'sparse_hwc' else 'invalid'
        elif name == 'curtailment':
            states['sensor.emhass_current_pv_input_mode']['state'] = 'export_limit'
            states['sensor.solcast_pv_forecast_power_now']['attributes'].update(estimate=4321.7, estimate10=3456.1, estimate90=5678.9)
        elif name in ('empty_power', 'zero_power'):
            for entity, key in [('sensor.dh_p_load_forecast', 'dh_p_load_forecast'),
                                ('sensor.dh_p_pv_forecast', 'dh_p_pv_forecast')]:
                rows = states[entity]['attributes']['forecasts']
                if name == 'empty_power':
                    states[entity]['attributes']['forecasts'] = []
                else:
                    for row in rows:
                        row[key] = 0
        elif name.startswith('boundary_'):
            at = boundary(now, 30) + timedelta(minutes=30)
            variant['captured_at'] = (at + timedelta(microseconds=-1 if name == 'boundary_before' else 1)).isoformat()
        elif name == 'dst':
            # Shift the whole frozen timeline, including JSON-encoded HWC dates,
            # so forecasts remain populated across Adelaide's spring transition.
            at = datetime.fromisoformat('2026-10-03T16:29:59+00:00')
            delta = at-now
            def shift(value):
                if isinstance(value, dict):
                    return {key: shift(item) for key, item in value.items()}
                if isinstance(value, list):
                    return [shift(item) for item in value]
                if isinstance(value, str):
                    if value.startswith('['):
                        try:
                            return json.dumps(shift(json.loads(value)))
                        except ValueError:
                            pass
                    try:
                        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
                        if parsed.tzinfo is not None:
                            return (parsed+delta).isoformat()
                    except ValueError:
                        pass
                return value
            variant = shift(variant)
        yield name, variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='*', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--container', default='hass')
    parser.add_argument('--variants', action='store_true', help='also replay deterministic edge-case mutations')
    parser.add_argument('--handoff-db', type=Path, help='replay stored shadow handoff snapshots from a readonly SQLite journal')
    args = parser.parse_args()
    paths = list(args.snapshots)
    if args.capture:
        snapshot = capture()
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        args.capture.write_text(json.dumps(snapshot, indent=2) + '\n')
        paths.append(args.capture)
    if not paths and not args.handoff_db:
        parser.error('provide snapshots or --capture')
    failed = False
    for path in paths:
        snapshot = json.loads(path.read_text())
        cases = scenarios(snapshot) if args.variants else [('recorded', snapshot)]
        for label, variant in cases:
            mismatches, payloads = replay(variant, args.container)
            print(f'{path} [{label}]: {len(mismatches)} mismatches; DH/MPC load slots '
                  f"{len(payloads['dh']['load_power_forecast'])}/{len(payloads['mpc']['load_power_forecast'])}", flush=True)
            for mismatch in mismatches[:20]:
                print('  ' + mismatch)
            failed |= bool(mismatches)
    if args.handoff_db:
        with closing(sqlite3.connect(args.handoff_db.resolve().as_uri()+'?mode=ro', uri=True)) as db:
            records = db.execute('SELECT record FROM handoffs ORDER BY rowid').fetchall()
        if not records:
            parser.error('handoff journal contains no captured payloads')
        for raw, in records:
            record = json.loads(raw)
            mismatches, payloads = replay(record['input_snapshot'], args.container)
            mismatches.extend(differences(record['payloads'], payloads, 'stored_payloads'))
            print(f"handoff {record['price_run_id']}: {len(mismatches)} mismatches; readiness={record['readiness']}", flush=True)
            for mismatch in mismatches[:20]:
                print('  '+mismatch)
            failed |= bool(mismatches)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
