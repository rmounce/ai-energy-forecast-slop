"""Read-only export of independent economic telemetry; never change shared CQs/data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import urllib.request

import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_utils import load_config
from eval.measured_actuals import SOURCES, integrate, endpoint_samples, window


def inventory(client, ha_config):
    request = urllib.request.Request(ha_config['url'].rstrip('/')+'/api/states',
        headers={'Authorization': 'Bearer '+ha_config['token']})
    with urllib.request.urlopen(request, timeout=20) as response:
        states = {row['entity_id']: row for row in json.load(response)}
    names = '|'.join(re.escape(source.entity.split('.', 1)[1]) for source in SOURCES)
    keys = list(client.query(f'SHOW SERIES WHERE "entity_id" =~ /^({names})$/').get_points())
    catalog = {}
    for source in SOURCES:
        name = source.entity.split('.', 1)[1]
        measurements = sorted({row['key'].split(',', 1)[0] for row in keys
                               if re.search(r'(?:^|,)entity_id='+re.escape(name)+r'(?:,|$)', row['key'])})
        state = states.get(source.entity, {})
        attrs = state.get('attributes', {})
        preferred = source.entity.split('.', 1)[0]+(
            '__'+attrs['device_class'] if attrs.get('device_class') else '')
        # SHOW SERIES also finds CQ outputs with the same entity tag. Prefer
        # the raw HA measurement explicitly; never silently substitute an aggregate.
        measurement = preferred if preferred in measurements else None
        fields = list(client.query(f'SHOW FIELD KEYS FROM "{measurement}"').get_points()) if measurement else []
        catalog[source.column] = {'entity': source.entity, 'role': source.role,
            'expected_unit': source.unit, 'current_unit': attrs.get('unit_of_measurement'),
            'device_class': attrs.get('device_class'), 'current_state': state.get('state'),
            'last_updated': state.get('last_updated'), 'measurements': measurements,
            'measurement': measurement, 'fields': fields}
    return catalog


def query_samples(client, measurement, entity, field, start, end):
    # All identifiers come from fixed entity allowlist/discovered schema. Reject
    # unexpected syntax rather than interpolate arbitrary measurement/tag text.
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', value) for value in (measurement, entity, field)):
        raise ValueError('unsupported measurement/entity/field identifier')
    query = (f'SELECT "{field}" FROM "rp_raw"."{measurement}" '
             f'WHERE "entity_id" = \'{entity}\' AND time >= \'{start.isoformat()}\' '
             f'AND time < \'{end.isoformat()}\'')
    rows = list(client.query(query).get_points())
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz='UTC')), query
    return pd.Series([row.get(field) for row in rows],
                     index=pd.to_datetime([row['time'] for row in rows], utc=True)), query


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--output', type=Path, required=True, help='new ignored output directory')
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--max-hold-seconds', type=float, default=120)
    args = parser.parse_args()
    start, end = window(args.start, args.end)
    if end > pd.Timestamp.now(tz='UTC').floor('5min'):
        parser.error('end must exclude the current incomplete interval')
    if not 0 < args.max_hold_seconds <= 300:
        parser.error('power hold bound must be positive and at most 300 seconds')
    if args.output.exists():
        parser.error('output directory already exists')
    config = load_config(ROOT/'config.yaml')
    ic = config['influxdb']
    client = InfluxDBClient(host=ic['host'], port=ic.get('port', 8086),
        username=ic['username'], password=ic['password'], database=ic['database'], timeout=20, retries=0)
    manifest = {'schema': 1, 'mode': 'measured_economic_actuals',
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'start': start.isoformat(), 'end': end.isoformat(),
        'max_hold_seconds': args.max_hold_seconds, 'publication_authorized': False,
        'assumptions': ['current HA unit metadata does not establish historical unit stability',
                        'held observations expire; gaps remain missing',
                        'delivered PV is not counterfactual available solar',
                        'derived balance sensors are not independent validation meters'],
        'sources': {}, 'queries': {}, 'coverage': {}}
    frame = pd.DataFrame(index=pd.date_range(start, end, freq='5min', inclusive='left'))
    try:
        manifest['sources'] = inventory(client, config['home_assistant'])
        if not args.inventory_only:
            for source in SOURCES:
                info = manifest['sources'][source.column]
                if not info['measurement']:
                    info['export_status'] = 'missing_or_ambiguous_measurement'
                    continue
                if info['current_unit'] != source.unit:
                    info['export_status'] = 'unit_mismatch'
                    continue
                fields = {row['fieldKey']: row['fieldType'] for row in info['fields']}
                field = 'value' if 'value' in fields else None
                if source.unit is None:
                    field = 'state' if fields.get('state') == 'string' else field if fields.get(field) == 'string' else None
                if field is None:
                    info['export_status'] = 'unsupported_value_field'
                    continue
                hold = (86400 if source.role in ('applied_limit', 'curtailment_context') else
                        900 if source.role == 'forecast_proxy_comparison_only' else args.max_hold_seconds)
                info['query_entity'] = source.entity
                info['query_measurement'] = info['measurement']
                info['query_field'] = field
                samples, query = query_samples(client, info['measurement'], source.entity.split('.', 1)[1],
                    field, start-pd.Timedelta(seconds=hold), end)
                if samples.empty and source.role == 'applied_limit':
                    # Stable number entities need not emit recent state changes.
                    # The mode sensor records the applied caps as numeric attributes.
                    mode = manifest['sources']['curtailment_fraction']
                    mode_fields = {row['fieldKey'] for row in mode['fields']}
                    attribute = {'pv_limit_kw': 'pv_limit_kw', 'export_limit_kw': 'export_limit_kw',
                                 'charge_limit_kw': 'charge_limit_kw'}[source.column]
                    if mode['measurement'] and attribute in mode_fields:
                        samples, query = query_samples(client, mode['measurement'],
                            'emhass_current_pv_input_mode', attribute, start-pd.Timedelta(seconds=hold), end)
                        info.update(query_entity='sensor.emhass_current_pv_input_mode',
                                    query_measurement=mode['measurement'], query_field=attribute,
                                    fallback='recorded_mode_attribute_not_current_state')
                manifest['queries'][source.column] = query
                info['raw_samples'] = len(samples)
                if source.unit is None:
                    labels = {'measured': 0., 'transition': 1., 'pv_limit': 1., 'export_limit': 1.}
                    info['observed_states'] = sorted(str(value) for value in samples.dropna().unique())
                    samples = samples.map(labels)
                summary = integrate(samples, start, end, max_hold_seconds=hold)
                for column in summary:
                    label = source.column if column == 'mean' else source.column+'_'+column
                    frame[label] = summary[column]
                if source.unit == '%':
                    frame[source.column+'_end'] = endpoint_samples(samples, frame.index+pd.Timedelta(minutes=5),
                                                                   max_hold_seconds=hold)
                if source.column in ('grid_import_w', 'battery_charge_w'):
                    directions = ('import', 'export') if source.column == 'grid_import_w' else ('charge', 'discharge')
                    numeric = pd.to_numeric(samples, errors='coerce')
                    for direction, sign in zip(directions, (1, -1)):
                        split = integrate((numeric*sign).clip(lower=0), start, end, max_hold_seconds=hold)
                        frame[source.column+'_'+direction+'_kwh'] = split['mean']/1000*5/60
                info['export_status'] = 'exported'
                manifest['coverage'][source.column] = {
                    'complete_intervals': int(summary['mean'].notna().sum()), 'intervals': len(summary),
                    'observed_seconds_fraction': float(summary['coverage'].mean()),
                    'max_hold_seconds': hold}
                print(json.dumps({'source': source.column, **manifest['coverage'][source.column]}), flush=True)
    finally:
        client.close()
    args.output.mkdir(parents=True)
    if not args.inventory_only:
        path = args.output/'actuals.parquet'
        frame.to_parquet(path)
        manifest['parquet_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['export_complete'] = True
    manifest['exporter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['integration_sha256'] = hashlib.sha256((ROOT/'eval/measured_actuals.py').read_bytes()).hexdigest()
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'inventory_only': args.inventory_only,
                     'sources': {key: {'unit': value['current_unit'],
                         'measurement': value['measurement'],
                         'export_status': value.get('export_status', 'inventory_only')}
                         for key, value in manifest['sources'].items()}}), flush=True)


if __name__ == '__main__':
    main()
