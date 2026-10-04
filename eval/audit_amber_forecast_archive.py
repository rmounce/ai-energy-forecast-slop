"""Bounded read-only freeze of archived Amber APF revisions; no dispatch reconstruction."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENTITIES = {leg: 'amber_5min_forecasts_extended_'+leg+'_price' for leg in ('general', 'feed_in')}
PRICES = ('per_kwh', 'spot_per_kwh', 'advanced_price_low', 'advanced_price_predicted', 'advanced_price_high')


def utc(value):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError('timestamp must be aware and finite')
    return stamp.tz_convert('UTC')


def parse_revision(record, leg):
    """Keep raw Amber monetary convention; feed APF labels order by export value."""
    if leg not in ENTITIES or record.get('unit_of_measurement_str') != '$/kWh':
        raise ValueError('unknown leg or unsupported unit')
    receipt = utc(record['time'])
    raw = record.get('Forecasts_str')
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise ValueError('missing or oversized payload')
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = ast.literal_eval(raw)
    if not isinstance(payload, list) or not 1 <= len(payload) <= 1000:
        raise ValueError('unsupported forecast payload')
    ends, starts = [], []
    durations = {}
    for item in payload:
        if not isinstance(item, dict) or item.get('type') != 'ForecastInterval':
            raise ValueError('unsupported forecast row')
        duration = item.get('duration')
        if isinstance(duration, bool) or duration not in (5, 30):
            raise ValueError('unsupported duration')
        start, end = utc(item['start_time']), utc(item['end_time'])
        if end-start not in (pd.Timedelta(minutes=duration), pd.Timedelta(minutes=duration)-pd.Timedelta(seconds=1)):
            raise ValueError('interval does not match declared duration')
        if end != end.floor('5min'):
            raise ValueError('unaligned interval end')
        for field in PRICES:
            if isinstance(item.get(field), bool) or not isinstance(item.get(field), (int, float)) or not math.isfinite(item[field]):
                raise ValueError('missing or nonfinite price')
        low, predicted, high = [item[field] for field in PRICES[2:]]
        if not (low <= predicted <= high if leg == 'general' else low >= predicted >= high):
            raise ValueError('incoherent advanced price bounds')
        starts.append(end-pd.Timedelta(minutes=duration))
        ends.append(end)
        durations[str(duration)] = durations.get(str(duration), 0)+1
    if any(starts[i] != ends[i-1] for i in range(1, len(ends))):
        raise ValueError('forecast intervals overlap, duplicate or have gaps')
    return {'receipt': receipt.isoformat(), 'payload_sha256': hashlib.sha256(raw.encode()).hexdigest(),
            'intervals': len(payload), 'durations': durations, 'start': starts[0].isoformat(),
            'end': ends[-1].isoformat(), 'hours_after_receipt': (ends[-1]-receipt).total_seconds()/3600,
            'rows': payload}


def select_asof(revisions, origin, max_age_minutes=15):
    """No future receipt permitted; revisions are independently received per leg."""
    origin = utc(origin)
    if not math.isfinite(max_age_minutes) or max_age_minutes <= 0:
        raise ValueError('invalid maximum age')
    available = [row for row in revisions if utc(row['receipt']) <= origin]
    if not available:
        return None
    latest = max(available, key=lambda row: utc(row['receipt']))
    same_receipt = [row for row in available if utc(row['receipt']) == utc(latest['receipt'])]
    if len({row['payload_sha256'] for row in same_receipt}) != 1:
        raise ValueError('conflicting latest APF revision')
    if origin-utc(latest['receipt']) > pd.Timedelta(minutes=max_age_minutes):
        return None
    return latest


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-revisions', type=int, default=12)
    args = parser.parse_args()
    start, end = utc(args.start), utc(args.end)
    if not start < end <= start+pd.Timedelta(days=7) or not 1 <= args.max_revisions <= 24:
        parser.error('window must be positive and <=7 days; max revisions 1–24')
    if args.output.exists():
        parser.error('output already exists')
    from config_utils import load_config
    from influxdb import InfluxDBClient
    config = load_config(ROOT/'config.yaml')['influxdb']
    client = InfluxDBClient(host=config['host'], port=config.get('port', 8086), username=config['username'],
                           password=config['password'], database=config['database'], timeout=20, retries=0)
    raw, revisions, queries, inventory = {}, {}, {}, {}
    try:
        for leg, entity in ENTITIES.items():
            where = (f'FROM "rp_raw"."sensor__monetary" WHERE "entity_id" = \'{entity}\' '
                     f'AND time >= \'{start.isoformat()}\' AND time < \'{end.isoformat()}\'')
            queries[leg] = {'count': 'SELECT count("Forecasts_str") '+where,
                'first': 'SELECT "Forecasts_str","unit_of_measurement_str" '+where+f' ORDER BY time ASC LIMIT {args.max_revisions}',
                'last': 'SELECT "Forecasts_str","unit_of_measurement_str" '+where+' ORDER BY time DESC LIMIT 1'}
            count = list(client.query(queries[leg]['count']).get_points())
            first = list(client.query(queries[leg]['first']).get_points())
            last = list(client.query(queries[leg]['last']).get_points())
            raw[leg] = {'first': first, 'last': last}
            revisions[leg] = [parse_revision(row, leg) for row in first]
            last_parsed = [parse_revision(row, leg) for row in last]
            summary = lambda row: {key: value for key, value in row.items() if key != 'rows'}
            inventory[leg] = {'archived_count': count[0]['count'] if count else 0,
                'frozen_revisions': len(first), 'first': summary(revisions[leg][0]) if first else None,
                'last': summary(last_parsed[0]) if last else None}
    finally:
        client.close()
    origins = sorted({row['receipt'] for rows in revisions.values() for row in rows})
    pairs = []
    for origin in origins:
        pair = {leg: select_asof(rows, origin) for leg, rows in revisions.items()}
        if all(pair.values()):
            pairs.append({'origin': origin, 'legs': {leg: {key: row[key] for key in ('receipt', 'payload_sha256', 'hours_after_receipt')}
                for leg, row in pair.items()}})
        if len(pairs) == 24:
            break
    args.output.mkdir(parents=True)
    freeze = args.output/'revisions.json'
    freeze.write_text(json.dumps(raw, indent=2, allow_nan=False)+'\n')
    manifest = {'mode': 'bounded_apf_archive_feasibility_not_production_replay', 'start': start.isoformat(), 'end': end.isoformat(),
        'queries': queries, 'inventory': inventory, 'asof_pairs': pairs,
        'raw_revisions_sha256': hashlib.sha256(freeze.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'publication_authorized': False,
        'limitations': ['Influx event timestamp treated as recorder receipt, not proven HA availability timestamp',
            'general and feed latest revisions are independently received, not an atomic API snapshot',
            'only first bounded revisions plus final revision frozen; not full-window coverage proof',
            'extended 5m archive does not prove original billing-interval APF consumed by logged DH curve',
            'weights, current effective prices, allowance, accepted DH/PV/HWC state not reconstructed',
            'no wholesale conversion, tariff reconstruction, dispatch or savings calculation']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'inventory': inventory, 'asof_pairs': len(pairs)}, allow_nan=False))


if __name__ == '__main__':
    main()
