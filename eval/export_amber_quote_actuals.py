"""Freeze raw Amber quote revisions and price complete measured grid flows offline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd
from influxdb import InfluxDBClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_utils import load_config
from eval.amber_quote_actuals import canonical_rates, adjusted_rates, accounting
from eval.measured_actuals import window

ENTITIES = {'general': 'amber_5min_current_general_price',
            'feed': 'amber_5min_current_feed_in_price',
            'adjusted_feed': 'amber_adjusted_confirmed_feed_in_price'}
FIELDS = ['value', 'unit_of_measurement_str', 'type_str', 'estimate', 'duration',
          'start_time_str', 'end_time_str', 'update_time_str', 'confirmed_end_time_str',
          'raw_price', 'export_allowance_adjustment', 'adjustments_str']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--receipt-lag-hours', type=float, default=6)
    parser.add_argument('--revision-archive', type=Path, help='re-audit a frozen archive without network access')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    if not 0 < args.receipt_lag_hours <= 24:
        parser.error('receipt lag must be positive and at most 24 hours')
    manifest_path = args.dataset/'manifest.json'
    measured = json.loads(manifest_path.read_text())
    actuals_path = args.dataset/'actuals.parquet'
    if not measured.get('export_complete') or sha(actuals_path) != measured.get('parquet_sha256'):
        parser.error('measured dataset incomplete or hash mismatch')
    start, end = window(measured['start'], measured['end'])
    query_start = start-pd.Timedelta(minutes=5)
    query_end = min(end+pd.Timedelta(hours=args.receipt_lag_hours), pd.Timestamp.now(tz='UTC'))
    rows, queries = {}, {}
    archive_manifest_hash = None
    if args.revision_archive:
        archive_path = args.revision_archive/'manifest.json'
        archive = json.loads(archive_path.read_text())
        if (not archive.get('export_complete') or archive.get('mode') != 'archived_amber_observed_flow_accounting'
                or archive.get('sources') != ENTITIES
                or archive.get('measured_actuals_sha256') != sha(actuals_path)
                or archive.get('start') != start.isoformat() or archive.get('end') != end.isoformat()):
            parser.error('revision archive does not match measured dataset/source contract')
        query_start, query_end = pd.Timestamp(archive['query_start']), pd.Timestamp(archive['query_end'])
        queries = archive['queries']
        archive_manifest_hash = sha(archive_path)
        for source in ENTITIES:
            filename = source+'_revisions.json'
            path = args.revision_archive/filename
            if sha(path) != archive['files'].get(filename):
                parser.error('revision archive hash mismatch')
            rows[source] = json.loads(path.read_text())
    else:
        config = load_config(ROOT/'config.yaml')['influxdb']
        client = InfluxDBClient(host=config['host'], port=config.get('port', 8086),
            username=config['username'], password=config['password'], database=config['database'], timeout=20, retries=0)
        try:
            for source, entity in ENTITIES.items():
                fields = ','.join('"'+field+'"' for field in FIELDS)
                query = (f'SELECT {fields} FROM "rp_raw"."sensor__monetary" '
                         f'WHERE "entity_id" = \'{entity}\' AND time >= \'{query_start.isoformat()}\' '
                         f'AND time < \'{query_end.isoformat()}\'')
                queries[source] = query
                rows[source] = list(client.query(query).get_points())
        finally:
            client.close()
    general, general_errors = canonical_rates(rows['general'], start, end)
    feed, feed_errors = canonical_rates(rows['feed'], start, end)
    adjusted, adjusted_errors = adjusted_rates(rows['adjusted_feed'], feed, rows['feed'])
    actuals = pd.read_parquet(actuals_path)
    frame, summary = accounting(actuals, general, feed, adjusted)
    post_general, post_general_errors = canonical_rates(rows['general'], start, end, require_post_end_receipt=True)
    post_feed, post_feed_errors = canonical_rates(rows['feed'], start, end, require_post_end_receipt=True)
    post_adjusted, post_adjusted_errors = adjusted_rates(rows['adjusted_feed'], post_feed, rows['feed'])
    post_frame, post_summary = accounting(actuals, post_general, post_feed, post_adjusted)
    args.output.mkdir(parents=True)
    files = {}
    for source, history in rows.items():
        path = args.output/(source+'_revisions.json')
        path.write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
        files[path.name] = sha(path)
    for label, data in [('general_rates', general), ('feed_rates', feed), ('adjusted_rates', adjusted), ('priced_flows', frame), ('post_end_priced_flows', post_frame)]:
        path = args.output/(label+'.parquet')
        data.to_parquet(path)
        files[path.name] = sha(path)
    report = {'schema': 1, 'mode': 'archived_amber_observed_flow_accounting',
        'exported_at': datetime.now(timezone.utc).isoformat(), 'start': start.isoformat(), 'end': end.isoformat(),
        'query_start': query_start.isoformat(), 'query_end': query_end.isoformat(),
        'requested_receipt_lag_hours': args.receipt_lag_hours if not args.revision_archive else archive['requested_receipt_lag_hours'],
        'revision_archive_manifest_sha256': archive_manifest_hash, 'publication_authorized': False,
        'measured_manifest_sha256': sha(manifest_path), 'measured_actuals_sha256': sha(actuals_path),
        'exporter_sha256': sha(Path(__file__)), 'accounting_sha256': sha(ROOT/'eval/amber_quote_actuals.py'),
        'sources': ENTITIES, 'queries': queries, 'files': files,
        'rate_conventions': {'general_state': 'positive import cost',
            'feed_state': 'positive export revenue; MQTT state negates raw API per_kwh',
            'feed_forecast_attributes': 'raw API sign; negative is export revenue; not used for observed accounting'},
        'transition_filter': 'exclude identified <=100ms MQTT state-before-next-interval-attribute pairs; preserve raw rows',
        'raw_revision_rows': {source: len(history) for source, history in rows.items()},
        'rejections': {'general': general_errors, 'feed': feed_errors, 'adjusted_feed': adjusted_errors},
        'summary': summary,
        'post_end_receipt_sensitivity': {'summary': post_summary, 'rejections': {'general': post_general_errors,
            'feed': post_feed_errors, 'adjusted_feed': post_adjusted_errors}},
        'semantics_source': 'https://github.com/amberelectric/public-api/discussions/214',
        'limitations': ['archived CurrentInterval non-estimated quotes are not reconciled invoice settlement',
                        'CurrentInterval estimate=false denotes non-estimated API quote; receipt-before-end alone does not establish provisional status',
                        'separate post-end-receipt sensitivity deliberately reduces coverage and is not a validated truth filter',
                        'latest revision only within bounded receipt query; later revisions remain unknown',
                        'local adjusted feed scenario preserves recorded allowance; billing meaning unverified',
                        'excludes fixed charges and any flow/rate coverage gaps',
                        'observed cashflow is not counterfactual savings',
                        'explicit source interval convention: quoted start equals end minus 5 minutes plus 1 second'],
        'export_complete': True}
    # Missing totals must serialize as null, never NaN.
    def sanitize(value):
        if isinstance(value, dict): return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list): return [sanitize(item) for item in value]
        if isinstance(value, float) and not pd.notna(value): return None
        return value
    report = sanitize(report)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'summary': report['summary'], 'rejections': report['rejections']}, allow_nan=False))


if __name__ == '__main__':
    main()
