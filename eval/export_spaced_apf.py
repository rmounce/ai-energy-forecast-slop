"""Fixed six-hour APF feed sampling; read-only, no selection by realised price."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.audit_amber_forecast_archive import parse_revision, utc


def schedule(start,end):
    start,end = utc(start),utc(end)
    if not start < end <= start+pd.Timedelta(days=7):
        raise ValueError('positive <=7d sample window required')
    if start != start.floor('D') or end != end.floor('D'):
        raise ValueError('UTC midnight boundaries required')
    return list(pd.date_range(start,end,freq='6h',inclusive='left'))


def query_for(at):
    # Fixed entity/fields and two-minute receipt window; no arbitrary-query surface.
    at = utc(at)
    return ('SELECT "Forecasts_str","unit_of_measurement_str" FROM "rp_raw"."sensor__monetary" '
        'WHERE "entity_id" = \'amber_5min_forecasts_extended_feed_in_price\' '
        f'AND time >= \'{at.isoformat()}\' AND time < \'{(at+pd.Timedelta(minutes=2)).isoformat()}\' '
        'ORDER BY time ASC LIMIT 1')


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('start','end'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try: origins = schedule(args.start,args.end)
    except ValueError as exc: parser.error(str(exc))
    if args.output.exists(): parser.error('new output directory required')
    if utc(args.end)>pd.Timestamp.now(tz='UTC').floor('D'): parser.error('complete past UTC days required')
    from config_utils import load_config
    from influxdb import InfluxDBClient
    config = load_config(ROOT/'config.yaml')['influxdb']
    client = InfluxDBClient(host=config['host'],port=config.get('port',8086),
        username=config['username'],password=config['password'],database=config['database'],timeout=20,retries=0)
    rows, receipts = [],[]
    try:
        for at in origins:
            query = query_for(at)
            selected = list(client.query(query).get_points())
            if len(selected)>1: raise ValueError('sample row budget exceeded')
            receipt = {'scheduled_window_start':at.isoformat(),'query':query,'status':'missing_receipt'}
            if selected:
                row = selected[0]
                parsed = parse_revision(row,'feed_in')
                if not at <= utc(row['time']) < at+pd.Timedelta(minutes=2):
                    raise ValueError('receipt outside predefined sample window')
                rows.append(row)
                receipt.update(status='observed_receipt',receipt=parsed['receipt'],payload_sha256=parsed['payload_sha256'])
            receipts.append(receipt)
    finally: client.close()
    args.output.mkdir(parents=True)
    path = args.output/'history.json'
    path.write_text(json.dumps({'mpc_apf_feed_in':rows},indent=2,allow_nan=False)+'\n')
    manifest = {'mode':'predeclared_six_hour_apf_feed_sample','start':utc(args.start).isoformat(),'end':utc(args.end).isoformat(),
        'scheduled_windows':len(origins),'observed_receipts':len(rows),'receipts':receipts,
        'history_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_sha256':hashlib.sha256((ROOT/'eval/audit_amber_forecast_archive.py').read_bytes()).hexdigest(),
        'export_complete':True,'publication_authorized':False,
        'limitations':['first receipt in each fixed00/06/12/18 UTC two-minute window, missing windows retained',
            'selection independent of realised prices; seven days still too short for seasonal claims',
            'recorder receipt not proven atomic HA availability',
            'overlapping forecast horizons/targets remain dependent',
            'feed APF only; no controller/load/PV/spare export capacity reconstruction']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'scheduled_windows':len(origins),'observed_receipts':len(rows),'output':str(args.output)}))


if __name__ == '__main__': main()
