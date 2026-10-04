"""Bounded read-only control/telemetry archive for replay fidelity diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Fixed fields/entities: no credentials, arbitrary queries, or control writes.
SOURCES = {
    'mpc_battery': ('sensor__power', 'mpc_p_batt_forecast', ['value', 'battery_scheduled_power_str']),
    'mpc_soc': ('sensor__battery', 'mpc_soc_batt_forecast', ['value', 'battery_scheduled_soc_str']),
    'dh_soc': ('sensor__battery', 'dh_soc_batt_forecast', ['value', 'battery_scheduled_soc_str']),
    'dh_load': ('sensor__power', 'dh_p_load_forecast', ['value', 'forecasts_str']),
    'dh_pv': ('sensor__power', 'dh_p_pv_forecast', ['value', 'forecasts_str']),
    'dh_anchor': ('input_number', 'dh_last_soc_init', ['value']),
    'mpc_anchor': ('input_number', 'mpc_last_soc_init', ['value']),
    'hwc': ('sensor', 'emhass_dh_hwc_power_plan_snapshot', ['state', 'deferrables_schedule_json_str']),
    'mode': ('sensor', 'emhass_current_pv_input_mode', ['state', 'gross_pv_w', 'plant_pv_w',
        'grid_w', 'battery_w', 'battery_soc', 'pv_limit_kw', 'export_limit_kw', 'charge_limit_kw']),
    'pv': ('sensor__power', 'sigen_power_pv_gross', ['value']),
    'load': ('sensor__power', 'sigen_plant_consumed_power', ['value']),
    'battery': ('sensor__power', 'sigen_inverter_battery_power', ['value']),
    'grid': ('sensor__power', 'sigen_plant_grid_active_power', ['value']),
    'loss': ('sensor__power', 'sigen_inverter_conversion_loss', ['value']),
    'soc': ('sensor__battery', 'sigen_plant_battery_state_of_charge_derived', ['value']),
}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if (start.tzinfo is None or end.tzinfo is None or pd.isna(start) or pd.isna(end) or
            not start < end <= start+pd.Timedelta(minutes=90) or args.output.exists()):
        parser.error('aware positive window <=90m and new output directory required')
    start, end = start.tz_convert('UTC'), end.tz_convert('UTC')
    from config_utils import load_config
    from influxdb import InfluxDBClient
    config = load_config(ROOT/'config.yaml')['influxdb']
    client = InfluxDBClient(host=config['host'], port=config.get('port', 8086),
        username=config['username'], password=config['password'], database=config['database'],
        timeout=20, retries=0)
    raw, queries, schema = {}, {}, {}
    try:
        for measurement in sorted({source[0] for source in SOURCES.values()}):
            schema[measurement] = {row['fieldKey']: row['fieldType'] for row in
                client.query(f'SHOW FIELD KEYS FROM "{measurement}"').get_points()}
        for name, (measurement, entity, requested) in SOURCES.items():
            fields = [field for field in requested if field in schema[measurement]]
            if not fields:
                raw[name] = []
                continue
            columns = ','.join('"'+field+'"' for field in fields)
            query = (f'SELECT {columns} FROM "rp_raw"."{measurement}" '
                f'WHERE "entity_id" = \'{entity}\' AND time >= \'{start.isoformat()}\' '
                f'AND time < \'{end.isoformat()}\' ORDER BY time ASC LIMIT 10001')
            queries[name] = query
            raw[name] = list(client.query(query).get_points())
            if len(raw[name]) > 10000:
                raise ValueError('source exceeds bounded row budget: '+name)
            print(json.dumps({'source': name, 'rows': len(raw[name])}), flush=True)
    finally:
        client.close()
    args.output.mkdir(parents=True)
    archive = args.output/'history.json'
    archive.write_text(json.dumps(raw, indent=2, allow_nan=False)+'\n')
    manifest = {'mode': 'read_only_control_fidelity_archive', 'start': start.isoformat(),
        'end': end.isoformat(), 'queries': queries, 'sources': SOURCES,
        'selected_schema': {key: {field: schema[measurement].get(field) for field in fields}
            for key, (measurement, _, fields) in SOURCES.items()},
        'history_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'export_complete': True, 'publication_authorized': False,
        'limitations': ['Influx event time is not a proven solver-input capture timestamp',
            'independently recorded entities are not an atomic solver input snapshot',
            'published plan revisions can include state/attribute publication transitions',
            'a plan publication is not proof of a separate successful solve',
            'missing historical state remains missing; no current-state fallback']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
