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

DH_SOURCES = {
    **{'mpc_apf_'+leg: ('sensor__monetary', 'amber_5min_forecasts_extended_'+leg+'_price',
        ['Forecasts_str', 'unit_of_measurement_str']) for leg in ('general', 'feed_in')},
    'dh_price': (None, 'ai_price_forecast', ['forecasts_str']),
    'dh_price_low': (None, 'ai_price_forecast_low', ['forecasts_str']),
    'dh_price_high': (None, 'ai_price_forecast_high', ['forecasts_str']),
    'load_forecast': (None, 'ai_load_forecast_high', ['forecasts_str']),
    **{'solcast_'+day: (None, 'solcast_pv_forecast_forecast_'+day, ['detailedForecast_str'])
       for day in ('today', 'tomorrow', 'day_3', 'day_4')},
    **{name: ('input_number', entity, ['value']) for name, entity in (
        ('buy_weight', 'emhass_weight_buy_forecast'), ('sell_weight', 'emhass_weight_sell_forecast'),
        ('pv_weight', 'emhass_weight_pv_forecast'), ('discharge_weight', 'emhass_weight_battery_discharge'),
        ('minimum_soc', 'battery_soc_min_target'), ('soc_buffer', 'battery_soc_min_buffer'),
        ('target_offset', 'emhass_target_soc_offset'), ('export_allowance', 'sapn_free_exports'))},
    'reground_block': ('input_text', 'dh_last_reground_block', ['state', 'value', 'value_str']),
    'rated_capacity': (None, 'sigen_plant_rated_energy_capacity', ['value', 'unit_of_measurement_str']),
    'battery_health': (None, 'sigen_plant_battery_state_of_health', ['value', 'unit_of_measurement_str']),
}


EMS_SOURCES = {
    **{name: (None, entity, ['value', 'forecasts_str']) for name, entity in (
        ('mpc_grid', 'mpc_p_grid_forecast'), ('mpc_load', 'mpc_p_load_forecast'),
        ('mpc_pv', 'mpc_p_pv_forecast'), ('mpc_hybrid', 'mpc_p_hybrid_inverter'),
        ('mpc_curtailment', 'mpc_p_pv_curtailment'))},
    **{name: (None, entity, ['value', 'state', 'value_str']) for name, entity in (
        ('ems_action', 'emhass_battery_action'), ('ems_mode', 'sigen_plant_remote_ems_control_mode'),
        ('grid_status', 'sigen_plant_grid_connection_status'),
        ('grid_export_limit', 'sigen_plant_grid_export_limitation'),
        ('pcs_export_limit', 'sigen_plant_pcs_export_limitation'),
        ('discharge_limit', 'sigen_plant_ess_max_discharging_limit'),
        ('desired_export_limit', 'desired_export_limit'),
        ('flexible_export_limit', 'flexible_export_limit'),
        ('transient_pcs_cap', 'transient_pcs_export_cap'),
        ('export_ramp_timer', 'sigen_export_ramp'),
        ('minimum_export_soc', 'battery_soc_min_export'),
        ('effective_feed', 'amber_effective_feed_in_price'))},
}


ENERGY_SOURCES = {
    **{name: (None, entity, ['value', 'unit_of_measurement_str']) for name, entity in (
        ('inverter_ac', 'sigen_inverter_active_power'),
        ('pv1', 'sigen_inverter_pv1_power'), ('pv2', 'sigen_inverter_pv2_power'),
        ('available_charge', 'sigen_plant_available_max_charging_capacity'),
        ('available_discharge', 'sigen_plant_available_max_discharging_capacity'),
        ('derived_capacity', 'sigen_plant_battery_capactiy_derived'),
        ('bms_soc', 'sigen_plant_battery_state_of_charge'))},
    **{key: DH_SOURCES[key] for key in ('rated_capacity', 'battery_health')},
}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-dh-inputs', action='store_true', help='include strategic forecast/settings lineage')
    parser.add_argument('--include-ems-inputs', action='store_true', help='include published controller curves and device limits/modes')
    parser.add_argument('--include-energy-balance', action='store_true', help='include AC/DC and available battery capacity reconciliation')
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
    sources = dict(SOURCES)
    if args.include_dh_inputs: sources.update(DH_SOURCES)
    if args.include_ems_inputs: sources.update(EMS_SOURCES)
    if args.include_energy_balance: sources.update(ENERGY_SOURCES)
    raw, queries, schema, resolved = {}, {}, {}, {}
    try:
        for measurement in sorted({source[0] for source in sources.values() if source[0]}):
            schema[measurement] = {row['fieldKey']: row['fieldType'] for row in
                client.query(f'SHOW FIELD KEYS FROM "{measurement}"').get_points()}
        for name, (measurement, entity, requested) in sources.items():
            if measurement is None:
                keys = list(client.query(f'SHOW SERIES WHERE "entity_id" = \'{entity}\'').get_points())
                available = sorted({row['key'].split(',', 1)[0] for row in keys
                    if row['key'].split(',', 1)[0] in ('sensor', 'number', 'select', 'input_number', 'input_text', 'timer')
                    or row['key'].split(',', 1)[0].startswith(('sensor__', 'number__', 'input_number__'))})
                matching = []
                for candidate in available:
                    if candidate not in schema:
                        schema[candidate] = {row['fieldKey']: row['fieldType'] for row in
                            client.query(f'SHOW FIELD KEYS FROM "{candidate}"').get_points()}
                    if any(field in schema[candidate] for field in requested): matching.append(candidate)
                if len(matching) != 1:
                    raw[name] = []
                    resolved[name] = {'entity': entity, 'status': 'missing_or_ambiguous_raw_measurement', 'candidates': matching}
                    continue
                measurement = matching[0]
            resolved[name] = {'entity': entity, 'measurement': measurement}
            fields = [field for field in requested if field in schema[measurement]]
            if not fields:
                raw[name] = []
                continue
            columns = ','.join('"'+field+'"' for field in fields)
            query = (f'SELECT {columns} FROM "rp_raw"."{measurement}" '
                f'WHERE "entity_id" = \'{entity}\' AND time >= \'{start.isoformat()}\' '
                f'AND time < \'{end.isoformat()}\' ORDER BY time ASC LIMIT 10001')
            # Stable settings/forecasts may not emit inside a short window.
            # Retain the latest prior record; downstream admission controls age.
            prior_query = (f'SELECT {columns} FROM "rp_raw"."{measurement}" '
                f'WHERE "entity_id" = \'{entity}\' AND time < \'{start.isoformat()}\' ORDER BY time DESC LIMIT 1')
            queries[name] = {'window': query, 'prior': prior_query}
            raw[name] = list(client.query(query).get_points())
            if len(raw[name]) > 10000:
                raise ValueError('source exceeds bounded row budget: '+name)
            raw[name] = list(client.query(prior_query).get_points())+raw[name]
            print(json.dumps({'source': name, 'rows': len(raw[name])}), flush=True)
    finally:
        client.close()
    args.output.mkdir(parents=True)
    archive = args.output/'history.json'
    archive.write_text(json.dumps(raw, indent=2, allow_nan=False)+'\n')
    manifest = {'mode': 'read_only_control_fidelity_archive', 'start': start.isoformat(),
        'end': end.isoformat(), 'queries': queries, 'sources': sources, 'resolved_sources': resolved,
        'selected_schema': {key: {field: schema.get(resolved[key].get('measurement'), {}).get(field) for field in fields}
            for key, (_, _, fields) in sources.items()},
        'history_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'exporter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'export_complete': True, 'publication_authorized': False,
        'limitations': ['Influx event time is not a proven solver-input capture timestamp',
            'independently recorded entities are not an atomic solver input snapshot',
            'published plan revisions can include state/attribute publication transitions',
            'a plan publication is not proof of a separate successful solve',
            'latest prior record included; downstream admission must enforce source-specific age/coverage',
            'missing historical state remains missing; no current-state fallback']}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
