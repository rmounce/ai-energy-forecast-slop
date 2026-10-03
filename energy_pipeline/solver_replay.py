"""Historical, isolated solver requests/results; never live admission or publication."""
from copy import deepcopy
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd


POWER_PRICE_KEYS = ('load_power_forecast', 'pv_power_forecast',
                    'load_cost_forecast', 'prod_price_forecast')
RETRIEVAL_KEYS = ('optimization_time_step', 'time_zone',
                  'sensor_power_photovoltaics', 'sensor_power_load_no_var_loads')
PLANT_OVERRIDES = ('battery_minimum_state_of_charge', 'battery_nominal_energy_capacity')
OPTIM_OVERRIDES = ('weight_battery_charge', 'weight_battery_discharge')
PAYLOAD_KEYS = frozenset(POWER_PRICE_KEYS + PLANT_OVERRIDES + OPTIM_OVERRIDES + (
    'soc_init', 'soc_final', 'optimization_time_step', 'prediction_horizon',
    'delta_forecast_daily', 'entity_save', 'publish_prefix'))
RESULT_COLUMNS = ('P_Load', 'P_PV', 'P_batt', 'P_grid_pos', 'P_grid_neg', 'SOC_opt')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} requires a finite number')
    return value


def prepare_request(record, config, *, kind, optimization_sha256):
    """Freeze a recorded payload and credential-free config for a historical rehearsal.

    Capture-time readiness is checked, never renewed. Source freshness, original-time
    plant config and operational validity are deliberately not asserted by this API.
    """
    if kind not in ('dh', 'mpc'):
        raise ValueError('unsupported solve kind')
    if not re.fullmatch(r'[0-9a-f]{64}', optimization_sha256):
        raise ValueError('require pinned optimization source SHA-256')
    ready = record['readiness'][kind]
    if not ready['coverage_ready'] or ready['reasons']:
        raise ValueError('recorded handoff coverage is not ready')
    payload = deepcopy(record['payloads'][kind])
    unexpected = set(payload) - PAYLOAD_KEYS
    if unexpected:
        raise ValueError(f'unsupported solver payload fields: {sorted(unexpected)}')
    step, count = (30, 144) if kind == 'dh' else (5, 168)
    if payload['optimization_time_step'] != step or payload['prediction_horizon'] != count:
        raise ValueError('unexpected recorded solve horizon')
    for key in POWER_PRICE_KEYS:
        values = payload[key]
        if not isinstance(values, list) or len(values) != count:
            raise ValueError(f'{key}: wrong horizon')
        for value in values:
            _number(value, key)
            if 'power' in key and value < 0:
                raise ValueError(f'{key}: negative power')
    for key in PLANT_OVERRIDES + OPTIM_OVERRIDES + ('soc_init', 'soc_final'):
        _number(payload[key], key)
    for key in ('soc_init', 'soc_final', 'battery_minimum_state_of_charge'):
        if not 0 <= payload[key] <= 1:
            raise ValueError(f'{key}: outside SoC fractions')
    if payload['battery_nominal_energy_capacity'] <= 0:
        raise ValueError('nonpositive battery capacity')
    if set(config) != {'retrieve_hass_conf', 'optim_conf', 'plant_conf'}:
        raise ValueError('require only retrieval, optimisation and plant configuration')
    if set(config['retrieve_hass_conf']) != set(RETRIEVAL_KEYS):
        raise ValueError('retrieval config must contain only selected noncredential fields')
    # Solver configuration is numeric/policy data; reject credential-bearing keys
    # recursively rather than relying on a caller to strip them.
    def check_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if re.search(r'token|password|secret|api_key|hass_url|influxdb', key, re.I):
                    raise ValueError('credential/network configuration forbidden')
                check_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_keys(item)
    check_keys(config)
    effective = deepcopy(config)
    if effective['optim_conf']['number_of_deferrable_loads'] != 0:
        raise ValueError('deferrable solver loads not supported; HWC must be in recorded load')
    for key in PLANT_OVERRIDES:
        effective['plant_conf'][key] = payload[key]
    for key in OPTIM_OVERRIDES:
        effective['optim_conf'][key] = payload[key]
    effective['retrieve_hass_conf']['optimization_time_step'] = step
    effective['optim_conf']['delta_forecast_daily'] = payload['delta_forecast_daily']
    # Resource bound only; separately reported from production's thread setting.
    effective['optim_conf']['num_threads'] = 1
    effective['optim_conf']['lp_solver_timeout'] = min(
        float(effective['optim_conf'].get('lp_solver_timeout', 45)), 45)
    if effective['optim_conf']['lp_solver_timeout'] <= 0:
        raise ValueError('nonpositive solver timeout')
    plant = effective['plant_conf']
    if plant.get('inverter_is_hybrid') and any(
        not isinstance(plant.get(key), (int, float)) or plant[key] <= 0
        for key in ('inverter_ac_input_max', 'inverter_ac_output_max')
    ):
        raise ValueError('require explicit inverter limits; no model-file lookup')
    at = pd.Timestamp(record['captured_at'])
    if at.tzinfo is None:
        raise ValueError('capture timestamp requires timezone')
    start = at.tz_convert('UTC').floor(f'{step}min')
    request = {'schema': 1, 'mode': 'historical_solver_replay', 'kind': kind,
               'captured_at': at.isoformat(), 'forecast_start': start.isoformat(),
               'publication_id': record['publication_id'],
               'handoff_revision': digest(record), 'config_revision': digest(config),
               'optimization_sha256': optimization_sha256,
               'configuration': effective, 'payload': payload,
               'resource_overrides': {'num_threads': 1,
                                      'lp_solver_timeout': effective['optim_conf']['lp_solver_timeout']}}
    request['request_id'] = digest(request)
    return request


def validate_result(request, result):
    """Check a historical result; never mint an AcceptedDHSolve/live authority."""
    if result['request_id'] != request['request_id']:
        raise ValueError('solver result belongs to another request')
    if result['optimization_sha256'] != request['optimization_sha256']:
        raise ValueError('solver source differs from pinned source')
    if result['status'] != 'Optimal':
        raise ValueError('solver result is not exactly Optimal')
    payload = request['payload']
    frame = pd.DataFrame(result['values'], columns=result['columns'],
                         index=pd.to_datetime(result['targets'], utc=True))
    expected = pd.date_range(request['forecast_start'], periods=payload['prediction_horizon'],
                             freq=f"{payload['optimization_time_step']}min")
    if not frame.index.equals(expected) or not set(RESULT_COLUMNS) <= set(frame):
        raise ValueError('solver result interval/column coverage differs from request')
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite solver output')
    tolerance = 1e-5
    for key, column in (('load_power_forecast', 'P_Load'), ('pv_power_forecast', 'P_PV')):
        if not np.allclose(frame[column], payload[key], rtol=0, atol=tolerance):
            raise ValueError('solver changed input load/PV')
    if (frame['P_grid_pos'] < -tolerance).any() or (frame['P_grid_neg'] > tolerance).any():
        raise ValueError('solver grid signs invalid')
    plant = request['configuration']['plant_conf']
    for key, column in (('load_cost_forecast', 'unit_load_cost'),
                        ('prod_price_forecast', 'unit_prod_price')):
        if column not in frame or not np.allclose(frame[column], payload[key], rtol=0,
                                                  atol=tolerance):
            raise ValueError('solver changed input tariff')
    imported, exported = frame['P_grid_pos'], -frame['P_grid_neg']
    if ((imported > tolerance) & (exported > tolerance)).any():
        raise ValueError('simultaneous grid import/export')
    for power, key in ((imported, 'maximum_power_from_grid'),
                       (exported, 'maximum_power_to_grid')):
        if (power > np.asarray(plant.get(key, 9000)) + tolerance).any():
            raise ValueError('solver exceeded grid limit')
    curtailed = frame.get('P_PV_curtailment', pd.Series(0., index=frame.index))
    if (curtailed < -tolerance).any() or (curtailed > frame['P_PV']+tolerance).any():
        raise ValueError('invalid PV curtailment')
    if not plant.get('compute_curtailment', False) and (curtailed > tolerance).any():
        raise ValueError('unexpected PV curtailment')
    dc = frame['P_PV'] - curtailed + frame['P_batt']
    if plant['inverter_is_hybrid']:
        ac = np.where(dc >= 0, dc * plant.get('inverter_efficiency_dc_ac', 1.),
                      dc / plant.get('inverter_efficiency_ac_dc', 1.))
        if 'P_hybrid_inverter' not in frame or not np.allclose(
                frame['P_hybrid_inverter'], ac, rtol=0, atol=tolerance):
            raise ValueError('solver hybrid DC/AC conversion differs')
        if (ac > plant['inverter_ac_output_max']+tolerance).any() or (
                ac < -plant['inverter_ac_input_max']-tolerance).any():
            raise ValueError('solver exceeded inverter limit')
    else:
        ac = dc
    if not np.allclose(ac + imported - exported, frame['P_Load'], rtol=0,
                       atol=tolerance):
        raise ValueError('solver AC power balance differs')
    soc = frame['SOC_opt'].to_numpy(dtype=float)
    low = min(payload['soc_init'], plant['battery_minimum_state_of_charge'])
    high = max(payload['soc_init'], plant['battery_maximum_state_of_charge'])
    if (soc < low-tolerance).any() or (soc > high+tolerance).any():
        raise ValueError('solver SoC outside physical/recovery bounds')
    # EMHASS P_batt > 0 is DC discharge; SOC_opt is END-of-interval.
    batt = frame['P_batt'].to_numpy(dtype=float)
    delta = np.where(batt >= 0, batt / plant['battery_discharge_efficiency'],
                     batt * plant['battery_charge_efficiency'])
    reconstructed = payload['soc_init'] - np.cumsum(delta) * (
        payload['optimization_time_step'] / 60 / plant['battery_nominal_energy_capacity'])
    if not np.allclose(soc, reconstructed, rtol=0, atol=tolerance):
        raise ValueError('solver SoC transition convention differs')
    if abs(soc[-1]-payload['soc_final']) > tolerance:
        raise ValueError('solver terminal SoC differs from requested endpoint')
    return frame


def forecast_summary(request, frame):
    """Forecast economics only: no realised outcomes, savings or oracle attribution."""
    payload = request['payload']
    hours = payload['optimization_time_step'] / 60
    imported = frame['P_grid_pos'].to_numpy() * hours / 1000
    exported = -frame['P_grid_neg'].to_numpy() * hours / 1000
    import_cost = float(np.dot(imported, payload['load_cost_forecast']))
    export_revenue = float(np.dot(exported, payload['prod_price_forecast']))
    return {'scope': 'forecast_cashflow_only', 'steps': len(frame),
            'import_kwh': float(imported.sum()), 'export_kwh': float(exported.sum()),
            'import_cost_aud': import_cost, 'export_revenue_aud': export_revenue,
            'cashflow_aud': export_revenue-import_cost,
            'initial_soc': payload['soc_init'], 'final_soc': float(frame['SOC_opt'].iloc[-1]),
            'battery_throughput_kwh': float(frame['P_batt'].abs().sum()*hours/1000)}
