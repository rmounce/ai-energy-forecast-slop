from copy import deepcopy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from energy_pipeline.solver_replay import prepare_request, validate_result, forecast_summary


HASH = 'a'*64


@pytest.fixture
def inputs():
    payload = {'load_power_forecast': [1000.]*144, 'pv_power_forecast': [0.]*144,
               'load_cost_forecast': [.2]*144, 'prod_price_forecast': [.1]*144,
               'soc_init': .5, 'soc_final': .5, 'optimization_time_step': 30,
               'prediction_horizon': 144, 'delta_forecast_daily': 3,
               'battery_minimum_state_of_charge': .15, 'battery_nominal_energy_capacity': 40300,
               'weight_battery_charge': 0, 'weight_battery_discharge': .04,
               'entity_save': True, 'publish_prefix': 'dh_'}
    record = {'captured_at': '2026-10-03T01:51:09+00:00', 'publication_id': 'historical',
              'readiness': {'dh': {'coverage_ready': True, 'reasons': []}},
              'payloads': {'dh': payload}}
    config = {'retrieve_hass_conf': {'optimization_time_step': 30, 'time_zone': 'Etc/UTC',
                                    'sensor_power_photovoltaics': '',
                                    'sensor_power_load_no_var_loads': 'load'},
              'optim_conf': {'number_of_deferrable_loads': 0, 'num_threads': 0,
                             'lp_solver_timeout': 100, 'delta_forecast_daily': 1},
              'plant_conf': {'battery_nominal_energy_capacity': 420000,
                             'battery_minimum_state_of_charge': .1,
                             'battery_maximum_state_of_charge': 1,
                             'battery_charge_efficiency': .99,
                             'battery_discharge_efficiency': .99,
                             'inverter_is_hybrid': True,
                             'inverter_efficiency_dc_ac': .95,
                             'inverter_efficiency_ac_dc': .95,
                             'maximum_power_from_grid': 15000,
                             'maximum_power_to_grid': 10000,
                             'inverter_ac_input_max': 9980, 'inverter_ac_output_max': 9980}}
    return record, config


def make_request(inputs):
    return prepare_request(*inputs, kind='dh', optimization_sha256=HASH)


def result(request):
    frame = pd.DataFrame({'P_Load': [1000.]*144, 'P_PV': [0.]*144,
                          'P_batt': [0.]*144, 'P_grid_pos': [1000.]*144,
                          'P_grid_neg': [0.]*144, 'SOC_opt': [.5]*144,
                          'P_hybrid_inverter': [0.]*144,
                          'unit_load_cost': [.2]*144, 'unit_prod_price': [.1]*144},
                         index=pd.date_range(request['forecast_start'], periods=144, freq='30min'))
    return {'request_id': request['request_id'], 'optimization_sha256': HASH, 'status': 'Optimal',
            'targets': [at.isoformat() for at in frame.index], 'columns': list(frame),
            'values': frame.to_numpy().tolist()}


def test_runtime_capacity_and_weights_override_static_config_without_mutation(inputs):
    original = deepcopy(inputs)
    req = make_request(inputs)
    assert inputs == original
    assert req['configuration']['plant_conf']['battery_nominal_energy_capacity'] == 40300
    assert req['configuration']['optim_conf']['weight_battery_discharge'] == .04
    assert req['configuration']['optim_conf']['lp_solver_timeout'] == 45
    assert req['forecast_start'] == '2026-10-03T01:30:00+00:00'
    assert req['request_id'] == make_request(inputs)['request_id']
    inputs[0]['payloads']['dh']['load_cost_forecast'][0] = -.1
    assert make_request(inputs)['request_id'] != req['request_id']


@pytest.mark.parametrize('bad', ['coverage', 'extra_payload', 'secret', 'network', 'nan',
                                 'power', 'horizon', 'deferrable', 'inverter'])
def test_incomplete_or_unsupported_inputs_rejected(inputs, bad):
    record, config = inputs
    if bad == 'coverage': record['readiness']['dh']['coverage_ready'] = False
    elif bad == 'extra_payload': record['payloads']['dh']['maximum_power_to_grid'] = 0
    elif bad == 'secret': config['optim_conf']['nested'] = {'long_lived_token': 'never-export'}
    elif bad == 'network': config['retrieve_hass_conf']['hass_url'] = 'http://ha'
    elif bad == 'nan': record['payloads']['dh']['load_cost_forecast'][0] = float('nan')
    elif bad == 'power': record['payloads']['dh']['pv_power_forecast'][0] = -1
    elif bad == 'horizon': record['payloads']['dh']['load_power_forecast'].pop()
    elif bad == 'deferrable': config['optim_conf']['number_of_deferrable_loads'] = 1
    else: config['plant_conf'].pop('inverter_ac_input_max')
    with pytest.raises(ValueError): make_request(inputs)


def test_forecast_cashflow_is_not_realised_profit(inputs):
    request = make_request(inputs)
    frame = validate_result(request, result(request))
    summary = forecast_summary(request, frame)
    assert summary['scope'] == 'forecast_cashflow_only'
    assert summary['import_kwh'] == 72
    assert summary['cashflow_aud'] == pytest.approx(-14.4)


@pytest.mark.parametrize('bad', ['parent', 'source', 'status', 'time', 'power', 'soc',
                                 'terminal', 'balance', 'tariff', 'conversion', 'grid_limit',
                                 'simultaneous_grid', 'inverter_limit', 'curtailment'])
def test_invalid_results_never_count_as_replay_success(inputs, bad):
    request = make_request(inputs)
    output = result(request)
    if bad == 'parent': output['request_id'] = 'other'
    elif bad == 'source': output['optimization_sha256'] = 'b'*64
    elif bad == 'status': output['status'] = 'Optimal_Inaccurate'
    elif bad == 'time': output['targets'][0] = '2026-10-03T01:00:00+00:00'
    elif bad == 'power': output['values'][0][0] = 999
    elif bad == 'soc': output['values'][0][5] = .6
    elif bad == 'terminal': request['payload']['soc_final'] = .6
    elif bad == 'balance': output['values'][0][3] = 1100
    elif bad == 'tariff': output['values'][0][7] = .3
    elif bad == 'conversion': output['values'][0][6] = 100
    elif bad == 'grid_limit': request['configuration']['plant_conf']['maximum_power_from_grid'] = 900
    elif bad == 'simultaneous_grid':
        output['values'][0][3] = 1100
        output['values'][0][4] = -100
    elif bad == 'inverter_limit':
        request['configuration']['plant_conf']['inverter_ac_output_max'] = -1
    else:
        output['columns'].append('P_PV_curtailment')
        for row in output['values']: row.append(1.)
    with pytest.raises(ValueError): validate_result(request, output)


def test_discharge_positive_and_soc_end_of_interval(inputs):
    request = make_request(inputs)
    output = result(request)
    # One stored-energy-neutral charge/discharge pair with unequal efficiencies.
    output['values'][0][2] = -1000
    output['values'][1][2] = 1000*.99*.99
    output['values'][0][5] = .5+1000*.99*.5/40300
    output['values'][0][6] = -1000/.95
    output['values'][0][3] = 1000+1000/.95
    output['values'][1][6] = 1000*.99*.99*.95
    output['values'][1][3] = 1000-1000*.99*.99*.95
    frame = validate_result(request, output)
    assert frame['SOC_opt'].iloc[0] > request['payload']['soc_init']
    assert frame['SOC_opt'].iloc[-1] == request['payload']['soc_final']


def test_container_has_no_production_mounts_or_network():
    path = Path(__file__).resolve().parents[2]/'scripts/replay_energy_solves.py'
    spec = importlib.util.spec_from_file_location('replay_energy_solves', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cmd = module.container_command('sha256:'+HASH, '/tmp/private-request')
    assert cmd[cmd.index('--network')+1] == 'none'
    assert '--read-only' in cmd and '--user' in cmd
    mounts = [cmd[i+1] for i, value in enumerate(cmd) if value == '--mount']
    assert mounts == ['type=bind,src=/tmp/private-request/worker.py,dst=/work/worker.py,readonly',
                      'type=bind,src=/tmp/private-request/request.json,dst=/work/request.json,readonly']
    assert '/var/run/docker.sock' not in ' '.join(cmd)
    with pytest.raises(ValueError): module.container_command('latest', '/tmp/request')
