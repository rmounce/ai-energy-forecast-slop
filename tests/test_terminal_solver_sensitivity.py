from copy import deepcopy

import numpy as np
import pytest

from energy_pipeline.solver_replay import digest
from eval.compare_terminal_solver_sensitivity import compare, terminal_request


def artifact():
    request = {'forecast_start': '2026-10-03T00:00:00+00:00', 'kind': 'dh',
        'handoff_revision': 'original', 'optimization_sha256': 'a'*64,
        'payload': {'prediction_horizon': 2, 'optimization_time_step': 30,
            'soc_init': .5, 'soc_final': .5, 'load_power_forecast': [1000., 1000.],
            'pv_power_forecast': [0., 0.], 'load_cost_forecast': [.2, .2],
            'prod_price_forecast': [.1, .1]},
        'configuration': {'plant_conf': {'battery_nominal_energy_capacity': 10000,
            'battery_minimum_state_of_charge': .1, 'battery_maximum_state_of_charge': 1.,
            'battery_charge_efficiency': 1., 'battery_discharge_efficiency': 1.,
            'inverter_is_hybrid': False}}}
    request['request_id'] = digest(request)
    columns = ['P_Load', 'P_PV', 'P_batt', 'P_grid_pos', 'P_grid_neg', 'SOC_opt',
               'unit_load_cost', 'unit_prod_price']
    result = {'request_id': request['request_id'], 'optimization_sha256': 'a'*64,
        'status': 'Optimal', 'columns': columns,
        'targets': ['2026-10-03T00:00:00+00:00', '2026-10-03T00:30:00+00:00'],
        'values': [[1000., 0., 0., 1000., 0., .5, .2, .1]]*2}
    return {'request': request, 'result': result, 'image': 'sha256:'+'b'*64,
            'publication_authorized': False}


def charge_challenger(baseline):
    challenger = deepcopy(baseline)
    challenger['request'] = terminal_request(baseline, 60)
    challenger['result']['request_id'] = challenger['request']['request_id']
    challenger['result']['values'] = [[1000., 0., -2000., 3000., 0., .6, .2, .1],
                                      [1000., 0., 0., 1000., 0., .6, .2, .1]]
    return challenger


def test_endpoint_change_preserves_inputs_and_creates_distinct_identity():
    baseline = artifact()
    original = deepcopy(baseline)
    request = terminal_request(baseline, 60)
    assert baseline == original
    assert request['request_id'] != baseline['request']['request_id']
    assert request['payload']['soc_init'] == .5
    assert request['payload']['soc_final'] == .6
    assert request['payload']['load_power_forecast'] == [1000., 1000.]


@pytest.mark.parametrize('endpoint', [True, np.nan, np.inf, -1, 9, 101])
def test_invalid_endpoint_rejected(endpoint):
    with pytest.raises(ValueError): terminal_request(artifact(), endpoint)


def test_cashflow_reduction_for_stored_energy_is_not_loss_ranking():
    baseline = artifact()
    result = compare(baseline, charge_challenger(baseline))
    assert result['ending_inventory_delta_kwh'] == pytest.approx(1)
    assert result['cashflow_delta_aud'] == pytest.approx(-.2)
    assert result['inventory_value_break_even_aud_per_kwh'] == pytest.approx(.2)
    assert result['battery_charge_delta_kwh'] == pytest.approx(1)
    assert result['battery_discharge_delta_kwh'] == 0
    assert result['first_battery_discharge_delta_w'] == -2000


def test_changed_forecast_rejected_even_when_result_matches_modified_request():
    baseline = artifact()
    challenger = charge_challenger(baseline)
    challenger['request']['payload']['prod_price_forecast'][0] = .9
    challenger['result']['values'][0][-1] = .9
    challenger['request']['request_id'] = digest({k: v for k, v in challenger['request'].items() if k != 'request_id'})
    challenger['result']['request_id'] = challenger['request']['request_id']
    with pytest.raises(ValueError, match='more than terminal'):
        compare(baseline, challenger)


def test_tampered_baseline_or_authorised_challenger_rejected():
    baseline = artifact()
    challenger = charge_challenger(baseline)
    challenger['publication_authorized'] = True
    with pytest.raises(ValueError, match='provenance'): compare(baseline, challenger)
    baseline['request']['payload']['soc_final'] = .9
    with pytest.raises(ValueError, match='identity'): terminal_request(baseline, 60)
