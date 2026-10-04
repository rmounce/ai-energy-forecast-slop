from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from eval.sequential_core_replay import execute, snapshot_at, simulate, decision_bundle_identity


@pytest.fixture
def plant():
    return {'battery_nominal_energy_capacity': 40300, 'battery_minimum_state_of_charge': .15,
        'battery_maximum_state_of_charge': 1., 'battery_charge_efficiency': .99,
        'battery_discharge_efficiency': .99, 'inverter_efficiency_dc_ac': .95,
        'inverter_efficiency_ac_dc': .95, 'battery_charge_power_max': 20000,
        'battery_discharge_power_max': 20000, 'inverter_ac_output_max': 9980,
        'inverter_ac_input_max': 9980, 'maximum_power_from_grid': 15000,
        'maximum_power_to_grid': 10000, 'inverter_is_hybrid': True}


def test_hybrid_execution_conserves_dc_soc_and_ac_grid(plant):
    result = execute(plant, .5, -6000, 0, 5000, 1000, 10000)
    assert result['end_soc'] == pytest.approx(.5+6000*.99*(5/60)/40300)
    assert result['inverter_ac_w'] == pytest.approx(-1000/.95)
    assert result['grid_import_kwh'] == pytest.approx((1000+1000/.95)*(5/60)/1000)
    assert result['grid_export_kwh'] == 0
    assert not result['command_clipped']


def test_negative_export_guard_curtailed_supply_and_excess_battery_discharge(plant):
    result = execute(plant, .5, 2000, 0, 5000, 1000, 0)
    assert result['curtailed_pv_w'] == 5000
    assert result['battery_discharge_w'] == pytest.approx(1000/.95)
    assert result['grid_export_kwh'] == pytest.approx(0)
    assert result['command_clipped']


def test_energy_capacity_and_inverter_bounds_without_soc_clipping(plant):
    result = execute(plant, .9999, -20000, 0, 0, 1000, 10000)
    assert result['end_soc'] == pytest.approx(1.)
    result = execute(plant, .5, -20000, 0, 0, 1000, 10000)
    assert result['inverter_ac_w'] == pytest.approx(-9980)
    assert result['grid_import_kwh']*12000 == pytest.approx(10980)
    assert result['command_clipped']


def test_unservable_load_fails_instead_of_silent_energy_creation(plant):
    with pytest.raises(ValueError, match='cannot execute'):
        execute(plant, .15, 0, 0, 0, 25000, 10000)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -1])
def test_missing_or_invalid_delivered_pv_rejected(plant, bad):
    with pytest.raises(ValueError): execute(plant, .5, 0, 0, bad, 1000, 10000)


def revision(leg):
    rows = [{'start_time': '2026-10-03T02:00:01Z', 'end_time': '2026-10-03T02:05:00Z',
        'duration': 5, 'per_kwh': .2 if leg == 'general' else -.1,
        'advanced_price_low': .1 if leg == 'general' else -.05,
        'advanced_price_predicted': .2 if leg == 'general' else -.1,
        'advanced_price_high': .3 if leg == 'general' else -.2}]
    return {'receipt': '2026-10-03T01:59:00Z', 'payload_sha256': leg, 'rows': rows}


def test_current_forecast_fallback_is_causal_and_sign_consistent():
    revisions = {leg: [revision(leg)] for leg in ('general', 'feed_in')}
    past = {'pv_dc_w': 5000, 'load_site_w': 1000, 'conversion_loss_w': 100}
    quotes = {leg: [] for leg in ('general', 'feed', 'adjusted_feed')}
    states, provenance = snapshot_at({}, .5, past, '2026-10-03T02:00:00Z', revisions, quotes)
    assert float(states['sensor.amber_adjusted_confirmed_feed_in_price']['state']) == .1
    assert float(states['sensor.amber_5min_current_general_price']['state']) == .2
    assert provenance['general']['current_source'] == 'already_issued_per_kwh_forecast'
    assert states['sensor.amber_5min_forecasts_extended_general_price']['attributes']['Forecasts'] == []
    future = deepcopy(revisions)
    future['general'].append(revision('general') | {'receipt': '2026-10-03T02:01:00Z',
        'payload_sha256': 'future', 'rows': []})
    other, _ = snapshot_at({}, .5, past, '2026-10-03T02:00:00Z', future, quotes)
    assert other == states


def test_unreceived_or_stale_apf_cannot_enter_decision():
    revisions = {leg: [revision(leg)] for leg in ('general', 'feed_in')}
    for origin in ('2026-10-03T01:58:00Z', '2026-10-03T02:15:00Z'):
        with pytest.raises(ValueError, match='missing causal APF'):
            snapshot_at({}, .5, {'pv_dc_w': 0, 'load_site_w': 0, 'conversion_loss_w': 0}, origin, revisions, {})


def test_subminute_origin_uses_current_five_minute_interval_with_causal_receipts():
    revisions = {leg: [revision(leg)] for leg in ('general', 'feed_in')}
    past = {'pv_dc_w': 5000, 'load_site_w': 1000, 'conversion_loss_w': 100}
    quotes = {leg: [] for leg in ('general', 'feed', 'adjusted_feed')}
    states, provenance = snapshot_at({}, .5, past, '2026-10-03T02:01:25Z', revisions, quotes)
    assert float(states['sensor.amber_5min_current_general_price']['state']) == .2
    assert provenance['general']['current_source'] == 'already_issued_per_kwh_forecast'
    assert states['sensor.amber_5min_forecasts_extended_general_price']['attributes']['Forecasts'] == []


def toy_bundle(plant):
    dates = pd.date_range('2026-10-03T02:00:00Z', periods=144, freq='30min')
    parent = {
        'sensor.sigen_plant_rated_energy_capacity': {'state': '40.3'},
        'sensor.sigen_plant_battery_state_of_health': {'state': '100'},
        'input_number.dh_last_soc_init': {'state': '50'},
        'input_number.battery_soc_min_target': {'state': '15'},
        'sensor.dh_soc_batt_forecast': {'attributes': {'battery_scheduled_soc': [
            {'date': date.isoformat(), 'dh_soc_batt_forecast': 50} for date in dates]}},
        'sensor.dh_p_load_forecast': {'attributes': {'forecasts': [
            {'date': date.isoformat(), 'dh_p_load_forecast': 500} for date in dates]}},
        'sensor.dh_p_pv_forecast': {'attributes': {'forecasts': [
            {'date': date.isoformat(), 'dh_p_pv_forecast': 0} for date in dates]}}}
    corrected = deepcopy(parent)
    for row in corrected['sensor.dh_p_load_forecast']['attributes']['forecasts']:
        row['dh_p_load_forecast'] = 400
    apf = {}
    for leg in ('general', 'feed_in'):
        first = revision(leg)
        rows = []
        for date in pd.date_range(dates[0], periods=169, freq='5min'):
            row = deepcopy(first['rows'][0])
            row['start_time'] = (date+pd.Timedelta(seconds=1)).isoformat()
            row['end_time'] = (date+pd.Timedelta(minutes=5)).isoformat()
            rows.append(row)
        apf[leg] = [first | {'rows': rows}]
    configuration = {'plant_conf': plant,
        'retrieve_hass_conf': {'optimization_time_step': 30, 'time_zone': 'Etc/UTC',
            'sensor_power_photovoltaics': '', 'sensor_power_load_no_var_loads': ''},
        'optim_conf': {'number_of_deferrable_loads': 0}}
    actuals = [{'time': '2026-10-03T02:00:00Z', 'pv_dc_w': 0, 'load_site_w': 1000,
        'conversion_loss_w': 0, 'general_rate': .2, 'feed_rate': .1},
        {'time': '2026-10-03T02:05:00Z', 'pv_dc_w': 0, 'load_site_w': 9000,
         'conversion_loss_w': 0, 'general_rate': .2, 'feed_rate': .1}]
    return {'parents': {'baseline': parent, 'calibrated_load': corrected},
        'initial_soc': .5, 'actuals': actuals, 'past': deepcopy(actuals[0]),
        'apf_revisions': apf, 'quote_rows': {leg: [] for leg in ('general', 'feed', 'adjusted_feed')},
        'configuration': configuration, 'optimization_sha256': 'a'*64,
        'source_publication_id': 'historical', 'parent_request_ids': {'baseline': 'original', 'calibrated_load': 'corrected'}}


def neutral_solve(request):
    # Feasible neutral battery plan, independently checked by production result validator.
    p = request['payload']
    n = p['prediction_horizon']
    frame = pd.DataFrame({'P_Load': p['load_power_forecast'], 'P_PV': p['pv_power_forecast'],
        'P_batt': [0.]*n, 'P_grid_pos': p['load_power_forecast'], 'P_grid_neg': [0.]*n,
        'SOC_opt': [p['soc_init']]*n, 'P_hybrid_inverter': [0.]*n,
        'unit_load_cost': p['load_cost_forecast'], 'unit_prod_price': p['prod_price_forecast']},
        index=pd.date_range(request['forecast_start'], periods=n, freq='5min'))
    return {'request_id': request['request_id'], 'optimization_sha256': request['optimization_sha256'],
        'status': 'Optimal', 'solve_seconds': 0., 'columns': list(frame),
        'values': frame.to_numpy().tolist(), 'targets': [date.isoformat() for date in frame.index]}


def test_same_measured_consumption_not_cheaper_predicted_load_scores_both_arms(plant):
    report = simulate(toy_bundle(plant), neutral_solve)
    for arm in ('baseline', 'calibrated_load'):
        assert report['summary'][arm]['variable_cost_aud'] == pytest.approx(1/6)
    assert report['comparison']['cashflow_delta_aud'] == 0
    assert report['comparison']['ending_inventory_delta_kwh'] == 0
    assert report['comparison']['changed_executed_battery_steps'] == 0


def test_future_actual_mutation_does_not_change_current_decision(plant):
    first = toy_bundle(plant)
    second = deepcopy(first)
    second['actuals'][0]['load_site_w'] = 8000
    a, b = simulate(first, neutral_solve), simulate(second, neutral_solve)
    assert a['solves'][0]['request'] == b['solves'][0]['request']
    assert a['summary']['baseline']['variable_cost_aud'] != b['summary']['baseline']['variable_cost_aud']


def test_solver_configuration_mutation_rejected_before_acceptance(plant):
    def bad(request):
        result = neutral_solve(request)
        request['configuration']['retrieve_hass_conf']['optimization_time_step'] = 999
        return result
    with pytest.raises(ValueError, match='mutated frozen request'):
        simulate(toy_bundle(plant), bad)


def test_rescore_can_only_change_financial_labels_not_decisions_or_physics(plant):
    bundle = toy_bundle(plant)
    changed = deepcopy(bundle)
    changed['actuals'][0]['general_rate'] = 999
    assert decision_bundle_identity(bundle) == decision_bundle_identity(changed)
    changed['actuals'][0]['pv_dc_w'] = 999
    assert decision_bundle_identity(bundle) != decision_bundle_identity(changed)
