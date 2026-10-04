"""Economic mechanism checks; these do not validate historical load calibration skill."""
from copy import deepcopy
import math

import pytest

from eval.controlled_cycle_scenarios import run_arm
from eval.controlled_load_information import ARMS, objective, scenario, select_export, suite


@pytest.fixture(scope='module')
def results():
    return suite()


def row(results, stock, load, solar=False):
    return next(s for s in results['scenarios'] if (s['stock'], s['future_demand'],
                s['otherwise_curtailed_later_solar']) == (stock, load, solar))


def test_all_arms_execute_identical_actuals_without_resets(results):
    for case in results['scenarios']:
        for name in ARMS:
            arm = case['arms'][name]
            phases = deepcopy(case['common_realized_phases'])
            phases[0]['export_cap_kw'] = arm['decision']['export_cap_kw']
            reproduced = run_arm(phases, case['initial_soc'], phases[0]['export_cap_kw'] > 0)
            assert arm['variable_cost_aud'] == reproduced['variable_cost_aud']
            assert arm['ending_inventory_kwh'] == reproduced['ending_inventory_kwh']
            assert arm['initial_soc'] == case['initial_soc']
            for previous, following in zip(arm['phases'], arm['phases'][1:]):
                assert following['start_soc'] == previous['end_soc']


def test_overprediction_withholds_profitable_export(results):
    case = row(results, 'scarce', 'low')
    assert case['arms']['incumbent']['decision']['export_cap_kw'] == 0
    assert case['arms']['modest_correction']['decision']['export_cap_kw'] > 0
    assert case['comparisons']['modest_correction']['net_value_aud_at_decision_assumptions'] > 0
    assert case['comparisons']['modest_correction']['ending_inventory_delta_kwh'] < 0


def test_underprediction_causes_expensive_import_with_equal_endpoint(results):
    case = row(results, 'scarce', 'high')
    base, corrected = (case['arms'][name] for name in ('incumbent', 'modest_correction'))
    assert base['grid_import_kwh'] > corrected['grid_import_kwh']
    assert base['ending_inventory_kwh'] == pytest.approx(corrected['ending_inventory_kwh'])
    assert case['comparisons']['modest_correction']['net_value_aud_at_decision_assumptions'] > 0
    assert corrected['phases'][-1]['floor_binding_steps'] > 0


def test_worsened_bias_can_hurt(results):
    case = row(results, 'scarce', 'high')
    assert case['comparisons']['worsened_bias']['net_value_aud_at_decision_assumptions'] < 0
    assert case['arms']['worsened_bias']['absolute_future_load_error_kwh'] > case['arms']['incumbent']['absolute_future_load_error_kwh']


@pytest.mark.parametrize('load', ['low', 'high'])
def test_ample_energy_makes_load_forecast_irrelevant(results, load):
    case = row(results, 'ample', load)
    assert {a['decision']['export_cap_kw'] for a in case['arms'].values()} == {2.0}
    for comparison in case['comparisons'].values():
        assert comparison['net_value_aud_at_decision_assumptions'] == 0


def test_later_solar_can_remove_reservation_value(results):
    case = row(results, 'scarce', 'high', True)
    for comparison in case['comparisons'].values():
        assert comparison['net_value_aud_at_decision_assumptions'] == pytest.approx(0, abs=1e-10)
    full = row(results, 'ample', 'high', True)
    assert full['arms']['incumbent']['phases'][1]['capacity_binding_steps'] > 0
    assert full['arms']['incumbent']['curtailed_pv_dc_kwh'] > 0


def test_oracle_is_policy_class_upper_bound_at_stated_values(results):
    for case in results['scenarios']:
        oracle = objective(case['arms']['perfect_load_oracle'])
        for arm in case['arms'].values():
            assert oracle <= objective(arm) + 1e-10


def test_zero_overhead_equal_endpoint_cash_matches_analytic_energy_value():
    case = scenario('scarce', 'high', False, dc_fixed_loss_w=0.)
    base, corrected = (case['arms'][name] for name in ('incumbent', 'modest_correction'))
    assert base['ending_inventory_kwh'] == pytest.approx(corrected['ending_inventory_kwh'])
    analytic = ((base['grid_import_kwh']-corrected['grid_import_kwh'])*.50
                - (base['grid_export_kwh']-corrected['grid_export_kwh'])*.30)
    assert analytic > 0
    assert case['comparisons']['modest_correction']['variable_credit_gain_aud'] == pytest.approx(analytic)
    assert base['dc_throughput_kwh'] == pytest.approx(corrected['dc_throughput_kwh'])


def test_terminal_sensitivity_can_reverse_low_load_benefit(results):
    values = row(results, 'scarce', 'low')['comparisons']['modest_correction']['value_sensitivity']
    assert next(v['net_value_aud'] for v in values if v['terminal_value_aud_per_kwh'] == .20 and v['wear_aud_per_dc_kwh'] == .04) > 0
    assert next(v['net_value_aud'] for v in values if v['terminal_value_aud_per_kwh'] == .50 and v['wear_aud_per_dc_kwh'] == .04) < 0


def test_decision_does_not_mutate_forecast(results):
    case = row(results, 'scarce', 'low')
    forecast = deepcopy(case['common_realized_phases'])
    saved = deepcopy(forecast)
    select_export(forecast, case['initial_soc'])
    assert forecast == saved


@pytest.mark.parametrize('value', [-1., math.inf, math.nan])
def test_invalid_value_assumptions_rejected(results, value):
    with pytest.raises(ValueError):
        objective(row(results, 'scarce', 'low')['arms']['incumbent'], terminal_value=value)


def test_scope_and_json_are_explicit(results):
    import json
    json.dumps(results, allow_nan=False)
    assert not results['publication_authorized']
    assert 'synthetic' in results['scope']
    assert len(results['scenarios']) == 8
    assert any('not a calibration replay' in line for line in results['limitations'])
