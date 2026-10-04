from copy import deepcopy

import pytest

from eval.controlled_cycle_scenarios import (ARMS, PLANT, net_value, paired, phase,
                                             scarce_later_energy, solar_replacement, suite)


@pytest.mark.parametrize('exportable', [False, True])
def test_full_replacement_equal_stock_by_capacity_not_reset(exportable):
    row = solar_replacement(1., exportable=exportable)
    assert row['comparison']['ending_inventory_delta_kwh'] == pytest.approx(0, abs=1e-11)
    for arm in row['arms'].values():
        assert arm['final_soc'] == pytest.approx(1.)
        assert arm['phases'][-1]['capacity_binding_steps'] > 0
        assert arm['phases'][1]['start_soc'] == arm['phases'][0]['end_soc']
    gap = row['construction']['initial_extra_stock_spent_kwh']
    expected = gap*(PLANT['battery_discharge_efficiency']+1/PLANT['battery_charge_efficiency'])
    assert row['comparison']['dc_throughput_delta_kwh'] == pytest.approx(expected)
    assert net_value(row['comparison'], 0., .04) == pytest.approx(net_value(row['comparison'], .40, .04))


@pytest.mark.parametrize('fraction', [0., .25, .5, .75, 1., 1.25])
def test_recovery_fraction_physically_closes_stock_gap(fraction):
    row = solar_replacement(fraction)
    gap = row['construction']['initial_extra_stock_spent_kwh']
    assert row['comparison']['ending_inventory_delta_kwh'] == pytest.approx(-gap*max(0., 1-fraction))
    assert row['arms']['baseline']['phases'][-1]['capacity_binding_steps'] > 0
    base, extra = (row['arms'][name] for name in ARMS)
    assert base['curtailed_pv_dc_kwh']-extra['curtailed_pv_dc_kwh'] == pytest.approx(
        gap*min(fraction, 1)/PLANT['battery_charge_efficiency'])


def test_exportable_solar_subtracts_foregone_revenue_not_free_replacement():
    capped = solar_replacement(1., dc_fixed_loss_w=0.)
    exported = solar_replacement(1., exportable=True, later_feed=.10, dc_fixed_loss_w=0.)
    gap = capped['construction']['initial_extra_stock_spent_kwh']
    foregone = gap/PLANT['battery_charge_efficiency']*PLANT['inverter_efficiency_dc_ac']*.10
    assert capped['comparison']['variable_credit_gain_aud']-exported['comparison']['variable_credit_gain_aud'] == pytest.approx(foregone)
    assert exported['arms']['baseline']['phases'][1]['grid_export_kwh'] > 0
    assert exported['arms']['extra_early_export']['phases'][1]['grid_export_kwh'] == pytest.approx(0., abs=1e-10)
    assert exported['arms']['baseline']['curtailed_pv_dc_kwh'] == pytest.approx(0., abs=1e-10)


@pytest.mark.parametrize('exportable', [False, True])
def test_analytic_energy_and_cash_oracle(exportable):
    row = solar_replacement(1., exportable=exportable, early_feed=.25, later_feed=.10, dc_fixed_loss_w=0.)
    d = row['construction']['initial_extra_stock_spent_kwh']
    ed, ec, eta = (PLANT[key] for key in ('battery_discharge_efficiency', 'battery_charge_efficiency', 'inverter_efficiency_dc_ac'))
    cash = d*eta*ed*.25-(d*eta*.10/ec if exportable else 0.)
    assert row['comparison']['variable_credit_gain_aud'] == pytest.approx(cash)
    assert net_value(row['comparison'], .30, .04) == pytest.approx(cash-.04*d*(ed+1/ec))


@pytest.mark.parametrize('early_feed, sign', [(0., -1), (.20, -1), (.40, 0), (.60, 1)])
def test_scarce_floor_reprices_export_to_later_import_and_wear_cancels(early_feed, sign):
    row = scarce_later_energy(early_feed=early_feed, dc_fixed_loss_w=0.)
    base, extra = (row['arms'][name] for name in ARMS)
    assert row['comparison']['ending_inventory_delta_kwh'] == pytest.approx(0., abs=1e-11)
    assert row['comparison']['dc_throughput_delta_kwh'] == pytest.approx(0., abs=1e-11)
    for arm in (base, extra):
        assert arm['phases'][-1]['floor_binding_steps'] > 0
        assert arm['final_soc'] == pytest.approx(PLANT['battery_minimum_state_of_charge'])
    early_gain = extra['phases'][0]['grid_export_kwh']-base['phases'][0]['grid_export_kwh']
    later_import = extra['phases'][1]['grid_import_kwh']-base['phases'][1]['grid_import_kwh']
    assert later_import == pytest.approx(early_gain)
    assert row['comparison']['variable_credit_gain_aud'] == pytest.approx(early_gain*(early_feed-.40))
    net = net_value(row['comparison'], .40, .04)
    if sign == 0:
        assert net == pytest.approx(0., abs=1e-11)
    else:
        assert net*sign > 0


def test_common_inputs_no_mutation_and_stock_stays_inside_bounds():
    before = deepcopy(PLANT)
    row = paired([phase('early_export', 15, export_cap_kw=2.)], .95)
    assert PLANT == before
    assert row['arms']['baseline']['initial_soc'] == row['arms']['extra_early_export']['initial_soc']
    for arm in row['arms'].values():
        assert .15 <= arm['final_soc'] <= 1.
    assert row['comparison']['variable_credit_gain_aud'] > 0
    assert row['comparison']['ending_inventory_delta_kwh'] < 0


@pytest.mark.parametrize('fraction', [-.1, 1.26, float('nan'), float('inf')])
def test_invalid_recovery_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        solar_replacement(fraction)


def test_report_sweeps_values_without_claiming_savings_or_promoting():
    result = suite()
    assert result['publication_authorized'] is False
    assert len(result['scenarios']) == 16
    assert all(len(row['value_sensitivity']) == 8 for row in result['scenarios'])


@pytest.mark.parametrize('early_feed', [-.10, 0.])
def test_unrewarded_extra_export_cannot_be_saved_by_free_solar(early_feed):
    row = solar_replacement(1., early_feed=early_feed)
    assert net_value(row['comparison'], .40, .04) < 0


@pytest.mark.parametrize('terminal,wear', [(-1., .04), (.20, -1.), (float('nan'), .04)])
def test_invalid_valuation_rejected(terminal, wear):
    with pytest.raises(ValueError):
        net_value(scarce_later_energy()['comparison'], terminal, wear)
