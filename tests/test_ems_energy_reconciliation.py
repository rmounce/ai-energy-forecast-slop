import pytest

from eval.reconcile_ems_energy import held_energy, reconcile


def rows(values):
    return [{'time':f'2026-09-30T22:0{i}:00Z','value':v} for i,v in enumerate(values)]


def test_raw_signed_ledger_separates_charge_discharge_and_holds_exact_edges():
    r = held_energy(rows([1000,-2000]),'2026-09-30T22:00:30Z','2026-09-30T22:01:30Z')
    assert r['positive_kwh'] == pytest.approx(1000*30/3600000)
    assert r['negative_kwh'] == pytest.approx(2000*30/3600000)
    assert r['signed_kwh'] == pytest.approx(r['positive_kwh']-r['negative_kwh'])


@pytest.mark.parametrize('data,end',[(rows([1000]),'22:03:00Z'),(rows([1000,float('nan')]),'22:02:00Z')])
def test_expiry_and_missing_observation_terminate_support(data,end):
    with pytest.raises(ValueError,match='incomplete'):
        held_energy(data,'2026-09-30T22:00Z','2026-09-30T'+end)


def test_future_power_is_not_used_before_receipt():
    a = held_energy(rows([1000,2000]),'2026-09-30T22:00Z','2026-09-30T22:01Z')
    b = held_energy(rows([1000,99999]),'2026-09-30T22:00Z','2026-09-30T22:01Z')
    assert a == b


def history():
    h = {k:rows([v,v]) for k,v in [('pv1',100),('pv2',0),('pv',100),('battery',-1000),
                                   ('inverter_ac',1000),('grid',-700),('load',300),('loss',100)]}
    # Capacity-estimate movement at fixed ratio; no physical-energy truth implied.
    h.update(soc=rows([50,50]),available_charge=rows([20,19]),available_discharge=rows([20,19]),
             derived_capacity=rows([40,38]))
    return h


def test_capacity_identity_is_separate_from_dc_power_and_nominal_soc():
    r = reconcile(history(),'2026-09-30T22:00Z','2026-09-30T22:01Z',nominal_kwh=40.3)
    i = r['inventory']
    assert i['nominal_soc_energy_change_kwh'] == 0
    assert i['dc_power_energy_change_kwh'] == pytest.approx(-1/.99/60)
    assert i['available_ratio_change_component_kwh'] == 0
    assert i['available_capacity_change_component_kwh'] == -1
    assert i['available_discharge_change_kwh'] == -1
    assert i['available_discharge_minus_dc_power_change_kwh'] != 0


def test_raw_dc_ac_and_site_balance_use_independent_sensor_streams():
    r = reconcile(history(),'2026-09-30T22:00Z','2026-09-30T22:01Z',nominal_kwh=40.3)
    b = r['balance']
    assert b['raw_dc_minus_ac_kwh'] == pytest.approx(.1/60)
    assert b['derived_clipped_loss_kwh'] == pytest.approx(.1/60)
    assert b['grid_plus_inverter_minus_site_kwh'] == pytest.approx(0)
    assert b['pv_sum_minus_gross_kwh'] == 0


def test_invalid_capacity_or_efficiency_is_not_silently_repaired():
    for kwargs in [{'nominal_kwh':0},{'nominal_kwh':40,'discharge_efficiency':0}]:
        with pytest.raises(ValueError): reconcile(history(),'2026-09-30T22:00Z','2026-09-30T22:01Z',**kwargs)
