from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from eval.audit_ems_delivery import (CURVES, audit, bounded_self_consume_branch,
                                    control_segments, execute_ems, selected_plan)
from test_sequential_core_replay import plant


def actual(mode='Command Discharging (PV First)', grid=2., pcs=100., discharge=24., charge=21., **values):
    return {'start':'2026-09-30T22:05:00Z','end':'2026-09-30T22:06:00Z','duration_seconds':60,
            'pv_dc_w':100.,'load_site_w':300.,'observed_battery_charge_w':-2000.,
            'observed_grid_import_w':-1500.,'general_rate':.4,'feed_rate':.2,
            'controls': {'ems_mode':{'state':mode},'grid_export_limit':{'value':grid},
                         'pcs_export_limit':{'value':pcs},'discharge_limit':{'value':discharge},'charge_limit':{'value':charge}}, **values}


def history():
    h = {}
    for name, (column,key) in CURVES.items():
        values = {'battery':315.79,'grid':0.,'load':300.,'pv':0.,'hybrid':300.,'curtailment':0.}
        h['mpc_'+name] = [{'time':'2026-09-30T22:04:26Z',column:json.dumps([
            {'date':'2026-09-30T22:00:00Z',key:1000.},
            {'date':'2026-09-30T22:05:00Z',key:values[name]}])}]
        # Must not contaminate selection at the tick.
        h['mpc_'+name].append({'time':'2026-09-30T22:05:25Z',column:json.dumps([
            {'date':'2026-09-30T22:05:00Z',key:5000.}])})
    h.update(mode=[{'time':'2026-09-30T21:00Z','charge_limit_kw':21.}],grid_status=[{'time':'2026-09-30T21:00Z','state':'On Grid'}],ems_action=[{'time':'2026-09-30T22:05:00.2Z','state':'Self-consume from battery'}],
             ems_mode=[{'time':'2026-09-30T22:04:00Z','state':'Command Discharging (PV First)'},
                       {'time':'2026-09-30T22:05:05Z','state':'Maximum Self Consumption'},
                       {'time':'2026-09-30T22:05:30Z','state':'Command Discharging (PV First)'}])
    for name,value in [('grid_export_limit',2.),('pcs_export_limit',100.),('discharge_limit',24.)]:
        h[name] = [{'time':'2026-09-30T21:00:00Z','value':value}]
    return h


def test_fallback_selects_next_slot_from_prior_plan_and_excludes_future_publication():
    values,refs = selected_plan(history(),'2026-09-30T22:05:00Z')
    assert values['grid'] == 0
    assert values['battery'] == 315.79
    assert refs['battery']['publication'] == '2026-09-30T22:04:26Z'
    assert refs['battery']['target'] == '2026-09-30T22:05:00Z'
    assert bounded_self_consume_branch(values,'On Grid') == 'Self-consume from battery'


def test_supported_branch_matches_current_yaml_and_does_not_guess_other_cases():
    controller = yaml.safe_load((Path(__file__).parents[1]/'hass/automation-sigenergy-emhass.yaml').read_text())
    branches = controller['actions'][1]['choose']
    branch = next(b for b in branches if b['conditions'][0].get('value_template','').strip() == '{{ p_grid == 0 }}')
    assert branch['sequence'][0]['data_template']['value'] == 'Self-consume from battery'
    assert branch['sequence'][1]['data']['control_mode'] == 'Maximum Self Consumption'
    plan = selected_plan(history(),'2026-09-30T22:05Z')[0]
    for change in ({'grid':-1000.},{'battery':-300.},{'curtailment':100.},{'load':0.},{'hybrid':-100.}):
        assert bounded_self_consume_branch(plan | change,'On Grid') is None
    assert bounded_self_consume_branch(plan,'Off Grid') is None


def test_mode_selection_changes_energy_and_preserves_input_plant(plant):
    before = deepcopy(plant)
    self_consume = execute_ems(plant,.8,actual(mode='Maximum Self Consumption'))
    export = execute_ems(plant,.8,actual())
    assert self_consume['grid_export_kwh'] == pytest.approx(0.,abs=1e-12)
    assert self_consume['battery_discharge_w'] == pytest.approx(300/.95-100)
    assert export['grid_export_kwh'] == pytest.approx(2/60)
    assert export['end_soc'] < self_consume['end_soc']
    assert plant == before


def test_pcs_and_discharge_limits_are_physical_constraints(plant):
    capped = execute_ems(plant,.8,actual(pcs=.5))
    assert capped['inverter_ac_w'] == pytest.approx(500)
    assert capped['grid_export_kwh'] == pytest.approx(.2/60)
    discharge = execute_ems(plant,.8,actual(discharge=.1))
    assert discharge['battery_discharge_w'] == pytest.approx(100)
    assert discharge['grid_import_kwh'] > 0
    ignore_pcs = execute_ems(plant,.8,actual(pcs=.5),use_pcs_limit=False)
    assert ignore_pcs['grid_export_kwh'] > capped['grid_export_kwh']


def test_discharge_mode_does_not_charge_from_excess_pv_and_respects_soc_floor(plant):
    surplus = execute_ems(plant,.8,actual(pv_dc_w=10000,grid=0.))
    assert surplus['battery_discharge_w'] == pytest.approx(0)
    assert surplus['curtailed_pv_w'] > 0
    empty = execute_ems(plant,plant['battery_minimum_state_of_charge'],actual())
    assert empty['battery_discharge_w'] == pytest.approx(0)
    assert empty['end_soc'] == plant['battery_minimum_state_of_charge']


@pytest.mark.parametrize('values', [{'mode':'Standby'},{'pcs':float('nan')},{'grid':-1.}])
def test_unsupported_mode_or_invalid_register_fails(plant,values):
    with pytest.raises(ValueError): execute_ems(plant,.8,actual(**values))


def test_trace_cuts_at_recorded_changes_without_using_future_registers():
    h = history()
    h['grid_export_limit'].append({'time':'2026-09-30T22:05:20Z','value':0.})
    rows = control_segments(h,[actual()])
    assert sum(r['duration_seconds'] for r in rows) == 60
    assert rows[0]['controls']['ems_mode']['state'] == 'Command Discharging (PV First)'
    assert rows[1]['controls']['ems_mode']['state'] == 'Maximum Self Consumption'
    assert rows[1]['controls']['grid_export_limit']['value'] == 2.
    assert rows[2]['controls']['grid_export_limit']['value'] == 0.
    with pytest.raises(ValueError,match='no causal'): control_segments({**h,'pcs_export_limit':[]},[actual()])


def test_audit_reports_supported_action_mode_clock_and_diagnostic_scope(plant):
    out = audit(history(),[actual()],plant,.8,{'grid_import_kwh':0.,'grid_export_kwh':.1,'ending_inventory_kwh':31.},.79)
    assert out['diagnostics']['supported_tick_action_matches'] == 1
    assert out['ticks'][0]['next_mpc_publication'] == '2026-09-30T22:05:25+00:00'
    assert out['ticks'][0]['mode_changes_before_publication'][0]['time'] == '2026-09-30T22:05:05Z'
    assert out['observed']['duration_seconds'] == 60
    assert out['publication_authorized'] is False
    assert 'not a policy challenger' in __import__('eval.audit_ems_delivery',fromlist=['']).__doc__


def test_self_consumption_respects_recorded_charge_cutoff(plant):
    values = {'mode':'Maximum Self Consumption','pv_dc_w':1000.}
    allowed = execute_ems(plant,.8,actual(**values))
    blocked = execute_ems(plant,.8,actual(charge=0.,**values))
    assert allowed['battery_discharge_w'] < 0
    assert allowed['grid_export_kwh'] == pytest.approx(0.,abs=1e-12)
    assert blocked['battery_discharge_w'] == 0
    assert blocked['grid_export_kwh'] > 0
    assert blocked['end_soc'] == .8


def test_charging_limit_attribute_change_cuts_timeline_and_missing_readback_fails():
    h = history()
    h['mode'].append({'time':'2026-09-30T22:05:15Z','charge_limit_kw':0.})
    rows = control_segments(h,[actual()])
    assert rows[0]['controls']['charge_limit']['value'] == 21.
    assert next(r for r in rows if pd.Timestamp(r['start'])==pd.Timestamp('2026-09-30T22:05:15Z'))['controls']['charge_limit']['value'] == 0.
    h['mode'][-1]['charge_limit_kw'] = None
    with pytest.raises(ValueError,match='charging-limit'): control_segments(h,[actual()])
