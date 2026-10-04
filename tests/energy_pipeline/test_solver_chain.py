from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from energy_pipeline.handoff import build_handoff
from energy_pipeline.payloads import Inputs, interpolate, soc_points, mpc_soc
from energy_pipeline.solver_chain import build_chained_handoff, project_dh_entities
from energy_pipeline.solver_replay import digest, prepare_request
from test_handoff import coherent_plan, snapshot
from test_publication import plan
from test_solver_replay import inputs, HASH


def artifact(record, config):
    request = prepare_request(record, config, kind='dh', optimization_sha256=HASH)
    payload = request['payload']
    index = pd.date_range(request['forecast_start'], periods=144, freq='30min')
    batt = np.zeros(144)
    batt[:2] = [-1000, 980.1]
    change = np.where(batt > 0, batt/.99, batt*.99)
    soc = payload['soc_init']-np.cumsum(change)*.5/payload['battery_nominal_energy_capacity']
    dc = np.asarray(payload['pv_power_forecast'])+batt
    ac = np.where(dc > 0, dc*.95, dc/.95)
    net = np.asarray(payload['load_power_forecast'])-ac
    frame = pd.DataFrame({'P_Load': payload['load_power_forecast'],
        'P_PV': payload['pv_power_forecast'], 'P_batt': batt,
        'P_grid_pos': np.maximum(net, 0), 'P_grid_neg': np.minimum(net, 0),
        'P_hybrid_inverter': ac, 'SOC_opt': soc,
        'unit_load_cost': payload['load_cost_forecast'],
        'unit_prod_price': payload['prod_price_forecast']}, index=index)
    result = {'request_id': request['request_id'], 'optimization_sha256': HASH,
        'status': 'Optimal', 'targets': [at.isoformat() for at in index],
        'columns': list(frame), 'values': frame.to_numpy().tolist(),
        'projected_dh_entities': project_dh_entities(frame)}
    return {'mode': 'historical_solver_replay', 'publication_authorized': False,
            'request': request, 'result': result}


@pytest.fixture
def chain_inputs(plan, inputs):
    record = build_handoff(coherent_plan(plan), snapshot())
    return record, artifact(record, inputs[1])


def test_projection_keeps_start_labels_for_end_soc_and_percent_strings(chain_inputs):
    record, parent = chain_inputs
    projected = parent['result']['projected_dh_entities']
    rows = projected['sensor.dh_soc_batt_forecast']['attributes']['battery_scheduled_soc']
    start = pd.Timestamp(parent['request']['forecast_start']).to_pydatetime()
    assert rows[0] == {'date': start.isoformat(), 'dh_soc_batt_forecast': '55.99'}
    points = soc_points(rows, 55)
    assert points[0] == (start, 55)
    assert points[1] == (start+pd.Timedelta(minutes=30), 55.99)
    assert interpolate(points, start+pd.Timedelta(minutes=15), 0) == pytest.approx(55.495)


def test_new_dh_parent_replaces_all_three_channels_and_helper_without_mutation(chain_inputs):
    record, parent = chain_inputs
    original = deepcopy((record, parent))
    chained = build_chained_handoff(record, parent)
    assert (record, parent) == original
    states = chained['input_snapshot']['states']
    assert states['input_number.dh_last_soc_init']['state'] == '55.0'
    assert states['input_text.dh_last_reground_block']['state'] == 'dh-20261002T0630Z'
    for entity in parent['result']['projected_dh_entities']:
        assert states[entity] == parent['result']['projected_dh_entities'][entity]
    assert chained['lineage']['mpc_dh_parent'] == digest(parent['result'])
    assert chained['lineage']['mpc_dh_price_parent'] == record['publication_id']
    assert chained['lineage']['mpc_price_parent'] == record['lineage']['mpc_price_parent']
    for key in ('load_cost_forecast', 'prod_price_forecast'):
        assert chained['payloads']['mpc'][key] == record['payloads']['mpc'][key]
    assert all(not value['solve_authorized'] for value in chained['readiness'].values())
    chained['input_snapshot']['states']['input_number.dh_last_soc_init']['state'] = '1'
    assert (record, parent) == original


def test_anchor_preserves_pre_payload_percent_precision(plan, inputs):
    capture = snapshot()
    capture['states']['sensor.sigen_plant_battery_state_of_charge_derived']['state'] = '55.123456'
    record = build_handoff(coherent_plan(plan), capture)
    parent = artifact(record, inputs[1])
    chained = build_chained_handoff(record, parent)
    assert parent['request']['payload']['soc_init'] == .5512
    assert chained['historical_chain']['dh_last_soc_init_pct'] == 55.1235


@pytest.mark.parametrize('actual', [40, 60, 100])
def test_chained_soc_uses_signed_init_positive_lockin_and_full_guard(chain_inputs, actual):
    chained = build_chained_handoff(*chain_inputs)
    snap = chained['input_snapshot']
    snap['states']['sensor.sigen_plant_battery_state_of_charge_derived']['state'] = str(actual)
    inputs = Inputs(snap['states'], snap['captured_at'], snap['timezone'])
    policy = mpc_soc(inputs)
    assert policy.soc_final_pct == (55 if actual == 40 else pytest.approx(55+max(policy.deviation_pct, 0)))
    if actual == 100:
        assert policy.soc_init_pct == 100
    elif actual == 40:
        assert policy.deviation_pct < 0 and policy.soc_init_pct < 40


@pytest.mark.parametrize('bad', ['publish', 'identity', 'handoff', 'kind', 'projection',
                                 'status', 'coverage', 'policy', 'snapshot'])
def test_unproven_or_mismatched_chain_rejected(chain_inputs, bad):
    record, parent = chain_inputs
    if bad == 'publish': parent['publication_authorized'] = True
    elif bad == 'identity': parent['request']['payload']['soc_init'] = .4
    elif bad == 'handoff': record['publication_id'] = 'other'
    elif bad == 'kind': parent['request']['kind'] = 'mpc'
    elif bad == 'projection': parent['result']['projected_dh_entities'] = {}
    elif bad == 'status': parent['result']['status'] = 'Infeasible'
    else:
        if bad == 'coverage': record['readiness']['mpc']['coverage_ready'] = False
        elif bad == 'policy': record['payloads']['mpc']['soc_final'] = .7
        else: record['input_snapshot']['captured_at'] = '2026-10-02T06:40:00Z'
        # Keep request binding valid to isolate the specific downstream check.
        parent['request']['handoff_revision'] = digest(record)
        parent['request']['request_id'] = digest({k: v for k, v in parent['request'].items()
                                                if k != 'request_id'})
        parent['result']['request_id'] = parent['request']['request_id']
    with pytest.raises(ValueError): build_chained_handoff(record, parent)


def test_chained_request_retains_parent_evidence(chain_inputs, inputs):
    chained = build_chained_handoff(*chain_inputs)
    request = prepare_request(chained, inputs[1], kind='mpc', optimization_sha256=HASH)
    assert request['historical_chain'] == chained['historical_chain']
    assert request['historical_chain']['publication_authorized'] is False
    with pytest.raises(ValueError, match='only supports'):
        prepare_request(chained, inputs[1], kind='dh', optimization_sha256=HASH)
