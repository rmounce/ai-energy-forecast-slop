from copy import deepcopy
import hashlib
import json

import pandas as pd
import pytest

from energy_pipeline.handoff import PRICE_ENTITIES, ENTITIES, bundle_snapshot, build_handoff
from energy_pipeline.publication import encoded, ShadowPublication
from energy_pipeline.accepted_store import StoreError
from test_accepted_store import NOW
from test_publication import plan


def coherent_plan(plan):
    copied = deepcopy(plan)
    for write, entity in zip(copied['writes'], PRICE_ENTITIES): write['entity'] = entity
    del copied['id']
    copied['id'] = hashlib.sha256(encoded(copied).encode()).hexdigest()
    return copied


def snapshot():
    start = NOW.floor('30min')
    def entity(state='0', **attrs): return {'state': str(state), 'attributes': attrs}
    states = {key: entity() for key in ENTITIES}
    states['sensor.sigen_plant_battery_state_of_charge_derived'] = entity(55)
    states['sensor.sigen_plant_rated_energy_capacity'] = entity(50)
    states['sensor.sigen_plant_battery_state_of_health'] = entity(100)
    states['input_number.battery_soc_min_target'] = entity(10)
    states['sensor.ai_load_forecast_high'] = entity(forecasts=[
        {'timestamp': (start+pd.Timedelta(minutes=30*i)).isoformat(), 'power_load': 1000} for i in range(144)])
    states['sensor.solcast_pv_forecast_forecast_today'] = entity(detailedForecast=[
        {'period_start': (start+pd.Timedelta(minutes=30*i)).isoformat(), 'pv_estimate': 2,
         'pv_estimate10': 1, 'pv_estimate90': 3} for i in range(144)])
    for channel in ('general', 'feed_in'):
        states[f'sensor.amber_5min_forecasts_extended_{channel}_price'] = entity(Forecasts=[
            {'start_time': (NOW.floor('5min')+pd.Timedelta(minutes=5*i)).isoformat(),
             'advanced_price_low': .1, 'advanced_price_predicted': .2, 'advanced_price_high': .3, 'per_kwh': .2}
            for i in range(1, 168)])
    for channel in ('load', 'pv'):
        key = f'dh_p_{channel}_forecast'
        states['sensor.'+key] = entity(forecasts=[{'date': (start+pd.Timedelta(minutes=30*i)).isoformat(), key: 1000} for i in range(30)])
    states['sensor.dh_soc_batt_forecast'] = entity(battery_scheduled_soc=[
        {'date': start.isoformat(), 'dh_soc_batt_forecast': 55}])
    states['sensor.unrelated'] = entity(secret='never copied')
    return {'capture_started_at': NOW.isoformat(), 'captured_at': (NOW+pd.Timedelta(seconds=1)).isoformat(),
            'timezone': 'Australia/Adelaide', 'states': states}


def test_bundle_replaces_three_price_parents_without_mutating_inputs(plan):
    plan = coherent_plan(plan)
    capture = snapshot()
    original = deepcopy(capture)
    frozen = bundle_snapshot(plan, capture)
    for write in plan['writes']:
        assert frozen['states'][write['entity']] == write['payload']
    assert 'sensor.unrelated' not in frozen['states']
    assert capture == original
    frozen['states'][PRICE_ENTITIES[0]]['attributes']['forecasts'][0]['general_price'] = 999
    assert plan['writes'][0]['payload']['attributes']['forecasts'][0]['general_price'] != 999


def test_payload_lineage_distinguishes_dh_bundle_from_current_mpc_inputs(plan):
    plan = coherent_plan(plan)
    record = build_handoff(plan, snapshot())
    assert record['lineage']['dh_price_parent'] == plan['id']
    assert record['lineage']['mpc_price_parent'] != plan['id']
    assert record['lineage']['mpc_dh_price_parent'] == 'unknown'
    assert record['readiness']['dh']['coverage_ready']
    assert record['readiness']['mpc']['coverage_ready']
    assert not any(value['solve_authorized'] for value in record['readiness'].values())
    assert len(record['payloads']['dh']['load_cost_forecast']) == 144
    assert len(record['payloads']['mpc']['load_cost_forecast']) == 168


@pytest.mark.parametrize('bad', ['expiry', 'capture', 'targets'])
def test_invalid_handoff_boundary_rejected(plan, bad):
    plan = coherent_plan(plan)
    capture = snapshot()
    if bad == 'expiry': capture['captured_at'] = (NOW+pd.Timedelta(minutes=4)).isoformat(); capture['capture_started_at'] = capture['captured_at']
    elif bad == 'capture': capture['capture_started_at'] = (NOW-pd.Timedelta(minutes=1)).isoformat()
    else: plan['writes'][0]['entity'] = 'sensor.other'
    with pytest.raises(ValueError): build_handoff(plan, capture)


@pytest.mark.parametrize('bad', ['load_length', 'load_alignment', 'mpc_power', 'mpc_price'])
def test_partial_or_misaligned_payload_is_diagnostic_not_ready(plan, bad):
    plan = coherent_plan(plan)
    capture = snapshot()
    states = capture['states']
    if bad == 'load_length': states['sensor.ai_load_forecast_high']['attributes']['forecasts'].pop()
    elif bad == 'load_alignment': states['sensor.ai_load_forecast_high']['attributes']['forecasts'][0]['timestamp'] = (NOW-pd.Timedelta(hours=1)).isoformat()
    elif bad == 'mpc_power': states['sensor.dh_p_load_forecast']['attributes']['forecasts'] = []
    else: states['sensor.amber_5min_forecasts_extended_general_price']['attributes']['Forecasts'] = []
    record = build_handoff(plan, capture)
    kind = 'dh' if bad.startswith('load') else 'mpc'
    assert not record['readiness'][kind]['coverage_ready']
    assert not record['readiness'][kind]['solve_authorized']


def test_handoff_storage_requires_matching_commit_and_bounds_raw_capture(tmp_path, plan):
    plan = coherent_plan(plan)
    record = build_handoff(plan, snapshot())
    journal = ShadowPublication(tmp_path/'journal.sqlite')
    try:
        with pytest.raises(StoreError, match='not the committed'): journal.save_handoff(plan, record)
        assert journal.execute(plan, lambda: True, now=lambda: NOW)
        journal.save_handoff(plan, record)
        import sqlite3
        with sqlite3.connect(journal.path) as db:
            saved = json.loads(db.execute('SELECT record FROM handoffs').fetchone()[0])
        assert saved == record
        assert 'sensor.unrelated' not in saved['input_snapshot']['states']
    finally:
        journal.close()


def test_capture_manifest_covers_all_template_entity_dependencies():
    import re
    from scripts.replay_energy_payloads import templates
    referenced = set(re.findall(r'(?:sensor|input_number|input_text)\.[a-z0-9_]+', json.dumps(templates())))
    assert referenced <= ENTITIES


@pytest.mark.parametrize('target', ['pv', 'dh_power', 'mpc_price'])
def test_matching_lengths_do_not_hide_misaligned_timestamps(plan, target):
    plan = coherent_plan(plan)
    capture = snapshot()
    states = capture['states']
    if target == 'pv':
        states['sensor.solcast_pv_forecast_forecast_today']['attributes']['detailedForecast'][4]['period_start'] = NOW.isoformat()
    elif target == 'dh_power':
        states['sensor.dh_p_load_forecast']['attributes']['forecasts'][4]['date'] = NOW.isoformat()
    else:
        states['sensor.amber_5min_forecasts_extended_general_price']['attributes']['Forecasts'][4]['start_time'] = (NOW+pd.Timedelta(minutes=1)).isoformat()
    record = build_handoff(plan, capture)
    kind = 'dh' if target == 'pv' else 'mpc'
    assert not record['readiness'][kind]['coverage_ready']
    assert any('misaligned' in reason for reason in record['readiness'][kind]['reasons'])


@pytest.mark.parametrize('offset,ready', [(1, True), (2, False), (-1, False)])
def test_confirmed_extended_amber_one_second_offset_preserves_slots(plan, offset, ready):
    plan = coherent_plan(plan)
    capture = snapshot()
    for channel in ('general', 'feed_in'):
        rows = capture['states'][f'sensor.amber_5min_forecasts_extended_{channel}_price']['attributes']['Forecasts']
        for row in rows:
            row['start_time'] = (pd.Timestamp(row['start_time'])+pd.Timedelta(seconds=offset)).isoformat()
    record = build_handoff(plan, capture)
    assert record['readiness']['mpc']['coverage_ready'] == ready
    assert record['input_snapshot']['states']['sensor.amber_5min_forecasts_extended_general_price'] == capture['states']['sensor.amber_5min_forecasts_extended_general_price']


def test_unknown_live_state_is_not_mistaken_for_safe_numeric_fallback(plan):
    capture = snapshot()
    capture['states']['sensor.sigen_plant_battery_state_of_charge_derived']['state'] = 'unavailable'
    record = build_handoff(coherent_plan(plan), capture)
    assert not record['readiness']['dh']['coverage_ready']
    assert not record['readiness']['mpc']['coverage_ready']
    assert any('unusable_live_state' in reason for reason in record['readiness']['mpc']['reasons'])
