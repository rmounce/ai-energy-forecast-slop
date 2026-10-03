"""Frozen accepted-price → DH/MPC payload rehearsal; never solve or control."""
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math

from energy_pipeline.accepted_store import RecoveredBundle
from energy_pipeline.payloads import Inputs, boundary, build_dh_payload, build_mpc_payload, timestamp

PRICE_ENTITIES = ('sensor.ai_price_forecast_low', 'sensor.ai_price_forecast', 'sensor.ai_price_forecast_high')
ENTITIES = frozenset(PRICE_ENTITIES + (
    'sensor.sigen_plant_battery_state_of_charge_derived', 'sensor.sigen_plant_rated_energy_capacity',
    'sensor.sigen_plant_battery_state_of_health', 'input_number.dh_last_soc_init',
    'input_text.dh_last_reground_block', 'input_number.battery_soc_min_target',
    'input_number.emhass_target_soc_offset', 'input_number.emhass_weight_battery_discharge',
    'input_number.emhass_weight_pv_forecast', 'input_number.emhass_weight_buy_forecast',
    'input_number.emhass_weight_sell_forecast', 'input_number.sapn_free_exports',
    'sensor.emhass_dh_hwc_power_plan_snapshot', 'sensor.hwc_power_plan',
    'sensor.dh_soc_batt_forecast', 'sensor.ai_load_forecast_high',
    'sensor.amber_5min_current_general_price', 'sensor.amber_adjusted_confirmed_feed_in_price',
    'sensor.amber_5min_forecasts_extended_general_price', 'sensor.amber_5min_forecasts_extended_feed_in_price',
    'sensor.dh_p_load_forecast', 'sensor.dh_p_pv_forecast',
    'sensor.sigen_inverter_conversion_loss', 'sensor.sigen_power_pv_gross',
    'sensor.sigen_plant_consumed_power', 'sensor.emhass_current_pv_input_mode',
    'sensor.solcast_pv_forecast_power_now', 'input_number.battery_soc_min_buffer',
    'input_number.battery_soc_min_export', 'input_number.mpc_last_soc_init',
) + tuple(f'sensor.solcast_pv_forecast_forecast_{day}' for day in ('today', 'tomorrow', 'day_3', 'day_4')))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bundle_snapshot(plan, snapshot):
    """Replace all three DH price parents together in an owned frozen HA view."""
    bundle = RecoveredBundle(plan['bundle'])
    bundle.validate()
    now = timestamp(snapshot['captured_at'])
    started = timestamp(snapshot['capture_started_at'])
    if not 0 <= (now-started).total_seconds() <= 30:
        raise ValueError('handoff HA capture outside 30-second budget')
    if not bundle.time_current(now):
        raise ValueError('handoff accepted price bundle expired or interval changed')
    if tuple(write['entity'] for write in plan['writes']) != PRICE_ENTITIES:
        raise ValueError('handoff price targets differ from current payload policy')
    frozen = deepcopy(snapshot)
    frozen['states'] = {key: value for key, value in frozen['states'].items() if key in ENTITIES}
    for write in plan['writes']:
        frozen['states'][write['entity']] = deepcopy(write['payload'])
    return frozen


def _coverage(payload, count):
    reasons = []
    for key in ('load_cost_forecast', 'prod_price_forecast', 'load_power_forecast', 'pv_power_forecast'):
        values = payload[key]
        if len(values) != count:
            reasons.append(f'{key}:slots={len(values)},expected={count}')
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            reasons.append(f'{key}:nonfinite')
        if 'power' in key and any(value < 0 for value in values):
            reasons.append(f'{key}:negative')
    for key in ('soc_init', 'soc_final', 'battery_minimum_state_of_charge'):
        if not math.isfinite(payload[key]) or not 0 <= payload[key] <= 1:
            reasons.append(f'{key}:outside_fraction_range')
    if payload['battery_nominal_energy_capacity'] <= 0:
        reasons.append('battery_nominal_energy_capacity:not_positive')
    return reasons


def build_handoff(plan, snapshot):
    frozen = bundle_snapshot(plan, snapshot)
    now = timestamp(frozen['captured_at'])
    inputs = Inputs(frozen['states'], now, frozen['timezone'])
    payloads = {'dh': build_dh_payload(inputs), 'mpc': build_mpc_payload(inputs)}
    readiness = {}
    for kind, count in (('dh', 144), ('mpc', 168)):
        reasons = _coverage(payloads[kind], count)
        readiness[kind] = {'coverage_ready': not reasons, 'reasons': reasons, 'solve_authorized': False}
    # DH load is consumed positionally by incumbent templates; length alone cannot
    # establish that its first row belongs to this price interval.
    load = inputs.attr('sensor.ai_load_forecast_high', 'forecasts') or []
    expected = boundary(now, 30)
    if len(load) != 144 or any(timestamp(row['timestamp']) != expected+timedelta(minutes=30*i)
                               for i, row in enumerate(load)):
        readiness['dh']['reasons'].append('load_targets:misaligned')
        readiness['dh']['coverage_ready'] = False
    pv_targets = [timestamp(row['period_start']) for day in ('today', 'tomorrow', 'day_3', 'day_4')
                  for row in inputs.attr(f'sensor.solcast_pv_forecast_forecast_{day}', 'detailedForecast') or []
                  if timestamp(row['period_start']) > now-timedelta(minutes=30)]
    if len(pv_targets) < 144 or pv_targets[:144] != [expected+timedelta(minutes=30*i) for i in range(144)]:
        readiness['dh']['reasons'].append('pv_targets:misaligned')
    mpc_start = boundary(now, 5)
    last_power = boundary(mpc_start+timedelta(minutes=5*167), 30)
    power_count = int((last_power-expected).total_seconds()//1800)+1
    for channel in ('load', 'pv'):
        entity = f'sensor.dh_p_{channel}_forecast'
        targets = [timestamp(row['date']) for row in inputs.attr(entity, 'forecasts') or []
                   if timestamp(row['date']) >= expected]
        if targets[:power_count] != [expected+timedelta(minutes=30*i) for i in range(power_count)]:
            readiness['mpc']['reasons'].append(f'dh_{channel}_targets:misaligned')
    for channel in ('general', 'feed_in'):
        targets = [timestamp(row['start_time']) for row in inputs.attr(
            f'sensor.amber_5min_forecasts_extended_{channel}_price', 'Forecasts') or []
            if timestamp(row['start_time']) > now]
        expected_prices = [mpc_start+timedelta(minutes=5*i) for i in range(1, 168)]
        # Live extended Amber rows use a +1s start offset with exact 300s cadence.
        # Accept only that bounded offset; preserve original timestamps/payloads.
        if len(targets) < 167 or any(not 0 <= (at-wanted).total_seconds() <= 1
                                    for at, wanted in zip(targets[:167], expected_prices)):
            readiness['mpc']['reasons'].append(f'{channel}_price_targets:misaligned')
    for value in readiness.values():
        value['coverage_ready'] = not value['reasons']
    common_live = ('sensor.sigen_plant_battery_state_of_charge_derived',
                   'sensor.sigen_plant_rated_energy_capacity', 'sensor.sigen_plant_battery_state_of_health')
    mpc_live = ('sensor.amber_5min_current_general_price', 'sensor.amber_adjusted_confirmed_feed_in_price',
                'sensor.sigen_inverter_conversion_loss', 'sensor.sigen_power_pv_gross',
                'sensor.sigen_plant_consumed_power')
    for kind in readiness:
        for entity in common_live + (mpc_live if kind == 'mpc' else ()):
            try:
                usable = math.isfinite(float(inputs.state(entity)))
            except (TypeError, ValueError):
                usable = False
            if not usable:
                readiness[kind]['reasons'].append(f'{entity}:unusable_live_state')
                readiness[kind]['coverage_ready'] = False
    states = frozen['states']
    return {'schema': 1, 'mode': 'handoff_shadow', 'publication_id': plan['id'],
        'price_run_id': plan['bundle']['run_id'], 'captured_at': frozen['captured_at'],
        'input_revision': digest(states), 'input_snapshot': frozen, 'payloads': payloads,
        'readiness': readiness, 'lineage': {
            'dh_price_parent': plan['id'],
            'mpc_price_parent': digest({key: states.get(key) for key in (
                'sensor.amber_5min_current_general_price', 'sensor.amber_adjusted_confirmed_feed_in_price',
                'sensor.amber_5min_forecasts_extended_general_price', 'sensor.amber_5min_forecasts_extended_feed_in_price')}),
            'mpc_dh_parent': digest({key: states.get(key) for key in (
                'sensor.dh_p_load_forecast', 'sensor.dh_p_pv_forecast', 'sensor.dh_soc_batt_forecast')}),
            'mpc_dh_price_parent': 'unknown',
        }}
