import json
import sqlite3

import pandas as pd
import pytest

from eval.audit_ems_delivery import CURVES
from eval.replay_ems_fallback import guarded_controls, policy, publication_events, replay, static_guard
from test_sequential_core_replay import plant


def fixture():
    export = dict(battery=2400., grid=-2000., load=300., pv=100., hybrid=2300., curtailment=0.)
    idle = dict(battery=215.79, grid=0., load=300., pv=100., hybrid=300., curtailment=0.)
    history = {}
    for name, (column, key) in CURVES.items():
        history['mpc_'+name] = [
            {'time': '2026-09-30T22:04:25Z', column: json.dumps([
                {'date': '2026-09-30T22:00Z', key: export[name]},
                {'date': '2026-09-30T22:05Z', key: idle[name]}])},
            {'time': '2026-09-30T22:05:25Z', column: json.dumps([
                {'date': '2026-09-30T22:05Z', key: export[name]}])}]
    history.update(grid_status=[{'time': '2026-09-30T22:00Z', 'state': 'On Grid'}],
                   effective_feed=[{'time': '2026-09-30T22:00Z', 'value': .2}],
                   flexible_export_limit=[{'time': '2026-09-30T22:00Z', 'value': 9.999}])
    segments = [{'start': '2026-09-30T22:04:30Z', 'end': '2026-09-30T22:06:00Z',
                 'duration_seconds': 90., 'pv_dc_w': 100., 'load_site_w': 300.,
                 'general_rate': .4, 'feed_rate': .2}]
    return history, segments, export, idle


def test_fallback_hold_gains_credit_with_inventory_cost_and_own_soc(plant):
    history, segments, _, _ = fixture()
    result = replay(history, segments, plant, .8, .05)
    compare = result['comparison']
    assert compare['variable_credit_gain_aud'] == pytest.approx(2000*25/3_600_000*.2)
    assert compare['ending_inventory_delta_kwh'] < 0
    assert compare['dc_throughput_delta_kwh'] > 0
    assert compare['break_even_inventory_value_aud_per_kwh_before_wear'] == pytest.approx(.2*.95*.99)
    assert result['summary']['hold_accepted_command']['final_soc'] < result['summary']['preceding_plan_fallback']['final_soc']
    assert result['scope'] == 'frozen_incumbent_plan_conditioned_timing_sensitivity'
    assert result['publication_authorized'] is False


def test_incumbent_mode_soc_and_future_observations_do_not_control_challenger(plant):
    history, segments, _, _ = fixture()
    before = replay(history, segments, plant, .8, .05)
    history['ems_mode'] = [{'time': '2026-09-30T22:05Z', 'state': 'Standby'}]
    history['soc'] = [{'time': '2026-09-30T22:05Z', 'value': 99.}]
    history['effective_feed'].append({'time': '2026-09-30T22:06:01Z', 'value': -1.})
    assert replay(history, segments, plant, .8, .05) == before


def test_price_change_and_flexible_limit_apply_to_held_command(plant):
    history, segments, _, _ = fixture()
    history['effective_feed'].append({'time': '2026-09-30T22:05Z', 'value': -.01})
    result = replay(history, segments, plant, .8, .05)
    assert result['comparison']['variable_credit_gain_aud'] == pytest.approx(0., abs=1e-12)
    history['effective_feed'][-1]['value'] = .2
    history['flexible_export_limit'].append({'time': '2026-09-30T22:05Z', 'value': .5})
    result = replay(history, segments, plant, .8, .05)
    assert result['comparison']['variable_credit_gain_aud'] == pytest.approx(500*25/3_600_000*.2)


def test_physical_soc_floor_limits_economic_gain(plant):
    history, segments, _, _ = fixture()
    result = replay(history, segments, plant, plant['battery_minimum_state_of_charge'], .05)
    assert result['comparison']['variable_credit_gain_aud'] == pytest.approx(0., abs=1e-12)
    assert result['comparison']['ending_inventory_delta_kwh'] == 0
    with pytest.raises(ValueError, match='intra-segment guard'):
        replay(history, segments, plant, .8, .5)


@pytest.mark.parametrize('change', [{'battery': -1.}, {'battery': 0.}, {'grid': 1.}, {'curtailment': 1.}, {'hybrid': -1.}])
def test_unsupported_controller_branches_fail(change):
    _, _, export, _ = fixture()
    with pytest.raises(ValueError, match='unsupported'):
        policy(export | change, .8, .05)


def test_stale_missing_and_offgrid_sources_fail():
    history, _, export, _ = fixture()
    command = policy(export, .8, .05)
    with pytest.raises(ValueError):
        guarded_controls(history=history, command=command, at=pd.Timestamp('2026-09-30T22:16Z'), soc=.8, minimum_export_soc=.05)
    history['grid_status'][0]['state'] = 'Off Grid'
    with pytest.raises(ValueError, match='off-grid'):
        guarded_controls(command, history, pd.Timestamp('2026-09-30T22:05Z'), .8, .05)


def test_current_plan_activates_after_latest_changed_receipt_not_first(plant):
    history, segments, _, _ = fixture()
    history['mpc_grid'][-1]['time'] = '2026-09-30T22:05:25.5Z'
    events = publication_events(history, pd.Timestamp(segments[0]['start']), pd.Timestamp(segments[-1]['end']))
    assert events[0][0] == pd.Timestamp('2026-09-30T22:05:25.5Z')
    result = replay(history, segments, plant, .8, .05)
    assert result['comparison']['variable_credit_gain_aud'] == pytest.approx(2000*25.5/3_600_000*.2)


def test_hold_does_not_silently_extend_stale_command(plant):
    history, segments, _, _ = fixture()
    segments[0].update(end='2026-09-30T22:10Z', duration_seconds=330.)
    with pytest.raises(ValueError, match='fresh MPC command'):
        replay(history, segments, plant, .8, .05)


def test_fixed_dc_overhead_consumes_inventory_without_inventing_grid_credit(plant):
    history, segments, _, _ = fixture()
    before = replay(history, segments, plant, .8, .05)
    after = replay(history, segments, plant, .8, .05, dc_fixed_loss_w=140.)
    assert after['comparison']['variable_credit_gain_aud'] == pytest.approx(before['comparison']['variable_credit_gain_aud'])
    for arm in before['summary']:
        base, loss = before['summary'][arm], after['summary'][arm]
        assert loss['grid_export_kwh'] == pytest.approx(base['grid_export_kwh'])
        assert loss['ending_inventory_kwh'] < base['ending_inventory_kwh']
        assert loss['dc_throughput_kwh']-base['dc_throughput_kwh'] == pytest.approx(140.*90/3_600_000)
    assert after['dc_fixed_loss_w'] == 140.


def test_fixed_loss_and_device_dc_limit_reduce_export_physically(plant):
    history, segments, _, _ = fixture()
    plant['battery_discharge_power_max'] = 500.
    before = replay(history, segments, plant, .8, .05)
    after = replay(history, segments, plant, .8, .05, dc_fixed_loss_w=140.)
    assert after['summary']['hold_accepted_command']['grid_export_kwh'] < before['summary']['hold_accepted_command']['grid_export_kwh']
    assert after['comparison']['variable_credit_gain_aud'] < before['comparison']['variable_credit_gain_aud']
    with pytest.raises(ValueError, match='0/140W'):
        replay(history, segments, plant, .8, .05, dc_fixed_loss_w=-1.)


def test_static_guard_fallback_requires_unchanged_older_capture_clocks(tmp_path):
    path = tmp_path/'journal.sqlite'
    record = {'captured_at': '2026-10-03T00:00Z', 'input_snapshot': {'states': {
        'input_number.battery_soc_min_export': {'entity_id': 'input_number.battery_soc_min_export', 'state': '5',
            'last_changed': '2026-09-18T00:00Z', 'last_updated': '2026-09-18T00:00Z', 'last_reported': '2026-09-18T00:00Z'}}}}
    with sqlite3.connect(path) as connection:
        connection.execute('create table handoffs (record text)')
        connection.execute('insert into handoffs values (?)', (json.dumps(record),))
    value, provenance = static_guard(path, '2026-09-30T22:00Z')
    assert value == .05
    assert 'not_independent_historical' in provenance['kind']
    with pytest.raises(ValueError):
        static_guard(path, '2026-09-01T00:00Z')
