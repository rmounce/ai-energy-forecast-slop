import hashlib
import json

import pandas as pd
import pytest

from eval.apf_packet_regret import evaluate, load_inputs


def fixture(prices=(.1, .3), realised=(.4, .2), origin='2026-09-27T00:00:01Z'):
    start = pd.Timestamp('2026-09-27T00:05:00Z')
    rows = []
    for i, price in enumerate(prices):
        end = start+pd.Timedelta(minutes=5*(i+1))
        rows.append({'end_time': end.isoformat(), 'duration': 5,
                     'advanced_price_predicted': -price, 'advanced_price_low': -price+.05})
    revision = {'receipt': origin, 'rows': rows}
    rates = pd.DataFrame({'rate': realised}, index=pd.date_range(start, periods=len(prices), freq='5min'))
    return revision, rates, origin


def run(revision, rates, origin, **kwargs):
    return evaluate(revision, rates, origin, 15/60, wear_aud_per_dc_kwh=0, inverter_efficiency=1, **kwargs)


def test_raw_feed_sign_and_realised_regret():
    result = run(*fixture(), battery_discharge_efficiency=1)
    assert result['candidates']['predicted']['selected_target'].endswith('00:10:00+00:00')
    assert result['oracle_target'].endswith('00:05:00+00:00')
    assert result['candidates']['predicted']['regret_aud'] == pytest.approx(.05)
    assert result['candidates']['predicted']['regret_cents_per_stored_kwh'] == pytest.approx(20)


def test_terminal_hold_is_comparable_and_negative_actual_can_lose():
    revision, rates, origin = fixture(prices=(.3, .4), realised=(-.1, -.2))
    result = run(revision, rates, origin, battery_discharge_efficiency=1, terminal_value_aud_per_dc_kwh=.2)
    assert result['oracle_target'] is None
    assert result['candidates']['predicted']['realised_incremental_value_aud'] == pytest.approx(-.1)
    assert result['candidates']['predicted']['regret_aud'] == pytest.approx(.1)


def test_abstain_preserves_packet_and_efficiency_cost():
    result = run(*fixture(prices=(.1, .2)), battery_discharge_efficiency=.8,
                 terminal_value_aud_per_dc_kwh=.2)
    assert result['candidates']['predicted']['selected_target'] is None
    assert result['candidates']['predicted']['ending_packet_kwh'] == .25
    assert result['packet_export_ac_kwh'] == .2
    assert result['required_spare_export_kw'] == pytest.approx(2.4)


def test_conservative_bound_can_abstain_while_predicted_exports():
    result = run(*fixture(prices=(.1, .2)), battery_discharge_efficiency=1,
                 terminal_value_aud_per_dc_kwh=.175)
    assert result['candidates']['predicted']['selected_target'] is not None
    assert result['candidates']['conservative']['selected_target'] is None


@pytest.mark.parametrize('mutation,reason', [
    ('future', 'future'), ('stale', 'stale'), ('missing_quote', 'incomplete confirmed'),
    ('missing_apf', 'incomplete APF'), ('duplicate_quote', 'duplicate quote'),
    ('duplicate_apf', 'duplicate APF')])
def test_rejects_incomplete_or_noncausal(mutation, reason):
    revision, rates, origin = fixture()
    if mutation == 'future': revision['receipt'] = '2026-09-27T00:01:00Z'
    if mutation == 'stale': revision['receipt'] = '2026-09-26T23:00:00Z'
    if mutation == 'missing_quote': rates = rates.iloc[:1]
    if mutation == 'missing_apf': revision['rows'].pop()
    if mutation == 'duplicate_quote': rates = pd.concat([rates, rates.iloc[:1]])
    if mutation == 'duplicate_apf': revision['rows'].append(revision['rows'][0])
    with pytest.raises(ValueError, match=reason): run(revision, rates, origin)


@pytest.mark.parametrize('kwargs', [{'packet_kwh': 1}, {'packet_kwh': 0},
    {'battery_discharge_efficiency': 1.1}, {'battery_discharge_efficiency': float('nan')},
    {'terminal_value_aud_per_dc_kwh': -.1}])
def test_settings_and_power(kwargs):
    with pytest.raises(ValueError): run(*fixture(), **kwargs)


def test_partial_current_interval_excluded():
    revision, rates, origin = fixture()
    revision['rows'].insert(0, {'end_time': '2026-09-27T00:05:00Z', 'duration': 5,
        'advanced_price_predicted': -999, 'advanced_price_low': -999})
    assert run(revision, rates, origin)['first_future_slot'].endswith('00:05:00+00:00')


def test_30minute_prices_expand_without_inventing_prices():
    origin = '2026-09-27T00:00:00Z'
    revision = {'receipt': origin, 'rows': [{'end_time': '2026-09-27T00:30:00Z',
        'duration': 30, 'advanced_price_predicted': -.2, 'advanced_price_low': -.15}]}
    rates = pd.DataFrame({'rate': [.1]*6}, index=pd.date_range(origin, periods=6, freq='5min'))
    result = evaluate(revision, rates, origin, .5, battery_discharge_efficiency=1, wear_aud_per_dc_kwh=0, inverter_efficiency=1)
    assert result['five_minute_slots'] == 6
    assert result['candidates']['predicted']['forecast_incremental_value_aud'] == .05


def test_changed_apf_hash_rejected(tmp_path):
    archive, quotes = tmp_path/'apf', tmp_path/'quotes'
    archive.mkdir(); quotes.mkdir()
    (archive/'revisions.json').write_text('{}')
    (archive/'manifest.json').write_text(json.dumps({'raw_revisions_sha256': 'wrong'}))
    with pytest.raises(ValueError, match='changed APF'): load_inputs(archive, quotes)


def test_dc_throughput_wear_differs_from_stock_value():
    revision, rates, origin = fixture(prices=(1., 1.), realised=(1., 1.))
    result = evaluate(revision, rates, origin, .25, battery_discharge_efficiency=.8,
        inverter_efficiency=.5, wear_aud_per_dc_kwh=.1, terminal_value_aud_per_dc_kwh=.2)
    assert result['packet_discharge_dc_kwh'] == .2
    assert result['packet_export_ac_kwh'] == .1
    assert result['candidates']['predicted']['realised_incremental_value_aud'] == pytest.approx(.03)


@pytest.mark.parametrize('mutation', ['sign', 'hash', 'canonical'])
def test_quote_provenance_rejected(tmp_path, mutation):
    from eval.amber_quote_actuals import canonical_rates
    archive, quotes = tmp_path/'apf', tmp_path/'quotes'
    archive.mkdir(); quotes.mkdir()
    apf = archive/'revisions.json'; apf.write_text('{}')
    (archive/'manifest.json').write_text(json.dumps({'raw_revisions_sha256': hashlib.sha256(apf.read_bytes()).hexdigest()}))
    row = {'time': '2026-09-27T00:05:23Z', 'start_time_str': '2026-09-27T00:00:01Z',
        'end_time_str': '2026-09-27T00:05:00Z', 'duration': 5, 'type_str': 'CurrentInterval',
        'estimate': 0, 'unit_of_measurement_str': '$/kWh', 'value': .2}
    (quotes/'feed_revisions.json').write_text(json.dumps([row]))
    rates, _ = canonical_rates([row], '2026-09-27T00:00:00Z', '2026-09-27T00:05:00Z')
    if mutation == 'canonical': rates.loc[:, 'rate'] = .3
    rates.to_parquet(quotes/'feed_rates.parquet')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in quotes.iterdir()}
    if mutation == 'hash': hashes['feed_rates.parquet'] = 'wrong'
    manifest = {'export_complete': True, 'start': '2026-09-27T00:00:00Z',
        'end': '2026-09-27T00:05:00Z', 'files': hashes, 'rate_conventions': {'feed_state':
        'positive export revenue; MQTT state negates raw API per_kwh' if mutation != 'sign' else 'wrong'}}
    (quotes/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, AssertionError)): load_inputs(archive, quotes)


def test_control_receipt_dedup_and_conflict():
    from eval.apf_packet_regret import add_control_histories
    revision = {'receipt': '2026-09-27T00:00:00Z', 'payload_sha256': 'a'}
    result, sources = add_control_histories([revision, revision], [])
    assert result == [revision]
    assert sources == []
    with pytest.raises(ValueError, match='conflicting'):
        add_control_histories([revision, revision | {'payload_sha256': 'b'}], [])


@pytest.mark.parametrize('count,hash_value,reason', [(201, None, 'cap'), (0, 'wrong', 'changed')])
def test_control_history_guards(tmp_path, count, hash_value, reason):
    from eval.apf_packet_regret import add_control_histories
    history = tmp_path/'history.json'
    history.write_text(json.dumps({'mpc_apf_feed_in': [{}]*count}))
    (tmp_path/'manifest.json').write_text(json.dumps({'export_complete': True,
        'history_sha256': hash_value or hashlib.sha256(history.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match=reason): add_control_histories([], [tmp_path])
