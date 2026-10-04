import json

import pytest

from eval.audit_amber_forecast_archive import parse_revision, select_asof


def record(feed=False):
    rows = []
    for minute in (5, 10):
        rows.append({'duration': 5, 'type': 'ForecastInterval',
            'start_time': f'2026-09-27T00:{minute-5:02}:01Z', 'end_time': f'2026-09-27T00:{minute:02}:00Z',
            'per_kwh': .2, 'spot_per_kwh': -.05, 'advanced_price_low': .3 if feed else .1,
            'advanced_price_predicted': .2, 'advanced_price_high': .1 if feed else .3})
    return {'time': '2026-09-27T00:00:25Z', 'unit_of_measurement_str': '$/kWh', 'Forecasts_str': repr(rows)}


def test_literal_json_and_feed_sign_order():
    row = record()
    first = parse_revision(row, 'general')
    row['Forecasts_str'] = json.dumps(first['rows'])
    second = parse_revision(row, 'general')
    assert first['intervals'] == second['intervals'] == 2
    assert first['start'] == '2026-09-27T00:00:00+00:00'
    assert first['payload_sha256'] != second['payload_sha256']
    assert parse_revision(record(True), 'feed_in')['intervals'] == 2
    with pytest.raises(ValueError, match='bounds'):
        parse_revision(record(True), 'general')


def test_asof_excludes_future_and_stale_receipts():
    row = parse_revision(record(), 'general')
    assert select_asof([row], '2026-09-27T00:00:24Z') is None
    assert select_asof([row], '2026-09-27T00:00:25Z') == row
    assert select_asof([row], '2026-09-27T00:16:00Z') is None


def test_conflicting_latest_receipt_is_not_arbitrarily_selected():
    row = parse_revision(record(), 'general')
    with pytest.raises(ValueError, match='conflicting'):
        select_asof([row, row | {'payload_sha256': 'different'}], row['receipt'])


@pytest.mark.parametrize('change', ['unit', 'naive', 'nan', 'gap', 'duplicate'])
def test_rejects_unusable_archive(change):
    row = record()
    import ast
    payload = ast.literal_eval(row['Forecasts_str'])
    if change == 'unit': row['unit_of_measurement_str'] = 'c/kWh'
    if change == 'naive': row['time'] = '2026-09-27T00:00:25'
    if change == 'nan': payload[0]['per_kwh'] = float('nan')
    if change == 'gap': payload.pop(0); payload[0]['start_time'] = '2026-09-27T00:00:01Z'
    if change == 'duplicate': payload[1] = payload[0].copy()
    row['Forecasts_str'] = json.dumps(payload)
    with pytest.raises(ValueError): parse_revision(row, 'general')
