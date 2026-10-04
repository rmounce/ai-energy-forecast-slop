import pandas as pd
import pytest

from eval.amber_quote_actuals import accounting, adjusted_rates, canonical_rates

START = '2026-09-27T00:00:00Z'
END = '2026-09-27T00:05:00Z'


def quote(**changes):
    row = {'time': '2026-09-27T00:05:23Z', 'start_time_str': '2026-09-27T00:00:01Z',
           'end_time_str': END, 'duration': 5, 'type_str': 'CurrentInterval',
           'estimate': 0, 'unit_of_measurement_str': '$/kWh', 'value': 0.2}
    return row | changes


def adjusted(**changes):
    return {'time': '2026-09-27T00:05:24Z', 'confirmed_end_time_str': END,
            'unit_of_measurement_str': '$/kWh', 'raw_price': 0.2,
            'export_allowance_adjustment': 0.01, 'value': 0.21} | changes


def test_receipt_is_not_priced_interval():
    rates, errors = canonical_rates([quote()], START, END)
    assert not errors
    assert rates.index[0] == pd.Timestamp(START)
    assert rates.iloc[0].receipt_lag_seconds == 23


@pytest.mark.parametrize('changes,reason', [
    ({'estimate': 1}, 'not_confirmed_current_interval'),
    ({'type_str': 'ForecastInterval'}, 'not_confirmed_current_interval'),
    ({'unit_of_measurement_str': 'c/kWh'}, 'unit_mismatch'),
    ({'duration': 30}, 'unsupported_duration'),
    ({'start_time_str': START}, 'start_convention_mismatch'),
    ({'value': float('inf')}, 'invalid_attributes'),
    ({'time': '2026-09-26T23:59:00Z'}, 'receipt_before_interval')])
def test_invalid_quote_excluded(changes, reason):
    rates, errors = canonical_rates([quote(**changes)], START, END)
    assert rates.empty
    assert errors[reason] == 1


def test_invalid_latest_never_falls_back():
    rates, errors = canonical_rates([quote(), quote(time='2026-09-27T00:06:00Z', estimate=1)], START, END)
    assert rates.empty
    assert errors['not_confirmed_current_interval'] == 1


def test_conflicting_latest_revision_rejected():
    rates, errors = canonical_rates([quote(), quote(value=0.3)], START, END)
    assert rates.empty
    assert errors['conflicting_latest_revision'] == 1


def test_latest_revision_and_history_count():
    rates, errors = canonical_rates([quote(), quote(value=0.3, time='2026-09-27T00:06:00Z')], START, END)
    assert not errors
    assert rates.iloc[0]['rate'] == 0.3
    assert rates.iloc[0].revision_count == 2


def test_adjustment_requires_at_receipt_and_final_quote_match():
    rows = [quote()]
    raw, _ = canonical_rates(rows, START, END)
    accepted, errors = adjusted_rates([adjusted()], raw, rows)
    assert not errors and accepted.iloc[0]['rate'] == 0.21
    rejected, errors = adjusted_rates([adjusted(time='2026-09-27T00:05:22Z')], raw, rows)
    assert rejected.empty and errors['no_raw_quote_at_receipt'] == 1
    revised_rows = rows+[quote(time='2026-09-27T00:06:00Z', value=0.3)]
    revised, _ = canonical_rates(revised_rows, START, END)
    rejected, errors = adjusted_rates([adjusted()], revised, revised_rows)
    assert rejected.empty and errors['final_raw_revision_mismatch'] == 1


def test_adjustment_arithmetic_and_conflict():
    rows = [quote()]
    raw, _ = canonical_rates(rows, START, END)
    rejected, errors = adjusted_rates([adjusted(value=0.22)], raw, rows)
    assert rejected.empty and errors['adjustment_identity_mismatch'] == 1
    rejected, errors = adjusted_rates([adjusted(), adjusted(value=0.22)], raw, rows)
    assert rejected.empty and errors['conflicting_latest_revision'] == 1


def test_cashflow_sign_and_complete_energy_only():
    idx = pd.date_range(START, periods=2, freq='5min')
    actual = pd.DataFrame({'grid_import_w_import_kwh': [1, float('nan')],
                           'grid_import_w_export_kwh': [2, 3]}, index=idx)
    general = pd.DataFrame({'rate': [.2, .2]}, index=idx)
    feed = pd.DataFrame({'rate': [.1, .1]}, index=idx)
    adj = pd.DataFrame({'rate': [.11, .11]}, index=idx)
    priced, result = accounting(actual, general, feed, adj)
    assert priced.iloc[0].raw_cost == 0
    assert pd.isna(priced.iloc[1].raw_cost)
    assert result['raw_priced_energy_intervals'] == 1
    assert result['paired_adjustment_cost_dollars'] == pytest.approx(-.02)
    assert result['raw_export_revenue_dollars'] == pytest.approx(.2)
    assert result['raw_chargeable_export_cost_dollars'] == 0


def test_negative_mqtt_feed_state_means_paid_export():
    idx = pd.date_range(START, periods=1, freq='5min')
    actual = pd.DataFrame({'grid_import_w_import_kwh': [0.], 'grid_import_w_export_kwh': [2.]}, index=idx)
    general = pd.DataFrame({'rate': [.2]}, index=idx)
    feed = pd.DataFrame({'rate': [-.1]}, index=idx)
    adj = pd.DataFrame({'rate': [-.09]}, index=idx)
    _, result = accounting(actual, general, feed, adj)
    assert result['raw_observed_cost_dollars'] == pytest.approx(.2)
    assert result['raw_chargeable_export_cost_dollars'] == pytest.approx(.2)
    assert result['raw_export_revenue_dollars'] == 0
    assert result['chargeable_export_kwh'] == 2


def test_pre_end_receipt_preserved_with_separate_sensitivity():
    row = quote(time='2026-09-27T00:00:23Z')
    rates, errors = canonical_rates([row], START, END)
    assert not errors and not rates.iloc[0].received_at_or_after_end
    rates, errors = canonical_rates([row], START, END, require_post_end_receipt=True)
    assert rates.empty and errors['latest_receipt_before_interval_end'] == 1


def test_mqtt_rollover_new_value_with_old_interval_is_not_a_quote_revision():
    stable = quote(value=.0261, time='2026-09-27T00:00:16Z', update_time_str='2026-09-27T00:00:16')
    transitional = quote(value=.03, time='2026-09-27T00:05:19.395265Z',
                         update_time_str='2026-09-27T00:00:16')
    next_interval = quote(value=.03, time='2026-09-27T00:05:19.396310Z',
        start_time_str='2026-09-27T00:05:01Z', end_time_str='2026-09-27T00:10:00Z',
        update_time_str='2026-09-27T00:05:19')
    rates, errors = canonical_rates([stable, transitional, next_interval], START, END)
    assert rates.iloc[0]['rate'] == .0261
    assert errors['state_attribute_transition'] == 1
    # Without an already received successor, do not use future evidence.
    unpaired, _ = canonical_rates([stable, transitional], START, END)
    assert unpaired.iloc[0]['rate'] == .03


def test_same_interval_genuine_revision_not_removed_by_rollover_filter():
    a = quote(value=.2, time='2026-09-27T00:04:00Z', update_time_str='a')
    b = quote(value=.3, time='2026-09-27T00:04:00.001Z', update_time_str='b')
    rates, errors = canonical_rates([a, b], START, END)
    assert not errors and rates.iloc[0]['rate'] == .3


def test_adjusted_rollover_retains_matching_complete_raw_quote():
    stable = quote(time='2026-09-27T00:00:16Z', update_time_str='a')
    old_metadata = quote(value=.3, time='2026-09-27T00:05:19.001Z', update_time_str='a')
    next_metadata = quote(value=.3, time='2026-09-27T00:05:19.002Z', update_time_str='b',
        start_time_str='2026-09-27T00:05:01Z', end_time_str='2026-09-27T00:10:00Z')
    raw_rows = [stable, old_metadata, next_metadata]
    raw, _ = canonical_rates(raw_rows, START, END)
    adjusted_rows = [adjusted(time='2026-09-27T00:00:16.001Z'),
        adjusted(time='2026-09-27T00:05:19.003Z', value=.31, raw_price=.3),
        adjusted(time='2026-09-27T00:05:19.004Z', value=.31, raw_price=.3,
                 confirmed_end_time_str='2026-09-27T00:10:00Z')]
    result, errors = adjusted_rates(adjusted_rows, raw, raw_rows)
    assert result.iloc[0]['rate'] == .21
    assert errors['state_attribute_transition'] == 1
