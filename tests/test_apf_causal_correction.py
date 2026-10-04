import copy

import pandas as pd
import pytest

from eval.apf_causal_correction import bucket, clusters, fit_bias, corrected_revision


def revision(receipt, target, predicted=-.10, duration=5):
    return {'receipt': pd.Timestamp(receipt).isoformat(), 'rows': [{'end_time': (pd.Timestamp(target)+pd.Timedelta(minutes=duration)).isoformat(),
        'duration': duration, 'advanced_price_predicted': predicted,
        'advanced_price_low': predicted+.02, 'advanced_price_high': predicted-.02}]}


def fixture():
    first = revision('2026-09-27T00:00Z', '2026-09-27T01:00Z')
    later = revision('2026-09-27T00:05Z', '2026-09-27T01:00Z', -.20)
    current = revision('2026-09-28T00:00Z', '2026-09-28T01:00Z')
    rows = [first, later, current]
    rates = pd.DataFrame({'rate': [.30], 'receipt': [pd.Timestamp('2026-09-27T01:03Z')]},
                         index=pd.DatetimeIndex(['2026-09-27T01:00Z']))
    return rows, rates


def fit(rows, rates, **kwargs):
    return fit_bias(rows, rates, rows[-1]['receipt'], clusters(rows),
        minimum_clusters=1, minimum_targets=1, ridge_targets=0, max_bias=1, **kwargs)


def test_newest_forecast_deduplicated_and_sign():
    rows, rates = fixture()
    fitted = fit(rows, rates)
    assert fitted['bins'][0]['targets'] == 1
    assert fitted['bins'][0]['bias_aud_per_kwh'] == pytest.approx(.10)
    assert fitted['training_labels'][0]['forecast_receipt'] == rows[1]['receipt']


@pytest.mark.parametrize('late_field', ['receipt', 'target_end'])
def test_late_or_equal_label_never_trains(late_field):
    rows, rates = fixture()
    if late_field == 'receipt':
        rates['receipt'] = pd.Timestamp(rows[-1]['receipt'])
    else:
        rows[-1]['receipt'] = '2026-09-27T01:05:00+00:00'
        # Explicit clusters test end cutoff independently of current-burst exclusion.
    groups = {r['receipt']: i if i == 2 else 0 for i, r in enumerate(rows)}
    fitted = fit_bias(rows, rates, rows[-1]['receipt'], groups, minimum_clusters=1, minimum_targets=1)
    assert fitted['training_labels'] == []


def test_current_cluster_and_future_receipts_excluded():
    rows, rates = fixture()
    rows[0]['receipt'] = '2026-09-28T00:00:00+00:00'
    rows[1]['receipt'] = '2026-09-28T00:05:00+00:00'
    rows[-1]['receipt'] = '2026-09-28T00:10:00+00:00'
    assert fit(rows, rates)['training_labels'] == []


def test_insufficient_history_zero_correction():
    rows, rates = fixture()
    fitted = fit_bias(rows, rates, rows[-1]['receipt'], clusters(rows))
    assert not fitted['supported']
    assert all(r['bias_aud_per_kwh'] == 0 for r in fitted['bins'])


def test_ridge_and_cap():
    rows, rates = fixture()
    fitted = fit_bias(rows, rates, rows[-1]['receipt'], clusters(rows), minimum_clusters=1,
        minimum_targets=1, ridge_targets=1, max_bias=.01)
    assert fitted['bins'][0]['bias_aud_per_kwh'] == .01


def test_release_lag_blocks_recent_label():
    rows, rates = fixture()
    assert fit(rows, rates, release_lag_minutes=24*60)['training_labels'] == []


def test_future_quote_mutation_cannot_change_fit():
    rows, rates = fixture()
    rates.loc[pd.Timestamp('2026-09-28T01:00Z')] = [.99, pd.Timestamp('2026-09-28T01:01Z')]
    before = fit(rows, rates)
    rates.loc[pd.Timestamp('2026-09-28T01:00Z'), 'rate'] = 999
    assert before == fit(rows, rates)


def test_split_horizon_boundary_and_preserve_original():
    source = revision('2026-09-28T00:00Z', '2026-09-28T01:50Z', duration=30)
    original = copy.deepcopy(source)
    fitted = {'bins': [{'bias_aud_per_kwh': .01}, {'bias_aud_per_kwh': -.02}, {'bias_aud_per_kwh': .03}]}
    changed = corrected_revision(source, source['receipt'], fitted)
    assert source == original
    assert len(changed['rows']) == 6
    assert changed['rows'][0]['advanced_price_predicted'] == pytest.approx(-.11)
    assert changed['rows'][2]['advanced_price_predicted'] == pytest.approx(-.08)


@pytest.mark.parametrize('hours,expected', [(0, 0), (2, 1), (6, 2), (14, None), (-1, None)])
def test_bins(hours, expected):
    assert bucket(hours) == expected


@pytest.mark.parametrize('settings', [{'ridge_targets': -1}, {'max_bias': float('nan')}, {'minimum_clusters': 0}])
def test_invalid_fit(settings):
    rows, rates = fixture()
    with pytest.raises(ValueError):
        fit_bias(rows, rates, rows[-1]['receipt'], clusters(rows), **settings)


def test_groups_hold_out_whole_utc_day_despite_six_hour_jitter():
    rows = [revision('2026-09-27T00:00Z', '2026-09-27T01:00Z'),
            revision('2026-09-27T12:00:01Z', '2026-09-27T13:00Z'),
            revision('2026-09-28T00:00Z', '2026-09-28T01:00Z')]
    groups = clusters(rows)
    assert groups[rows[0]['receipt']] == groups[rows[1]['receipt']]
    assert groups[rows[2]['receipt']] != groups[rows[0]['receipt']]


def test_preindexed_fit_matches_direct_fit():
    from eval.apf_causal_correction import prepare_labels
    rows, rates = fixture()
    groups = clusters(rows)
    prepared = prepare_labels(rows, rates, groups)
    assert fit_bias(rows, rates, rows[-1]['receipt'], groups, prepared=prepared) == fit_bias(rows, rates, rows[-1]['receipt'], groups)
