from datetime import timedelta
import numpy as np
import pandas as pd
import pytest
from energy_pipeline.source_cache import SourceCache, SourcePolicy, SourceUnavailable
from energy_pipeline.freshness import FreshnessEvidence

START = pd.Timestamp('2026-10-03T16:00:00Z')
POLICY = SourcePolicy(('value',), 300, 1800)


def frame(periods=150, start=START):
    return pd.DataFrame({'value': np.arange(periods, dtype=float)},
                        index=pd.date_range(start, periods=periods, freq='30min'))


def test_snapshots_are_isolated_and_unchanged_refresh_has_stable_revision():
    cache = SourceCache({'aemo': POLICY})
    data = frame()
    assert cache.put('aemo', data, START)
    revision = cache.snapshot(START)['aemo'].revision
    data.iloc[0, 0] = -100
    snapshot = cache.snapshot(START)
    snapshot['aemo'].frame.iloc[0, 0] = -200
    assert cache.snapshot(START)['aemo'].frame.iloc[0, 0] == 0
    assert not cache.put('aemo', frame(), START+timedelta(minutes=5))
    assert cache.snapshot(START+timedelta(minutes=5))['aemo'].revision == revision


def test_failed_refresh_retains_last_good_without_resetting_age():
    cache = SourceCache({'aemo': POLICY})
    evidence = [FreshnessEvidence('aemo', 'http_response_created', START.isoformat(), True, False)]
    cache.put('aemo', frame(), START, evidence=evidence)
    evidence.clear()
    with pytest.raises(SourceUnavailable):
        cache.put('aemo', frame(10), START+timedelta(minutes=10))
    assert cache.snapshot(START+timedelta(minutes=29))['aemo'].fetched_at == START
    assert cache.snapshot(START+timedelta(minutes=29))['aemo'].evidence[0].timestamp == START.isoformat()
    with pytest.raises(SourceUnavailable, match='age'):
        cache.snapshot(START+timedelta(minutes=31))
    # A recovered identical source must re-arm dependent work.
    assert cache.put('aemo', frame(), START+timedelta(minutes=31))


def test_coverage_is_rechecked_after_interval_rollover_and_dst():
    cache = SourceCache({'aemo': SourcePolicy(('value',), 300, 3600)})
    cache.put('aemo', frame(144), START)
    assert len(cache.snapshot(START+timedelta(minutes=29))['aemo'].frame) == 144
    with pytest.raises(SourceUnavailable, match='coverage'):
        cache.snapshot(START+timedelta(minutes=30))
    # UTC intervals cross Adelaide's skipped local hour without duplicate targets.
    cache.put('aemo', frame(150), START+timedelta(minutes=30))
    assert cache.snapshot(START+timedelta(minutes=31))['aemo'].frame.index.is_unique


@pytest.mark.parametrize('mutation', ['missing', 'nan', 'inf', 'duplicate', 'reverse', 'naive'])
def test_invalid_source_cannot_replace_usable_snapshot(mutation):
    cache = SourceCache({'aemo': POLICY})
    cache.put('aemo', frame(), START)
    data = frame()
    if mutation == 'missing': data = data.rename(columns={'value': 'other'})
    elif mutation == 'nan': data.iloc[3, 0] = np.nan
    elif mutation == 'inf': data.iloc[3, 0] = np.inf
    elif mutation == 'duplicate': data = pd.concat([data, data.iloc[-1:]])
    elif mutation == 'reverse': data = data.iloc[::-1]
    elif mutation == 'naive': data.index = data.index.tz_localize(None)
    with pytest.raises((SourceUnavailable, TypeError)):
        cache.put('aemo', data, START)
    assert cache.snapshot(START)['aemo'].frame.iloc[3, 0] == 3


def test_missing_source_out_of_order_and_future_capture_are_rejected():
    cache = SourceCache({'aemo': POLICY})
    with pytest.raises(SourceUnavailable, match='not acquired'):
        cache.snapshot(START)
    cache.put('aemo', frame(), START+timedelta(minutes=1))
    with pytest.raises(SourceUnavailable, match='out-of-order'):
        cache.put('aemo', frame(), START)
    with pytest.raises(SourceUnavailable, match='age'):
        cache.snapshot(START)


def test_history_checks_observation_age_and_lookback_independently_of_fetch():
    policy = SourcePolicy(('value',), 300, 2400, historical=True, min_history_hours=24)
    cache = SourceCache({'history': policy})
    history = frame(49, START-timedelta(hours=24.5))
    cache.put('history', history, START)
    stale = history.iloc[:-3]
    with pytest.raises(SourceUnavailable, match='90 minutes'):
        cache.put('history', stale, START)
    with pytest.raises(SourceUnavailable, match='lookback'):
        cache.put('history', history.iloc[5:], START)


def test_only_confirmed_zero_capacity_ratio_nan_is_allowed_and_preserved():
    ratios = (('solar_ratio', 'solar_available', 'solar_capacity'),)
    policy = SourcePolicy(('solar_ratio', 'solar_available', 'solar_capacity'), 300, 1800,
                          zero_capacity_ratios=ratios)
    cache = SourceCache({'aemo': policy})
    data = frame().rename(columns={'value': 'solar_ratio'})
    data['solar_available'], data['solar_capacity'] = 0., 0.
    data.loc[data.index[0], 'solar_ratio'] = np.nan
    cache.put('aemo', data, START)
    assert pd.isna(cache.snapshot(START)['aemo'].frame.iloc[0]['solar_ratio'])
    bad = data.copy()
    bad.loc[bad.index[0], 'solar_capacity'] = 1
    with pytest.raises(SourceUnavailable, match='coverage'):
        cache.put('aemo', bad, START)
    all_missing = data.copy()
    all_missing['solar_ratio'] = np.nan
    with pytest.raises(SourceUnavailable, match='coverage'):
        cache.put('aemo', all_missing, START)
