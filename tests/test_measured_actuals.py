import numpy as np
import pandas as pd
import pytest

from eval.measured_actuals import clean_samples, endpoint_samples, integrate, window

START = pd.Timestamp('2026-10-01T00:00:00Z')


def samples(seconds, values):
    return pd.Series(values, index=START+pd.to_timedelta(seconds, unit='s'))


def test_irregular_observations_are_time_weighted_not_event_averaged():
    result = integrate(samples([0, 10, 290], [1000, 2000, 3000]), START,
                       START+pd.Timedelta(minutes=5), max_hold_seconds=300)
    assert result['coverage'].iloc[0] == 1
    assert result['mean'].iloc[0] == pytest.approx(2000)
    # More observations of the same value do not change integrated energy.
    repeated = integrate(samples([0, 10, 20, 30, 40, 290], [1000, 2000, 2000, 2000, 2000, 3000]),
                         START, START+pd.Timedelta(minutes=5), max_hold_seconds=300)
    pd.testing.assert_series_equal(result['mean'], repeated['mean'])


def test_expired_sample_is_missing_not_zero_or_infinite_hold():
    result = integrate(samples([0], [1200]), START, START+pd.Timedelta(minutes=10), max_hold_seconds=120)
    assert result['coverage'].tolist() == [.4, 0]
    assert result['mean'].isna().all()
    assert result['observed_mean'].iloc[0] == 1200
    assert np.isnan(result['observed_mean'].iloc[1])


def test_future_sample_never_fills_earlier_gap_and_nan_terminates_previous():
    result = integrate(samples([100, 200, 250], [1000, np.nan, 2000]), START,
                       START+pd.Timedelta(minutes=5), max_hold_seconds=300)
    assert result['coverage'].iloc[0] == .5
    assert result['observed_mean'].iloc[0] == pytest.approx(4000/3)
    assert np.isnan(result['mean'].iloc[0])


def test_pre_window_sample_and_cross_bin_hold_preserve_energy():
    result = integrate(samples([-60], [600]), START, START+pd.Timedelta(minutes=15), max_hold_seconds=1000)
    assert result['mean'].tolist() == [600]*3
    assert result['coverage'].tolist() == [1]*3
    assert (result['mean']*5/60/1000).sum() == pytest.approx(.15)


def test_crossing_boundary_allocates_each_side_and_excludes_right_endpoint():
    result = integrate(samples([290, 310, 600], [1000, 2000, 9999]), START,
                       START+pd.Timedelta(minutes=10), max_hold_seconds=300)
    assert result['coverage'].tolist() == pytest.approx([10/300, 1])
    assert result['mean'].iloc[1] == pytest.approx((10*1000+290*2000)/300)
    assert result['sample_count'].tolist() == [1, 1]


def test_identical_duplicates_collapse_conflicting_series_rejected():
    assert len(clean_samples(samples([0, 0], [1000, 1000]))) == 1
    with pytest.raises(ValueError, match='conflicting'):
        integrate(samples([0, 0], [1000, 2000]), START, START+pd.Timedelta(minutes=5))


def test_endpoint_soc_uses_past_observation_only_and_expires():
    result = endpoint_samples(samples([0, 299, 301], [50, 60, 70]),
        pd.DatetimeIndex([START-pd.Timedelta(seconds=1), START+pd.Timedelta(seconds=300),
                          START+pd.Timedelta(seconds=500)]), max_hold_seconds=120)
    assert np.isnan(result[0]) and result[1] == 60 and np.isnan(result[2])


def test_opposite_grid_directions_retain_energy_even_when_mean_is_zero():
    raw = samples([0, 150], [1000, -1000])
    end = START+pd.Timedelta(minutes=5)
    net = integrate(raw, START, end, max_hold_seconds=300)['mean'].iloc[0]
    imported = integrate(raw.clip(lower=0), START, end, max_hold_seconds=300)['mean'].iloc[0]/12000
    exported = integrate((-raw).clip(lower=0), START, end, max_hold_seconds=300)['mean'].iloc[0]/12000
    assert net == 0 and imported == pytest.approx(1/24) and exported == pytest.approx(1/24)


def test_empty_series_gives_no_coverage():
    result = integrate(samples([], []), START, START+pd.Timedelta(minutes=5))
    assert result['coverage'].iloc[0] == 0 and result['mean'].isna().all()


@pytest.mark.parametrize('start,end', [('2026-10-01', '2026-10-02'),
    ('2026-10-01T00:01Z', '2026-10-02T00:00Z'),
    ('2026-10-02T00:00Z', '2026-10-01T00:00Z'),
    ('2026-10-01T00:00Z', '2026-10-09T00:00Z')])
def test_unbounded_or_unaligned_window_rejected(start, end):
    with pytest.raises(ValueError): window(start, end)


def test_timezone_offsets_convert_to_utc_across_adelaide_dst():
    start, end = window('2026-10-04T01:30:00+09:30', '2026-10-04T03:30:00+10:30')
    assert end-start == pd.Timedelta(hours=1)
    assert str(start.tz) == 'UTC'
