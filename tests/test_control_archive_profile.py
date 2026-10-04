import pandas as pd
import pytest

from eval.export_control_history import archive_profile, SOURCES


def test_long_cycle_profile_cannot_expand_into_forecast_controller_queries():
    start = pd.Timestamp('2026-09-27T08:40Z')
    end = start+pd.Timedelta(hours=21)
    sources,budget = archive_profile(start,end,cycle_support=True)
    assert set(sources)=={'pv','pv1','pv2','battery','inverter_ac','loss','soc','last_full'}
    assert budget==100000
    for flag in ('dh','ems','energy'):
        with pytest.raises(ValueError,match='cannot include'):
            archive_profile(start,end,cycle_support=True,**{flag:True})


def test_existing_control_profile_preserves_small_query_bound():
    start = pd.Timestamp('2026-09-27T08:40Z')
    sources,budget = archive_profile(start,start+pd.Timedelta(minutes=90))
    assert sources==SOURCES and budget==10000
    with pytest.raises(ValueError,match='duration bound'):
        archive_profile(start,start+pd.Timedelta(minutes=91))


@pytest.mark.parametrize('start,end',[('2026-09-27','2026-09-28'),
    ('2026-09-27T00:00Z','2026-09-28T00:01Z'),('2026-09-28T00:00Z','2026-09-27T00:00Z')])
def test_cycle_duration_and_timezone_bounds(start,end):
    with pytest.raises(ValueError): archive_profile(pd.Timestamp(start),pd.Timestamp(end),cycle_support=True)
