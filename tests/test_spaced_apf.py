import pandas as pd
import pytest

from eval.export_spaced_apf import schedule,query_for


def test_sampling_is_fixed_and_bounded_not_price_selected():
    rows = schedule('2026-09-27T00:00Z','2026-10-04T00:00Z')
    assert len(rows)==28 and set(at.hour for at in rows)=={0,6,12,18}
    assert rows[-1]==pd.Timestamp('2026-10-03T18:00Z')
    query = query_for(rows[0])
    assert 'ORDER BY time ASC LIMIT 1' in query
    assert '2026-09-27T00:02:00+00:00' in query


@pytest.mark.parametrize('start,end',[('2026-09-27','2026-10-04'),
    ('2026-09-27T00:00Z','2026-10-05T00:00Z'),('2026-09-27T00:01Z','2026-10-04T00:00Z'),
    ('2026-10-04T00:00Z','2026-09-27T00:00Z')])
def test_invalid_sample_windows_reject(start,end):
    with pytest.raises(ValueError): schedule(start,end)
