from types import SimpleNamespace

import pandas as pd
import pytest

from energy_pipeline.freshness import collect_freshness
from energy_pipeline.weather import capture_weather, MARKERS


def adapter(markers):
    states = iter(markers)
    frames = []
    def read(method, endpoint):
        assert (method, endpoint) == ('GET', 'states/weather.hourly')
        attributes = next(states)
        return {'entity_id': 'weather.hourly', 'attributes': attributes}
    def forecast():
        frame = pd.DataFrame({'temperature': [len(frames)]})
        frames.append(frame)
        return frame
    return SimpleNamespace(CONFIG={'home_assistant': {'weather_entity': 'weather.hourly'}},
                           call_ha_api=read, get_weather_forecast=forecast), frames


def clocks(issue='2026-10-02T14:00:00+09:30', fetched='2026-10-02T04:35:00Z'):
    return dict(zip(MARKERS, (issue, fetched)))


def test_stable_clocks_are_normalized_and_do_not_use_capture_time():
    fc, frames = adapter([clocks(), clocks()])
    with collect_freshness() as evidence:
        result = capture_weather(fc)
    assert result is frames[0]
    assert [(item.basis, item.timestamp) for item in evidence] == [
        ('hourly_issue_observed', '2026-10-02T04:30:00+00:00'),
        ('hourly_successful_fetch_observed', '2026-10-02T04:35:00+00:00')]


def test_update_retries_and_discards_first_frame_and_its_clocks():
    new = clocks(fetched='2026-10-02T05:00:00Z')
    fc, frames = adapter([clocks(), new, new, new])
    with collect_freshness() as evidence:
        result = capture_weather(fc)
    assert result is frames[1]
    assert len(evidence) == 2
    assert evidence[1].timestamp == '2026-10-02T05:00:00+00:00'


def test_repeated_updates_fail_without_freshness_evidence():
    fc, frames = adapter([clocks(), {}, {}, clocks()])
    with collect_freshness() as evidence:
        with pytest.raises(RuntimeError, match='both capture attempts'):
            capture_weather(fc)
    assert len(frames) == 2
    assert evidence == []


@pytest.mark.parametrize('attributes', [{}, clocks(None, None), clocks('bad', '2026-10-02T04:35:00')])
def test_absent_null_or_invalid_clocks_remain_unknown(attributes):
    fc, _ = adapter([attributes, attributes])
    with collect_freshness() as evidence:
        capture_weather(fc)
    assert [(item.basis, item.timestamp) for item in evidence] == [
        ('provider_issue_unknown', None), ('provider_fetch_unknown', None)]


def test_failed_state_read_cannot_commit_a_frame():
    fc, frames = adapter([None])
    with pytest.raises(RuntimeError, match='attributes unavailable'):
        capture_weather(fc)
    assert not frames
