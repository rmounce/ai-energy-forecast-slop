from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
import json

import numpy as np
import pandas as pd
import pytest

from eval.archived_forecasts import parse_rows
from eval.audit_dh_source_history import source_coverage, load_alignment_score, source_states, SETTINGS
from eval.select_economic_windows import select_windows, REQUIRED


def source_fixture():
    # 72h horizon crosses Adelaide's spring clock change.
    dates = pd.date_range('2026-10-03T02:00Z', periods=144, freq='30min')
    states = {}
    for entity in ('sensor.ai_price_forecast', 'sensor.ai_price_forecast_low', 'sensor.ai_price_forecast_high'):
        states[entity] = {'attributes': {'forecasts': [dict(timestamp=at.isoformat(), general_price=.2,
            feed_in_price=.1) for at in dates]}}
    states['sensor.ai_load_forecast_high'] = {'attributes': {'forecasts': [
        dict(timestamp=at.isoformat(), power_load=500) for at in dates]}}
    for day in ('today', 'tomorrow', 'day_3', 'day_4'):
        states['sensor.solcast_pv_forecast_forecast_'+day] = {'attributes': {'detailedForecast': []}}
    states['sensor.solcast_pv_forecast_forecast_today']['attributes']['detailedForecast'] = [
        dict(period_start=at.tz_convert('Australia/Adelaide').isoformat(), pv_estimate=1,
            pv_estimate10=.5, pv_estimate90=2) for at in dates]
    return states, dates


def test_equal_length_forecasts_do_not_admit_half_hour_load_shift():
    states, dates = source_fixture()
    assert source_coverage(states, dates[0]+pd.Timedelta(seconds=40))['coverage_ready']
    for row in states['sensor.ai_load_forecast_high']['attributes']['forecasts']:
        row['timestamp'] = (pd.Timestamp(row['timestamp'])-pd.Timedelta(minutes=30)).isoformat()
    report = source_coverage(states, dates[0]+pd.Timedelta(seconds=40))
    assert report['source_counts']['load'] == 144
    assert report['reasons'] == ['load_targets:misaligned_or_incomplete']


def test_dst_constructor_archive_matches_json_and_correct_utc_targets():
    states, dates = source_fixture()
    rows = states['sensor.solcast_pv_forecast_forecast_today']['attributes']['detailedForecast']
    python_rows = deepcopy(rows)
    for row in python_rows:
        row['period_start'] = datetime.fromisoformat(row['period_start']).astimezone(ZoneInfo('Australia/Adelaide'))
    assert parse_rows(repr(python_rows)) == parse_rows(json.dumps(rows))
    assert {pd.Timestamp(row['period_start']).utcoffset().total_seconds() for row in rows} == {34200,37800}
    states['sensor.solcast_pv_forecast_forecast_today']['attributes']['detailedForecast'] = parse_rows(repr(python_rows))
    assert source_coverage(states, dates[0]+pd.Timedelta(seconds=40))['coverage_ready']


@pytest.mark.parametrize('raw', ["[{'x': __import__('os').getcwd()}]", "[{'x': (1).__class__}]",
    "[{'x': datetime.datetime(2026,1,1)}]", "[{'x': zoneinfo.ZoneInfo(key='Europe/London')}]",
    "[{'x': 1, 'x': 2}]", '[]', '[1]', '[x for x in []]'])
def test_unrecognised_archive_syntax_fails_closed(raw):
    with pytest.raises((ValueError, SyntaxError)): parse_rows(raw)


def test_timestamp_diagnostic_uses_same_vintage_overlap_and_no_tail_fill():
    dates = pd.date_range('2026-10-03T01:30Z', periods=144, freq='30min')
    rows = [dict(timestamp=at.isoformat(), power_load=i*100) for i,at in enumerate(dates)]
    measured = pd.DataFrame({'load_base_w': np.repeat(np.arange(1,144)*100.,6)},
        index=pd.date_range(dates[1], periods=143*6, freq='5min'))
    score = load_alignment_score(rows, dates[1], measured)
    assert score['paired_complete_targets'] == 143
    assert score['aligned_mae_w'] == 0
    assert score['shifted_mae_w'] == 100
    assert score['aligned_minus_shifted_14h_kwh'] == pytest.approx(1.4)
    measured.iloc[1,0] = np.nan
    assert load_alignment_score(rows,dates[1],measured)['paired_complete_targets'] == 142
    assert load_alignment_score(rows,dates[0],measured) is None


def test_later_updated_capture_cannot_supply_missing_historical_setting(monkeypatch):
    monkeypatch.setattr('eval.audit_dh_source_history.recorded_states',
        lambda history,captured,at,with_parent: (deepcopy(captured),{}))
    entity = SETTINGS['buy_weight']
    captured = {entity: {'state': '.5','last_updated': '2026-10-03T02:01Z'}}
    with pytest.raises(ValueError,match='no causal setting evidence: buy_weight'):
        source_states({},captured,'2026-10-03T02:00Z')


def regime_frame():
    frame = pd.DataFrame(1., index=pd.date_range('2026-09-30T21:00Z', periods=19, freq='5min'), columns=REQUIRED)
    frame['soc_pct_end'] = 80
    frame['feed_rate'] = .2
    frame['buy_rate'] = .4
    frame['pv_dc_w'] = 100
    frame['grid_import_w'] = -1000
    return frame


def test_regimes_are_deterministic_nonoverlapping_and_do_not_invent_negative_buy():
    frame = regime_frame()
    report = select_windows(frame)
    high, low = [report['selected'][key] for key in ('high_value_export','low_solar_export')]
    assert high['start'] == frame.index[1].isoformat()
    assert high['end'] == low['start']
    assert report == select_windows(frame)
    frame['soc_pct_end'] = 100
    frame['pv_dc_w'] = 4000
    report = select_windows(frame)
    assert report['candidate_counts']['near_full_negative_buy'] == 0
    assert report['candidate_counts']['near_full_pv'] > 0


def test_regimes_require_complete_targets_and_use_signed_raw_export_energy():
    frame = regime_frame()
    frame['grid_import_w_export_kwh'] = .3
    report = select_windows(frame)
    high = report['selected']['high_value_export']
    assert high['observed_export_kwh'] == pytest.approx(1.8)
    assert high['observed_export_revenue_aud'] == pytest.approx(.36)
    frame['pv_dc_w'] = np.nan
    assert select_windows(frame)['selected'] == {}
    frame = regime_frame()
    frame['soc_pct_end'] = 101
    with pytest.raises(ValueError,match='physical'): select_windows(frame)
    with pytest.raises(ValueError,match='ordered unique'): select_windows(pd.concat([frame,frame]))


def test_selector_rejects_off_grid_origins_and_missing_raw_export_energy():
    frame = regime_frame()
    frame.index += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError,match='boundaries'): select_windows(frame)
    frame = regime_frame()
    frame['grid_import_w_export_kwh'] = np.nan
    assert select_windows(frame)['selected'] == {}
