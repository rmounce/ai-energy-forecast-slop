from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import threading
import io
import zipfile
import pandas as pd
import forecast

from energy_pipeline.freshness import (
    FreshnessEvidence, collect_freshness, record_evidence, record_http_response)


def test_expired_cached_response_keeps_original_time():
    original = datetime.now(timezone.utc)-timedelta(hours=2)
    response = SimpleNamespace(from_cache=True, is_expired=True, created_at=original)
    with collect_freshness() as evidence:
        record_http_response('aemo', response)
    assert evidence == [FreshnessEvidence('aemo', 'http_response_created', original.isoformat(), True, True)]


def test_cached_response_with_unknown_time_is_never_retimed():
    with collect_freshness() as evidence:
        record_http_response('aemo', SimpleNamespace(from_cache=True))
    assert evidence[0].timestamp is None


def test_original_response_and_legacy_cache_times_are_utc():
    with collect_freshness() as evidence:
        record_http_response('fresh', SimpleNamespace())
        record_http_response('legacy', SimpleNamespace(from_cache=True, created_at=datetime(2026, 10, 2)))
    assert datetime.fromisoformat(evidence[0].timestamp).tzinfo == timezone.utc
    assert evidence[1].timestamp == '2026-10-02T00:00:00+00:00'


def test_collectors_are_isolated_between_threads_and_reset_after_exit():
    barrier = threading.Barrier(2)
    def run(name):
        with collect_freshness() as evidence:
            barrier.wait(timeout=2)
            record_evidence(FreshnessEvidence(name, 'unknown', None))
            with collect_freshness() as nested:
                record_evidence(FreshnessEvidence('nested', 'unknown', None))
            assert len(nested) == 1
            record_evidence(FreshnessEvidence(name, 'unknown', None))
        record_evidence(FreshnessEvidence('outside', 'unknown', None))
        return evidence
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, name) for name in ('history', 'aemo')]
        results = [future.result() for future in futures]
    assert [[item.source for item in result] for result in results] == [['history']*2, ['aemo']*2]


def test_aemo_report_captures_listing_and_payload_cache_times(monkeypatch):
    created = datetime(2026, 10, 1, tzinfo=timezone.utc)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('report.csv', 'I,INTERVAL_DATETIME,REGIONID,SCHEDULED_DEMAND,NET_INTERCHANGE\n'
                        'D,"2026/10/02 12:00:00",SA1,1200,0\n')
    responses = iter([
        SimpleNamespace(text='PUBLIC_SEVENDAYOUTLOOK_FULL_20261002110000_123.zip',
                        from_cache=True, is_expired=True, created_at=created, raise_for_status=lambda: None),
        SimpleNamespace(content=archive.getvalue(), from_cache=True, is_expired=False,
                        created_at=created+timedelta(minutes=1), raise_for_status=lambda: None)])
    monkeypatch.setattr(forecast, '_aemo_session', SimpleNamespace(get=lambda *args, **kwargs: next(responses)))
    monkeypatch.setattr(forecast, 'CONFIG', {'aemo_forecast': {'regions': ['SA1']}})
    with collect_freshness() as evidence:
        frame = forecast._get_aemo_7_day_outlook_forecast()
    assert frame.iloc[0]['total_demand_sa1'] == 1200
    assert [item.source for item in evidence] == ['nemweb_listing', 'nemweb_report']
    assert [item.timestamp for item in evidence] == [created.isoformat(), (created+timedelta(minutes=1)).isoformat()]
    assert evidence[0].expired and not evidence[1].expired


def test_aemo_short_term_records_transport_time_without_provider_issue_claim(monkeypatch):
    created = datetime(2026, 10, 1, tzinfo=timezone.utc)
    response = SimpleNamespace(from_cache=True, is_expired=True, created_at=created,
        raise_for_status=lambda: None, json=lambda: {'5MIN': [
            {'SETTLEMENTDATE': '2026-10-02 12:00:00', 'REGIONID': 'SA1', 'TOTALDEMAND': 1200, 'NETINTERCHANGE': 0}]})
    monkeypatch.setattr(forecast, '_aemo_session', SimpleNamespace(post=lambda *args, **kwargs: response))
    monkeypatch.setattr(forecast, 'CONFIG', {'aemo_forecast': {'regions': ['SA1']}})
    with collect_freshness() as evidence:
        frame = forecast._get_aemo_short_term_forecast()
    assert frame.index[0] == pd.Timestamp('2026-10-02T01:30:00Z')
    assert evidence == [FreshnessEvidence('aemo_short_term', 'http_response_created', created.isoformat(), True, True)]
