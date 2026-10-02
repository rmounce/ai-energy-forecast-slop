import asyncio
from datetime import datetime, timezone
import threading
from unittest.mock import patch
import pandas as pd
import pytest
from energy_pipeline.source_cache import SourceCache, SourcePolicy
from energy_pipeline.source_refresh import AcquiredSource, SourceRefreshers, SourceRefreshTimeout
import energy_pipeline.source_refresh as refresh_module
from energy_pipeline.freshness import FreshnessEvidence


def frame():
    return pd.DataFrame({'value': [1.] * 150}, index=pd.date_range(
        pd.Timestamp.now(tz='UTC').floor('30min'), periods=150, freq='30min'))


def test_source_timeout_never_commits_late_frame():
    async def run():
        cache = SourceCache({'aemo': SourcePolicy(('value',), 300, 1800)})
        trigger, shutdown, release = asyncio.Event(), asyncio.Event(), threading.Event()
        def acquire(name):
            release.wait(2)
            return AcquiredSource(frame())
        workers = SourceRefreshers(cache, trigger, shutdown, acquire)
        try:
            with patch.object(refresh_module, 'SOURCE_TIMEOUT_SECONDS', .01):
                with pytest.raises(SourceRefreshTimeout):
                    await workers.refresh('aemo')
            assert shutdown.is_set()
            release.set()
            await asyncio.sleep(.02)
            assert cache._sources == {}
            assert not trigger.is_set()
        finally:
            release.set()
            for executor in workers.executors.values(): executor.shutdown(wait=True)
    asyncio.run(run())


def test_slow_refresh_does_not_block_cached_consumption():
    async def run():
        cache = SourceCache({'aemo': SourcePolicy(('value',), 300, 1800)})
        cache.put('aemo', frame(), datetime.now(timezone.utc))
        trigger, shutdown, release = asyncio.Event(), asyncio.Event(), threading.Event()
        started = threading.Event()
        def acquire(name):
            started.set()
            release.wait(2)
            return AcquiredSource(frame()*2, (FreshnessEvidence('aemo', 'unknown', None),))
        workers = SourceRefreshers(cache, trigger, shutdown, acquire)
        try:
            task = asyncio.create_task(workers.refresh('aemo'))
            async def poll():
                while not started.is_set(): await asyncio.sleep(.001)
            await asyncio.wait_for(poll(), 1)
            assert cache.snapshot(datetime.now(timezone.utc))['aemo'].frame.iloc[0, 0] == 1
            assert not task.done()
            release.set()
            await asyncio.wait_for(task, 1)
            assert trigger.is_set()
            assert cache.snapshot(datetime.now(timezone.utc))['aemo'].frame.iloc[0, 0] == 2
            assert cache.snapshot(datetime.now(timezone.utc))['aemo'].evidence == (FreshnessEvidence('aemo', 'unknown', None),)
        finally:
            release.set()
            for executor in workers.executors.values(): executor.shutdown(wait=True)
    asyncio.run(run())


def test_weather_acquisition_explicitly_marks_unknown_provider_fetch(monkeypatch):
    monkeypatch.setattr(refresh_module, '_acquire_frame', lambda name: frame())
    acquired = refresh_module.acquire_source('weather')
    assert acquired.evidence == (FreshnessEvidence('bom', 'provider_fetch_unknown', None),)


def test_failed_refresh_preserves_snapshot_and_does_not_arm_trigger():
    async def run():
        cache = SourceCache({'aemo': SourcePolicy(('value',), 300, 1800)})
        now = datetime.now(timezone.utc)
        cache.put('aemo', frame(), now)
        def acquire(name): raise OSError('offline')
        trigger, shutdown = asyncio.Event(), asyncio.Event()
        workers = SourceRefreshers(cache, trigger, shutdown, acquire)
        try:
            with pytest.raises(OSError): await workers.refresh('aemo')
            assert not trigger.is_set()
            assert cache.snapshot(datetime.now(timezone.utc))['aemo'].fetched_at == now
        finally:
            for executor in workers.executors.values(): executor.shutdown(wait=True)
    asyncio.run(run())
