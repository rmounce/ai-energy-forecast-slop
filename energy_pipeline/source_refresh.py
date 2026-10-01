"""Independent source workers; no forecast or publication work on these threads."""
from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import logging
import time

from energy_pipeline.source_cache import SourceCache

log = logging.getLogger('source_refresh')
SOURCE_TIMEOUT_SECONDS = 180


class SourceRefreshTimeout(RuntimeError):
    pass


def acquire_source(name):
    import forecast as fc
    if name == 'aemo':
        return fc.get_aemo_forecast()
    if name == 'weather':
        return fc.get_weather_forecast()
    if name == 'history':
        start = datetime.now(timezone.utc)
        start = start.replace(minute=start.minute//30*30, second=0, microsecond=0)
        client = fc.InfluxDBClient(**fc.CONFIG['influxdb'])
        try:
            return fc.get_historical_data(client, start-timedelta(days=fc.CONFIG['prediction_history_days']),
                                          start-timedelta(minutes=30))
        finally:
            client.close()
    raise ValueError(f'unknown source: {name}')


class SourceRefreshers:
    def __init__(self, cache: SourceCache, trigger, shutdown, acquire=acquire_source):
        self.cache, self.trigger, self.shutdown, self.acquire = cache, trigger, shutdown, acquire
        self.executors = {name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
                          for name in cache.policies}

    async def refresh(self, name):
        started = time.monotonic()
        # Timestamp the request start, conservatively including slow acquisition
        # in its age. Provider issuance age is a separate, still-unverified contract.
        fetched_at = datetime.now(timezone.utc)
        future = asyncio.get_running_loop().run_in_executor(self.executors[name], self.acquire, name)
        future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        try:
            frame = await asyncio.wait_for(asyncio.shield(future), SOURCE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            self.shutdown.set()
            raise SourceRefreshTimeout(f'{name}: acquisition deadline exceeded') from exc
        if self.shutdown.is_set():
            return
        changed = self.cache.put(name, frame, fetched_at)
        if changed:
            self.trigger.set()
        log.info('source refreshed name=%s changed=%s elapsed=%.3fs', name, changed, time.monotonic()-started)

    async def loop(self, name):
        while not self.shutdown.is_set():
            try:
                await self.refresh(name)
            except SourceRefreshTimeout:
                raise
            except Exception:
                # Keep previous snapshot; snapshot admission checks its age and
                # coverage every run. Failed refresh never resets usable-data age.
                log.exception('source refresh failed: %s; retaining last usable snapshot', name)
            try:
                await asyncio.wait_for(self.shutdown.wait(), self.cache.policies[name].refresh_seconds)
            except asyncio.TimeoutError:
                pass

    def close(self):
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
