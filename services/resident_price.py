#!/usr/bin/env python3
"""Opt-in resident price shadow; production listener and publishers stay unchanged."""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_utils import load_config
from energy_pipeline.price_worker import PriceWorker
from services.ha_listener import Listener, _install_signal_handlers

log = logging.getLogger('resident_price')
WORKER_TIMEOUT_SECONDS = 180
MAX_RSS_MIB = 2048


class WorkerTimeout(RuntimeError):
    pass


class WorkerResourceLimit(RuntimeError):
    pass


class ResidentPriceListener(Listener):
    def __init__(self, config, predictor=None):
        super().__init__(config)
        self.predictor = predictor or PriceWorker()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='price')
        self.completed = None

    async def _subscribe_state_changed(self, ws):
        await super()._subscribe_state_changed(ws)
        # Reconcile latest source state after startup and every reconnect.
        self.trigger.set()

    async def _run_predict_price(self):
        async with self.run_lock:
            loop = asyncio.get_running_loop()
            future = loop.run_in_executor(self.executor, self.predictor.predict)
            # Retrieve late exceptions even when timeout/cancellation discards the result.
            future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            try:
                result = await asyncio.wait_for(asyncio.shield(future), WORKER_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as exc:
                self.shutdown.set()
                log.error('resident worker deadline exceeded; process restart required')
                # Cancelling an await cannot stop its thread. No retry or second
                # inference in this process; main exits to let supervision restart it.
                raise WorkerTimeout('price worker exceeded deadline') from exc
            except asyncio.CancelledError:
                self.shutdown.set()
                raise
            except Exception:
                log.exception('resident shadow generation failed')
                self._schedule_failure_retry()
                return
            if self.shutdown.is_set():
                return
            if result.rss_mib > MAX_RSS_MIB:
                self.shutdown.set()
                raise WorkerResourceLimit(f'resident RSS {result.rss_mib:.1f} MiB exceeds {MAX_RSS_MIB} MiB')
            self.completed = result
            self.last_run_at = time.monotonic()
            self._retry_not_before = 0.0
            log.info('resident shadow accepted run=%s parent=%s', result.outcome.run_id, result.parent_revision)

    def _record_health_status(self, exit_code):
        # A shadow must never overwrite the production listener's health record.
        log.info('resident shadow status=%s', exit_code)

    async def run(self):
        tasks = [asyncio.create_task(self.consume_websocket()),
                 asyncio.create_task(self.worker()), asyncio.create_task(self.heartbeat())]
        stop = asyncio.create_task(self.shutdown.wait())
        try:
            done, _ = await asyncio.wait([*tasks, stop], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task is not stop:
                    task.result()  # propagate a fatal worker deadline or ingress failure
            if not self.shutdown.is_set():
                raise RuntimeError('resident listener task stopped unexpectedly')
        finally:
            self.shutdown.set()
            for task in [*tasks, stop]:
                task.cancel()
            await asyncio.gather(*tasks, stop, return_exceptions=True)
            if self._retry_task:
                self._retry_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._retry_task
            self.executor.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, help='run N shadow predictions sequentially, then exit (benchmark)')
    parser.add_argument('--verify-reload', action='store_true', help='compare each quantile against a freshly loaded model on identical inputs')
    args = parser.parse_args()
    if args.runs is not None and args.runs < 1:
        parser.error('--runs must be positive')
    config = load_config(ROOT / 'config.yaml')
    if not config['home_assistant'].get('token'):
        parser.error('home_assistant.token missing')
    listener = ResidentPriceListener(config, PriceWorker(verify_reload=args.verify_reload))

    async def run():
        _install_signal_handlers(listener, asyncio.get_running_loop())
        if args.runs is None:
            await listener.run()
        else:
            try:
                for _ in range(args.runs):
                    if listener.shutdown.is_set():
                        break
                    await listener._run_predict_price()
                    if listener._retry_not_before:
                        raise RuntimeError('shadow benchmark failed')
            finally:
                listener.shutdown.set()
                if listener._retry_task:
                    listener._retry_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await listener._retry_task
                listener.executor.shutdown(wait=False, cancel_futures=True)
    rc = 0
    try:
        asyncio.run(run())
    except (Exception, KeyboardInterrupt):
        log.exception('resident shadow stopped')
        rc = 1
    finally:
        listener.executor.shutdown(wait=False, cancel_futures=True)
    # Python normally joins executor threads at exit. A stuck worker would
    # prevent systemd restart; shadow mode has no publication/file transaction
    # to commit, so terminate the process after flushing its diagnostic logs.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == '__main__':
    main()
