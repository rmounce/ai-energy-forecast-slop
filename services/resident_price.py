#!/usr/bin/env python3
"""Opt-in resident price shadow; production listener and publishers stay unchanged."""
from __future__ import annotations
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config_utils import load_config
from energy_pipeline.price_worker import PriceWorker
from energy_pipeline.acceptance import AcceptanceDecision
from energy_pipeline.accepted_store import AcceptedStore, StoreError
from energy_pipeline.publication import ShadowPublication, prepare_plan
from energy_pipeline.handoff import ENTITIES, build_handoff
from energy_pipeline.source_cache import SourceCache, SourceUnavailable, price_source_policies
from energy_pipeline.source_refresh import SourceRefreshers
from services.ha_listener import Listener, _install_signal_handlers

log = logging.getLogger('resident_price')
WORKER_TIMEOUT_SECONDS = 180
MAX_RSS_MIB = 2048


class WorkerTimeout(RuntimeError):
    pass


class WorkerResourceLimit(RuntimeError):
    pass


class ResidentPriceListener(Listener):
    def __init__(self, config, predictor=None, *, store=None, publication=None, handoff=False):
        super().__init__(config)
        if publication is not None and store is None:
            raise ValueError('local publication requires accepted checkpoint storage')
        if handoff and publication is None:
            raise ValueError('handoff shadow requires local publication')
        self.handoff = handoff
        self._publication_config = config
        self.store = store
        self.publication = publication
        if publication is not None:
            log.info('local publication startup abandoned pending=%s; fresh reconciliation required',
                     publication.abandon_pending())
        self.recovered = store.recover() if store is not None else None
        if self.recovered is not None:
            log.info('recovered historical shadow checkpoint run=%s time_current=%s; fresh reconciliation required',
                     self.recovered.payload['run_id'], self.recovered.time_current())
        self.predictor = predictor or PriceWorker(SourceCache(price_source_policies(config)))
        self.refreshers = SourceRefreshers(self.predictor.sources, self.trigger, self.shutdown) if isinstance(self.predictor, PriceWorker) else None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='price')
        self.completed = None
        self.last_decision = None
        ha = config['home_assistant']
        self._input_entities = {self.entity_id, *ha.get('solcast_entities', [])}
        if ha.get('solcast_last_polled_entity'):
            self._input_entities.add(ha['solcast_last_polled_entity'])
        self._input_generation = 0

    def _on_state_changed(self, data):
        if data.get('entity_id') in self._input_entities:
            self._input_generation += 1
            self.trigger.set()

    async def _subscribe_state_changed(self, ws):
        await super()._subscribe_state_changed(ws)
        # Reconcile latest source state after startup and every reconnect.
        self._input_generation += 1
        self.trigger.set()

    async def _run_predict_price(self):
        async with self.run_lock:
            loop = asyncio.get_running_loop()
            async def submit(function, *args):
                future = loop.run_in_executor(self.executor, function, *args)
                # Retrieve late exceptions even after deadline/cancellation.
                future.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                return await asyncio.shield(future)

            async def generate_and_check():
                result = await submit(self.predictor.predict)
                if result.rss_mib > MAX_RSS_MIB:
                    return result, None
                generation = self._input_generation
                source_generation = self.predictor.sources.generation if self.refreshers else None
                acceptance_started = time.monotonic()
                decision = await submit(self.predictor.evaluate_completion, result)
                if generation != self._input_generation:
                    decision = AcceptanceDecision(('ha_changed_during_acceptance',), True)
                elif self.refreshers and source_generation != self.predictor.sources.generation:
                    decision = AcceptanceDecision(('source_changed_during_acceptance',), True)
                log.info('resident acceptance checked run=%s elapsed=%.3fs accepted=%s',
                         result.outcome.run_id, time.monotonic()-acceptance_started, decision.accepted)
                if decision.accepted and self.store is not None and not self.shutdown.is_set():
                    saved = await submit(self.store.save, result, datetime.now(timezone.utc))
                    if generation != self._input_generation:
                        decision = AcceptanceDecision(('ha_changed_during_checkpoint',), True)
                    elif self.refreshers and source_generation != self.predictor.sources.generation:
                        decision = AcceptanceDecision(('source_changed_during_checkpoint',), True)
                    elif saved is not None and not saved.time_current():
                        decision = AcceptanceDecision(('completion_expired_during_checkpoint',), True)
                    if decision.accepted and self.publication is not None:
                        def rehearse():
                            plan = prepare_plan(saved, self._publication_config)
                            last_validation = AcceptanceDecision()
                            def revalidate():
                                nonlocal last_validation
                                last_validation = self.predictor.evaluate_completion(result)
                                return (last_validation.accepted and not self.shutdown.is_set()
                                    and generation == self._input_generation
                                    and (not self.refreshers or source_generation == self.predictor.sources.generation))
                            committed = self.publication.execute(plan, revalidate)
                            if committed and self.handoff:
                                import forecast as fc
                                started = datetime.now(timezone.utc).isoformat()
                                rows = fc.call_ha_api('GET', 'states')
                                if not isinstance(rows, list):
                                    raise StoreError('cannot capture handoff HA inputs')
                                snapshot = {'capture_started_at': started,
                                    'captured_at': datetime.now(timezone.utc).isoformat(),
                                    'timezone': self._publication_config['timezone'],
                                    'states': {row['entity_id']: row for row in rows if row['entity_id'] in ENTITIES}}
                                record = build_handoff(plan, snapshot)
                                if saved.time_current() and revalidate():
                                    self.publication.save_handoff(plan, record)
                                    log.info('handoff shadow run=%s readiness=%s', result.outcome.run_id, record['readiness'])
                                else:
                                    committed = False
                            log.info('local publication rehearsal run=%s committed=%s', result.outcome.run_id, committed)
                            return committed, last_validation
                        committed, validation = await submit(rehearse)
                        if (not committed or generation != self._input_generation
                            or (self.refreshers and source_generation != self.predictor.sources.generation)
                            or not saved.time_current()):
                            decision = validation if not validation.accepted else AcceptanceDecision(('local_publication_superseded',), True)
                return result, decision
            try:
                result, decision = await asyncio.wait_for(generate_and_check(), WORKER_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as exc:
                self.shutdown.set()
                log.error('resident worker deadline exceeded; process restart required')
                # Cancelling an await cannot stop its thread. No retry or second
                # inference in this process; main exits to let supervision restart it.
                raise WorkerTimeout('price worker exceeded deadline') from exc
            except asyncio.CancelledError:
                self.shutdown.set()
                raise
            except StoreError:
                self.shutdown.set()
                raise
            except SourceUnavailable as exc:
                # Dependency acquisition will arm another trigger on readiness;
                # do not penalize initial cache warming with a five-minute retry.
                log.info('resident inputs not ready: %s', exc)
                return
            except Exception:
                log.exception('resident shadow generation failed')
                self._schedule_failure_retry()
                return
            if self.shutdown.is_set():
                return
            if result.rss_mib > MAX_RSS_MIB:
                self.shutdown.set()
                raise WorkerResourceLimit(f'resident RSS {result.rss_mib:.1f} MiB exceeds {MAX_RSS_MIB} MiB')
            self.last_decision = decision
            if not decision.accepted:
                log.info('resident shadow rejected run=%s reasons=%s', result.outcome.run_id, decision.reasons)
                if decision.reconcile:
                    self._retry_not_before = 0.0
                    self.trigger.set()
                elif not any(reason.startswith(('inputs_unavailable:', 'input_changed:config')) for reason in decision.reasons):
                    self._schedule_failure_retry()
                return
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
        if self.refreshers is not None:
            tasks.extend(asyncio.create_task(self.refreshers.loop(name)) for name in self.predictor.sources.policies)
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
            if self.refreshers is not None:
                self.refreshers.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lifetime = parser.add_mutually_exclusive_group()
    lifetime.add_argument('--runs', type=int, help='warm sources once, run N shadow predictions, then exit')
    lifetime.add_argument('--duration', type=float, help='run normal event/refresh loops for N seconds, then stop')
    parser.add_argument('--verify-reload', action='store_true', help='compare each quantile against a freshly loaded model on identical inputs')
    parser.add_argument('--state-file', type=Path, help='opt-in single-owner atomic accepted shadow checkpoint')
    parser.add_argument('--publication-db', type=Path, help='opt-in SQLite local publication rehearsal; no HA writes')
    parser.add_argument('--handoff-shadow', action='store_true', help='rehearse DH/MPC payload handoff locally; no solver requests')
    args = parser.parse_args()
    if args.publication_db and not args.state_file:
        parser.error('--publication-db requires --state-file')
    if args.handoff_shadow and not args.publication_db:
        parser.error('--handoff-shadow requires --publication-db')
    if args.runs is not None and args.runs < 1:
        parser.error('--runs must be positive')
    if args.duration is not None and args.duration <= 0:
        parser.error('--duration must be positive')
    config = load_config(ROOT / 'config.yaml')
    if not config['home_assistant'].get('token'):
        parser.error('home_assistant.token missing')
    store = AcceptedStore(args.state_file) if args.state_file else None
    publication = None
    try:
        publication = ShadowPublication(args.publication_db) if args.publication_db else None
        listener = ResidentPriceListener(config, PriceWorker(SourceCache(price_source_policies(config)), verify_reload=args.verify_reload), store=store, publication=publication, handoff=args.handoff_shadow)
    except BaseException:
        if publication is not None:
            publication.close()
        if store is not None:
            store.close()
        raise

    async def run():
        _install_signal_handlers(listener, asyncio.get_running_loop())
        if args.runs is None:
            async def expire():
                await asyncio.sleep(args.duration)
                listener.shutdown.set()
            timer = asyncio.create_task(expire()) if args.duration is not None else None
            try:
                await listener.run()
            finally:
                if timer:
                    timer.cancel()
                    with suppress(asyncio.CancelledError):
                        await timer
        else:
            try:
                # Finite benchmark warms each dependency once; subsequent runs
                # measure inference against the same cached data, not API latency.
                await asyncio.gather(*(listener.refreshers.refresh(name) for name in listener.predictor.sources.policies))
                for _ in range(args.runs):
                    if listener.shutdown.is_set():
                        break
                    previous = listener.completed
                    await listener._run_predict_price()
                    if listener._retry_not_before or listener.completed is previous:
                        raise RuntimeError('shadow benchmark failed or inputs not ready')
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
        if listener.refreshers is not None:
            listener.refreshers.close()
    # Python normally joins executor threads at exit. A stuck worker would
    # prevent supervision restart. A timed-out checkpoint may finish late;
    # recovery treats every surviving record as historical, never a publication lease.
    # Keep the file ownership lock until process exit, including late worker activity.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == '__main__':
    main()
