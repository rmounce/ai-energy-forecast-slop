"""Calculation-only resident price adapter; no production output writes."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import time
import pandas as pd

from config_utils import load_config
from energy_pipeline.model_cache import ModelCache
from energy_pipeline.memory import reclaim_transient_memory
from energy_pipeline.freshness import FreshnessEvidence, serialize_evidence
from energy_pipeline.source_cache import SourceCache, SourcePolicy, validate_frame
from production_contract import PredictionInputs, PredictionOutcome, ForecastContractError, validate_forecast_family
from energy_pipeline.acceptance import AcceptanceDecision, compare_revisions
from energy_pipeline.source_cache import SourceUnavailable


def content_revision(value):
    """Opaque digest: provenance logs never contain configuration credentials."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_revision(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return 'missing'


def utc_now():
    return pd.Timestamp.now(tz='UTC')


@dataclass(frozen=True)
class PriceCompletion:
    outcome: PredictionOutcome
    parent_revision: str
    captured_at: str
    elapsed_seconds: float
    model_load_seconds: float
    model_loads: int
    rss_mib: float
    source_revisions: dict[str, str]
    source_ages_seconds: dict[str, float]
    memory_maintenance_seconds: float
    rss_before_maintenance_mib: float
    allocator_trimmed: bool
    input_revisions: dict[str, str]
    source_freshness: dict[str, list[dict]]


class PriceWorker:
    """Called by one dedicated executor thread, never concurrently.

    HA state is frozen once per run. Independent refresh workers supply validated
    weather/AEMO/history snapshots; inference never refreshes those sources.
    """
    def __init__(self, source_cache: SourceCache, *, verify_reload=False):
        self.sources = source_cache
        self.verify_reload = verify_reload
        self.cache = ModelCache()

    def predict(self):
        started = time.monotonic()
        import forecast as fc
        from tariff_utils import load_tariff_profile
        # Source workers read the same immutable process configuration. Reloading
        # its module globals while they run could mix inputs; restart on changes.
        if load_config() != fc.CONFIG:
            raise RuntimeError('configuration changed; restart resident shadow')
        fc.GENERAL_TARIFF_MAP, fc.FEED_IN_TARIFF_MAP, fc.NETWORK_LOSS_FACTOR = load_tariff_profile(fc.CONFIG, fc.ROOT)
        tariff_revision = file_revision(fc.ROOT / fc.CONFIG['paths']['tariff_file'])
        pointer_revision = file_revision(fc._production_bundle_store().active_pointer('price'))
        now = datetime.now(timezone.utc)
        sources = self.sources.snapshot(now)
        rows = fc.call_ha_api('GET', 'states')
        if not isinstance(rows, list):
            raise RuntimeError('cannot capture HA inputs')
        needed = set(fc.CONFIG['home_assistant']['solcast_entities'])
        needed.add(fc.CONFIG['home_assistant']['amber_billing_entity'])
        solcast_poll_entity = fc.CONFIG['home_assistant'].get('solcast_last_polled_entity')
        if solcast_poll_entity:
            needed.add(solcast_poll_entity)
        states = {row['entity_id']: row for row in rows if row['entity_id'] in needed}
        del rows
        apf = states.get(fc.CONFIG['home_assistant']['amber_billing_entity'])
        if not apf or not apf.get('attributes', {}).get('Forecasts'):
            raise RuntimeError('snapshot missing Amber APF')
        captured_at = datetime.now(timezone.utc).isoformat()
        revision = content_revision(apf)
        freshness = {name: serialize_evidence(source.evidence) for name, source in sources.items()}
        freshness['apf'] = serialize_evidence([
            FreshnessEvidence('apf', 'ha_entity_updated', apf.get('last_updated')),
            # Preserve bridge's naive clock string. It is neither an aware
            # source-issue time nor proof of a successful Amber API fetch.
            FreshnessEvidence('apf', 'bridge_publication_naive_clock',
                              apf['attributes'].get('update_time'))])
        poll = states.get(solcast_poll_entity, {})
        freshness['solcast'] = serialize_evidence([FreshnessEvidence('solcast',
            'integration_successful_fetch' if poll else 'provider_fetch_unknown', poll.get('state'))])
        input_revisions = {
            'config': content_revision(fc.CONFIG),
            'tariff': content_revision([fc.GENERAL_TARIFF_MAP, fc.FEED_IN_TARIFF_MAP, fc.NETWORK_LOSS_FACTOR]),
            'tariff_file': tariff_revision,
            'model_pointer': pointer_revision,
            'apf': revision,
            **{f'ha:{entity}': content_revision(states.get(entity))
               for entity in fc.CONFIG['home_assistant']['solcast_entities']},
            **{f'source:{name}': source.revision for name, source in sources.items()},
        }
        if solcast_poll_entity:
            input_revisions[f'ha:{solcast_poll_entity}'] = content_revision(poll)
        with fc.prediction_resources(entity_states=states, model_cache=self.cache, verify_reload=self.verify_reload):
            solcast = fc.get_solcast_forecast()
            # Solcast's four-day horizon can span DST and produce an object
            # index of mixed offsets. Match the incumbent preparation's UTC
            # normalization before admission, rather than rejecting valid rows.
            solcast.index = pd.to_datetime(solcast.index, utc=True)
            start = now.replace(minute=now.minute//30*30, second=0, microsecond=0)
            validate_frame(solcast, SourcePolicy(('power_pv',), 0, 0), start)
            inputs = PredictionInputs(
                {'solcast': solcast, 'weather': sources['weather'].frame, 'aemo': sources['aemo'].frame},
                sources['history'].frame, start)
            outcome = fc.run_predictions(['price'], False, True, False, calculation_only=True, input_snapshot=inputs)
        memory = reclaim_transient_memory()
        input_revisions['model_artifacts'] = content_revision(self.cache.loaded_signature)
        completion = PriceCompletion(outcome, revision, captured_at, time.monotonic()-started,
                                     self.cache.last_load_seconds, self.cache.loads, memory.after_mib,
                                     {name: source.revision for name, source in sources.items()},
                                     {name: (now-source.fetched_at).total_seconds() for name, source in sources.items()},
                                     memory.elapsed_seconds, memory.before_mib, memory.trimmed, input_revisions, freshness)
        logging.info(json.dumps({'event': 'resident_price_complete', 'run_id': outcome.run_id,
            'parent_revision': revision, 'captured_at': captured_at,
            'source_revisions': completion.source_revisions, 'source_ages_seconds': completion.source_ages_seconds,
            'input_revisions': completion.input_revisions,
            'source_freshness': completion.source_freshness,
            'memory_maintenance_seconds': round(memory.elapsed_seconds, 3),
            'rss_before_maintenance_mib': round(memory.before_mib, 1), 'allocator_trimmed': memory.trimmed,
            'model_bundle_id': outcome.model_bundle_id, 'point_counts': outcome.point_counts,
            'publication_result': outcome.publication_result, 'verify_reload': self.verify_reload,
            'elapsed_seconds': round(completion.elapsed_seconds, 3),
            'model_load_seconds': round(completion.model_load_seconds, 3),
            'model_loads': completion.model_loads, 'rss_mib': round(completion.rss_mib, 1)}, sort_keys=True))
        return completion

    def evaluate_completion(self, completion, *, now=None):
        """Point-in-time shadow admission; no output writes or model promotion."""
        import forecast as fc
        from tariff_utils import load_tariff_profile
        live_clock = now is None
        now = utc_now() if live_clock else pd.Timestamp(now)
        if now.tzinfo is None:
            raise ValueError('acceptance time must be timezone-aware')
        capture = pd.Timestamp(completion.captured_at)
        age = (now-capture).total_seconds()
        if age < 0 or age > 180:
            return AcceptanceDecision(('completion_age_outside_budget',), True)
        first = pd.Timestamp(completion.outcome.forecasts['price'].index[0])
        if first != now.floor('30min'):
            return AcceptanceDecision(('forecast_interval_changed',), True)
        try:
            validate_forecast_family(completion.outcome.forecasts, 'price', expected_start=now.floor('30min'))
        except ForecastContractError as exc:
            return AcceptanceDecision((f'forecast_invalid:{exc}',))
        try:
            sources = self.sources.snapshot(now)
        except SourceUnavailable as exc:
            # Wait for source recovery; immediately rerunning cannot repair it.
            return AcceptanceDecision((f'inputs_unavailable:{exc}',))
        rows = fc.call_ha_api('GET', 'states')
        if not isinstance(rows, list):
            raise RuntimeError('cannot verify current HA inputs')
        # Recheck interval and cache admission after the network read; it may
        # have crossed a boundary or taken long enough to expire dependencies.
        finish = utc_now() if live_clock else now
        if first != finish.floor('30min'):
            return AcceptanceDecision(('forecast_interval_changed',), True)
        if (finish-capture).total_seconds() > 180:
            return AcceptanceDecision(('completion_age_outside_budget',), True)
        try:
            sources = self.sources.snapshot(finish)
        except SourceUnavailable as exc:
            return AcceptanceDecision((f'inputs_unavailable:{exc}',))
        current = {
            'config': content_revision(load_config()),
            'tariff': content_revision(load_tariff_profile(fc.CONFIG, fc.ROOT)),
            'tariff_file': file_revision(fc.ROOT / fc.CONFIG['paths']['tariff_file']),
            'model_pointer': file_revision(fc._production_bundle_store().active_pointer('price')),
            'model_artifacts': content_revision(self.cache.current_signature()),
            **{f'source:{name}': source.revision for name, source in sources.items()},
        }
        needed = {key.removeprefix('ha:') for key in completion.input_revisions if key.startswith('ha:')}
        apf_entity = fc.CONFIG['home_assistant']['amber_billing_entity']
        needed.add(apf_entity)
        states = {row['entity_id']: row for row in rows if row['entity_id'] in needed}
        current['apf'] = content_revision(states.get(apf_entity))
        for entity in needed-{apf_entity}:
            # The initial optional poll marker uses {} when missing.
            fallback = {} if entity == fc.CONFIG['home_assistant'].get('solcast_last_polled_entity') else None
            current[f'ha:{entity}'] = content_revision(states.get(entity, fallback))
        return compare_revisions(completion.input_revisions, current)
