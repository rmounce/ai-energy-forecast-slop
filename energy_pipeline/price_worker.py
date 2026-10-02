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
from production_contract import PredictionInputs, PredictionOutcome


def content_revision(value):
    """Opaque digest: provenance logs never contain configuration credentials."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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
