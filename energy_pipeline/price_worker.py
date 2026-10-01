"""Calculation-only resident price adapter; no production output writes."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time

from config_utils import load_config
from energy_pipeline.model_cache import ModelCache


@dataclass(frozen=True)
class PriceCompletion:
    outcome: object
    parent_revision: str
    captured_at: str
    elapsed_seconds: float
    model_load_seconds: float
    model_loads: int
    rss_mib: float


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])/1024
    return 0.0


class PriceWorker:
    """Called by one dedicated executor thread, never concurrently.

    HA state is frozen once per run. Weather service and AEMO/history remain
    independent acquisitions, as in production; caching those is a later step.
    """
    def __init__(self, *, verify_reload=False):
        self.verify_reload = verify_reload
        self.cache = ModelCache()

    def predict(self):
        started = time.monotonic()
        import forecast as fc
        from tariff_utils import load_tariff_profile
        # Long-lived process observes changed config/tariff profile next run.
        fc.CONFIG = load_config()
        fc.GENERAL_TARIFF_MAP, fc.FEED_IN_TARIFF_MAP, fc.NETWORK_LOSS_FACTOR = load_tariff_profile(fc.CONFIG, fc.ROOT)
        rows = fc.call_ha_api('GET', 'states')
        if not isinstance(rows, list):
            raise RuntimeError('cannot capture HA inputs')
        needed = set(fc.CONFIG['home_assistant']['solcast_entities'])
        needed.add(fc.CONFIG['home_assistant']['amber_billing_entity'])
        states = {row['entity_id']: row for row in rows if row['entity_id'] in needed}
        del rows
        apf = states.get(fc.CONFIG['home_assistant']['amber_billing_entity'])
        if not apf or not apf.get('attributes', {}).get('Forecasts'):
            raise RuntimeError('snapshot missing Amber APF')
        captured_at = datetime.now(timezone.utc).isoformat()
        revision = hashlib.sha256(json.dumps(apf, sort_keys=True).encode()).hexdigest()
        with fc.prediction_resources(entity_states=states, model_cache=self.cache, verify_reload=self.verify_reload):
            outcome = fc.run_predictions(['price'], False, True, False, calculation_only=True)
        completion = PriceCompletion(outcome, revision, captured_at, time.monotonic()-started,
                                     self.cache.last_load_seconds, self.cache.loads, rss_mib())
        logging.info(json.dumps({'event': 'resident_price_complete', 'run_id': outcome.run_id,
            'parent_revision': revision, 'captured_at': captured_at,
            'model_bundle_id': outcome.model_bundle_id, 'point_counts': outcome.point_counts,
            'publication_result': outcome.publication_result, 'verify_reload': self.verify_reload,
            'elapsed_seconds': round(completion.elapsed_seconds, 3),
            'model_load_seconds': round(completion.model_load_seconds, 3),
            'model_loads': completion.model_loads, 'rss_mib': round(completion.rss_mib, 1)}, sort_keys=True))
        return completion
