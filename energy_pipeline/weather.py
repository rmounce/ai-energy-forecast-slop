"""Associate observed hourly BOM clocks with resident weather captures."""
from datetime import datetime, timezone

from energy_pipeline.freshness import FreshnessEvidence, record_evidence


MARKERS = ('hourly_forecast_issue_time', 'hourly_forecast_last_fetched')


def _markers(fc):
    entity = fc.CONFIG['home_assistant']['weather_entity']
    state = fc.call_ha_api('GET', f'states/{entity}')
    if not isinstance(state, dict) or state.get('entity_id') != entity:
        raise RuntimeError('BOM hourly state unavailable during capture')
    attributes = state.get('attributes')
    if not isinstance(attributes, dict):
        raise RuntimeError('BOM hourly state attributes unavailable during capture')
    return tuple(attributes.get(key) for key in MARKERS)


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def capture_weather(fc):
    # HA state and forecast are separate reads. This detects observed updates;
    # it cannot prove atomicity inside the integration's collector/entity lifecycle.
    for _ in range(2):
        before = _markers(fc)
        frame = fc.get_weather_forecast()
        after = _markers(fc)
        if before != after:
            continue
        issue, fetched = map(_timestamp, after)
        record_evidence(FreshnessEvidence('bom',
            'hourly_issue_observed' if issue else 'provider_issue_unknown', issue))
        record_evidence(FreshnessEvidence('bom',
            'hourly_successful_fetch_observed' if fetched else 'provider_fetch_unknown', fetched))
        return frame
    raise RuntimeError('BOM hourly markers changed during both capture attempts')
