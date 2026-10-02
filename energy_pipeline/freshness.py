"""Bounded, per-acquisition freshness evidence; receipt is not provider issuance."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class FreshnessEvidence:
    source: str
    basis: str
    timestamp: str | None
    cached: bool | None = None
    expired: bool | None = None


_COLLECTOR = ContextVar('source_freshness_evidence', default=None)


@contextmanager
def collect_freshness():
    evidence = []
    token = _COLLECTOR.set(evidence)
    try:
        yield evidence
    finally:
        _COLLECTOR.reset(token)


def record_evidence(evidence):
    collector = _COLLECTOR.get()
    if collector is not None:
        if len(collector) >= 16:
            raise RuntimeError('source freshness evidence exceeded bounded collection')
        collector.append(evidence)


def record_http_response(source, response):
    if _COLLECTOR.get() is None:
        return
    cached = bool(getattr(response, 'from_cache', False))
    created = getattr(response, 'created_at', None)
    # A cached response without its original creation time remains unknown;
    # never make its freshness equal to the time we read it from disk.
    if created is None and not cached:
        created = datetime.now(timezone.utc)
    if created is not None and created.tzinfo is None:
        # requests-cache timestamps are UTC; normalize its legacy naive form.
        created = created.replace(tzinfo=timezone.utc)
    record_evidence(FreshnessEvidence(source, 'http_response_created',
        created.astimezone(timezone.utc).isoformat() if created else None,
        cached, bool(getattr(response, 'is_expired', False))))


def serialize_evidence(evidence):
    return [asdict(item) for item in evidence]
