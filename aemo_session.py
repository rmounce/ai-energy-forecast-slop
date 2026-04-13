"""
Shared requests-cache session for AEMO/NEMWeb HTTP calls.

Used by both forecast.py and ingest scripts so that a successful fetch by
any process is cached on disk and reused by the next caller within the TTL.

Cache file: data/http_cache.sqlite  (project root, persists across reboots)
"""
from pathlib import Path

from requests_cache import CachedSession

# Stable path relative to this file (project root)
CACHE_PATH = Path(__file__).parent / "data" / "http_cache"


def make_aemo_session() -> CachedSession:
    """Return a CachedSession configured for AEMO/NEMWeb endpoints.

    TTLs:
      - AEMO visualisations API (5MIN JSON):  300 s  (data refreshes ~5 min)
      - NEMWeb reports (ZIPs, listings):     1800 s  (reports update every 30 min)

    stale_if_error=True: on network failure, returns the most recent cached
    response rather than raising, so a transient AEMO outage doesn't abort
    the forecast run.
    """
    return CachedSession(
        str(CACHE_PATH),
        backend="sqlite",
        stale_if_error=True,
        urls_expire_after={
            "*visualisations.aemo.com.au*": 300,
            "*nemweb.com.au*": 1800,
        },
    )
