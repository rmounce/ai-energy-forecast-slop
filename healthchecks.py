#!/usr/bin/env python3
"""Shared Healthchecks.io project-key support for scheduled repo jobs."""

from __future__ import annotations

import argparse
import os
import re
import sys
from urllib.error import URLError
from urllib.request import urlopen


PING_ROOT = "https://hc-ping.com"
SLUG_RE = re.compile(r"[a-z0-9_-]+\Z")
PROJECT_KEY_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


class HealthcheckPingError(RuntimeError):
    """A sanitized error for a failed ping; never includes the secret URL."""


def build_ping_url(slug: str, *, failed: bool = False, project_key: str | None = None,
                   legacy_url: str | None = None) -> str | None:
    """Build a project-key slug URL, falling back to a legacy UUID URL if given."""
    if not SLUG_RE.fullmatch(slug):
        raise ValueError("Healthchecks slug must use lowercase letters, digits, hyphens, or underscores")
    key = project_key if project_key is not None else os.environ.get("HC_REPO_PING_KEY", "")
    key = key.strip()
    if key:
        if not PROJECT_KEY_RE.fullmatch(key):
            raise ValueError("HC_REPO_PING_KEY contains invalid characters")
        endpoint = f"{PING_ROOT}/{key}/{slug}"
        if failed:
            endpoint += "/fail"
        return f"{endpoint}?create=1"
    if legacy_url:
        endpoint = legacy_url.rstrip("/")
        if failed:
            endpoint += "/fail"
        return endpoint
    return None


def ping_check(slug: str, *, failed: bool = False, legacy_url: str | None = None,
               timeout: float = 10) -> bool:
    """Ping a check. Return False when no check key is configured."""
    url = build_ping_url(slug, failed=failed, legacy_url=legacy_url)
    if url is None:
        return False
    try:
        with urlopen(url, timeout=timeout):
            return True
    except (OSError, URLError) as exc:
        raise HealthcheckPingError(
            f"Healthchecks request failed ({type(exc).__name__})"
        ) from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slug", help="stable unique slug for the job")
    parser.add_argument("--exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        ping_check(args.slug, failed=args.exit_code != 0,
                   legacy_url=os.environ.get("HC_PREDICT_URL"))
    except (HealthcheckPingError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
