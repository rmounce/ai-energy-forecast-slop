"""Immutable, worker-local tariff profile captured from one verified file read."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class TariffSnapshot:
    revision: str
    general_tariff: tuple[tuple[str, float], ...]
    feed_in_tariff: tuple[tuple[str, float], ...]
    network_loss_factor: float
    amber_api_scaling_factor: float

    @property
    def present(self):
        return self.revision != 'missing'

    @property
    def effective_profile(self):
        # Return owned copies: caller mutation cannot alter the captured profile.
        return dict(self.general_tariff), dict(self.feed_in_tariff), self.network_loss_factor


def capture_tariffs(path):
    try:
        with Path(path).open('rb') as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read()
            after = os.fstat(handle.fileno())
        stamp = lambda stat: (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stamp(before) != stamp(after):
            raise RuntimeError('tariff file changed during capture')
    except FileNotFoundError:
        return TariffSnapshot('missing', (), (), 1.05, 1.10)
    profile = json.loads(raw)
    def numeric(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('tariff profile contains nonfinite value')
        return number
    def schedule(name):
        return tuple(sorted((slot, numeric(value)) for slot, value in profile.get(name, {}).items()))
    return TariffSnapshot(hashlib.sha256(raw).hexdigest(), schedule('general_tariff'),
        schedule('feed_in_tariff'), numeric(profile.get('network_loss_factor', 1.05)),
        numeric(profile.get('amber_api_scaling_factor', 1.10)))


_CURRENT = ContextVar('frozen_tariff_profile', default=None)


def current_tariffs():
    return _CURRENT.get()


@contextmanager
def frozen_tariffs(snapshot):
    token = _CURRENT.set(snapshot)
    try:
        yield
    finally:
        _CURRENT.reset(token)
