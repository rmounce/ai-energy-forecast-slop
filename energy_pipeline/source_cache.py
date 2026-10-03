"""Thread-safe, bounded last-usable dataframe snapshots for resident inference."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import threading
from typing import Mapping
import numpy as np
import pandas as pd
from energy_pipeline.freshness import FreshnessEvidence


class SourceUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class SourcePolicy:
    columns: tuple[str, ...]
    refresh_seconds: float
    max_age_seconds: float
    historical: bool = False
    min_history_hours: float = 0
    zero_capacity_ratios: tuple[tuple[str, str, str], ...] = ()
    max_tail_gap_seconds: float = 0


@dataclass(frozen=True)
class SourceSnapshot:
    frame: pd.DataFrame
    fetched_at: datetime
    revision: str
    evidence: tuple[FreshnessEvidence, ...] = ()


def validate_frame(frame, policy, start):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise SourceUnavailable('empty dataframe')
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise SourceUnavailable('index must be timezone-aware')
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise SourceUnavailable('index must be sorted and unique')
    missing = set(policy.columns)-set(frame.columns)
    if missing:
        raise SourceUnavailable(f'missing columns: {sorted(missing)}')
    selected = frame.loc[:, list(policy.columns)].apply(pd.to_numeric, errors='coerce')
    # STPASA availability fractions are undefined when both capacity and
    # available generation are zero (observed nightly solar 0/0). Preserve the
    # raw NaNs for incumbent ffill/bfill; permit only this documented case during
    # coverage validation, with at least one finite ratio available to fill from.
    for ratio, numerator, denominator in policy.zero_capacity_ratios:
        if ratio in selected and numerator in frame and denominator in frame:
            allowed = selected[ratio].isna() & frame[numerator].eq(0) & frame[denominator].eq(0)
            if selected[ratio].notna().any():
                selected.loc[allowed, ratio] = 0
    if np.isinf(selected.to_numpy()).any():
        raise SourceUnavailable('infinite values')
    if policy.historical:
        usable = selected.dropna()
        # Existing predictor fills historical gaps; bound latest usable observation
        # independently rather than pretending fetch time means observation freshness.
        if usable.empty or usable.index[-1] < start-timedelta(minutes=90):
            raise SourceUnavailable('latest complete history row is more than 90 minutes behind')
        if usable.index[0] > start-timedelta(hours=policy.min_history_hours):
            raise SourceUnavailable('insufficient historical lookback')
    else:
        required = pd.date_range(start, periods=144, freq='30min')
        window = selected.reindex(required)
        if policy.max_tail_gap_seconds:
            # BOM's hourly horizon can end shortly before the price horizon.
            # Preserve raw inputs: incumbent preparation fills AFTER adjustments.
            # Only waive absent trailing targets, never present-but-invalid rows,
            # internal holes or a missing leading target.
            tail_gap = (required[-1]-selected.index[-1]).total_seconds()
            if 0 < tail_gap <= policy.max_tail_gap_seconds:
                window = window.loc[window.index <= selected.index[-1]]
        if not np.isfinite(window.to_numpy()).all():
            raise SourceUnavailable('missing or nonfinite 72-hour forecast coverage')


class SourceCache:
    def __init__(self, policies: Mapping[str, SourcePolicy]):
        self.policies = dict(policies)
        self._sources = {}
        self._lock = threading.Lock()
        self._generation = 0

    @property
    def generation(self):
        with self._lock:
            return self._generation

    def put(self, name, frame, fetched_at, *, evidence=()):
        fetched_at = pd.Timestamp(fetched_at).to_pydatetime()
        if fetched_at.tzinfo is None:
            raise SourceUnavailable('fetch timestamp must be timezone-aware')
        try:
            validate_frame(frame, self.policies[name], pd.Timestamp(fetched_at).floor('30min'))
        except SourceUnavailable as exc:
            raise SourceUnavailable(f'{name}: {exc}') from exc
        frozen = frame.copy(deep=True)
        frozen.index = frozen.index.tz_convert('UTC')
        digest = hashlib.sha256(pd.util.hash_pandas_object(frozen, index=True).values.tobytes()
                                + repr(list(frozen.columns)).encode()).hexdigest()
        replacement = SourceSnapshot(frozen, fetched_at, digest, tuple(evidence))
        with self._lock:
            prior = self._sources.get(name)
            if prior and prior.fetched_at > fetched_at:
                raise SourceUnavailable('out-of-order source completion')
            recovered = prior is not None and (fetched_at-prior.fetched_at).total_seconds() > self.policies[name].max_age_seconds
            self._sources[name] = replacement
            changed = prior is None or prior.revision != digest or recovered
            if changed:
                self._generation += 1
        return changed

    def snapshot(self, now):
        now = pd.Timestamp(now)
        if now.tzinfo is None:
            raise SourceUnavailable('snapshot time must be timezone-aware')
        start = now.floor('30min')
        with self._lock:
            sources = dict(self._sources)
        result = {}
        for name, policy in self.policies.items():
            source = sources.get(name)
            if source is None:
                raise SourceUnavailable(f'{name}: not acquired')
            age = (now-pd.Timestamp(source.fetched_at)).total_seconds()
            if age < 0 or age > policy.max_age_seconds:
                raise SourceUnavailable(f'{name}: acquisition age {age:.1f}s outside budget')
            try:
                validate_frame(source.frame, policy, start)
            except SourceUnavailable as exc:
                raise SourceUnavailable(f'{name}: {exc}') from exc
            result[name] = SourceSnapshot(source.frame.copy(deep=True), source.fetched_at, source.revision, source.evidence)
        return result


def price_source_policies(config):
    features = config['models']['price']['feature_cols']
    weather = tuple(col for col in features if col in
                    ('temperature_adelaide', 'humidity_adelaide', 'wind_speed_adelaide'))
    aemo = tuple(col for col in features if col not in (*weather, 'power_pv'))
    target = config['models']['price']['target_column']
    lags = config['models']['price'].get('target_lags', [])
    lookback = abs(min(lags)) / 2 if isinstance(lags, list) and lags else abs(lags) / 2 if isinstance(lags, int) else 0
    ratios = tuple((f'stpasa_{kind}_avail_frac', f'stpasa_ss_{kind}_uigf', f'stpasa_ss_{kind}_capacity')
                   for kind in ('wind', 'solar') if f'stpasa_{kind}_avail_frac' in features)
    return {'weather': SourcePolicy(weather, 1800, 7200, max_tail_gap_seconds=3600),
            'aemo': SourcePolicy(aemo, 300, 1800, zero_capacity_ratios=ratios),
            'history': SourcePolicy(tuple(dict.fromkeys([target, *features])), 300, 2400, historical=True, min_history_hours=lookback, zero_capacity_ratios=ratios)}
