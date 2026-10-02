"""Single-owner atomic shadow checkpoint; recovered data never authorizes publication."""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

from production_contract import validate_forecast_family

MAX_BYTES = 256 * 1024


class StoreError(RuntimeError):
    pass


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _clock(value):
    clock = pd.Timestamp(value)
    if pd.isna(clock) or clock.tzinfo is None:
        raise ValueError('checkpoint clock must be aware')
    return clock.tz_convert('UTC')


@dataclass(frozen=True)
class RecoveredBundle:
    payload: dict

    def time_current(self, now=None):
        """Necessary time gate only; current inputs must still be revalidated."""
        now = _clock(now if now is not None else datetime.now(timezone.utc))
        capture = _clock(self.payload['captured_at'])
        return (0 <= (now-capture).total_seconds() <= 180
                and now < _clock(self.payload['expires_at'])
                and now.floor('30min') == _clock(self.payload['forecast_start']))


def _validate(payload):
    if payload['schema'] != 1 or payload['mode'] != 'shadow':
        raise ValueError('unsupported checkpoint schema/mode')
    capture, accepted, expires, start = map(_clock, (payload['captured_at'],
        payload['accepted_at'], payload['expires_at'], payload['forecast_start']))
    if not 0 <= (accepted-capture).total_seconds() <= 180:
        raise ValueError('checkpoint acceptance outside capture budget')
    if start != accepted.floor('30min') or expires != min(capture+pd.Timedelta(seconds=180), start+pd.Timedelta(minutes=30)):
        raise ValueError('invalid checkpoint interval/expiry')
    if not payload['run_id'] or not payload['parent_revision'] or not payload['model_bundle_id']:
        raise ValueError('missing checkpoint identity')
    if not isinstance(payload['input_revisions'], dict) or not payload['input_revisions']:
        raise ValueError('missing checkpoint input revisions')
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in payload['input_revisions'].items()):
        raise ValueError('invalid checkpoint revisions')
    if payload['tariff_snapshot']['revision'] != payload['input_revisions']['tariff_file']:
        raise ValueError('checkpoint tariff identity mismatch')
    frames = {key: pd.DataFrame({key: item['values']},
              index=pd.DatetimeIndex([_clock(value) for value in item['timestamps']]))
              for key, item in payload['forecasts'].items()}
    if set(frames) != {'price_p30', 'price', 'price_p70'}:
        raise ValueError('invalid checkpoint forecast family')
    validate_forecast_family(frames, 'price', expected_start=start)
    if any(not frame.index.equals(frames['price'].index) for frame in frames.values()):
        raise ValueError('checkpoint quantile timestamps differ')
    # Also reject nonfinite metadata, not only nonfinite forecast values.
    _encode(payload)


class AcceptedStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = os.open(str(self.path)+'.lock', os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lock)
            self._lock = None
            raise StoreError('shadow checkpoint already owned') from exc

    def close(self):
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def recover(self):
        if self._lock is None:
            raise StoreError('checkpoint store closed')
        try:
            with self.path.open('rb') as handle:
                raw = handle.read(MAX_BYTES+1)
            if len(raw) > MAX_BYTES:
                raise ValueError('checkpoint exceeds size limit')
            envelope = json.loads(raw)
            payload = envelope['payload']
            if envelope['sha256'] != hashlib.sha256(_encode(payload)).hexdigest():
                raise ValueError('checkpoint checksum mismatch')
            _validate(payload)
            return RecoveredBundle(payload)
        except FileNotFoundError:
            return None
        except Exception as exc:
            raise StoreError('invalid shadow checkpoint; retained for inspection') from exc

    def save(self, completion, accepted_at):
        if self._lock is None:
            raise StoreError('checkpoint store closed')
        temporary = None
        try:
            outcome = completion.outcome
            start = _clock(outcome.forecasts['price'].index[0])
            captured = _clock(completion.captured_at)
            payload = dict(schema=1, mode='shadow', run_id=outcome.run_id,
                model_bundle_id=outcome.model_bundle_id, parent_revision=completion.parent_revision,
                captured_at=captured.isoformat(), accepted_at=_clock(accepted_at).isoformat(),
                forecast_start=start.isoformat(),
                expires_at=min(captured+pd.Timedelta(seconds=180), start+pd.Timedelta(minutes=30)).isoformat(),
                input_revisions=completion.input_revisions, source_revisions=completion.source_revisions,
                source_freshness=completion.source_freshness, tariff_snapshot=asdict(completion.tariff_snapshot),
                forecasts={key: {'timestamps': [_clock(value).isoformat() for value in frame.index],
                                 'values': frame.iloc[:, 0].astype(float).tolist()}
                           for key, frame in outcome.forecasts.items()})
            _validate(payload)
            raw = _encode({'payload': payload, 'sha256': hashlib.sha256(_encode(payload)).hexdigest()})
            if len(raw) > MAX_BYTES:
                raise ValueError('checkpoint exceeds size limit')
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=self.path.name+'.',
                                             suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            temporary = None
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return RecoveredBundle(payload)
        except Exception as exc:
            # Failure after replace has an uncertain durability outcome. Stop;
            # restart validates whichever complete historical record survived.
            raise StoreError('shadow checkpoint commit failed') from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
