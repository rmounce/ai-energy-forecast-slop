from types import SimpleNamespace
import json
import os

import pandas as pd
import pytest

from energy_pipeline.accepted_store import AcceptedStore, StoreError, MAX_BYTES
from energy_pipeline.tariffs import TariffSnapshot
from production_contract import PredictionOutcome


NOW = pd.Timestamp('2026-10-02T06:35:00Z')


def completion(run_id='first'):
    index = pd.date_range(NOW.floor('30min'), periods=144, freq='30min')
    return SimpleNamespace(outcome=PredictionOutcome('price', 'test', 'model',
        {key: pd.DataFrame({key: [value]*144}, index=index)
         for key, value in [('price_p30', -1.), ('price', 0.), ('price_p70', 1.)]}, run_id=run_id),
        parent_revision='parent', captured_at=NOW.isoformat(), input_revisions={'tariff_file': 'missing'},
        source_revisions={'weather': 'weather-revision'}, source_freshness={'weather': []},
        tariff_snapshot=TariffSnapshot('missing', (), (), 1.05, 1.1))


def test_roundtrip_recovery_is_historical_with_expiry_and_exact_values(tmp_path):
    path = tmp_path/'checkpoint.json'
    store = AcceptedStore(path)
    assert store.recover() is None
    store.save(completion(), NOW+pd.Timedelta(seconds=1))
    store.close()
    store = AcceptedStore(path)
    try:
        bundle = store.recover()
        assert bundle.payload['run_id'] == 'first'
        assert bundle.payload['forecasts']['price_p30']['values'] == [-1.]*144
        assert bundle.payload['tariff_snapshot']['network_loss_factor'] == 1.05
        assert bundle.time_current(NOW+pd.Timedelta(seconds=2))
        assert not bundle.time_current(NOW-pd.Timedelta(seconds=1))
        assert not bundle.time_current(NOW+pd.Timedelta(seconds=180))
        assert not bundle.time_current(NOW+pd.Timedelta(minutes=30))
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        store.close()


def test_single_owner_and_release(tmp_path):
    path = tmp_path/'checkpoint.json'
    store = AcceptedStore(path)
    with pytest.raises(StoreError, match='already owned'):
        AcceptedStore(path)
    store.close()
    other = AcceptedStore(path)
    other.close()
    with pytest.raises(StoreError, match='closed'):
        store.save(completion(), NOW)


@pytest.mark.parametrize('failure', ['replace', 'fsync'])
def test_failed_commit_preserves_previous_complete_record_and_cleans_temp(tmp_path, monkeypatch, failure):
    path = tmp_path/'checkpoint.json'
    store = AcceptedStore(path)
    try:
        store.save(completion(), NOW)
        def fail(*args): raise OSError('injected failure')
        monkeypatch.setattr(os, failure, fail)
        with pytest.raises(StoreError, match='commit failed'):
            store.save(completion('second'), NOW)
        assert store.recover().payload['run_id'] == 'first'
        assert not list(tmp_path.glob('*.tmp'))
    finally:
        store.close()


def test_directory_sync_failure_leaves_only_complete_record(tmp_path, monkeypatch):
    path = tmp_path/'checkpoint.json'
    store = AcceptedStore(path)
    try:
        store.save(completion(), NOW)
        fsync = os.fsync
        calls = []
        def fail_directory(fd):
            calls.append(fd)
            if len(calls) == 2: raise OSError('directory sync failure')
            fsync(fd)
        monkeypatch.setattr(os, 'fsync', fail_directory)
        with pytest.raises(StoreError): store.save(completion('second'), NOW)
        assert store.recover().payload['run_id'] == 'second'
    finally:
        store.close()


@pytest.mark.parametrize('raw', [b'{', b'x'*(MAX_BYTES+1)])
def test_corrupt_or_oversized_checkpoint_fails_closed(tmp_path, raw):
    path = tmp_path/'checkpoint.json'
    path.write_bytes(raw)
    store = AcceptedStore(path)
    try:
        with pytest.raises(StoreError, match='retained for inspection'): store.recover()
        assert path.read_bytes() == raw
    finally:
        store.close()


def test_checksum_and_contract_validation(tmp_path):
    from energy_pipeline.accepted_store import _encode
    import hashlib
    path = tmp_path/'checkpoint.json'
    store = AcceptedStore(path)
    try:
        store.save(completion(), NOW)
        envelope = json.loads(path.read_bytes())
        envelope['payload']['forecasts']['price']['values'][0] = 99
        path.write_bytes(_encode(envelope))
        with pytest.raises(StoreError): store.recover()
        envelope['sha256'] = hashlib.sha256(_encode(envelope['payload'])).hexdigest()
        path.write_bytes(_encode(envelope))
        with pytest.raises(StoreError): store.recover()  # quantiles cross even with valid digest
    finally:
        store.close()


def test_invalid_family_cannot_replace_existing_record(tmp_path):
    path = tmp_path/'checkpoint.json'
    store = AcceptedStore(path)
    try:
        store.save(completion(), NOW)
        invalid = completion('bad')
        invalid.outcome.forecasts['price'].iloc[0, 0] = float('nan')
        with pytest.raises(StoreError): store.save(invalid, NOW)
        assert store.recover().payload['run_id'] == 'first'
    finally:
        store.close()


def test_checkpoint_survives_abrupt_process_exit_and_releases_owner(tmp_path):
    import subprocess
    import sys
    path = tmp_path/'checkpoint.json'
    code = """import os, runpy, sys
ns = runpy.run_path('tests/energy_pipeline/test_accepted_store.py')
store = ns['AcceptedStore'](sys.argv[1])
store.save(ns['completion'](), ns['NOW'])
os._exit(0)
"""
    process = subprocess.run([sys.executable, '-c', code, str(path)], timeout=10)
    assert process.returncode == 0
    store = AcceptedStore(path)
    try:
        assert store.recover().payload['run_id'] == 'first'
    finally:
        store.close()
