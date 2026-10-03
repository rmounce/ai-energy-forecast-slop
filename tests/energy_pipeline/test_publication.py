import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from energy_pipeline.accepted_store import AcceptedStore, StoreError
from energy_pipeline.publication import ShadowPublication, prepare_plan, encoded
from energy_pipeline.tariffs import TariffSnapshot, frozen_tariffs
from test_accepted_store import completion, NOW

CONFIG = {'home_assistant': {'publish_entities':
    {key: 'sensor.test_'+key for key in ('price_p30', 'price', 'price_p70')}}}


@pytest.fixture
def plan(tmp_path):
    store = AcceptedStore(tmp_path/'accepted.json')
    try:
        bundle = store.save(completion(), NOW)
        return prepare_plan(bundle, CONFIG)
    finally:
        store.close()


def rows(path, table):
    with sqlite3.connect(path) as db:
        return db.execute('SELECT * FROM '+table).fetchall()


@pytest.mark.parametrize('with_tariffs', [False, True])
def test_incumbent_payload_parity_including_tariffs_and_stable_clock(tmp_path, monkeypatch, with_tariffs):
    import forecast as fc
    store = AcceptedStore(tmp_path/'accepted.json')
    completed = completion()
    if with_tariffs:
        slots = [slot.strftime('%H:%M:%S') for slot in pd.date_range('2000-01-01', periods=48, freq='30min')]
        completed.tariff_snapshot = TariffSnapshot('test-profile', tuple((slot, .2) for slot in slots),
            tuple((slot, .05) for slot in slots), 1.07, 1.1)
        completed.input_revisions['tariff_file'] = 'test-profile'
    try:
        bundle = store.save(completed, NOW)
        plan = prepare_plan(bundle, CONFIG)
        actual = []
        monkeypatch.setattr(fc, 'CONFIG', {**fc.CONFIG, **CONFIG})
        monkeypatch.setattr(fc, 'call_ha_api', lambda method, endpoint, payload:
                            actual.append({'entity': endpoint.removeprefix('states/'), 'payload': payload}) or {})
        class Clock:
            @staticmethod
            def now(tz): return NOW.to_pydatetime()
        monkeypatch.setattr(fc, 'datetime', Clock)
        with frozen_tariffs(completed.tariff_snapshot):
            fc._publish_lgbm_model_to_hass('price', {'forecasts': completed.outcome.forecasts})
        assert plan['writes'] == actual
    finally:
        store.close()


def test_success_and_duplicate_have_one_stable_commit_marker(tmp_path, plan, monkeypatch):
    path = tmp_path/'publication.db'
    publisher = ShadowPublication(path)
    try:
        assert publisher.execute(plan, lambda: True, now=lambda: NOW)
        assert rows(path, 'marker') == [(1, plan['id'])]
        assert len(rows(path, 'outputs')) == 3
        monkeypatch.setattr(publisher, '_write', lambda *args: pytest.fail('duplicate sink write'))
        assert publisher.execute(plan, lambda: pytest.fail('no lease renewal'), now=lambda: NOW)
        assert not publisher.recover()
    finally:
        publisher.close()


@pytest.mark.parametrize('acknowledged', [False, True])
def test_crash_window_resumes_only_missing_receipts_with_identical_payloads(tmp_path, plan, monkeypatch, acknowledged):
    path = tmp_path/'publication.db'
    publisher = ShadowPublication(path)
    original = publisher._ack
    def fail_ack(job, receipts):
        if acknowledged: original(job, receipts)
        raise OSError('injected crash after sink write')
    monkeypatch.setattr(publisher, '_ack', fail_ack)
    with pytest.raises(StoreError): publisher.execute(plan, lambda: True, now=lambda: NOW)
    assert len(rows(path, 'outputs')) == 1 and not rows(path, 'marker')
    publisher.close()
    publisher = ShadowPublication(path)
    try:
        pending = publisher.recover()
        assert pending[0][0] == plan
        assert len(pending[0][1]) == int(acknowledged)
        writes = []
        original_write = publisher._write
        def track(job, write):
            writes.append(write)
            original_write(job, write)
        monkeypatch.setattr(publisher, '_write', track)
        assert publisher.execute(plan, lambda: True, now=lambda: NOW)
        assert len(writes) == 3-int(acknowledged)
        assert rows(path, 'marker') == [(1, plan['id'])]
        assert json.loads(rows(path, 'outputs')[0][2]) == plan['writes'][0]['payload']
    finally:
        publisher.close()


@pytest.mark.parametrize('failure_at', [1, 2, 3, 4])
def test_parent_change_between_writes_never_commits_family(tmp_path, plan, failure_at):
    path = tmp_path/'publication.db'
    publisher = ShadowPublication(path)
    calls = []
    def valid():
        calls.append(1)
        return len(calls) <= failure_at
    try:
        assert not publisher.execute(plan, valid, now=lambda: NOW)
        assert not rows(path, 'marker')
        assert len(rows(path, 'outputs')) == failure_at-1
    finally:
        publisher.close()


def test_expired_recovery_and_startup_abandon_do_not_replay(tmp_path, plan, monkeypatch):
    path = tmp_path/'publication.db'
    publisher = ShadowPublication(path)
    monkeypatch.setattr(publisher, '_ack', lambda *args: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(StoreError): publisher.execute(plan, lambda: True, now=lambda: NOW)
    try:
        assert not publisher.execute(plan, lambda: True, now=lambda: NOW+pd.Timedelta(minutes=4))
        assert not rows(path, 'marker')
        assert publisher.abandon_pending() == 0  # expired execute already abandoned it
        assert not publisher.execute(plan, lambda: True, now=lambda: NOW)
    finally:
        publisher.close()


def test_plan_tamper_and_competing_owner_fail(tmp_path, plan):
    publisher = ShadowPublication(tmp_path/'publication.db')
    try:
        with pytest.raises(StoreError): ShadowPublication(tmp_path/'publication.db')
        plan['writes'][0]['payload']['state'] = 999
        with pytest.raises(StoreError): publisher.execute(plan, lambda: True, now=lambda: NOW)
    finally:
        publisher.close()


def next_plan(plan, run_id):
    import hashlib
    new = json.loads(encoded(plan))
    new['bundle']['run_id'] = run_id
    del new['id']
    new['id'] = hashlib.sha256(encoded(new).encode()).hexdigest()
    return new


def test_mixed_sink_generation_cannot_pass_old_marker(tmp_path, plan, monkeypatch):
    publisher = ShadowPublication(tmp_path/'publication.db')
    try:
        assert publisher.execute(plan, lambda: True, now=lambda: NOW)
        assert publisher.committed_family() == plan
        second = next_plan(plan, 'second')
        monkeypatch.setattr(publisher, '_ack', lambda *args: (_ for _ in ()).throw(OSError('crash')))
        with pytest.raises(StoreError): publisher.execute(second, lambda: True, now=lambda: NOW)
        assert publisher.committed_family() is None
        assert publisher.abandon_pending() == 1
        assert not publisher.execute(second, lambda: True, now=lambda: NOW)
    finally:
        publisher.close()


def test_journal_and_sink_remain_bounded(tmp_path, plan):
    path = tmp_path/'publication.db'
    publisher = ShadowPublication(path)
    try:
        for number in range(15):
            assert publisher.execute(next_plan(plan, str(number)), lambda: True, now=lambda: NOW)
        assert len(rows(path, 'jobs')) <= 10
        assert len(rows(path, 'outputs')) == 3
        assert len(rows(path, 'marker')) == 1
    finally:
        publisher.close()


def test_durable_journal_recovers_after_process_exit_between_write_and_ack(tmp_path, plan):
    import subprocess
    import sys
    path = tmp_path/'publication.db'
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(encoded(plan))
    code = """import json, os, sys
from energy_pipeline.publication import ShadowPublication
from test_accepted_store import NOW
publisher = ShadowPublication(sys.argv[1])
publisher._ack = lambda *args: os._exit(23)
publisher.execute(json.load(open(sys.argv[2])), lambda: True, now=lambda: NOW)
"""
    # Add test directory explicitly; child otherwise imports only production code.
    from pathlib import Path
    import os
    env = dict(os.environ, PYTHONPATH=str(Path('tests/energy_pipeline').resolve())+os.pathsep+str(Path.cwd()))
    child = subprocess.run([sys.executable, '-c', code, str(path), str(plan_path)], env=env, timeout=10)
    assert child.returncode == 23
    publisher = ShadowPublication(path)
    try:
        assert len(publisher.recover()) == 1
        assert len(rows(path, 'outputs')) == 1 and not rows(path, 'marker')
        assert publisher.execute(plan, lambda: True, now=lambda: NOW)
        assert publisher.committed_family() == plan
    finally:
        publisher.close()
