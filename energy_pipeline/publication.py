"""Durable local publication rehearsal. No HA output/network transport exists here."""
from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3

import pandas as pd

from energy_pipeline.accepted_store import RecoveredBundle, StoreError
from energy_pipeline.tariffs import TariffSnapshot, frozen_tariffs

KEYS = ('price_p30', 'price', 'price_p70')


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def prepare_plan(bundle, config):
    """Freeze incumbent HA payloads once; retries preserve values and last_updated."""
    import forecast as fc
    bundle.validate()
    data = bundle.payload
    tariff = dict(data['tariff_snapshot'])
    for name in ('general_tariff', 'feed_in_tariff'):
        tariff[name] = tuple(tuple(pair) for pair in tariff[name])
    writes = []
    with frozen_tariffs(TariffSnapshot(**tariff)):
        for key in KEYS:
            item = data['forecasts'][key]
            frame = pd.DataFrame({'wholesale_price': item['values']},
                                 index=pd.to_datetime(item['timestamps'], utc=True))
            fc.apply_tariffs_to_forecast(frame)
            records = frame.copy()
            records.insert(0, 'timestamp', [value.isoformat() for value in frame.index])
            writes.append({'entity': config['home_assistant']['publish_entities'][key],
                'payload': {'state': round(float(frame.iloc[0]['general_price']), 4),
                    'attributes': {'forecasts': records.to_dict('records'),
                        'last_updated': data['accepted_at'],
                        'friendly_name': f"AI {key.replace('_', ' ').title()} Forecast",
                        'icon': 'mdi:chart-line'}}})
    if len({write['entity'] for write in writes}) != 3:
        raise ValueError('publication targets must be distinct')
    plan = {'schema': 1, 'mode': 'local_shadow', 'bundle': data, 'writes': writes}
    # Identity includes targets, tariffs, values and stable publication timestamp.
    plan['id'] = hashlib.sha256(encoded(plan).encode()).hexdigest()
    return json.loads(encoded(plan))  # owned JSON types; identical after restart


class ShadowPublication:
    """One cooperating process, one worker; bounded journal + local mock sink.

    Sink write and receipt are separate durable operations to model the real
    remote-write/acknowledgement crash window. Commit marker is local simulation.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = os.open(str(self.path)+'.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not self.path.exists():
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                os.close(fd)
            with self._connect() as db:
                db.executescript('''
                    CREATE TABLE IF NOT EXISTS jobs
                    (id TEXT PRIMARY KEY, plan TEXT NOT NULL, status TEXT NOT NULL, receipts TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS outputs
                    (entity TEXT PRIMARY KEY, job TEXT NOT NULL, payload TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS marker
                    (singleton INTEGER PRIMARY KEY CHECK(singleton=1), job TEXT NOT NULL);
                ''')
        except Exception as exc:
            self.close()
            raise StoreError('cannot open local publication journal') from exc

    @contextmanager
    def _connect(self):
        if self._lock is None:
            raise StoreError('local publication journal closed')
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def close(self):
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def recover(self):
        """Inspect pending jobs. Caller must revalidate before explicit resume."""
        with self._connect() as db:
            return [(json.loads(plan), json.loads(receipts)) for plan, receipts in
                    db.execute("SELECT plan, receipts FROM jobs WHERE status='pending'")]

    def abandon_pending(self):
        # Resident startup deliberately reconciles fresh inputs instead of replaying
        # historical accepted bundles. Keep unfinished plans as diagnostic evidence.
        with self._connect() as db:
            count = db.execute("UPDATE jobs SET status='abandoned' WHERE status='pending'").rowcount
        return count

    def committed_family(self):
        """Local consumer guard: reject mixed sink generations even with an old marker."""
        with self._connect() as db:
            db.execute('BEGIN')
            row = db.execute('SELECT plan FROM jobs JOIN marker ON jobs.id=marker.job WHERE jobs.status=\'committed\'').fetchone()
            if row is None:
                return None
            plan = json.loads(row[0])
            for write in plan['writes']:
                current = db.execute('SELECT job, payload FROM outputs WHERE entity=?', (write['entity'],)).fetchone()
                if current != (plan['id'], encoded(write['payload'])):
                    return None
            return plan

    def _write(self, job, write):
        with self._connect() as db:
            db.execute('INSERT INTO outputs VALUES (?, ?, ?) ON CONFLICT(entity) DO UPDATE SET job=excluded.job, payload=excluded.payload',
                       (write['entity'], job, encoded(write['payload'])))

    def _ack(self, job, receipts):
        with self._connect() as db:
            db.execute('UPDATE jobs SET receipts=? WHERE id=?', (encoded(receipts), job))

    def execute(self, plan, revalidate, *, now=None):
        """Return False if superseded/expired; interrupted plans can be resumed explicitly."""
        try:
            return self._execute(plan, revalidate, now=now)
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError('local publication rehearsal failed; journal retained') from exc

    def _execute(self, plan, revalidate, *, now=None):
        if self._lock is None:
            raise StoreError('local publication journal closed')
        job = plan['id']
        unsigned = {key: value for key, value in plan.items() if key != 'id'}
        if plan.get('schema') != 1 or plan.get('mode') != 'local_shadow' or job != hashlib.sha256(encoded(unsigned).encode()).hexdigest():
            raise StoreError('invalid local publication plan')
        if len(plan['writes']) != 3 or len({write['entity'] for write in plan['writes']}) != 3:
            raise StoreError('invalid local publication targets')
        RecoveredBundle(plan['bundle']).validate()
        if len(encoded(plan).encode()) > 512 * 1024:
            raise StoreError('local publication plan exceeds size limit')
        clock = now or (lambda: datetime.now(timezone.utc))
        def valid():
            return RecoveredBundle(plan['bundle']).time_current(clock()) and revalidate()
        with self._connect() as db:
            prior = db.execute('SELECT plan, status, receipts FROM jobs WHERE id=?', (job,)).fetchone()
            if prior:
                if prior[0] != encoded(plan):
                    raise StoreError('publication identity collision')
                if prior[1] == 'committed':
                    return True  # No sink replay, no lease renewal.
                if prior[1] == 'abandoned':
                    return False
                receipts = json.loads(prior[2])
            else:
                if not valid():
                    return False
                db.execute("UPDATE jobs SET status='abandoned' WHERE status='pending'")
                # Bound history; active job/marker retained. Local sink uses only current targets.
                db.execute("DELETE FROM jobs WHERE id IN (SELECT id FROM jobs WHERE status != 'pending' AND id NOT IN (SELECT job FROM marker) ORDER BY rowid DESC LIMIT -1 OFFSET 8)")
                db.execute('INSERT INTO jobs VALUES (?, ?, ?, ?)', (job, encoded(plan), 'pending', '[]'))
                receipts = []
        for write in plan['writes']:
            if not valid():
                with self._connect() as db:
                    db.execute("UPDATE jobs SET status='abandoned' WHERE id=?", (job,))
                return False
            if write['entity'] not in receipts:
                # Crash after this write but before receipt →repeat identical local payload.
                self._write(job, write)
                receipts.append(write['entity'])
                self._ack(job, receipts)
        if not valid():
            with self._connect() as db:
                db.execute("UPDATE jobs SET status='abandoned' WHERE id=?", (job,))
            return False
        with self._connect() as db:
            # Verify receipts correspond to the simulated sink, not merely journal progress.
            for write in plan['writes']:
                row = db.execute('SELECT job, payload FROM outputs WHERE entity=?', (write['entity'],)).fetchone()
                if row != (job, encoded(write['payload'])):
                    raise StoreError('local publication sink differs from receipt')
            db.execute('INSERT INTO marker VALUES (1, ?) ON CONFLICT(singleton) DO UPDATE SET job=excluded.job', (job,))
            db.execute("UPDATE jobs SET status='committed' WHERE id=?", (job,))
            db.execute('DELETE FROM outputs WHERE entity NOT IN (?, ?, ?)', tuple(write['entity'] for write in plan['writes']))
        return True
