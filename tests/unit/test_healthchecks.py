from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from urllib.error import URLError

import pytest

import healthchecks

NOW = datetime(2026, 9, 24, 12, tzinfo=timezone.utc)


def test_registry_has_one_status_schedule_per_monitored_job():
    assert set(healthchecks.JOBS) == {
        "predict-load", "price-listener", "aemo-pec-mi-transition",
    }
    assert healthchecks.JOBS["aemo-pec-mi-transition"] == {
        "period_seconds": 300, "grace_seconds": 600, "failure_runs": 2,
    }


def test_one_capture_failure_recovers_without_alert(tmp_path):
    calls = []
    ping = lambda _url, *, failed: calls.append(failed)
    healthchecks.record_job_status("aemo-pec-mi-transition", 0, root=tmp_path, now=NOW)
    healthchecks.record_job_status(
        "aemo-pec-mi-transition", 1, root=tmp_path, now=NOW + timedelta(minutes=5),
    )
    first = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=6), ping_fn=ping,
    )
    healthchecks.record_job_status(
        "aemo-pec-mi-transition", 0, root=tmp_path, now=NOW + timedelta(minutes=10),
    )
    second = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=11), ping_fn=ping,
    )
    assert first["status"] == second["status"] == "success"
    assert calls == [False, False]


def test_two_capture_failures_remain_pending_through_quick_recovery(tmp_path):
    healthchecks.record_job_status("aemo-pec-mi-transition", 0, root=tmp_path, now=NOW)
    for minutes in (5, 10):
        healthchecks.record_job_status(
            "aemo-pec-mi-transition", 1, root=tmp_path, now=NOW + timedelta(minutes=minutes),
        )
    healthchecks.record_job_status(
        "aemo-pec-mi-transition", 0, root=tmp_path, now=NOW + timedelta(minutes=11),
    )
    calls = []
    ping = lambda _url, *, failed: calls.append(failed)
    failed = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=11), ping_fn=ping,
    )
    recovered = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=12), ping_fn=ping,
    )
    assert failed["status"] == "failure"
    assert recovered["status"] == "success"
    assert calls == [True, False]


def test_job_failure_remains_pending_until_aggregate_reports_it(tmp_path):
    healthchecks.record_job_status("price-listener", 1, root=tmp_path, now=NOW)
    healthchecks.record_job_status("price-listener", 0, root=tmp_path, now=NOW + timedelta(seconds=10))
    state = healthchecks._read_json(tmp_path / "data/healthcheck_status/price-listener.json")
    assert state["last_outcome"] == "success"
    assert state["failure_pending"] is True
    assert state["last_success_at"] == "2026-09-24T12:00:10Z"


def test_aggregate_fails_once_for_a_job_failure_and_recovery_clears_check(tmp_path):
    calls = []
    ping = lambda _url, *, failed: calls.append(failed)
    healthchecks.record_job_status("price-listener", 1, root=tmp_path, now=NOW)

    first = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id", now=NOW, ping_fn=ping,
    )
    second = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=1), ping_fn=ping,
    )
    assert first["status"] == second["status"] == "failure"
    assert first["pinged"] is True and second["pinged"] is False
    assert calls == [True]

    healthchecks.record_job_status("price-listener", 0, root=tmp_path, now=NOW + timedelta(minutes=2))
    recovered = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=2), ping_fn=ping,
    )
    assert recovered["status"] == "success"
    assert recovered["pinged"] is True
    assert calls == [True, False]


def test_force_resynchronizes_remote_failure_even_if_local_state_says_failed(tmp_path):
    healthchecks.record_job_status("aemo-pec-mi-transition", 1, root=tmp_path, now=NOW)
    healthchecks.record_job_status("aemo-pec-mi-transition", 1, root=tmp_path, now=NOW)
    calls = []
    ping = lambda _url, *, failed: calls.append(failed)
    healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id", now=NOW, ping_fn=ping,
    )
    result = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(seconds=30), ping_fn=ping, force=True,
    )
    assert result["status"] == "failure" and result["pinged"] is True
    assert calls == [True, True]


def test_new_failure_while_remote_is_already_down_does_not_block_recovery(tmp_path):
    calls = []
    ping = lambda _url, *, failed: calls.append(failed)
    healthchecks.record_job_status("aemo-pec-mi-transition", 1, root=tmp_path, now=NOW)
    healthchecks.record_job_status("aemo-pec-mi-transition", 1, root=tmp_path, now=NOW)
    healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id", now=NOW, ping_fn=ping,
    )
    healthchecks.record_job_status(
        "aemo-pec-mi-transition", 1, root=tmp_path, now=NOW + timedelta(minutes=1),
    )
    already_down = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=1), ping_fn=ping,
    )
    state = healthchecks._read_json(tmp_path / "data/healthcheck_status/aemo-pec-mi-transition.json")
    assert already_down["status"] == "failure" and already_down["pinged"] is False
    assert state["failure_pending"] is False

    healthchecks.record_job_status(
        "aemo-pec-mi-transition", 0, root=tmp_path, now=NOW + timedelta(minutes=2),
    )
    recovered = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=2), ping_fn=ping,
    )
    assert recovered["status"] == "success" and recovered["pinged"] is True
    assert calls == [True, False]


def test_failure_recorded_during_aggregate_is_reported_on_next_pass(tmp_path):
    healthchecks.record_job_status("price-listener", 0, root=tmp_path, now=NOW)
    started = Event()
    finished = Event()
    worker = None

    def ping(_url, *, failed):
        nonlocal worker
        assert failed is False

        def record_failure():
            started.set()
            healthchecks.record_job_status(
                "price-listener", 1, root=tmp_path, now=NOW + timedelta(seconds=1),
            )
            finished.set()

        worker = Thread(target=record_failure)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.1)

    healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW, ping_fn=ping,
    )
    worker.join(timeout=1)
    assert finished.is_set()
    calls = []
    result = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(seconds=2),
        ping_fn=lambda _url, *, failed: calls.append(failed),
    )
    assert result["status"] == "failure"
    assert calls == [True]


def test_new_failure_cannot_be_cleared_by_earlier_failure_ping(tmp_path):
    healthchecks.record_job_status("price-listener", 1, root=tmp_path, now=NOW)
    started = Event()
    finished = Event()
    worker = None

    def ping(_url, *, failed):
        nonlocal worker
        assert failed is True

        def record_failure():
            started.set()
            healthchecks.record_job_status(
                "price-listener", 1, root=tmp_path, now=NOW + timedelta(seconds=1),
            )
            finished.set()

        worker = Thread(target=record_failure)
        worker.start()
        assert started.wait(1)
        assert not finished.wait(0.1)

    healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW, ping_fn=ping,
    )
    worker.join(timeout=1)
    assert finished.is_set()
    state = healthchecks._read_json(tmp_path / "data/healthcheck_status/price-listener.json")
    assert state["failure_pending"] is True


def test_success_from_one_job_cannot_mask_another_jobs_failure(tmp_path):
    healthchecks.record_job_status("predict-load", 0, root=tmp_path, now=NOW)
    healthchecks.record_job_status("aemo-pec-mi-transition", 1, root=tmp_path, now=NOW)
    healthchecks.record_job_status("aemo-pec-mi-transition", 1, root=tmp_path, now=NOW)
    calls = []
    result = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id", now=NOW,
        ping_fn=lambda _url, *, failed: calls.append(failed),
    )
    assert result["status"] == "failure"
    assert any("aemo-pec-mi-transition" in item for item in result["failures"])
    assert calls == [True]


def test_missing_job_status_is_allowed_only_during_initial_grace(tmp_path):
    healthchecks.record_job_status("predict-load", 0, root=tmp_path, now=NOW)
    healthchecks.record_job_status("price-listener", 0, root=tmp_path, now=NOW)
    calls = []
    ping = lambda _url, *, failed: calls.append(failed)
    first = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id", now=NOW, ping_fn=ping,
    )
    still_starting = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=7), ping_fn=ping,
    )
    overdue = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=16), ping_fn=ping,
    )
    assert first["status"] == still_starting["status"] == "success"
    assert overdue["status"] == "failure"
    assert any("aemo-pec-mi-transition: no successful run" in item for item in overdue["failures"])
    assert calls == [False, False, True]


def test_stale_capture_and_stale_listener_fail_independently(tmp_path):
    healthchecks.record_job_status("aemo-pec-mi-transition", 0, root=tmp_path, now=NOW)
    healthchecks.record_job_status("price-listener", 0, root=tmp_path, now=NOW)
    result = healthchecks.aggregate_once(
        root=tmp_path, healthcheck_url="https://hc-ping.com/private-id",
        now=NOW + timedelta(minutes=41), ping_fn=lambda *_args, **_kwargs: None,
    )
    assert result["status"] == "failure"
    assert any("aemo-pec-mi-transition: stale success" in item for item in result["failures"])
    assert any("price-listener: stale success" in item for item in result["failures"])


def test_missing_url_and_ping_transport_errors_are_sanitized(monkeypatch):
    monkeypatch.delenv("HC_PREDICT_URL", raising=False)
    with pytest.raises(healthchecks.HealthcheckError, match="HC_PREDICT_URL is unset"):
        healthchecks.aggregate_once()

    secret = "private-uuid"
    monkeypatch.setattr(
        healthchecks.urllib.request, "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError(f"failed {secret}")),
    )
    with pytest.raises(healthchecks.HealthcheckError) as error:
        healthchecks._ping(f"https://hc-ping.com/{secret}", failed=False)
    assert secret not in str(error.value)


def test_unknown_job_is_rejected(tmp_path):
    with pytest.raises(healthchecks.HealthcheckError, match="unknown monitored job"):
        healthchecks.record_job_status("arbitrary", 0, root=tmp_path, now=NOW)
