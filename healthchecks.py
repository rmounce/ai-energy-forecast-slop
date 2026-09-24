"""Coalesce repository job health into the single configured Healthchecks check.

Jobs write local status. A one-minute aggregator pings the existing check only
when every job is healthy, and sends ``/fail`` when any job fails or goes stale.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
STATUS_ROOT = ROOT / "data" / "healthcheck_status"
AGGREGATE_STATE = STATUS_ROOT / "aggregate.json"
JOBS = {
    "predict-load": {"period_seconds": 1800, "grace_seconds": 600},
    "price-listener": {"period_seconds": 1800, "grace_seconds": 600},
    "aemo-pec-mi-transition": {"period_seconds": 300, "grace_seconds": 180},
}


class HealthcheckError(RuntimeError):
    """A sanitized healthcheck configuration or transport error."""


@contextmanager
def _registry_lock(status_root: Path):
    """Serialize status writers with aggregate evaluation across processes."""
    status_root.mkdir(parents=True, exist_ok=True)
    with (status_root / ".lock").open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _utc_now(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    if not isinstance(value, dict):
        raise HealthcheckError("health status file is not a JSON object")
    return value


def record_job_status(
    job: str,
    exit_code: int,
    *,
    root: Path | str = ROOT,
    now: datetime | None = None,
) -> dict:
    """Atomically write this job's latest outcome and preserve unseen failures."""
    if job not in JOBS:
        raise HealthcheckError("unknown monitored job")
    directory = Path(root) / "data" / "healthcheck_status"
    with _registry_lock(directory):
        return _record_job_status_unlocked(job, exit_code, directory, now)


def _record_job_status_unlocked(
    job: str, exit_code: int, directory: Path, now: datetime | None,
) -> dict:
    at = _utc_now(now)
    path = directory / f"{job}.json"
    previous = _read_json(path)
    succeeded = int(exit_code) == 0
    status = {
        "job": job,
        "updated_at": _iso(at),
        "last_success_at": _iso(at) if succeeded else previous.get("last_success_at"),
        "last_outcome": "success" if succeeded else "failure",
        "last_failure_at": previous.get("last_failure_at") if succeeded else _iso(at),
        "failure_pending": bool(previous.get("failure_pending")) or not succeeded,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, sort_keys=True) + "\n")
    temporary.replace(path)
    return status


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _failures(status_root: Path, started_at: datetime, now: datetime) -> list[str]:
    failures = []
    for job, schedule in JOBS.items():
        status = _read_json(status_root / f"{job}.json")
        last_success = _parse_time(status.get("last_success_at"))
        if status.get("failure_pending") or status.get("last_outcome") == "failure":
            failures.append(f"{job}: recorded failure")
        elif last_success is not None:
            allowed_age = schedule["period_seconds"] + schedule["grace_seconds"]
            if (now - last_success).total_seconds() > allowed_age:
                failures.append(f"{job}: stale success")
        elif (now - started_at).total_seconds() > (
            schedule["period_seconds"] + schedule["grace_seconds"]
        ):
            failures.append(f"{job}: no successful run")
    return failures


def _clear_reported_failures(status_root: Path) -> None:
    """Clear local failure latches once the shared check is known to be down."""
    for job in JOBS:
        path = status_root / f"{job}.json"
        job_state = _read_json(path)
        if job_state.get("failure_pending"):
            job_state["failure_pending"] = False
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(job_state, sort_keys=True) + "\n")
            temporary.replace(path)


def _ping(url: str, *, failed: bool, timeout: int = 10) -> None:
    endpoint = url.rstrip("/") + ("/fail" if failed else "")
    request = urllib.request.Request(endpoint, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if not 200 <= response.status < 300:
                raise HealthcheckError("Healthchecks endpoint returned an error")
    except HealthcheckError:
        raise
    except Exception as exc:
        # URL errors contain the credential-bearing URL; never include their text.
        raise HealthcheckError(f"Healthchecks ping failed ({type(exc).__name__})") from None


def aggregate_once(
    *,
    root: Path | str = ROOT,
    healthcheck_url: str | None = None,
    now: datetime | None = None,
    ping_fn: Callable[..., None] | None = None,
    force: bool = False,
) -> dict:
    """Evaluate local job state and heartbeat the one configured remote check."""
    url = (healthcheck_url if healthcheck_url is not None else os.environ.get("HC_PREDICT_URL", "")).strip()
    if not url:
        raise HealthcheckError("HC_PREDICT_URL is unset")
    status_root = Path(root) / "data" / "healthcheck_status"
    with _registry_lock(status_root):
        return _aggregate_once_unlocked(status_root, url, now, ping_fn, force)


def _aggregate_once_unlocked(
    status_root: Path, url: str, now: datetime | None,
    ping_fn: Callable[..., None] | None, force: bool,
) -> dict:
    current = _utc_now(now)
    state_path = status_root / AGGREGATE_STATE.name
    state = _read_json(state_path)
    started_at = _parse_time(state.get("monitor_started_at")) or current
    failures = _failures(status_root, started_at, current)
    desired = "failure" if failures else "success"
    previous_remote = state.get("last_sent_status")
    should_ping = force or desired == "success" or previous_remote != "failure"
    if should_ping:
        (ping_fn or _ping)(url, failed=desired == "failure")
        state["last_sent_status"] = desired
    if desired == "failure" and (should_ping or previous_remote == "failure"):
        # A successful /fail ping or the last recorded remote failure means
        # the condition is already visible. Later job success can recover it.
        _clear_reported_failures(status_root)
    state.setdefault("monitor_started_at", _iso(started_at))
    state["last_evaluated_at"] = _iso(current)
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
    temporary.replace(state_path)
    return {"status": desired, "failures": failures, "pinged": should_ping}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="record one local job result")
    record.add_argument("job", choices=JOBS)
    record.add_argument("--exit-code", type=int, required=True)
    aggregate = commands.add_parser("aggregate", help="evaluate and ping the shared check")
    aggregate.add_argument(
        "--force", action="store_true",
        help="send the current state even if it matches the last recorded remote state",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            record_job_status(args.job, args.exit_code)
            return 0
        result = aggregate_once(force=args.force)
        if result["failures"]:
            print("Repository healthcheck failed: " + "; ".join(result["failures"]))
        return 0
    except HealthcheckError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
