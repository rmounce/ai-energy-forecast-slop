from urllib.error import URLError

import pytest

import healthchecks


def test_project_ping_key_gives_each_job_its_own_auto_provisioned_check():
    assert healthchecks.build_ping_url(
        "predict-load", project_key="project-key"
    ) == "https://hc-ping.com/project-key/predict-load?create=1"
    assert healthchecks.build_ping_url(
        "price-listener", project_key="project-key"
    ) == "https://hc-ping.com/project-key/price-listener?create=1"
    assert healthchecks.build_ping_url(
        "aemo-pec-mi-transition", failed=True, project_key="project-key"
    ) == "https://hc-ping.com/project-key/aemo-pec-mi-transition/fail?create=1"


def test_project_ping_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("HC_REPO_PING_KEY", "project-key")

    assert healthchecks.build_ping_url("predict-load") == (
        "https://hc-ping.com/project-key/predict-load?create=1"
    )


def test_legacy_uuid_url_is_used_only_when_project_key_is_unset(monkeypatch):
    monkeypatch.delenv("HC_REPO_PING_KEY", raising=False)
    legacy_url = "https://hc-ping.com/old-uuid"

    assert healthchecks.build_ping_url("predict-load", legacy_url=legacy_url) == legacy_url
    assert healthchecks.build_ping_url(
        "predict-load", failed=True, legacy_url=legacy_url
    ) == f"{legacy_url}/fail"


def test_missing_key_skips_ping_and_ping_errors_do_not_reveal_project_key(monkeypatch):
    monkeypatch.delenv("HC_REPO_PING_KEY", raising=False)
    assert healthchecks.ping_check("aemo-pec-mi-transition") is False

    secret = "hidden-project-key"
    monkeypatch.setenv("HC_REPO_PING_KEY", secret)
    monkeypatch.setattr(
        healthchecks, "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError(f"bad URL {secret}")),
    )
    with pytest.raises(healthchecks.HealthcheckPingError) as error:
        healthchecks.ping_check("aemo-pec-mi-transition")
    assert secret not in str(error.value)
