import json
from unittest.mock import patch

import pytest

from model_bundles import BundleError, BundleStore


def report():
    return {"eligible_for_manual_promotion": True}


def test_candidate_and_atomic_promotion_and_rollback(tmp_path):
    store = BundleStore(tmp_path / "models")
    first = store.write_candidate("price", "one", {"price_model.pkl": b"one"}, {})
    assert first.is_dir()
    store.promote("price", "one", report=report())
    second = store.write_candidate("price", "two", {"price_model.pkl": b"two"}, {})
    store.promote("price", "two", report=report())
    assert store.resolve_active("price")[0] == "two"
    assert store.rollback("price") == "one"
    assert store.resolve_active("price")[0] == "one"


def test_hash_mismatch_cannot_validate_or_promote(tmp_path):
    store = BundleStore(tmp_path / "models")
    path = store.write_candidate("load", "one", {"load_model.pkl": b"good"}, {})
    (path / "load_model.pkl").write_bytes(b"bad")
    with pytest.raises(BundleError, match="hash mismatch"):
        store.validate("load", "one")


def test_migration_refuses_partial_and_is_idempotent(tmp_path):
    store = BundleStore(tmp_path / "models")
    existing = tmp_path / "load_model.pkl"
    existing.write_bytes(b"model")
    with pytest.raises(BundleError, match="every root artifact"):
        store.migrate_root_artifacts("load", "initial", [existing, tmp_path / "missing.json"])
    params = tmp_path / "load_params.json"
    params.write_text("{}")
    path = store.migrate_root_artifacts("load", "initial", [existing, params])
    assert json.loads((path / "manifest.json").read_text())["family"] == "load"
    with pytest.raises(BundleError, match="already exists"):
        store.migrate_root_artifacts("load", "initial", [existing, params])


def test_report_replacement_refreshes_manifest_hash(tmp_path):
    store = BundleStore(tmp_path / "models")
    path = store.write_candidate("price", "one", {"price_model.pkl": b"model"}, {})
    report = {"family": "price", "bundle_id": "one", "smoke_result": True}
    store.replace_candidate_report("price", "one", report)
    manifest = store.validate("price", "one")
    report_names = [name for name in manifest["artifacts"] if name.startswith("candidate_report")]
    assert len(report_names) == 1
    assert json.loads((path / report_names[0]).read_text())["smoke_result"] is True
    assert manifest["bundle_id"] == "one"


def test_interrupted_report_install_preserves_previous_valid_report(tmp_path):
    store = BundleStore(tmp_path / "models")
    old = {"family": "price", "bundle_id": "one", "smoke_result": True}
    path = store.write_candidate("price", "one", {
        "price_model.pkl": b"model", "candidate_report.json": json.dumps(old).encode(),
    }, {})
    original_replace = __import__("model_bundles").os.replace
    calls = {"n": 0}

    def fail_manifest(source, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("injected manifest interruption")
        return original_replace(source, target)

    with patch("model_bundles.os.replace", side_effect=fail_manifest):
        with pytest.raises(OSError):
            store.replace_candidate_report("price", "one", {**old, "revision": 2})
    assert store.validate("price", "one")["bundle_id"] == "one"
    assert json.loads((path / "candidate_report.json").read_text()) == old


def test_interrupted_candidate_write_leaves_active_bundle_unchanged(tmp_path):
    store = BundleStore(tmp_path / "models")
    store.write_candidate("price", "active", {"model.pkl": b"active"}, {})
    store.promote("price", "active", report=report())
    original_write = __import__("pathlib").Path.write_bytes

    def fail_second_artifact(path, content):
        if path.name == "second.pkl":
            raise OSError("injected artifact interruption")
        return original_write(path, content)

    with patch("pathlib.Path.write_bytes", autospec=True, side_effect=fail_second_artifact):
        with pytest.raises(OSError, match="artifact interruption"):
            store.write_candidate(
                "price",
                "candidate",
                {"first.pkl": b"one", "second.pkl": b"two"},
                {},
            )

    assert store.resolve_active("price")[0] == "active"
    assert not store.bundle_dir("price", "candidate").exists()
