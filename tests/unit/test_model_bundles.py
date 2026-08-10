import json

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
    assert json.loads((path / "candidate_report.json").read_text())["smoke_result"] is True
    assert store.validate("price", "one")["bundle_id"] == "one"
