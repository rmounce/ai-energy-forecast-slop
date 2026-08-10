"""Atomic, versioned production model bundle lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class BundleError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BundleStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def family_dir(self, family: str) -> Path:
        return self.root / family

    def bundle_dir(self, family: str, bundle_id: str) -> Path:
        return self.family_dir(family) / "bundles" / bundle_id

    def active_pointer(self, family: str) -> Path:
        return self.family_dir(family) / "active.json"

    def resolve_active(self, family: str) -> tuple[str, Path]:
        pointer = self.active_pointer(family)
        try:
            data = json.loads(pointer.read_text())
            bundle_id = data["bundle_id"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
            raise BundleError(f"no valid active pointer for {family}") from exc
        path = self.bundle_dir(family, bundle_id)
        self.validate(family, bundle_id)
        return bundle_id, path

    def validate(self, family: str, bundle_id: str) -> dict:
        path = self.bundle_dir(family, bundle_id)
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            raise BundleError(f"incomplete bundle: {family}/{bundle_id}")
        try:
            manifest = json.loads(manifest_path.read_text())
            artifacts = manifest["artifacts"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise BundleError(f"invalid manifest: {manifest_path}") from exc
        for name, expected in artifacts.items():
            artifact = path / name
            if not artifact.is_file() or sha256(artifact) != expected:
                raise BundleError(f"hash mismatch or missing artifact: {artifact}")
        if manifest.get("family") != family or manifest.get("bundle_id") != bundle_id:
            raise BundleError(f"manifest identity mismatch: {manifest_path}")
        return manifest

    def write_candidate(self, family: str, bundle_id: str, artifacts: dict[str, bytes], manifest: dict) -> Path:
        """Write a complete candidate; interruption cannot touch active state."""
        destination = self.bundle_dir(family, bundle_id)
        if destination.exists():
            raise BundleError(f"bundle already exists: {family}/{bundle_id}")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix=f".{bundle_id}.", dir=parent))
        try:
            for name, content in artifacts.items():
                target = temp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            complete = dict(manifest)
            complete.update({"bundle_id": bundle_id, "family": family})
            complete.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
            complete["artifacts"] = {name: sha256(temp / name) for name in artifacts}
            (temp / "manifest.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
            self.validate_temp(family, bundle_id, temp, complete)
            os.replace(temp, destination)
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise
        return destination

    @staticmethod
    def validate_temp(family: str, bundle_id: str, path: Path, manifest: dict) -> None:
        if manifest.get("family") != family or manifest.get("bundle_id") != bundle_id:
            raise BundleError("candidate manifest identity mismatch")
        for name, expected in manifest.get("artifacts", {}).items():
            if not (path / name).is_file() or sha256(path / name) != expected:
                raise BundleError(f"candidate artifact failed validation: {name}")

    def promote(self, family: str, bundle_id: str, *, report: dict | None = None) -> None:
        manifest = self.validate(family, bundle_id)
        if report is None or report.get("eligible_for_manual_promotion") is not True:
            raise BundleError("candidate report does not permit manual promotion")
        pointer = self.active_pointer(family)
        pointer.parent.mkdir(parents=True, exist_ok=True)
        old = None
        if pointer.exists():
            old = json.loads(pointer.read_text()).get("bundle_id")
        payload = {"bundle_id": bundle_id, "family": family, "previous_bundle_id": old, "promoted_utc": datetime.now(timezone.utc).isoformat()}
        temp = pointer.with_name(f".{pointer.name}.tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temp, pointer)

    def rollback(self, family: str) -> str:
        pointer = self.active_pointer(family)
        try:
            data = json.loads(pointer.read_text())
            previous = data.get("previous_bundle_id")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise BundleError(f"cannot rollback {family}: invalid active pointer") from exc
        if not previous:
            raise BundleError(f"cannot rollback {family}: no predecessor")
        self.validate(family, previous)
        self.promote(family, previous, report={"eligible_for_manual_promotion": True})
        return previous

    def migrate_root_artifacts(self, family: str, bundle_id: str, files: Iterable[Path]) -> Path:
        """Import existing root artifacts once; refuse partial/conflicting state."""
        destination = self.bundle_dir(family, bundle_id)
        if destination.exists() or self.active_pointer(family).exists():
            raise BundleError(f"migration state already exists for {family}")
        paths = list(files)
        if not paths or not all(path.is_file() for path in paths):
            raise BundleError("migration requires every root artifact to exist")
        artifacts = {path.name: path.read_bytes() for path in paths}
        return self.write_candidate(family, bundle_id, artifacts, {
            "migration": "root-artifacts",
            "git_commit": "unknown",
            "config_digest": "unknown",
            "training_start_utc": None,
            "training_end_utc": None,
            "row_count": None,
            "requested_quantiles": [],
            "feature_list": [],
            "lightgbm_parameters": {},
            "shift_values": {},
            "producing_command": "./forecast.py migrate-<family>-bundle",
            "parent_bundle_id": None,
        })
