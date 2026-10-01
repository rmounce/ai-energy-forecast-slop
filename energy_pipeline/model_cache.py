"""Single-worker model family cache; stage replacements before committing them."""
from __future__ import annotations
import json
from pathlib import Path
import time
import joblib


class ModelCache:
    def __init__(self, loader=joblib.load):
        self.loader = loader
        self._signature = None
        self._family = {}
        self.loads = 0
        self.last_load_seconds = 0.0

    @staticmethod
    def signature(paths):
        result = []
        for name, artifacts in sorted(paths.items()):
            for kind in ('model', 'params'):
                path = Path(artifacts[kind]).resolve()
                stat = path.stat()
                result.append((name, kind, str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns))
        return tuple(result)

    @property
    def loaded_signature(self):
        """Identity of the installed family, never a newly resolved pointer."""
        return self._signature

    def load_family(self, paths):
        signature = self.signature(paths)
        self.last_load_seconds = 0.0
        if signature == self._signature:
            return self._family
        started = time.monotonic()
        replacement = {}
        for name, artifacts in paths.items():
            model = self.loader(artifacts['model'])
            self.loads += 1
            with Path(artifacts['params']).open() as handle:
                params = json.load(handle)
            replacement[name] = (model, params)
        if self.signature(paths) != signature:
            raise RuntimeError('model artifacts changed while loading family')
        self._family, self._signature = replacement, signature
        self.last_load_seconds = time.monotonic()-started
        return self._family
