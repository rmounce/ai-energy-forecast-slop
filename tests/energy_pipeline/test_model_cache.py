import json
from pathlib import Path
import pytest
from energy_pipeline.model_cache import ModelCache


def artifacts(tmp_path, prefix='a'):
    result = {}
    for name in ('low', 'mid', 'high'):
        model = tmp_path / f'{prefix}_{name}.pkl'
        params = tmp_path / f'{prefix}_{name}.json'
        model.write_text(name)
        params.write_text(json.dumps({'shift_value': 1}))
        result[name] = {'model': model, 'params': params}
    return result


def test_reuses_family_and_observes_parameter_replacement(tmp_path):
    calls = []
    def load(path):
        calls.append(path)
        return object()
    cache = ModelCache(load)
    paths = artifacts(tmp_path)
    original = cache.load_family(paths)
    assert cache.load_family(paths) is original
    assert len(calls) == 3
    paths['mid']['params'].write_text('{"shift_value": 222}')
    changed = cache.load_family(paths)
    assert changed is not original
    assert changed['mid'][1] == {'shift_value': 222}
    assert len(calls) == 6


def test_failed_promotion_keeps_previous_complete_family(tmp_path):
    cache = ModelCache(lambda path: Path(path).read_text())
    old_paths = artifacts(tmp_path)
    previous = cache.load_family(old_paths)
    new_paths = artifacts(tmp_path, 'b')
    new_paths['high']['params'].write_text('broken')
    with pytest.raises(ValueError):
        cache.load_family(new_paths)
    assert cache.load_family(old_paths) is previous
    new_paths['high']['params'].write_text('{"shift_value": 3}')
    assert cache.load_family(new_paths) is not previous


def test_artifact_change_during_load_rejects_entire_family(tmp_path):
    paths = artifacts(tmp_path)
    def load(path):
        paths['low']['params'].write_text('{"shift_value": 123456}')
        return object()
    cache = ModelCache(load)
    with pytest.raises(RuntimeError, match='changed while loading'):
        cache.load_family(paths)
