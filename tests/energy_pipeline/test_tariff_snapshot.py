from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import json
import threading
import pandas as pd
import pytest
import forecast as fc
from energy_pipeline import tariffs
from tariff_utils import load_tariff_profile


def profile(scale=1.1, loss=1.05, rate=.03):
    return {'amber_api_scaling_factor': scale, 'network_loss_factor': loss,
            'general_tariff': {'00:00:00': rate}, 'feed_in_tariff': {'00:00:00': .01}}


def test_frozen_readers_preserve_profile_after_file_changes_and_match_legacy(tmp_path, monkeypatch):
    path = tmp_path/'tariffs.json'
    path.write_text(json.dumps(profile()))
    config = {'paths': {'tariff_file': str(path)}, 'timezone': 'UTC', 'gst_rate': 1.1}
    monkeypatch.setattr(fc, 'CONFIG', config)
    snapshot = tariffs.capture_tariffs(path)
    frame = pd.DataFrame({'wholesale_price': [.2, -.2]},
                         index=pd.date_range('2026-10-02T00:00:00Z', periods=2, freq='1D'))
    legacy = frame.copy()
    fc.apply_tariffs_to_forecast(legacy)
    with fc.prediction_resources(entity_states={}, model_cache=object(), tariff_snapshot=snapshot):
        path.write_text(json.dumps(profile(scale=2, loss=2, rate=.8)))
        assert fc.get_amber_api_scaling_factor() == 1.1
        assert fc.get_network_loss_factor() == 1.05
        assert load_tariff_profile(config, tmp_path) == snapshot.effective_profile
        frozen = frame.copy()
        fc.apply_tariffs_to_forecast(frozen)
        pd.testing.assert_frame_equal(frozen, legacy, check_exact=True)
        # Even deleting the file cannot change any captured accessor.
        path.unlink()
        assert fc.get_amber_api_scaling_factor() == 1.1
        assert fc.get_network_loss_factor() == 1.05
    assert fc.get_amber_api_scaling_factor() == 1.10  # incumbent missing-file fallback restored
    assert tariffs.current_tariffs() is None


def test_missing_file_conversion_keeps_incumbent_no_tariff_fallback(tmp_path, monkeypatch):
    path = tmp_path/'missing.json'
    monkeypatch.setattr(fc, 'CONFIG', {'paths': {'tariff_file': str(path)}})
    snapshot = tariffs.capture_tariffs(path)
    assert not snapshot.present
    frame = pd.DataFrame({'wholesale_price': [.2, -.2]})
    with tariffs.frozen_tariffs(snapshot):
        fc.apply_tariffs_to_forecast(frame)
    assert frame['general_price'].tolist() == [.2, -.2]
    assert frame['feed_in_price'].tolist() == [.2, -.2]


def test_snapshot_copies_and_nested_thread_contexts_are_isolated(tmp_path):
    path = tmp_path/'tariff.json'
    path.write_text(json.dumps(profile()))
    first = tariffs.capture_tariffs(path)
    path.write_text(json.dumps(profile(scale=2)))
    second = tariffs.capture_tariffs(path)
    first.effective_profile[0]['00:00:00'] = 99
    assert first.effective_profile[0]['00:00:00'] == .03
    barrier = threading.Barrier(2)
    def run(snapshot):
        with tariffs.frozen_tariffs(snapshot):
            barrier.wait(timeout=2)
            assert tariffs.current_tariffs() is snapshot
            with pytest.raises(RuntimeError), tariffs.frozen_tariffs(second):
                raise RuntimeError('test context restoration')
            assert tariffs.current_tariffs() is snapshot
        return tariffs.current_tariffs()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, snapshot) for snapshot in (first, second)]
        assert [future.result() for future in futures] == [None, None]


def test_malformed_nonfinite_and_mid_read_file_changes_fail_capture(tmp_path, monkeypatch):
    path = tmp_path/'tariff.json'
    path.write_text('{')
    with pytest.raises(ValueError): tariffs.capture_tariffs(path)
    path.write_text(json.dumps(profile(loss=float('nan'))))
    with pytest.raises(ValueError, match='nonfinite'): tariffs.capture_tariffs(path)
    path.write_text(json.dumps(profile()))
    stamps = iter([SimpleNamespace(st_ino=1, st_size=10, st_mtime_ns=1, st_ctime_ns=1),
                   SimpleNamespace(st_ino=1, st_size=10, st_mtime_ns=2, st_ctime_ns=2)])
    monkeypatch.setattr(tariffs.os, 'fstat', lambda fd: next(stamps))
    with pytest.raises(RuntimeError, match='changed during capture'): tariffs.capture_tariffs(path)
