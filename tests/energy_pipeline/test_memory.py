from unittest.mock import Mock
import energy_pipeline.memory as memory


def test_high_rss_collects_cycles_then_trims_and_reports_cost(monkeypatch):
    events = []
    gc = Mock(side_effect=lambda: events.append('gc'))
    trim = Mock(side_effect=lambda pad: events.append(('trim', pad)) or 1)
    samples = iter([2000, 1900, 1400])
    monkeypatch.setattr(memory, 'rss_mib', lambda: next(samples))
    monkeypatch.setattr(memory.gc, 'collect', gc)
    monkeypatch.setattr(memory, 'allocator_trim', lambda: trim)
    result = memory.reclaim_transient_memory()
    assert events == ['gc', ('trim', 0)]
    assert result.before_mib == 2000
    assert result.after_mib == 1400
    assert result.trimmed


def test_below_threshold_does_not_call_allocator(monkeypatch):
    monkeypatch.setattr(memory, 'rss_mib', lambda: 1000)
    monkeypatch.setattr(memory.gc, 'collect', lambda: None)
    allocator = Mock()
    monkeypatch.setattr(memory, 'allocator_trim', allocator)
    assert not memory.reclaim_transient_memory().trimmed
    allocator.assert_not_called()


def test_missing_gnu_extension_leaves_resource_guard_to_handle_high_rss(monkeypatch):
    monkeypatch.setattr(memory, 'rss_mib', lambda: 3000)
    monkeypatch.setattr(memory.gc, 'collect', lambda: None)
    monkeypatch.setattr(memory, 'allocator_trim', lambda: None)
    result = memory.reclaim_transient_memory()
    assert result.after_mib == 3000
    assert not result.trimmed
