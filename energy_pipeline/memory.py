"""Reclaim transient Python cycles and, where available, unused GNU libc pages."""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
from functools import lru_cache
import gc
from pathlib import Path
import time

TRIM_THRESHOLD_MIB = 1536


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])/1024
    return 0.0


@lru_cache(maxsize=1)
def allocator_trim():
    # GNU extension; optional on other libc implementations. Documented MT-Safe.
    # Returning unused pages cannot release allocations still owned by a source worker.
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


@dataclass(frozen=True)
class MemoryMaintenance:
    before_mib: float
    after_mib: float
    elapsed_seconds: float
    trimmed: bool


def reclaim_transient_memory():
    started = time.monotonic()
    before = rss_mib()
    gc.collect()
    trimmed = False
    if rss_mib() > TRIM_THRESHOLD_MIB:
        trim = allocator_trim()
        if trim is not None:
            trimmed = bool(trim(0))
    return MemoryMaintenance(before, rss_mib(), time.monotonic()-started, trimmed)
