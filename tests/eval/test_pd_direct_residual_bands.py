import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pd_direct_baseline import (
    apply_pd_residual_bands,
    horizon_bucket_from_hours,
    pd_level_bucket,
)


def test_pd_residual_bucket_classification():
    assert horizon_bucket_from_hours(1.0) == "1-6h"
    assert horizon_bucket_from_hours(6.0) == "6-14h"
    assert horizon_bucket_from_hours(14.0) == "14-30h"
    assert horizon_bucket_from_hours(30.0) == "30h+"

    assert pd_level_bucket(-1.0) == "<0"
    assert pd_level_bucket(59.9) == "0-60"
    assert pd_level_bucket(149.9) == "60-150"
    assert pd_level_bucket(299.9) == "150-300"
    assert pd_level_bucket(300.0) == ">300"


def test_apply_pd_residual_bands_clamps_around_q50_and_falls_back():
    idx = pd.date_range("2025-09-01T00:00:00Z", periods=2, freq="30min")
    q50 = pd.Series([100.0, 200.0], index=idx)
    bands = pd.DataFrame(
        [
            {
                "granularity": "horizon_hod_level",
                "horizon_bucket": "1-6h",
                "hod_30m": 0,
                "pd_level_bucket": "60-150",
                "q20": -30.0,
                "q80": 50.0,
            },
            {
                "granularity": "horizon",
                "horizon_bucket": "1-6h",
                "q20": 20.0,
                "q80": -20.0,
            },
        ]
    )

    lower, upper = apply_pd_residual_bands(q50, idx[0], bands)

    assert lower.iloc[0] == 70.0
    assert upper.iloc[0] == 150.0
    # Fallback row has inverted positive/negative residuals, so monotonic clamp
    # should collapse both sides to q50 rather than crossing.
    assert lower.iloc[1] == 200.0
    assert upper.iloc[1] == 200.0
