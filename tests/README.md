# Test Framework

## Structure

```
tests/
  unit/                        # Fast tests — no external deps, run in <60s
    conftest.py                # Shared fixtures, monkeypatching InfluxDB + HA calls
    test_tier1_tactical.py     # Tier 1 LightGBM: shape [12], no NaN, no quantile crossing
    test_tft_price.py          # Tier 2 TFT: shape [144,6], no NaN, decoder 15 features wired
    test_combined_sensor.py    # Amber-format attrs: Tier 1 5-min, Tier 2 30-min, 144 total items
    test_tariffs.py            # wholesale → general/feed-in: sign, GST, loss factor
    test_feed_in_sign.py       # feed-in negative for positive wholesale (regression guard)
    test_data_pipeline.py      # scaler keys present for all features; no dimension mismatches
  fixtures/
    influx_snapshot.json       # Canned InfluxDB query responses (get_historical_data + SDO)
    ha_entities.json           # Canned HA entity states (weather, Solcast, Amber)
    capture_fixtures.py        # Script: run live predict cycle, write fixtures above
  eval/
    test_financial_gate.py     # Runs holistic_eval on stratified set; asserts thresholds
```

## Running

```bash
# Fast unit tests only (no InfluxDB or HA required)
pytest tests/unit/ -v

# Full suite including financial gate (requires InfluxDB)
pytest tests/ -v
```

## Fixture capture

Before writing unit tests, capture a live run snapshot:

```bash
source .venv/bin/activate
python tests/fixtures/capture_fixtures.py
```

This runs one full `predict-all` cycle with mocked HA publishing, intercepts all InfluxDB
queries and HA REST calls, and writes the responses to `tests/fixtures/`. Commit the
resulting fixtures — they are frozen and must not be regenerated silently.

**To refresh fixtures** after a deliberate pipeline change: re-run `capture_fixtures.py`,
review the diff carefully, then commit.

---

## Phase 8 — Two Layers

### Layer 1: Unit/regression tests

**No dependency on Phase 6. Start immediately.**

Key invariants checked:

| Test | What it catches |
|------|----------------|
| `test_tier1_tactical` | Shape mismatch, NaN output, quantile ordering |
| `test_tft_price` | Decoder feature count (must be 15), shape, value range |
| `test_combined_sensor` | Tier 1 items have 5-min intervals, Tier 2 have 30-min; feed-in sign convention |
| `test_tariffs` | GST applied correctly, loss factor sign, feed-in negation |
| `test_feed_in_sign` | Regression guard: positive wholesale → negative feed-in `per_kwh` |
| `test_data_pipeline` | All scaler keys present (`sd_demand`, `sd_net_interchange`, etc.) |

These tests caught two real bugs in April 2026: inverted feed-in sign and missing TFT
decoder features (13 vs 15). Keeping them prevents recurrence.

### Layer 2: Financial eval gate

**Depends on Phase 6 baseline. Wire up after `holistic_eval.py` produces results.**

Gate thresholds (set after Phase 6 runs):
- AI pipeline mean $/day ≥ legacy LightGBM baseline
- Spike stratum: no worse than legacy − 5%
- Low/normal stratum: no worse than legacy − 2%

**Both layers must pass before Phase 5 sub-tasks 4–8 resume.**
