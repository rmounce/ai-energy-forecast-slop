# Pipeline Audit — 2026-05-11

Triggered after the user-visible "30-minute offset at a couple of points" in
`sensor.ai_spot_price_forecast` led to a chain of fixes (`f8bd8ea`, `f184320`,
`5a09c72`). User asked whether similar bugs exist elsewhere across data fetch,
ingest, training, prediction, and publish paths.

This doc catalogues what was checked and what was found.

## Bug shapes searched for

1. `SELECT last(field) GROUP BY time(N)` patterns on `run_time`-tagged Influx
   measurements — these mix stale rows from different forecast runs because
   Influx write order is not forecast-run order. The "first" bug of the
   session was this shape in `_get_influx_pd_prices`.
2. Interval-end vs interval-start mismatches between forecasts (AEMO native
   interval-end) and downstream readers (HA chart surfaces; the
   continuous-query-aggregated 30-min actuals which use Influx's default
   GROUP-BY-start bucketing). The PD-direct publish shift fix addressed this.
3. Timezone confusion (Brisbane AEST vs UTC) at conversion boundaries.
4. Implicit duration assumptions (the implementer's `_get_aemo_short_term_forecast`
   "first-row 5-min fallback" was an example).

## Findings (status)

### A. Stale-run-mixing `last() GROUP BY time()` instances

| Location | Status | Affects |
|---|---|---|
| `_get_influx_pd_prices` (forecast.py ~L1021) | **Fixed 2026-05-11 (`f8bd8ea`)** | PD-direct + TFT inference |
| `_execute_raw_aemo_stitched_price_forecast` PREDISPATCH (forecast.py ~L2146) | **Fixed by other implementer 2026-05-11 (`b815928`)** | `sensor.ai_aemo_price_forecast` |
| **`_get_influx_sdo_demand` (forecast.py ~L1127)** | **Fixed 2026-05-11 (this commit)** | TFT inference SDO covariate |
| `_get_influx_latest_pd7day_prices` (forecast.py ~L1078) | OK — uses `GROUP BY run_time`, picks max | — |
| `_execute_tactical_prediction` P5MIN (forecast.py ~L1343) | OK — uses `GROUP BY run_time`, picks max | — |
| `_execute_raw_aemo_stitched_price_forecast` P5MIN (forecast.py ~L2112) | OK — uses `GROUP BY run_time`, picks max | — |

Other queries with `GROUP BY time(30m)` (load/PV/weather/dispatch CQ-aggregated)
do not have `run_time` tags so cannot mix stale runs. Those are fine.

### B. Interval-end vs interval-start publish-side mismatches

| Surface | Status |
|---|---|
| `sensor.ai_aemo_price_forecast` (raw stitched) | **Fixed by other implementer (`b39eace`)**: P5MIN `-5min`, PREDISPATCH/PD7Day `-30min` |
| `sensor.ai_pd_direct_price_forecast(_low/_high)` | **Fixed 2026-05-11 (`f184320`)**: `-30min` on publish |
| `sensor.ai_p5min_price_forecast(_low/_high)` | OK by construction — uses `pd.date_range(start=run_time, periods=12, freq='5min')` which is already interval-start |
| `sensor.ai_price_forecast(_low/_high)` (incumbent APF/LightGBM) | Likely OK — index is created from internal forecast dates rather than AEMO source. Not separately audited; no chart-visible offset reported. |
| `sensor.ai_tft_price_forecast(_low/_high)` (TFT shadow) | Likely affected by the same training-side alignment as PD-direct (see C below). Publish-layer fix would mirror PD-direct's. TFT is on a sunset clock so deferred. |

### C. Deeper internal alignment issue (training data path)

`data/build_training_dataset.py` line 428:

```python
target_actual = actuals.loc[run_t + dt30 : run_t + output_length * dt30, "rrp"]
                .reindex(dec_intervals)
```

Joins forecasts at `interval_dt` (interval-end per AEMO source) against
`actuals.rrp` at `time` (interval-start because the CQ uses default
`GROUP BY time(30m)` bucketing). For run_t=12:00, decoder step 0:

- Decoder feature for step 0: PREDISPATCH at `interval_dt=12:30` →
  forecast for AEMO interval `[12:00, 12:30)`.
- Target for step 0: `actuals.loc[12:30, "rrp"]` → actual price for
  CQ-bucket `[12:30, 13:00)` (interval-start).

**These refer to different half-hours.** The trained TFT and any model
descendants (residual bands, PD7Day debiaser, active15) have learned a
30-min-shifted forecast→target relationship.

This was already flagged in `docs/timestamp_convention_audit_2026-05-11.md`
under "Deeper Internal Alignment Finding". The implementer's "Do Not Silently
Patch Yet" guidance still applies — needs a controlled dataset rebuild +
small side-by-side eval before any retrain.

`eval/rolling_mpc_eval.py` likely has the same alignment when joining
forecasts against actuals, biasing all dispatch-eval results in the same
direction. Not separately audited.

### D. Ingest pipeline (root of the interval convention asymmetry)

Verified ingest scripts are self-consistent:

- **AEMO forecast ingest** (`ingest-predispatch.py`, `ingest-pd7day.py`,
  `ingest-p5min.py`, `ingest-sevendayoutlook.py`): all parse AEMO's
  PERIODID / interval_datetime field (AEST naive), localize to NEM_TZ,
  convert to UTC, write `time=interval_utc`. The Influx `time` field
  for these measurements is therefore **interval-end** (AEMO native).
- **Dispatch actuals ingest** (`ingest-nem-data.py`): same pattern;
  `rp_5m.aemo_dispatch_sa1_5m.time` is interval-end at ingest time, but
  downstream CQ aggregation (`cq_aemo_5m_sa1_to_30m`, etc.) buckets by
  `GROUP BY time(30m)` which uses Influx's default **interval-start**
  labelling, producing `rp_30m.aemo_dispatch_sa1_30m.time` =
  **interval-start**.

So the asymmetry is structural: forecasts pass through ingest with
interval-end labels preserved, while actuals get re-labelled to
interval-start by the CQs. Both `data/export_parquet.py` and HA-facing
readers inherit those conventions.

This is not a "bug" per se — it's the consequence of mixing source-faithful
ingest (interval-end) with CQ aggregation defaults (interval-start). But it
is the root cause of every "30-min misalignment" symptom seen in this
session.

### E. Pre-existing bugs flagged for follow-up (not fixed in this audit)

1. **Spike-classifier dtype error** (forecast.py `_apply_pd_debiaser` ~L1537):
   `Cannot compare dtypes int64 and datetime64[us, UTC]` on every
   live run. The classifier silently falls back to `prob_spike=0.0`
   (maximum debiasing), so output is still sensible, but the adaptive
   per-step compression based on spike probability isn't actually firing.
   Probably caused by `historical_df` arriving with a non-DatetimeIndex
   in some code path. Worth a dedicated fix.
2. **TFT shadow publish timestamps**: same likely interval-end vs
   interval-start mismatch on publish as PD-direct had. Deferred because
   TFT is on a sunset clock (review 2026-06-05).

### F. Areas not audited in detail this round

- **Eval-side queries** (`holistic_eval.py`, `retro_tft_inference.py`,
  `retro_tier1_inference.py`, `build_holistic_eval_set.py`,
  `export_holistic_actuals.py`) all use `GROUP BY time(30m)` on
  CQ-aggregated measurements (not run_time-tagged), so no
  stale-run-mixing risk. But they may inherit the C alignment issue.
- **Tariff lookups** — use local AEST/ACST half-hour-of-day keys; the
  conversion path was implementer-checked in `apply_tariffs_to_forecast`.
  Not re-audited here.
- **`_get_aemo_short_term_forecast`** — already fixed by the
  implementer (now treats the AEMO 30MIN viz API rows as interval-ending
  and subtracts 30 min consistently).
- **HA template stitching** — the `sensor.ai_spot_price_forecast`
  template just merges underlying entity timestamps verbatim with no
  interpretation, so it inherits whatever convention the underlying
  entities publish.

## Recommended next moves

In priority order:

1. **Spike-classifier dtype fix** (~30 min) — see E.1. Easy win once the
   exact failure point is traced.
2. **TFT shadow publish-layer shift** if it stays alive past the
   2026-06-05 sunset — mirror the PD-direct fix.
3. **Dataset rebuild + side-by-side eval** for the C alignment issue.
   This is the substantive piece. Build a clearly named interval-start
   dataset variant (without mutating raw parquet), run a small
   pd_direct strategic eval to size the effect, then decide whether to
   rebuild and retrain the TFT/debiaser/residual-band stack.
4. **Eval framework alignment audit** — confirm whether
   `rolling_mpc_eval.py`'s actuals-vs-forecast joins are affected by
   the same asymmetry. If yes, all historical eval results have a
   known direction of bias.
