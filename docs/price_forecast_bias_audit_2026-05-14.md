# Price Forecast Bias Audit — 2026-05-14

Companion to `docs/pd_direct_debiaser_audit_2026-05-13.md`. Run B v3's negative
absolute PnL under `--strategic-soc-handoff exact` raised the question: is
the strategic LP being misled by a biased price-forecast curve, and if so,
*which* forecast? The PD-direct debiaser audit characterised one Tier 2
source, but in Run B the active sources are `amber_apf_lgbm` (strategic curve
= LGBM extrapolated) and `model_a_hybrid` (strategic curve = TFT) —
neither uses PD-direct in `build_strategic_curve`. So this audit scores
LGBM and TFT against realised RRP directly.

## Method

`eval/audit_price_forecast.py` joins each forecast log row with realised RRP:

- LGBM: `price_forecast_log.csv` (model_name=`price`). 83% of recent rows
  have `actual` backfilled via `forecast.py backfill-actuals`; remainder are
  filled from `actuals_sa1.parquet`.
- TFT:  `tft_price_forecast_log.csv` (model_name=`tft_price`). Zero rows
  have backfilled `actual`; all values come from the parquet join.

Both logs store predictions in $/kWh; multiplied by 1000 for comparison
with the parquet's $/MWh `rrp`. `forecast_target_time` is interval-END,
so actuals are looked up at `target_time − 30min`.

Window: forecasts created since `2026-04-01T00:00Z`, matching Run B v3 /
C v3. Horizon clipped to 0–72h.

Sample sizes:
- LGBM: n=254,589 (target window `2026-04-01 → 2026-05-13`)
- TFT:  n=146,781 (target window `2026-04-12 → 2026-05-12`; the TFT log
  doesn't go back as far as LGBM)

## LGBM (model='price') — the `amber_apf_lgbm` strategic source

| bucket | n | actual | pred | MAE | **bias** | p90 overshoot |
|---|---:|---:|---:|---:|---:|---:|
| **OVERALL** | 254589 | 57.30 | 60.04 | 36.55 | **+2.74** | 62.20 |
| overnight | 60803 | 58.52 | 73.24 | 37.56 | **+14.73** | 79.58 |
| morning | 62876 | 43.09 | 37.00 | 35.12 | −6.10 | 47.58 |
| solar | 56057 | 43.10 | 29.31 | 34.67 | −13.80 | 35.88 |
| evening | 43608 | 86.15 | 98.12 | 37.36 | **+11.97** | 66.37 |
| late | 31245 | 68.73 | 82.69 | 39.74 | **+13.96** | 77.22 |
| h=0-1h | 3628 | 56.47 | 50.28 | 13.98 | −6.20 | 12.85 |
| h=1-4h | 10855 | 56.55 | 49.14 | 17.92 | −7.41 | 18.54 |
| h=4-12h | 28787 | 56.74 | 52.04 | 23.48 | −4.70 | 19.57 |
| h=12-24h | 43292 | 56.91 | 53.77 | 25.69 | −3.13 | 26.93 |
| h=24h+ | 168027 | 57.57 | 63.94 | 43.28 | **+6.37** | 74.24 |

Overall bias is nearly zero (+$2.74/MWh) but that hides a sharp regime
split:

- **Over-forecasts** the high-price periods: overnight +$14.73, evening
  +$11.97, late +$13.96.
- **Under-forecasts** the low-price periods: morning −$6.10, solar −$13.80.
- **Over-forecasts at long horizons** (24h+ bias +$6.37) — the regime the
  strategic 72h LP cares about.
- p90 overshoot at overnight/late is $77-$80/MWh: 10% of forecasts in
  those buckets are over by $80+/MWh, exactly the tail that makes the
  strategic LP think a spike is imminent.

## TFT (model='tft_price') — the `model_a_hybrid` strategic source

| bucket | n | actual | pred | MAE | **bias** | p90 overshoot |
|---|---:|---:|---:|---:|---:|---:|
| **OVERALL** | 146781 | 56.98 | 52.22 | 40.07 | **−4.76** | 69.06 |
| overnight | 36420 | 53.40 | 47.91 | 37.09 | −5.48 | 61.62 |
| morning | 35839 | 49.97 | 42.39 | 44.80 | −7.58 | 78.51 |
| solar | 32627 | 42.04 | 35.18 | 37.09 | −6.87 | 64.60 |
| evening | 24207 | 88.37 | 88.78 | 42.36 | **+0.41** | 73.81 |
| late | 17688 | 63.17 | 62.43 | 38.98 | **−0.74** | 65.58 |
| h=0-1h | 2135 | 57.29 | 28.84 | 35.96 | **−28.46** | 11.40 |
| h=1-4h | 6385 | 57.91 | 35.74 | 36.14 | −22.17 | 29.34 |
| h=4-12h | 16848 | 58.34 | 43.13 | 38.27 | −15.22 | 48.39 |
| h=12-24h | 25071 | 59.40 | 49.37 | 38.25 | −10.03 | 59.05 |
| h=24h+ | 96342 | 56.05 | 56.16 | 41.21 | **+0.12** | 74.90 |

Strikingly different shape from LGBM:

- **Under-forecasts at short horizons** (0-1h bias −$28.46, 1-4h bias
  −$22.17). Largely the "double compression" problem documented in
  `docs/tft_price_forecast.md`.
- **Essentially unbiased at long horizons** (24h+ bias +$0.12).
- Evening and late biases near zero (+$0.41, −$0.74) — TFT does *not*
  systematically over-forecast evening peaks the way LGBM does.

## Verdict on the Run B v3 Negative PnL

LGBM's bias profile lines up with the strategic over-charging pattern:

| strategic-LP-relevant signal | LGBM | TFT |
|---|---|---|
| Long-horizon (24h+) bias | **+$6.37** over | +$0.12 ≈ zero |
| Overnight bias | **+$14.73** over | −$5.48 under |
| Evening bias | **+$11.97** over | +$0.41 ≈ zero |
| p90 long-horizon overshoot | **+$74.24** | +$74.90 (similar) |

For `amber_apf_lgbm`: the strategic 72h LP sees LGBM's consistent over-forecast
on overnight/evening/late at 24h+ horizons, decides those future spikes warrant
pre-charging, races toward the battery cap. When the realised prices are
$12-15/MWh lower on average than forecast, the discharge revenue doesn't cover
the import cost. **This is the mechanism for Run B v3's −$0.18 PnL on
`amber_apf_lgbm` + LGBM-load.**

For `model_a_hybrid` with TFT strategic curve: the bias profile predicts
*less* over-confidence than LGBM because TFT doesn't over-forecast at long
horizons. But Run B v3's `model_a_hybrid` result is confounded by the
20%-skip / 100%-repair issue (`skipped_missing_curve=2331`, `repaired_invalid_curve=9474`),
so we can't cleanly compare. The repair-curve issue is itself probably a
TFT-feature alignment problem orthogonal to the bias question.

## Caveats

- Window is 41 days for LGBM, 30 days for TFT. Wholesale price regimes shift
  seasonally; biases that held over April-May 2026 may not hold in other
  months.
- `actuals_sa1.parquet` for the join was refreshed 2026-05-12T14:13Z + later
  during this session. Should be current to 2026-05-13.
- LGBM "p90 overshoot" is the 90th percentile of `pred - actual`, *not* the
  90th percentile of `|pred - actual|`. So p90=80 means 10% of forecasts
  overshoot by ≥$80/MWh — the relevant tail for "LP thinks a spike is coming".
- The Adelaide-time bucket breakdown reflects the *target* time, not the
  creation time. A forecast made at 14:00Z targeting 04:00Z next day lands in
  the "late" bucket.

## Connection to Other Audits

- `docs/pd_direct_debiaser_audit_2026-05-13.md`: PD-direct OOF debiaser
  POST-promotion bias is −$5.13 overall (slight under-forecast), so PD-direct
  *would* probably *not* drive the same over-charging pattern as LGBM if
  used as a strategic curve. But PD-direct is not currently the strategic
  source for any production-equivalent source contract in Run B/C.
- `docs/tft_price_forecast.md`: documents TFT's known short-horizon "double
  compression" pathology, which this audit reproduces (0-1h bias −$28.46).

## Possible Next Steps

a) **Calibrate LGBM** — add a horizon × Adelaide-bucket bias correction to
   the LGBM `amber_apf_lgbm` strategic curve. Cheap to test, would directly
   probe whether removing the +$14/MWh overnight over-forecast eliminates
   the strategic LP over-charging.

b) **Investigate the model_a_hybrid repair-curve issue** — separate from
   the bias question. The 20% skip + 100% repair pattern on `model_a_hybrid`
   needs explaining before its dispatch numbers are trustworthy. Likely a
   TFT decoder-feature alignment problem.

c) **Per-day case study** — pick a worst-PnL day in Run B v3 and walk the
   strategic LP's decisions step by step against the LGBM long-horizon
   curve at that moment. Would confirm the mechanism story tightly.

d) **Park it** — the load-source verdict is robust; the strategic-LP-bias
   interaction is a known structural property of LP MPC with biased
   forecasts, not specific to the current setup. Documented; no production
   action required because production uses Amber+LGBM and the same bias is
   presumably in real EMHASS-driven dispatch already.

## Files

- Audit script: `eval/audit_price_forecast.py`
- LGBM log: `price_forecast_log.csv`
- TFT log: `tft_price_forecast_log.csv`
- Realised RRP: `data/parquet/actuals_sa1.parquet`
