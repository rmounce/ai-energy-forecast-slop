# APF-Free Price Forecast Review — 2026-07-02

Fresh holistic opinion on the suspended price-forecasting / debiasing work:
is an Amber-APF-free price source viable, what open data would help, and
what model construction suits CPU-only training and inference.

Written after reviewing `docs/roadmap.md` (strategic pivot + Phase α-prime),
`docs/price/tft_price_forecast.md` (Structural Critique 2026-05-05),
`docs/price/pd_direct_debiaser_audit_2026-05-13.md`,
`docs/price/lgbm_residual_driver_audit_2026-06-14.md`,
`docs/price/aemo_renewable_availability_discovery_2026-06-14.md`,
`docs/price/price_forecast_sources.md`, and the current logs/model artifacts.

## Caveman Summary

- Viable: yes — PD-direct + residual bands + tightatten already beat Amber
  APF on all four eval windows in May (Step 4 tightatten: 4/4 PnL wins).
- The work stalled on trust and eval noise, not on forecast accuracy.
- Economic stakes are small (~$0.1–0.5/day between decent sources); the real
  value is retailer portability, resilience, and debuggability.
- Do not revive the TFT or any deep net. Stay LightGBM + deterministic layers.
- Structural fix: predict the *residual* (`actual − raw_PD`), never absolute
  price, with automated rolling retrains so nothing goes stale again.
- Highest-value new open data: PDPASA REGIONSOLUTION (renewables at 0–29h),
  PREDISPATCH interconnector/constraint solutions, PREDISPATCH price
  sensitivities, AEMO rooftop-PV actual/forecast.
- Make forecast-quality metrics the iteration gate; run dispatch PnL less
  often with bootstrap error bars — several past kill decisions were likely
  made on noise.
- Cheapest first step: score the ~5 weeks of live PD-direct shadow logs
  (2026-05-08 → ~2026-06-14) already sitting in `pd_direct_forecast_log.csv`
  via `eval/compare_shadow_forecasts.py`, and find out why that log stopped
  mid-June.

---

## 1. Is APF-free forecasting viable and worthwhile?

**Technically viable: already demonstrated.** The Phase α-prime Step 4
"tightatten" config (PD-direct + horizon×HoD×level residual bands + terminal
salvage + low-volatility attenuation) beat Amber on every window of the
same-run matrix: shoulder3 +1.2%, WB2 +1.9%, WB7 +25%, WA7 +17%. The
corrected tactical-boundary comparison in Step 6 still won 3 of 4 windows.
Amber's APF is itself largely AEMO PREDISPATCH plus proprietary short-horizon
dressing — the repo already ingests the same raw signal, and the Tier 1
tactical LGBM (which passed both the accuracy and dispatch gates in April)
covers the 0–60 min zone where Amber's proprietary edge actually lives.

**Economically honest framing:** the PnL deltas between decent sources were
~$0.1–0.5/day in the eval matrix, i.e. roughly $50–150/year. If the goal were
purely money, the `amber_apf_lgbm` incumbent is fine. The case for the work is
retailer portability (freedom to leave Amber, or survive their API/product
changes), resilience to Amber outages, and the fact that the APF is a black
box that cannot be debugged when it misbehaves. Those were the original
charter goals and they still hold.

**What actually killed the work** — worth naming, because none of it is
"the forecast can't be good enough":

1. **The TFT was structurally wrong.** The 2026-05-05 critique stands and
   should not be relitigated. A model whose decoder covariate is a forecast of
   its own target is learning AEMO's forecast-error distribution — a low-SNR
   problem TFT is mismatched for. It regressed to the encoder median across
   four independent training runs (011b, 014, 015, active15). Correctly
   abandoned.
2. **The trained PD debiaser went stale.** The 2026-05-13 audit found raw
   PREDISPATCH beating debiased PD ($26.86 vs $29.38 MAE) after the alignment
   fix. A frozen LGBM trained on an old regime is a liability. This is a
   process failure (no scheduled retrain), not a modelling failure.
3. **The dispatch eval is noisier than the effect sizes.** The PD7Day-covariate
   rejection (MAE −10% at 0–6h, PnL −3.4% over 55 days) has the signature of a
   decision made on noise. WA7 SoC depletion turned out to be
   source-insensitive under the eval objective (Step 5b). Iterating against a
   noisy gate is what made the project feel like looping.
4. **PD7Day training data was thin** (~2–3 months when the tail debiaser was
   built). There are now ~5 months and it grows passively.

**Verdict: viable and worthwhile if Amber-independence is a goal, which it
is.** The revival should be a lean consolidation of what already worked, not
a new architecture hunt.

## 2. Open data sources worth adding

The 2026-06-14 residual audit already identified where the money is. Top
drivers of incumbent error by bias spread: SDO net-interchange forecast error
($57.62/MWh), actual interchange regime ($53.34), PV forecast error ($47.41),
market-wide wind (local BOM wind explicitly *not* a substitute). Demand error
was weak ($10.99). Accordingly:

| Source | What / why | Cost to add | Expected value |
|---|---|---|---|
| **PDPASA REGIONSOLUTION** | UIGF / SS_WIND_UIGF / SS_SOLAR_UIGF at 0–29h — the STPASA fields (proven: tail MAE $31.16→$20.97 in ablation; 23.3% importance in retrain) but at the PREDISPATCH horizon where most dispatch value lives. Path already found in the discovery doc. | Low — mirror the STPASA ingest | **Highest** |
| **PREDISPATCH interconnector/constraint solution** (`PREDISPATCHINTERCONNECTORRES`) | Forecast Heywood/Murraylink flows and binding limits. Interchange error was the #1 residual driver; currently only SDO's coarse version is visible. | Very low — it's in files already downloaded | High |
| **PREDISPATCH price sensitivities** (`PREDISPATCH_PRICESENSITIVITIES`) | SA1 price under demand offsets — AEMO publishing the local slope of the supply curve. A direct "how fragile is this PD price" spike-risk feature. | Low | High for spike/band work |
| **ROOFTOP_PV_ACTUAL / ROOFTOP_PV_FORECAST** (NEMweb) | SA operational demand is dominated by distributed PV; targets the solar-hours under-forecast bias (−$13.95/MWh). Solcast is site-level, not state-level. | Low | Medium |
| **Open-Meteo gridded wind** at Mid-North SA wind-farm coordinates | Weather-model-based market wind proxy, independent of AEMO UIGF, all horizons. Free API. | Medium | Optional — only if PDPASA/STPASA UIGF prove insufficient |

**Skip:** bid-stack data (BIDPEROFFER is open but ~1 GB/month for a feature
price sensitivities approximate), MTPASA (wrong horizon), gas-market data
(slow regime covariate, low value at 72h), anything proprietary.

## 3. Model construction (CPU-only)

**Stay with LightGBM + deterministic layers; deep nets stay dead.** The
literature edge of transformers here came almost entirely from PREDISPATCH
being an input (Sinclair et al.'s own SHAP: >60% of importance), and trees
exploit that input equally well, train in minutes on CPU, and retrain cheaply
on a schedule — which is the real requirement, because failure #2 above was
staleness and the fix for staleness is frequent automated retraining. That is
exactly what CPU-friendly models buy and TFT doesn't. N-BEATS / N-HiTS /
PatchTST on CPU would re-fight the same structural battle, slower.

### Proposed "PD-direct v2" stack

1. **Tier 1 (0–60 min):** existing tactical LGBM unchanged — it passed both
   gates.
2. **1–28h:** publish `raw_PD + predicted_residual`, never absolute price.
   Quantile LGBMs (q20/q50/q80, pinball loss) on `actual − raw_PD` with
   features: horizon, HoD, PD level and local shape, PDPASA renewables,
   interconnector flow/limit headroom, price sensitivities, and **recent
   realised PD error** (last 2–6h) as an online feedback feature. The residual
   framing is the structural fix for both "model over-discounts its own input"
   and "debiaser goes stale": the default output when the model sees nothing
   is raw PD, not a compressed median. Shrink the residual toward zero when
   the feature regime is out-of-distribution. (This is the roadmap's Phase β
   design, minus the TFT.)
3. **28–72h tail:** PD7Day de-capped via the cap-materialisation stats (only
   0.5% of ≥$300 flags materialise ≥$300 — a soft cap-replacement policy is
   nearly free), the PD7Day q50 debiaser refreshed on the now-~5-months of
   data and retrained monthly, STPASA renewables, HoD seasonal blend. Per the
   Step 4 finding, this zone only needs sane inventory posture, not
   minute-perfect shape.
4. **Bands:** keep the empirical residual bands with Step 4b low-volatility
   attenuation — they were the thing that actually beat Amber. Recompute on a
   rolling window (monthly cron) instead of frozen artifacts.
5. **Retraining discipline:** rolling-origin monthly retrains, time-based
   holdout, automated under `nice -n 19`. The 2026-05-13 failure mode (frozen
   model, shifted regime) should become structurally impossible.

### Evaluation changes (as important as the model)

- Forecast-quality metrics (stratified pinball loss, MAE by horizon/regime,
  Step 5a shape diagnostics) become the *iteration* gate.
- Dispatch PnL runs less often, with paired-bootstrap-over-days error bars, so
  a −3.4% over 55 days can be recognised as indistinguishable from zero.
- The 2026-05-09 three-gate contract (shape / economics / production safety)
  stays; the missing piece was error bars on gate 2.

## 4. Recommended first steps (cheap → committed)

1. **Score the accumulated live shadow data.** `pd_direct_forecast_log.csv`
   holds ~5 weeks of live PD-direct publishes (2026-05-08 → ~2026-06-14) and
   `eval/compare_shadow_forecasts.py` already exists to score them against the
   incumbent. An afternoon's work that measures the real gap before any build
   commitment. Also: find out why the log stopped mid-June (last row created
   2026-06-14; possibly a casualty of predispatch-service changes).
2. **Ingest PDPASA + interconnector + price sensitivities** and rerun the
   residual-driver decomposition with them joined.
3. **Replace the frozen PD debiaser with the residual-LGBM design above**,
   only if step 1/2 evidence still supports the build.
4. **Refresh the PD7Day tail debiaser and residual bands** on current data and
   put both on a monthly retrain timer.

Nothing here requires new architecture research, GPU time, or proprietary
data.
