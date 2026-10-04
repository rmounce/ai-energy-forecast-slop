# Spaced APF causal correction — 2026-10-05

- Parallel continuation of [APF packet timing](apf_packet_timing_2026-10-05.md) and
  [load-information mechanisms](load_information_value_2026-10-05.md). APF retained; no promotion,
  production/device writes or optimizer calls. Only28 fixed read-only APF snapshot queries.
- `export_spaced_apf.py`: Sep27–Oct4 UTC,00/06/12/18 UTC daily; first feed receipt inside each
  predeclared two-minute window.28/28 captured; selection independent of realised price.
  Maximum7 complete UTC days, fixed entity/fields, one row/query; missing windows retained.
- New sparse snapshot archive supplements earlier frozen event receipts for fitting. Evaluation
  uses only predeclared28 receipts, not dense event windows. Still one week, no seasonal evidence.

## Correction contract

- `apf_causal_correction.py`: fixed0–2/2–6/6–14h revenue-bias bands. Ridge denominator adds24
  target equivalents; bias clipped±10c/kWh; need≥12 unique target/bin pairs and≥2 earlier forecast
  UTC days in each band. Unsupported bins use zero; no claimed fitted evidence during warm-up.
- Entire current UTC day's **forecast receipts** excluded from fit. Earlier-day forecasts may
  use labels completed/received earlier in current day. Both quote receipt and interval end must
  strictly precede evaluation origin; no forecast future-target labels in fitting.
- Newest fully future earlier forecast vintage per target/bin retained. Same actual target can
  appear in different bins; correlation remains explicit. Five-minute labels are not independent.
- Final canonical quote received after origin excluded even if preliminary receipt earlier;
  conservative training availability, not retrospective substitution of final labels.
- Apply differential band biases to APF point/bounds in raw feed sign; split interval at5min
  boundaries. Uniform bias alone could change withholding but not slot rank. This simple band
  design cannot repair arbitrary ordering errors inside same band.
- Packet economics unchanged:0.25kWh stock,0.2475kWh DC throughput,0.235125kWh AC output,
 4c/DC-kWh wear,0/20c stored-energy value, assumed spare2.8215kW export power. All arms/oracle
  share complete fully future slots; missing quote horizons excluded, never filled.
- Forecast slots and quote availability pre-indexed once; no repeated heavy model fitting.

## Result

- 28 origins;26 retain≥1 complete horizon.6/12/14h×0/20c terminal assumptions:146 eligible paired
  comparisons,22 excluded incomplete confirmed-quote horizons.48 warm-up comparisons (Sep27/28),
 98 fully supported comparisons from Sep29 onward. Overlap not independent/additive savings.
- **Only one selected action changed.** Oct3 06:00:23.585436 UTC,12h,terminal0:
  baseline exports12:00; correction12:05. Realised packet value1.681020c→1.909091c,
  improvement0.228071c per0.25kWh hypothetical packet; regret3.395205→2.482920c/stored-kWh.
- Every20c-terminal action unchanged. No broad economic benefit established; do not promote
  horizon-bias correction or claim weekly/system savings from this result.
- Small conditional improvement is not achievable-site headroom: no load/PV/controller,
  replenishment or available inverter/grid headroom reconstructed.
- Fit parameters fixed before new spaced evaluation; not tuned on its result. No independent
  untouched week. Broad causal sampling and more relevant features needed before a new claim.

## Verification and next effort

- 19 correction tests +5 sampler tests pass;25 existing packet tests pass. Root combined39 new
  sampler/correction/load tests pass. Tests cover label cutoffs, same-day holdout, newest-vintage
  dedup, future-label mutation, cold-start, shrinkage/cap, horizon boundaries and sampling limits.
- Root verifies report's evaluation-code hash and all50,610 recorded fit labels' forecast-day,
  quote-receipt and interval-end cutoffs. Saved training evidence is correlated, not50,610 independent
  training examples. Dependencies/source manifests recorded through existing input contracts.
- Next priority: empirical existing load-calibration comparison at constrained reserves, with
  common explicit PV scenarios. Simple APF band correction remains offline; later price experiments
  should target revision/spread/calendar structure on broader causal data, not heavier architectures
  merely to move forecast-error scores. Keep reserve value and forecast effects separate.
- Private authoritative archives/results:
  `data/energy_replay/spaced_apf_20261005_week/`,
  `data/energy_replay/apf_causal_correction_20261005_spaced/report.json`.
  Initial burst-based/slow scratch runs stopped; no completed authoritative output from them.

```bash
nice -n 19 ./.venv/bin/python eval/apf_causal_correction.py \
  --archive data/energy_replay/amber_apf_archive_20261004 \
  --quotes data/energy_replay/amber_observed_week_20261004_v6 \
  --control-history data/energy_replay/dh_source_history_20261004_near_full \
    data/energy_replay/dh_source_history_20261004_export_extended \
    data/energy_replay/constraint_history_20261005_scarce \
  --evaluation-history data/energy_replay/spaced_apf_20261005_week \
  --output data/energy_replay/NEW_CAUSAL_APF
```
