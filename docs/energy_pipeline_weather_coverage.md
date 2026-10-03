# Resident weather horizon admission

2026-10-03 Adelaide. Calculation-only shadow; production forecasting/control unchanged.

## Confirmed mismatch

- Live HA BOM capture at 09:31 Adelaide: 145 interpolated half-hour points from
  `2026-10-02T23:00:00Z` through `2026-10-05T23:00:00Z` (72 hours inclusive).
  Current price targets: `2026-10-03T00:00:00Z` through `2026-10-05T23:30:00Z`.
  Exactly one absent trailing target, three weather feature NaNs on reindex. No internal hole.
- Previous 19:33 attempt returned 145 points but failed coverage; exact missing intervals were
  not captured then. Current probe confirms a normal hourly-anchor mismatch, not a proven outage.
- Incumbent `forecast._prepare_prediction_covariates` concatenates weather/AEMO/Solcast, applies
  time-dependent covariate adjustments, THEN ffill/bfill. Longer source horizons carry weather
  tail from final adjusted sample. Strict shadow raw-source admission previously prevented this path.

## Corrected shadow contract

- `SourcePolicy.max_tail_gap_seconds`: default 0; only price weather policy sets 3600.
  Allow absent trailing targets up to one hour after final raw timestamp, at most two half-hour slots.
  This bounded exception preserves incumbent fill; it is not proof of provider freshness.
- No raw frame extension, new values, adjustments or timestamps. Do not fill before adjustment:
  it would apply different time-slot biases and change forecast inputs.
- Leading gaps, internal holes, existing nonfinite final rows and >1h tail remain rejected.
  Cache snapshot rechecks cap as targets roll forward. AEMO/Solcast full-window checks unchanged;
  source acquisition-age budgets unchanged. Prepared model covariates still require complete finite
  coverage over all 144 targets before inference.
- Eight new checks: 30/60-minute tails unchanged, strict AEMO isolation, leading/internal/invalid/long
  gaps rejected, interval-rollover rejection, exact incumbent adjusted-tail preparation.
  Focused regression suite: 159 passed; existing Influx deprecation warning only.

## Live verification

- 09:34 Adelaide: `--runs 1 --verify-reload --state-file /tmp/resident-recovery-20261003.json`;
  accepted 144 points per quantile, exact cached/reloaded p30/p50/p70 parity. Three cached model loads,
  generation 12.832s including reference reload, RSS 1322.7 MiB, acceptance 0.093s. Exit 0.
- 09:41 restart with same state path: prior run `20261003T000408.580607Z` recovered with
  `time_current=False`, explicitly historical; fresh capture/prediction accepted and checkpoint replaced.
  New run `20261003T001146.517852Z`, 144 points per quantile, three loads, generation 6.880s,
  RSS 1119.9 MiB. Exit 0. Both shadows stopped; no publications/production health writes/HA reload.
- These are short startup/restart checks, not sustained memory or live fault/interval/DST proof.
  BOM metadata patch still unapplied; provider issue/fetch clocks unknown.

Next: publication transaction/idempotency/partial-write recovery in shadow; sustained operation and
provider freshness validation remain production cutover gates.
