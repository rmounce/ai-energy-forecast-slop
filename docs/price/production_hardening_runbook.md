# Production forecast hardening runbook

The weekly timer creates isolated candidate bundles. It never promotes them.
Prediction resolves one active price/load bundle at the start of each run.

Run commands from the repository root with the project virtual environment active:

```bash
source .venv/bin/activate
```

Without this step, `./forecast.py` may select the system Python and fail on project dependencies.

## One-time migration

Run from the repository root after reviewing the root artifacts:

```bash
./forecast.py migrate-load-bundle
./forecast.py migrate-price-bundle
```

Migration refuses missing artifacts and existing pointer state. It copies artifacts into
`models/production/<family>/bundles/<bundle_id>/` and atomically creates `active.json`.
It is not run by deployment automation.

## Inspect, promote, rollback

```bash
./forecast.py validate-bundle --family price --bundle <id>
./forecast.py screen-bundle --family price --bundle <id> --metrics screening.json
./forecast.py promote-bundle --family price --bundle <id>
./forecast.py rollback-bundle --family price
```

Promotion requires a complete manifest, matching SHA-256 hashes, and a report containing
`eligible_for_manual_promotion: true`, plus current quantile/feature contract and smoke checks.
The built-in candidate report is conservatively ineligible until screening metrics and a valid
144-point inference smoke are recorded. Rollback selects the previous complete pointer.

## Diagnose a failed refresh

- A missing/stale/misaligned Amber APF or incomplete quantile family exits nonzero.
- `predictions.json` and forecast logs are not changed until generation validation passes.
- HA publication validates the complete family before its first POST. A later POST failure
  reports potentially changed entity IDs; HA publication is not transactional.
- The price listener pings its healthcheck only after exit 0, retries failures every five minutes,
  and coalesces APF events received during a run.

The screening report records that historical training uses realised PV/weather/demand and selects
STPASA differently from live inference. Metrics are screening evidence, not causal promotion proof.
`screen-bundle` runs the independent 144-point smoke first, verifies a SHA-256-identified row file,
derives every metric on identical candidate/incumbent rows, then atomically switches the manifest
to a versioned report. Do not edit bundle JSON by hand.

The descriptor JSON contains `schema_version: 1`, `family`, `units`, `row_count`, and provenance
fields `command`, `rows_file`, and `rows_sha256`. `rows_file` is resolved relative to the descriptor
and may be CSV or Parquet. Required row columns are:

```text
forecast_issue_time, forecast_target_time, actual
price: candidate_p30/p50/p70, incumbent_p30/p50/p70
load:  candidate_p50/p65/p75, incumbent_p50/p65/p75
```

Rows must be unique by issue/target time, finite, half-hour aligned, and within 0–72h. Price uses
fixed 0–16.5h, 16.5–28h, 28–48h, and 48–72h buckets; load uses 0–24h, 24–48h, and 48–72h. Every
bucket must contain rows. The report derives MAE, bias, pinball loss, empirical coverage, incumbent
regressions, and evaluation ranges. Eligibility requires regression ≤ 0.05, price absolute-bias
worsening ≤ 10 $/MWh, and load p65 coverage 0.55–0.85. Missing, non-finite, hash-mismatched, or
wrong-unit evidence fails closed.

Candidate creation also records artifact sizes, serialized training-series ranges, and finite
coverage for every stored future covariate (including STPASA price features), alongside smoke time.
