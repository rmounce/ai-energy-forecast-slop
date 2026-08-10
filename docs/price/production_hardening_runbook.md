# Production forecast hardening runbook

The weekly timer creates isolated candidate bundles. It never promotes them.
Prediction resolves one active price/load bundle at the start of each run.

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
`screen-bundle` runs the independent 144-point smoke first, then atomically replaces the report
and refreshes its manifest hash from the supplied machine-readable metrics. It is the supported
way to add screening evidence; do not edit bundle JSON by hand.

The metrics file supplies `comparable: true`, `primary_regressions` (fractional values), and
either `bias_worsening_mwh` for price or `p65_coverage` for load. The report applies the versioned
thresholds: regression ≤ 0.05, price bias worsening ≤ 10 $/MWh, and load p65 coverage 0.55–0.85.
Missing evidence remains ineligible.
