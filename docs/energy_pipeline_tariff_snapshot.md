# Resident tariff snapshot

Status: implemented/tested in calculation-only shadow; valid-profile behaviour preserved.
2026-10-02 Adelaide. Parent: [price acceptance](energy_pipeline_price_acceptance.md).

- `energy_pipeline/tariffs.py`: one binary file read, fstat before/after; reject observed in-place
  changes. SHA256 comes from the SAME bytes parsed. Invalid JSON/nonfinite values fail generation.
- Frozen tuples contain import/export schedules, network loss and Amber scaling. Accessors return
  owned map copies. Missing-file defaults preserve incumbent scalar defaults; tariff application
  still skips GST/loss/adders when the file is absent, matching existing missing-file behaviour.
- `forecast.prediction_resources(..., tariff_snapshot=...)` installs worker-local ContextVar;
  nested/error exits restore context, independent source threads cannot inherit a price profile.
- Amber scaling/network getters, `load_tariff_profile`, tariff application and tactical tariff
  features use the snapshot inside this context. Outside it, incumbent paths read as before.
- Price worker no longer writes shared tariff globals. Completion retains the immutable snapshot
  for later accepted-bundle/publication work; lineage includes effective profile, scaling and bytes.
- Acceptance captures current tariff file once too; persistent changes reject the old result and
  coalesce reconciliation. An external change-and-revert cannot mix tariffs inside generation.
  This is not an atomic lock spanning every remote input or future publication.
- Tests: file replacement/deletion during context; map-copy isolation; nested/thread/error
  restoration; malformed/nonfinite/mid-read writes; generation uses old scaling/loss throughout
  a live-file change and is then rejected; exact positive/negative tariff-conversion parity.
- 126 focused tests pass, including incumbent tariff/forecast/listener/model regressions.
- Live two-run `--verify-reload` probe, exit 0: exact cached/reloaded p30/p50/p70 parity; 144 points
  per quantile, three cache model loads. Generation 9.7s / 4.2s (includes reload comparison), RSS
  1306 /1334 MiB. Acceptance 0.079 /0.084s. No publication or production unit/config change.

Next: durable accepted bundle and publication transaction; separate missing-source freshness work.
