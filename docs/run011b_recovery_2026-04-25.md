# Run 011b Snapshot Recovery Note — 2026-04-25

Purpose: record the provenance and validation status of the recovered Run 011b-era TFT price
artifact pair now copied into `models/tft_price/`.

Recovered local copies:
- [models/tft_price/checkpoint_run011b_snapshot_candidate.pt](../models/tft_price/checkpoint_run011b_snapshot_candidate.pt)
- [models/tft_price/scalers_run011b_snapshot_candidate.pkl](../models/tft_price/scalers_run011b_snapshot_candidate.pkl)

Source snapshot:
- `/home/saltspork/.zfs/snapshot/autosnap_2026-04-17_00:00:10_daily/src/ai-energy-forecast-slop/models/tft_price/checkpoint_best.pt`
- `/home/saltspork/.zfs/snapshot/autosnap_2026-04-17_00:00:10_daily/src/ai-energy-forecast-slop/models/tft_price/scalers.pkl`

Why this appears to be the correct Run 011b-era asset:
- checkpoint metadata matches the documented Run 011b note in [docs/training_runs.md](./training_runs.md):
  - epoch `5`
  - val loss `0.053756...`
  - quantiles `[0.05, 0.10, 0.50, 0.90, 0.95, 0.99]`
  - 15-feature decoder contract with `covar_missing`
- snapshot `evaluation_results.csv` matches the documented Run 011b horizon table
- the current rolling eval harness can load the snapshot paths directly and complete a smoke run

Checksums of the copied local files:
- checkpoint: `fb976ec4a0f9ff6224811d235431b27fcaa5c402196c17fd8f712a7f13056033`
- scalers: `4458deabc72e80b9fd0cbaa59f7d06dde577e8f9c4b0bd566a9831fc75b978a8`

Caveat:
- this pair is still best described as a **recovered Run 011b candidate** rather than a
  formally promoted incumbent artifact
- it is suitable for rolling-eval diagnostic and counterfactual work
- broader architecture conclusions should still wait for fuller incumbent-backed validation
