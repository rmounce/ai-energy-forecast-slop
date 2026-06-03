# HWC Stratified Model Validation

Offline validation against clean compressor-only cycles in `data/hwc_cop_cycles.csv`.

This report uses cycle-level probe-rise milestones, not raw HA temperature curves.
The stratified model is therefore a shape diagnostic only; it is not yet a live-control model.
The stratified replay uses observed per-cycle thermal input, so its end-temperature error is
not a fair forecast benchmark; the milestone timing errors are the more useful signal.

## Parameters

| parameter | value |
| --- | --- |
| `probe_height_fraction` | `0.62` |
| `thermocline_width_fraction` | `0.59` |
| `hot_target_c` | `60.0` |
| `standing_loss_ua_kw_per_c` | `0.0025` |

## Aggregate Error

| metric | single_node | stratified |
| --- | --- | --- |
| end temp C | `2.71` | `0.46` |
| rise 10% min | MAE `20.0`; reached `6/6`; misses `0` | MAE `7.67`; reached `6/6`; misses `0` |
| rise 50% min | MAE `8.83`; reached `6/6`; misses `0` | MAE `4.5`; reached `6/6`; misses `0` |
| rise 90% min | MAE `3.0`; reached `1/6`; misses `5` | MAE `8.08`; reached `6/6`; misses `0` |

## Cycle Detail

| start | cycle_class | dur_min | tank_start | tank_end_obs | block_end | block_end_err | strat_end | strat_end_err | rise_10_obs | block_rise_10 | block_rise_10_err | strat_rise_10 | strat_rise_10_err | rise_50_obs | block_rise_50 | block_rise_50_err | strat_rise_50 | strat_rise_50_err | rise_90_obs | block_rise_90 | block_rise_90_err | strat_rise_90 | strat_rise_90_err |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 20:35 | full_reheat | 81 | 47.9 | 59.9 | 54.43 | -5.47 | 60.38 | 0.48 | 13.0 | 15.0 | 2.0 | 31.0 | 18.0 | 46.0 | 75.0 | 29.0 | 49.0 | 3.0 | 71.5 | nan | nan | 67.0 | -4.5 |
| 2026-05-29 10:01 | top_up | 53 | 55.5 | 60.0 | 59.73 | -0.27 | 60.18 | 0.18 | 13.5 | 6.0 | -7.5 | 20.0 | 6.5 | 33.0 | 29.0 | -4.0 | 32.0 | -1.0 | 48.0 | 51.0 | 3.0 | 44.0 | -4.0 |
| 2026-05-31 10:54 | full_reheat | 134 | 45.2 | 59.9 | 56.06 | -3.84 | 60.57 | 0.67 | 52.0 | 18.0 | -34.0 | 50.0 | -2.0 | 84.0 | 91.0 | 7.0 | 80.0 | -4.0 | 122.0 | nan | nan | 110.0 | -12.0 |
| 2026-06-01 11:31 | full_reheat | 118 | 47.0 | 60.0 | 56.57 | -3.43 | 60.58 | 0.58 | 43.0 | 16.0 | -27.0 | 44.0 | 1.0 | 78.0 | 80.0 | 2.0 | 71.0 | -7.0 | 108.5 | nan | nan | 97.0 | -11.5 |
| 2026-06-02 10:07 | top_up | 63 | 53.9 | 60.0 | 58.96 | -1.04 | 60.29 | 0.29 | 16.0 | 8.0 | -8.0 | 24.0 | 8.0 | 36.0 | 38.0 | 2.0 | 38.0 | 2.0 | 55.5 | nan | nan | 52.0 | -3.5 |
| 2026-06-03 10:24 | full_reheat | 125 | 47.7 | 60.0 | 57.77 | -2.23 | 60.58 | 0.58 | 57.5 | 16.0 | -41.5 | 47.0 | -10.5 | 85.0 | 76.0 | -9.0 | 75.0 | -10.0 | 115.0 | nan | nan | 102.0 | -13.0 |
