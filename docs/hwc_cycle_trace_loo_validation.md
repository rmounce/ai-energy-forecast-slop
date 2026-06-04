# HWC Cycle Trace Leave-One-Out Validation

Held-out raw-trace validation against clean compressor-only cycles from
`data/hwc_cop_cycles.csv`.

For each row, the stratified probe-shape parameters are fitted from the other
clean cycles, then replayed against the held-out cycle. The stratified trace still
uses observed per-cycle thermal input, so this validates probe-shape transfer, not
forecast-ready heat input.

## Aggregate Held-Out Trace Error

| metric | single_node | stratified_loo |
| --- | --- | --- |
| mean trace MAE C | `1.331` | `0.832` |
| mean max abs error C | `2.758` | `1.94` |
| mean end error C | `-2.512` | `0.468` |
| mean +0.5C flat-lag error min | `-15.714` | `8.429` |

## Cycle Detail

| start | cycle_class | train_cycles | loo_probe_height_fraction | loo_thermocline_width_fraction | points | duration_min | observed_start | observed_end | single_node_mae_c | single_node_max_err_c | single_node_end_err_c | single_node_flat_lag_err_min | stratified_mae_c | stratified_max_err_c | stratified_end_err_c | stratified_flat_lag_err_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 20:35 | full_reheat | 6 | 0.62 | 0.59 | 76 | 81.0 | 48.09 | 59.95 | 2.609 | 5.516 | -5.516 | -2.0 | 1.312 | 3.356 | 0.434 | 19.0 |
| 2026-05-29 10:01 | top_up | 6 | 0.62 | 0.59 | 39 | 53.0 | 56.13 | 59.96 | 0.27 | 0.57 | -0.235 | -8.0 | 0.462 | 0.909 | 0.215 | 6.0 |
| 2026-05-31 10:54 | full_reheat | 6 | 0.62 | 0.64 | 135 | 134.0 | 45.37 | 59.83 | 1.918 | 3.774 | -3.774 | -27.0 | 0.903 | 2.168 | 0.736 | 8.0 |
| 2026-06-01 11:31 | full_reheat | 6 | 0.62 | 0.64 | 116 | 118.0 | 47.02 | 59.93 | 1.491 | 3.364 | -3.364 | -15.0 | 0.892 | 1.868 | 0.646 | 15.0 |
| 2026-06-02 10:07 | top_up | 6 | 0.62 | 0.6 | 51 | 63.0 | 54.11 | 60.0 | 0.568 | 1.066 | -1.04 | -8.0 | 0.454 | 1.169 | 0.29 | 8.0 |
| 2026-06-03 10:24 | full_reheat | 6 | 0.62 | 0.64 | 126 | 125.0 | 47.85 | 60.0 | 1.86 | 3.589 | -2.23 | -43.0 | 1.136 | 2.879 | 0.58 | -10.0 |
| 2026-06-04 10:01 | full_reheat | 6 | 0.62 | 0.59 | 62 | 74.0 | 53.06 | 59.94 | 0.603 | 1.424 | -1.424 | -7.0 | 0.667 | 1.229 | 0.376 | 13.0 |
