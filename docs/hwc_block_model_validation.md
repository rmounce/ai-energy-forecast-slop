# HWC Block Model Validation

Offline validation of the live block scheduler's single-node transition model against
clean compressor-only cycles in `data/hwc_cop_cycles.csv`.

This evaluates the model used by `hwc_planner.py`, not the offline stratified shape
diagnostic. It is intended to show whether block scheduling errors are coming from
duration/heat-rate assumptions, nominal electrical power, or end-temperature replay.

## Aggregate Error

| class | n | duration_mae_min | duration_bias_min | end_temp_mae_c | end_temp_bias_c | elec_mae_kwh | elec_bias_kwh |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 7 | 10.286 | 8.857 | 0.769 | -0.769 | 0.134 | 0.13 |
| full_reheat | 5 | 12.6 | 10.6 | 0.968 | -0.968 | 0.173 | 0.173 |
| top_up | 2 | 4.5 | 4.5 | 0.27 | -0.27 | 0.037 | 0.022 |

## Cycle Detail

| start | cycle_class | tank_start | tank_end_obs | wet_bulb | observed_duration_min | predicted_duration_min | duration_err_min | observed_elec_kwh | predicted_elec_kwh | elec_err_kwh | predicted_end_temp_at_observed_runtime | end_temp_err_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 20:35 | full_reheat | 47.9 | 59.9 | 12.8 | 81.0 | 120.0 | 39.0 | 1.1 | 1.56 | 0.46 | 56.35 | -3.55 |
| 2026-05-29 10:01 | top_up | 55.5 | 60.0 | 13.8 | 53.0 | 55.0 | 2.0 | 0.73 | 0.715 | -0.015 | 60.0 | 0.0 |
| 2026-05-31 10:54 | full_reheat | 45.2 | 59.9 | 12.9 | 134.0 | 145.0 | 11.0 | 1.6 | 1.885 | 0.285 | 59.23 | -0.67 |
| 2026-06-01 11:31 | full_reheat | 47.0 | 60.0 | 13.0 | 118.0 | 125.0 | 7.0 | 1.53 | 1.625 | 0.095 | 59.38 | -0.62 |
| 2026-06-02 10:07 | top_up | 53.9 | 60.0 | 14.7 | 63.0 | 70.0 | 7.0 | 0.85 | 0.91 | 0.06 | 59.46 | -0.54 |
| 2026-06-03 10:24 | full_reheat | 47.7 | 60.0 | 12.3 | 125.0 | 120.0 | -5.0 | 1.54 | 1.56 | 0.02 | 60.0 | 0.0 |
| 2026-06-04 10:01 | full_reheat | 52.6 | 60.0 | 10.5 | 74.0 | 75.0 | 1.0 | 0.97 | 0.975 | 0.005 | 60.0 | 0.0 |
