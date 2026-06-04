# HWC Calibration Cycles

Curated cycle-level calibration output from `hwc_cop_analysis.py`.

- Source window: existing CSV merged with extracted window (since `2026-06-04`)
- Rows: 9 total, 7 clean
- Method: compressor-on windows from HA/InfluxDB; electrical input from baseline-subtracted `sensor.remaining_power_load`; thermal output from tank probe delta plus standing loss.
- Caveat: tank stratification means single-probe thermal output is approximate; use clean flags and cycle context before fitting model parameters.

| start | dur_min | tank_start | tank_end | ambient | wet_bulb | baseline_w | hp_mean_w | hp_p95_w | elec_kwh | therm_kwh | cop | probe_lag_min | probe_rise_10_min | probe_rise_50_min | probe_rise_90_min | exhaust_start | exhaust_max | exhaust_end | element_on | defrost_on | four_way_on | clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 19:35 | 38 | 43.7 | 47.1 | 14.5 | 13.6 | 186 | 800 | 1609 | 0.52 | 0.97 | 1.87 | 10.0 | 8.0 | 21.5 | 33.0 | 30.0 | 57.0 | 54.0 | False | False | False | False |
| 2026-05-28 20:35 | 81 | 47.9 | 59.9 | 13.5 | 12.8 | 196 | 806 | 879 | 1.1 | 3.32 | 3.03 | 8.0 | 13.0 | 46.0 | 71.5 | 48.6 | 82.0 | 81.1 | False | False | False | True |
| 2026-05-29 10:01 | 53 | 55.5 | 60.0 | 14.5 | 13.8 | 140 | 814 | 885 | 0.73 | 1.27 | 1.75 | 14.5 | 13.5 | 33.0 | 48.0 | 24.8 | 70.9 | 67.2 | False | False | False | True |
| 2026-05-30 10:25 | 123 | 48.0 | 60.0 | 15.8 | 14.4 | 820 | 130 | 346 | 0.27 | 3.39 | 12.63 | 26.5 | 52.0 | 88.5 | 114.0 | 28.2 | 82.9 | 81.0 | False | False | False | False |
| 2026-05-31 10:54 | 134 | 45.2 | 59.9 | 15.6 | 12.9 | 153 | 715 | 862 | 1.6 | 4.11 | 2.56 | 33.0 | 52.0 | 84.0 | 122.0 | 29.4 | 78.0 | 72.5 | False | False | False | True |
| 2026-06-01 11:31 | 118 | 47.0 | 60.0 | 17.2 | 13.0 | 117 | 771 | 905 | 1.53 | 3.63 | 2.38 | 22.0 | 43.0 | 78.0 | 108.5 | 27.8 | 82.9 | 79.4 | False | False | False | True |
| 2026-06-02 10:07 | 63 | 53.9 | 60.0 | 16.5 | 14.7 | 123 | 810 | 901 | 0.85 | 1.72 | 2.02 | 15.5 | 16.0 | 36.0 | 55.5 | 26.9 | 72.0 | 65.3 | False | False | False | True |
| 2026-06-03 10:24 | 125 | 47.7 | 60.0 | 14.5 | 12.3 | 107 | 738 | 876 | 1.54 | 3.46 | 2.24 | 50.0 | 57.5 | 85.0 | 115.0 | 20.3 | 81.9 | 76.1 | False | False | False | True |
| 2026-06-04 10:01 | 74 | 52.6 | 60.0 | 13.5 | 10.5 | 126 | 780 | 861 | 0.97 | 2.08 | 2.15 | 13.5 | 17.0 | 45.0 | 67.5 | 21.0 | 78.9 | 74.0 | False | False | False | True |
