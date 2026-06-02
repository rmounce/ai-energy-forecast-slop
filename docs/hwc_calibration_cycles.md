# HWC Calibration Cycles

Curated cycle-level calibration output from `hwc_cop_analysis.py`.

- Source window: since `2026-05-28`
- Rows: 7 total, 5 clean
- Method: compressor-on windows from HA/InfluxDB; electrical input from baseline-subtracted `sensor.remaining_power_load`; thermal output from tank probe delta plus standing loss.
- Caveat: tank stratification means single-probe thermal output is approximate; use clean flags and cycle context before fitting model parameters.

| start | dur_min | tank_start | tank_end | ambient | wet_bulb | baseline_w | elec_kwh | therm_kwh | cop | clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 19:35 | 38 | 43.7 | 47.1 | 14.5 | 13.6 | 186 | 0.52 | 0.97 | 1.87 | False |
| 2026-05-28 20:35 | 81 | 47.9 | 59.9 | 13.5 | 12.8 | 196 | 1.1 | 3.32 | 3.03 | True |
| 2026-05-29 10:01 | 53 | 55.5 | 60.0 | 14.5 | 13.8 | 140 | 0.73 | 1.27 | 1.75 | True |
| 2026-05-30 10:25 | 123 | 48.0 | 60.0 | 15.8 | 14.4 | 820 | 0.27 | 3.39 | 12.63 | False |
| 2026-05-31 10:54 | 134 | 45.2 | 59.9 | 15.6 | 12.9 | 153 | 1.6 | 4.11 | 2.56 | True |
| 2026-06-01 11:31 | 118 | 47.0 | 60.0 | 17.2 | 13.0 | 117 | 1.53 | 3.63 | 2.38 | True |
| 2026-06-02 10:07 | 63 | 53.9 | 60.0 | 16.5 | 14.7 | 123 | 0.85 | 1.72 | 2.02 | True |
