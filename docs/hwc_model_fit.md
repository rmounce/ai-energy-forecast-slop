# HWC Thermal Model Fit

Derived from clean, compressor-only cycles in `data/hwc_cop_cycles.csv`.

- Target temperature: `60.0` C
- Top-up class starts at: `tank_start >= 53.0` C
- Usable cycles: `7`

## Suggested Existing Planner Parameters

| parameter | suggested | note |
| --- | --- | --- |
| `thermal.nominal_power_w` | `780` | median clean HP proxy watts, rounded to 10 W |
| `thermal.heat_rate_c_per_hour` | `6.6` | median full-reheat probe lift rate where available |

## Optional Top-Up Model Parameters

| parameter | suggested | note |
| --- | --- | --- |
| `thermal.top_up_start_temp_c` | `53.0` | use the top-up rate at or above this modelled tank temp |
| `thermal.top_up_heat_rate_c_per_hour` | `5.5` | median near-target top-up probe lift rate |

## Diagnostic Split

- Top-up median heat rate: `5.5` C/h
- Top-up mean COP: `1.89`
- Mean usable-cycle COP: `2.3`

## Stratified Model Hints

| parameter | estimate | note |
| --- | --- | --- |
| `probe_height_fraction` | `0.62` | estimated from median 50% probe-rise timing as a fraction of cycle duration |
| `thermocline_width_fraction` | `0.63` | estimated from median 10%→90% probe-rise timing span |
| `probe_rise_10_cycle_fraction` | `0.25` | diagnostic only |
| `probe_rise_90_cycle_fraction` | `0.91` | diagnostic only |

## Cycle Class Summary

| class | n | median_heat_rate_c_per_hour | median_hp_mean_w | mean_cop | median_elec_kwh_per_c | median_probe_lag_min | median_probe_rise_10_min | median_probe_rise_50_min | median_probe_rise_90_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_usable | 7 | 6.0 | 780 | 2.3 | 0.125 | 15.5 | 17.0 | 46.0 | 71.5 |
| full_reheat | 5 | 6.58 | 771 | 2.47 | 0.118 | 22.0 | 43.0 | 78.0 | 108.5 |
| top_up | 2 | 5.45 | 812 | 1.89 | 0.151 | 15.0 | 14.8 | 34.5 | 51.8 |

Do not treat this as a stratified tank model yet; it is a parameter fit for the current single-node block planner.
