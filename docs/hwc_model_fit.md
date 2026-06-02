# HWC Thermal Model Fit

Derived from clean, compressor-only cycles in `data/hwc_cop_cycles.csv`.

- Target temperature: `60.0` C
- Top-up class starts at: `tank_start >= 53.0` C
- Usable cycles: `5`

## Suggested Existing Planner Parameters

| parameter | suggested | note |
| --- | --- | --- |
| `thermal.nominal_power_w` | `810` | median clean HP proxy watts, rounded to 10 W |
| `thermal.heat_rate_c_per_hour` | `6.6` | median full-reheat probe lift rate where available |

## Optional Top-Up Model Parameters

| parameter | suggested | note |
| --- | --- | --- |
| `thermal.top_up_start_temp_c` | `53.0` | use the top-up rate at or above this modelled tank temp |
| `thermal.top_up_heat_rate_c_per_hour` | `5.5` | median near-target top-up probe lift rate |

## Diagnostic Split

- Top-up median heat rate: `5.5` C/h
- Top-up mean COP: `1.89`
- Mean usable-cycle COP: `2.35`

## Cycle Class Summary

| class | n | median_heat_rate_c_per_hour | median_hp_mean_w | mean_cop | median_elec_kwh_per_c | median_probe_lag_min |
| --- | --- | --- | --- | --- | --- | --- |
| all_usable | 5 | 6.58 | 806 | 2.35 | 0.118 | 15.5 |
| full_reheat | 3 | 6.61 | 771 | 2.66 | 0.109 | 22.0 |
| top_up | 2 | 5.45 | 812 | 1.89 | 0.151 | 15.0 |

Do not treat this as a stratified tank model yet; it is a parameter fit for the current single-node block planner.
