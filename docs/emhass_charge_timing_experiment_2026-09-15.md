# EMHASS charge-timing experiment — 2026-09-15

## Confirmed baseline

- Live EMHASS: `0.17.9`; `/healthz` and `/api/v1/last-run` confirmed 2026-09-15.
- EMHASS `0.17.7+`: tiny objective tie-break schedules economically equivalent PV
  curtailment as late as possible.
- HA DH and MPC payloads: time-dependent `weight_battery_charge` ramp.
  - Window: 14:30–21:00 local.
  - Increment: `$0.0002/kWh` per 5 minutes.
  - Peak: about `$0.0156/kWh`.
- External EMHASS config: `battery_stress_cost: 0.005`, 50 PWL segments.
- External EMHASS config: `inverter_stress_cost: 0.02`, 20 PWL segments.
- HA supplies the separate battery-discharge weight at runtime.

## Experiment decision

- Remove the HA battery-charge cost ramp; send scalar zero charge weight.
- Disable the battery-power PWL stress cost.
- Keep inverter AC-power PWL stress cost unchanged.
- Keep the battery-discharge weight unchanged.

## Rationale

- Battery: 40 kWh; 20 kW charge capability exceeds available PV/inverter power.
- Peak battery power is not the main stress concern.
- Thermal heat soak matters more; earlier charging precedes peak garage temperature.
- Native late-curtailment tie-break should prefer early PV absorption when schedules
  are otherwise economically equal.
- Forecast negative import prices create real value for retained headroom and therefore
  outrank the tie-break. Surprise negative prices remain a forecast-risk problem.
- Planned resistive dump loads and narrow live-price overrides are more direct hedges
  against surprise negative-price events than a permanent afternoon charge penalty.

## Watch during trial

- First curtailment time and curtailed energy.
- First/full battery time and success reaching 100% for top balancing.
- Peak battery power and battery/garage temperature.
- Response to forecast and surprise negative import prices.
- Cases where retained inverter stress suppresses mildly profitable grid charging.

## External rollback values

```json
"battery_stress_cost": 0.005,
"battery_stress_segments": 50,
"inverter_stress_cost": 0.02,
"inverter_stress_segments": 20
```
