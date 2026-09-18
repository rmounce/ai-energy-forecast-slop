# Dump-load control

## Loads

- PowerWaster 1: `switch.snf15_snf15`; fan heater; assumed 2,000 W.
- PowerWaster 2: `switch.snf18_snf18`; fan heater; assumed 2,000 W.
- PowerWaster 3: `switch.powerwaster_3_athom_c3_d97c`; oil heater; measured power; 2,150 W nominal.
- PowerWaster 4: `switch.powerwaster_4_athom_c3_dfb0`; oil heater; measured power; 2,150 W nominal.
- PowerWaster 3/4 run ESPHome 2024.12.2. Framework issue prevents firmware updates; do not require an update for operation.

## Controller

- HA automation: `automation.amber_negative_price_dump_loads` (ID `1773472044412`).
- Non-negative effective import price: turn all loads off.
- Invalid curtailment policy or unavailable planned curtailment: turn all loads off.
- Negative price plus planned curtailment above the aggressive threshold: turn all available loads on.
- Otherwise add one load only when grid import plus its conservative allowance is at most 14,500 W.
- Admission priority: fan heaters, then oil heaters.
- Shedding priority above 14,500 W: oil heaters, then fan heaters.
- Each pair sorts by `last_changed`: longest-off enters first; longest-on sheds first.
- Control allowances: fan 2,000 W; oil 2,250 W.
- One-minute reevaluation handles thermostat cycling and missed threshold crossings.
- A 15-second all-load change guard prevents immediate reversal.

## Measurement

- `sensor.estimated_dump_load_power` is the aggregate despite its legacy name.
- Fan heaters contribute 2,000 W each while their switches are on.
- Oil heaters contribute their live plug power while available and 2,150 W fallback while switched on if metering is unavailable.
- A thermostat-open oil heater correctly contributes approximately 0 W while its plug remains on.

## Commissioning observations

- 2026-09-18, individual live switching test: both new plug mappings and relays worked.
- PowerWaster 3 reached 2,106 W after its meter settled; later samples were 2,063 W.
- PowerWaster 4 reached 2,096 W during its test.
- Plug voltage/current/power entities update asynchronously. Ignore the first few samples after switching; aggregate power follows the plug's reported power.
- The 2,250 W control allowance remains conservative relative to the observed loads.
