# Dump-load control

## Loads

- PowerWaster 1: `switch.snf15_snf15`; fan heater; assumed 2,000 W.
- PowerWaster 2: `switch.snf18_snf18`; fan heater; assumed 2,000 W.
- PowerWaster 3: `switch.powerwaster_3_athom_c3_d97c`; oil heater; measured power; 2,150 W nominal.
- PowerWaster 4: `switch.powerwaster_4_athom_c3_dfb0`; oil heater; measured power; 2,150 W nominal.
- PowerWaster 3/4 run ESPHome 2024.12.2. Framework issue prevents firmware updates; do not require an update for operation.

## Controller

- HA automation: `automation.amber_negative_price_dump_loads` (ID `1773472044412`).
- Any grid status other than `On Grid`: turn all four loads off in one service call.
- Grid-status changes trigger immediate reevaluation; restart mode interrupts an admission sequence.
- Tracked automation: `hass/automation-dump-loads.yaml`.
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
- HA counts only the live PowerWaster 3/4 meter entities in
  `sensor.individually_metered_load`, so their measured power is known rather than inflating
  `sensor.remaining_power_load`. PowerWaster 1/2 fixed estimates do not enter the known total.
- The main dashboard's state-sorted `Individual Loads` card shows PowerWaster 3 and 4
  separately. It does not show the aggregate estimated dump-load entity.
- A separate `Dump Loads` dashboard card shows the aggregate as `Total (estimated)` and the
  direct switch-based PowerWaster 1/2 estimate. Estimated values are not mixed into the
  state-sorted metered-load list.
- `sensor.estimated_unmetered_dump_load_power` reports 0, 2,000, or 4,000 W directly from the
  PowerWaster 1/2 switch states; it does not subtract meters from the aggregate.
- Dump loads remain part of `sensor.deferrable_load_power` and are therefore excluded from
  `sensor.power_consumed_without_deferrable_loads`, the base load-forecast input.
- The HWC heat-pump meter follows the same split: it is a known individual load, but is also
  deferrable and excluded from the base forecast before its schedule is added to EMHASS demand.

## Oil-heater circulation fans

- PowerWaster 3 drives Bed 3 fan `fan.fan_1` through
  `automation.powerwaster_3_bed_3_circulation_fan` (ID `1789795626001`).
- PowerWaster 4 drives Bed 2 fan `fan.fan_2` through
  `automation.powerwaster_4_bed_2_circulation_fan` (ID `1789795626002`).
- Plug switch on: request forward direction and 50% speed.
- Plug switch off: keep the fan running for six minutes, then stop it if the plug remains off.
- Each room uses a separate restart-mode automation. A new switch transition cancels that
  room's pending shutdown without affecting the other room.
- HA startup reconciles enabled heaters. A fan stopped or reversed while its heater remains
  enabled is restored to forward 50%-speed operation.
- Control follows plug switch state, not measured heater power. Low/zero power while the plug
  is on can mean the heater thermostat is saturated; circulation remains useful then.
- Existing Bed 2/3 air-conditioning fan automations remain separate.

## Commissioning observations

- 2026-09-18, individual live switching test: both new plug mappings and relays worked.
- PowerWaster 3 reached 2,106 W after its meter settled; later samples were 2,063 W.
- PowerWaster 4 reached 2,096 W during its test.
- Plug voltage/current/power entities update asynchronously. Ignore the first few samples after switching; aggregate power follows the plug's reported power.
- The 2,250 W control allowance remains conservative relative to the observed loads.
- 2026-09-19: added and enabled both live oil-heater circulation-fan automations. Both heater
  switches were off at final verification, so the fans correctly remained off.
- 2026-09-19: increased both circulation fans from minimum speed (16%) to 50% and applied the
  change immediately while both heaters were enabled.
- 2026-09-19: added the live PowerWaster 3/4 meters separately to the known-load total and main
  dashboard through a template hot reload and Lovelace WebSocket save. Fixed PowerWaster 1/2
  estimates remain outside the known-load presentation. HA configuration check passed; the
  dashboard save passed read-after-write verification.
- 2026-09-19: added a separate main-dashboard `Dump Loads` card for the aggregate and
  PowerWaster 1/2 estimate; the measured-load list and known-load sum remain measured-only.
