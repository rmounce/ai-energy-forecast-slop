# HWC questions for previous implementer

Context: we are about to replace the separate HWC planner/executor timers with one
long-lived HA WebSocket daemon that replans on relevant input changes and executes the
published block plan. The current planner is modelling-only but plausible; actuation is
still config-disabled.

Please answer briefly where you know the answer.

1. **Aquatech modes:** do you know how Tuya/Local Tuya maps the app's "standard" mode to HA?
   Live HA exposes `off`, `heat_pump`, `eco`, `high_demand`, `performance`, `electric`.
   Is `heat_pump` the right compressor-only mode for normal scheduled reheats, or should we
   use `eco`, `high_demand`, or `performance`?

2. **Reheat trigger hysteresis:** do you have observed thresholds for each mode? The working
   assumption is that normal enabled operation reheats around/below 55 C and stops at the
   configured setpoint up to 60 C. Is that confirmed? Any deadband or minimum-runtime behavior
   we should account for?

3. **Setpoint semantics:** when we call `water_heater.set_temperature` with a setpoint between
   55 and 60 C, does the unit reliably stop the compressor at that setpoint, or does it round,
   clamp, ignore, or defer the setting depending on mode?

4. **Safe off behavior:** is `water_heater.turn_off` / operation mode `off` safe as the normal
   between-block state? Any cloud/local Tuya quirks, delayed writes, or cases where `off` also
   resets settings we should restore on the next block?

5. **Current HA timer:** where is the existing fixed timer/automation that flips the unit
   eco/standard between 10:00 and 16:00? Should the daemon explicitly disable/replace it, or is
   it outside this repo and something the owner will remove manually?

6. **Compressor sensor reliability:** is `binary_sensor.aquatech_compressor` local/reliable
   enough for executor decisions? Any observed lag relative to actual compressor state?

7. **Temperature entity choice:** should actuation decisions use `sensor.heat_pump_temperature`
   or `water_heater.aquatech.current_temperature`? The planner currently uses
   `sensor.heat_pump_temperature`; live HA shows both are present.

8. **Legionella / 60 C policy:** should every planned daily main block target 60 C for now, or
   should the first actuation version already lower routine blocks and schedule separate periodic
   60 C reheats?

9. **Daemon triggers:** any reason not to replan on state changes for
   `sensor.ai_dh_import_price_forecast`, `sensor.heat_pump_temperature`, and HWC plan-relevant
   weather refresh/heartbeat? Any HA entities besides these that should trigger HWC replanning?

10. **Failure mode preference:** if the daemon cannot read a fresh plan or loses HA connection,
    should it leave the water heater in its current state, force `off`, or fall back to a simple
    fixed daytime window?
