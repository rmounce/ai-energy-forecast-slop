# HWC answers from previous implementer

Answers to `docs/hwc_questions_for_previous_implementer.md`. **Important scope caveat:**
my work was **modelling + telemetry analysis only — I never actuated the unit** (no
`water_heater` service calls, no mode changes). So Q1–Q4 and parts of Q6 are outside what I
verified; I mark those clearly rather than guess. The owner knows the app/mode behaviour
first-hand and is the best source for those.

1. **Aquatech modes / Tuya mapping — DON'T KNOW.** I never exercised the modes. I don't know
   how the app's "standard" maps to the HA list (`heat_pump`/`eco`/`high_demand`/`performance`/
   `electric`). Ask the owner, and/or mine history: the operation-mode entity should be in
   InfluxDB — correlate each mode against `aquatech_compressor`, `aquatech_element`, and the
   `exhaust`/power behaviour to see which are compressor-only vs element-assisted. Strong prior:
   `electric`/`performance` likely engage the 1800 W element (avoid for normal reheats);
   `heat_pump`/`eco` are likely compressor-only — but verify against `aquatech_element`.

2. **Reheat hysteresis — PARTIAL (from data).** In the cycles I analysed (owner's current
   timer/setpoint), reheats *started* with the tank probe ~45–48 °C and *stopped* at 60 °C
   (e.g. 45.2→59.9, 47.0→60.0, 47.9→60.0; plus one 55→60 top-up). So observed behaviour is
   consistent with "reheat when ≲48 °C, stop at setpoint." I did **not** vary mode or setpoint,
   so I can't give per-mode thresholds or confirm a deadband/min-runtime. Cycles ran ~80–135
   min; I saw no short-cycling in the data, but didn't probe minimum runtime.

3. **Setpoint semantics (55–60) — DON'T KNOW.** Never set it. Verify empirically: set 55,
   57, 60 and watch where the compressor actually stops (`aquatech_compressor` + tank probe).
   Note the unit clearly *can* hit 60 on compressor alone (observed), and the owner said the
   element is only needed 60→70 — so 55–60 should be compressor-only, but confirm it doesn't
   silently clamp/round per mode.

4. **Safe-off behaviour — DON'T KNOW.** Never used `turn_off`/`off`. Local Tuya can have
   delayed/dropped writes; test that `off` doesn't reset mode/setpoint that you then need to
   restore. Recommend reading back state after each write and reconciling.

5. **Existing HA timer (eco↔standard 10:00–16:00) — owner's, NOT in this repo.** The owner
   told me they have an HA timer flipping eco→standard 10:00–16:00 daily; it's in their HA
   config, not in `hass/` here (I never saw it). The daemon and that timer would be **two
   controllers fighting** — the owner should disable/remove it when the daemon takes over.
   I don't have its entity id.

6. **Compressor sensor reliability — LIKELY OK, lag unmeasured.** `binary_sensor.aquatech_
   compressor` transitions lined up cleanly with the onset of the exhaust-temp and power rises
   in every cycle, so it's a trustworthy *state* signal. It logs on change (query InfluxDB with
   `fill(previous)`). I did **not** measure its latency vs true compressor state, and I'm not
   certain whether it's local or cloud-sourced — for a tight control loop, confirm it's Local
   Tuya and check lag.

7. **Temperature entity — use Local Tuya for control.** The planner used
   `sensor.heat_pump_temperature` because it's the one **logged to InfluxDB** (cloud Tuya) for
   historical analysis. For real-time actuation, prefer `water_heater.aquatech.current_temperature`
   (Local Tuya — lower latency, no cloud dependency; the owner wants to drop cloud reliance).
   **First verify the two track each other** (they should be the same probe). Caveat from the
   characterisation: this probe is the *control* probe and **lags the condenser during a reheat
   due to stratification** — fine for "is the tank charged" decisions, but don't treat it as the
   instantaneous tank energy.

8. **Legionella / 60 °C policy — recommend lowering routine, with periodic 60.** The COP data
   is clear: the 55→60 °C tail runs at COP ~1.75 vs ~2.4–3.0 cycle-average, so **daily-to-60 is
   the expensive option**. Recommended policy: routine blocks target ~**55 °C** (comfortably
   above legionella growth range), with a **periodic** (e.g. weekly) 60 °C reheat placed in the
   cheapest/warmest slot. Daily-60 is the owner's originally-stated preference and is the most
   conservative for legionella — so make this a **config policy** and let the owner choose;
   don't silently drop below ~50 °C for routine. (Keep `min_temperatures` ≥ ~48 °C as a floor.)

9. **Daemon replan triggers — your list is right; a few additions.** Replanning on
   `sensor.ai_dh_import_price_forecast` (the cost driver — note the price pipeline is already
   event-driven via `ai-energy-listener.service`), `sensor.heat_pump_temperature` (state), and a
   weather refresh is sensible. Additions: (a) a **periodic heartbeat** (e.g. 30 min) as a
   defensive fallback if no trigger fires; (b) **debounce** the tank-temp trigger (don't replan
   on every 0.1 °C wiggle — only on meaningful change or block boundaries); (c) replan if the
   unit's **mode/state changes unexpectedly** (manual intervention) so the daemon doesn't fight
   the user. Don't trigger on the noisy `exhaust`/coil sensors.

10. **Failure mode — fail to a safe fixed window, never cold.** Forcing `off` on a lost plan/
    connection is unsafe (risks no hot water / missed legionella); leaving "current state" can
    strand it off. Best: **fall back to a simple fixed daytime reheat window** (≈ the owner's
    original 10:00–16:00 behaviour) — or hand control back to the unit's native thermostat if
    that path still exists. Guiding principle: a daemon/optimiser failure must **never** leave
    the tank cold or skip legionella; degrade to "guaranteed daily daytime reheat," not "off."

— Prior implementer (modelling/characterisation phase). Detailed evidence:
`docs/hwc_thermal_characterisation.md`; data tool: `hwc_cop_analysis.py`.
