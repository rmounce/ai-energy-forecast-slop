# HWC surplus & negative-price strategy

How the heat-pump hot-water (HWC) unit should exploit **PV surplus / curtailment** and
**negative grid buy prices**, and how that fits the existing flexible-load controllers
(battery, A/C, dump-load fan heaters). Design + phased plan. Read alongside
`docs/hwc/handover.md`, `docs/hwc/dp_planner.md`, `docs/hwc/aquatech_entities.md`.

Status: **negative-price override enabled 2026-07-14; reactive surplus-element override enabled
2026-08-03.** The longer-term DP curtailment model below remains optional future work.

**Watch on the first live event:** element power is still *assumed* (1800 W) — confirm it from
the Athom ch2 circuit, since it sets the break-even's Δ. The default paths (`performance` while
the compressor runs, `electric` while it is idle) do not depend on Δ; only the
compressor-interrupt branch does.

## The two events (and why they're handled differently)

| | PV surplus / curtailment | Negative grid buy price |
|---|---|---|
| What it is | sunny, battery full / export-limited; PV being thrown away | paid to import; constraint is the grid connection limit |
| What's scarce | the curtailed kWh themselves (finite, free) | nothing — you earn per kWh |
| Right objective | max **useful heat per free kWh** → COP matters → heat pump | max **electrical draw** → COP irrelevant → element |
| **Forecastable?** | **yes** — EMHASS publishes `mpc_p_pv_curtailment` as a forward series | **usually not** — appears by surprise |
| **Who owns it** | **the DP** (plan it optimally from the forecast) | **the executor** (reactive override) |
| Frequency | starts in spring, more often than negative price | rarer; neither happens in winter |

The split is on **forecastability**: curtailment is forecast, so the DP can plan around it;
negative price is a surprise, so a reactive executor override is the right tool.

## How the existing flexible loads coordinate (cross-repo map)

There is **no central arbiter** — a federation of independent controllers, each thresholded
off the same shared signals (`sensor.amber_effective_general_price`,
`sensor.mpc_p_pv_curtailment`, `sensor.sigen_plant_grid_import_power`). Priority emerges from
who acts first on negative price and from the connection-limit band controllers voluntarily
yield.

1. **Battery EMS** — `hass/.../automations.yaml` "Battery EMS Control Based on Amber Prices
   and SOC" (+ `packages/sigenergy_ems.yaml`). Large `choose:` tree over EMHASS MPC outputs
   (`p_grid`, `p_batt`, `p_pv`, `p_pv_curtailment`, `soc`) + Amber prices. Explicit
   negative-price branches: `price < 0` and `planned_pv_curtailment_kwh > 0` → proactive
   full-rate grid charge to `max_grid_import` (14999 W) + curtail PV export. The only
   reusable / arbitrageable store. **Auto-limits its own charge rate to respect the 15 kW
   import limit** — no external coordination needed.
2. **A/C thermal-mass absorption** — `~/actrl/actrl.py` (AppDaemon). Reads next-hour average of
   `sensor.mpc_p_pv_curtailment`, feeds a "grid-surplus integral" that nudges A/C setpoints to
   soak curtailed PV into the house thermal mass. Keys off forecast curtailment magnitude only,
   not price. (`statctrl.py` is unrelated — adaptive thermostat start.)
3. **Fan-heater dump loads** — `automations.yaml` "Amber negative price dump loads"
   (`switch.snf15_snf15`, `switch.snf18_snf18`). Only when `price < 0`. Surplus present
   (`planned_pv_curtailment_kwh > 1`) → all on; else modulate to hold grid import in a
   [10, 12.5] kW band (hard back-off at 14.5 kW). Lowest tier — pure dump, no useful product.
4. **HWC** — `hwc_dp_planner.py` + `services/hwc_daemon.py`. Today: cost-min import only,
   `heat_pump` mode, setpoint clamped 55–60 °C. **Not in the surplus/negative-price
   coordination at all** — this doc adds it.

Grid connection ceiling ≈ 15 kW (`number.sigen_plant_grid_import_limitation` default 15,
`max_grid_import` 14999, fan-heater hard-off at 14.5 kW).

## Priority placement for HWC

- **Negative price:** HWC is the **top** claimant — it heats **unconditionally** (no
  available-headroom margin required), though *which* source runs depends on the compressor's
  state (see below). Max HWC draw is 1800 W and the battery auto-limits its charge to respect
  the 15 kW ceiling, so simultaneous draw never trips the limit; no special battery coordination
  is needed.
- **Surplus:** HWC ranks above the fan heaters (useful product, COP-efficient) and is fed
  *upstream* of the battery — HWC is planned first and fed into EMHASS as load, so the battery
  mops up whatever surplus remains (it "reshapes" around HWC).

## Actuation primitive (Aquatech RAPID X6 modes)

**Boost is confirmed on the installed Aquatech (2026-08-02).** With the controller on, hold panel
`M + Up` for three seconds to establish a hidden Boost latch. While latched, remote Standard/60
suppresses the element and remote Hybrid+/70 starts it immediately above 60 °C. `turn_off` clears
the latch. The latch is not present in any of the 50 raw Tuya DPs, so it cannot currently be created
or observed remotely. A target-only 60→70 change within Hybrid+ does not start the element; dispatch
requires a real mode transition from Standard/60 into Hybrid+/70. Manual sterilisation remains unobserved. See
`controller_family_manual_sweep.md`.

`water_heater.aquatech` operation modes (HA advertises 15–75 °C, but the physical maximum is
70 °C; confirmed 2026-07-27):

- **`heat_pump`** — heat pump only, ~700 W, capped at 60 °C. COP ~2.4–3.0 (the 55→60 tail is
  ~1.75). Today's only mode.
- **`electric`** — resistive element only, **1800 W, COP 1**, target 70 °C; an ordinary new cycle
  starts at the inclusive 60 °C trigger, not above it.
- **`performance` (Hybrid+)** — heat pump to 60 °C, then element 60→70.
  Ordinary new-cycle trigger is 50 °C. With the physical Boost latch established, selecting this
  mode starts the element directly above 60 °C and suppresses the compressor.

**The two sources are mutually exclusive — physically, not just by mode logic.** The unit is on
a 10 A plug (~2400 W) and heat pump + element together would be ~2500 W, so it can never run
both. `performance` is therefore strictly *sequential* (compressor to 60, then element), and
**1800 W is the unit's maximum possible draw** in any mode. This is what makes the
negative-price rule below a genuine either/or.

Heat rates are similar: 1800 W element ≈ 6.9 °C/hr for 225 L; the ~700 W heat pump at COP ~2.5
delivers ~1750 W thermal ≈ similar. The element is not *faster* — it is *available above 60*
and *cheaper to start* (no compressor short-cycle constraint).

The executor currently issues only `heat_pump` / `off` (`setpoint_max_c: 60`). **Learning
`electric` mode + setpoint-to-70 is shared groundwork for both phases.**

The confirmed Boost primitive is not integrated into the executor. Exploiting it requires a policy
that deliberately avoids `turn_off` while the latch must persist, uses a ≤60 °C mode as the idle
state, verifies physical element power on every request, and treats any off/restart as requiring
manual re-latching.

## Design — negative price (executor override, reactive)

Lives **inside `hwc_daemon`** as a layer on top of the DP plan (not a separate HA automation —
a second controller fighting for `water_heater.aquatech` is the one thing guaranteed to
misbehave). Externally still federated (keys off the same `amber_effective_general_price`).

Live `amber_effective_general_price < 0` overrides the DP's plan for that interval. **What we
force depends on whether the compressor is already running** (amended 2026-07-14 — the original
decision was unconditional `electric`):

| Compressor state | Action | Why |
|---|---|---|
| **off** | **`electric` @ 70 °C** | nothing to interrupt; arm max draw. If already above ~60 °C, the common element hysteresis may delay restart until the probe falls to its trigger |
| **running** | **`performance` @ 70 °C** | keeps the compressor uninterrupted to 60 °C, then the element takes it 60→70 automatically — full dump on any event long enough to matter, no restart |
| **running, deeply negative** | **`electric` @ 70 °C**, latched | only when the break-even below says the restart pays for itself |

### Why not unconditional `electric`

Interrupting a running compressor to force the element **buys no extra heat**: 1800 W of element
at COP 1 ≈ 1800 W thermal, and ~700 W of heat pump at COP ~2.5 ≈ 1750 W thermal (see heat-rate
note above). It buys only the extra **Δ = 1.8 − 0.7 = 1.1 kW** of grid draw we are being paid
for — and it costs one compressor restart when the price flips back.

Below 60 °C, 700 W is simply the most this unit can draw without stopping the compressor (10 A
plug, mutually exclusive sources). That sub-60 gap is a physical constraint, not a choice.

### Break-even (when a deep negative price *does* justify the restart)

Switch a **running** compressor to `electric` when

```
|price| × 1.1 kW × gain_hours  >  transition_cost_aud
    ⇔   price < − transition_cost_aud / (1.1 × gain_hours)
```

`gain_hours` is **not** simply the remaining negative window. If we *don't* switch,
`performance` escalates to the element by itself once the tank reaches 60 °C, and past that
point we draw 1800 W either way — the switch has bought nothing. So

```
gain_hours = min(remaining_negative_window, time_to_reach_60C)
```

with `time_to_reach_60C` from the current tank temperature and the heat-pump thermal rate the DP
already models. The correction bites hardest exactly when the tank is nearly hot, which is when
interrupting the compressor is least worthwhile.

`remaining_negative_window` comes from the leading run of forecast-negative Amber intervals
(Amber publishes forward prices, so the override is not fully blind). **If the forecast is
unavailable, assume one interval** — pessimistic, so we won't switch.

At today's constants (`transition_cost_aud` 0.05, Δ 1.1 kW):

| `gain_hours` | Break-even price |
|---|---|
| 5 min | −54 c/kWh |
| 15 min | −18 c/kWh |
| 30 min | −9 c/kWh |
| 60 min | −4.5 c/kWh |

So this fires only on genuinely deep or sustained events — which is the intent.

**`transition_cost_aud` (0.05) is a modelling proxy for compressor wear + short-cycle loss, not
a measured cost.** Here it *is* the threshold. Tightening this behaviour means revisiting that
number, not the formula.

### Price entity cadence (conservative → confirmed)

`sensor.amber_effective_general_price` updates **twice per 5-minute interval**: a *conservative*
forecast at the start of the interval, then the *confirmed actual* ~30 s later. It is a bare
template sensor — **no attributes**, so there is no flag distinguishing the two updates and no
forward prices on it. Forward intervals come from
`sensor.amber_billing_interval_forecasts_general_price` (`Forecasts`, 5-minute resolution;
config `amber_billing_entity`), which is what `remaining_negative_window` is computed from.

With no flag, the two updates are separated by **which decisions each is allowed to make**,
split by what a wrong call costs:

- **Entry acts on the first (conservative) value.** A false entry costs ~30 s of element on a
  positive price (~15 Wh, well under a cent); waiting for confirmation would forfeit 10 % of
  every genuine 5-minute dump window.
- **The break-even compressor interrupt waits for the confirmed price** — it is the only
  decision that costs a restart. "Confirmed" is implemented as *negative across an update
  boundary* (two consecutive negative reads, or negative ≥ 45 s), not by trying to identify
  which update we are on. The delay barely affects `gain_hours`.
- **Exit below 60 °C acts on the first value.** Reverting is cheap and it is what guards
  `performance`'s ungated element leg. At/above 60 °C, exit waits for the confirmed current
  5-minute price (`sensor.amber_5min_current_general_price`) to be **strictly positive**;
  confirmed zero continues the event.

This assumes *conservative* means biased **high** for a buy price (pessimistic for the buyer), so
a conservative-negative read is strong evidence the confirmed price is negative too.

### Latching and exit

- **Latch:** once switched to `electric` under the break-even, stay there until the price goes
  positive at/above 60 °C — do not re-evaluate and flip back to `heat_pump`/`performance` mid-event. A
  flip-back pays the very restart we were trying to avoid *and* gives up the dump.
- **Exit:** below 60 °C, the effective price may release the override at zero. At/above 60 °C,
  only a **strictly positive confirmed current price** releases it; zero continues heating.
  When released, **actively revert to the DP plan's mode and setpoint** — do not merely stop
  asserting the override. `performance`'s 60→70 element leg is **ungated**: left in place above
  60 °C it will keep importing at 1800 W to reach setpoint.
- **Physical confirmation:** HA mode + target only prove that Local Tuya accepted the request.
  After 60 s and at/below the 60 °C trigger, element-only commands must also have
  `binary_sensor.aquatech_element == on`; otherwise retry. Above 60 °C, leave the accepted
  command armed: fresh `electric`, `performance`, and likely `high_demand` share the element
  controller's re-trigger hysteresis, and repeated writes may reset its start processing.

Forecast negative prices (occasional) get optimal DP treatment automatically (price < 0 →
negative cost → DP runs the element hard); the override is the safety net for the *un*forecast
ones.

## Design — surplus / curtailment

### Implemented first: reactive element event

The daemon gives HWC first claim on a large current surplus without duplicating EMHASS's
curtailment calculation. `sensor.mpc_p_pv_curtailment >= 2200 W` for 120 s, tank `<= 60 C`,
valid non-adverse power telemetry, and both compressor and element off starts `electric @ 70 C`.
It does not interrupt a planned heat-pump cycle.

Once started, the event ignores the curtailment value: the new 1800 W load naturally consumes
the signal that caused entry. It exits after either grid import exceeds 400 W or battery power
is below -400 W (discharging) continuously for 180 s, or when the tank reaches 69.8 C. Missing
grid/battery telemetry is adverse while active and blocks entry. Negative-price control has
higher precedence and clears any surplus latch. State persists across daemon restarts.

This deliberately needs no Boost latch. The ordinary electric cycle is initiated at or below
the confirmed inclusive 60 C trigger, then runs toward 70 C. HVAC remains independently
controlled and absorbs smaller residual surplus; explicit cross-repo arbitration is deferred
until observations show it is needed.

### Future option: forecast-driven DP integration

Model curtailment in the DP so the optimal plan emerges rather than being hand-coded. Most of
the earlier reactive thresholds (2 kWh / 0.5 kWh / instantaneous gates) **disappear** — they
become emergent from the DP's costs.

**Actions per step:** `{off, heat_pump, electric}`.
- `heat_pump`: ~700 W, available < 60 °C, compressor start charged `transition_cost_aud` (0.05).
- `electric`: 1800 W, available up to 70 °C, **start cost ~0** (no short-cycle constraint).
- This reproduces "compressor commits, element bang-bangs" *emergently* — no thresholds.

**State range:** extend 45 → **70 °C** (was capped at 60).

**Curtailment-aware pricing — the accounting, moved from reactive to planned.** For each
action's electrical draw at step `t`:
```
grid_power[t] = max(0, action_power − available_curtailment[t])
cost[t]       = grid_power[t] × price[t] × dt   (+ transition cost on compressor start)
```
In a spill interval marginal cost → ~0, so the DP pulls heating into it and picks element vs
heat pump automatically (free → element's COP penalty costs nothing and it reaches higher /
absorbs more spill; priced → heat pump's COP wins).

**Availability of curtailment — two conditions, both required:**
1. **Budget (economic cap):** forecast curtailment kWh, summed over a forward window (≤ 8 h),
   **use-it-or-lose-it**. This is the *genuinely-free* energy — what's truly being wasted vs
   what the battery wants for arbitrage. On a day the battery can soak all surplus, budget ≈ 0
   → no free HWC (correct: HWC there would just lower battery SoC and cost arbitrage).
2. **Per-slot feasibility (physical):** `PV_available[t] > base_load[t] + HWC_power[t]`, using
   **PV-available (Solcast potential), not PV-actual (clipped)**. We do **not** gate on reported
   curtailed power ≥ 1800 W: EMHASS reshapes battery charging around the HWC load (battery
   yields), so the binding instantaneous constraint is PV *capacity* vs load, not the
   residual-after-battery curtailment.

**Terminal value:** the DP already has `terminal_penalty_aud_per_c` (0.05, one-sided below
`terminal_target` = start temp). Free-curtailment soak is driven not by a terminal *reward* but
by **forward cost displacement within the 48 h horizon** (heating cheap/free now reduces the
cost of later draws + the min-temp/legionella obligations), which the DP already accounts for.
No new terminal reward needed; speculative over-storage beyond horizon draws is marginal once
standing loss is counted.

## Coordination caveats (not blockers)

- **EMHASS ↔ HWC coupling is circular.** The curtailment EMHASS publishes is from a solve that
  doesn't fully reflect the *latest* HWC plan; once HWC soaks some, true curtailment shrinks.
  The existing DH snapshot + frequent replan handle this lag approximately — don't expect the
  budget to be a fixed point.
- **Federation blind spot.** A/C and the fan heaters claim the *same* curtailment budget and no
  controller sees the others. Total claims can exceed actual curtailment → a little import on a
  cheap surplus day. Acceptable for a federated v1; revisit if it bites. HWC's high value
  (useful, efficient heat) makes it a good priority claimant.
- **Connection limit** is auto-handled — the battery limits its own charge to respect 15 kW, and
  the 1800 W element is small.

## Phased plan

- **Phase 1 — executor `electric`/`performance` support + negative-price override.**
  **Implemented and enabled 2026-07-14** (brought forward because negative prices arrived early).
  What shipped:
  - `hwc_executor.Decision` gained `mode` (planned heating defaults to `actuation.operation_mode`)
    and `uses_compressor`; `apply_decision` commands `decision.mode`.
  - **`command_key` includes the mode.** It was `(action, setpoint)`, so an override that changed
    only the mode would have been dedup-skipped as "unchanged" and never actuated.
  - `hwc_negative_price.py` — the rule, the break-even, the window, the latch (pure functions).
  - `hwc_daemon` applies the override between `decide_current` and the actuation guards, watches
    the price entity (execute, not replan), and persists the latch in the daemon state file.
  - **Element-only heating is a compressor *stop*.** `min_off_seconds` no longer gates it (no
    compressor to rest), and it no longer feeds `last_heat_command_at` / `last_command_action` —
    those drive the start-grace in `effective_compressor_running` and the unconfirmed-start
    warning, both of which would otherwise wait on a confirmation that can never arrive. A
    separate `last_compressor_command_action` carries the compressor's view.
  - **SoC tracker** credits element heat at **COP 1** (`soc.step` applies the heat pump's COP to
    whatever power it is given, so the element is passed as a COP-equivalent power — exact in the
    build regime, the only one reachable below 60 °C). Otherwise the tracker would coast, and shed
    its draw prior, while the tank gained ~1800 W.
  - **Cycle reporting needs no change:** the reporter keys off compressor edges, and the
    compressor is off during `electric`, so no cycle opens and the COP stats stay clean.
- **Phase 2 — DP curtailment + element modelling.** Add the `electric` action, extend state to
  70 °C, ingest the curtailment forecast + Solcast PV-available + base-load series, apply
  curtailment-aware pricing with the budget + per-slot feasibility conditions. Reuses Phase 1's
  electric actuation. Target before spring.

## Data sources / entities

| Need | Source |
|---|---|
| Import price | already in the DP (`sensor.ai_dh_import_price_forecast` + MPC short-term) |
| Buy price (live, for override) | `sensor.amber_effective_general_price` (template; no attributes; conservative→confirmed twice per 5 min) |
| Forward prices (for `remaining_negative_window`) | `sensor.amber_billing_interval_forecasts_general_price` (`Forecasts`, 5-min resolution; config `amber_billing_entity`) |
| Curtailment budget | `sensor.mpc_p_pv_curtailment` (`forecasts` attribute series) |
| PV-available (potential, unclipped) | `sensor.solcast_pv_forecast_forecast_today` / `_tomorrow` (`pv_estimate`); EMHASS already nets this as `pv_net` |
| Base load (HWC-excluded) | the LGBM load forecast already produced for EMHASS |
| Actuation | `water_heater.aquatech` modes `heat_pump` / `electric` (`performance` deliberately unused for surplus) |
| Element circuit power (to verify 1800 W) | Athom ch2 (`sensor.athom_energy_monitor_02a3c8_..._power_2`) first time the element runs |

## Decisions settled (2026-06-28)

- Coordination style: **federated** (extend the existing pattern; no central broker).
- Negative price → HWC heats **unconditionally**, top priority. *(Mode choice amended
  2026-07-14 — see below; the original decision was unconditional `electric`.)*
- Surplus → **DP-planned** (not reactive heuristic); negative price → **executor override**.
- Surplus mode handling: **self-managed `heat_pump` + `electric`**, not `performance` (whose
  element is ungated and would import on modest-surplus days).
- Curtailment model: **budget (≤ 8 h, use-it-or-lose-it) + per-slot `PV_available > load + HWC`**.
- DP-vs-reactive for surplus: do the DP properly (Phase 2); skip the reactive-surplus interim.

## Decisions amended (2026-07-14)

Negative buy prices arrived earlier than expected, forcing Phase 1 to be built. Reviewing the
"unconditional `electric`" rule against a *running compressor* changed it:

- The 10 A plug makes the two heat sources **mutually exclusive** (~2500 W combined vs ~2400 W
  available), so `performance` is sequential and 1800 W is the unit's hard maximum draw.
- Interrupting a running compressor for the element **gains no heat** (1800 W element ≈ 1750 W
  thermal from the heat pump) — only Δ 1.1 kW of paid draw — while costing a restart.
- Therefore: compressor **off** → `electric` @ 70; compressor **running** → `performance` @ 70
  (uninterrupted to 60, element 60→70), **unless** the break-even
  `price < −transition_cost_aud / (1.1 × gain_hours)` passes, where
  `gain_hours = min(remaining_negative_window, time_to_60C)`.
- The `electric` switch is **latched** for the event; at/above 60 °C the daemon actively reverts
  to the DP plan only after the confirmed price is strictly positive (`performance`'s element
  leg is ungated and would import to setpoint).
- Still to verify: element draw really is 1800 W (Athom ch2, first time the element runs).
