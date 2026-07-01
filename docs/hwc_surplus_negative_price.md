# HWC surplus & negative-price strategy

How the heat-pump hot-water (HWC) unit should exploit **PV surplus / curtailment** and
**negative grid buy prices**, and how that fits the existing flexible-load controllers
(battery, A/C, dump-load fan heaters). Design + phased plan. Read alongside
`docs/hwc_handover.md`, `docs/hwc_dp_planner.md`, `docs/hwc_aquatech_entities.md`.

Status: **design agreed 2026-06-28, not yet implemented.** Neither event fires in winter, so
there is no rush; Phase 2 has runway to be done properly rather than as a stopgap.

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

- **Negative price:** HWC resistive element is the **top** claimant — the element turns on
  **unconditionally** (no available-headroom margin required). The element is only 1800 W and
  the battery auto-limits its charge to respect the 15 kW ceiling, so simultaneous draw never
  trips the limit; no special battery coordination is needed.
- **Surplus:** HWC ranks above the fan heaters (useful product, COP-efficient) and is fed
  *upstream* of the battery — HWC is planned first and fed into EMHASS as load, so the battery
  mops up whatever surplus remains (it "reshapes" around HWC).

## Actuation primitive (Aquatech RAPID X6 modes)

`water_heater.aquatech` operation modes (HA accepts setpoints 15–75 °C):

- **`heat_pump`** — heat pump only, ~700 W, capped at 60 °C. COP ~2.4–3.0 (the 55→60 tail is
  ~1.75). Today's only mode.
- **`electric`** — resistive element only, **1800 W, COP 1**, available at any temp up to 75 °C.
- **`performance` (Hybrid+)** — heat pump to 60 °C, then element 60→75 (default 70).
  **Element portion is ungated** (runs to setpoint regardless of available surplus), so we
  **do not** use it for surplus — see below. Self-managing `heat_pump` + `electric` gives the
  curtailment-gated control we want.

Heat rates are similar: 1800 W element ≈ 6.9 °C/hr for 225 L; the ~700 W heat pump at COP ~2.5
delivers ~1750 W thermal ≈ similar. The element is not *faster* — it is *available above 60*
and *cheaper to start* (no compressor short-cycle constraint).

The executor currently issues only `heat_pump` / `off` (`setpoint_max_c: 60`). **Learning
`electric` mode + setpoint-to-75 is shared groundwork for both phases.**

## Design — negative price (executor override, reactive)

Lives **inside `hwc_daemon`** as a layer on top of the DP plan (not a separate HA automation —
a second controller fighting for `water_heater.aquatech` is the one thing guaranteed to
misbehave). Externally still federated (keys off the same `amber_effective_general_price`).

- Live `amber_effective_general_price < 0` → force **`electric` to 75 °C**, unconditional, top
  priority, overriding the DP's plan for that interval.
- `electric` (not `performance`): below 60 °C, electric draws the full 1800 W while performance
  would draw only the 700 W heat pump — electric *dumps more*, which is the point.
- Forecast negative prices (occasional) get optimal DP treatment automatically (price < 0 →
  negative cost → DP runs the element hard); the override is purely the safety net for the
  *un*forecast ones.

## Design — surplus / curtailment (DP, forecast-driven)

Model curtailment in the DP so the optimal plan emerges rather than being hand-coded. Most of
the earlier reactive thresholds (2 kWh / 0.5 kWh / instantaneous gates) **disappear** — they
become emergent from the DP's costs.

**Actions per step:** `{off, heat_pump, electric}`.
- `heat_pump`: ~700 W, available < 60 °C, compressor start charged `transition_cost_aud` (0.05).
- `electric`: 1800 W, available up to 75 °C, **start cost ~0** (no short-cycle constraint).
- This reproduces "compressor commits, element bang-bangs" *emergently* — no thresholds.

**State range:** extend 45 → **75 °C** (was capped at 60).

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

- **Phase 1 — executor `electric` support + negative-price override.** Small, and the genuinely
  reactive path. Executor learns `electric` mode + setpoint-to-75; daemon adds the live
  `price < 0 → electric to 75` override on top of the DP plan. Ships the surprise-handler.
- **Phase 2 — DP curtailment + element modelling.** Add the `electric` action, extend state to
  75 °C, ingest the curtailment forecast + Solcast PV-available + base-load series, apply
  curtailment-aware pricing with the budget + per-slot feasibility conditions. Reuses Phase 1's
  electric actuation. Target before spring.

## Data sources / entities

| Need | Source |
|---|---|
| Import price | already in the DP (`sensor.ai_dh_import_price_forecast` + MPC short-term) |
| Buy price (live, for override) | `sensor.amber_effective_general_price` |
| Curtailment budget | `sensor.mpc_p_pv_curtailment` (`forecasts` attribute series) |
| PV-available (potential, unclipped) | `sensor.solcast_pv_forecast_forecast_today` / `_tomorrow` (`pv_estimate`); EMHASS already nets this as `pv_net` |
| Base load (HWC-excluded) | the LGBM load forecast already produced for EMHASS |
| Actuation | `water_heater.aquatech` modes `heat_pump` / `electric` (`performance` deliberately unused for surplus) |
| Element circuit power (to verify 1800 W) | Athom ch2 (`sensor.athom_energy_monitor_02a3c8_..._power_2`) first time the element runs |

## Decisions settled (2026-06-28)

- Coordination style: **federated** (extend the existing pattern; no central broker).
- Negative price → **`electric` to 75**, unconditional, top priority.
- Surplus → **DP-planned** (not reactive heuristic); negative price → **executor override**.
- Surplus mode handling: **self-managed `heat_pump` + `electric`**, not `performance` (whose
  element is ungated and would import on modest-surplus days).
- Curtailment model: **budget (≤ 8 h, use-it-or-lose-it) + per-slot `PV_available > load + HWC`**.
- DP-vs-reactive for surplus: do the DP properly (Phase 2); skip the reactive-surplus interim.
