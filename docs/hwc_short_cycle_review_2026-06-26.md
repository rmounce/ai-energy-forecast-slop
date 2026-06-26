# HWC short-cycling — diagnosis for independent review (2026-06-26)

**Audience:** a reviewer with no access to the repository. Everything needed to assess the
diagnosis — system description, the relevant code, the production log evidence, and a
reproduction — is embedded below.

**One-line summary:** the hot-water heat pump short-cycles (≈2 min on / ≈1 min off) when the
tank temperature sits on the `top_up_start_temp_c = 53 °C` regime boundary. It is a
*cross-replan limit cycle*: each individual plan is fine, but the dynamic-programming (DP)
planner's decision *for the current 5-minute slot* flips depending on whether the compressor
is currently on or off, and the executor has a minimum-on guard but no minimum-off guard, so
nothing damps the oscillation.

---

## Resolution (2026-06-26) — shipped after review

Reviewer and owner converged. **Option A (a probe-only continuous heat-rate curve) was rejected:**
the measured two-stage rate is stratification/state-of-charge driven, not a function of the
single mid-tank probe, so a memoryless `f(T_probe)` would discard validated physics. Two changes
shipped:

1. **Carried regime seed (the real fix).** The daemon now seeds the in-progress run's regime
   (`block_regime_full` = FULL iff the run started below `top_up_start_temp_c`, tracked as the
   coldest probe temp since it began) instead of letting the DP re-deduce it from the current
   probe. A cold-started run carried past 53 °C stays FULL, so "continue" is priced correctly and
   the artifact stop disappears. Falls back to the temp-based guess on a daemon restart mid-block.
2. **Symmetric `min_off_seconds` guard (hardware protection).** A `heat` within the minimum
   compressor rest after an `off` is suppressed — model-agnostic short-cycle protection. No strict
   min-*on*: deferring a stop would overshoot the setpoint, whereas deferring a start is safe.

A reproduction confirmed the seed flips the present-slot decision at the boundary, and the fix is
covered by unit tests (`test_dp_block_regime_full_seed_continues_through_boundary`,
`test_block_regime_*`, `test_*_min_off_*`).

**Deferred (agreed destination, not built):** a continuous **state-of-charge** tank model (2-node
or hot-fraction), with condensing/exhaust temperature as a *calibration/observation* signal (it is
only meaningful while the compressor runs, so it indexes charging, not the discharge/draw problem).
This dissolves the latent future-block discontinuity entirely; until then it is a logged known
limitation. The remaining minor inaccuracy: the published-temperature replay
(`simulate_block_temperatures`) still latches regime on block-start temp, so a carried-FULL run's
predicted temps are modelled slightly conservatively — pessimistic, not a control problem.

The sections below are the original diagnostic record as sent for review.

---

## 1. System under review

A residential heat-pump hot-water cylinder (Aquatech RAPID X6, 225 L) is scheduled by a
custom controller. The relevant pieces:

- **Planner** (`hwc_dp_planner.build_dp_plan`): a dynamic program that, every ≥60 s (and on
  any forecast/state change), chooses a binary compressor on/off sequence over a 576-step ×
  5-minute horizon (48 h). It minimises a single monetary objective:

  > `J = imported energy cost  +  transition_cost_aud per off→on start  +  soft penalties`

  Soft penalties enforce a min-temperature floor and a daily 60 °C legionella obligation.
  There is **no hard minimum-runtime constraint** — short cycles are discouraged *only* by a
  per-start `transition_cost_aud` (production value **0.05 AUD**). The DP only picks the
  on/off bits; the published temperatures come from an exact thermal replay.

- **Executor** (`hwc_executor.decide` + daemon loop): every ~60 s it looks at the *current*
  slot of the published plan and the *current* compressor state, and issues `heat` or `off`
  to Home Assistant.

- **Compressor-state seed**: the planner is told whether the compressor is currently running
  (`compressor_initially_on`). Continuing an already-running compressor is **free** (no
  transition cost); a fresh start costs `transition_cost_aud`.

### 1.1 The regime latch (key detail)

The heat-pump heats faster from cold than when topping up. The model captures this with a
**regime** that latches on the *block-start* temperature:

- Below `top_up_start_temp_c` (53 °C) at block start → **FULL** regime, ~6.6 °C/h.
- At/above 53 °C at block start → **TOP-UP** regime, ~5.5 °C/h.

Crucially the regime is *carried* for the duration of a heating block (a cold reheat keeps the
full rate even after it climbs past 53 °C). The DP therefore tracks regime as part of its
state, and seeds it from the current temperature **and** the current on/off state.

### 1.2 Relevant production config

```yaml
hwc:
  optimization_time_step: 5          # minutes
  transition_cost_aud: 0.05          # AUD per off->on start — the ONLY short-cycle deterrent
  thermal:
    min_temp: 45
    desired_temp: 60                 # daily legionella obligation
    max_temp: 60
    heat_rate_c_per_hour: 6.6        # FULL regime
    top_up_start_temp_c: 53.0        # regime boundary
    top_up_heat_rate_c_per_hour: 5.5 # TOP-UP regime
  daemon:
    minimum_replan_interval_seconds: 60
    execution_interval_seconds: 60
    heat_command_grace_seconds: 120  # minimum-ON guard (see §4)
```

---

## 2. Observed behaviour (production journal, 2026-06-26, tank pinned at 53.0 °C)

Times are local (ACST). `start_temp` is the tank temperature fed to the planner;
`compressor_on` is the seed; `starts`/`stops` count edges in the resulting 48 h plan.

```
11:42:24  HWC plan: start_temp=52.0°C, compressor_on=True
11:42:25  dp plan: starts=3, stops=3, objective=$0.929
11:42:26  executor: heat (inside planned block)            -> compressor running

11:43:26  HWC plan: start_temp=53.0°C, compressor_on=True
11:43:27  dp plan: starts=3, stops=4, objective=$0.954      <-- extra stop = close the
11:43:27  executor: off (outside planned block;                 in-progress run immediately
                         stopping running compressor)            and defer
11:43:27  HWC command: turn off water_heater.aquatech       -> compressor OFF

11:44:27  HWC plan: start_temp=53.0°C, compressor_on=False
11:44:28  dp plan: starts=3, stops=3, objective=$0.903
11:44:28  executor: heat (inside planned block)             -> compressor ON again
11:44:35  HWC command: set ... mode=heat_pump setpoint=60.0C

11:45:28  HWC plan: start_temp=53.0°C, compressor_on=True
11:45:29  dp plan: starts=3, stops=4, objective=$0.952
11:45:30  executor: off (stopping running compressor)
11:45:30  WARNING Suppressing HWC off command 52.3s after heat command   <-- min-ON guard
11:46:19  WARNING Suppressing HWC off command 101.6s after heat command
11:46:49  HWC command: turn off water_heater.aquatech       -> compressor OFF (after ~130 s)

11:47:34  HWC plan: start_temp=53.0°C, compressor_on=False
11:47:35  dp plan: starts=3, stops=3, objective=$0.903
          ... cycle repeats ...
```

The tell-tale pair, at an **identical 53.0 °C tank** and near-identical forecast:

| seed `compressor_on` | DP plan         | executor decision for **now** |
|----------------------|-----------------|-------------------------------|
| `True`  (running)    | starts=3 stops=4 | **off** — stop now, defer     |
| `False` (off)        | starts=3 stops=3 | **heat** — start now          |

So the present-slot action depends on the seed, and in the *destabilising* direction:
**being on makes it want to stop; being off makes it want to start.** That is anti-hysteresis,
and it sustains a limit cycle with period ≈ 2–3 min.

(The objective values across the two seeds are not directly comparable — they are evaluated
from different starting states — but the opposing *present action* is the problem.)

---

## 3. Root cause

### 3.1 Why the present action flips with the seed

Normally `transition_cost_aud` is *stabilising*: continuing is free, restarting costs money,
so once on you tend to stay on and once off you tend to stay off. Here that hysteresis is
**inverted** by the regime latch at the 53 °C boundary, where the tank was parked:

- **Seeded ON at 53.0 °C** → the in-progress block is latched **TOP-UP** (slow, 5.5 °C/h).
  Continuing is modelled as inefficient, so the DP prefers to **stop now** and restart later in
  a clean FULL-rate block — even though stopping+restarting costs a transition. The "slow
  continue" penalty exceeds the 0.05 transition cost.
- **Seeded OFF at 53.0 °C** → starting drops the temperature a hair below 53 (standing loss +
  draw within the first step), so `regime_for_start` returns **FULL** (fast, 6.6 °C/h).
  Starting now is modelled as efficient → the DP **heats now**.

The same 53.0 °C tank is therefore modelled as "slow if I'm already on" vs "fast if I start
fresh", which is precisely backwards for stability. The regime-latch penalty on *continue*
overwhelms the small transition cost and flips the optimal present action.

The DP code that produces this (annotated):

```python
# regime carried from the seed when already running; otherwise chosen from the (slightly
# cooled) post-step temperature t1.
if action_on:
    if on_prev and regime_prev != _OFF:
        regime = regime_prev                 # seeded-ON: carries TOP-UP at 53 °C
    else:
        regime = regime_for_start(t1)        # seeded-OFF: t1 < 53 -> FULL
    rate = _rate_for_regime(th, regime, wbp, t1)
    t_next = min(max_temp, t1 + rate * step_h)
    ...
    trans = 0.0 if on_prev else transition_cost   # continue free, restart costs 0.05
```

```python
def regime_for_start(t1: float) -> int:
    if top_up_start is not None and t1 >= top_up_start:
        return _TOPUP
    return _FULL
```

### 3.2 Why nothing damps it

The executor (daemon) has a **minimum-ON guard** but **no minimum-OFF guard**. It suppresses
an `off` issued shortly after a `heat` (to bridge a ~50 s lag in the Tuya compressor sensor on
start), but the `off→heat` edge is completely unguarded:

```python
def should_suppress_off_after_heat(*, decision_action, now, last_heat_command_at,
                                   grace_seconds, compressor_on):
    if decision_action != "off":
        return False        # <-- only 'off' is ever suppressed; 'heat' is never debounced
    if compressor_on:
        return False
    if last_heat_command_at <= 0:
        return False
    return now - last_heat_command_at < grace_seconds   # grace_seconds = 120
```

Consequence: the ON pulse is bounded to ~120 s by the guard, but the moment the guard lapses
and the next replan seeds `off`, the executor immediately re-issues `heat`. The cycle's
frequency is unbounded from the off side.

### 3.3 Reproduction (no proprietary data needed)

Running the production planner over a synthetic 48 h horizon (price cheap now → expensive
later, matching the logged `$0.136 → $1.0/kWh`), sweeping start temperature and seed:

```
start=52.0C  seed_off: t0=HEAT starts=3   seed_on: t0=HEAT starts=3
start=52.5C  seed_off: t0=HEAT starts=3   seed_on: t0=HEAT starts=3
start=53.0C  seed_off: t0=HEAT starts=3   seed_on: t0=HEAT starts=3
start=53.5C  seed_off: t0=off  starts=3   seed_on: t0=HEAT starts=3   <-- FLIP
start=54.0C  seed_off: t0=off  starts=3   seed_on: t0=HEAT starts=3   <-- FLIP
```

The seed-dependent flip appears within ±1 °C of the 53 °C boundary. Its exact polarity and
temperature shift with the precise price/draw shape (in this synthetic it lands the *other*
way — seed-on heats, seed-off defers — which would be self-correcting; in the production run
it landed on the destabilising side). The robust, reproducible fact is: **near the regime
boundary the present-slot decision is seed-dependent**, i.e. the model has a knife-edge there.
Whether a given day's forecast tips it to the stabilising or destabilising side is luck.

---

## 4. Proposed remedies (for the reviewer to weigh in on)

Listed in the implementer's order of preference. We have **not** implemented any of these yet —
this document is for review first.

1. **Remove the regime knife-edge (root-cause fix, smallest behavioural change).**
   The "continue vs fresh-start" asymmetry at the boundary is arguably a *modelling artefact*:
   a compressor that is genuinely mid-block at 53 °C is not really slower than one that starts
   fresh at 53 °C — the latch exists to model a *cold* reheat carrying its momentum, not to
   penalise a warm continue. Make the present-slot regime evaluation continuous across the seed
   so 53.0 °C yields the same t0 action whether on or off (e.g. evaluate "continue" at the same
   effective rate a fresh start would get at that temperature, or hysteresis-band the boundary).
   *Risk:* must not regress the legitimate cold-reheat momentum behaviour the latch was added
   for.

2. **Add a symmetric minimum-off guard in the executor (cheap safety net).**
   Mirror `should_suppress_off_after_heat` with a `suppress_heat_after_off` debounce so the
   compressor cannot restart within N minutes of stopping. Bounds the cycle to the heat pump's
   safe duty period regardless of planner behaviour. *This is a band-aid* — the tank still
   toggles, just slower — but it is low-risk and protects the hardware while (1) is evaluated.

3. **Commitment / stickiness on the current compressor state.**
   Forbid a replan from reversing the *current* physical state unless the modelled improvement
   exceeds `transition_cost_aud`. This neutralises *any* anti-hysteresis source, not just the
   regime latch. *Risk:* adds state/latch to the controller; slightly more complex; needs care
   so it cannot defer a genuinely needed legionella heat.

4. **Raise `transition_cost_aud` (blunt instrument).**
   A larger per-start cost would swamp the regime-latch penalty and restore net hysteresis.
   *Risk:* globally distorts the economic optimisation (suppresses legitimate price-arbitrage
   splits), and only raises the bar the artefact must clear rather than removing it.

The implementer's recommendation: **(1) as the real fix, with (2) as a cheap concurrent safety
net.** Open question for the reviewer: is the regime latch's "warm continue = slow" treatment
ever physically correct (in which case (1) needs to preserve it for true cold reheats), or is
it purely an artefact of latching on a single block-start temperature?

---

## 5. Questions for the reviewer

1. Is the regime-latch flip (§3.1) the right root cause, or is there a simpler explanation for
   the seed-dependent present-slot decision we've missed?
2. Of the four remedies, which best fits a controller whose design goals are *lean,
   slightly-conservative, minimal config knobs*?
3. Is a minimum-runtime concept (hard constraint or commitment latch) worth introducing at all,
   given the explicit 2026-06 design decision to rely solely on `transition_cost_aud`?
```
