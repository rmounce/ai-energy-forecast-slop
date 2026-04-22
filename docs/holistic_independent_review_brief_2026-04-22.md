# Holistic Independent Review Brief — 2026-04-22

Purpose: orient an external reviewer to the repo and the current decision space at a
system level.

This brief is intentionally short and neutral. It is not meant to argue for a particular
architecture. The reviewer is expected to read the repo directly and form their own view.

Related repo entry points:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/roadmap.md](./roadmap.md)
- [docs/data_sources.md](./data_sources.md)
- [eval/README.md](../eval/README.md)
- [docs/training_runs.md](./training_runs.md)
- [forecast.py](../forecast.py)
- [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

---

## 1. Review Goal

Please review the system holistically and challenge any assumptions that look like local
optima within the current design space.

The objective is simple:

- maximize **real financial performance** of the battery system across all relevant market
  conditions

There is one important secondary preference:

- if possible, achieve that performance **without depending on Amber's APF as a core input**

This is a strong design preference, but **not** a hard requirement. If a reviewer concludes
that the best-performing architecture should still depend on Amber APF, that is still a valid
answer.

This includes, but is not limited to:
- forecast architecture
- optimizer / execution architecture
- control timing and handoff design
- evaluation methodology
- whether the current decomposition into tiers is actually the right one

Existing decisions should be treated as revisable unless they are forced by external
constraints.

This includes not just the latest active questions, but also earlier framing choices that the
repo may currently take for granted.

---

## 2. Real-World Problem Setting

The system controls a home battery in South Australia using:
- AEMO price and market-forecast inputs
- household load / PV / weather data
- EMHASS for optimization
- Home Assistant automations for orchestration and actuation

The deployed control problem has two different practical timescales:
- a **strategic / longer-horizon** planning problem
- a **tactical / short-horizon** execution problem

The repo currently expresses this as:
- `30m / 72h` strategic forecasting
- `5m / 60min` tactical forecasting
- `72h × 30m` day-ahead optimization
- `14h × 5m` rolling MPC optimization

That decomposition is current practice, not a protected truth.

The reviewer should feel free to question, for example:
- whether those are the right horizons and resolutions
- whether the current split between forecasting and control is the right one
- whether the problem should instead be formulated around a different state / action /
  boundary-value contract
- whether the current use of EMHASS is ideal, merely convenient, or actively constraining
  better outcomes

---

## 3. Hard External Constraints

The main hard constraints are external data products and their timing.

Key AEMO inputs currently used:
- **P5MIN**: every 5 minutes, short-horizon signal
- **PREDISPATCH**: every 30 minutes
- **PD7Day**: 3 times per day
- **SevenDayOutlook**: every 30 minutes
- dispatch actuals

Key local / HA inputs:
- household load
- PV generation
- weather
- Amber price feeds / existing production forecast logs

The important question is not only "what data exists?" but:
- when each source becomes available
- how stale it is when decisions are made
- how often the controller can realistically replan
- whether the current architecture makes the best use of those arrival times

See:
- [docs/data_sources.md](./data_sources.md)
- [ARCHITECTURE.md](../ARCHITECTURE.md)

---

## 4. Current System Shape

### Forecasting

The repo currently has:
- **Tier 1 tactical LightGBM** for short-horizon `5m / 60min`
- **Tier 2 TFT price model** for `30m / 72h`
- an archived **strategic LightGBM** exploration that did not become the preferred path

There is also active / recent work on enhanced-input TFT variants using richer market
decoder inputs such as PREDISPATCH and PD7Day.

### Optimization / execution

The live control stack uses EMHASS for:
- day-ahead `72h × 30m`
- rolling MPC `14h × 5m`

Operationally, the current production picture is that the strategic layer influences the
tactical layer through a **SoC target around the 14-hour boundary**, while the tactical
layer responds more frequently to:
- current confirmed price
- fresh short-horizon price forecasts
- real-time load / PV conditions

### Evaluation

The repo currently contains multiple evaluation layers:
- one-shot holistic dispatch simulation
- tactical 5-minute accuracy / dispatch eval
- rolling MPC evaluation

The rolling MPC work is currently the most important live architectural evaluation track.

See:
- [eval/README.md](../eval/README.md)
- [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

---

## 5. What Seems Established vs What Is Still Open

### More established

The repo currently treats the following as reasonably supported:
- financial evaluation matters more than pure forecast error metrics
- a tactical short-horizon model is useful
- a longer-horizon planning model is useful
- rolling evaluation is more representative than one-shot horizon evaluation for the
  intended MPC use case
- the exact interaction between long-horizon and short-horizon layers matters materially

Even these should be treated as reviewable conclusions rather than axioms if the reviewer
thinks the repo evidence points elsewhere.

### Still open

The following should be treated as open design questions:
- whether the current two-tier decomposition is optimal
- whether the current tactical horizon is the right one
- whether long-horizon opportunity cost should be expressed through:
  - state handoff,
  - objective bias,
  - quantile / risk-policy tilt,
  - a value function / water-value style contract,
  - or some other mechanism
- whether the present rolling evaluation setup is fully aligned with production intent
- whether current optimizer timing / refresh cadence is the right use of available data
- whether EMHASS should remain the core optimization engine unchanged, or be wrapped /
  modified more substantially
- whether the present emphasis on forecast-layer iteration has been too strong relative to
  control-layer or timing-layer redesign
- whether any current evaluation objective, baseline choice, or data slice is steering the
  project toward a local maximum

---

## 6. Current Architectural Tension

The central current tension is roughly this:

- the system wants long-horizon awareness so it can preserve energy for downstream value
- but it also wants very fresh tactical execution using higher-frequency information

Recent work in the repo suggests that:
- restoring strategic boundary-state information into the rolling MPC eval matters
- but it may not be sufficient on its own
- there may still be residual execution-policy or forecast-posture weaknesses on ordinary
  days even after a more production-aligned handoff is restored

The repo has explored several mechanisms in that area, but none should be assumed to be the
final answer.

More broadly, the reviewer should feel free to step back from this tension entirely and ask
whether the system is solving the right intermediate problem.

---

## 7. What Review Would Be Most Valuable

The most useful review would address questions like:

1. Is the current overall decomposition of:
   - `30m / 72h` strategic forecast
   - `5m / 60min` tactical forecast
   - `14h × 5m` tactical MPC
   actually the right structure for the objective?

2. Is the present way that long-horizon information influences tactical execution the best
   available contract, or just a workable intermediate solution?

3. Is there a better way to formulate the control problem given the true external data
   timing constraints?

4. Are any of the current evaluation loops likely to bias design decisions toward local
   maxima that would not hold in live operation?

5. If you were redesigning the system from scratch using the same external inputs and the
   same HA / EMHASS environment, would you keep this architecture, adapt it, or replace it?

6. What are the highest-value next experiments to distinguish:
   - better forecast architecture
   from
   - better execution policy
   from
   - better system-level timing / handoff design?

7. Which currently accepted repo assumptions are most likely to be wrong, incomplete, or
   overfit to the current way the project has been partitioned?

8. Are there simpler formulations that would likely outperform the current direction even if
   they require backing out some recent work?

---

## 8. Suggested Review Path Through The Repo

If helpful, a reasonable order is:

1. [ARCHITECTURE.md](../ARCHITECTURE.md)
2. [docs/data_sources.md](./data_sources.md)
3. [docs/roadmap.md](./roadmap.md)
4. [eval/README.md](../eval/README.md)
5. [forecast.py](../forecast.py)
6. [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

Optional deeper context:
- [docs/training_runs.md](./training_runs.md)
- [docs/tft_price_forecast.md](./tft_price_forecast.md)
- [docs/track10a_handoff_analysis_2026-04-22.md](./track10a_handoff_analysis_2026-04-22.md)
- [docs/option_b_plan_2026-04-22.md](./option_b_plan_2026-04-22.md)

---

## 9. Requested Reviewer Mindset

Please do not assume that:
- the current stratification scheme is sacred
- the current tiering is sacred
- the current optimizer contract is sacred
- the current direction of travel is necessarily correct
- the current roadmap phases imply the right architectural boundaries
- the current use of EMHASS, Home Assistant timing, or forecast publication cadence is
  already close to optimal

The main ask is:

Given the real external inputs, their timing, and the operational environment in this repo,
what system design would you back if the objective were simply **best financial performance
in live operation across changing conditions**, with a secondary preference to reduce or
eliminate dependence on Amber APF if that can be done without sacrificing too much
performance?
