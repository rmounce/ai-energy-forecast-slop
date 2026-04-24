# Fresh Independent Strategy Review Brief — 2026-04-24

Purpose: provide a clean, neutral orientation for an independent expert reviewing the
overall system strategy after the recent Track 10A handoff and dynamic-bridge work.

This brief is intentionally compact. It is meant to help the reviewer get their bearings
quickly, then read the repo directly and form their own view.

Primary repo entry points:
- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [docs/roadmap.md](./roadmap.md)
- [docs/data_sources.md](./data_sources.md)
- [eval/README.md](../eval/README.md)
- [forecast.py](../forecast.py)
- [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)
- [docs/training_runs.md](./training_runs.md)

Recent review / experiment notes:
- [docs/independent_review_brief_2026-04-21.md](./independent_review_brief_2026-04-21.md)
- [docs/holistic_independent_review_brief_2026-04-22.md](./holistic_independent_review_brief_2026-04-22.md)
- [docs/dynamic_bridge_results_2026-04-24.md](./dynamic_bridge_results_2026-04-24.md)
- [docs/codex_review_response_2026-04-23.md](./codex_review_response_2026-04-23.md)

---

## 1. Review Goal

Please review the whole strategy, not just the latest live implementation question.

The objective is:

- maximize real financial performance of the battery system under live operating conditions

There is one important secondary preference:

- if possible, maintain a credible path that does not depend on Amber APF as a core input

That preference is real, but it is **not** a hard requirement. If the best-performing
architecture still depends on Amber APF, that is still a valid conclusion.

Assumptions that should be treated as challengeable:
- the current two-timescale decomposition
- the current forecast horizons / resolutions
- the current use of EMHASS and the present optimizer contract
- the current evaluation framing
- the current emphasis on forecast-path improvement relative to control-contract redesign

The reviewer should feel free to reassess earlier design choices, not only the latest open
question.

---

## 2. Real-World Setting

The system controls a home battery in South Australia using:
- AEMO market data and forecasts
- household load / PV / weather data
- Home Assistant orchestration
- EMHASS optimization

The current production-like picture has:
- a strategic / longer-horizon planning component
- a tactical / short-horizon execution component

The repo currently expresses that as:
- `30m / 72h` strategic forecasting
- `5m / 60min` tactical forecasting
- `72h × 30m` day-ahead optimization
- `14h × 5m` rolling MPC optimization

That decomposition is current practice, not a protected truth.

---

## 3. Hard Constraints

The key hard constraints are the available external data products and their timing.

Important inputs include:
- confirmed current dispatch price
- **P5MIN**
- **PREDISPATCH**
- **PD7Day**
- **SevenDayOutlook**
- dispatch actuals
- local load / PV / weather

The important question is not only what data exists, but:
- when each source arrives
- how stale it is when decisions are made
- how often the controller can realistically replan
- whether the current architecture makes best use of those arrival patterns

See:
- [docs/data_sources.md](./data_sources.md)
- [ARCHITECTURE.md](../ARCHITECTURE.md)

---

## 4. Current System Shape

### Forecasting

The repo currently has:
- **Tier 1 tactical LightGBM** at `5m / 60min`
- **Tier 2 TFT** at `30m / 72h`
- recent enhanced-input TFT work using richer market decoder inputs such as PREDISPATCH and PD7Day

### Optimization / execution

The current live-control picture uses EMHASS for:
- day-ahead `72h × 30m`
- rolling MPC `14h × 5m`

Operationally, the strategic layer is currently described as influencing the tactical layer
through a SoC target around the `14h` boundary, while the tactical layer replans frequently
using:
- confirmed current interval price
- fresh short-horizon forecasts
- real-time local load / PV conditions

### Evaluation

The repo currently contains:
- one-shot holistic financial eval
- tactical forecast / dispatch eval
- rolling MPC eval

The rolling MPC work is now the most important live architectural evaluation track.

See:
- [eval/README.md](../eval/README.md)
- [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

---

## 5. What Seems More Established

The repo currently treats the following as reasonably supported, though still reviewable:

- financial evaluation matters more than forecast error metrics alone
- a short-horizon tactical layer is useful
- long-horizon awareness also matters
- rolling evaluation is more representative than one-shot horizon evaluation for the intended
  MPC use case
- the contract between long-horizon value and short-horizon execution matters materially

---

## 6. What Is Still Open

These should be treated as live design questions:

- whether the current two-tier decomposition is optimal
- whether the tactical horizon should remain `60min` forecast + `14h` MPC
- whether the strategic layer should primarily produce a detailed `72h` path, or a different
  object such as:
  - a terminal target / target band
  - an opportunity-cost signal
  - a reserve / posture signal
  - a richer value function
- whether EMHASS should remain the core optimizer unchanged
- whether the current evaluation stack is steering the project toward local optima
- whether current effort has been too focused on improving forecast paths instead of changing
  the strategic-to-tactical contract

---

## 7. Most Relevant Recent Findings

### 7.1 Handoff mattered

Rolling Track 10A work showed that restoring the strategic handoff materially changed the
evaluation picture. The hybrid remained behind Amber on the important `2025-09-01 -> 2025-10-13`
Window B slice, but the gap narrowed relative to the earlier no-handoff setup.

### 7.2 Naive path tilt failed

The first fixed q50->q90 blend sweep was strongly negative. Treating the whole future path as
more conservative was too blunt.

### 7.3 First bridge-contract variants were active but inert

The first dynamic bridge runs on the 6-week Window B slice:
- dynamic terminal value under `exact`
- dynamic upward band without a value signal

produced the same economics and the same tactical dispatch path as the handoff-enabled baseline.

The current interpretation in the repo is that this was a formulation lesson, not a blanket
rejection of bridge contracts.

### 7.4 Short pilots narrowed the bridge space further

Short 2-day pilots over `2025-09-01 -> 2025-09-03` then tested:
- `floor + dynamic target uplift`
- `band + terminal(all)`
- same-window `exact`
- `band + terminal_scope=extra_band`

Findings:
- all of these pilot variants produced the same first-step tactical dispatch on that slice
- the bridge metadata changed
- even the more selective `extra_band` formulation, which values only the energy above the q50
  floor, did not change realized tactical action relative to the same-window `exact` baseline

Current repo reading:
- bridge-only tuning is looking increasingly diagnostic rather than clearly promising
- one stronger short-window probe may still be defensible
- but the project may be close to the point where it should pivot from bridge tuning to
  diagnosing the residual Amber gap more directly

See:
- [docs/dynamic_bridge_results_2026-04-24.md](./dynamic_bridge_results_2026-04-24.md)

---

## 8. Current Strategic Tension

The central tension is still:

- the system needs long-horizon awareness so it can preserve inventory for downstream value
- the tactical layer also needs to react to fresher, higher-frequency information

The repo is currently asking whether the remaining problem is mainly:
- forecast quality
- control-contract quality
- optimizer timing / cadence
- or some deeper architecture mismatch

The reviewer should feel free to step back from that framing too, and ask whether the system
is solving the right intermediate problem at all.

---

## 9. What Review Would Be Most Valuable

The most useful review would address questions like:

1. Is the current decomposition of:
   - `30m / 72h` strategic forecast
   - `5m / 60min` tactical forecast
   - `14h × 5m` tactical MPC
   actually the right structure for the live objective?

2. Is the current way that long-horizon information influences tactical execution the best
   available contract, or just a workable intermediate solution?

3. Given the real external data timing constraints, is there a better control formulation?

4. Is the current rolling eval sufficiently aligned with production intent to guide the
   architecture, or are there still hidden mismatches?

5. If redesigning from scratch under the same HA / EMHASS / AEMO environment, would you:
   - keep this architecture,
   - adapt it,
   - or replace it?

6. Are the repo’s current local optima more likely to be in:
   - forecast-path design,
   - strategic-to-tactical contract design,
   - optimization design,
   - timing / orchestration,
   - or evaluation framing?

7. What are the highest-value next experiments to separate:
   - better forecast architecture
   - better control contract
   - better tactical optimizer behavior
   - better system-level timing

---

## 10. Suggested Review Path

If helpful, a reasonable order is:

1. [ARCHITECTURE.md](../ARCHITECTURE.md)
2. [docs/data_sources.md](./data_sources.md)
3. [docs/roadmap.md](./roadmap.md)
4. [eval/README.md](../eval/README.md)
5. [forecast.py](../forecast.py)
6. [eval/rolling_mpc_eval.py](../eval/rolling_mpc_eval.py)

Then, for recent context:
- [docs/dynamic_bridge_results_2026-04-24.md](./dynamic_bridge_results_2026-04-24.md)
- [docs/holistic_independent_review_brief_2026-04-22.md](./holistic_independent_review_brief_2026-04-22.md)
- [docs/codex_holistic_review_draft_2026-04-22.md](./codex_holistic_review_draft_2026-04-22.md)
