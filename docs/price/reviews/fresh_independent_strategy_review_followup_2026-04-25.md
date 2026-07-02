# Fresh Independent Strategy Review Follow-Up — 2026-04-25

Purpose: provide a concise follow-up after the first production-fidelity rolling-eval upgrade.
This is meant to brief an external reviewer on what changed since the previous strategy review,
what is now established, and what questions remain genuinely open.

This note assumes the reviewer can inspect the repo and prior briefs. It is not intended to
duplicate all earlier background.

Related documents:
- [fresh_independent_strategy_review_brief_2026-04-24.md](./fresh_independent_strategy_review_brief_2026-04-24.md)
- [rolling_eval_fidelity_pilot_2026-04-25.md](./rolling_eval_fidelity_pilot_2026-04-25.md)
- [rolling_eval_fidelity_full_windows_2026-04-25.md](./rolling_eval_fidelity_full_windows_2026-04-25.md)
- [roadmap.md](./roadmap.md)
- [eval/README.md](../eval/README.md)

---

## 1. What Changed Since The Last Review

Two important implementation changes landed:

1. The AI combined forecast publisher now expands Tier 2 `30m` steps into `5m` forecast items,
   removing a known integration mismatch between published combined sensors and the MPC-side
   timebase.

2. Track 10A rolling eval now supports a more production-like economic mode:
   `netload_tariffed`
   - actual `30m` load/PV expanded to `5m` net load
   - separate tariffed import and feed-in price curves
   - same rolling `14h x 5m` MPC and SoC carryover
   - same current-interval price treatment
   - same strategic handoff semantics

This created the first meaningful comparison between the legacy `price_only` architecture gate
and a closer-to-production rolling gate.

Important caveat:
- the current tariffed runs use the locally available **Run 014 Phase 7** TFT asset
- repo docs still record **Run 011b + binary routing** as the evaluated incumbent
- so the results below should be interpreted as **provisional / artifact-limited** until the
  strongest intended incumbent asset is recovered or reproduced under the same tariffed gate

---

## 2. Main New Result

Under the old `price_only` gate:
- Window A had looked favorable to the hybrid stack
- Window B had looked moderately unfavorable but not catastrophic

Under the newer `netload_tariffed` gate:
- Amber beats the hybrid on the initial `2-day` pilot
- Amber beats the hybrid on the `7-day` follow-up
- Amber beats the hybrid on full Window B (`2025-09-01 -> 2025-10-13`)
- Amber beats the hybrid on full Window A (`2025-07-21 -> 2025-09-01`)

Headline numbers:

### 2-day pilot (`2025-09-01 -> 2025-09-03`)
- `price_only`: hybrid **+2.6%**
- `netload_tariffed`: hybrid **-6.3%**

### 7-day tariffed follow-up (`2025-09-01 -> 2025-09-08`)
- Amber: **$1.476/day**
- Hybrid: **$0.952/day**
- hybrid vs amber: **-35.5%**

### Full Window B
- `price_only` exact handoff refresh:
  - Amber: **$2.451/day**
  - Hybrid: **$2.271/day**
  - hybrid vs amber: **-7.4%**
- `netload_tariffed` exact handoff:
  - Amber: **$0.427/day**
  - Hybrid: **$0.298/day**
  - hybrid vs amber: **-30.2%**

### Full Window A (`netload_tariffed`)
- Amber: **-$0.570/day**
- Hybrid: **-$0.901/day**
- hybrid vs amber: **-58.1%**

So the stronger gate does not just narrow the hybrid’s edge. It changes the overall
architecture reading across both major windows for the currently tested Run 014-based local
setup.

---

## 3. Important Interpretation

The tariffed mode is not just a different accounting layer.

Comparing the same 2-day exact-handoff slice under `price_only` vs `netload_tariffed` showed
materially different control actions:
- `soc_kwh`: `892` changed rows
- `discharge_kw`: `295` changed rows
- `charge_kw`: `289` changed rows

So the more production-like economic objective changes tactical behavior itself, not just the
score assigned to a fixed policy.

This matters because it means older `price_only` conclusions are not reliable architecture
gates once site economics are modeled more faithfully. It does **not** yet mean the best
documented AEMO-native incumbent has been defeated under the tariffed gate, because that
incumbent asset is not currently present locally.

---

## 4. What The New Diagnostics Suggest

The new tariffed diagnostics do **not** point to a single simple failure mode, but they do
narrow the space.

### Window B diagnostic read

Under full-window `netload_tariffed`:
- Amber earns about **$8.0** more export revenue
- import cost is almost the same
- Hybrid finishes with **less** final stored energy, not more (`0.0 kWh` vs `9.1 kWh`)

So on Window B, the hybrid does not appear to be losing because it held too much inventory to
the end. The larger signal is weaker realized export monetization.

Regime view for Window B:
- `low`: Hybrid worse
- `normal`: Hybrid worse
- `spike`: Hybrid slightly better

That is important because it suggests the residual gap is not “Amber only wins on spike days.”
The bigger weakness appears to be on the more common low/normal days.

### Window A diagnostic read

Under full-window `netload_tariffed`:
- Amber again wins mostly on export monetization
- Hybrid is somewhat better on import/degradation in places, but not enough to offset the
  realized spread gap

Regime view for Window A:
- `low`: Hybrid slightly worse
- `normal`: Hybrid slightly better
- `spike`: Hybrid much worse

So Window A and Window B fail differently:
- Window B gap is more concentrated in `low` and `normal`
- Window A gap is more concentrated in `spike`

This makes the control problem look more like a *family* of monetization failures than one
single posture bug.

---

## 5. What Now Looks Less Likely

The following now look less likely as primary explanations:

1. “The main issue is just the missing strategic handoff.”
   The exact-handoff runs still lose under the tariffed gate.

2. “The main issue is just terminal inventory preservation.”
   On full Window B, Hybrid ends with *less* final stored energy than Amber.

3. “Another round of generic 72h forecast-path tuning should be the next priority.”
   The more production-like rolling gate suggests the larger bottleneck is tactical monetization
   under realistic site economics.

4. “Bridge metadata activation is meaningful by itself.”
   Earlier bridge experiments activated metadata without changing dispatch, and bridge tuning is
   currently not the main uncertainty.

---

## 6. Current Best Read

The repo direction still looks directionally plausible:
- AEMO-native tactical + strategic stack
- rolling `14h x 5m` MPC
- less dependence on Amber over time

But the main bottleneck now appears more like:
- tactical control / inventory monetization under realistic site economics

than:
- one more strategic path variant

In other words:
- the production-real gate strengthened the case for making rolling MPC the primary architecture
  gate
- it weakened the case for continuing to treat `price_only` or pure long-horizon forecast-path
  improvement as the main roadmap driver
- it also introduced an asset-selection caveat: interpretation should stay provisional until the
  strongest intended incumbent TFT checkpoint is re-run

---

## 7. Questions For Review

The following are the main questions where outside judgment would be useful:

1. Given that the tariffed gate changes both dispatch and ranking, should `netload_tariffed`
   now fully replace `price_only` as the main architecture gate, or should both continue in
   parallel for a while?

2. How should the mixed failure pattern be interpreted?
   - Window B: weakness on `low` and `normal`
   - Window A: weakness on `spike`

   Does that pattern point more toward:
   - tactical objective misalignment,
   - forecast calibration by regime,
   - execution timing,
   - or some interaction of all three?

3. What is the highest-value next intervention under this evidence?
   Options might include:
   - a bounded water-value / inventory-value signal
   - a different tactical objective formulation
   - explicit discharge/posture shaping
   - better tactical forecast conditioning
   - some other execution-policy structure

4. Does the current evidence argue for staying within the existing EMHASS-style LP + handoff
   framing, or is this now strong enough evidence to explore a more structural tactical-control
   redesign?

5. What is the cleanest way to separate:
   - “Amber monetizes inventory better because its forecast shape is better”
   from
   - “Amber monetizes inventory better because the current tactical controller is using future
     inventory value poorly”?

---

## 8. Suggested Reviewer Lens

If helpful, the strongest current evidence to inspect is:

- the tariffed full-window summaries:
  - `rolling_mpc_eval_tracka_followup_6week_netload_exact_20260425_*`
  - `rolling_mpc_eval_tracka_windowa_6week_netload_exact_20260425_*`
- the diagnostics CSVs:
  - `*_diagnostics_overall.csv`
  - `*_diagnostics_daily.csv`
- the updated roadmap and eval README

The core strategic question is now:

**What is the most promising way to improve tactical monetization of stored energy under
production-like site economics, while continuing to reduce dependence on Amber APF?**
