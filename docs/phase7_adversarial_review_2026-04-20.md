# Phase 7 "Federated Forecast" — Adversarial Review
**Date:** 2026-04-20  
**Purpose:** Red-team critique of the Output Stitching proposal before committing to Phase 7.

---

## Debate Point 1: The Boundary Discontinuity Trap

**This is the most serious risk, and smoothing alone doesn't fully fix it.**

The discontinuity isn't just a cosmetic jump — it's a calibration problem. Model A is trained on
PREDISPATCH features with actual SA1 prices as target. Model B is trained on PD7Day with the same
target. Their output distributions will be different in ways that a simple blend can't fix, because
they've learned different things about what "normal" looks like.

The specific EMHASS failure mode: the LP doesn't care about smoothness, it cares about price
deltas. If the stitch creates a $60 step at h=28, EMHASS will plan to discharge just before and
recharge just after — not because that's good arbitrage, but because the optimiser is doing exactly
what it was designed to do. A ±2h Gaussian blend reduces this but doesn't eliminate it if the
underlying calibrations differ by more than the blend width.

**Counter-argument in your favour:** the day-ahead runs 3× per day, so the stitch boundary is in
a different position on each run. The day-ahead plan gets overwritten every 8 hours anyway. The
artefact may not be persistent enough to cause real dispatching harm — the MPC will see the real
price before the stitch point ever gets executed.

**Net read:** the discontinuity is real but probably manageable in practice, provided the MPC's
near-term view always overrides it before execution. The bigger risk is the day-ahead plan during
that 8h window influencing the `soc_final` target in a way that degrades MPC performance at the
*stitch horizon* — not at the stitch point itself.

---

## Debate Point 2: Managing the Dynamic Splice

**Fix the stitch at a static horizon. The dynamic approach has a subtle but nasty failure mode.**

The accordion swings from ~32 steps (16h, just after PREDISPATCH cutover) to ~80 steps (40h, just
before). If the stitch moves with availability, the day-ahead plan can shift the handoff by up to
24h between consecutive runs. EMHASS will see a fundamentally different forecast shape every 30
minutes, not just updated values. The LP optimises over the entire 72h each time — if the plan for
hours 20–30 keeps changing character (Model A today, Model B next run, Model A the run after), the
day-ahead SoC trajectory becomes erratic and the `soc_final` target passed to the MPC bounces
around.

**Recommendation:** fix the stitch at a horizon shorter than the minimum reliable PREDISPATCH
coverage. From the training data, PREDISPATCH covers 32 steps (16h) on ~98% of runs. Fix the
boundary at **h=16** (32 steps). This is conservative — you discard PREDISPATCH signal from 16h
to 28h when it's available — but it gives you a stable, predictable stitch point and Model A
always has complete decoder inputs.

The cost: you're giving up ~12h of good PREDISPATCH coverage on half of all runs. That's worth
paying for a stable day-ahead plan.

---

## Debate Point 3: Evaluating Model B's Actual Value

**Model B almost certainly shouldn't be a trained ML model yet.**

You have ~75 days of PD7Day training data. Realistic training/val split gives you ~60 days train,
~15 days val. SA1 spike events — the whole point of getting the terminal SoC target right — are
rare. You might see 5–8 meaningful spike events in 60 days of training. A model trained on 5
examples of the thing you care most about is not a model; it's a coincidence engine.

**The honest baseline test:** take raw PD7Day `rrp` values as "Model B." Apply a simple additive
bias correction fitted on those 75 days. Compare against a static 20% reserve rule on a held-out
eval set of multi-day arbitrage windows. If bias-corrected raw PD7Day doesn't beat the static
reserve, you don't need Model B at all — and if it does, the delta is your upper bound on what a
trained model can add.

**Recommendation:** build Model B as a rules-based system first (bias-corrected PD7Day + a "spike
day" flag that lifts the terminal SoC floor based on SevenDayOutlook demand signal) and only
replace it with ML once you have 6+ months of PD7Day coverage. The complexity of a trained model
is not justified at 75 days.

---

## Debate Point 4: The Demise of the Debiaser

**You can drop it for Model A, but you need a controlled comparison before you trust the result.**

The argument for dropping it: Model A's encoder sees 96 steps of actual historical prices. With
PREDISPATCH as a direct decoder feature and actual prices as target, the TFT can learn
PREDISPATCH bias implicitly through the attention mechanism — it can look back at how PREDISPATCH
overshot on the last 5 constraint events and discount accordingly. The debiaser was a preprocessing
step that handled this explicitly because the original single TFT didn't have a strong enough
training signal at short horizons. At 0–16h, PREDISPATCH coverage is nearly 100%, so implicit
learning has enough examples.

The real risk: the spike routing (+9.7%) worked. You don't actually know how much of that gain
came from the TFT vs the debiaser doing the heavy lifting on spike windows. Dropping the debiaser
and retraining from scratch means you lose the +9.7% reference point and can't distinguish
"Model A natively handles PREDISPATCH bias" from "Model A is just bad on spike windows again."

**Recommendation:** keep the debiaser for the first Model A training run. Evaluate against the
existing +9.7% benchmark. Only then run an ablation (Model A without debiaser) to see if it
regresses. Don't drop both the architecture change and the debiaser simultaneously — you won't
be able to attribute the result.

---

## Two Concerns Not in the Original Framing

### 5. Model A as TFT may be over-specified

For 0–16h with complete PREDISPATCH decoder inputs, does Model A need to be a TFT at all? The
existing Tier 1 LightGBM already beats P5MIN by +24% MAE at 0–60min. A well-engineered LightGBM
on PREDISPATCH features for 0–16h (32 × 30-min steps) with horizon as a feature might match or
beat a TFT — it trains in minutes, is trivial to retrain weekly, and has no attention-head
pathologies. The TFT's advantage is learning complex temporal dependencies over long encoder
windows; PREDISPATCH *is* the temporal signal here, so the encoder may be redundant.

Worth benchmarking a LightGBM baseline before committing to TFT for Model A.

### 6. The stitch is invisible to EMHASS — but the MPC SoC inheritance logic is not

Even with a perfectly stitched 72h array, the asymmetric `soc_final` Jinja logic (positive
deviation only, `+0.5` bias) still means the MPC can drift ahead of the day-ahead SoC plan but
not behind. If Model A correctly predicts a near-term spike and the day-ahead plan conserves
battery for it, but the actual spike arrives 2h earlier than predicted, the MPC enters the spike
with lower SoC than planned and the deviation correction won't help — you can't reclaim energy
you already discharged.

This isn't a reason to abandon stitching. It's a reason to revisit the `soc_final` Jinja logic
in parallel — the asymmetric deviation correction made sense when the day-ahead plan was
unreliable, but if Model A is genuinely accurate at 0–16h it may be counterproductive.

---

## Net Assessment

Output stitching is the right direction. The concerns above are all solvable. Recommended priority
order for the design:

1. **Fix the stitch boundary at h=16** (static, not dynamic) before anything else
2. **Model B = bias-corrected raw PD7Day** until 6 months of data exists
3. **Keep the debiaser for Model A's first run**, ablate it later
4. **Benchmark LightGBM vs TFT for Model A** before committing to the heavier architecture
5. **Revisit `soc_final` Jinja logic** once Model A is validated — the asymmetric correction is
   separate debt that stitching exposes

The "silver bullet" framing is slightly optimistic — you're not eliminating EMHASS optimisation
debt, you're trading one kind (bad 72h forecast corrupting day-ahead plan) for a better-understood
kind (stitch boundary artefact that MPC overrides before execution). That's a good trade.
