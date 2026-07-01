# Amber GST: billing behaviour vs. API representation — open reconciliation

**Status:** open investigation (started 2026-07-01). Cross-machine collaboration with a
second agent that holds the Amber **PDF bills** and **detailed usage CSVs**. This repo owns
the holistic model; that agent owns bill-level sanity checking.

## The question

`docs/tariff_gst_regime.md` and the fix committed on 2026-07-01 assume Amber made a
**behavioural change** at the FY26 → FY27 boundary (~1 Jul 2026): the feed-in (export)
`per_kwh` went from **GST-inclusive** to **GST-free**.

The owner's competing hypothesis: **billing behaviour never changed.** Resi feed-in credits
are GST-free under AU tax law regardless, and the ~1 Jul Amber *API* change merely started
*reporting* `per_kwh` in a way that better reflects what was always billed. If so, the
reconstruction break we saw at 1 Jul is a **representation** change (and/or a coincident
loss-factor / adder reset), not an economic one — and the "why it broke" narrative in
`tariff_gst_regime.md` needs revising.

The distinction matters because eval baselines (`eval/retariff_dispatch.py`,
`eval/rolling_mpc_eval.py`) still carry the OLD conditional-GST-on-feed-in logic on the
assumption it was *correct* for pre-FY27 windows. If feed-in was GST-free all along, those
pre-FY27 baselines are biased.

## Preliminary evidence from this repo (leans toward "billing didn't change")

Source: `amber-usage.csv` — Amber usage export, **general (E1)** and **feedIn (B1)**,
5-min intervals, **2025-11-01 → 2026-01-21** (i.e. mid-FY26, ~5 months *before* the API
change). Columns: `Price` (c/kWh, the customer-facing per_kwh), `Usage` (kWh), `Cost`.

- `Cost = Price × Usage` exactly on both legs (median ratio 1.000000). So `Price` is the
  billed per_kwh; nothing else is layered on at the interval level.
- Regressing **import Price on feed-in Price** at the same NEM interval (same underlying
  spot; intercepts absorb the fixed adders):

  | subset | n | OLS slope | Theil–Sen |
  | --- | --- | --- | --- |
  | all intervals | 23 616 | −1.125 | −1.35 (near-zero-denominator noise) |
  | \|feed\|>5c | 12 504 | −1.117 | −1.132 |
  | \|feed\|>10c | 8 045 | −1.111 | −1.100 |
  | real credits (feed<−2c) | 12 197 | −1.109 | −1.100 |

  On the clean subsets the slope is **−1.10 = −GST** (assuming equal import/export loss
  factors). If FY26 feed-in `per_kwh` were GST-**inclusive**, GST would cancel between the
  legs and this slope would be **−1.00**. It is not.

**Interpretation (provisional):** the Nov 2025–Jan 2026 usage data shows feed-in `per_kwh`
already moving GST-**free** vs. import, *contradicting* the current doc's claim that "FY26
was pristine because feed-in was GST-inclusive then." This supports the owner: the billed
feed-in leg looks GST-free well before 1 Jul 2026.

**Caveats / why this isn't conclusive on its own:**
- The import↔feed slope can't separate "feed GST-free, equal loss factors" from "both
  GST-inclusive, but L_import/L_export ≈ 1.10" — a 10% directional loss-factor split is
  physically unlikely but not disproven from this CSV alone (it has no raw AEMO spot column
  to regress against directly, unlike the original doc's method).
- The import leg has a sign-conditional GST guard (GST only when a net cost) and a large
  network adder, so very-negative-spot intervals are mildly non-linear — mitigated by the
  magnitude cuts above.
- This CSV ends Jan 2026; it does not contain the 1 Jul boundary itself.

## What only the bills can settle (asks for the bill-analysis agent)

See the outbound message. The decisive artifacts are (a) whether FY26 bills show GST on the
feed-in credit line, and (b) whether the effective feed-in credit **c/kWh** is continuous
across the 1 Jul 2026 boundary or steps down ~10%.

## Outbound message #1 (to bill-analysis agent, 2026-07-01)

```
Subject: Did Amber's feed-in GST treatment actually change on 1 Jul 2026, or did only the API catch up?

Context: I model Amber (Adelaide, SAPN) tariffs by stripping the wholesale component out of
Amber's reported per_kwh to recover fixed time-of-day adders + a network loss factor. Around
1 Jul 2026 our reconstruction broke on the FEED-IN (export) leg, and we patched it on the
assumption Amber changed behaviour: feed-in per_kwh went GST-inclusive -> GST-free at the FY
boundary. The owner suspects instead that billing NEVER changed (resi feed-in is GST-free
under AU tax law regardless) and Amber's API just started reporting per_kwh the way it was
always billed. You hold the ground truth (PDF bills + usage CSVs); I need you to settle it.

My model's constants (for your cross-check): GST = 1.10; distribution/loss factor ~1.08
(FY26) rising to ~1.1245 (FY27); import per_kwh = +GST*loss*rawspot + adder; feed-in per_kwh
(post-fix) = -loss*rawspot + adder, with adder ~0 off-peak and about -1.0 c/kWh for the
solar-sponge export charge. rawspot = Amber's spot_per_kwh / 1.10.

My preliminary finding that prompts this (from a Nov 2025-Jan 2026 usage CSV, i.e. mid-FY26,
BEFORE the boundary): regressing import per_kwh on feed-in per_kwh at the same interval gives
slope ~ -1.10, not -1.00. GST-inclusive feed-in would make GST cancel between the legs
(-1.00). So mid-FY26 feed-in already looks GST-FREE -- which contradicts our own "FY26 was
GST-inclusive" story and supports the owner. I can't fully rule out a ~10% import-vs-export
loss-factor split from the CSV alone; the bills remove that ambiguity.

Please check, from the bills/CSVs, and report back:

1. FEED-IN GST ON AN FY26 BILL (before 1 Jul 2026): Does the bill apply GST to the solar
   feed-in / export credit, or is that line GST-free? Quote the GST summary lines.

2. FY26 CSV -> FY26 BILL tie-out: Does the bill's total feed-in credit ($) equal the plain
   sum of the usage-CSV feed-in `Cost` for that period, or is a GST factor (x1.10 or /1.10)
   applied between CSV and bill?

3. THE DECISIVE ONE -- continuity across 1 Jul 2026: For comparable wholesale conditions,
   did the effective feed-in credit in c/kWh (bill feed-in credit $ / exported kWh) step
   DOWN by ~10% at 1 Jul, or stay continuous? Continuous => billing unchanged, API caught up
   (owner right). ~10% drop => a real economic change (our current story right). If you have
   a June bill and a July bill, a per-kWh comparison on similar days is ideal.

4. IMPORT leg: Confirm GST IS applied to the general/import charge. Note how any negative
   import intervals (paid to consume, during negative spot) are treated -- GST on/off?

5. SOLAR-SPONGE EXPORT CHARGE (intervals where you PAY to export, negative feed-in): Is GST
   applied to that charge? My model says NO (GST-free in both directions on the export leg).

6. Customer GST status: Any sign on the bill that Amber treats the account as NOT
   GST-registered (which would make feed-in GST-free by ATO rule, permanently)?

Reply format: for each item, the answer + the exact bill/CSV figures you used. If you can
give me feed-in credit c/kWh for one representative day in each of Jun 2026 and Jul 2026,
that alone likely closes item 3.

I'll fold your findings into docs/tariff_gst_billing_reconciliation.md and, if billing was
GST-free all along, revise docs/tariff_gst_regime.md's "why it broke" section and flag the
pre-FY27 eval baselines that assume GST'd feed-in.
```

## Findings from the bills (bill-analysis agent reply #1, 2026-07-01)

**Verdict: billing behaviour never changed — feed-in has been GST-free the entire time.**
The owner's hypothesis is confirmed as strongly as the on-disk data allows (see the one
remaining gap below).

Evidence, by item:

1. **Feed-in GST — GST-free on every bill.** All 12 consecutive invoices from period
   29/06/2025–29/07/2025 through 28/05/2026–27/06/2026 show `GST INCLUDED $0.00` in the
   credits section, plus standing boilerplate: *"for certain types of credits — most
   commonly solar exports — we are required to exclude GST for residential customers… the
   ATO considers these to be earnings."* This is on **all** bills, not just near the
   boundary — a standing ATO rule, not an FY-specific one.

2. **CSV → bill tie-out — no GST factor anywhere.** Usage-CSV `Cost = Price × Usage` to
   <$0.0001. Bill feed-in credit vs raw CSV sum, e.g. invoice 2949539: CSV $382.01 vs bill
   $382.61 (ratio 1.0016, **not** 1.10 or 1/1.10). The whole gap is explained exactly by the
   **9 kWh/day free-export-threshold + 90-day rollover** correction (matched to the cent in
   6/7 fully-covered periods). No GST arithmetic on the export leg at any stage.

3. **Continuity — no July data exists yet.** Last bill ends 27/06/2026; last usage CSV ends
   2026-06-30 23:25. The literal June-vs-July head-to-head can't be done. But 12 straight
   months are uniformly GST-free with no flip anywhere in the sequence. Late-June 2026
   feed-in rates to compare a future July bill against:

   | date | export kWh | avg feed-in c/kWh |
   | --- | --- | --- |
   | 2026-06-20 | 3.535 | 13.10 |
   | 2026-06-25 | 0.032 | 14.59 |
   | 2026-06-27 | 0.158 | 11.65 |
   | 2026-06-29 | 9.209 | 35.33 |
   | 2026-06-30 | 5.402 | 21.34 |

   If July per-kWh lands in a similar band under similar spot (not ~10% lower), continuity is
   confirmed.

4. **Import GST — applied to the net aggregate, including negative-usage months.** Bills
   carry a single `GST – 10%` line over Usage + Network + Amber Fee combined. Invoice 3471900
   (28/02–27/03/2026) had a *negative* net usage (avg wholesale −0.0777 $/kWh → Usage excl
   GST −$1.42) yet GST = $4.71 ≈ 10% × (Network $27.77 + Amber Fee $20.92 + Usage −$1.42) =
   10% × $47.27. **GST is not suspended or floored at zero when usage goes negative.** →
   direct evidence *against* our per-interval import sign guard (see implications).

5. **"Solar-sponge export charge" — the bill agent flagged a naming mismatch; owner
   clarified it is NOT one.** On the *bills*, *"Solar Sponge Energy"* is an **import** usage
   fee (cheap-midday consumption, ~2.8–4.7 c/kWh), and there is no separately billed
   export-side line — the ~1 c/kWh DNSP two-way export threshold is embedded inside the
   GST-free wholesale export price. But **SAPN's own tariff document names the 1 c/kWh charge
   "Export Charge – Solar Sponge"**, using "Solar Sponge" as the *period* label. So our
   `feed_in_tariff` ≈ −0.01 adder and its name are **correct and vendor-aligned** — the
   import "Solar Sponge Energy" fee and the export "Export Charge – Solar Sponge" simply share
   SAPN's period name. No rename. SAPN period definitions (already matched by
   `tariff_utils.tariff_bucket`, `_PEAK/_SOLAR` bounds):
   - **Solar Sponge** 10:00–16:00
   - **Peak** 17:00–21:00
   - **Shoulder** (our "off_peak") all other hours

6. **GST-registration — implicit, permanent.** No ABN/registration line, but the boilerplate
   frames GST exclusion as a standing residential-customer rule, not FY-specific. Consistent
   with "always the law."

## Refined narrative (what actually happened)

The break in the pipeline at ~1 Jul 2026 was **not** Amber changing billing behaviour. Two
sources have to be kept distinct — and the preliminary CSV analysis above conflated them:

- The **usage/billing** representation (PDF bills, "Amber Usage" CSV export) has been
  feed-in-GST-free the whole time (bills = ground truth; the mid-FY26 usage CSV slope ≈ −1.10
  agrees).
- Our pipeline consumes the **live pricing/forecast API** `per_kwh`, a *different* endpoint.
  The change at ~1 Jul was Amber's **price API representation** catching up to the
  long-standing GST-free billing reality. So `tariff_gst_regime.md`'s framing ("Amber changed
  how GST is applied") should become "Amber's price-API `per_kwh` representation was corrected
  to match already-GST-free billing" — a **representation** change, not economic.

Still not fully nailed: whether the live price API's feed-in `per_kwh` was *literally*
GST-inclusive before 1 Jul (a real API-field flip) or whether our pipeline was mis-modelling
already-GST-free feed-in all along. The bills can't see the price API. Pipeline historical
price logs, or the July usage/price comparison, would close this last mechanistic gap. Either
way the fix committed on 2026-07-01 (feed-in GST-free going forward) is correct.

## Implications for owner decision

1. **Import sign guard (`per_kwh > 0`) — DROPPED (owner-approved, applied 2026-07-01).** Item 4
   shows GST applies to negative import too. The rule is now fully directional on *both* legs:
   import always GST-inclusive, feed-in always GST-free, no sign guards. Guard removed at all
   five sites — `add_gst`/`remove_gst` (forecast.py), `apply_tariffs_to_forecast`,
   `_reconstruct_per_interval`, `_fit_tariff_profile`, and
   `tariff_utils.tariffed_price_frame_from_wholesale_mwh` — forward and reverse kept exact
   inverses. Test `test_general_price_negative_wholesale_still_has_gst` updated; unit suite
   green (230 passed).
2. **~~Rename the "solar-sponge export charge"~~ — no change.** Owner confirmed SAPN's own
   term is "Export Charge – Solar Sponge"; our naming is correct (see item 5).
3. **Pre-FY27 eval baselines** (`eval/retariff_dispatch.py:95`,
   `eval/rolling_mpc_eval.py:489-499`) assume conditional-GST-on-feed-in was *correct* for
   pre-FY27 windows. On this evidence it never was — those baselines are biased on the export
   leg. Flag before any eval rerun; left unchanged pending owner decision.

## Resolution log

- 2026-07-01 — opened; preliminary usage-CSV finding (feed-in slope ≈ −1.10, GST-free
  mid-FY26); message #1 drafted to bill agent.
- 2026-07-01 — bill agent reply #1: **confirmed** GST-free feed-in across all 12 FY26 bills;
  CSV↔bill gap = free-export-threshold, not GST; import GST on net incl. negative months;
  "solar-sponge" is an import fee (naming mismatch). Findings + implications above. **Open:**
  July continuity (no data yet) and the price-API-flip-vs-always-mismodelled mechanism.
