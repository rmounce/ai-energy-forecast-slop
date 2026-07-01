# Amber GST regime and the FY26 → FY27 tariff change

## Summary

The pipeline reconstructs deterministic time-of-day tariff *adders* (`general_tariff`,
`feed_in_tariff` in `tariff_profile.json`) by stripping the wholesale component out of
Amber's reported `per_kwh` prices. Around the FY26 → FY27 boundary (~1 July 2026) the
feed-in (export) `per_kwh` from Amber's **live pricing API** stopped carrying GST, which
broke that reconstruction until the code was updated to match.

> **Correction (2026-07-01, from bill reconciliation):** this was *not* a billing/behavioural
> change. PDF bills confirm feed-in credits have been **GST-free the entire time** (all 12
> FY26 bills; standing ATO residential rule). What changed at ~1 July was Amber's price-**API
> representation** catching up to the always-GST-free billing reality — a representation
> change, not economic. See `docs/tariff_gst_billing_reconciliation.md`. That same
> reconciliation retired the **import sign guard** (the bills apply GST to negative-usage
> months too): the import leg now carries GST unconditionally, as reflected in the rule below.

**Rule (verified against live forecasts on 2026-07-01):**

- **Import (general) leg** carries GST **unconditionally**, both signs — including a
  negative import price (being paid to consume). Bills levy GST on the net usage regardless
  of sign; the old `> 0` sign guard was dropped 2026-07-01 (see
  `docs/tariff_gst_billing_reconciliation.md`).
- **Feed-in (export) leg is GST-free in both directions** — export *credits* (you are
  paid) *and* the solar-sponge export *charge* (you pay to export) alike.

This is a **directional** (buy-vs-sell) rule, not a sign-conditional one: GST is decided by
the *leg* (import vs export), never by the sign of the price. We explicitly tested and
rejected the "GST on whatever costs you, either leg" hypothesis: the feed-in export charge
is not GST'd.

## Evidence

Regressing Amber's reported `per_kwh` against raw AEMO spot (`spot_per_kwh / api_scaling`),
both fits r = ±1.000:

| leg | slope | interpretation |
| --- | --- | --- |
| Import | `+gst · net_loss` (≈ +1.2372) | GST on the whole price |
| Feed-in | `−net_loss` (≈ −1.1248, **not** −1.2369) | no GST on the wholesale credit |

Splitting feed-in by sign of spot (credit vs export charge) gives the same slope
`−net_loss` and an implied GST multiplier of ≈ 1.00 in **both** cases, including the
solar-sponge export charge. So the export leg is uniformly GST-free.

Rarely exercised: negative **import** prices (import stays positive except during summer
solar-sponge negative-spot events) and large feed-in export charges. The import leg was
originally given a `> 0` sign guard as a low-regret default; the FY26 bills later showed
GST *is* levied on negative-usage months, so the guard was removed (import GST is now
unconditional — see `tariff_gst_billing_reconciliation.md`).

## Why it broke

In FY26 the **live price API's** feed-in `per_kwh` was reported GST-*inclusive* (slope
`−gst·net_loss`) — even though the underlying billing was always GST-free (see the
correction note above and `tariff_gst_billing_reconciliation.md`). The reverse
path did `-remove_gst(per_kwh) - wholesale·loss`, and the two GST factors cancelled
exactly, so `feed_in_tariff` reconstructed to a clean `0` / `−0.01` regardless of spot.

When Amber dropped GST from the feed-in leg, `remove_gst` became an over-division. The
residual gained a term `−net_loss·(1 − 1/gst)·spot ≈ −0.102·spot` that rides the
wholesale curve — the "slightly off" `feed_in_tariff` values (0.0005 / −0.0091 / 0.001
after median smoothing) that first flagged the change.

Worse, `_calculate_forecasted_network_loss_factor` derives `network_loss_factor` from the
same GST-free feed-in via `-remove_gst(per_kwh)/raw_spot`. On the next `update-tariffs`
run this would have collapsed the loss factor from **1.1245 → ~1.0225** (= 1.1245 / gst),
in-bounds and silently accepted, which then contaminates the GST-inclusive
`general_tariff` with a `+0.102·spot` error — a much larger, import-side problem.

## The fix

Forward and reverse paths are now exact inverse pairs on both legs.

- `forecast.py::_get_tariff_data` — feed-in: no `remove_gst`; import: `remove_gst` only
  when `per_kwh > 0`.
- `forecast.py::_calculate_forecasted_network_loss_factor` — derive from the raw
  (GST-free) feed-in `per_kwh`, no `remove_gst`.
- `forecast.py::apply_tariffs_to_forecast` and
  `tariff_utils.py::tariffed_price_frame_from_wholesale_mwh` — feed-in price is GST-free
  unconditionally; general price keeps its `> 0` sign guard.

Round-trip verification against the live 2026-07-01 forecast reconstructs
`feed_in_tariff` back to `0.00000` (off-peak) / `0.00003` (peak) / `−0.00999`
(solar-sponge), each with std ≈ 0.00004, and the loss-factor re-derivation returns
~1.119 instead of the corrupt ~1.018.

## Estimation methodology (precision)

Amber quantizes both `per_kwh` and `spot_per_kwh` to 4 dp ($/kWh) = **0.01 c/kWh**, so
each field carries ±0.00005 rounding noise. Two design choices compensate:

1. **Snap `amber_api_scaling_factor` to GST.** This factor reconciles Amber's
   GST-inclusive `spot_per_kwh` display with the raw AEMO wholesale price, so its true
   value *is* the GST rate (1.1). Estimating it from the Amber-vs-AEMO-predispatch ratio
   only injects forecast/rounding noise, which compounds into `network_loss_factor`
   (which is derived as `≈ 1.022 × api_scaling`). We therefore compute it only as a
   sanity check and, when it lands within tolerance of GST, use **exactly** 1.1. A drift
   beyond tolerance is logged as an error and the measured value kept, since it would
   signal Amber changed their spot basis. The reverse-path fixed-adder reconstruction is
   unchanged by this (it depends only on `net_loss / api_scaling`); the benefit is to the
   forward path, which uses `net_loss` alone against the AEMO-scale price forecast.

   GST = 1.1 is confirmed independently of this estimate: within a band, regressing import
   `per_kwh` on feed-in `per_kwh` (same underlying spot) gives slope −1.0998 (r = 0.99999),
   i.e. `−gst·(L_import/L_export)` with equal loss factors.

2. **Pooled per-band OLS instead of per-interval median.** The reconstruction is linear
   (`gst-adjusted price = loss·wholesale + fixed_adder`), so one least-squares fit with a
   shared slope (the loss factor) and a per-(leg, bucket) intercept (the fixed adder)
   estimates everything jointly and pools the rounding noise. Empirically this tightened
   the off-peak fixed adder from a per-interval spread of ~1×10⁻⁴ to a standard error of
   ~7×10⁻⁶ (~15×), and returns the loss factor with an error bar (e.g. 1.1247 ± 0.0003
   from one day). If the slope is poorly determined (a flat-spot day, large SE) the
   previous loss factor is retained; if the fit is unavailable it falls back to bucket
   medians. A rolling multi-day window would tighten this further and is a possible future
   step, but is deferred to avoid straddling FY and export-credit-season boundaries.

## Observed FY26 → FY27 rate changes (Adelaide, SAPN / Amber)

Fixed adders from `tariff_profile.json` (ex-GST, $/kWh) and the calibration scalars:

| item | FY26 (final) | FY27 (first) | Δ |
| --- | --- | --- | --- |
| general — off-peak/shoulder | 0.1331 | 0.1462 | +0.0131 |
| general — solar sponge | 0.0666 | 0.0720 | +0.0054 |
| general — peak | 0.3580 | 0.4001 | +0.0421 |
| feed-in — solar-sponge export charge | −0.0100 | −0.0100 | 0 |
| `amber_api_scaling_factor` | 1.100000 | 1.100201 | ~0 |
| `network_loss_factor` (effective) | 1.079767 | 1.124457 | +0.0447 |

Consistent with Amber's notified changes effective 1 July 2026: network shoulder
+1.31 c/kWh (exact match on the general off-peak adder), peak and solar-sponge usage up,
and SAPN's distribution loss factor rising 1.0811 → 1.1316 (the effective
`network_loss_factor` tracks this, bundling the Amber API scaling separation). The
environmental-certificate cut, higher market charges, and removal of the carbon-neutral
add-on net into the residual general-adder movements. The one behavioural change beyond
the rate table is the **GST-free feed-in leg** documented above.
