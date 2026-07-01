# Amber GST regime and the FY26 → FY27 tariff change

## Summary

The pipeline reconstructs deterministic time-of-day tariff *adders* (`general_tariff`,
`feed_in_tariff` in `tariff_profile.json`) by stripping the wholesale component out of
Amber's reported `per_kwh` prices. Around the FY26 → FY27 boundary (~1 July 2026) Amber
changed how GST is applied to the **feed-in (export) leg**, which broke that
reconstruction until the code was updated to match.

**Rule (verified against live forecasts on 2026-07-01):**

- **Import (general) leg** carries GST, and only when the price is a net **cost** (`> 0`).
  A negative import price (being paid to consume) is GST-free.
- **Feed-in (export) leg is GST-free in both directions** — export *credits* (you are
  paid) *and* the solar-sponge export *charge* (you pay to export) alike.

This is a **directional** (buy-vs-sell) rule, not a sign-conditional one. We explicitly
tested and rejected the "GST on whatever costs you, either leg" hypothesis: the feed-in
export charge is not GST'd.

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

Untestable at present: negative **import** prices (import stays positive except during
summer solar-sponge negative-spot events) and large feed-in export charges. We keep the
import sign guard (GST only on positive cost) as the low-regret choice for the former.

## Why it broke

In FY26 the feed-in `per_kwh` was GST-*inclusive* (slope `−gst·net_loss`). The reverse
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
