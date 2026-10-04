# Controlled load-information value — 2026-10-05

- Parallel follow-up to [cycle mechanisms](controlled_cycle_value_2026-10-05.md); APF retained,
  no production/network/optimizer calls. Price/PV paths known equally to all arms.
- `controlled_load_information.py`: eight synthetic scarce/ample×low/high demand×solar/no-solar
  cases. Four information arms: incumbent assumption, modest correction, worse bias, perfect load.
- Enumerate early export cap0–2kW in0.05kW steps over1h; minimise forecast cost +4c/DC-kWh wear
  −20c/stored-kWh ending stock. Carry each decision through identical realised phases and physics.
- No endpoint reset; later self-consumption uses common actuals. Physical floor15%,40.3kWh stock,
 99% battery/95% inverter efficiencies,140W overhead. Scarce starts2.5kWh above floor; ample95%.
- Stipulated demand errors/correction directions know the synthetic case: not a trained forecast.
  ±0.5kWh correction resembles magnitude of recorded14h correction, but is concentrated in1h;
  cannot transfer its measured skill or economic gain from prior historical replay.

## Default results

Early export30c/AC-kWh; later import50c/AC-kWh. Terminal20c/stored-kWh,wear4c/DC-kWh:

| Regime without later solar | Incumbent → modest-correction export cap kW | Net value change cents | Ending stock difference kWh |
|---|---:|---:|---:|
| Scarce, actual later demand1.2kWh; forecast1.8→1.3kWh |0→0.45 |+2.0359 |−0.478469 |
| Scarce, actual later demand2.4kWh; forecast1.3→1.8kWh |0.45→0 |+9.1796 |0 |
| Ample stock, low or high demand |2→2 |0 |0 |

- Low-demand correction releases profitable export but ends with less stock; gain reverses if
  ending energy valued50c instead of20c. Perfect-load arm selects0.55kW, gain2.488c at stated values.
- High-demand correction avoids≈0.454kWh later imports. Equal ending floor and DC throughput;
  value is real within this synthetic case rather than unpriced extra stock depletion.
- Worsened high-demand forecast0.8kWh selects0.95kW early export, loses≈10.12c versus incumbent.
- Known later replenishing solar removes modest-correction value in scarce cases; ample cases
  all insensitive. This isolates a mechanism, not evidence that uncertain real PV is always known.
- Oracle upper bound applies only to enumerated export-cap policy and stated objective. Changing
  terminal/wear sensitivity revalues fixed decisions; it does not rerun their selection.

## Verification and decision

- 15 tests pass: same actuals, physical continuity, scarce floor/capacity, oracle policy-class bound,
  adverse bias, terminal reversal, analytic zero-overhead cash, immutable forecast and invalid values.
- Root reviewed selection/economics: perfect information scores same realised physical objective,
  all forecast arms see same price/PV inputs, with all-leg throughput and ending value retained.
- Prior40min zero historical calibration gain remains valid for that window. These scenarios show
  why it cannot rule out load value at scarce reserves; they do not show current correction will help.
- Next empirical load gate: incumbent versus actual causal calibrated vintages in an admitted
  reserve-constrained window, with common PV scenarios and perfect-load headroom reported separately.
  Do not increase model complexity until this decision sensitivity and existing calibration skill
  hold across more than selected examples.
- Ignored output `data/energy_replay/controlled_load_information_20261005.json`; scenario/decision
  choices, common actuals, phase constraints and code hashes retained.

```bash
nice -n 19 ./.venv/bin/python eval/controlled_load_information.py \
  --output data/energy_replay/NEW_LOAD_INFORMATION.json
```
