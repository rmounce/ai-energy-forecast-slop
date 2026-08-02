# HWC fan-speed controller family and thresholds

Understanding of the Aquatech RAPID/X6 fan controls, recorded 2026-08-01 and updated 2026-08-02.
The threshold logic is
partly black-box: parameter meanings come from a closely matching R290 controller manual; transition
direction and persistence come from this unit's telemetry. Treat the inferred Boolean rule as strong,
not vendor-confirmed firmware documentation.

## Aquatech options

The Aquatech owner manual calls F30/F35 the **fan high/low-speed running parameters** and gives two
profiles:

| profile | F30 | F35 | practical result |
|---|---:|---:|---|
| Factory | 25 °C | 55 °C | High fan through almost every Adelaide-winter reheat |
| Manufacturer-documented quiet | 10 °C | 30 °C | Low fan normally; high fan retained for cold conditions |

Aquatech documents the menu operation and values, but not the sensor associated with each parameter
or the transition state machine. Its quiet procedure changes F30 `25→10` and F35 `55→30`; this is a
manufacturer-supported profile, not an installer-invented setting.

Source: [Aquatech RAPID/X6 and DYNAMIC/X8 owner manual, p. 37](https://www.aquatechheatpumps.com.au/_files/ugd/228c32_84eb2f4659664c0e92f0dd0fe910d8fa.pdf).

On **2026-08-02**, the owner read every available parameter from the physical controller before
restoring quiet mode. The complete as-found factory snapshot is in `aquatech-settings.csv`; its
`Custom` column records F30 `10` and F35 `30`. This snapshot supersedes related-product defaults for
the installed unit. F39 was absent from the menu. F70 displayed `998`, matching the password the
owner successfully used to enter the settings menu; this confirms the related manual's F70
“menu password” description on the installed controller.

The CSV cross-references descriptions available in the related HI-WATER/Hisense R90 manual. Its
status column is deliberately conservative: `Unconfirmed` means only that the same F-number exists
in the family manual, even when the recorded value also matches; `Partly confirmed` means Aquatech
documents the broad purpose but not the detailed state logic; `Confirmed` requires Aquatech-specific
documentation or direct observation. Blank descriptions have no sufficiently close published match
yet.

## Parameter meanings and likely rule

The matching HI-WATER/Hisense R90 R290 factory table identifies:

- **F30:** ambient-air temperature point associated with fan shutoff/changeover; allowed range
  `10–40 °C`, default `25 °C`.
- **F35:** high/low fan transfer tank-water temperature; allowed range `10–60 °C`, default `48 °C`
  in that manual.

The F35 default differs from Aquatech's explicit `55 °C`, so the manuals are not interchangeable
firmware specifications. The matching parameter numbers, descriptions, ranges, neighbouring fan
parameters and R290 control table are nevertheless strong evidence that they use the same controller
or firmware family.

Sources: [HI-WATER R290 manual](https://device.report/manuals/hi-water-r290-heat-pump-water-heater-manual),
[Hisense R90 rendering of the same parameter table](https://manuals.plus/hisense/r90-heat-pump-water-heater-manual).

Best current model of the Aquatech behaviour:

```text
high fan if ambient <= F30 OR tank <= F35
low fan only if ambient > F30 AND tank > F35
```

A low-temperature condition appears to latch high fan for at least much of the active compressor
cycle. Exact comparison operators, hysteresis and unlatching rules are unknown. In particular, the
poorly translated F30 label says "turn off fan motor ambient temp. point", while this unit exposes
`Off / Low / High`; do not interpret that phrase as proof the complete fan stops during an ordinary
reheat.

## Evidence from this unit

The 30-second fan trace in `data/hwc_cycles.sqlite` supports the model:

- Under quiet `10/30`, normal June daytime reheats were low fan with ambient roughly `12–22 °C`
  and tank starts roughly `35–54 °C`.
- Quiet-mode high fan appeared on the cold runs: 2026-06-22 at ambient `4–6 °C` and 2026-06-28 at
  `8–10 °C`. It continued while tank temperature rose well above `30 °C`.
- Brief high-fan samples also occurred at ambient below `10 °C` with tank around `57 °C` on
  2026-06-20 and 2026-06-27. This isolates F30: cold ambient alone can request high fan.
- After the owner moved the thresholds back towards factory defaults on **2026-07-04 09:55**, every
  analysed winter daytime cycle ran high fan. That is expected when ambient `11–17 °C` is below a
  `25 °C` F30 threshold, even as the tank approaches `60 °C`.
- On **2026-08-02**, after recording the full factory parameter set, the owner restored quiet
  F30=`10`, F35=`30`: the measured efficiency penalty was considered small and the noise reduction
  noticeable. Treat cycles from this date as quiet-profile cycles again; exact change time was not
  recorded.

The trace therefore explains the remembered behaviour: quiet mode **prefers** low fan but is not a
low-fan lock. Cold overnight recovery still invokes high fan.

## Suspected related products

See `controller_family_manual_sweep.md` for the broader YT/Solareast/Airtherm/Ecostar/SIPH manual
trail and its element Boost/sterilisation findings.

Relationship confidence matters; shared parameter tables or Tuya datapoints do not prove identical
compressors, plumbing, tuning or firmware.

| product/manual | relationship evidence | safe conclusion |
|---|---|---|
| **Aquatech RAPID/X6** | Installed unit; current Aquatech manual covers it directly | F30 `25→10`, F35 `55→30` procedure applies |
| **Aquatech DYNAMIC/X8** | Same current Aquatech manual and fan procedure cover both X6 and X8 | Same exposed fan menu; other hardware may differ |
| **Hydrotherm DYNAMIC/X8** | Hydrotherm documents name Aquatech Solar Technologies as product/service authority; repo design history records X6 Gen2 as using the same compressor/electronics | Very close Australian-market sibling; do not assume every firmware revision matches |
| **HI-WATER R290 all-in-one** | Near-exact F-code table: F30/F32/F33/F35/F36/F37/F38, matching ranges and neighbouring refrigeration controls | Strong controller-family evidence; its F35 factory default is `48`, not Aquatech's `55` |
| **Hisense R90** | Public R90 manual reproduces the HI-WATER R290 table and control behaviour | Strong manual/controller-family lead; manufacturer/OEM chain is not confirmed |
| **Emerald all-in-one / Rinnai DemandDuo Tuya variants** | Same T1–T5-style refrigeration sensors and closely matching Tuya datapoint semantics, documented in `aquatech_entities.md` | Useful for decoding sensors only; no present evidence that F30/F35 values or fan logic match |

Hydrotherm sources: [DYNAMIC/X8 owner manual](https://www.hydrothermhotwatersystems.com.au/wp-content/uploads/2024/02/HYD_DYNAMIC_USER_MANUAL_WEB.pdf),
[older DYNAMIC/X8 manual](https://www.hydrothermhotwatersystems.com.au/wp-content/uploads/2020/03/Dynamic-X8-Owner-Manual-web.pdf).

## HI-WATER/Hisense R90 factory-parameter backup

Raw transcription of every entry in the manual's **Factory Parameters List** on 2026-08-01.
Missing code numbers are missing in the source table; they are not transcription omissions. Wording,
ranges and defaults are preserved except for whitespace/punctuation cleanup. This is a reference
backup, **not an Aquatech configuration recommendation**. The manual says these parameters are for
professional engineers; several control safety, defrost and refrigeration behaviour.

| code | manual description / enumerated options | range | unit | default | remark |
|---|---|---:|---|---:|---|
| F01 | Set target heating temperature | 15–55 | °C | 55 | Adjustable |
| F02 | Set target cooling temperature (N/A for DHW) | 7–30 | °C | 12 | Factory |
| F03 | Water-temperature control difference | 1–15 | °C | 5 | Factory |
| F04 | Heating-temperature setting range: `0` = 15–55; `1` = 15–75; `2` = 15–60; `3` = 15–40 | 0–3 | — | 0 | Factory; R134: F04=1, R290: F04=0 |
| F05 | Temperature-setting deviation in automatic mode | -10–20 | °C | 0 | Factory |
| F08 | Maximum tank temperature with heat pump only | 30–75 | °C | 60 | Factory |
| F09 | Heat-pump lowest working ambient temperature | -15–5 | °C | -7 | Factory |
| F10 | Auxiliary electric-heater starting ambient temperature | -10–35 | °C | 5 | Factory |
| F11 | Tank-temperature sensor calibration value | -20–20 | °C | 0 | Factory |
| F12 | Outlet-water-temperature sensor calibration value | -20–20 | °C | 0 | Factory |
| F13 | Auto fast-heating mode: `0` = on; `1` = forbidden | 0–1 | — | 1 | Factory |
| F14 | Auto fast-heating temperature difference | 2–70 | °C | 40 | Factory |
| F15 | “Between setting temp. and real water temp.” (description incomplete in source) | 50–99 | °C | 68 | Factory |
| F16 | Warning high temperature | 2–15 | °C | 5 | Factory |
| F20 | Defrosting period | 1–90 | min | 40 | Factory |
| F21 | Defrost time each time | 6–90 | min | 10 | Factory |
| F22 | Maximum ambient temperature for defrosting | 0–50 | °C | 12 | Factory |
| F23 | Defrost starting coil temperature | -30–30 | °C | -3 | Factory |
| F24 | Defrost stopping coil temperature | 0–50 | °C | 18 | Factory |
| F25 | Defrost temperature difference between ambient and coil | 0–15 | °C | 10 | Factory |
| F26 | Compressor continuous-run time before defrosting | 0–40 | min | 6 | Factory |
| F30 | Turn off fan motor ambient-temperature point | 10–40 | °C | 25 | Factory |
| F32 | Turn off fan motor exhaust-temperature point | 10–125 | °C | 100 | Factory |
| F33 | Turn on fan motor exhaust-temperature difference | 1–50 | °C | 5 | Factory |
| F35 | High/low fan transfer tank temperature | 10–60 | °C | 48 | Factory |
| F36 | Tank temperature when fan motor turns off | 15–75 | °C | 52 | Factory |
| F37 | Coil temperature when fan motor turns off | 10–30 | °C | 18 | Factory |
| F38 | Coil-temperature setting when fan restarts | 0–15 | °C | 7 | Factory |
| F40 | Low-pressure switch: `0` = alarm when switch on; `1` = alarm when switch off; `2` = forbidden | 0–2 | — | 2 | Factory |
| F43 | Low-voltage fault detection delay | 0–60 | min | 3 | Factory |
| F44 | High-pressure switch: `0` = alarm when switch on; `1` = alarm when switch off; `2` = forbidden | 0–2 | — | 1 | Factory |
| F45 | Maximum automatic recoveries from low/high-voltage faults | 0–10 | — | 3 | Factory |
| F47 | Water-flow switch: `0` = switch on means failure; `1` = switch off means failure; `2` = forbidden | 0–2 | — | 2 | Factory |
| F50 | Electronic expansion-valve (EEV) control cycle | 20–90 | sec | 30 | Factory |
| F51 | Target superheat when ambient >15 °C | -8–15 | °C | 1 | Factory |
| F52 | Expansion-valve allowed exhaust temperature | 70–120 | °C | 92 | Factory |
| F53 | Defrost expansion-valve setting | 20–450 | P | 400 | Factory |
| F54 | Minimum expansion-valve opening when ambient ≥5 °C | 80–250 | P | 100 | Factory |
| F55 | Expansion-valve selection: `0` = automatic; `1` = manual | 0–1 | — | 0 | Factory |
| F56 | Manual expansion-valve step count | 20–450 | P | 350 | Factory |
| F60 | Exhaust high-temperature protection value | 50–110 | °C | 100 | Factory |
| F61 | Tank-temperature compensation: `0` = automatic; `1` = cancel | 0–1 | — | 0 | Factory |
| F62 | Cooling/heating selection: `0` = cooling; `1` = heating | 0–1 | — | 1 | Factory |
| F63 | System working mode: `0` = manual; `1` = automatic | 0–1 | — | 0 | Factory |
| F66 | Electric disinfection: `0` = disabled; `1` = enabled | 0–1 | — | 1 | Factory |
| F68 | Low-temperature anti-freezing: `0` = disabled; `1` = enabled | 0–1 | — | 1 | Factory |
| F70 | Menu password; `0` cancels password | 0–999 | — | 0 | Factory |
| F92 | Temperature unit: `0` = Celsius; `1` = Fahrenheit (reserved) | 0–1 | — | 0 | Factory |
| F93 | Ventilation function: `0` = heat pump preferred; `1` = ventilation priority | 0–1 | — | 0 | Factory |

Primary backup source: [HI-WATER R90 PDF](https://www.energyduegi.com/schede_tecniche/Manuale%20d%27uso%20HI-WATER%20R90.pdf).
Cross-check rendering: [Hisense R90 manual](https://manuals.plus/hisense/r90-heat-pump-water-heater-manual).

## Operational cautions

- Record the timestamp and exact F30/F35 values whenever changing profiles.
- Split COP/calibration data at each change; `fan_high_on` means high occurred at least once during
  the cycle, not that the whole cycle was high.
- Do not copy other models' factory tables wholesale. Safety, defrost, element and refrigeration
  parameters adjacent to F30/F35 are out of scope and must remain untouched.
- If exact transition/hysteresis semantics become operationally important, test one threshold at a
  time while recording ambient, tank and raw fan state; do not infer them by changing both together.
