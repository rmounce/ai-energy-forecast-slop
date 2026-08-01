# HWC controller-family manual sweep — 2026-08-01

Target: find better documentation for the Aquatech RAPID/X6 controller, especially a way to start
the resistive element above its observed ~60 °C ordinary-mode re-trigger threshold. This is an
evidence map, not proof that commands or parameters are portable across products.

## Main result

The strongest trail is the **YT-200/250/300TB2** all-in-one HPWH family. The same model identifiers,
physical format and controller vocabulary appear under Solareast/Sunrain, Airtherm Aqua, Ecostar,
Sacon and Chameleon/SIPH. Hisense's AH-200/300NH4GHB manual retains YT model identifiers in some
drawings/tables, consistent with an adapted controller/OEM-family manual.

This materially broadens the research space. A clearer Airtherm manual documents two element
commands distinct from ordinary `ELE`/hybrid modes:

- **Boost:** while heating, a panel chord turns Boost on/off; the compressor stops (or never starts)
  and the element turns on until setpoint.
- **Manual sterilisation:** a three-key chord heats to 70 °C and holds 65–70 °C for 30 minutes,
  with a two-hour timeout if it cannot reach target.

Either command could use a different internal request path from Aquatech's ordinary
`electric @ 70` mode. A corresponding hidden Tuya datapoint is the best prospective automated
workaround; a panel test above 60 °C is the best first discriminator.

Source: [Airtherm Aqua 1.2 manual, controller and operation sections pp. 19–24](https://brookvent.ie/wp-content/uploads/2024/06/airtherm-aqua_1-2_manual-20.05.20241.pdf).

## Candidate family

| branding/model | evidence | manual value | confidence/relevance |
|---|---|---|---|
| **Aquatech RAPID/X6, DYNAMIC/X8** | Current Aquatech manual covers both; installed X6 telemetry | Exact local target | Confirmed product documentation |
| **Hydrotherm DYNAMIC/X8** | Manuals name Aquatech Solar Technologies as authority; repo history records shared X6/X8 electronics | Australian sibling documentation and service clues | High, but firmware revision may differ |
| **Hisense AH-200NH4GHB / AH-300NH4GHB** and `C` variants | Full F-code table; same diagnostics 00–16; certified/distributed by Hisense | Parameter meanings, quick heat, disinfection, multilingual newer manual, SG Ready | Strong controller-family evidence |
| **Airtherm Aqua 1.2 200/250/300 L** | Internal model table says YT-200/250/300TB2; same controller diagnostics and key-chord functions | Clearest operational description found | Strong command/state evidence; hardware differs |
| **Solareast/Sunrain YT-200/250/300TB2** | Manufacturer/product catalog uses the YT identifiers; R290, 620 mm cylinder, element, Wi-Fi/PV/Modbus features | Likely upstream product/OEM trail | Strong product-family lead; no register map found |
| **Ecostar YT-200/250/300TB2** | Exact identifiers and matching published specification table | Alternate distributor/support channel | Strong rebadge evidence |
| **Chameleon SIPH-200/250/300TB2** | Manual names both YT and SIPH identifiers | Alternate manual/support channel | Strong rebadge evidence |
| **Sacon YT-200TB2** | Exact identifier in supplier listing | Alternate brand/search term | Moderate; listing rather than service manual |
| **Emerald / Rinnai DemandDuo Tuya variants** | Similar T1–T5 and Tuya datapoint vocabulary only | Sensor/DP decoding leads | Weak for element logic |

Selected sources:

- [Solareast/Sunrain YT family catalog](https://marketdirectory.messefrankfurt.com/images/original/document_downloads/10000006202501/0015050370/1671769866768_1729264797.pdf)
- [Ecostar YT product page](https://www.ecostar.com.tr/en/products/condensing-boilers/heat-pump/heat-pump)
- [Chameleon YT/SIPH manual](https://chameleon.co.ke/wp-content/uploads/2025/08/CSL-R290-HEATPUMP-MANUAL.pdf)
- [Hisense multilingual AH-200/300NH4GHB manual index](https://www.manualslib.de/manual/1034512/Hisense-Ah-200Nh4Ghb.html)
- [Hisense SG Ready certification/model list](https://sgready.waermepumpe.de/database/?cHash=ea8e378a41fff7e3ef8686ddc329892f&tx_bwpsgreadydatabase_frontend%5Baction%5D=downloadLabel&tx_bwpsgreadydatabase_frontend%5Bcontroller%5D=Frontend&tx_bwpsgreadydatabase_frontend%5Blabel%5D=526)

## What the clearer manuals add

### Airtherm/YT controller

- Panel `M` long-press enters parameter settings; Up/Down long-press enters live status query.
- Diagnostics match the Hisense table closely: compressor 11, four-way valve 12, high fan 13,
  low fan 14, circulation pump 15, element 16.
- `ELE` is element-only with a 15–75 °C target range.
- `HYB1` uses heat pump/element logic below 60 °C, then element above 60 °C; documented restart
  differential is 5 °C.
- Boost explicitly stops/suppresses the compressor and requests the element.
- Sterilisation explicitly requests 70 °C independently of the ordinary setpoint flow.
- Smart Life device category is `Large Home Appliance → Smart Heat Pump (Wi-Fi)`.
- The product advertises PV and Modbus capability, but the sweep did **not** find a YT-family Modbus
  register map.

### Newer Hisense manual

The newer multilingual AH-200/300NH4GHB(C) manual is much better translated than the earlier
HI-WATER PDF. It documents:

- Boost as element on/off from a two-key chord while heating.
- HYB1 handover at 60 °C and a 5 °C restart differential.
- A distinct element-only mode.
- SG Ready inputs with four external grid states.

It appears to describe a newer touchscreen/firmware variant, so its key chords and mode semantics
are leads, not Aquatech instructions.

## Important non-equivalences

- Airtherm/YT specifies a **2.0 kW** element and **3.0 kW** maximum input; it describes heat pump and
  element together below 60 °C in HYB1.
- The installed Aquatech element is measured around **1.78 kW**, and available evidence says its
  heat sources are sequential/mutually exclusive on the 10 A supply.
- Hisense's related F35 default is 48 °C; Aquatech explicitly documents 55 °C.
- Tank sizes, refrigerant charges, controller panels and firmware generations vary.

Therefore copy **concepts and search vocabulary**, not factory values, wiring or safety settings.

## Workaround leads, ranked

1. **Panel Boost test:** with Aquatech tank just above 60 °C and target 70 °C, invoke the matching
   Boost chord if the physical panel/manual layout supports it. Observe element binary + circuit
   power. Do not change service parameters.
2. **Panel sterilisation test:** if Boost is absent/blocked, test the documented sterilisation chord.
   Be ready to cancel it; verify tempering and safe maximum temperature first.
3. **Full Tuya DP inventory:** compare raw Aquatech datapoints before/during a successful panel Boost
   or sterilisation event. Look for a momentary command/boolean not mapped by Local Tuya.
4. **Controller program code and PCB/display labels:** record diagnostic program code, firmware,
   PCB model and display-controller markings; search those exact identifiers across YT/Hisense docs.
5. **Ask OEM/distributors:** request the YT-200/250/300TB2 Modbus map and controller service manual
   from Solareast, Airtherm/Brookvent, Ecostar or Chameleon. Phrase the request around Boost,
   sterilisation and element output 16.

Do not use tank-sensor calibration, protection thresholds or direct element rewiring as a dump-load
workaround. Those paths can defeat temperature/safety control and are outside this research scope.

## Chinese-language search result

Searches for the F30/F35 Chinese concepts, AH model identifiers and likely translations of the
quick-heat parameters did not locate an indexed original Chinese service manual. The upstream
Solareast trail is stronger than the Chinese search results: YT identifiers persist across catalogs
and rebadged manuals. Future searches should use photographed PCB/controller identifiers rather than
generic translated phrases.
