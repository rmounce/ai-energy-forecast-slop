# HWC controller-family manual sweep — 2026-08-01

Target: find better documentation for the Aquatech RAPID/X6 controller, especially a way to start
the resistive element above its observed ~60 °C ordinary-mode re-trigger threshold. This is an
evidence map, not proof that commands or parameters are portable across products.

## Main result

The current Aquatech manual confirms that the apparent element threshold is intentional mode
policy: its mode table lists `ELEMENT - 60 °C/70 °C` (trigger/target). The observed refusal to
re-start the element above about 60 °C therefore matches the published factory behaviour.

The same Aquatech manual also confirms a separate controller-supported element path: F66 enables a
weekly legionella cycle that heats **from the current tank temperature to 70 °C with the element**
and then holds at temperature for 32 minutes. That does not provide on-demand dispatch by itself,
but it proves this Aquatech firmware can request the element independently of Element mode's 60 °C
whole-cycle trigger. F66 is now confirmed in `aquatech-settings.csv`; its installed factory value is
`0` (disabled). Enabling it is a documented configuration change, not authorised by this research
pass.

The hardware does not appear to require that restriction. Australian-certified 270 L units with
the same 1.2 kW heat-pump input, 1.8 kW element, 14 A/3 kW combined maximum, R290/400 g charge,
pressure ratings and 620 mm tank format expose direct element-only and Boost modes. Firmware and
controller policy vary even where the refrigeration/electrical platform is a close match.

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

## Aquatech factory modes and parameter hypotheses — 2026-08-02

The Aquatech table describes the trigger as the tank temperature that starts a **new heating
cycle**, not the temperature where one heat source hands over to another. It also warns that a
timer-interrupted cycle will not begin again until the selected mode's trigger is reached. This
explains why accepting a new Element/70 command above 60 °C need not energise the element.

| mode | factory target | whole-cycle trigger | target minus trigger | heat-source sequence |
|---|---:|---:|---:|---|
| ECO | 60 °C | 48 °C | 12 K | heat pump only to 60 °C |
| STANDARD | 60 °C | 55 °C | 5 K | heat pump only to 60 °C |
| HYBRID | 65 °C | 55 °C | 10 K | heat pump to 60 °C, then element to 65 °C |
| HYBRID+ | 70 °C | 50 °C | 20 K | heat pump to 60 °C, then element to 70 °C |
| ELEMENT | 70 °C | 60 °C | 10 K | element only to 70 °C |

Primary sources: [current Aquatech owner manual, heating-mode table pp. 29–30](https://www.aquatechheatpumps.com.au/_files/ugd/228c32_84eb2f4659664c0e92f0dd0fe910d8fa.pdf),
[Aquatech RAPID/X6 design document, controller section pp. 33–34](https://www.aquatechheatpumps.com.au/_files/ugd/228c32_d38fa4bab7d74dbe8380bf3856f37858.pdf).

The table helps interpret the installed F-code snapshot, but does **not** produce a defensible
mode-to-F-code map yet:

- F01=`60`, F03=`5` and F08=`60` align with the related service table's global target, global
  water-control differential and heat-pump maximum respectively. They also align with STANDARD's
  60/55 behaviour. This is family-manual support, not proof that F03 governs every Aquatech mode.
- The differences `12`, `5`, `10` and `20` all occur in F71–F100, including F83/F86=`12`,
  F80/F96=`5`, F88=`10` and F87=`20`. The repetitions and many unrelated controller settings with
  the same ordinary values make a positional assignment unsafe. Treat the matches as search
  fingerprints only.
- F69=`7` and F95=`30` superficially match related firmware's seven-day disinfection interval and
  30-minute hold. Aquatech documents a 32-minute hold, and the related manuals identify F67—not
  F69—as an automatic-disinfection setting. These are low-confidence coincidences until a complete
  table or controlled observation supports them.
- F101–F116 split naturally into 6/5/5 monotonic groups, and their `150–500` magnitude resembles
  EEV opening steps (the controller diagnostics expose roughly `100–480`; known F53/F56 EEV values
  are `400`/`350`). They are therefore more plausibly refrigeration lookup curves than mode
  temperatures. This is a moderate structural hypothesis, not a parameter interpretation.

Changing the user target does not yet discriminate fixed trigger tables from target-relative
deadbands. Hydrotherm's matching Tuya guide says the set temperature overrides a mode preset while
the tank temperature triggers reheating “based on the mode”, which leans toward mode-specific fixed
triggers. Direct Aquatech behaviour at altered targets remains the authority.

## Stronger panel-command evidence — 2026-08-02

The current official Hisense AHS-210/270HF4GHB manual shows the same five-button controller face,
display layout, diagnostic indices 00–22 and five-mode vocabulary as Aquatech. It explicitly assigns:

- **`M + Up`, hold 3 seconds while heating:** toggle Boost; compressor stops or stays off and the
  element turns on until target.
- **`Power + Clock + Down`, hold 5 seconds while on:** toggle manual sterilisation; heat to 70 °C,
  hold 65–70 °C for 30 minutes, then exit (two-hour failure timeout).

Source: [official Hisense AHS-210/270HF4GHB installation guide, controller and operation pp. 19–24](https://dtc-aus-api.hisense.com/medias/AHS-210HF4GHB-IG.pdf?context=bWFzdGVyfG1hbnVhbHwzMDM0NjQ3fGFwcGxpY2F0aW9uL3BkZnxhR1ExTDJnM05pODRPRFkzT0RJd056WTFNakUwTDBGSVV5MHlNVEJJUmpSSFNFSXRTVWN1Y0dSbXw1NTE3M2MwZThiYWUxZDRmMWMxNWJiNjhiNjFkZmUxZDMyNmYyNTExYjlkOTU3ZGFmNmNmOTg3MjNlYjdmMmY3).

The Airtherm and Power Bay manuals independently document the same chords and state transitions on
the same controller layout. Aquatech omits these chords and uses a different parameter-entry hold
time, so they remain candidate firmware functions rather than Aquatech instructions. The visual,
diagnostic and behavioural match nevertheless makes `M + Up` the strongest read-only-observation /
controlled-test lead for an immediate element request.

### Live trigger-boundary result — 2026-08-02

The owner authorised supported-mode experiments while the installed X6 was idle at a displayed
60 °C. The event-driven HWC daemon was stopped first so it could not overwrite commands.

| initial state | single compound command | observed result |
|---|---|---|
| off, 60 °C, both relays off, ~1.8 W | `performance @ 70 °C` | accepted but remained idle |
| performance armed, 60 °C, still idle | `electric @ 70 °C` | element on; compressor off; ~1.79–1.81 kW |
| electric running at 60 °C | change target to 61 °C | normal element cut-out at displayed 61 °C |
| electric idle at 61 °C | re-arm `electric @ 70 °C` | accepted but remained idle at ~1.9 W |

This directly confirms the current Aquatech table's mode-specific whole-cycle interpretation:
HYBRID+ does not start a new cycle at 60 °C because its trigger is 50 °C, whereas ELEMENT starts at
its inclusive 60 °C trigger and will not re-trigger at 61 °C. It also creates a clean state for the
next discriminator: `electric/70` armed at 61 °C with both heat sources off, followed by the
candidate `M + Up` Boost chord. **That chord has not yet been attempted.**

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
| **Hisense AHS-270HF4GHB** | Exact 270 L electrical/refrigerant/pressure specification match under SAA-231065-EA | Current Australian manual and ConnectLife implementation | Very strong hardware-platform lead |
| **Aether HP270** | Exact specification match under SAA-240641-EA; detailed 45-page manual | Explicit simultaneous Boost and direct element-only mode | Very strong hardware and alternate-firmware evidence |
| **Power Bay PB-270RE** | Exact specification match under SAA-231203-EA; detailed manual | STAN/HYB1/ELE behaviour, Boost chord and diagnostics | Very strong hardware and alternate-controller evidence |
| **Soltaro HPWSTR003** | Exact specification match under SAA-231204-EA; detailed manual | Five modes, direct ELE, Boost and sterilisation | Very strong hardware and alternate-controller evidence |
| **Viessmann Vitocal 161-A 270 SOC/SO** | Exact ERAC specification match under SAA-241129-EA | Alternate service/support channel | Strong hardware lead; useful manual not located |
| **Warmth WNZ270L-2in1** | Exact ERAC specification match under SAA-240388-EA | Alternate service/support channel | Strong hardware lead; useful manual not located |
| **Emerald / Rinnai DemandDuo Tuya variants** | Similar T1–T5 and Tuya datapoint vocabulary only | Sensor/DP decoding leads | Weak for element logic |

Selected sources:

- [Current Aquatech RAPID/X6 and DYNAMIC/X8 manual](https://www.aquatechheatpumps.com.au/_files/ugd/228c32_84eb2f4659664c0e92f0dd0fe910d8fa.pdf)
- [Solareast/Sunrain YT family catalog](https://marketdirectory.messefrankfurt.com/images/original/document_downloads/10000006202501/0015050370/1671769866768_1729264797.pdf)
- [Ecostar YT product page](https://www.ecostar.com.tr/en/products/condensing-boilers/heat-pump/heat-pump)
- [Chameleon YT/SIPH manual](https://chameleon.co.ke/wp-content/uploads/2025/08/CSL-R290-HEATPUMP-MANUAL.pdf)
- [Hisense multilingual AH-200/300NH4GHB manual index](https://www.manualslib.de/manual/1034512/Hisense-Ah-200Nh4Ghb.html)
- [Hisense SG Ready certification/model list](https://sgready.waermepumpe.de/database/?cHash=ea8e378a41fff7e3ef8686ddc329892f&tx_bwpsgreadydatabase_frontend%5Baction%5D=downloadLabel&tx_bwpsgreadydatabase_frontend%5Bcontroller%5D=Frontend&tx_bwpsgreadydatabase_frontend%5Blabel%5D=526)
- [Aether HP200/HP270 manual](https://aetheraustralia.com.au/wp-content/uploads/2025/03/userManualFinals.pdf)
- [Power Bay PB-270RE manual](https://powerbay.com.au/wp-content/uploads/2024/09/Power-Bay-PB270RE-Heat-Pump-User-Manual.pdf)
- [Soltaro HPWSTR002/003 manual](https://australia.a.bigcontent.io/v1/static/14361239_1_P_PROD_DET_SoltaroASHPManual)
- [Hisense AHS-270HF4GHB product/manual page](https://hisense.com.au/product/AHS-270HF4GHB/270l-heat-pump-hot-water)

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

### Exact-spec Australian siblings

The certification match is more probative than appearance alone. Aquatech Dynamic/X8 and the six
270 L candidates above share the principal nameplate values. It does not prove identical PCB,
sensor placement or firmware, but makes their manuals the best source of alternate control logic.

- **Power Bay:** STAN is heat-pump-only; HYB1 says heat pump and element work together; ELE is
  element-only with a 5 °C restart differential. Its text says a 15–75 °C range while the adjacent
  summary chart says 15–70 °C, an internal manual inconsistency. Its Boost chord turns the element
  on until target. Neither description imposes a 60 °C minimum before ELE or Boost can operate.
- **Soltaro:** ECO has a 12 °C restart differential; HYB/HYB1 hand over from compressor to element at
  60 °C; ELE is element-only from 15–70 °C with a 10 °C restart differential. A separate Boost chord
  turns the element on until setpoint. This is especially useful: `ELE` hysteresis and the hybrid
  60 °C handover are separate concepts, whereas Aquatech packages 60/70 as the Element mode.
- **Aether:** factory STAN is 55 °C with 5 K deadband. The table says its 270 L element setting is
  61 °C/5 K and is used below -7 °C. User-selectable `booS` runs heat pump and element together;
  `ELE` is element-only, both with 15–75 °C target range. Non-standard modes revert within 24 hours.
- **Hisense AHS:** current Australian product literature advertises multiple modes to 70 °C,
  ConnectLife control and weekly sterilisation. The public 38-page installation guide was found,
  but indexed copies expose less control detail than the manuals above.

These mutually inconsistent policies on matching hardware are evidence that a controller command,
firmware option or hidden datapoint could work around Aquatech's 60 °C trigger. They are not evidence
that another brand's parameter values or firmware can safely be copied.

## Important non-equivalences

- Airtherm/YT specifies a **2.0 kW** element and **3.0 kW** maximum input; it describes heat pump and
  element together below 60 °C in HYB1.
- The installed Aquatech element is measured around **1.78 kW**, and available evidence says its
  heat sources are sequential/mutually exclusive on the 10 A supply.
- Hisense's related F35 default is 48 °C; Aquatech explicitly documents 55 °C.
- Tank sizes, refrigerant charges, controller panels and firmware generations vary.

Therefore copy **concepts and search vocabulary**, not factory values, wiring or safety settings.

## Workaround leads, ranked

1. **Panel Boost discriminator:** the Aquatech panel layout matches the official Hisense controller
   that uses `M + Up` for Boost. If the owner later authorises a controlled test, use a tank just
   above 60 °C and a target below the mechanical thermostat limit; observe the physical element
   binary and circuit power. A changed/flashing element icon is supporting UI evidence, not proof of
   load. Do not change service parameters.
2. **Aquatech F66 weekly cycle:** manufacturer-confirmed to request the element from current
   temperature to 70 °C, so it is the safest supported proof of an above-60 element path. It is
   scheduled rather than on demand; whether enabling F66 starts immediately or only after its
   internal interval is unknown. No change is authorised here.
3. **Panel manual-sterilisation discriminator:** the matching-controller chord is
   `Power + Clock + Down` for five seconds. It is less attractive than Boost because it deliberately
   targets 70 °C and holds temperature. Aquatech documents weekly sterilisation but not this manual
   chord. Verify tempering and cancellation behaviour before any later test.
4. **Full Tuya DP inventory:** compare raw Aquatech datapoints before/during a successful panel Boost
   or sterilisation event. Look for a momentary command/boolean not mapped by Local Tuya.
5. **Controller program code and PCB/display labels:** record diagnostic program code, firmware,
   PCB model and display-controller markings; search those exact identifiers across YT/Hisense docs.
6. **Ask OEM/distributors:** request the YT-200/250/300TB2 Modbus map and controller service manual
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

## Other desktop findings and dead ends

- Aquatech's older *Hydrotherm & your Solar PV v2.0* describes timer scheduling against ordinary
  hysteresis, not a PV/SG dry-contact or element override. It says Constant/Timer modes reheat below
  55 °C and treats Element Booster as extra capacity for large loads.
- An indexed `Dynamic X8 Gen 2 Settings Screen Manual` filename was found, but its original Aquatech
  PDF URL no longer resolves through the public site. Archive/vendor recovery remains worthwhile.
- No public YT-family Modbus register map, Aquatech Tuya schema, service firmware image, or original
  Chinese controller manual was located.
- The public certification records establish nameplate equivalence but do not identify the OEM,
  PCB, controller supplier or firmware lineage. Hisense and Solareast remain strong trails, not a
  proven manufacturer attribution for the installed Aquatech.
- A SolarEast R290 Home Assistant integration found online is for a monobloc space-heating product;
  its registers should not be assumed portable to this tank controller.

### Best remaining no-device actions

1. Recover the missing Aquatech `Dynamic X8 Gen 2 Settings Screen Manual` from web archives or
   Aquatech/Hydrotherm support.
2. Request service/controller manuals and Tuya datapoint lists for Aquatech X6/X8, Hisense
   AHS-210/270HF4GHB, Aether HP270, PB-270RE and HPWSTR003. Ask specifically whether Boost/ELE may
   start above 60 °C and whether output 16 is externally commandable.
3. Request the YT-200/250/300TB2 Modbus register map from Solareast/Sunrain and distributors.
4. When device access returns, identify the panel/PCB/program code before trying any sibling key
   chord. That identifier is now more valuable than another broad model-name search.
