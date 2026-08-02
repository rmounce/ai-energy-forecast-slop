# HWC controller-family manual sweep — 2026-08-01

Target: find better documentation for the Aquatech RAPID/X6 controller, especially a way to start
the resistive element above its observed ~60 °C ordinary-mode re-trigger threshold. This is an
evidence map, not proof that commands or parameters are portable across products.

## Main result

The installed Aquatech **does implement the matching-controller Boost chord**. At 61 °C, `M + Up`
for three seconds latched Boost while the unit was on in STANDARD/60. Switching to HYBRID+/70 then
closed the element relay immediately: compressor off, element on, about 1.795 kW. The latch survived
remote mode changes, allowing HA to dispatch the element above 60 °C with HYBRID+/70 and suppress it
with STANDARD/60. `turn_off` cleared the latch. The latch itself is controller-local and did not
appear in any of the 50 reported Tuya DPs.

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

Boost uses a different internal request path from ordinary `electric @ 70` mode. It is now
confirmed on Aquatech, although a physical chord is required to establish the hidden latch after
each `turn_off`. Manual sterilisation remains unobserved.

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

On 2026-08-02 the owner visually confirmed that the current Australian Hisense AHS and German Tech
YT-200/270TD2 panels appear identical to the installed Aquatech controller. The Australian
Viessmann Vitocal 161-A shares the mechanical platform but uses a different colour touchscreen.
Emerald uses an apparently identical controller on a visibly different, squarer tank/chassis. Panel
identity strengthens the Hisense/German Tech/Emerald controller-document trail, but does not prove
an identical PCB, firmware revision, parameter set, Tuya mapping or complete-appliance OEM.

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
its inclusive 60 °C trigger and will not re-trigger at 61 °C.

### Live panel-chord result — 2026-08-02

The owner performed both candidate family chords at an idle 61 °C:

- `M + Up` for three seconds beeped and briefly flashed the element icon, then returned to idle.
  HA continued to report both relays off and circuit power around 1.9 W. This is evidence that the
  controller recognised the chord, but not a valid Boost test: the family instructions require an
  already-running heating cycle.
- `Power + Clock + Down` for five seconds beeped without a visible display change. Immediate and
  delayed HA checks remained `electric/70`, both relays off and around 1.9 W. This first attempt was
  not a valid family-protocol discriminator because the related manuals inhibit sterilisation when
  the ordinary target is at least 70 °C.
- The owner repeated `Power + Clock + Down` with the controller on in STANDARD/60 but idle at
  61 °C. There was no display or power response. This satisfies the related protocol's on-state and
  target-below-70 preconditions but still did not start sterilisation. The matching-family chord is
  therefore not an observed Aquatech command.

The first Boost attempt obscured the result because ELECTRIC already had a 70 °C target but did not
retain the visible latch. A controlled retest established the actual state machine:

| action/state | observed result |
|---|---|
| STANDARD/60 on and idle at 61 °C; hold `M + Up` 3 s | beep; element icon flashed continuously; no relay/power rise because target was already satisfied |
| while latched, select HYBRID+/70 at panel | element icon solid and relay closed immediately; compressor off; element on; ~1.795 kW |
| remote compound STANDARD/60 | element off and ~1.8 W; icon off, but hidden Boost latch persisted |
| remote compound HYBRID+/70 | element restarted immediately at 61 °C; compressor remained off; ~1.795 kW |
| while latched in HYBRID+/60, target-only change 60→70 | target was accepted and icon kept flashing, but relay/power remained off; target change alone did not re-evaluate Boost |
| then remote STANDARD/60 → HYBRID+/70 | icon went flashing→off in STANDARD, then solid in HYBRID+ as the relay closed and element started (~1.775 kW), proving the hidden latch persisted |
| remote `turn_off`, then re-arm HYBRID+/70 | first command stopped the element and cleared Boost; re-arm remained idle |

This is a supported on-demand element path above the ordinary 60 °C trigger. For remote dispatch,
Boost must first be latched physically while the controller is on. HA can then use STANDARD/60 as
the no-element state and HYBRID+/70 as the element state while the tank is above 60 °C. Any
`turn_off` clears the latch and requires the physical chord again. The latch is invisible through
the reported Tuya DPs; infer it only from controlled transitions and verified element power. A
target-only increase does not dispatch the latched element: use a genuine mode transition into
HYBRID+/70.

### Live F66 enable result — 2026-08-02

The owner enabled the documented F66 legionella setting (`0` to `1`) while the controller was on in
STANDARD/60 but thermally idle at 61 °C. There was no visible panel response. Immediate and
one-controller-interval HA checks remained at 61 °C with both relays off and circuit power around
1.9 W. Enabling F66 therefore does not immediately start disinfection in this state. It remains a
weekly cadence enable, not an observed on-demand command; the result does not reveal whether its
internal counter starts or resets when F66 is enabled.
The owner then restored F66 to its installed value `0`; the controller remained idle.
The panel exposes only time-of-day, not a calendar date or weekday, so there is no documented safe
way to advance the weekly counter for an immediate test.

A subsequent read-only raw Tuya diff captured all 50 DPs reported by the device while F66 was saved
from `0` to `1`. No DP changed; only two duplicate temperature signals moved naturally by 1 °C.
F66 is therefore not represented in the currently reported local DP set. The official Tuya cloud
schema likewise exposes only power, target temperature, temperature-unit and defrost writes—no
Boost, sterilisation or direct-element command. F66 was restored to `0` after the capture. See
`aquatech_entities.md` for the raw inventory and unmapped-DP notes.

## Candidate family

| branding/model | evidence | manual value | confidence/relevance |
|---|---|---|---|
| **Aquatech RAPID/X6, DYNAMIC/X8** | Current Aquatech manual covers both; installed X6 telemetry | Exact local target | Confirmed product documentation |
| **Hydrotherm DYNAMIC/X8** | Manuals name Aquatech Solar Technologies as authority; repo history records shared X6/X8 electronics | Australian sibling documentation and service clues | High, but firmware revision may differ |
| **Hisense AH-200NH4GHB / AH-300NH4GHB** and `C` variants | Full F-code table; same diagnostics 00–16; certified/distributed by Hisense | Parameter meanings, quick heat, disinfection, multilingual newer manual, SG Ready | Strong controller-family evidence |
| **Airtherm Aqua 1.2 200/250/300 L** | Internal model table says YT-200/250/300TB2; same controller diagnostics and key-chord functions | Clearest operational description found | Strong command/state evidence; hardware differs |
| **Solareast/Sunrain YT-200/250/300TB2** | Manufacturer/product catalog uses the YT identifiers; R290, 620 mm cylinder, element, Wi-Fi/PV/Modbus features | Related controller branch | Strong controller-family lead; mechanical platform differs |
| **SolarEast/Sunrain YT-200/270TD2; SIHP-200/270TD2** | Exact Australian hardware tuple; SolarEast ERAC/WaterMark registrations and Sunrain OEM listing | Upstream complete-unit OEM trail | Very strong; see `oem_manufacturer_research.md` |
| **Ecostar YT-200/250/300TB2** | Exact identifiers and matching published specification table | Alternate distributor/support channel | Strong rebadge evidence |
| **Chameleon SIPH-200/250/300TB2** | Manual names both YT and SIPH identifiers | Alternate manual/support channel | Strong rebadge evidence |
| **Sacon YT-200TB2** | Exact identifier in supplier listing | Alternate brand/search term | Moderate; listing rather than service manual |
| **Hisense AHS-270HF4GHB** | Exact 270 L electrical/refrigerant/pressure specification match under SAA-231065-EA; owner confirms identical-looking panel | Current Australian manual and ConnectLife implementation | Very strong hardware and controller-panel lead |
| **German Tech YT-200/270TD2** | Exact SolarEast TD2 platform; owner confirms identical-looking panel | Alternate manual/support channel | Very strong hardware and controller-panel lead |
| **Aether HP270** | Exact specification match under SAA-240641-EA; detailed 45-page manual | Explicit simultaneous Boost and direct element-only mode | Very strong hardware and alternate-firmware evidence |
| **Power Bay PB-270RE** | Exact specification match under SAA-231203-EA; detailed manual | STAN/HYB1/ELE behaviour, Boost chord and diagnostics | Very strong hardware and alternate-controller evidence |
| **Soltaro HPWSTR003** | Exact specification match under SAA-231204-EA; detailed manual | Five modes, direct ELE, Boost and sterilisation | Very strong hardware and alternate-controller evidence |
| **Viessmann Vitocal 161-A 270 SOC/SO** | Exact ERAC specification match under SAA-241129-EA; visibly different colour touchscreen | Mechanical/service support only | Strong hardware lead; weak for Aquatech UI behaviour |
| **Warmth WNZ270L-2in1** | Exact ERAC specification match under SAA-240388-EA | Alternate service/support channel | Strong hardware lead; useful manual not located |
| **Emerald** | Owner confirms identical-looking controller panel; tank/chassis is visibly different and squarer | Controller and sensor/DP decoding leads only | Strong panel-family lead; weak mechanical/OEM evidence |
| **Rinnai DemandDuo Tuya variants** | Similar T1–T5 and Tuya datapoint vocabulary | Sensor/DP decoding leads | Weak for element logic |

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

The certification match is more probative than appearance alone. Aquatech Dynamic/X8 and the
270 L candidates above share the principal nameplate values. More decisively, a 2020 Aquatech TÜV
report names Guangdong Sunrain Air Source Energy Co., Ltd. as the RAPID/X6 and Dynamic/X8 factory,
while SolarEast registers the exact YT-200/270TD2 platform in Australia. This establishes the
complete-unit OEM trail, but not identical PCBs, sensor placement or firmware across every rebadge.
See `oem_manufacturer_research.md`.

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

1. **Confirmed panel Boost:** `M + Up` establishes a controller-local latch. Remote STANDARD/60 and
   HYBRID+/70 then suppress/start the element above 60 °C; `turn_off` clears the latch. The remaining
   implementation question is whether an operational policy can safely avoid off while retaining
   reliable normal heat-pump scheduling.
2. **Aquatech F66 weekly cycle:** manufacturer-confirmed to request the element from current
   temperature to 70 °C, so it is the safest supported proof of an above-60 element path. It is
   scheduled rather than on demand; whether enabling F66 starts immediately or only after its
   internal interval is unknown. No change is authorised here.
3. **Panel manual-sterilisation discriminator:** the matching-controller chord is
   `Power + Clock + Down` for five seconds. It is less attractive than Boost because it deliberately
   targets 70 °C and holds temperature. Aquatech documents weekly sterilisation but not this manual
   chord. Verify tempering and cancellation behaviour before any later test.
4. **Tuya command discovery:** the full reported-DP inventory and live Boost/F66 diffs found no latch
   DP. Further work requires an upstream command schema, protocol trace or controller documentation;
   do not brute-force unknown writes.
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
- Hardware OEM attribution is now strong: an original Aquatech TÜV report names Guangdong Sunrain,
  and SolarEast registers the exact YT-200/270TD2 Australian platform. The installed unit's current-
  batch factory, PCB supplier and firmware lineage remain unverified; see
  `oem_manufacturer_research.md`.
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
