# Aquatech hardware/OEM attribution

Research status: **2026-08-02**. Scope: RAPID/X6 and related Australian R290 all-in-one HPWHs.
Controller attribution is secondary.

## Conclusion

**Factory/OEM: Guangdong Sunrain Air Source Energy Co., Ltd., Shunde, Foshan.** This is a
SolarEast Group company/factory also presented internationally as SolarEast Heat Pump Ltd. The
attribution is direct for the 2020 RAPID/X6 and Dynamic/X8, and strongly supported for the current
Gen 2/Gen 6 hardware family.

Do not reduce this to “made by Viessmann”, “made by Hisense” or merely “a generic YT unit”. Those
are customer brands/branches of a SolarEast/Sunrain OEM platform. Current-batch manufacturing has
not been checked from the installed unit's production record, so retain that narrow caveat.

## Direct evidence

- A [2020 TÜV technical report for RAPID/X6 and Dynamic/X8](https://www.mcmplumbing.com.au/wp-content/themes/mcm/templates/Rapid-x6-docs%26reports/usermanual.pdf?_t=1671625712)
  names:
  - client/manufacturer: Aquatech Solar Technologies Pty Ltd, Australia;
  - **factory: Guangdong Sunrain Air Source Energy Co., Ltd.**;
  - factory address: No. 73 Defu Road, Xingtan Town, Shunde District, Foshan, Guangdong, China.
- SolarEast's Australian ERAC registration `SAA-230971-EA` names
  [SolarEast Heat Pump Ltd. as applicant](https://device.report/erac/SAA-230971-EA) for
  `YT-200TD2` and `YT-270TD2`. It records 1.2 kW/5.3 A heat-pump input, 1.8 kW/7.5 A element,
  R290/400 g, IPX4, and 200/270 L tanks.
- Australian WaterMark certificate `026823` lists
  [SolarEast `YT-270TD2`](https://watermark.abcb.gov.au/product-search/product/512811) as an
  integral heat-pump water heater.
- Sunrain sells the same unit explicitly as an OEM/ODM product. Its
  [SIHP-200TD2/SIHP-270TD2 factory page](https://en.sunrain.com/r290-monoblock-air-to-water-heat-pump-water-heater/)
  gives the complete Australian-platform tuple: 620 x 1518/1838 mm, 104/118 kg, 2.78 kW heat,
  COP 4.15, 1.2 kW/5.3 A heat-pump input, 1.8 kW/7.5 A element, 3.0 kW/14 A maximum, R290/400 g,
  3.0 MPa, 850 kPa TPR, IPX4 and 43 dBA. Sunrain also markets it as
  [`YT-200TD2/YT-270TD2`](https://sunrain.en.made-in-china.com/product/YwUAPxeuvgaK/China-Sunrain-Water-Mark-SAA-Stc-Certified-Electric-All-in-One-Heat-Pump-Water-Heater.html).

## Corporate/factory identity

Names encountered refer to the same group or operating lineage, not four independent candidates:

| name | role/evidence |
|---|---|
| **SolarEast Group / SolarEast Holdings, stock 603366** | Parent group; owns Sunrain and Micoe brands and multiple manufacturing bases. |
| **Guangdong Sunrain Air Source Energy Co., Ltd.** | Legal factory name in the Aquatech TÜV report; Shunde address. |
| **SolarEast Heat Pump Ltd.** | International/OEM applicant and trading identity at the same Shunde address. |
| **Jiangsu Sunrain Solar Energy Co., Ltd.** | Group/export entity listing the exact YT-200/270TD2 product. |
| **Sunrain / Micoe / Sacon** | Group brands, not evidence of different factories. |

The [China Chamber of Commerce machinery/electronics company profile](https://www.cccme.cn/shop/cn20201208128/introduction.aspx)
says the Shunde operation has domestic/commercial heat-pump, water-tank, controller and metal-
cabinet production lines. SolarEast says it operates both Shunde and Lianyungang heat-pump bases.
The 2020 Aquatech report identifies **Shunde** for the tested units; a current unit's serial/batch
record would be needed to prove which group plant assembled it.

## Hardware layers

### Tank and complete appliance

Leading attribution: **Guangdong Sunrain/SolarEast, very high confidence**.

- Direct factory statement in the Aquatech TÜV report.
- Exact SolarEast model registered in Australia.
- Sunrain advertises both complete appliances and OEM/ODM customization.
- Factory profile claims its own water-tank and cabinet lines.
- Aquatech specifies enamelled carbon steel, 3.0 mm domes/2.5 mm wall, concave top/convex base,
  40 mm polyurethane, wrap-around microchannel condenser and impressed-current anode.

Remaining narrow alternative: Sunrain could subcontract a pressure-vessel batch or use another
SolarEast group plant. Its OEM page also advertises top-part kits for mating to a customer's local
tank. Nothing found supports that arrangement for Aquatech; the direct factory report and claimed
in-house tank line point the other way.

### Refrigeration package

Integrator: **Guangdong Sunrain/SolarEast, very high confidence**. The same factory is named for the
complete tested appliance and advertises the exact assembled refrigerant package.

Known component origins:

- compressor: **GMCC rotary**, confirmed by both the Aquatech manual and SolarEast/Sunrain product
  material. GMCC is Guangdong Meizhi Refrigeration Equipment Co., Ltd., part of Midea's component
  business; “GMCC Toshiba” describes its Toshiba-derived/JV lineage, not a Toshiba-built appliance;
- condenser: aluminium-alloy microchannel wrap around the tank; component maker not identified;
- evaporator: 420 x 350 mm, three-row copper on current Aquatech documentation;
- expansion/defrost: EEV plus four-way valve; suppliers not identified;
- element: 1.8 kW flange/Incoloy; Aquatech names a Robertshaw thermostat, not the element maker.

### Controller/firmware — secondary

Owner visual comparison on 2026-08-02 establishes two visible controller branches on the same
mechanical platform:

- **Hisense AHS and German Tech YT-200/270TD2:** control panel appears identical to the installed
  Aquatech panel. Treat their controller manuals as especially relevant, subject to firmware and
  parameter-default differences.
- **Viessmann Vitocal 161-A:** uses a visibly different, colour touchscreen controller despite the
  matching tank/heat-pump chassis. Its UI behaviour is not transferable to Aquatech.

An identical panel does not by itself prove identical main PCB, firmware build or Tuya datapoints.

Candidates, in order:

1. **Sunrain/SolarEast in-house controller operation.** Its factory profile claims a controller
   production line. The official Hisense PDF retains internal document identifiers
   `14400003001340` and `TY-YS-05-SM415`; the same numbering family occurs in Micoe/SolarEast group
   manuals. This supports SolarEast document/engineering provenance, but does not identify the PCB
   designer.
2. **Unidentified specialist control-board supplier.** Similar F-code/UI behaviour spans products
   with different mechanical platforms, so Sunrain may manufacture or populate a supplier design.
   PCB silkscreen and MCU markings are required to resolve this.
3. **Tuya.** Confirmed communications/cloud layer on Aquatech, but no evidence Tuya designed the
   refrigeration controller or mode logic.

## Related Australian brands

These are useful evidence of OEM customization, not competing manufacturer attributions:

| brand/model | relationship evidence |
|---|---|
| **Aquatech RAPID/X6, Dynamic/X8; Hydrotherm** | Direct Sunrain factory report. Current X6/X8 are customized size/performance/firmware branches. |
| **SolarEast YT-200/270TD2** | OEM's own registered Australian model. |
| **Viessmann Vitocal 161-A 210/270 SO/SOC** | Same 620 x 1518/1838 mm chassis and exact certification tuple. Owner visually identifies the Aquatech tank/body, but Viessmann uses a different colour touchscreen. |
| **Hisense AHS-210/270HF4GHB/C** | Same dimensions/specification; owner identifies the control panel as identical to Aquatech. Official PDF retains SolarEast/Micoe-like internal document numbering. |
| **German Tech YT-200/270TD2** | Exact SolarEast model family; owner identifies the control panel as identical to Aquatech. |
| **Aether HP200/HP270, Power Bay PB-270RE, Soltaro HPWSTR002/003, Warmth WNZ270L-2in1, Eurosun, Versopump** | Exact or near-exact Australian specification and mechanical platform; alternate controls/firmware exist. |

The older European 308 L Vitocal 161-A is unrelated. Comparisons must use the current Australian
200/270 L product.

## Visual comparison links

- [Sunrain/SolarEast exact OEM product and image gallery](https://sunrain.en.made-in-china.com/product/YwUAPxeuvgaK/China-Sunrain-Water-Mark-SAA-Stc-Certified-Electric-All-in-One-Heat-Pump-Water-Heater.html)
- [Sunrain SIHP-200/270TD2 factory page](https://en.sunrain.com/r290-monoblock-air-to-water-heat-pump-water-heater/)
- [Australian Viessmann Vitocal 161-A](https://www.viessmann.com.au/en/products/heat-pump/vitocal-161-a.html)
- [Aquatech RAPID/X6](https://www.aquatechheatpumps.com.au/product-page/rapid-x6)
- [Hisense AHS-270HF4GHB](https://hisense.com.au/product/AHS-270HF4GHB/270l-heat-pump-hot-water)
- [German Tech YT-200/270TD2 brochure](https://www.germantech.com.au/_files/ugd/ee60cf_b092a2b10ed7470c9b49669ccdf0a48f.pdf)
- [Power Bay PB-270RE dimensional sheet](https://powerbay.com.au/wp-content/uploads/2024/09/PB-270RE-Heat-Pump-Datasheet-.pdf)

Diagnostic visual features: shoulder/top moulding, opposing horizontal air grilles, lower service
hatch, base ring/feet, condensate outlet, hot/cold/PTR port heights, anode ports and fastener layout.

## Best remaining proof

1. Read the installed unit's factory/serial label and photograph PCB, compressor and tank labels at
   its next service—not worth reopening solely for this.
2. Ask Aquatech for the current Gen 2 factory model cross-reference and replacement-parts catalogue.
3. Obtain the full current SAA/WaterMark test reports; public registers usually expose the Australian
   applicant rather than the factory.
