# NEM counter-price flows: 6 October 2026 investigation

- Scope: public data feasibility + current NSW/VIC/SA case; no website built.
- Latest fetched dispatch: interval ending 2026-10-06 08:30 AEST / 09:00 Adelaide ACDT. Published approximately 08:25 AEST. Earlier cross-check: 08:20 AEST.
- Market timestamps: AEST year-round. Match prices, flows and constraint solutions by interval and intervention flag.
- Working tree clean before investigation. Downloads/parsed data: `/tmp/nem-*`; temporary investigation artifacts, not committed.

## Observed dispatch

| Region | RRP ($/MWh), 08:30 AEST |
| --- | ---: |
| NSW | 41.65839 |
| VIC | -14.38946 |
| SA | -5.99994 |

| Link | Dispatch target | Initial metered flow | Binding limit setter |
| --- | ---: | ---: | --- |
| EnergyConnect, NSW to SA | 100 MW | 75.71094 MW | Export: `NS_100_DYN`; opposite limit: `N>>6CGH_060_051` |
| VNI, NSW to VIC | 61.76219 MW | 270.39844 MW | `N^^N_6CGH_WGLT` |
| Heywood, VIC to SA | 326.57762 MW | 244.45313 MW | Export limit 433.43515 MW; target below limit |
| Murraylink, VIC to SA | 153.1396 MW | 158 MW | `S>NIL_MHNW1_MHNW2`; opposite limit: `N>>6CGH_060_051` |

- VNI convention: positive `VIC1-NSW1` = VIC to NSW. Target and export limit both **-61.76219 MW**. Given the other dispatch quantities, Wagga constraint restricts VIC→NSW so far that NSW→VIC is required.
- Targets differ from metered flows; don't present target as measured flow. Both NSW exports are counter-price in this snapshot.
- 08:20 cross-check: NSW 28.18895, VIC -13.29223, SA -7.01 $/MWh; NSW→SA target 100 MW, NSW→VIC target 107.3098 MW.

## Binding equations and interpretation

| Equation, 08:30 | Marginal value | Meaning / role |
| --- | ---: | --- |
| `N^^N_6CGH_WGLT` | -155.72019 | Wagga voltage-stability family; directly sets VNI counter-price target boundary. Family protects against voltage collapse following loss of Wagga–Lower Tumut 051. Outage variant identified through change report. |
| `N>>6CGH_060_051` | -927.82871 | Jindera–Wodonga 060 thermal-overload family for loss of Wagga–Lower Tumut 051; October presentation explicitly gives negative PEC coefficient, so NSW→SA exports relieve this constraint. Active outage variant. |
| `NS_100_DYN` | -680.88566 | PEC discretionary export cap at 100 MW. Limits export; does not by itself explain why export is desirable. Both directions' calculated boundaries are 100 MW in this solution. |
| `L_PEC_X_6C_6G_6H` | +22.7762 | Physical loop relationship for initial PEC configuration. Equality equations bind by design; not evidence of an overloaded line. Excluded from AEMO interconnector limit-setter calculation. |
| `V^^V_NIL_KGTS` | -846.3246 | Kerang voltage-stability constraint; active at 08:30. Earlier 08:20 used `V^^V_6CGH_KGTS` (-728.22764). |
| `S>NIL_MHNW1_MHNW2` | -802.27192 | Avoid Monash–North West Bend #2 overload after loss of #1; limits Murraylink VIC→SA. |

- Listed constraints: zero reported violation degree. Binding is different from violating.
- 2026 week 40 change report: `N-X_6C_6G_6H` = outage set for Buronga–Dinawan lines 6C, 6G, 6H. New loop constraints and PEC discretionary constraints effective 1 October.
- AEMO added separate `NSW1-SA1` dispatch interconnector 1 October. Strong timing/context link to user's observation; no before/after frequency analysis performed.
- Plain English: the NSW price describes its regional reference point. Southern NSW generation cannot freely reach all NSW demand. Exporting to SA helps relieve southern NSW security constraints; increasing VIC→NSW transfers would worsen the binding Wagga constraint. Physical loop equations couple the routes. Hence regional price order alone cannot predict each arrow.
- Local adjustment evidence: Bomen/Glenellen/Wagga North solar adjustments -1041.66; Darlington Point solar -993.4 $/MWh. Large differences from NSW reference price support internal congestion. These adjustments are not separate settlement prices.
- Confidence: observed flows/limit setters confirmed; family meanings supported by AEMO reports/slides; full current equation metadata/coefficient decomposition not obtained. Avoid claiming precise MW attribution, full nodal prices, or relaxed-constraint counterfactual dispatch.

## Public explainer feasibility

- Available anonymously: five-minute prices, target/metered interconnector flows, limits/limit setters, constraint RHS/LHS/marginal values/violations/local adjustments (`DispatchIS_Reports`); binding marginal costs (`MCCDispatch`); constraint change spreadsheets.
- Published model: `GENCONDATA` descriptions/type/impact/source/notes; `SPDINTERCONNECTORCONSTRAINT` coefficients; generator connection-point coefficients; constraint sets/invocations. Use exact effective date/version from dispatch result, not an arbitrary latest definition.
- AEMO FAQ describes baseline `PUBLIC_GENCON_*` / `PUBLIC_GENCON_ACTIVE_*` plus incremental updates. Attempted `/Data_Interchange/` endpoint returned HTTP 403; usable anonymous current baseline location remains unverified. Monthly archive is delayed: inspected 2026 index only listed through August.
- AEMO's existing Plain English converter requires participant registration. Some single-term constraint results are delayed until the following morning; handle missing/confidential results explicitly.
- Minimum reliable explanation: identify counter-price link → binding equations containing link → coefficient direction + marginal effect → description/outage context → distinguish driver from cap and physical equality.
- Rank link-specific effects, not raw marginal-value magnitude across differently scaled equations. Use losses, intervention/pricing runs and timestamps correctly. Loop equations require joint interpretation.
- Stronger causal claims need dispatch replay/sensitivity analysis. Start with interval case studies and deterministic explanations; unknown metadata must remain unknown.

## Sources

- [08:30 raw dispatch ZIP](https://nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/PUBLIC_DISPATCHIS_202610060830_0000000541410261.zip)
- [08:20 raw dispatch ZIP](https://nemweb.com.au/Reports/CURRENT/DispatchIS_Reports/PUBLIC_DISPATCHIS_202610060820_0000000541409087.zip)
- [Dispatch downloads](https://nemweb.com.au/Reports/Current/DispatchIS_Reports/)
- [2026 constraint change spreadsheets ZIP, week 40](https://nemweb.com.au/Reports/CURRENT/Weekly_Constraint_Reports/2026.zip)
- [PEC constraints and loop flows, 6 August 2026, pp. 11–22](https://www.aemo.com.au/-/media/files/initiatives/project-energyconnect/meeting-documents/2026/pec-mi-constraints-and-loop-flows-6-august.pdf)
- [PEC market integration dates](https://www.aemo.com.au/initiatives/major-programs/nem-reform-program/nem-reform-program-initiatives/project-energyconnect-market-integration-project)
- [AEMO constraint FAQ](https://www.aemo.com.au/energy-systems/electricity/national-electricity-market-nem/system-operations/congestion-information-resource/constraint-faq)
- [February 2026 constraint report, family descriptions](https://www.aemo.com.au/-/media/files/electricity/nem/security_and_reliability/congestion-information/statistics/2026/february-2026.pdf)
- [April 2026 constraint report, Wagga/Murraylink descriptions](https://www.aemo.com.au/-/media/files/electricity/nem/security_and_reliability/congestion-information/statistics/2026/april-2026.pdf)
