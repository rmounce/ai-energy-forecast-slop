# Grid interruption — 2026-10-08

- Evidence: read-only HA history, 13:45–14:05 Australia/Adelaide; focused MPC/AC history
  13:49–13:54. Times below local (UTC+10:30). Recorder timestamps describe observed
  telemetry, not exact electrical transfer times.
- Grid status: `Off Grid (Auto)` at 13:51:15.869; `On Grid` at 13:51:52.855.
  Observed off-grid duration 36.986 s.
- Battery action: backup self-consumption selected 13:51:27.576, 11.707 s after status
  transition. MPC battery publication 13:51:27.474 immediately preceded the selection;
  consistent with the MPC publication → battery-policy trigger path.
- EMS mode already `Maximum Self Consumption`; no recorded mode change.
- Discharge-limit readback: 0 → 24 kW at 13:51:32.871, 17.002 s after off-grid status.
- PV limit already 100 kW throughout; event does not test releasing a curtailed PV limit.
- Normal action restored 13:52:27.034; discharge limit returned to 0 at 13:52:31.898.
- All four dump switches off throughout sampled window. Retail price positive
  (0.0522/0.0535); event does not test outage shedding or negative-price admission.
- Off-grid samples: household power 1.235–1.471 kW; gross PV approximately 6.44–6.46 kW;
  battery power positive approximately 4.73–4.92 kW (charging); grid power near zero.
  No sampled supply collapse. Five-second telemetry cannot exclude a brief transfer transient.
- Reconnection: grid import peaked at 1.095 kW in focused samples, then approached zero.
- MPC continued publishing: 13:51:27 and 13:52:27; scheduled battery charging approximately
  4.880 and 4.996 kW respectively; current-slot grid plan remained zero.
  No evidence of an island-mode optimisation switch.
- actrl: compressor remained reported on; climate remained cool. Surplus integral continued
  normal decay, approximately 0.1 per 10 s through event (0.6/min), then 0.05 per 10 s
  from 13:54:05; reached zero 13:59:55. Integral exceeded 1; do not assume a 1-degree
  integral ceiling or equate the shared integral with applied room offsets.
- Archived curtailment forecast at window start: first 12 slots average zero; current
  curtailment state zero throughout. History does not establish every forecast attribute refresh.
- Limits: this low-load, PV-surplus event confirms backup policy selection and recovery;
  does not establish battery-discharge delivery under deficit, MCB/street fault origin,
  high-load backup capacity, or dump-load shedding.
