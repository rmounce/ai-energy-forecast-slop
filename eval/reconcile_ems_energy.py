"""Frozen power ledger and BMS capacity-ratio diagnostics; no fitted battery model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.audit_control_fidelity import asof, numeric_series
from eval.measured_actuals import clean_samples


def held_energy(rows, start, end):
    """Signed/positive/negative kWh over exact aware boundaries, max120s raw holds."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or not start < end:
        raise ValueError('aware positive energy window required')
    samples = clean_samples(numeric_series(rows))
    times = samples.index.as_unit('ns').asi8
    values = samples.to_numpy(dtype=float)
    left = np.maximum(times, start.value)
    right = np.minimum(np.minimum(np.r_[times[1:], end.value], times+120*10**9), end.value)
    seconds = np.maximum(0, right-left)/1e9
    valid = np.isfinite(values)
    if abs(seconds[valid].sum()-(end-start).total_seconds()) > 1e-6:
        raise ValueError('incomplete raw power support')
    dt, power = seconds[valid]/3_600_000, values[valid]
    return {'signed_kwh': float(np.sum(power*dt)), 'positive_kwh': float(np.sum(np.maximum(power,0)*dt)),
            'negative_kwh': float(np.sum(np.maximum(-power,0)*dt))}


def endpoint(history, at):
    fields = ('soc','available_charge','available_discharge','derived_capacity')
    rows = {key: asof(history[key],at,120) for key in fields}
    values = {key:float(row['value']) for key,row in rows.items()}
    if not np.isfinite(list(values.values())).all() or min(values.values()) < 0:
        raise ValueError('invalid available capacity/SoC endpoint')
    total = values['available_charge']+values['available_discharge']
    if total <= 0 or not 0 <= values['soc'] <= 100:
        raise ValueError('invalid capacity ratio')
    return values | {'available_total_kwh':total, 'available_ratio_soc':values['available_discharge']/total,
                     'receipt_times':{key:row['time'] for key,row in rows.items()}}


def reconcile(history, start, end, *, nominal_kwh, charge_efficiency=.99, discharge_efficiency=.99):
    if (not np.isfinite([nominal_kwh,charge_efficiency,discharge_efficiency]).all()
            or nominal_kwh <= 0 or not 0 < min(charge_efficiency,discharge_efficiency)
            or max(charge_efficiency,discharge_efficiency)>1):
        raise ValueError('invalid ledger capacity/efficiency')
    powers = {key:held_energy(history[key],start,end) for key in ('pv1','pv2','pv','battery','inverter_ac','grid','load','loss')}
    a,z = endpoint(history,start),endpoint(history,end)
    dc = powers['pv1']['signed_kwh']+powers['pv2']['signed_kwh']-powers['battery']['signed_kwh']
    ac = powers['inverter_ac']['signed_kwh']
    energy_change = powers['battery']['positive_kwh']*charge_efficiency-powers['battery']['negative_kwh']/discharge_efficiency
    nominal_soc_change = nominal_kwh*(z['soc']-a['soc'])/100
    # D=C*s. Ratio identity is algebra, not proof of physical energy inventory.
    ratio_component = a['available_total_kwh']*(z['available_ratio_soc']-a['available_ratio_soc'])
    capacity_component = z['available_ratio_soc']*(z['available_total_kwh']-a['available_total_kwh'])
    return {'start':pd.Timestamp(start).isoformat(),'end':pd.Timestamp(end).isoformat(),
            'duration_seconds':(pd.Timestamp(end)-pd.Timestamp(start)).total_seconds(), 'initial':a,'final':z,
            'power_ledger':powers,
            'balance':{'raw_dc_minus_ac_kwh':dc-ac,'derived_clipped_loss_kwh':powers['loss']['signed_kwh'],
                'pv_sum_minus_gross_kwh':powers['pv1']['signed_kwh']+powers['pv2']['signed_kwh']-powers['pv']['signed_kwh'],
                'grid_plus_inverter_minus_site_kwh':powers['grid']['signed_kwh']+ac-powers['load']['signed_kwh'],
                'aggregate_ac_dc_ratio':ac/dc if dc>1e-8 else None,
                'eta95_ac_residual_kwh':.95*dc-ac,
                'eta95_after140dc_ac_residual_kwh':.95*(dc-140*(pd.Timestamp(end)-pd.Timestamp(start)).total_seconds()/3_600_000)-ac},
            'inventory':{'nominal_capacity_kwh':nominal_kwh,'dc_power_energy_change_kwh':energy_change,
                'nominal_soc_energy_change_kwh':nominal_soc_change,
                'power_ledger_minus_nominal_soc_change_kwh':energy_change-nominal_soc_change,
                'available_discharge_change_kwh':z['available_discharge']-a['available_discharge'],
                'available_ratio_change_component_kwh':ratio_component,
                'available_capacity_change_component_kwh':capacity_component,
                'available_discharge_minus_dc_power_change_kwh':z['available_discharge']-a['available_discharge']-energy_change}}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('history','output'): parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--start',required=True)
    parser.add_argument('--end',required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output file required')
    manifest = json.loads((args.history/'manifest.json').read_text())
    path = args.history/'history.json'
    if not manifest['export_complete'] or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['history_sha256']:
        raise ValueError('changed/incomplete energy archive')
    start,end = pd.Timestamp(args.start),pd.Timestamp(args.end)
    if not pd.Timestamp(manifest['start']) <= start < end <= pd.Timestamp(manifest['end']):
        raise ValueError('window outside frozen archive')
    h = json.loads(path.read_text())
    rated = asof(h['rated_capacity'],start,31*86400)
    health = asof(h['battery_health'],start,31*86400)
    nominal = float(rated['value'])*float(health['value'])/100
    result = reconcile(h,start,end,nominal_kwh=nominal)
    result.update(scope='power_and_reported_capacity_diagnostic_not_true_inventory',publication_authorized=False,
                  provenance={'history_sha256':manifest['history_sha256'],'rated_capacity':rated,'health':health,
                              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'dependency_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                              for name in ('eval/audit_control_fidelity.py', 'eval/measured_actuals.py')}},
                  limitations=['independent sensor receipt clocks; endpoints and raw holds not atomic',
                      'available-discharge and total capacity are BMS estimates, not independently measured physical inventory',
                      'derived loss is a clipped residual of the same raw powers, not independent efficiency truth',
                      'nominal ledger uses fixed99% charge/discharge efficiencies for diagnosis; no parameter fitting',
                      'current installed SoC formula inspected separately; no proof of unchanged historical configuration'])
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'balance':result['balance'],'inventory':result['inventory']}))


if __name__ == '__main__': main()
