"""Screen observed battery excursions; coverage diagnostics, never replay admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def screen(frame):
    index = frame.index
    if (not isinstance(index, pd.DatetimeIndex) or index.tz is None or index.has_duplicates
            or not index.is_monotonic_increasing or not index.equals(index.floor('5min'))):
        raise ValueError('ordered unique aware five-minute boundaries required')
    columns = ('soc_pct_end', 'pv_dc_w', 'load_site_w', 'grid_import_w',
               'battery_charge_w_charge_kwh', 'battery_charge_w_discharge_kwh')
    if not set(columns) <= set(frame):
        raise ValueError('missing measured columns')
    values = frame[list(columns)].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('infinite measured value')
    soc = frame.soc_pct_end
    if ((soc.dropna() < 0) | (soc.dropna() > 100)).any():
        raise ValueError('invalid SoC')
    if (frame[['pv_dc_w','load_site_w']] < 0).any().any():
        raise ValueError('negative PV/load')
    energy = frame[list(columns[-2:])]
    if (energy < 0).any().any():
        raise ValueError('negative unsigned energy')

    def describe(start, end):
        # Input index labels interval starts; SoC samples describe interval ends.
        sample = frame.loc[(index >= start) & (index < end)]
        expected = int((end-start)/pd.Timedelta(minutes=5))
        finite_soc = sample.soc_pct_end.dropna()
        return {'start':start.isoformat(), 'end':end.isoformat(),
            'hours':(end-start).total_seconds()/3600,
            'minimum_observed_soc_pct':float(finite_soc.min()),
            'maximum_observed_soc_pct':float(finite_soc.max()),
            'expected_intervals':expected, 'present_intervals':len(sample),
            'finite_intervals':{key:int(sample[key].notna().sum()) for key in columns},
            'complete_measured_power':len(sample)==expected and bool(sample[list(columns[1:])].notna().all().all()),
            'observed_charge_kwh':float(sample[columns[-2]].sum()) if len(sample)==expected and sample[columns[-2]].notna().all() else None,
            'observed_discharge_kwh':float(sample[columns[-1]].sum()) if len(sample)==expected and sample[columns[-1]].notna().all() else None}

    # Collapse repeated full samples until a material excursion below95%.
    # Unknown SoC stays unknown; any cycle crossing gaps reports them explicitly.
    cycles, anchor, departed = [], None, False
    for at, value in soc.items():
        if pd.isna(value):
            continue
        if value >= 99.5:
            if anchor is not None and departed:
                start, end = anchor+pd.Timedelta(minutes=5), at+pd.Timedelta(minutes=5)
                cycles.append(describe(start,end) | {'initial_observed_soc_pct':float(soc.loc[anchor]),
                    'final_observed_soc_pct':float(value), 'kind':'observed_full_to_full_excursion'})
            anchor, departed = at, False
        elif anchor is not None and value <= 95:
            departed = True

    scarce = []
    begin, previous = None, None
    for at, value in soc.items():
        eligible = pd.notna(value) and value <= 15
        adjacent = previous is not None and at == previous+pd.Timedelta(minutes=5)
        if begin is not None and (not eligible or not adjacent):
            scarce.append(describe(begin,previous+pd.Timedelta(minutes=5)))
            begin = None
        if eligible and begin is None:
            begin = at
        previous = at
    if begin is not None:
        scarce.append(describe(begin,previous+pd.Timedelta(minutes=5)))
    return {'mode':'retrospective_inventory_screen_not_economic_validation',
        'publication_authorized':False, 'thresholds':{'full_soc_pct':99.5,'departure_soc_pct':95,'scarce_soc_pct':15},
        'cycles':sorted(cycles,key=lambda row:(row['minimum_observed_soc_pct'],row['hours'],row['start'])),
        'scarce_episodes':sorted(scarce,key=lambda row:(row['minimum_observed_soc_pct'],-row['hours'],row['start'])),
        'minimum_observed_soc_pct':float(soc.min()) if soc.notna().any() else None,
        'observed_intervals_at_or_below_10_pct':int(soc.le(10).sum()),
        'limitations':['observed SoC endpoints are changing BMS ratios, not independent energy inventory',
            'equal full SoC does not impose equal counterfactual ending inventory',
            'missing samples never filled; incomplete unsigned energy totals remain null',
            '15% is a screening threshold; reference physical floor10% is not necessarily binding',
            'retrospective stress selection cannot estimate average achievable savings',
            'measured coverage does not establish causal forecast/controller availability']}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output file required')
    path = args.dataset/'actuals.parquet'
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if not manifest['export_complete'] or sha(path)!=manifest['parquet_sha256']:
        raise ValueError('changed or incomplete measured archive')
    report = screen(pd.read_parquet(path))
    report['provenance'] = {'actuals_sha256':sha(path),'manifest_sha256':sha(args.dataset/'manifest.json'),
                            'code_sha256':sha(Path(__file__))}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False))


if __name__ == '__main__': main()
