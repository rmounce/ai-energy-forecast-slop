"""PV-string recovery and full-helper evidence; never fill missing actuals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from eval.audit_control_fidelity import numeric_series
from eval.measured_actuals import integrate


def helper_at(rows, at, zone):
    available = [row for row in rows if pd.Timestamp(row['time']) <= at]
    if not available: return {'status':'missing_historical_helper'}
    row = max(available,key=lambda row:pd.Timestamp(row['time']))
    value = pd.Timestamp(row['state'])
    if pd.isna(value): raise ValueError('invalid full-helper timestamp')
    if value.tzinfo is None: value = value.tz_localize(zone,ambiguous='raise',nonexistent='raise')
    value = value.tz_convert('UTC')
    if value > pd.Timestamp(row['time']): raise ValueError('full-helper value later than receipt')
    return {'status':'observed_prior_helper_not_counterfactual', 'receipt':row['time'],
        'state':row['state'],'last_full_utc':value.isoformat(),
        'holdoff_active':bool(pd.Timedelta(0) <= at-value < pd.Timedelta(hours=8))}


def audit(history,start,end,zone):
    start,end = pd.Timestamp(start),pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None or not start < end <= start+pd.Timedelta(hours=24):
        raise ValueError('aware positive <=24h window required')
    keys = ('pv','pv1','pv2','battery','inverter_ac','loss')
    summaries = {key:integrate(numeric_series(history[key]),start,end) for key in keys}
    frame = pd.DataFrame({key:summary['mean'] for key,summary in summaries.items()})
    if (frame[['pv','pv1','pv2']] < 0).any().any(): raise ValueError('negative measured PV')
    strings = frame.pv1+frame.pv2
    recovered = frame.pv.isna() & strings.notna()
    unresolved = frame.pv.isna() & strings.isna()
    runs = []
    for at in frame.index[unresolved]:
        if runs and pd.Timestamp(runs[-1]['end'])==at:
            runs[-1]['end'] = (at+pd.Timedelta(minutes=5)).isoformat()
            runs[-1]['intervals'] += 1
        else: runs.append({'start':at.isoformat(),'end':(at+pd.Timedelta(minutes=5)).isoformat(),'intervals':1})
    # Loss is derived from these same powers. Its algebraic PV residual cannot
    # become an independent target or make stale raw strings fresh.
    residual = (frame.inverter_ac+frame.battery+frame.loss)[unresolved].dropna()
    overlap = (frame.pv-strings).dropna()
    return {'mode':'cycle_support_diagnostic_not_replay_admission','start':start.isoformat(),'end':end.isoformat(),
        'publication_authorized':False,'replay_admission_authorized':False,
        'intervals':len(frame),'complete_intervals':{key:int(summary['mean'].notna().sum()) for key,summary in summaries.items()},
        'string_recoverable_intervals':int(recovered.sum()),'unresolved_intervals':int(unresolved.sum()),
        'unresolved_runs':runs,'maximum_gross_minus_strings_abs_w':float(overlap.abs().max()) if len(overlap) else None,
        'unresolved_derived_balance':{'complete_intervals':len(residual),
            'maximum_abs_w':float(residual.abs().max()) if len(residual) else None,
            'mean_w':float(residual.mean()) if len(residual) else None,
            'independent_validation':False},
        'full_helper':{key:helper_at(history.get('last_full',[]),at,zone) for key,at in (('start',start),('end',end))},
        'helper_local_time_zone':zone,
        'limitations':['raw powers held <=120s, gaps not filled or age-relaxed',
            'overlap confirms string-sum identity, not independent solar validation',
            'derived loss balance is algebraic, not independent evidence of zero PV',
            'full-helper receipt is historical; each counterfactual needs its own threshold transitions',
            'repo helper trigger above99.99% differs from controller full threshold99.5%; historical code stability unproven',
            'available delivered PV does not establish counterfactual solar under curtailment',
            'this eight-source archive contains no forecast/controller capture-clock completeness evidence']}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('history','output'): parser.add_argument('--'+flag,type=Path,required=True)
    for flag in ('start','end','local-time-zone'): parser.add_argument('--'+flag,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output file required')
    sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = args.history/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    path = args.history/'history.json'
    if not manifest['export_complete'] or manifest['history_sha256']!=sha(path):
        raise ValueError('changed/incomplete cycle archive')
    start,end = pd.Timestamp(args.start),pd.Timestamp(args.end)
    if not pd.Timestamp(manifest['start']) <= start < end <= pd.Timestamp(manifest['end']):
        raise ValueError('window outside cycle archive')
    result = audit(json.loads(path.read_text()),start,end,args.local_time_zone)
    result['provenance'] = {'history_sha256':sha(path),'manifest_sha256':sha(manifest_path),'code_sha256':sha(Path(__file__)),
        'dependency_sha256':{name:sha(ROOT/name) for name in ('eval/audit_control_fidelity.py','eval/measured_actuals.py')}}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__': main()
