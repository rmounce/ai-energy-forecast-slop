"""Check modeled MPC publication against pinned installed pure formatter; no solves/writes."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHANNELS = {'battery': ('sensor.mpc_p_batt_forecast', 'battery_scheduled_power', 'battery'),
            'grid': ('sensor.mpc_p_grid_forecast', 'forecasts', 'power'),
            'load': ('sensor.mpc_p_load_forecast', 'forecasts', 'power'),
            'pv': ('sensor.mpc_p_pv_forecast', 'forecasts', 'power'),
            'hybrid': ('sensor.mpc_p_hybrid_inverter', 'forecasts', 'power'),
            'curtailment': ('sensor.mpc_p_pv_curtailment', 'forecasts', 'power')}


def installed(bundle):
    # Import only in disposable pinned container. Never instantiate an HA client.
    from emhass.retrieve_hass import RetrieveHass
    source = hashlib.sha256(Path(inspect.getfile(RetrieveHass)).read_bytes()).hexdigest()
    if source != bundle['formatter_sha256']:
        raise ValueError('installed formatter source differs from pin')
    cases = []
    for case in bundle['cases']:
        projected = {}
        for name, (entity, attribute, device) in CHANNELS.items():
            series = pd.Series(case['columns'][name], index=pd.to_datetime(case['targets'], utc=True))
            data = RetrieveHass.get_attr_data_dict(series, 0, entity, device, 'W', entity,
                                                  attribute, np.round(series.iloc[0], 2))
            projected[name] = {'state':data['state'], 'rows':data['attributes'][attribute]}
        cases.append({'id':case['id'], 'projected':projected})
    return {'formatter_sha256':source, 'cases':cases, 'publication_authorized':False}


def compare(case, result, expected):
    if case['id'] != result['id']: raise ValueError('formatter case identity differs')
    points = []
    for i, target in enumerate(case['targets']):
        point = {'target':target}
        for name, (entity, _, _) in CHANNELS.items():
            channel = result['projected'][name]
            if len(channel['rows']) != len(case['targets']): raise ValueError('formatter target coverage differs')
            row = channel['rows'][i]
            if pd.Timestamp(row['date']) != pd.Timestamp(target): raise ValueError('formatter target label differs')
            point[name] = float(row[entity.removeprefix('sensor.')])
            if i == 0 and float(channel['state']) != point[name]: raise ValueError('formatter current state differs')
        points.append(point)
    actual = {'accepted_at':case['accepted_at'], 'points':points}
    if actual != expected: raise ValueError('MPC publication projection differs: '+case['id'])
    return len(points)*len(CHANNELS)


def case_from_frame(identity, frame, accepted_at):
    return {'id':identity, 'accepted_at':accepted_at,
            'targets':[at.isoformat() for at in frame.index], 'columns': {
                'battery':frame.P_batt.tolist(), 'grid':(frame.P_grid_pos+frame.P_grid_neg).tolist(),
                'load':frame.P_Load.tolist(), 'pv':frame.P_PV.tolist(),
                'hybrid':frame.P_hybrid_inverter.tolist(),
                'curtailment':frame.get('P_PV_curtailment',pd.Series(0.,index=frame.index)).tolist()}}


def main():
    os.nice(19)
    if sys.argv[1:2] == ['--worker']:
        print(json.dumps(installed(json.loads(Path(sys.argv[2]).read_text())),allow_nan=False))
        return
    from energy_pipeline.solver_replay import digest, validate_result
    from eval.ems_feedback_policy import project
    from eval.summarize_feedback_chain import summarize
    from scripts.replay_energy_solves import run_batch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,nargs='+',required=True)
    parser.add_argument('--formatter-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output required')
    chain = summarize(args.replay)
    bundles = [json.loads((folder/'bundle.json').read_text()) for folder in args.replay]
    images = {bundle['image'] for bundle in bundles}
    if len(images) != 1: raise ValueError('mixed installed images')
    cases, expected = [], {}
    for folder in args.replay:
        report = json.loads((folder/'report.json').read_text())
        for artifact in report['solves']:
            request = artifact['request']
            if request['kind'] != 'mpc': continue
            frame = validate_result(request,artifact['result'])
            accepted = next(step['activation'] for bundle in bundles for step in bundle['steps']
                            if pd.Timestamp(step['origin']) == pd.Timestamp(request['captured_at']))
            identity = request['request_id']
            cases.append(case_from_frame(identity,frame,accepted))
            expected[identity] = project(frame,accepted)
    # Rounding/sign boundaries, zeros and an Adelaide DST transition; no physics claims.
    index = pd.date_range('2026-10-03T16:25Z',periods=8,freq='5min')
    values = np.array([-0.0,-0.004,-0.005,-1.125,0.,0.005,1.125,9979.995])
    frame = pd.DataFrame({'P_batt':values,'P_grid_pos':np.maximum(values,0),
        'P_grid_neg':np.minimum(values,0),'P_Load':np.abs(values),'P_PV':np.abs(values),
        'P_hybrid_inverter':values,'P_PV_curtailment':np.abs(values)},index=index)
    cases.append(case_from_frame('synthetic_rounding_dst',frame,index[0].isoformat()))
    expected[cases[-1]['id']] = project(frame,cases[-1]['accepted_at'])
    bundle = {'image':images.pop(),'formatter_sha256':args.formatter_sha256,'cases':cases}
    result, hashes = run_batch(bundle,'eval/audit_mpc_formatter.py',['eval/audit_mpc_formatter.py'])
    if len(result['cases']) != len(cases): raise ValueError('formatter case count differs')
    checked = sum(compare(case,output,expected[case['id']]) for case,output in zip(cases,result['cases']))
    output = {'scope':'installed_pure_mpc_formatter_parity','image':bundle['image'],
        'formatter_sha256':result['formatter_sha256'],'historical_mpc_cases':len(cases)-1,
        'synthetic_cases':1,'checked_power_points':checked,'bundle_sha256':digest(bundle),
        'replay_lineage':chain['lineage'],'code_sha256':hashes,
        'projection_sha256':hashlib.sha256((ROOT/'eval/ems_feedback_policy.py').read_bytes()).hexdigest(),
        'publication_authorized':False,'result':result,
        'limitations':['pure formatting only; no HA publication atomicity, script completion or device latency evidence',
                       'historical corpus plus numerical fixtures; not proof for every future result/configuration']}
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:output[key] for key in ('scope','historical_mpc_cases','checked_power_points','formatter_sha256')}))


if __name__ == '__main__': main()
