"""Bounded DH/MPC event replay with own battery feedback and exogenous HWC."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from energy_pipeline.payloads import Inputs, build_dh_payload, dh_soc, target_soc_offset
from energy_pipeline.solver_chain import project_dh_entities
from energy_pipeline.solver_replay import digest, prepare_request, validate_result
from eval.audit_dh_source_history import source_states, source_coverage, SETTINGS
from eval.audit_amber_forecast_archive import parse_revision
from eval.minute_core_replay import build_bundle as minute_bundle, advance, policy_payload, execution_segments, between
from eval.feedback_checkpoint import contract, verified_checkpoint, validate_resume
from eval.audit_control_fidelity import asof
from eval.amber_quote_actuals import canonical_rates

OWN = ('sensor.dh_soc_batt_forecast','sensor.dh_p_load_forecast','sensor.dh_p_pv_forecast',
    'input_number.dh_last_soc_init','input_text.dh_last_reground_block',
    'input_number.emhass_target_soc_offset')
SOC = 'sensor.sigen_plant_battery_state_of_charge_derived'


def overlay(exogenous, parent, soc):
    states = deepcopy(exogenous)
    for entity in OWN:
        if entity not in parent: raise ValueError('missing owned parent: '+entity)
        states[entity] = deepcopy(parent[entity])
    states[SOC] = {'state':str(soc*100)}
    return states


def solve_payload(bundle, payload, kind, origin, arm, refs, parent_revision, solve):
    record = {'captured_at':origin,'publication_id':bundle['source_publication_id'],
        'mode':'historical_own_dh_feedback','payloads':{kind:payload},
        'readiness':{kind:{'coverage_ready':True,'reasons':[]}}}
    request = prepare_request(record,bundle['configuration'],kind=kind,
        optimization_sha256=bundle['optimization_sha256'])
    request['counterfactual'] = {'scope':'own_battery_dh_feedback_exogenous_hwc',
        'arm':arm,'input_receipts':refs,'dh_parent_revision':parent_revision,'publication_authorized':False}
    request.pop('request_id')
    request['request_id'] = digest(request)
    result = solve(request)
    if request['request_id'] != digest({key:value for key,value in request.items() if key != 'request_id'}):
        raise ValueError('solver mutated frozen request')
    return {'request':request,'result':result},validate_result(request,result)


def simulate(bundle, solve):
    if not 1 <= len(bundle['steps']) <= 15 or len(bundle['dh_events']) > 8:
        raise ValueError('require 1–15 MPC origins and <=8 DH origins')
    start,end = pd.Timestamp(bundle['steps'][0]['origin']),pd.Timestamp(bundle['steps'][-1]['end'])
    if 'resume_checkpoint' in bundle: start = validate_resume(bundle)
    ems = bundle.get('ems_execution')
    if ems:
        from eval import ems_feedback_policy as controller
        if bundle.get('experiment') != 'ems_timing': raise ValueError('EMS execution requires timing experiment')
    events = []
    for step in bundle['steps']:
        events.extend([(pd.Timestamp(step['origin']),2,'mpc',step),
            (pd.Timestamp(step['activation']),0,'mpc_activation',step)])
    for event in bundle['dh_events']:
        at = pd.Timestamp(event['origin'])
        if not start <= at < end: raise ValueError('DH origin outside execution timeline')
        events.append((at,2,'dh',event))
        if event['ready']:
            activation = pd.Timestamp(event['activation'])
            if not at < activation < end: raise ValueError('DH activation outside bounded timeline')
            events.append((activation,0,'dh_activation',event))
    if ems:
        events.extend((stamp,1,'fallback',{}) for stamp in pd.date_range(start.ceil('5min'),end,freq='5min') if stamp < end)
        for stream in ems['guards'].values():
            events.extend((pd.Timestamp(row['time']),1,'guard',{}) for row in stream
                          if start < pd.Timestamp(row['time']) < end)
    events.sort(key=lambda event:(event[0],event[1],event[2]))
    segments = bundle.get('prelude',[])+[row for step in bundle['steps'] for key in ('before_activation','after_activation') for row in step[key]]
    execution_plant = bundle.get('execution_plant',bundle['configuration']['plant_conf'])
    experiment = bundle.get('experiment','terminal_policy')
    if experiment not in ('terminal_policy','load_calibration','ems_timing'): raise ValueError('unknown feedback experiment')
    if experiment == 'ems_timing' and not ems: raise ValueError('missing EMS execution contract')
    challenger = {'load_calibration':'calibrated_load','ems_timing':'hold_accepted_command'}.get(experiment,'without_positive_lockin')
    arms = ('baseline',challenger)
    summaries, rows, artifacts, checkpoints = {},[],[],{}
    started = time.monotonic()
    for arm in arms:
        parent = deepcopy(bundle['initial_parent'])
        soc, command, cursor = bundle['initial_soc'],deepcopy(bundle['initial_command']),start
        totals,pending,revision = {},{},'initial_archived_parent:'+digest(parent)
        if 'resume_checkpoint' in bundle:
            seed = bundle['resume_checkpoint']['arms'][arm]
            parent,soc,command,revision = deepcopy(seed['parent']),seed['soc'],deepcopy(seed['command']),seed['revision']
        elif ems:
            plan = deepcopy(bundle['initial_ems_plan'])
            # Retain original publication clock while selecting the current target.
            command = {'ems_plan':plan,'ems_command':controller.policy(controller.selected(plan,start),soc,
                ems['minimum_export_soc']),'plant':deepcopy(execution_plant)}
        initial_soc = soc
        def execute_to(stamp):
            held = between(segments,cursor,stamp)
            if ems: return controller.advance(command.get('plant',execution_plant),soc,command,held,ems)
            return advance(command.get('plant',execution_plant),soc,command,held)
        for stamp,_,kind,event in events:
            if time.monotonic()-started > 150: raise TimeoutError('feedback batch budget exceeded')
            soc, flows = execute_to(stamp)
            for key,value in flows.items(): totals[key] = totals.get(key,0.)+value
            cursor = stamp
            row = {'origin':stamp.isoformat(),'kind':kind,'arm':arm,'soc':soc,'dh_parent_revision':revision}
            if kind in ('fallback','guard'):
                if kind == 'fallback' and arm == 'baseline':
                    command = controller.fallback(command,stamp,soc,ems)
                row['ems_command'] = deepcopy(command['ems_command'])
                rows.append(row)
                continue
            if kind.endswith('_activation'):
                accepted = pending.pop((kind.removesuffix('_activation'),event['origin']))
                if kind == 'mpc_activation':
                    command = controller.activate(accepted['ems_plan'],soc,ems,accepted['plant']) if ems else accepted
                    if ems: row['ems_command'] = deepcopy(command['ems_command'])
                else:
                    parent = accepted['parent']
                    # Model offset update as part of coherent acceptance, not a live helper write.
                    inputs = Inputs(overlay(event['states'],parent,soc),stamp.to_pydatetime())
                    parent['input_number.emhass_target_soc_offset'] = {'state':str(target_soc_offset(inputs))}
                    revision = accepted['revision']
                    row['accepted_parent_revision'] = revision
                rows.append(row)
                continue
            if kind == 'dh' and not event['ready']:
                rows.append(row | {'accepted':False,'reasons':event['reasons']})
                continue
            states = overlay(event['states'],parent,soc)
            if kind == 'dh':
                if arm == 'calibrated_load':
                    if 'calibrated_load_rows' not in event: raise ValueError('missing frozen load calibration')
                    states['sensor.ai_load_forecast_high'] = {'attributes':{'forecasts':deepcopy(event['calibrated_load_rows'])}}
                coverage = source_coverage(states,stamp)
                if not coverage['coverage_ready']: raise ValueError('admitted DH sources changed')
                inputs = Inputs(states,stamp.to_pydatetime())
                policy = dh_soc(inputs)
                payload = build_dh_payload(inputs)
            else:
                payload = policy_payload(states,stamp,soc,'baseline' if experiment in ('load_calibration','ems_timing') else arm)
            refs = event['input_receipts']
            if arm == 'calibrated_load' and kind == 'dh':
                refs = refs | {'load_calibration':event['load_calibration']}
            artifact,frame = solve_payload(bundle,payload,kind,event['origin'],arm,
                refs,revision,solve)
            artifacts.append(artifact)
            request = artifact['request']
            if payload['battery_nominal_energy_capacity'] != execution_plant['battery_nominal_energy_capacity']:
                raise ValueError('changing replay battery capacity is unsupported')
            row.update(request_id=request['request_id'],soc_init=payload['soc_init'],soc_final=payload['soc_final'])
            if kind == 'dh':
                projected = project_dh_entities(frame)
                if artifact['result'].get('projected_dh_entities') != projected:
                    raise ValueError('DH formatter evidence differs')
                updated = deepcopy(parent)
                updated.update(projected)
                updated['input_number.dh_last_soc_init'] = {'state':str(policy.soc_init_pct)}
                if policy.should_reground:
                    updated['input_text.dh_last_reground_block'] = {'state':policy.reground_block}
                pending[(kind,event['origin'])] = {'parent':updated,'revision':request['request_id']}
            else:
                pending[(kind,event['origin'])] = {'battery_w':float(frame.P_batt.iloc[0]),
                    'plant':deepcopy(request['configuration']['plant_conf']),
                    'curtail_w':float(frame.get('P_PV_curtailment',pd.Series(0.,index=frame.index)).iloc[0]),
                    'export_limit_w':request['configuration']['plant_conf']['maximum_power_to_grid']
                        if payload['prod_price_forecast'][0] > 0 else 0.}
                if ems:
                    pending[(kind,event['origin'])]['ems_plan'] = controller.project(frame,event['activation'])
                row['requested_battery_discharge_w'] = float(frame.P_batt.iloc[0])
                row['published_battery_discharge_w'] = event['published_battery_discharge_w']
            rows.append(row)
        soc, flows = execute_to(end)
        for key,value in flows.items(): totals[key] = totals.get(key,0.)+value
        if pending: raise ValueError('unactivated plans at end of replay')
        capacity = execution_plant['battery_nominal_energy_capacity']/1000
        summaries[arm] = totals | {'initial_soc':initial_soc,'final_soc':soc,'ending_inventory_kwh':soc*capacity,
            'final_parent_revision':revision,'final_offset_pct':float(parent['input_number.emhass_target_soc_offset']['state'])}
        checkpoints[arm] = {'parent':deepcopy(parent),'soc':soc,'command':deepcopy(command),'revision':revision}
    cash = summaries['baseline']['variable_cost_aud']-summaries[challenger]['variable_cost_aud']
    energy = summaries[challenger]['ending_inventory_kwh']-summaries['baseline']['ending_inventory_kwh']
    return {'scope':'own_battery_dh_feedback_exogenous_hwc','summary':summaries,'events':rows,'solves':artifacts,
        'comparison':{'cashflow_delta_aud':cash,'ending_inventory_delta_kwh':energy,
            'inventory_value_break_even_aud_per_kwh':-cash/energy if abs(energy)>1e-8 else None},
        'publication_authorized':False,'observed':bundle['observed'],
        'checkpoint':{'schema':1,'cursor':end.isoformat(),'contract':contract(bundle),'arms':checkpoints}}


def build_bundle(args):
    history = json.loads((args.history/'history.json').read_text())
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro',uri=True) as db:
        capture = json.loads(db.execute('SELECT record FROM handoffs ORDER BY rowid DESC LIMIT 1').fetchone()[0])
    captured = capture['input_snapshot']['states']
    apf = {leg:[parse_revision(row,leg) for row in history['mpc_apf_'+leg]] for leg in ('general','feed_in')}
    bundle = minute_bundle(args,apf_revisions=apf,captured_states=captured)
    if len(bundle['steps']) > 15: raise ValueError('pilot limited to15 MPC origins')
    first,last = pd.Timestamp(bundle['steps'][0]['origin']),pd.Timestamp(bundle['steps'][-1]['end'])
    resume = None
    if getattr(args,'resume_from',None):
        resume,parent_sha = verified_checkpoint(args.resume_from)
        cursor = pd.Timestamp(resume['cursor'])
        manifest = json.loads((args.history/'manifest.json').read_text())
        if not pd.Timestamp(manifest['start']) <= cursor <= first <= cursor+pd.Timedelta(seconds=120):
            raise ValueError('continuation cursor outside history or gap exceeds120s')
        previous = json.loads((args.replay/'bundle.json').read_text())
        rates = {key:canonical_rates(previous['quote_rows'][key],cursor.floor('5min'),first.ceil('5min'))[0]
            for key in ('general','feed')}
        prelude = execution_segments(history,cursor,first,rates) if cursor < first else []
        bundle['prelude'] = prelude
        for key,sign in (('grid_import_kwh',1),('grid_export_kwh',-1)):
            bundle['observed'][key] += sum(max(sign*r['observed_grid_import_w'],0)*r['duration_seconds']/3_600_000 for r in prelude)
        bundle['observed']['variable_cost_aud'] += sum((max(r['observed_grid_import_w'],0)*r['general_rate']
            -max(-r['observed_grid_import_w'],0)*r['feed_rate'])*r['duration_seconds']/3_600_000 for r in prelude)
        bundle['observed']['initial_soc'] = float(asof(history['soc'],cursor,120)['value'])/100
        bundle['provenance']['resume_parent_report_sha256'] = parent_sha
        bundle['provenance']['execution_start'] = cursor.isoformat()
    initial,initial_refs,initial_inherited = source_states(history,captured,first)
    bundle['initial_parent'] = {key:deepcopy(initial[key]) for key in OWN}
    bundle['initial_parent_evidence'] = {'input_receipts':initial_refs,'capture_confirmed_settings':initial_inherited}
    events = []
    for anchor in history['dh_anchor']:
        at = pd.Timestamp(anchor['time'])-pd.Timedelta(microseconds=1)
        if not (pd.Timestamp(resume['cursor']) if resume else first) <= at < last: continue
        states,refs,inherited = source_states(history,captured,at)
        coverage = source_coverage(states,at)
        event = {'origin':at.isoformat(),'ready':coverage['coverage_ready'],'reasons':coverage['reasons'],
            'states':{key:value for key,value in states.items() if key not in OWN+(SOC,)},
            'input_receipts':{key:value for key,value in refs.items()
                if key not in ('soc','dh_soc','dh_load','dh_pv','dh_anchor','target_offset','reground_block')},
            'capture_confirmed_settings':inherited}
        if event['ready']:
            pubs = [pd.Timestamp(row['time']) for row in history['dh_soc'] if at < pd.Timestamp(row['time']) < last]
            if not pubs or min(pubs)-at > pd.Timedelta(seconds=20):
                raise ValueError('missing bounded DH publication clock')
            event['activation'] = min(pubs).isoformat()
        events.append(event)
    for step in bundle['steps']:
        restored,refs,inherited = source_states(history,captured,step['origin'])
        # In particular, historical export allowance must replace the later capture.
        for key,entity in SETTINGS.items():
            if entity not in OWN:
                step['states'][entity] = deepcopy(restored[entity])
        step['input_receipts']['settings'] = {key:refs[key] for key,entity in SETTINGS.items() if entity not in OWN}
        step['capture_confirmed_settings'] = inherited
        for key in OWN: step['states'].pop(key,None)
        for key in ('dh_soc','dh_load','dh_pv','dh_anchor'):
            step['input_receipts']['telemetry_and_parent'].pop(key,None)
    initial_payload = policy_payload(overlay(bundle['steps'][0]['states'],bundle['initial_parent'],
        bundle['initial_soc']),first,bundle['initial_soc'],'baseline')
    # Base EMHASS config has placeholder capacity (420 kWh here); runtime
    # overrides are authoritative for both solve and pre-first-activation physics.
    bundle['execution_plant'] = deepcopy(bundle['configuration']['plant_conf'])
    for key in ('battery_nominal_energy_capacity','battery_minimum_state_of_charge'):
        bundle['execution_plant'][key] = initial_payload[key]
    bundle['dh_events'] = events
    bundle['provenance']['captured_handoff_sha256'] = digest(capture)
    if getattr(args,'experiment','terminal_policy') == 'load_calibration':
        from eval.prepare_load_feedback import prepare_calibrated_events
        calibration = json.loads((args.calibration/'report.json').read_text())
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        path = args.calibration/'challenger.parquet'
        measured = json.loads((args.dataset/'manifest.json').read_text())
        if (sha(path) != calibration['challenger_sha256'] or not measured['export_complete']
                or sha(args.dataset/'manifest.json') != calibration['dataset_manifest_sha256']
                or sha(args.dataset/'actuals.parquet') != measured['parquet_sha256']):
            raise ValueError('changed or incompatible calibration/measurement archive')
        bundle['experiment'] = 'load_calibration'
        bundle = prepare_calibrated_events(bundle,pd.read_parquet(path),calibration['settings'])
        bundle['provenance']['load_calibration'] = {'report_sha256':sha(args.calibration/'report.json'),
            'challenger_sha256':sha(path),'measured_actuals_sha256':measured['parquet_sha256'],
            'settings':calibration['settings']}
    if getattr(args,'experiment','terminal_policy') == 'ems_timing':
        from eval.ems_feedback_policy import configure, validate_guard_archive
        # Configure before validating resume: the controller/physics contract is required.
        if resume: bundle['resume_checkpoint'] = resume
        bundle['experiment'] = 'ems_timing'
        configure(bundle,args.ems_history,args.journal,args.dc_fixed_loss_w)
        if resume:
            validate_guard_archive(bundle,json.loads((args.resume_from/'bundle.json').read_text()))
    if resume:
        bundle['resume_checkpoint'] = resume
        validate_resume(bundle)
    return bundle


def main():
    os.nice(19)
    if sys.argv[1:2] == ['--worker']:
        from scripts.emhass_solver_worker import solve_request
        print(json.dumps(simulate(json.loads(Path(sys.argv[2]).read_text()),solve_request),allow_nan=False))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('history','replay','journal','output'): parser.add_argument('--'+flag,type=Path,required=True)
    for flag in ('start','end'): parser.add_argument('--'+flag,required=True)
    parser.add_argument('--experiment',choices=('terminal_policy','load_calibration','ems_timing'),default='terminal_policy')
    parser.add_argument('--ems-history',type=Path)
    parser.add_argument('--dc-fixed-loss-w',type=float,choices=(0.,140.),default=0.)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--resume-from',type=Path,help='verified preceding replay; carry own parents, inventory and command')
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    if args.experiment == 'load_calibration' and (args.calibration is None or args.dataset is None):
        parser.error('load_calibration requires --calibration and --dataset')
    if args.experiment != 'load_calibration' and (args.calibration is not None or args.dataset is not None):
        parser.error('calibration/dataset apply only to load_calibration')
    if args.experiment == 'ems_timing' and args.ems_history is None:
        parser.error('ems_timing requires --ems-history')
    if args.experiment != 'ems_timing' and (args.ems_history is not None or args.dc_fixed_loss_w != 0.):
        parser.error('EMS history/DC loss apply only to ems_timing')
    bundle = build_bundle(args)
    from scripts.replay_energy_solves import run_batch
    files = ['scripts/emhass_solver_worker.py','eval/dh_feedback_replay.py','eval/minute_core_replay.py',
        'eval/sequential_core_replay.py','eval/audit_control_fidelity.py','eval/audit_dh_source_history.py',
        'eval/archived_forecasts.py','eval/audit_amber_forecast_archive.py','eval/amber_quote_actuals.py',
        'eval/measured_actuals.py','energy_pipeline/payloads.py','energy_pipeline/solver_replay.py',
        'energy_pipeline/solver_chain.py','eval/prepare_load_feedback.py',
        'eval/compare_load_solver_sensitivity.py','eval/calibrate_measured_load.py','eval/feedback_checkpoint.py',
        'eval/ems_feedback_policy.py','eval/replay_ems_fallback.py','eval/audit_ems_delivery.py',
        'eval/summarize_feedback_chain.py']
    result,hashes = run_batch(bundle,'eval/dh_feedback_replay.py',files)
    for artifact in result['solves']: validate_result(artifact['request'],artifact['result'])
    result.update(bundle_sha256=digest(bundle),code_sha256=hashes,provenance=bundle['provenance'],limitations=[
        'initial live DH parent/anchor/reground/offset seeded once; subsequent battery feedback endogenous',
        'HWC schedule and thermal state remain exogenous; no counterfactual HWC planner',
        'aligned-source admission and coherent parent/helper activation are modeled improvements, not live fidelity',
        'historical helper/publication timing proxies; no proof of actual capture or actuation clocks',
        'offset update modeled at coherent DH acceptance; live offset helper timing not replayed',
        'delivered measured PV lower bound, ideal physical executor, device mode/ramp behavior not simulated',
        'archived settings/captured static fallback and current frozen plant config; not independent historical availability',
        'retrospective stress selection; variable cashflow excludes wear, fixed charges and terminal inventory value'])
    if bundle.get('experiment') == 'load_calibration':
        result['experiment'] = 'load_calibration'
        result['limitations'].extend(['calibration fits at logged forecast creation with assumed30m measurement release lag',
            'p65 version inferred from exact overlapping vector; archived HA lacks model-version identity',
            'common initial archived parent; calibration begins at first admitted DH refresh; tail labels not used'])
    if bundle.get('experiment') == 'ems_timing':
        result['experiment'] = 'ems_timing'
        result['limitations'].remove('delivered measured PV lower bound, ideal physical executor, device mode/ramp behavior not simulated')
        result['limitations'].extend([
            'own MPC trajectories/controller modes/limits with endogenous DH/MPC battery feedback; forecasting/terminal policy common',
            'supported on-grid noncharging noncurtailing battery export/self-consumption branches; others fail',
            'instant modeled activation at historical publication clock; no script completion/ramp/transient PCS model',
            'delivered PV lower bound and nominal-capacity energy ledger; independently validated stock unavailable',
            'static export SOC capture fallback weaker than historical receipts; guard at/below physical floor',
            'explicit fixed DC-loss ablation; no parameter fit'])
    args.output.mkdir(parents=True)
    for name,content in [('bundle',bundle),('report',result)]:
        (args.output/(name+'.json')).write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'scope':result['scope'],'summary':result['summary'],'comparison':result['comparison'],
        'dh_origins':len(bundle['dh_events']),'dh_rejected':sum(not e['ready'] for e in bundle['dh_events']),
        'solve_count':len(result['solves'])}))


if __name__ == '__main__': main()
