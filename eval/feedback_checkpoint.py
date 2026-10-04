"""Verified offline feedback checkpoints; no historical-state re-grounding."""
from copy import deepcopy
import hashlib
import json

import pandas as pd

from energy_pipeline.solver_replay import digest


def contract(bundle):
    return {'image':bundle.get('image'),'configuration_sha256':digest(bundle['configuration']),
        'optimization_sha256':bundle['optimization_sha256'],
        'experiment':bundle.get('experiment','terminal_policy'),
        'source_publication_id':bundle['source_publication_id'],
        'calibration':bundle.get('provenance',{}).get('load_calibration'),
        'execution_capacity_wh':bundle.get('execution_plant',bundle['configuration']['plant_conf'])['battery_nominal_energy_capacity']}


def verified_checkpoint(folder):
    """Reproduce saved requests/economics before trusting a historical continuation."""
    from eval.dh_feedback_replay import simulate
    report_path = folder/'report.json'
    report = json.loads(report_path.read_text())
    bundle = json.loads((folder/'bundle.json').read_text())
    if digest(bundle) != report['bundle_sha256']: raise ValueError('changed preceding bundle')
    saved = {a['request']['request_id']:a for a in report['solves']}
    if len(saved) != len(report['solves']): raise ValueError('duplicate preceding requests')
    def solve(request):
        artifact = saved.get(request['request_id'])
        if artifact is None or artifact['request'] != request: raise ValueError('preceding saved request differs')
        return deepcopy(artifact['result'])
    reproduced = simulate(bundle,solve)
    for key in ('summary','comparison','events','solves'):
        if reproduced[key] != report[key]: raise ValueError('preceding replay no longer reproduces: '+key)
    checkpoint = reproduced['checkpoint']
    if 'checkpoint' in report and report['checkpoint'] != checkpoint:
        raise ValueError('saved checkpoint differs from reproduced state')
    return checkpoint,hashlib.sha256(report_path.read_bytes()).hexdigest()


def validate_resume(bundle):
    checkpoint = bundle['resume_checkpoint']
    if checkpoint['schema'] != 1 or checkpoint['contract'] != contract(bundle):
        raise ValueError('resume experiment/configuration contract differs')
    cursor,first = pd.Timestamp(checkpoint['cursor']),pd.Timestamp(bundle['steps'][0]['origin'])
    if cursor.tzinfo is None or not cursor <= first <= cursor+pd.Timedelta(seconds=120):
        raise ValueError('resume cursor is future or gap exceeds120s')
    return cursor
