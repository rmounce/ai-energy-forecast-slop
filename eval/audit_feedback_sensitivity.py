"""Paired saved core-plan sensitivity; projected paths are not realised savings."""
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
from energy_pipeline.solver_replay import digest, validate_result
from eval.calibrate_measured_load import BANDS


def forecast_diagnostics(bundle, actuals):
    """Score each issued vector once; measured futures never enter preparation."""
    truth = actuals.load_base_w.resample('30min').sum(min_count=6)/6
    seen,results = {},[]
    for event in bundle['dh_events']:
        if not event['ready'] or 'load_calibration' not in event: continue
        metadata = event['load_calibration']
        receipt = metadata['load_receipt']
        vector = event['calibrated_load_rows']
        if receipt in seen:
            if seen[receipt] != digest(vector): raise ValueError('correction changed within issued vintage')
            continue
        seen[receipt] = digest(vector)
        base = event['states']['sensor.ai_load_forecast_high']['attributes']['forecasts']
        if [row['timestamp'] for row in base] != [row['timestamp'] for row in vector]:
            raise ValueError('scored target arrays differ')
        dates = pd.to_datetime([row['timestamp'] for row in vector],utc=True)
        creation = pd.Timestamp(metadata['forecast_creation'])
        scored = pd.DataFrame({'baseline': [row['power_load'] for row in base],
            'calibrated': [row['power_load'] for row in vector], 'actual':truth.reindex(dates).to_numpy(),
            'lead':(dates-creation).total_seconds()/3600},index=dates)
        scored = scored.loc[scored.lead.between(0,72) & np.isfinite(scored.actual)]
        scored['band'] = pd.cut(scored.lead,[0,6,16.5,36,72],labels=BANDS,include_lowest=True)
        bands = []
        for band,group in scored.groupby('band',observed=True):
            row = {'horizon_band':str(band),'paired_complete_targets':len(group)}
            for name in ('baseline','calibrated'):
                error = group[name]-group.actual
                row[name] = {'mae_w':float(error.abs().mean()),
                    'pinball_w':float(np.where(error>=0,.35*error,-.65*error).mean())}
            bands.append(row)
        results.append({'receipt':receipt,'forecast_creation':creation.isoformat(),'bands':bands,
            'partial_interval_excluded':True,'scope':'posthoc_forecast_accuracy_not_dispatch_savings'})
    return results


def plan_diagnostics(report):
    challengers = set(report['summary'])-{'baseline'}
    if len(challengers) != 1: raise ValueError('one paired challenger required')
    challenger = challengers.pop()
    indexed = {arm:{} for arm in ('baseline',challenger)}
    for artifact in report['solves']:
        request = artifact['request']
        if request['request_id'] != digest({key:value for key,value in request.items() if key != 'request_id'}):
            raise ValueError('changed request identity')
        arm = request['counterfactual']['arm']
        key = (request['captured_at'],request['kind'])
        if arm not in indexed or key in indexed[arm]: raise ValueError('unknown or duplicate arm/origin')
        indexed[arm][key] = artifact
    if not indexed['baseline'] or set(indexed['baseline']) != set(indexed[challenger]):
        raise ValueError('unpaired saved origins')
    rows = []
    for key,left in indexed['baseline'].items():
        right = indexed[challenger][key]
        a,b = left['request'],right['request']
        if any(a[field] != b[field] for field in ('forecast_start','optimization_sha256')):
            raise ValueError('paired solver/target mismatch')
        f,g = validate_result(a,left['result']),validate_result(b,right['result'])
        if not f.index.equals(g.index): raise ValueError('paired result targets differ')
        differences = (g.P_batt-f.P_batt).abs()
        different = differences[differences > 1e-6]
        row = {'origin':key[0],'kind':key[1],
            'maximum_projected_soc_difference_pp':float((g.SOC_opt-f.SOC_opt).abs().max()*100),
            'maximum_projected_battery_difference_w':float(differences.max()),
            'first_different_projected_battery_target':different.index[0].isoformat() if len(different) else None,
            'terminal_target_difference_pp':float((b['payload']['soc_final']-a['payload']['soc_final'])*100),
            'first_battery_command_difference_w':float(g.P_batt.iloc[0]-f.P_batt.iloc[0])}
        if key[1] == 'mpc':
            row.update(baseline_first_inverter_at_output_limit=bool(np.isclose(float(f.P_hybrid_inverter.iloc[0]),
                a['configuration']['plant_conf']['inverter_ac_output_max'],rtol=0,atol=1e-6)),
                baseline_first_grid_net_zero=bool(abs(float(f.P_grid_pos.iloc[0]-f.P_grid_neg.iloc[0])) <= 1e-6))
        rows.append(row)
    return {'mode':'projected_plan_sensitivity_not_future_execution_or_savings',
        'challenger':challenger,'pairs':rows,'comparison':report['comparison'],
        'mpc_at_output_limit':sum(row.get('baseline_first_inverter_at_output_limit',False) for row in rows),
        'mpc_grid_net_zero':sum(row.get('baseline_first_grid_net_zero',False) for row in rows)}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('replay','output'): parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--dataset',type=Path,help='optional frozen measured targets for posthoc forecast scoring')
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    report_path = args.replay/'report.json'
    report = json.loads(report_path.read_text())
    bundle = json.loads((args.replay/'bundle.json').read_text())
    if digest(bundle) != report['bundle_sha256']: raise ValueError('changed saved bundle')
    diagnostics = plan_diagnostics(report)
    if args.dataset:
        manifest = json.loads((args.dataset/'manifest.json').read_text())
        actual_path = args.dataset/'actuals.parquet'
        actual_sha = hashlib.sha256(actual_path.read_bytes()).hexdigest()
        if (not manifest['export_complete'] or manifest['parquet_sha256'] != actual_sha
                or bundle['provenance']['load_calibration']['measured_actuals_sha256'] != actual_sha):
            raise ValueError('changed or incompatible measured scoring targets')
        diagnostics['forecast_scores'] = forecast_diagnostics(bundle,pd.read_parquet(actual_path))
        diagnostics['measured_actuals_sha256'] = actual_sha
    diagnostics.update(report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        bundle_sha256=digest(bundle),auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        publication_authorized=False)
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+'\n')
    print(json.dumps(diagnostics,allow_nan=False))


if __name__ == '__main__': main()
