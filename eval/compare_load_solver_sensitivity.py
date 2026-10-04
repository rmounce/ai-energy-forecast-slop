"""Bounded, isolated DH load-calibration sensitivity; never realised savings."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from energy_pipeline.payloads import Inputs, build_dh_payload
from energy_pipeline.solver_replay import digest, prepare_request, validate_result, forecast_summary
from scripts.replay_energy_solves import run_request
from eval.calibrate_measured_load import BANDS


def calibrated_handoff(record, rows, settings):
    """Verify incumbent p65 lineage, then change its base load, retaining HWC/losses.

    Model version is inferred from exact vector agreement with a single logged
    vintage; the HA entity does not archive the version itself. Historical origin
    uses that vintage's creation, not the later capture time, for calibration.
    """
    if (settings['history_days'] <= 0 or settings['min_targets'] < 1 or
            settings['availability_lag_minutes'] < 0):
        raise ValueError('invalid calibration settings')
    original = deepcopy(record)
    states = original['input_snapshot']['states']
    base = states['sensor.ai_load_forecast_high']['attributes']['forecasts']
    at = pd.Timestamp(original['captured_at'])
    original_payload = build_dh_payload(Inputs(states, at.to_pydatetime()))
    if original_payload != original['payloads']['dh']:
        raise ValueError('original frozen DH builder parity failed')
    required = ['forecast_target_time', 'forecast_creation_time', 'model_name',
                'model_version', 'prediction_type', 'prediction', 'actual_w', 'horizon_hours']
    if not set(required) <= set(rows):
        raise ValueError('missing frozen calibration columns')
    candidates = rows.loc[(rows.model_name == 'load_p65') & (rows.forecast_creation_time <= at)].copy()
    if candidates.empty:
        raise ValueError('no causal incumbent p65 vintage')
    creation = candidates.forecast_creation_time.max()
    latest = candidates.loc[candidates.forecast_creation_time == creation]
    if latest.model_version.nunique() != 1 or latest.prediction_type.nunique() != 1:
        raise ValueError('ambiguous incumbent model/version/type')
    if latest.forecast_target_time.duplicated().any():
        raise ValueError('duplicate incumbent target')
    lookup = latest.set_index('forecast_target_time').prediction
    matched = []
    for row in base:
        target = pd.Timestamp(row['timestamp'])
        if target in lookup:
            if not np.isclose(float(row['power_load']), lookup.loc[target], rtol=0, atol=1e-7):
                raise ValueError('frozen HA base does not match latest causal p65 vector')
            matched.append(target)
    if len(matched) < 12:
        raise ValueError('insufficient exact vector lineage evidence')
    version, kind = latest.model_version.iloc[0], latest.prediction_type.iloc[0]
    train = candidates.loc[(candidates.model_version == version) &
                           (candidates.prediction_type == kind)].copy()
    if not ((train.forecast_creation_time <= train.forecast_target_time) &
            train.horizon_hours.between(0, 72)).all():
        raise ValueError('noncausal training forecast')
    if not np.isfinite(train[['prediction', 'actual_w']].to_numpy(dtype=float)).all():
        raise ValueError('nonfinite training forecast/measurement')
    keys = ['forecast_creation_time', 'forecast_target_time']
    if train.duplicated(keys).any():
        raise ValueError('duplicate training keys')
    train['band'] = pd.cut(train.horizon_hours, [0, 6, 16.5, 36, 72], labels=BANDS, include_lowest=True)
    release_lag = pd.Timedelta(minutes=30+settings['availability_lag_minutes'])
    history = pd.Timedelta(days=settings['history_days'])
    corrections = {}
    for band in BANDS:
        selected = train.loc[train.band == band].sort_values('forecast_creation_time')
        selected = selected.drop_duplicates('forecast_target_time', keep='last')
        release = selected.forecast_target_time+release_lag
        selected = selected.loc[(release <= creation) & (release > creation-history)]
        count = len(selected)
        correction = float((selected.actual_w-selected.prediction).quantile(.65)) if count >= settings['min_targets'] else 0.
        corrections[band] = {'correction_w': correction, 'training_targets': count,
                             'latest_released_target': selected.forecast_target_time.max().isoformat() if count else None}
    synthetic = deepcopy(original)
    corrected_base = synthetic['input_snapshot']['states']['sensor.ai_load_forecast_high']['attributes']['forecasts']
    for row in corrected_base:
        lead = (pd.Timestamp(row['timestamp'])-creation).total_seconds()/3600
        # Current partial interval was not included in the causal forecast audit.
        if lead < 0 or lead > 72:
            continue
        band = BANDS[0 if lead <= 6 else 1 if lead <= 16.5 else 2 if lead <= 36 else 3]
        correction = corrections[band]['correction_w']
        row['power_load'] = max(0., float(row['power_load'])+correction)
    payload = build_dh_payload(Inputs(synthetic['input_snapshot']['states'], at.to_pydatetime()))
    for key in original_payload:
        if key != 'load_power_forecast' and payload[key] != original_payload[key]:
            raise ValueError('load challenger changed another input')
    synthetic['payloads']['dh'] = payload
    provenance = {'scope': 'synthetic_load_only_fixed_endpoint',
                  'original_handoff_revision': digest(original),
                  'original_input_revision': original['input_revision'],
                  'original_lineage': deepcopy(original['lineage']),
                  'original_publication_id': original['publication_id'],
                  'forecast_creation': creation.isoformat(), 'model_version': str(version),
                  'prediction_type': str(kind), 'matching_vector_targets': len(matched),
                  'settings': settings, 'corrections': corrections,
                  'publication_authorized': False}
    synthetic['input_revision'] = digest(synthetic['input_snapshot'])
    synthetic['mode'] = 'historical_load_counterfactual'
    synthetic['counterfactual'] = provenance
    return synthetic, provenance


def comparison(baseline, challenger):
    left, right = baseline['request'], challenger['request']
    if baseline['image'] != challenger['image'] or any(left[key] != right[key] for key in
            ('kind', 'captured_at', 'forecast_start', 'optimization_sha256')):
        raise ValueError('comparison requires identical origin and pinned solver')
    for key in left['payload']:
        if key != 'load_power_forecast' and left['payload'][key] != right['payload'][key]:
            raise ValueError('comparison requires fixed prices/PV/initial/terminal/policy')
    if left['configuration'] != right['configuration']:
        raise ValueError('comparison requires identical effective plant/config')
    a, b = validate_result(left, baseline['result']), validate_result(right, challenger['result'])
    original_summary, corrected_summary = forecast_summary(left, a), forecast_summary(right, b)
    return {'scope': 'forecast_cashflow_sensitivity_not_realised_savings',
            'baseline': original_summary, 'challenger': corrected_summary,
            'cashflow_difference_aud': corrected_summary['cashflow_aud']-original_summary['cashflow_aud'],
            'load_energy_difference_kwh': float((b.P_Load-a.P_Load).sum()*left['payload']['optimization_time_step']/60000),
            'first_battery_discharge_w': {'baseline': float(a.P_batt.iloc[0]), 'challenger': float(b.P_batt.iloc[0])},
            'soc_path_difference_max_percentage_points': float((a.SOC_opt-b.SOC_opt).abs().max()*100),
            'battery_path_difference_max_w': float((a.P_batt-b.P_batt).abs().max()),
            'ending_inventory_difference_kwh': float((b.SOC_opt.iloc[-1]-a.SOC_opt.iloc[-1])*left['payload']['battery_nominal_energy_capacity']/1000),
            'limitations': ['different forecast loads change predicted consumption: cashflow delta is not policy savings',
                            'single frozen origin; no sequential execution or independently measured available PV',
                            'p65 model version inferred from exact overlapping forecast vector, not archived HA version',
                            'historical measurement receipt lag assumed; frozen input/config/source provenance retained']}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--optimization-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    calibration = json.loads((args.calibration/'report.json').read_text())
    path = args.calibration/'challenger.parquet'
    if hashlib.sha256(path.read_bytes()).hexdigest() != calibration['challenger_sha256']:
        raise ValueError('changed calibration snapshot')
    with sqlite3.connect(args.journal.resolve().as_uri()+'?mode=ro', uri=True) as db:
        record = json.loads(db.execute('SELECT record FROM handoffs ORDER BY rowid DESC LIMIT 1').fetchone()[0])
    config = json.loads(args.config.read_text())
    synthetic, provenance = calibrated_handoff(record, pd.read_parquet(path), calibration['settings'])
    artifacts = []
    for handoff in (record, synthetic):
        request = prepare_request(handoff, config, kind='dh', optimization_sha256=args.optimization_sha256)
        if handoff is synthetic:
            request['counterfactual'] = provenance
            request.pop('request_id')
            request['request_id'] = digest(request)
        artifacts.append(run_request(request, args.image))
    report = comparison(*artifacts)
    report.update({'publication_authorized': False, 'counterfactual': provenance,
                   'calibration_report_sha256': hashlib.sha256((args.calibration/'report.json').read_bytes()).hexdigest()})
    args.output.mkdir(parents=True)
    for name, artifact in zip(('baseline', 'calibrated_load'), artifacts):
        (args.output/(name+'.json')).write_text(json.dumps(artifact, indent=2, allow_nan=False)+'\n')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__': main()
