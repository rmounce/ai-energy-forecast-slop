"""Offline causal load-calibration challenger on frozen, measured forecast windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


QUANTILES = {'load': .5, 'load_p65': .65, 'load_p75': .75}
BANDS = ['0–6h', '6–16.5h', '16.5–36h', '36–72h']


def calibrate(rows, frame, *, history_days=3, min_targets=48, availability_lag_minutes=30):
    """Rolling residual quantiles; each past target contributes once per model/band.

    Latest causal vintage within the band represents each training target. A target
    becomes available after its entire 30-minute interval plus an explicit lag.
    Every scored vintage uses a correction fitted before its own creation time.
    """
    if history_days <= 0 or min_targets < 1 or availability_lag_minutes < 0:
        raise ValueError('invalid calibration history/availability settings')
    target = frame.load_base_w.resample('30min').sum(min_count=6)/6
    rows = rows.loc[rows.model_name.isin(QUANTILES)].copy()
    rows['actual_w'] = rows.forecast_target_time.map(target)
    rows = rows.dropna(subset=['actual_w', 'prediction', 'model_version'])
    causal = (rows.forecast_creation_time <= rows.forecast_target_time) & rows.horizon_hours.between(0, 72)
    if not causal.all() or not np.isfinite(rows[['prediction', 'actual_w']].to_numpy(dtype=float)).all():
        raise ValueError('noncausal or nonfinite input')
    keys = ['model_name', 'model_version', 'prediction_type', 'forecast_creation_time', 'forecast_target_time']
    if rows.duplicated(keys).any():
        raise ValueError('duplicate forecast keys')
    rows['horizon_band'] = pd.cut(rows.horizon_hours, [0, 6, 16.5, 36, 72], labels=BANDS, include_lowest=True)
    rows['correction_w'] = 0.
    rows['training_targets'] = 0
    history = pd.Timedelta(days=history_days)
    lag = pd.Timedelta(minutes=30+availability_lag_minutes)
    grouping = ['model_name', 'model_version', 'prediction_type', 'horizon_band']
    for (model, _, _, _), group in rows.groupby(grouping, observed=True):
        train = group.sort_values('forecast_creation_time').drop_duplicates('forecast_target_time', keep='last')
        train = train.sort_values('forecast_target_time')
        release = train.forecast_target_time+lag
        residual = train.actual_w-train.prediction
        for creation, issued in group.groupby('forecast_creation_time'):
            available = (release <= creation) & (release > creation-history)
            count = int(available.sum())
            rows.loc[issued.index, 'training_targets'] = count
            if count >= min_targets:
                rows.loc[issued.index, 'correction_w'] = float(residual.loc[available].quantile(QUANTILES[model]))
    rows['calibrated_w'] = (rows.prediction+rows.correction_w).clip(lower=0)
    rows['calibration_ready'] = rows.training_targets >= min_targets
    return rows


def scores(rows):
    """Equal target weights within each day/model/band; paired eligible vintages."""
    results = []
    rows = rows.loc[rows.calibration_ready].copy()
    rows['utc_date'] = rows.forecast_target_time.dt.strftime('%Y-%m-%d')
    for (model, version, kind, band, day), group in rows.groupby(
            ['model_name', 'model_version', 'prediction_type', 'horizon_band', 'utc_date'], observed=True):
        q = QUANTILES[model]
        result = {'model': str(model), 'model_version': str(version), 'prediction_type': str(kind),
                  'horizon_band': str(band), 'utc_date': day, 'rows': len(group),
                  'targets': group.forecast_target_time.nunique(),
                  'mean_correction_w': float(group.groupby('forecast_target_time').correction_w.mean().mean())}
        for name, column in [('baseline', 'prediction'), ('calibrated', 'calibrated_w')]:
            error = group[column]-group.actual_w
            metrics = pd.DataFrame({'target': group.forecast_target_time, 'bias_w': error,
                'mae_w': error.abs(), 'pinball_w': np.where(error >= 0, (1-q)*error, -q*error),
                'coverage': (error >= 0).astype(float)})
            result[name] = {key: float(value) for key, value in metrics.groupby('target').mean().mean().items()}
        results.append(result)
    return results


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--history-days', type=float, default=3)
    parser.add_argument('--min-targets', type=int, default=48)
    parser.add_argument('--availability-lag-minutes', type=float, default=30)
    args = parser.parse_args()
    if args.output.exists(): parser.error('output already exists')
    manifest_path = args.dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    actual_path = args.dataset/'actuals.parquet'
    audit = json.loads((args.audit/'report.json').read_text())
    forecast_path = args.audit/'load_forecast_window.parquet'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if not manifest.get('export_complete') or digest(actual_path) != manifest['parquet_sha256']:
        raise ValueError('incomplete or changed measured dataset')
    evidence = audit['forecast_logs']['load']
    if (not evidence['stable_during_read'] or digest(forecast_path) != evidence['filtered_parquet_sha256']
            or digest(manifest_path) != audit['dataset_manifest_sha256']):
        raise ValueError('forecast snapshot changed or does not match measured dataset')
    rows = calibrate(pd.read_parquet(forecast_path), pd.read_parquet(actual_path),
        history_days=args.history_days, min_targets=args.min_targets,
        availability_lag_minutes=args.availability_lag_minutes)
    report = {'scope': 'causal_forecast_challenger_not_economic_savings',
        'settings': {'history_days': args.history_days, 'min_targets': args.min_targets,
                     'availability_lag_minutes': args.availability_lag_minutes},
        'method': 'rolling residual quantile; latest vintage per training target/band; equal scored target weights',
        'rows': len(rows), 'eligible_rows': int(rows.calibration_ready.sum()),
        'dataset_manifest_sha256': digest(manifest_path), 'forecast_parquet_sha256': digest(forecast_path),
        'code_sha256': digest(Path(__file__)), 'scores': scores(rows),
        'publication_authorized': False,
        'limitations': ['assumed measurement availability lag; historical receipt latency unverified',
            'short single-site window, overlapping vintages and days are dependent',
            'horizon-band correction may not transfer across seasons or regimes',
            'independent quantile corrections can cross; not a deployable quantile bundle',
            'no tariff, dispatch, ending-inventory or realised savings comparison']}
    args.output.mkdir(parents=True)
    rows.to_parquet(args.output/'challenger.parquet', index=False)
    report['challenger_sha256'] = digest(args.output/'challenger.parquet')
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'rows': len(rows), 'eligible_rows': report['eligible_rows']}))


if __name__ == '__main__': main()
