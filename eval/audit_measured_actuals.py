"""Offline coverage/consistency and as-issued load diagnostics; not savings attribution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def statistics(values):
    values = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    return {'count': len(values), 'mean': float(values.mean()) if len(values) else None,
            'median': float(values.median()) if len(values) else None,
            'absolute_p95': float(values.abs().quantile(.95)) if len(values) else None}


def measured_report(frame):
    report = {'scope': 'observed_telemetry_diagnostics_not_savings', 'intervals': len(frame),
              'coverage': {}, 'consistency': {}, 'daily_energy': [], 'pv_nowcast_diagnostic': {}}
    columns = ('pv_dc_w', 'grid_import_w', 'battery_charge_w', 'load_site_w', 'load_base_w',
               'planned_battery_discharge_w', 'soc_pct_end', 'curtailment_fraction')
    for column in columns:
        report['coverage'][column] = int(frame[column].notna().sum()) if column in frame else 0
    for name, columns, signs in (
            ('ac_site_minus_grid_inverter_w', ('load_site_w', 'grid_import_w', 'inverter_ac_w'), (1, -1, -1)),
            ('dc_balance_derived_loss_w', ('pv_dc_w', 'battery_charge_w', 'inverter_ac_w', 'conversion_loss_w'), (1, -1, -1, -1)),
            ('actual_minus_planned_charge_w', ('battery_charge_w', 'planned_battery_discharge_w'), (1, 1))):
        if all(column in frame for column in columns):
            values = sum(frame[column]*sign for column, sign in zip(columns, signs))
            report['consistency'][name] = statistics(values)
    if {'pv_dc_w', 'pv_proxy_w', 'curtailment_fraction'} <= set(frame):
        paired = frame[['pv_dc_w', 'pv_proxy_w', 'curtailment_fraction']].dropna()
        selected = paired[(paired.curtailment_fraction == 0) & (paired.pv_dc_w >= 500)]
        report['pv_nowcast_diagnostic'] = {
            'scope': 'logged_current_estimate_not_future_forecast_skill',
            'paired_complete_intervals': len(paired),
            'strict_measured_mode_above_500w': len(selected),
            'error_w': statistics(selected.pv_proxy_w-selected.pv_dc_w),
            'mae_w': float((selected.pv_proxy_w-selected.pv_dc_w).abs().mean()) if len(selected) else None,
            'estimate_to_delivered_energy_ratio': float(selected.pv_proxy_w.sum()/selected.pv_dc_w.sum()) if len(selected) else None,
            'reconstruction_context_time_fraction': float(frame.curtailment_fraction.mean())}
    energy_cols = ('grid_import_w_import_kwh', 'grid_import_w_export_kwh',
                   'battery_charge_w_charge_kwh', 'battery_charge_w_discharge_kwh')
    for date, group in frame.groupby(frame.index.date):
        report['daily_energy'].append({'utc_date': str(date),
            **{column: {'observed_kwh': float(group[column].sum()) if group[column].notna().any() else None,
                        'complete_intervals': int(group[column].notna().sum())}
               for column in energy_cols if column in group}})
    return report


def log_window(path, start, end):
    """Read causal rows for completed targets, excluding actual columns entirely."""
    before = path.stat()
    selected, invalid, negative_horizon = [], 0, 0
    columns = ['forecast_creation_time', 'forecast_target_time', 'model_name', 'model_version',
               'prediction_type', 'prediction', 'power_pv']
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250000, dtype='string'):
        target = pd.to_datetime(chunk.forecast_target_time, utc=True, format='mixed', errors='coerce')
        mask = (target >= start) & (target < end)
        if not mask.any():
            continue
        chunk = chunk.loc[mask].copy()
        chunk['forecast_target_time'] = target.loc[mask]
        chunk['forecast_creation_time'] = pd.to_datetime(chunk.forecast_creation_time, utc=True,
                                                        format='mixed', errors='coerce')
        chunk['prediction'] = pd.to_numeric(chunk.prediction, errors='coerce').replace([np.inf, -np.inf], np.nan)
        chunk['power_pv'] = pd.to_numeric(chunk.power_pv, errors='coerce')
        invalid += int(chunk[['forecast_creation_time', 'prediction']].isna().any(axis=1).sum())
        valid = chunk.forecast_creation_time.notna() & chunk.prediction.notna()
        horizon = (chunk.forecast_target_time-chunk.forecast_creation_time).dt.total_seconds()/3600
        negative_horizon += int((valid & (horizon < 0)).sum())
        chunk = chunk.loc[valid & (horizon >= 0) & (horizon <= 72)].copy()
        chunk['horizon_hours'] = horizon.loc[chunk.index]
        selected.append(chunk)
    after = path.stat()
    stable = (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    rows = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns+['horizon_hours'])
    keys = ['model_name', 'prediction_type', 'forecast_creation_time', 'forecast_target_time']
    duplicates = rows.duplicated(keys, keep=False)
    evidence = {'path': str(path), 'size_before': before.st_size, 'size_after': after.st_size,
        'mtime_ns_before': before.st_mtime_ns, 'mtime_ns_after': after.st_mtime_ns,
        'stable_during_read': stable, 'invalid_rows_in_target_window': invalid,
        'partial_interval_rows_excluded': negative_horizon, 'duplicate_rows_excluded': int(duplicates.sum()),
        'causal_rows': int((~duplicates).sum()), 'series': []}
    rows = rows.loc[~duplicates].copy()
    for (model, kind), group in rows.groupby(['model_name', 'prediction_type'], dropna=False):
        evidence['series'].append({'model': None if pd.isna(model) else str(model),
            'prediction_type': None if pd.isna(kind) else str(kind), 'rows': len(group),
            'model_versions': sorted(str(value) for value in group.model_version.dropna().unique()),
            'creations': group.forecast_creation_time.nunique(), 'targets': group.forecast_target_time.nunique()})
    return rows, evidence


def load_scores(rows, frame):
    if 'load_base_w' not in frame:
        return []
    # All six complete measured bins required. No resample gap fill or interpolation.
    target = frame.load_base_w.resample('30min').sum(min_count=6)/6
    rows = rows.copy()
    rows['measured_base_w'] = rows.forecast_target_time.map(target)
    rows = rows.dropna(subset=['measured_base_w'])
    rows['horizon_band'] = pd.cut(rows.horizon_hours, [0, 6, 16.5, 36, 72],
                                 labels=['0–6h', '6–16.5h', '16.5–36h', '36–72h'], include_lowest=True)
    scores = []
    for (model, kind, band), group in rows.groupby(['model_name', 'prediction_type', 'horizon_band'], observed=True):
        error = group.prediction-group.measured_base_w
        scores.append({'model': str(model), 'prediction_type': str(kind), 'horizon_band': str(band),
            'rows': len(group), 'distinct_targets': group.forecast_target_time.nunique(),
            'bias_w': float(error.mean()), 'mae_w': float(error.abs().mean()),
            'actual_at_or_below_prediction_fraction': float((error >= 0).mean())})
    return scores


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new ignored audit directory')
    parser.add_argument('--load-log', type=Path)
    parser.add_argument('--price-log', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output directory already exists')
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    path = args.dataset/'actuals.parquet'
    if not manifest.get('export_complete') or hashlib.sha256(path.read_bytes()).hexdigest() != manifest['parquet_sha256']:
        raise ValueError('measured export incomplete or content changed')
    frame = pd.read_parquet(path)
    report = measured_report(frame)
    report['dataset_manifest_sha256'] = hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
    report['auditor_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['forecast_logs'] = {}
    report['limitations'] = ['two-day diagnostics do not establish economic rankings',
        'load scores use all causal vintages, not a matched walk-forward experiment',
        'no settled price/rates or counterfactual available-PV target included',
        'current PV estimate is not a forecast-vintage archive']
    args.output.mkdir(parents=True)
    for name, log in (('load', args.load_log), ('price', args.price_log)):
        if log:
            print(f'Reading {name} forecast log for measured target window', flush=True)
            rows, evidence = log_window(log, pd.Timestamp(manifest['start']), pd.Timestamp(manifest['end']))
            report['forecast_logs'][name] = evidence
            if evidence['stable_during_read']:
                snapshot = args.output/(name+'_forecast_window.parquet')
                rows.to_parquet(snapshot, index=False)
                evidence['filtered_parquet_sha256'] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                if name == 'load': report['load_diagnostics'] = load_scores(rows, frame)
            else:
                evidence['excluded_reason'] = 'source_changed_during_read'
            print(json.dumps(evidence), flush=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'measured_consistency': report['consistency'],
                      'load_scores': report.get('load_diagnostics', [])}), flush=True)


if __name__ == '__main__':
    main()
