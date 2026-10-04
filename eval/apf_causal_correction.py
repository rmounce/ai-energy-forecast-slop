"""Bounded causal APF horizon-bias packet diagnostic, never a site savings estimate."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.apf_packet_regret import load_inputs, add_control_histories, evaluate, sha
from eval.audit_amber_forecast_archive import utc

BINS = ((0., 2.), (2., 6.), (6., 14.))


def bucket(hours):
    for index, (lower, upper) in enumerate(BINS):
        if lower <= hours < upper:
            return index
    return None


def clusters(revisions):
    """Predeclared UTC-day groups; exclude every forecast receipt in current day."""
    days = sorted({utc(row['receipt']).date().isoformat() for row in revisions})
    return {row['receipt']: days.index(utc(row['receipt']).date().isoformat()) for row in revisions}


def prepare_labels(revisions, rates, groups):
    """Expand/index once; fit origins use primitive dictionaries, not pandas loc."""
    quotes = {utc(target): (float(row['rate']), utc(row['receipt']))
              for target, row in rates.iterrows()}
    records = []
    for revision in sorted(revisions, key=lambda r: utc(r['receipt'])):
        receipt = utc(revision['receipt'])
        for row in revision['rows']:
            end = utc(row['end_time'])
            for target in pd.date_range(end-pd.Timedelta(minutes=row['duration']), end, freq='5min', inclusive='left'):
                index = bucket((target-receipt).total_seconds()/3600)
                if index is None or target <= receipt or target not in quotes:
                    continue
                rate, quote_receipt = quotes[target]
                residual = rate+float(row['advanced_price_predicted'])
                if not math.isfinite(residual):
                    raise ValueError('nonfinite residual')
                records.append({'target': target.isoformat(), 'bin': index,
                    'forecast_receipt': receipt.isoformat(), 'quote_receipt': quote_receipt.isoformat(),
                    'interval_end': (target+pd.Timedelta(minutes=5)).isoformat(),
                    'cluster': groups[revision['receipt']], 'residual': residual,
                    '_receipt': receipt, '_quote_receipt': quote_receipt,
                    '_end': target+pd.Timedelta(minutes=5)})
    return records


def fit_bias(revisions, rates, origin, groups, *, minimum_clusters=2,
             minimum_targets=12, ridge_targets=24., max_bias=.10, release_lag_minutes=0., prepared=None):
    """Newest fully future vintage per target/bin, completed earlier labels only.

    Final canonical quotes with later receipt are omitted even if a preliminary
    quote existed earlier. Entire current UTC day is held out.
    """
    if minimum_clusters < 1 or minimum_targets < 1 or not all(
            math.isfinite(x) and x >= 0 for x in (ridge_targets, max_bias, release_lag_minutes)):
        raise ValueError('invalid fit settings')
    origin = utc(origin)
    current_group = groups[origin.isoformat()]
    cutoff = origin-pd.Timedelta(minutes=release_lag_minutes)
    prepared = prepare_labels(revisions, rates, groups) if prepared is None else prepared
    records = {}
    for row in prepared:
        if (row['_receipt'] >= origin or row['cluster'] >= current_group or
                row['_end'] >= cutoff or row['_quote_receipt'] >= cutoff):
            continue
        key = (row['target'], row['bin'])
        records[key] = {k: v for k, v in row.items() if not k.startswith('_')}
    fitted = []
    for index, limits in enumerate(BINS):
        rows = [r for r in records.values() if r['bin'] == index]
        count = len(rows)
        group_count = len({r['cluster'] for r in rows})
        supported = count >= minimum_targets and group_count >= minimum_clusters
        raw = sum(r['residual'] for r in rows)/(count+ridge_targets) if count else 0.
        fitted.append({'hours': list(limits), 'targets': count, 'earlier_clusters': group_count,
            'supported': supported, 'bias_aud_per_kwh': max(-max_bias, min(max_bias, raw)) if supported else 0.})
    return {'origin': origin.isoformat(), 'current_cluster': current_group,
        'bins': fitted, 'training_labels': list(records.values()),
        'supported': all(row['supported'] for row in fitted)}


def corrected_revision(revision, origin, fit):
    """Split forecast intervals so horizon bins apply to identical five-minute slots."""
    result = copy.deepcopy(revision)
    rows = []
    for original in revision['rows']:
        end = utc(original['end_time'])
        for target in pd.date_range(end-pd.Timedelta(minutes=original['duration']), end, freq='5min', inclusive='left'):
            row = copy.deepcopy(original)
            row['duration'] = 5
            row['end_time'] = (target+pd.Timedelta(minutes=5)).isoformat()
            row['start_time'] = target.isoformat()
            index = bucket((target-utc(origin)).total_seconds()/3600)
            bias = fit['bins'][index]['bias_aud_per_kwh'] if index is not None else 0.
            # API feed sign is cost; revenue bias therefore subtracts from raw API price.
            for field in ('advanced_price_low', 'advanced_price_predicted', 'advanced_price_high'):
                row[field] = float(row[field])-bias
            rows.append(row)
    result['rows'] = rows
    return result


def run(revisions, rates, *, horizons=(6, 12, 14), terminals=(0., .20), evaluation_origins=None):
    groups = clusters(revisions)
    results, excluded, fits = [], [], []
    prepared = prepare_labels(revisions, rates, groups)
    for revision in sorted(revisions, key=lambda r: utc(r['receipt'])):
        origin = revision['receipt']
        if evaluation_origins is not None and origin not in evaluation_origins:
            continue
        fit = fit_bias(revisions, rates, origin, groups, prepared=prepared)
        fits.append(fit)
        corrected = corrected_revision(revision, origin, fit)
        for horizon in horizons:
            for terminal in terminals:
                try:
                    baseline = evaluate(revision, rates, origin, horizon, terminal_value_aud_per_dc_kwh=terminal)
                    challenger = evaluate(corrected, rates, origin, horizon, terminal_value_aud_per_dc_kwh=terminal)
                except ValueError as exc:
                    excluded.append({'origin': origin, 'horizon_hours': horizon, 'terminal': terminal, 'reason': str(exc)})
                    continue
                results.append({'origin': origin, 'cluster': groups[origin], 'horizon_hours': horizon,
                    'terminal': terminal, 'fit_supported': fit['supported'],
                    'baseline': baseline['candidates']['predicted'],
                    'corrected': challenger['candidates']['predicted'],
                    'oracle_incremental_value_aud': baseline['oracle_incremental_value_aud']})
    summary = []
    for group in sorted(set(groups.values())):
        for terminal in terminals:
            for horizon in horizons:
                rows = [r for r in results if r['cluster'] == group and r['terminal'] == terminal and r['horizon_hours'] == horizon]
                if not rows:
                    continue
                summary.append({'cluster': group, 'origin_date': rows[0]['origin'][:10], 'terminal': terminal,
                    'horizon_hours': horizon, 'cases': len(rows), 'supported_cases': sum(r['fit_supported'] for r in rows),
                    'mean_baseline_regret_cents_per_kwh': sum(r['baseline']['regret_cents_per_stored_kwh'] for r in rows)/len(rows),
                    'mean_corrected_regret_cents_per_kwh': sum(r['corrected']['regret_cents_per_stored_kwh'] for r in rows)/len(rows),
                    'changed_choices': sum(r['baseline']['selected_target'] != r['corrected']['selected_target'] for r in rows)})
    return {'results': results, 'fits': fits, 'exclusions': excluded, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--quotes', type=Path, required=True)
    parser.add_argument('--control-history', type=Path, nargs='*', default=[])
    parser.add_argument('--evaluation-history', type=Path, help='Predeclared receipt selection; included in training archive too')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    revisions, rates, provenance = load_inputs(args.archive, args.quotes)
    revisions, sources = add_control_histories(revisions, args.control_history)
    provenance['control_histories'] = sources
    selected = None
    if args.evaluation_history is not None:
        selection, selection_sources = add_control_histories([], [args.evaluation_history])
        selected = {row['receipt'] for row in selection}
        revisions, extra_sources = add_control_histories(revisions, [args.evaluation_history])
        provenance['evaluation_history'] = selection_sources
        provenance['evaluation_origins'] = sorted(selected)
    report = run(revisions, rates, evaluation_origins=selected)
    report.update({'scope': 'causal_horizon_bias_conditional_packet_not_site_savings',
        'publication_authorized': False, 'provenance': provenance, 'code_sha256': sha(Path(__file__)),
        'settings': {'minimum_prior_utc_days': 2, 'minimum_targets_per_bin': 12, 'ridge_targets': 24,
            'maximum_absolute_bias_aud_per_kwh': .10, 'release_lag_minutes': 0,
            'bin_hours': BINS, 'packet_kwh': .25, 'dc_efficiency': .99, 'inverter_efficiency': .95,
            'wear_per_dc_kwh': .04, 'terminal_values': [0., .20]},
        'limitations': ['Only earlier UTC-day forecast receipts train; target/bin pairs deduplicated but bins and UTC days remain correlated.',
            'Final canonical quote receipt and interval end strictly precede origin; later revisions omitted.',
            'Unsupported bins use zero correction; do not treat cold-start fallback as fitted evidence.',
            'Sparse frozen windows, no independent test week or site constraints; no deployable savings claim.',
            'Fixed bins, shrinkage and clipping are diagnostic defaults, not selected by held-out results.']})
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
