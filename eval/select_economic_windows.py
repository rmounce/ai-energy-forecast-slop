"""Select diagnostic trading regimes from frozen measured targets, not savings."""
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

REGIMES = ('high_value_export', 'low_solar_export', 'near_full_negative_buy', 'near_full_pv')
REQUIRED = ('pv_dc_w', 'load_site_w', 'battery_charge_w', 'grid_import_w', 'soc_pct_end', 'buy_rate', 'feed_rate')


def select_windows(frame, minutes=30):
    if minutes not in (15, 30) or not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError('aware index and 15/30m window required')
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError('ordered unique target times required')
    if any(at != at.floor('5min') for at in frame.index.tz_convert('UTC')):
        raise ValueError('five-minute target boundaries required')
    if not set(REQUIRED) <= set(frame): raise ValueError('missing measured/rate columns')
    count = minutes//5
    candidates = {key: [] for key in REGIMES}
    missing = 0
    for i in range(1, len(frame)-count+1):
        sample = frame.iloc[i:i+count]
        at = sample.index[0]
        dates = pd.date_range(at, periods=count, freq='5min')
        initial = frame.iloc[i-1].soc_pct_end
        columns = list(REQUIRED)+(['grid_import_w_export_kwh'] if 'grid_import_w_export_kwh' in sample else [])
        if (not sample.index.equals(dates) or frame.index[i-1] != at-pd.Timedelta(minutes=5)
                or not np.isfinite(initial) or not np.isfinite(sample[columns].to_numpy()).all()):
            missing += 1
            continue
        if min(sample.pv_dc_w.min(), sample.load_site_w.min(), sample.soc_pct_end.min(), initial) < 0 or max(initial, sample.soc_pct_end.max()) > 100:
            raise ValueError('invalid physical target')
        pv, feed, buy = float(sample.pv_dc_w.mean()), float(sample.feed_rate.mean()), float(sample.buy_rate.mean())
        export = float(sample.get('grid_import_w_export_kwh', (-sample.grid_import_w).clip(lower=0)*5/60000).sum())
        if 'grid_import_w_export_kwh' in sample and sample.grid_import_w_export_kwh.min() < 0:
            raise ValueError('invalid export energy')
        summary = {'start': at.isoformat(), 'end': (dates[-1]+pd.Timedelta(minutes=5)).isoformat(),
            'minutes': minutes, 'initial_soc_pct': float(initial), 'final_soc_pct': float(sample.soc_pct_end.iloc[-1]),
            'minimum_soc_pct': float(sample.soc_pct_end.min()), 'maximum_soc_pct': float(sample.soc_pct_end.max()),
            'mean_delivered_pv_w': pv, 'mean_feed_rate': feed, 'mean_buy_rate': buy,
            'observed_export_kwh': export,
            'observed_export_revenue_aud': float((sample.feed_rate*sample.get('grid_import_w_export_kwh',
                (-sample.grid_import_w).clip(lower=0)*5/60000)).sum())}
        eligible = {
            'high_value_export': (feed >= .15 and initial >= 25, feed),
            'low_solar_export': (pv <= 500 and feed >= .10 and initial >= 20, feed),
            'near_full_negative_buy': (sample.soc_pct_end.min() >= 97 and sample.soc_pct_end.max() >= 99.5
                and pv >= 1000 and buy < 0, -buy),
            'near_full_pv': (sample.soc_pct_end.min() >= 97 and sample.soc_pct_end.max() >= 99.5
                and pv >= 1000, pv),
        }
        for regime, (admitted, score) in eligible.items():
            if admitted: candidates[regime].append(summary | {'selection_score': score})
    selected, counts = {}, {key: len(rows) for key, rows in candidates.items()}
    # Priority order fixed before ranking; choose highest-score non-overlapping
    # window per regime, with earliest origin as deterministic tie-break.
    for regime in REGIMES:
        for row in sorted(candidates[regime], key=lambda row: (-row['selection_score'], row['start'])):
            if any(pd.Timestamp(row['start']) < pd.Timestamp(other['end']) and
                   pd.Timestamp(row['end']) > pd.Timestamp(other['start']) for other in selected.values()):
                continue
            selected[regime] = row
            break
    return {'mode': 'retrospective_regime_selection_not_economic_validation', 'selected': selected,
        'candidate_counts': counts, 'incomplete_candidate_windows': missing,
        'publication_authorized': False, 'selection': {
            'minutes': minutes, 'priority': REGIMES, 'price_threshold_export_aud_per_kwh': .15,
            'low_solar_max_mean_w': 500, 'low_solar_min_feed_aud_per_kwh': .10,
            'near_full_min_soc_pct': 97, 'near_full_peak_soc_pct': 99.5,
            'near_full_min_pv_w': 1000, 'near_full_mean_buy_must_be_negative': True},
        'limitations': ['selection uses realised prices/state to diagnose distinct stress regimes',
            'not random/out-of-sample selection; results cannot estimate average achievable savings',
            'complete delivered PV required; night stable-zero recording gaps not filled',
            'independent telemetry coverage is not archived forecast-input coverage',
            'observed export revenue is not missed-opportunity or controller headroom']}


def main():
    os.nice(19)
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('dataset', 'quotes', 'output'): parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--minutes', type=int, choices=(15,30), default=30)
    args = parser.parse_args()
    if args.output.exists(): parser.error('new output directory required')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    actual = args.dataset/'actuals.parquet'
    am, qm = [json.loads((folder/'manifest.json').read_text()) for folder in (args.dataset,args.quotes)]
    if (not am['export_complete'] or am['parquet_sha256'] != sha(actual) or not qm['export_complete']
            or qm['measured_actuals_sha256'] != sha(actual)):
        raise ValueError('incompatible or changed measured/quote datasets')
    frame = pd.read_parquet(actual)
    for leg, column in [('general','buy_rate'),('feed','feed_rate')]:
        path = args.quotes/(leg+'_rates.parquet')
        if qm['files'][path.name] != sha(path): raise ValueError('changed scoring rates')
        frame = frame.join(pd.read_parquet(path)[['rate']].rename(columns={'rate':column}))
    report = select_windows(frame,args.minutes)
    report['provenance'] = {'actuals_sha256':sha(actual), 'quote_manifest_sha256':sha(args.quotes/'manifest.json'),
        'selector_sha256':sha(Path(__file__))}
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False))


if __name__ == '__main__': main()
