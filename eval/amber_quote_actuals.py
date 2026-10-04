"""Strict archived Amber interval rates and observed-flow accounting, not invoices."""
from collections import Counter
import math

import pandas as pd


def utc(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('timestamp must be timezone-aware')
    return stamp.tz_convert('UTC')


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError('nonfinite numeric value')
    if not math.isfinite(result):
        raise ValueError('nonfinite numeric value')
    return result


def canonical_rates(rows, start, end, *, require_post_end_receipt=False):
    """Latest receipt per quoted interval; never substitute older valid revisions."""
    start, end = utc(start), utc(end)
    buckets, rejected = {}, Counter()
    for row in rows:
        try:
            quoted_end = utc(row.get('end_time_str'))
            receipt = utc(row.get('time'))
        except (ValueError, TypeError):
            rejected['invalid_identity'] += 1
            continue
        target = quoted_end-pd.Timedelta(minutes=5)
        if target < start or target >= end:
            rejected['outside_target_window'] += 1
            continue
        reason = None
        try:
            rate = finite(row.get('value'))
            if row.get('unit_of_measurement_str') != '$/kWh':
                reason = 'unit_mismatch'
            elif row.get('type_str') != 'CurrentInterval' or finite(row.get('estimate')) != 0:
                reason = 'not_confirmed_current_interval'
            elif finite(row.get('duration')) != 5:
                reason = 'unsupported_duration'
            elif quoted_end != quoted_end.floor('5min'):
                reason = 'unaligned_end'
            elif utc(row.get('start_time_str')) != target+pd.Timedelta(seconds=1):
                reason = 'start_convention_mismatch'
            elif receipt < target:
                reason = 'receipt_before_interval'
            elif require_post_end_receipt and receipt < quoted_end:
                reason = 'latest_receipt_before_interval_end'
        except (ValueError, TypeError):
            rate, reason = None, 'invalid_attributes'
        buckets.setdefault(target, []).append((receipt, rate, reason))
    accepted = []
    for target, revisions in sorted(buckets.items()):
        receipt = max(item[0] for item in revisions)
        latest = {(item[1], item[2]) for item in revisions if item[0] == receipt}
        if len(latest) != 1:
            rejected['conflicting_latest_revision'] += 1
            continue
        rate, reason = latest.pop()
        if reason:
            rejected[reason] += 1
            continue
        accepted.append({'target': target, 'rate': rate, 'receipt': receipt,
                         'revision_count': len(revisions),
                         'received_at_or_after_end': receipt >= target+pd.Timedelta(minutes=5),
                         'receipt_lag_seconds': (receipt-target-pd.Timedelta(minutes=5)).total_seconds()})
    frame = pd.DataFrame(accepted, columns=['target', 'rate', 'receipt', 'revision_count', 'received_at_or_after_end', 'receipt_lag_seconds'])
    frame = frame.set_index('target')
    frame.index = pd.DatetimeIndex(frame.index, tz='UTC') if frame.empty else pd.DatetimeIndex(frame.index)
    return frame, dict(rejected)


def adjusted_rates(rows, raw_rates, raw_rows):
    """Recorded local adjustment must match latest raw quote at receipt and final raw."""
    buckets, rejected = {}, Counter()
    raw_by_end = {}
    for row in raw_rows:
        try:
            raw_by_end.setdefault(utc(row.get('end_time_str')), []).append((utc(row.get('time')), row))
        except (ValueError, TypeError):
            continue
    for row in rows:
        try:
            target = utc(row.get('confirmed_end_time_str'))-pd.Timedelta(minutes=5)
            receipt = utc(row.get('time'))
        except (ValueError, TypeError):
            rejected['invalid_identity'] += 1
            continue
        if target not in raw_rates.index:
            rejected['no_confirmed_raw_interval'] += 1
            continue
        buckets.setdefault(target, []).append((receipt, row))
    accepted = []
    for target, revisions in sorted(buckets.items()):
        receipt = max(item[0] for item in revisions)
        latest = [row for stamp, row in revisions if stamp == receipt]
        try:
            signatures = {(finite(row.get('value')), finite(row.get('raw_price')),
                           finite(row.get('export_allowance_adjustment')), row.get('unit_of_measurement_str'))
                          for row in latest}
            if len(signatures) != 1:
                raise ValueError('conflicting_latest_revision')
            value, raw, adjustment, unit = signatures.pop()
            if unit != '$/kWh' or abs(value-raw-adjustment) > 1e-6:
                raise ValueError('adjustment_identity_mismatch')
            if abs(raw-raw_rates.loc[target, 'rate']) > 1e-6:
                raise ValueError('final_raw_revision_mismatch')
            # Raw source must have emitted the same quote by this adjusted receipt.
            prior = [row for stamp, row in raw_by_end.get(target+pd.Timedelta(minutes=5), []) if stamp <= receipt]
            if not prior:
                raise ValueError('no_raw_quote_at_receipt')
            stamp = max(utc(row['time']) for row in prior)
            matches = [row for row in prior if utc(row['time']) == stamp]
            snapshot, errors = canonical_rates(matches, target, target+pd.Timedelta(minutes=5))
            if errors or snapshot.empty or abs(snapshot.iloc[0]['rate']-raw) > 1e-6:
                raise ValueError('raw_quote_at_receipt_mismatch')
        except (ValueError, TypeError) as exc:
            rejected[str(exc)] += 1
            continue
        accepted.append({'target': target, 'rate': value, 'raw_rate': raw, 'adjustment': adjustment,
                         'receipt': receipt, 'revision_count': len(revisions)})
    frame = pd.DataFrame(accepted, columns=['target', 'rate', 'raw_rate', 'adjustment', 'receipt', 'revision_count']).set_index('target')
    frame.index = pd.DatetimeIndex(frame.index, tz='UTC') if frame.empty else pd.DatetimeIndex(frame.index)
    return frame, dict(rejected)


def accounting(actuals, general, feed, adjusted):
    frame = actuals[['grid_import_w_import_kwh', 'grid_import_w_export_kwh']].copy()
    frame.columns = ['import_kwh', 'export_kwh']
    frame['general_rate'] = general['rate']
    frame['feed_rate'] = feed['rate']
    frame['adjusted_feed_rate'] = adjusted['rate']
    # These are MQTT CURRENT SENSOR STATES, not raw Forecasts.per_kwh.
    # Installed amber2mqtt negates API feed-in per_kwh: positive state earns.
    frame['import_cost'] = frame.import_kwh*frame.general_rate
    frame['export_cost'] = -frame.export_kwh*frame.feed_rate
    frame['raw_cost'] = frame.import_cost+frame.export_cost
    frame['adjusted_cost'] = frame.import_kwh*frame.general_rate-frame.export_kwh*frame.adjusted_feed_rate
    paired = frame.dropna()
    raw_priced = frame.dropna(subset=['raw_cost'])
    report = {'intervals': len(frame), 'general_intervals': int(frame.general_rate.notna().sum()),
              'feed_intervals': int(frame.feed_rate.notna().sum()),
              'complete_energy_intervals': int(frame[['import_kwh', 'export_kwh']].notna().all(axis=1).sum()),
              'raw_priced_energy_intervals': int(frame.raw_cost.notna().sum()),
              'paired_raw_adjusted_intervals': len(paired),
              'raw_observed_cost_dollars': float(frame.raw_cost.sum(min_count=1)),
              'raw_import_cost_dollars': float(raw_priced.import_cost.sum(min_count=1)),
              'raw_export_cost_dollars': float(raw_priced.export_cost.sum(min_count=1)),
              'raw_chargeable_export_cost_dollars': float(raw_priced.export_cost.clip(lower=0).sum()),
              'raw_export_revenue_dollars': float(-raw_priced.export_cost.clip(upper=0).sum()),
              'chargeable_export_kwh': float(raw_priced.loc[raw_priced.feed_rate < 0, 'export_kwh'].sum()),
              'chargeable_export_intervals': int(((raw_priced.feed_rate < 0) & (raw_priced.export_kwh > 0)).sum()),
              'paired_raw_cost_dollars': float(paired.raw_cost.sum(min_count=1)),
              'paired_adjusted_scenario_cost_dollars': float(paired.adjusted_cost.sum(min_count=1)),
              'paired_adjustment_cost_dollars': float((paired.adjusted_cost-paired.raw_cost).sum(min_count=1)),
              'daily': []}
    for day, part in frame.groupby(frame.index.floor('D')):
        valid = part.dropna(subset=['raw_cost'])
        report['daily'].append({'date': day.date().isoformat(), 'priced_intervals': len(valid),
                              'raw_cost_dollars': float(valid.raw_cost.sum(min_count=1)),
                              'import_cost_dollars': float(valid.import_cost.sum(min_count=1)),
                              'export_cost_dollars': float(valid.export_cost.sum(min_count=1)),
                              'import_kwh': float(valid.import_kwh.sum()), 'export_kwh': float(valid.export_kwh.sum())})
    return frame, report
