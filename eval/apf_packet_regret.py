"""Offline one-shot APF export timing diagnostic; conditional market value, not site savings."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.audit_amber_forecast_archive import parse_revision, select_asof, utc
from eval.amber_quote_actuals import canonical_rates


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(revision, rates, origin, horizon_hours, *, packet_kwh=.25,
             battery_discharge_efficiency=.99, inverter_efficiency=.95, wear_aud_per_dc_kwh=.04,
             terminal_value_aud_per_dc_kwh=0., max_export_kw=9.98):
    """Compare one forecast-selected future slot or withholding against hindsight.

    Values are incremental relative to keeping the DC packet at the endpoint.
    Hindsight is only over the same complete, fully future slot set. No charging,
    site load, later decisions, counterfactual solar or production controller.
    """
    settings = (horizon_hours, packet_kwh, battery_discharge_efficiency, inverter_efficiency,
                wear_aud_per_dc_kwh, terminal_value_aud_per_dc_kwh, max_export_kw)
    if not all(math.isfinite(v) for v in settings):
        raise ValueError('nonfinite settings')
    if not (0 < horizon_hours <= 14 and packet_kwh > 0 and
            0 < battery_discharge_efficiency <= 1 and 0 < inverter_efficiency <= 1 and wear_aud_per_dc_kwh >= 0 and
            terminal_value_aud_per_dc_kwh >= 0 and max_export_kw > 0):
        raise ValueError('invalid settings')
    origin = utc(origin)
    if utc(revision['receipt']) > origin:
        raise ValueError('future APF revision')
    if origin-utc(revision['receipt']) > pd.Timedelta(minutes=15):
        raise ValueError('stale APF revision')
    first = origin.ceil('5min')
    stop = (origin+pd.Timedelta(hours=horizon_hours)).floor('5min')
    expected = pd.date_range(first, stop, freq='5min', inclusive='left')
    if expected.empty:
        raise ValueError('no fully future slots')
    slots = {}
    for row in revision['rows']:
        end = utc(row['end_time'])
        start = end-pd.Timedelta(minutes=row['duration'])
        for target in pd.date_range(start, end, freq='5min', inclusive='left'):
            if target not in expected:
                continue
            if target in slots:
                raise ValueError('duplicate APF slot')
            slots[target] = {'predicted': -float(row['advanced_price_predicted']),
                             'conservative': -float(row['advanced_price_low'])}
    if set(slots) != set(expected):
        raise ValueError('incomplete APF future horizon')
    if not rates.index.is_unique:
        raise ValueError('duplicate quote target')
    if not set(expected) <= set(rates.index):
        raise ValueError('incomplete confirmed quote horizon')
    realised = rates.loc[expected, 'rate'].astype(float)
    if not all(math.isfinite(v) for v in realised):
        raise ValueError('nonfinite confirmed quote')
    dc_kwh = packet_kwh*battery_discharge_efficiency
    ac_kwh = dc_kwh*inverter_efficiency
    required_kw = ac_kwh/(5/60)
    if required_kw > max_export_kw+1e-12:
        raise ValueError('packet exceeds assumed spare export power')
    def value(price):
        return ac_kwh*price-dc_kwh*wear_aud_per_dc_kwh-packet_kwh*terminal_value_aud_per_dc_kwh
    oracle_target = max(expected, key=lambda t: value(realised.loc[t]))
    oracle_value = max(0., value(realised.loc[oracle_target]))
    candidates = {}
    for field in ('predicted', 'conservative'):
        chosen = max(expected, key=lambda t: value(slots[t][field]))
        abstain = value(slots[chosen][field]) <= 0
        achieved = 0. if abstain else value(realised.loc[chosen])
        candidates[field] = {'selected_target': None if abstain else chosen.isoformat(),
            'forecast_incremental_value_aud': 0. if abstain else value(slots[chosen][field]),
            'realised_incremental_value_aud': achieved,
            'regret_aud': oracle_value-achieved,
            'regret_cents_per_stored_kwh': (oracle_value-achieved)*100/packet_kwh,
            'ending_packet_kwh': packet_kwh if abstain else 0.}
    return {'origin': origin.isoformat(), 'horizon_hours': horizon_hours,
        'first_future_slot': first.isoformat(), 'future_slot_end': stop.isoformat(),
        'five_minute_slots': len(expected), 'packet_stored_dc_kwh': packet_kwh,
        'packet_discharge_dc_kwh': dc_kwh, 'packet_export_ac_kwh': ac_kwh,
        'battery_discharge_efficiency': battery_discharge_efficiency, 'inverter_efficiency': inverter_efficiency, 'required_spare_export_kw': required_kw,
        'oracle_target': oracle_target.isoformat() if oracle_value > 0 else None,
        'oracle_incremental_value_aud': oracle_value, 'candidates': candidates}


def load_inputs(archive, quotes):
    apf_manifest = json.loads((archive/'manifest.json').read_text())
    apf_path = archive/'revisions.json'
    if sha(apf_path) != apf_manifest['raw_revisions_sha256']:
        raise ValueError('changed APF archive')
    quote_manifest = json.loads((quotes/'manifest.json').read_text())
    convention = 'positive export revenue; MQTT state negates raw API per_kwh'
    if not quote_manifest.get('export_complete') or quote_manifest['rate_conventions']['feed_state'] != convention:
        raise ValueError('unsupported quote archive/sign convention')
    for name in ('feed_revisions.json', 'feed_rates.parquet'):
        if sha(quotes/name) != quote_manifest['files'][name]:
            raise ValueError('changed quote archive')
    raw_quotes = json.loads((quotes/'feed_revisions.json').read_text())
    rates, _ = canonical_rates(raw_quotes, quote_manifest['start'], quote_manifest['end'])
    pd.testing.assert_frame_equal(rates, pd.read_parquet(quotes/'feed_rates.parquet'))
    raw = json.loads(apf_path.read_text())['feed_in']
    revisions = [parse_revision(row, 'feed_in') for row in raw['first']+raw['last']]
    return revisions, rates, {'apf_manifest_sha256': sha(archive/'manifest.json'),
        'apf_revisions_sha256': sha(apf_path), 'quote_manifest_sha256': sha(quotes/'manifest.json'),
        'feed_revisions_sha256': sha(quotes/'feed_revisions.json'),
        'feed_rates_sha256': sha(quotes/'feed_rates.parquet')}


def add_control_histories(revisions, paths):
    """Add bounded frozen APF feed receipts; conflicts never silently overwrite."""
    combined = list(revisions)
    sources = []
    for folder in paths:
        manifest_path, history_path = folder/'manifest.json', folder/'history.json'
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('export_complete') or sha(history_path) != manifest['history_sha256']:
            raise ValueError('changed or incomplete control history')
        raw = json.loads(history_path.read_text()).get('mpc_apf_feed_in', [])
        if len(raw) > 200:
            raise ValueError('control APF receipt cap exceeded')
        combined.extend(parse_revision(row, 'feed_in') for row in raw)
        sources.append({'folder': str(folder), 'manifest_sha256': sha(manifest_path),
            'history_sha256': sha(history_path), 'feed_receipts': len(raw)})
    unique = {}
    for row in combined:
        receipt = utc(row['receipt']).isoformat()
        if receipt in unique and unique[receipt]['payload_sha256'] != row['payload_sha256']:
            raise ValueError('conflicting APF receipt across archives')
        unique[receipt] = row
    return list(unique.values()), sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--quotes', type=Path, required=True)
    parser.add_argument('--control-history', type=Path, nargs='*', default=[])
    parser.add_argument('--horizons-hours', type=float, nargs='+', default=[6, 12, 14])
    parser.add_argument('--packet-kwh', type=float, default=.25)
    parser.add_argument('--battery-discharge-efficiency', type=float, default=.99)
    parser.add_argument('--inverter-efficiency', type=float, default=.95)
    parser.add_argument('--wear-aud-per-dc-kwh', type=float, default=.04)
    parser.add_argument('--terminal-value-aud-per-dc-kwh', type=float, default=0.)
    parser.add_argument('--max-export-kw', type=float, default=9.98)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.horizons_hours)) != len(args.horizons_hours):
        parser.error('duplicate horizons')
    if args.output.exists():
        parser.error('output already exists')
    revisions, rates, provenance = load_inputs(args.archive, args.quotes)
    revisions, control_sources = add_control_histories(revisions, args.control_history)
    provenance['control_histories'] = control_sources
    results, exclusions = [], []
    for origin in sorted({row['receipt'] for row in revisions}):
        revision = select_asof(revisions, origin)
        for horizon in args.horizons_hours:
            try:
                result = evaluate(revision, rates, origin, horizon, packet_kwh=args.packet_kwh,
                    battery_discharge_efficiency=args.battery_discharge_efficiency,
                    inverter_efficiency=args.inverter_efficiency,
                    wear_aud_per_dc_kwh=args.wear_aud_per_dc_kwh,
                    terminal_value_aud_per_dc_kwh=args.terminal_value_aud_per_dc_kwh,
                    max_export_kw=args.max_export_kw)
                result['apf_payload_sha256'] = revision['payload_sha256']
                results.append(result)
            except ValueError as exc:
                exclusions.append({'origin': origin, 'horizon_hours': horizon, 'reason': str(exc)})
    report = {'scope': 'conditional_one_shot_export_packet_regret_not_site_savings',
        'publication_authorized': False, 'provenance': provenance, 'code_sha256': sha(Path(__file__)),
        'settings': {k: v for k, v in vars(args).items() if k not in ('archive', 'quotes', 'output', 'control_history')},
        'results': results, 'exclusions': exclusions,
        'eligible_origins': len({row['origin'] for row in results}),
        'limitations': ['first bounded revisions cluster in one window; final revisions may lack future quotes',
            'overlapping origins and horizons are not independent or additive savings',
            'expanded five-minute APF slots may interpolate billing intervals, not independent market samples',
            'CurrentInterval non-estimated archived quotes are not invoice-reconciled settlement',
            'unadjusted raw feed quotes exclude local allowance adjustment',
            'assumes a stored packet and spare inverter/grid export headroom; no site load/PV/controller',
            'withholding retains terminal energy value; exporting incurs explicit DC wear and efficiency',
            'hindsight timing only bounds this restricted packet choice, not full-system forecast headroom']}
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'eligible_origins': report['eligible_origins'], 'cases': len(results),
                      'excluded': len(exclusions), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
