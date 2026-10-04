"""Verify and total bounded own-feedback batches without double-counting inventory."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.feedback_checkpoint import verified_checkpoint

FLOW_KEYS = ('variable_cost_aud', 'grid_import_kwh', 'grid_export_kwh',
             'battery_throughput_kwh', 'curtailed_pv_kwh', 'clipped_seconds', 'executed_discharge_ws')


def summarize(folders):
    if not folders:
        raise ValueError('require at least one batch')
    summaries, lineage = {}, []
    previous, previous_sha = None, None
    for folder in folders:
        checkpoint, report_sha = verified_checkpoint(folder)
        report = json.loads((folder/'report.json').read_text())
        bundle = json.loads((folder/'bundle.json').read_text())
        resumed = bundle.get('resume_checkpoint')
        if previous is None and resumed is not None:
            raise ValueError('chain must begin with an initial batch')
        if previous is not None:
            if resumed != previous:
                raise ValueError('discontinuous checkpoint chain')
            if bundle['provenance']['resume_parent_report_sha256'] != previous_sha:
                raise ValueError('preceding report lineage differs')
        for arm, values in report['summary'].items():
            if arm not in summaries:
                summaries[arm] = {key: 0. for key in FLOW_KEYS}
                summaries[arm]['initial_soc'] = values['initial_soc']
            for key in FLOW_KEYS:
                summaries[arm][key] += values[key]
            for key in ('final_soc', 'ending_inventory_kwh', 'final_parent_revision', 'final_offset_pct'):
                summaries[arm][key] = values[key]
        lineage.append({'folder': str(folder), 'report_sha256': report_sha,
                        'end': checkpoint['cursor'], 'solve_count': len(report['solves'])})
        previous, previous_sha = checkpoint, report_sha
    challenger = next(arm for arm in summaries if arm != 'baseline')
    cash = summaries['baseline']['variable_cost_aud'] - summaries[challenger]['variable_cost_aud']
    inventory = summaries[challenger]['ending_inventory_kwh'] - summaries['baseline']['ending_inventory_kwh']
    return {'summary': summaries, 'lineage': lineage, 'comparison': {
        'cashflow_delta_aud': cash, 'ending_inventory_delta_kwh': inventory,
        'inventory_value_break_even_aud_per_kwh': -cash/inventory if abs(inventory) > 1e-8 else None},
        'publication_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batches', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.batches)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['comparison']))


if __name__ == '__main__':
    main()
