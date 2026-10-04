"""Freeze causal p65 residual corrections into admitted historical DH events."""
from copy import deepcopy

import pandas as pd

from energy_pipeline.payloads import Inputs, build_dh_payload
from energy_pipeline.solver_replay import digest
from eval.compare_load_solver_sensitivity import calibrated_handoff

LOAD = 'sensor.ai_load_forecast_high'


def prepare_calibrated_events(bundle, rows, settings):
    """Match the received vector; fit at its creation, never at a later DH solve."""
    result = deepcopy(bundle)
    if result.get('experiment','terminal_policy') != 'load_calibration':
        raise ValueError('load preparation requires load_calibration experiment')
    for event in result['dh_events']:
        if not event['ready']: continue
        at,receipt = pd.Timestamp(event['origin']),pd.Timestamp(event['input_receipts']['load_forecast'])
        if receipt > at: raise ValueError('future load receipt')
        # Logger labels follow sensor publication by ~22ms in these archives.
        # Exact-vector proof plus bounded clock tolerance; never a future DH vintage.
        ceiling = min(at,receipt+pd.Timedelta(seconds=2))
        available = rows.loc[rows.forecast_creation_time <= ceiling]
        states = deepcopy(event['states'])
        states.update(deepcopy(result['initial_parent']))
        states['sensor.sigen_plant_battery_state_of_charge_derived'] = {'state':str(result['initial_soc']*100)}
        record = {'captured_at':at.isoformat(),'publication_id':result['source_publication_id'],
            'input_snapshot':{'states':states},'input_revision':digest(states),
            'lineage':{'load_receipt':receipt.isoformat()},
            'payloads':{'dh':build_dh_payload(Inputs(states,at.to_pydatetime()))}}
        corrected,provenance = calibrated_handoff(record,available,settings)
        base,changed = states[LOAD]['attributes']['forecasts'],corrected['input_snapshot']['states'][LOAD]['attributes']['forecasts']
        if [row['timestamp'] for row in base] != [row['timestamp'] for row in changed]:
            raise ValueError('calibration changed target timestamps')
        event['calibrated_load_rows'] = deepcopy(changed)
        event['load_calibration'] = {key:provenance[key] for key in
            ('forecast_creation','model_version','prediction_type','matching_vector_targets','settings','corrections')}
        event['load_calibration'].update(load_receipt=receipt.isoformat(),
            logged_creation_minus_receipt_seconds=(pd.Timestamp(provenance['forecast_creation'])-receipt).total_seconds(),
            lineage_clock_tolerance_seconds=2,
            base_load_sha256=digest(base),calibrated_load_sha256=digest(changed),
            next14h_base_energy_delta_kwh=sum(float(b['power_load'])-float(a['power_load'])
                for a,b in zip(base[:28],changed[:28]))*.5/1000,
            initial_parent_seed='common archived parent; calibration starts at first admitted DH refresh')
    return result
