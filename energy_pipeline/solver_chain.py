"""Historical DH result → recorded MPC inputs; no live admission or writes."""
from copy import deepcopy

import numpy as np

from energy_pipeline.payloads import Inputs, build_dh_payload, build_mpc_payload, dh_soc, target_soc_offset
from energy_pipeline.solver_replay import digest, validate_result


DH_CHANNELS = (
    ('P_Load', 'sensor.dh_p_load_forecast', 'forecasts', 1),
    ('P_PV', 'sensor.dh_p_pv_forecast', 'forecasts', 1),
    ('SOC_opt', 'sensor.dh_soc_batt_forecast', 'battery_scheduled_soc', 100),
)


def project_dh_entities(frame):
    """Consumed HA fields only; mirror EMHASS get_attr_data_dict's two decimals.

    Row dates remain interval STARTS, including end-of-interval SOC_opt values.
    No publishing, entity metadata, scheduler or current-state selection implied.
    """
    projected = {}
    for column, entity, attribute, scale in DH_CHANNELS:
        values = frame[column].to_numpy(dtype=float)*scale
        key = entity.removeprefix('sensor.')
        projected[entity] = {
            'state': f'{np.round(values[0], 2):.2f}',
            'attributes': {attribute: [
                {'date': at.isoformat(), key: str(np.round(value, 2))}
                for at, value in zip(frame.index, values)]}}
    return projected


def build_chained_handoff(record, dh_artifact):
    """Rehearse one ordered cycle at the capture clock, without solver latency.

    Requires a matching validated DH solve and its installed formatter evidence.
    Original Amber/live telemetry/HWC remain frozen. No new freshness is claimed.
    """
    if (dh_artifact.get('mode') != 'historical_solver_replay'
            or dh_artifact.get('publication_authorized') is not False):
        raise ValueError('require an unauthorised historical DH artifact')
    request, result = dh_artifact['request'], dh_artifact['result']
    content = {key: value for key, value in request.items() if key != 'request_id'}
    if request['request_id'] != digest(content):
        raise ValueError('DH request identity mismatch')
    if (request['kind'] != 'dh' or request['handoff_revision'] != digest(record)
            or request['publication_id'] != record['publication_id']):
        raise ValueError('DH result belongs to another historical handoff')
    frame = validate_result(request, result)
    ready = record['readiness']['mpc']
    if not ready['coverage_ready'] or ready['reasons']:
        raise ValueError('original MPC coverage is not ready')
    snapshot = deepcopy(record['input_snapshot'])
    if snapshot['captured_at'] != record['captured_at']:
        raise ValueError('snapshot capture differs from handoff')
    inputs = Inputs(snapshot['states'], snapshot['captured_at'], snapshot['timezone'])
    if (build_dh_payload(inputs) != record['payloads']['dh']
            or build_mpc_payload(inputs) != record['payloads']['mpc']):
        raise ValueError('current payload policy differs from recorded handoff')
    projected = project_dh_entities(frame)
    if result.get('projected_dh_entities') != projected:
        raise ValueError('DH projection differs from installed formatter evidence')
    policy = dh_soc(inputs)
    if round(policy.soc_init_pct/100, 4) != request['payload']['soc_init']:
        raise ValueError('DH anchor differs from solved input')
    states = snapshot['states']
    states.update(projected)
    # HA writes the helper from pre-payload percent precision, not the rounded
    # fractional solver input. Preserve that distinction in this ordered cycle.
    states['input_number.dh_last_soc_init']['state'] = str(policy.soc_init_pct)
    if policy.should_reground:
        states['input_text.dh_last_reground_block']['state'] = policy.reground_block
    chained_inputs = Inputs(states, snapshot['captured_at'], snapshot['timezone'])
    chained = deepcopy(record)
    chained['mode'] = 'historical_chained_handoff'
    chained['input_snapshot'] = snapshot
    chained['input_revision'] = digest(states)
    chained['payloads']['mpc'] = build_mpc_payload(chained_inputs)
    result_revision = digest(result)
    chained['lineage']['mpc_dh_parent'] = result_revision
    chained['lineage']['mpc_dh_price_parent'] = record['publication_id']
    chained['historical_chain'] = {
        'original_handoff_revision': digest(record),
        'dh_request_id': request['request_id'], 'dh_result_revision': result_revision,
        'dh_price_parent': record['publication_id'],
        'timing_assumption': 'ordered_dh_then_mpc_at_frozen_capture_clock',
        'dh_last_soc_init_pct': policy.soc_init_pct,
        'reground_block_updated': policy.should_reground,
        # Offset update affects a future DH run, not this MPC. Record preview
        # separately; do not invent an observed helper event/order.
        'next_offset_preview_pct': target_soc_offset(chained_inputs),
        'publication_authorized': False,
    }
    for value in chained['readiness'].values():
        value['solve_authorized'] = False
    return chained
