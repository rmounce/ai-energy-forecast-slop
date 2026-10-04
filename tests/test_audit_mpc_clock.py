import json

import pandas as pd
import pytest

from eval.audit_mpc_clock import audit, candidate, capacity_evidence, preceding_timer_evidence


def evidence():
    targets = pd.date_range('2026-09-30T22:40Z', periods=168, freq='5min')
    curve = lambda key, values: json.dumps([{'date': target.isoformat(), key: str(value)}
                                           for target, value in zip(targets, values)])
    current = {'time': '2026-09-30T22:41:27.337814Z', 'value': 194.05,
        'battery_scheduled_power_str': curve('mpc_p_batt_forecast', [194.05]*168)}
    history = {'mpc_anchor': [
        {'time': '2026-09-30T22:'+minute+':25.07Z', 'value': 71.75} for minute in ('36', '37', '38', '39')]+[
        {'time': '2026-09-30T22:40:18.75Z', 'value': 71.75}],
        'mpc_battery': [{'time': '2026-09-30T22:40:21.21Z', 'value': 162.47,
            'battery_scheduled_power_str': curve('mpc_p_batt_forecast', [162.47]*168)}, current],
        'mpc_soc': [{'time': '2026-09-30T22:41:27.36Z',
            'battery_scheduled_soc_str': curve('mpc_soc_batt_forecast', [71.71]*167+[71.75])}],
        'mode': [{'time': '2026-09-30T22:41:20Z', 'state': 'measured'}],
        'dh_anchor': [{'time': '2026-09-30T22:40:10Z', 'value': 71.75}],
        'dh_soc': [{'time': '2026-09-30T22:40:12Z', 'battery_scheduled_soc_str': curve('dh_soc_batt_forecast', [71.75]*168)}],
        'dh_load': [{'time': '2026-09-30T22:40:12Z', 'forecasts_str': curve('dh_p_load_forecast', [267]*168)}],
        'dh_pv': [{'time': '2026-09-30T22:40:12Z', 'forecasts_str': curve('dh_p_pv_forecast', [87]*168)}],
        'hwc': [{'time': '2026-09-30T22:40:12Z', 'deferrables_schedule_json_str': curve('hwc_power_plan', [0]*168)}],
        'rated_capacity': [{'time': '2026-09-30T21:00Z', 'value': 40.3, 'unit_of_measurement_str': 'kWh'}],
        'battery_health': [{'time': '2026-09-30T21:00Z', 'value': 100., 'unit_of_measurement_str': '%'}]}
    for name, value in [('soc', 71.75), ('load', 267.), ('pv', 227.247), ('loss', 180.)]:
        history[name] = [{'time': '2026-09-30T22:41:20Z', 'value': value}]
    controls = {'mpc_load': [{'time': '2026-09-30T22:41:27.31Z', 'forecasts_str': curve('mpc_p_load_forecast', [267]*168)}],
                'mpc_pv': [{'time': '2026-09-30T22:41:27.33Z', 'forecasts_str': curve('mpc_p_pv_forecast', [87]*168)}]}
    return history, controls, current, {'battery_discharge_efficiency': .99, 'battery_charge_efficiency': .99}


def test_unchanged_helper_supports_both_clock_sensitivities_but_never_admits_origin():
    history, controls, _, plant = evidence()
    report = audit(history, controls, {}, plant, '2026-09-30T22:41Z', '2026-09-30T22:42Z')
    row = report['rows'][0]
    assert row['status'] == 'missing_fresh_helper_clock'
    assert all(candidate['supported'] for candidate in row['candidates'])
    assert all(candidate['expected']['load_first_two_w'] == 267 for candidate in row['candidates'])
    assert all(candidate['expected']['pv_first_two_w'] == 87 for candidate in row['candidates'])
    assert all(candidate['expected']['soc_init_pct'] == 71.75 for candidate in row['candidates'])
    assert report['summary']['missing_clocks_with_both_supported_candidates'] == 1
    assert row['candidate_origins_admitted'] is False
    assert report['replay_admission_authorized'] is False


def test_known_helper_clock_is_preserved_exactly_without_substitution():
    history, controls, _, plant = evidence()
    history['mpc_anchor'].append({'time': '2026-09-30T22:41:25.089123Z', 'value': 71.75})
    report = audit(history, controls, {}, plant, '2026-09-30T22:41Z', '2026-09-30T22:42Z')
    row = report['rows'][0]
    assert row['observed_origin'] == '2026-09-30T22:41:25.089123Z'
    assert 'candidates' not in row


def test_future_telemetry_never_enters_candidate_inputs():
    history, controls, publication, plant = evidence()
    before = candidate(history, controls, {}, plant, publication, '2026-09-30T22:41:25Z')
    history['load'].append({'time': '2026-09-30T22:41:26Z', 'value': 999})
    history['soc'].append({'time': '2026-09-30T22:41:26Z', 'value': 99})
    after = candidate(history, controls, {}, plant, publication, '2026-09-30T22:41:25Z')
    assert after == before


def test_clock_sensitivity_reports_a_source_change_between_candidates():
    history, controls, _, plant = evidence()
    history['load'].append({'time': '2026-09-30T22:41:25.05Z', 'value': 999})
    report = audit(history, controls, {}, plant, '2026-09-30T22:41Z', '2026-09-30T22:42Z')
    first, second = report['rows'][0]['candidates']
    assert first['supported'] is True
    assert second['checks']['first_two_load_slots'] is False
    assert report['summary']['missing_clocks_with_both_supported_candidates'] == 0


@pytest.mark.parametrize('mutation', ['endpoint', 'helper', 'stale', 'missing_pair', 'insufficient_clocks'])
def test_failed_consistency_or_source_coverage_is_explicit(mutation):
    history, controls, publication, plant = evidence()
    if mutation == 'endpoint':
        rows = json.loads(history['mpc_soc'][0]['battery_scheduled_soc_str'])
        rows[0]['mpc_soc_batt_forecast'] = '70'
        history['mpc_soc'][0]['battery_scheduled_soc_str'] = json.dumps(rows)
    elif mutation == 'helper': history['mpc_anchor'][-1]['value'] = 70.
    elif mutation == 'stale': history['load'][0]['time'] = '2026-09-30T22:38Z'
    elif mutation == 'missing_pair': controls['mpc_load'] = []
    else: history['mpc_anchor'] = history['mpc_anchor'][-2:]
    result = candidate(history, controls, {}, plant, publication, '2026-09-30T22:41:25Z')
    assert result['supported'] is False
    assert 'rejection' in result or not all(result['checks'].values())


def test_five_minute_price_trigger_cannot_receive_timer_origin():
    history, controls, publication, plant = evidence()
    publication['time'] = '2026-09-30T22:45:27Z'
    result = candidate(history, controls, {}, plant, publication, '2026-09-30T22:45:25Z')
    assert result['supported'] is False
    assert 'non-five-minute' in result['rejection']


def test_clock_evidence_excludes_future_receipts():
    history, _, _, _ = evidence()
    before = preceding_timer_evidence(history, '2026-09-30T22:41:25Z')
    history['mpc_anchor'].append({'time': '2026-09-30T22:42:25.01Z', 'value': 71.75})
    assert preceding_timer_evidence(history, '2026-09-30T22:41:25Z') == before


def test_static_capacity_fallback_requires_older_capture_metadata():
    history, _, _, _ = evidence()
    history['rated_capacity'] = []
    captured = {'sensor.sigen_plant_rated_energy_capacity': {'state': '40.3', 'last_updated': '2026-09-18T00:00Z',
        'attributes': {'unit_of_measurement': 'kWh'}}}
    capacity, refs, inherited = capacity_evidence(history, captured, '2026-09-30T22:41:25Z')
    assert capacity == 40300
    assert 'not_independent_historical' in inherited[0]['evidence']
    captured['sensor.sigen_plant_rated_energy_capacity']['last_updated'] = '2026-10-03T00:00Z'
    with pytest.raises(ValueError, match='causal capacity'):
        capacity_evidence(history, captured, '2026-09-30T22:41:25Z')
