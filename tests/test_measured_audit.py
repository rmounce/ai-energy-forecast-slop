import json

import numpy as np
import pandas as pd
import pytest

from eval.audit_measured_actuals import load_scores, log_window, measured_report

START = pd.Timestamp('2026-10-01T00:00:00Z')


def test_causal_log_filter_excludes_partial_intervals_duplicates_and_nonfinite(tmp_path):
    rows = []
    for creation, target, prediction in [(0, 30, 1000), (31, 30, 9999),
                                         (0, 60, 1000), (0, 60, 2000), (0, 90, np.inf)]:
        rows.append({'forecast_creation_time': (START+pd.Timedelta(minutes=creation)).isoformat(),
            'forecast_target_time': (START+pd.Timedelta(minutes=target)).isoformat(),
            'model_name': 'load_p65', 'model_version': 'recorded-version',
            'prediction_type': 'simple', 'prediction': prediction,
            'power_pv': 3000, 'actual': 999999})
    path = tmp_path/'log.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    selected, evidence = log_window(path, START, START+pd.Timedelta(hours=2))
    assert len(selected) == 1 and selected.prediction.iloc[0] == 1000
    assert 'actual' not in selected
    assert evidence['partial_interval_rows_excluded'] == 1
    assert evidence['duplicate_rows_excluded'] == 2
    assert evidence['invalid_rows_in_target_window'] == 1
    assert evidence['stable_during_read']


def test_halfhour_load_target_requires_six_complete_measured_bins():
    frame = pd.DataFrame({'load_base_w': [1000]*6+[np.nan]+[1000]*5},
                         index=pd.date_range(START, periods=12, freq='5min'))
    rows = pd.DataFrame({'forecast_target_time': [START, START+pd.Timedelta(minutes=30)],
                         'prediction': [1100, 999999], 'model_name': ['load_p65']*2,
                         'prediction_type': ['simple']*2, 'horizon_hours': [1, 1.5]})
    scores = load_scores(rows, frame)
    assert len(scores) == 1 and scores[0]['rows'] == 1
    assert scores[0]['bias_w'] == 100 and scores[0]['actual_at_or_below_prediction_fraction'] == 1


def test_plan_tracking_uses_opposite_measured_and_solver_battery_signs():
    frame = pd.DataFrame({'battery_charge_w': [1000, -2000],
                          'planned_battery_discharge_w': [-1000, 2000]},
                         index=pd.date_range(START, periods=2, freq='5min'))
    report = measured_report(frame)
    assert report['consistency']['actual_minus_planned_charge_w']['absolute_p95'] == 0


def test_pv_diagnostic_excludes_curtailed_low_power_and_incomplete_samples():
    frame = pd.DataFrame({'pv_dc_w': [1000, 1000, 100, np.nan],
        'pv_proxy_w': [900, 99999, 99999, 99999], 'curtailment_fraction': [0, .1, 0, 0]},
        index=pd.date_range(START, periods=4, freq='5min'))
    report = measured_report(frame)
    assert report['pv_nowcast_diagnostic']['strict_measured_mode_above_500w'] == 1
    assert report['pv_nowcast_diagnostic']['estimate_to_delivered_energy_ratio'] == pytest.approx(.9)
    assert report['pv_nowcast_diagnostic']['error_w']['mean'] == -100
    json.dumps(report, allow_nan=False)


def test_missing_energy_is_unknown_not_zero():
    frame = pd.DataFrame({'grid_import_w_import_kwh': [np.nan]},
                         index=pd.date_range(START, periods=1, freq='5min'))
    report = measured_report(frame)
    assert report['daily_energy'][0]['grid_import_w_import_kwh']['observed_kwh'] is None
    json.dumps(report, allow_nan=False)
