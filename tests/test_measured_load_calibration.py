import numpy as np
import pandas as pd
import pytest

from eval.calibrate_measured_load import calibrate, scores

START = pd.Timestamp('2026-09-27T00:00:00Z')


def fixture():
    targets = pd.date_range(START, periods=12, freq='30min')
    frame = pd.DataFrame({'load_base_w': 100.}, index=pd.date_range(START, periods=72, freq='5min'))
    rows = pd.DataFrame({'forecast_target_time': targets, 'forecast_creation_time': targets,
        'horizon_hours': 0., 'prediction': 200., 'model_name': 'load_p65',
        'model_version': 'v1', 'prediction_type': 'simple'})
    return rows, frame


def test_completed_interval_plus_lag_and_cold_start():
    rows, frame = fixture()
    result = calibrate(rows, frame, min_targets=1)
    assert result.training_targets.tolist()[:4] == [0, 0, 1, 2]
    assert result.calibrated_w.tolist()[:4] == [200, 200, 100, 100]
    assert not result.calibration_ready.iloc[0]


def test_future_actuals_cannot_change_earlier_corrections():
    rows, frame = fixture()
    first = calibrate(rows, frame, min_targets=1)
    changed = frame.copy()
    changed.loc[changed.index >= START+pd.Timedelta(hours=1), 'load_base_w'] = 9000
    second = calibrate(rows, changed, min_targets=1)
    pd.testing.assert_series_equal(first.correction_w.iloc[:4], second.correction_w.iloc[:4])
    assert first.correction_w.iloc[8] != second.correction_w.iloc[8]


def test_training_targets_not_overweighted_by_repeated_vintages_and_version_isolation():
    rows, frame = fixture()
    extra = rows.iloc[[0]].copy()
    extra.forecast_creation_time -= pd.Timedelta(minutes=1)
    extra.horizon_hours = 1/60
    extra.prediction = 9999
    rows = pd.concat([rows, extra], ignore_index=True)
    rows.loc[3, 'model_version'] = 'v2'
    result = calibrate(rows, frame, min_targets=1)
    assert result.training_targets.iloc[2] == 1
    assert result.correction_w.iloc[2] == -100
    assert result.training_targets.iloc[3] == 0


def test_missing_halfhour_target_and_noncausal_input():
    rows, frame = fixture()
    frame.iloc[0, 0] = np.nan
    result = calibrate(rows, frame, min_targets=1)
    assert len(result) == 11
    assert result.training_targets.iloc[1] == 0
    rows.loc[2, 'forecast_creation_time'] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match='noncausal'):
        calibrate(rows, frame, min_targets=1)


def test_equal_target_score_weights_and_paired_eligible_rows():
    rows, frame = fixture()
    result = calibrate(rows, frame, min_targets=1)
    extra = result.iloc[[-1]].copy()
    extra.prediction = 1200
    scored = scores(pd.concat([result, extra], ignore_index=True))[0]
    assert scored['rows'] == 11 and scored['targets'] == 10
    assert scored['baseline']['mae_w'] == 150
    assert scored['calibrated']['mae_w'] == 0


def test_rolling_history_expires_old_targets():
    rows, frame = fixture()
    result = calibrate(rows, frame, min_targets=1, history_days=1/24)
    assert result.training_targets.iloc[-1] == 2


def test_correction_fits_requested_quantile_not_mean_residual():
    rows, frame = fixture()
    for index, value in enumerate([100, 200, 300]):
        frame.iloc[index*6:(index+1)*6, 0] = value
    result = calibrate(rows, frame, min_targets=3)
    assert result.correction_w.iloc[4] == pytest.approx(30)
