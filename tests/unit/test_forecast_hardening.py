import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import forecast


def screening_descriptor(tmp_path):
    issue = pd.Timestamp("2026-08-01T00:00:00Z")
    rows = []
    for horizon in (1.0, 20.0, 36.0, 60.0):
        rows.append({
            "forecast_issue_time": issue.isoformat(),
            "forecast_target_time": (issue + pd.Timedelta(hours=horizon)).isoformat(),
            "actual": 100.0,
            "candidate_p30": 99.0,
            "candidate_p50": 100.0,
            "candidate_p70": 101.0,
            "incumbent_p30": 99.0,
            "incumbent_p50": 101.0,
            "incumbent_p70": 102.0,
        })
    rows_path = tmp_path / "rows.csv"
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    descriptor = {
        "schema_version": 1,
        "family": "price",
        "units": "$/MWh",
        "row_count": len(rows),
        "provenance": {
            "command": "fixed-eval-command",
            "rows_file": rows_path.name,
            "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        },
    }
    descriptor_path = tmp_path / "screening.json"
    descriptor_path.write_text(json.dumps(descriptor))
    return descriptor_path


def test_screen_bundle_installs_row_derived_eligible_report(tmp_path, monkeypatch):
    descriptor_path = screening_descriptor(tmp_path)
    store = Mock()
    monkeypatch.setattr(forecast, "validate_bundle_command", lambda family, bundle: {"smoke_seconds": 1.25})
    monkeypatch.setattr(forecast, "_production_bundle_store", lambda: store)

    report = forecast.screen_bundle_command("price", "candidate-one", descriptor_path)

    assert report["eligible_for_manual_promotion"] is True
    assert report["metrics"]["row_count"] == 4
    assert report["metrics"]["comparable"] is True
    store.replace_candidate_report.assert_called_once_with("price", "candidate-one", report)


class MigrationStore:
    def __init__(self, events):
        self.events = events

    def migrate_root_artifacts(self, family, bundle_id, paths, *, manifest):
        self.events.append(("migrate", family, bundle_id, manifest))

    def promote(self, family, bundle_id, *, report):
        self.events.append(("promote", family, bundle_id, report))


def migration_config(tmp_path):
    paths = {}
    quantiles = {"load": {"alpha": 0.5}, "load_p65": {"alpha": 0.65}, "load_p75": {"alpha": 0.75}}
    for name in quantiles:
        for suffix in ("model_file", "params_file", "importance_file"):
            path = tmp_path / f"{name}_{suffix}"
            path.write_text("artifact")
            paths[f"{name}_{suffix}"] = str(path)
    return {
        "paths": paths,
        "models": {"load": {
            "quantile_models": quantiles,
            "feature_cols": ["power_pv"],
            "lgbm_params": {"n_estimators": 1},
        }},
    }


def test_migration_validates_before_promotion(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(forecast, "CONFIG", migration_config(tmp_path))
    monkeypatch.setattr(forecast, "_production_bundle_store", lambda: MigrationStore(events))
    monkeypatch.setattr(
        forecast,
        "validate_bundle_command",
        lambda family, bundle: events.append(("validate", family, bundle)),
    )

    bundle_id = forecast.migrate_root_family("load")

    assert [event[0] for event in events] == ["migrate", "validate", "promote"]
    assert events[1][2] == bundle_id


def test_failed_migration_validation_never_promotes(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(forecast, "CONFIG", migration_config(tmp_path))
    monkeypatch.setattr(forecast, "_production_bundle_store", lambda: MigrationStore(events))

    def fail_validation(family, bundle):
        events.append(("validate", family, bundle))
        raise RuntimeError("smoke failed")

    monkeypatch.setattr(forecast, "validate_bundle_command", fail_validation)
    with pytest.raises(RuntimeError, match="smoke failed"):
        forecast.migrate_root_family("load")
    assert [event[0] for event in events] == ["migrate", "validate"]


def test_prediction_resolves_one_bundle_even_if_pointer_would_change(tmp_path, monkeypatch):
    class ChangingStore:
        def __init__(self):
            self.calls = 0

        def resolve_active(self, family):
            self.calls += 1
            return ("one", tmp_path / "one") if self.calls == 1 else ("two", tmp_path / "two")

        def active_pointer(self, family):
            return tmp_path / "active.json"

    store = ChangingStore()
    monkeypatch.setattr(forecast, "CONFIG", {
        "paths": {},
        "models": {"price": {"quantile_models": {
            "price_p30": {}, "price": {}, "price_p70": {},
        }}},
    })
    monkeypatch.setattr(forecast, "_production_bundle_store", lambda: store)
    monkeypatch.setattr(forecast, "_PREDICTION_BUNDLE_PATHS", None)
    monkeypatch.setattr(forecast, "_PREDICTION_BUNDLE_IDS", None)

    first = forecast._prediction_model_paths("price")
    second = forecast._prediction_model_paths("price")

    assert store.calls == 1
    assert first == second
    assert all(path["model"].parent == tmp_path / "one" for path in first.values())


class FakeInfluxClient:
    def __init__(self, **kwargs):
        pass

    def close(self):
        pass


def prediction_config(tmp_path):
    return {
        "influxdb": {},
        "prediction_history_days": 1,
        "models": {"load": {
            "feature_cols": [],
            "target_column": "power_load",
            "primary_model_key": "load",
            "quantile_models": {
                "load": {"alpha": 0.5},
                "load_p65": {"alpha": 0.65},
                "load_p75": {"alpha": 0.75},
            },
        }},
        "paths": {
            "prediction_output_file": str(tmp_path / "predictions.json"),
            "load_forecast_log_file": str(tmp_path / "load_log.csv"),
        },
        "home_assistant": {"publish_entities": {}},
    }


def patch_prediction_boundaries(monkeypatch, tmp_path):
    monkeypatch.setattr(forecast, "CONFIG", prediction_config(tmp_path))
    monkeypatch.setattr(forecast, "InfluxDBClient", FakeInfluxClient)
    monkeypatch.setattr(forecast, "get_historical_data", lambda *args: pd.DataFrame({"power_load": [1.0]}))
    monkeypatch.setattr(forecast, "get_solcast_forecast", lambda: pd.DataFrame())
    monkeypatch.setattr(forecast, "get_weather_forecast", lambda: pd.DataFrame())
    monkeypatch.setattr(forecast, "get_aemo_forecast", lambda: pd.DataFrame())
    blank = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    monkeypatch.setattr(
        forecast,
        "_prepare_prediction_covariates",
        lambda *args: (blank.copy(), blank.copy(), blank.copy()),
    )


def valid_load_family():
    now = pd.Timestamp.now(tz="UTC").floor("30min")
    index = pd.date_range(now, periods=144, freq="30min")
    return {
        "load": pd.DataFrame({"load": [1.0] * 144}, index=index),
        "load_p65": pd.DataFrame({"load_p65": [2.0] * 144}, index=index),
        "load_p75": pd.DataFrame({"load_p75": [3.0] * 144}, index=index),
    }


def test_generation_failure_preserves_output_and_forecast_log(tmp_path, monkeypatch, caplog):
    patch_prediction_boundaries(monkeypatch, tmp_path)
    output = tmp_path / "predictions.json"
    output.write_text("sentinel")
    log_forecast = Mock()
    monkeypatch.setattr(forecast, "log_forecast_data", log_forecast)
    monkeypatch.setattr(forecast, "_execute_single_prediction", lambda **kwargs: ({}, "simple"))

    with caplog.at_level(logging.ERROR), pytest.raises(forecast.ForecastContractError):
        forecast.run_predictions(["load"], False, False, False)

    assert output.read_text() == "sentinel"
    log_forecast.assert_not_called()
    assert '"failure_stage": "generation_or_validation"' in caplog.text


def test_covariate_publication_failure_stops_before_forecast_publish_and_output(tmp_path, monkeypatch, caplog):
    patch_prediction_boundaries(monkeypatch, tmp_path)
    output = tmp_path / "predictions.json"
    output.write_text("sentinel")
    monkeypatch.setattr(
        forecast,
        "_execute_single_prediction",
        lambda **kwargs: (valid_load_family(), "simple"),
    )
    monkeypatch.setattr(
        forecast,
        "publish_adjusted_covariates_to_hass",
        Mock(side_effect=RuntimeError("covariate POST failed")),
    )
    publish_forecast = Mock()
    monkeypatch.setattr(forecast, "_publish_lgbm_model_to_hass", publish_forecast)

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="covariate POST failed"):
        forecast.run_predictions(["load"], True, False, True)

    assert output.read_text() == "sentinel"
    publish_forecast.assert_not_called()
    assert '"failure_stage": "covariate_publication"' in caplog.text


def test_forecast_publication_failure_preserves_local_output(tmp_path, monkeypatch, caplog):
    patch_prediction_boundaries(monkeypatch, tmp_path)
    output = tmp_path / "predictions.json"
    output.write_text("sentinel")
    monkeypatch.setattr(
        forecast,
        "_execute_single_prediction",
        lambda **kwargs: (valid_load_family(), "simple"),
    )
    monkeypatch.setattr(
        forecast,
        "_publish_lgbm_model_to_hass",
        Mock(side_effect=RuntimeError("forecast POST failed")),
    )

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="forecast POST failed"):
        forecast.run_predictions(["load"], True, False, False)

    assert output.read_text() == "sentinel"
    assert '"failure_stage": "publication"' in caplog.text


def test_calculation_only_has_no_output_publication_or_forecast_log(tmp_path, monkeypatch):
    patch_prediction_boundaries(monkeypatch, tmp_path)
    output = tmp_path / "predictions.json"
    output.write_text("sentinel")
    monkeypatch.setattr(forecast, "_execute_single_prediction",
                        lambda **kwargs: (valid_load_family(), "simple"))
    names = ("_publish_lgbm_model_to_hass", "publish_adjusted_covariates_to_hass",
             "log_forecast_data", "_persist_amber_spot_5min_forecasts")
    mocks = [Mock() for _ in names]
    for name, mock in zip(names, mocks):
        monkeypatch.setattr(forecast, name, mock)
    outcome = forecast.run_predictions(["load"], False, False, False, calculation_only=True)
    assert outcome.publication_result == "not_requested"
    assert outcome.point_counts == {"load": 144, "load_p65": 144, "load_p75": 144}
    assert output.read_text() == "sentinel"
    for mock in mocks:
        mock.assert_not_called()


def test_prediction_snapshot_is_frozen_and_missing_inputs_do_not_fetch_live(monkeypatch):
    states = {"sensor.apf": {"attributes": {"Forecasts": [1]}}}
    live = Mock()
    monkeypatch.setattr(forecast, "call_ha_api", live)
    with forecast.prediction_resources(entity_states=states, model_cache=object()):
        states["sensor.apf"]["attributes"]["Forecasts"].append(2)
        first = forecast.get_entity_state("sensor.apf")
        first["attributes"]["Forecasts"].append(3)
        assert forecast.get_entity_state("sensor.apf")["attributes"]["Forecasts"] == [1]
        assert forecast.get_entity_state("sensor.missing") is None
        live.assert_not_called()
    forecast.get_entity_state("sensor.live")
    live.assert_called_once()


def test_calculation_only_rejects_publish_flags():
    with pytest.raises(ValueError, match="cannot publish"):
        forecast.run_predictions(["price"], True, True, False, calculation_only=True)


def test_resident_worker_uses_one_ha_snapshot_for_all_quantiles(monkeypatch):
    from energy_pipeline import price_worker
    from types import SimpleNamespace
    config = {'paths': {'tariff_file': 'missing-tariff-test.json'},
              'home_assistant': {'amber_billing_entity': 'sensor.apf', 'solcast_entities': ['sensor.pv'],
                                 'solcast_last_polled_entity': 'sensor.pv_polled'}}
    monkeypatch.setattr(price_worker, 'load_config', lambda: config)
    monkeypatch.setattr('tariff_utils.load_tariff_profile', lambda *args: ({}, {}, 1))
    monkeypatch.setattr(forecast, 'CONFIG', config)
    monkeypatch.setattr(forecast, 'GENERAL_TARIFF_MAP', {})
    monkeypatch.setattr(forecast, 'FEED_IN_TARIFF_MAP', {})
    monkeypatch.setattr(forecast, 'NETWORK_LOSS_FACTOR', 1)
    rows = [{'entity_id': 'sensor.apf', 'attributes': {'Forecasts': [1]}},
            {'entity_id': 'sensor.pv', 'attributes': {}},
            {'entity_id': 'sensor.unrelated', 'attributes': {}},
            {'entity_id': 'sensor.pv_polled', 'state': '2026-10-02T00:00:00+00:00'}]
    api = Mock(return_value=rows)
    monkeypatch.setattr(forecast, 'call_ha_api', api)
    def predict(*args, **kwargs):
        assert kwargs['calculation_only'] is True
        for _ in range(3):
            assert forecast.get_entity_state('sensor.apf')['attributes']['Forecasts'] == [1]
            assert forecast.get_entity_state('sensor.unrelated') is None
        return SimpleNamespace(run_id='r', model_bundle_id='b', point_counts={'price': 144},
                               publication_result='not_requested')
    monkeypatch.setattr(forecast, 'run_predictions', predict)
    from energy_pipeline.source_cache import SourceSnapshot
    now = pd.Timestamp.now(tz='UTC')
    frame = pd.DataFrame({'power_pv': [0.] * 145}, index=pd.date_range(now.floor('30min'), periods=145, freq='30min'))
    source_cache = SimpleNamespace(snapshot=lambda at: {
        name: SourceSnapshot(frame, now.to_pydatetime(), name) for name in ('aemo', 'weather', 'history')})
    monkeypatch.setattr(forecast, 'get_solcast_forecast', lambda: frame)
    result = price_worker.PriceWorker(source_cache).predict()
    api.assert_called_once_with('GET', 'states')
    assert len(result.parent_revision) == 64
    assert result.outcome.publication_result == 'not_requested'
    assert result.input_revisions['apf'] == result.parent_revision
    assert result.input_revisions['source:aemo'] == 'aemo'
    assert result.input_revisions['ha:sensor.pv'] == price_worker.content_revision(rows[1])
    assert result.input_revisions['tariff'] == price_worker.content_revision([{}, {}, 1])
    assert result.input_revisions['ha:sensor.pv_polled'] == price_worker.content_revision(rows[3])
    assert result.source_freshness['solcast'][0]['timestamp'] == rows[3]['state']
    assert result.source_freshness['apf'][0]['timestamp'] is None


def test_input_revision_ignores_mapping_order_and_tracks_values():
    from energy_pipeline.price_worker import content_revision
    assert content_revision({'a': 1, 'b': 2}) == content_revision({'b': 2, 'a': 1})
    assert content_revision({'a': 1}) != content_revision({'a': 2})


def test_cached_input_path_performs_no_future_or_history_acquisition(tmp_path, monkeypatch):
    from production_contract import PredictionInputs
    patch_prediction_boundaries(monkeypatch, tmp_path)
    family = valid_load_family()
    frame = next(iter(family.values()))
    snapshot = PredictionInputs({'solcast': frame, 'weather': frame, 'aemo': frame}, frame, frame.index[0])
    for name in ('get_solcast_forecast', 'get_weather_forecast', 'get_aemo_forecast', 'get_historical_data', 'InfluxDBClient'):
        monkeypatch.setattr(forecast, name, Mock(side_effect=AssertionError('unexpected acquisition')))
    monkeypatch.setattr(forecast, '_execute_single_prediction', lambda **kwargs: (family, 'simple'))
    outcome = forecast.run_predictions(['load'], False, False, False, calculation_only=True, input_snapshot=snapshot)
    assert outcome.point_counts['load'] == 144
    with pytest.raises(ValueError, match='only supported'):
        forecast.run_predictions(['load'], False, False, False, input_snapshot=snapshot)
