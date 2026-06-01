import services.hwc_daemon as hd


def _config():
    return {
        "home_assistant": {
            "weather_entity": "weather.woodville_west_hourly",
        },
        "hwc": {
            "tank_temp_entity": "sensor.aquatech_current_temperature_local",
            "import_price_entity": "sensor.ai_dh_import_price_forecast",
            "predicted_temp_entity": "sensor.predicted_temp",
            "power_plan_entity": "sensor.power_plan",
            "publish_prefix": "hwc_",
            "actuation": {
                "water_heater_entity": "water_heater.aquatech",
                "compressor_entity": "binary_sensor.aquatech_compressor",
            },
            "daemon": {
                "tank_temp_replan_delta_c": 0.3,
            },
        },
    }


def _state(value):
    return {"state": str(value)}


def test_watched_entities_include_inputs_equipment_and_published_plan():
    assert hd.watched_entities(_config()) == {
        "sensor.aquatech_current_temperature_local",
        "sensor.ai_dh_import_price_forecast",
        "weather.woodville_west_hourly",
        "water_heater.aquatech",
        "binary_sensor.aquatech_compressor",
        "sensor.hwc_predicted_temp",
        "sensor.hwc_power_plan",
    }


def test_forecast_change_triggers_replan_only():
    decision = hd.classify_state_change(
        _config(),
        "sensor.ai_dh_import_price_forecast",
        {"state": "old"},
        {"state": "new"},
    )

    assert decision.replan is True
    assert decision.execute is False


def test_small_tank_temperature_change_is_ignored():
    decision = hd.classify_state_change(
        _config(),
        "sensor.aquatech_current_temperature_local",
        _state(57.0),
        _state(57.1),
    )

    assert decision.replan is False
    assert decision.execute is False


def test_meaningful_tank_temperature_change_triggers_replan():
    decision = hd.classify_state_change(
        _config(),
        "sensor.aquatech_current_temperature_local",
        _state(57.0),
        _state(56.6),
    )

    assert decision.replan is True
    assert decision.execute is False


def test_equipment_change_triggers_executor_only():
    decision = hd.classify_state_change(
        _config(),
        "binary_sensor.aquatech_compressor",
        {"state": "off"},
        {"state": "on"},
    )

    assert decision.replan is False
    assert decision.execute is True


def test_published_plan_change_triggers_executor_only():
    decision = hd.classify_state_change(
        _config(),
        "sensor.hwc_power_plan",
        {"state": "0"},
        {"state": "800"},
    )

    assert decision.replan is False
    assert decision.execute is True
