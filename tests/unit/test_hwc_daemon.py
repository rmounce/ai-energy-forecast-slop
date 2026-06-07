import services.hwc_daemon as hd


def _config():
    return {
        "home_assistant": {
            "weather_entity": "weather.woodville_west_hourly",
        },
        "timezone": "Australia/Adelaide",
        "hwc": {
            "tank_temp_entity": "sensor.aquatech_current_temperature_local",
            "import_price_entity": "sensor.ai_dh_import_price_forecast",
            "short_term_import_price_entity": "sensor.ai_mpc_import_price_forecast",
            "emhass_mpc_unit_load_cost_entity": "sensor.mpc_unit_load_cost",
            "emhass_dh_unit_load_cost_entity": "sensor.dh_unit_load_cost",
            "predicted_temp_entity": "sensor.predicted_temp",
            "power_plan_entity": "sensor.power_plan",
            "publish_prefix": "hwc_",
            "optimization_time_step": 5,
            "thermal": {
                "desired_temp": 60,
            },
            "actuation": {
                "water_heater_entity": "water_heater.aquatech",
                "compressor_entity": "binary_sensor.aquatech_compressor",
                "setpoint_min_c": 55,
                "setpoint_max_c": 60,
                "min_heat_start_delta_c": 2.0,
            },
            "daemon": {
                "tank_temp_replan_delta_c": 0.3,
                "heat_command_grace_seconds": 600,
                "fallback_enabled": True,
                "fallback_window_start": "10:00",
                "fallback_window_end": "16:00",
                "fallback_min_temp_c": 48,
                "fallback_setpoint_c": 60,
            },
        },
    }


def _state(value):
    return {"state": str(value)}


def test_watched_entities_include_inputs_equipment_and_published_plan():
    assert hd.watched_entities(_config()) == {
        "sensor.aquatech_current_temperature_local",
        "sensor.dh_unit_load_cost",
        "sensor.mpc_unit_load_cost",
        "weather.woodville_west_hourly",
        "water_heater.aquatech",
        "binary_sensor.aquatech_compressor",
        "sensor.hwc_predicted_temp",
        "sensor.hwc_power_plan",
    }


def test_forecast_change_triggers_replan_only():
    decision = hd.classify_state_change(
        _config(),
        "sensor.dh_unit_load_cost",
        {"state": "old"},
        {"state": "new"},
    )

    assert decision.replan is True
    assert decision.execute is False


def test_short_term_forecast_change_triggers_replan_only():
    decision = hd.classify_state_change(
        _config(),
        "sensor.mpc_unit_load_cost",
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


def test_suppresses_off_inside_heat_command_grace():
    assert hd.should_suppress_off_after_heat(
        decision_action="off",
        now=110.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_seen_on_since_heat=False,
    )


def test_does_not_suppress_heat_or_expired_grace():
    assert not hd.should_suppress_off_after_heat(
        decision_action="heat",
        now=110.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_seen_on_since_heat=False,
    )
    assert not hd.should_suppress_off_after_heat(
        decision_action="off",
        now=701.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_seen_on_since_heat=False,
    )
    assert not hd.should_suppress_off_after_heat(
        decision_action="off",
        now=110.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_seen_on_since_heat=True,
    )


def test_target_reached_local_date_maps_utc_to_local_date():
    assert hd.target_reached_local_date(
        _config(),
        "2026-06-02T14:45:00+00:00",
    ) == "2026-06-03"


def test_target_reached_local_date_ignores_missing_or_bad_timestamp():
    assert hd.target_reached_local_date(_config(), None) is None
    assert hd.target_reached_local_date(_config(), "not-a-date") is None


def test_fallback_heats_inside_fixed_window_when_tank_below_threshold():
    decision = hd.fallback_decision(
        _config(),
        now_utc=hd.datetime(2026, 6, 2, 1, 0, tzinfo=hd.timezone.utc),  # 10:30 ACST
        tank_temp_c=56.0,
        compressor_on=False,
    )

    assert decision.action == "heat"
    assert decision.setpoint_c == 60
    assert decision.reason.startswith("fallback fixed-window heat")


def test_fallback_emergency_heats_outside_window_below_floor():
    decision = hd.fallback_decision(
        _config(),
        now_utc=hd.datetime(2026, 6, 2, 12, 0, tzinfo=hd.timezone.utc),
        tank_temp_c=47.5,
        compressor_on=False,
    )

    assert decision.action == "heat"
    assert decision.reason.startswith("fallback emergency heat")


def test_fallback_waits_if_compressor_running_outside_window():
    decision = hd.fallback_decision(
        _config(),
        now_utc=hd.datetime(2026, 6, 2, 12, 0, tzinfo=hd.timezone.utc),
        tank_temp_c=55.0,
        compressor_on=True,
    )

    assert decision.action == "wait"


def test_fallback_can_be_disabled():
    config = _config()
    config["hwc"]["daemon"]["fallback_enabled"] = False

    assert hd.fallback_decision(
        config,
        now_utc=hd.datetime(2026, 6, 2, 1, 0, tzinfo=hd.timezone.utc),
        tank_temp_c=45.0,
        compressor_on=False,
    ) is None
