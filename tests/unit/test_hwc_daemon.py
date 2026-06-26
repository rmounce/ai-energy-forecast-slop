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
        compressor_on=False,
    )


def test_command_key_dedups_identical_commands_and_ignores_noops():
    import services.hwc_daemon as hd
    import hwc_executor as he

    off = he.Decision(action="off", reason="r")
    heat60 = he.Decision(action="heat", reason="r", setpoint_c=60.0)
    heat60b = he.Decision(action="heat", reason="other", setpoint_c=60.04)  # rounds to 60.0
    heat58 = he.Decision(action="heat", reason="r", setpoint_c=58.0)

    # Identical actuation => equal key (skipped); setpoint rounds to 0.1.
    assert hd.command_key(off) == hd.command_key(he.Decision(action="off", reason="x"))
    assert hd.command_key(heat60) == hd.command_key(heat60b)
    assert hd.command_key(heat60) != hd.command_key(heat58)
    assert hd.command_key(off) != hd.command_key(heat60)
    # No-op actions are never dedup-keyed (and never actuated).
    assert hd.command_key(he.Decision(action="idle", reason="r")) is None
    assert hd.command_key(he.Decision(action="wait", reason="r")) is None


def test_does_not_suppress_heat_or_expired_grace():
    assert not hd.should_suppress_off_after_heat(
        decision_action="heat",
        now=110.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_on=False,
    )
    assert not hd.should_suppress_off_after_heat(
        decision_action="off",
        now=701.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_on=False,
    )
    # Compressor confirmed running => the start registered, so an off is a genuine stop
    # (this is the already-running case the old edge latch missed).
    assert not hd.should_suppress_off_after_heat(
        decision_action="off",
        now=110.0,
        last_heat_command_at=100.0,
        grace_seconds=600,
        compressor_on=True,
    )


def test_suppresses_heat_inside_min_off_grace():
    # A restart within the minimum rest period is held back (hardware short-cycle protection,
    # symmetric with the off-after-heat guard).
    assert hd.should_suppress_heat_after_off(
        decision_action="heat", now=160.0, last_off_command_at=100.0, min_off_seconds=180
    )


def test_does_not_suppress_off_or_expired_min_off():
    # Only heat is gated; once the rest elapses a restart is allowed; and no prior off => nothing
    # to rest from.
    assert not hd.should_suppress_heat_after_off(
        decision_action="off", now=160.0, last_off_command_at=100.0, min_off_seconds=180
    )
    assert not hd.should_suppress_heat_after_off(
        decision_action="heat", now=300.0, last_off_command_at=100.0, min_off_seconds=180
    )
    assert not hd.should_suppress_heat_after_off(
        decision_action="heat", now=160.0, last_off_command_at=0.0, min_off_seconds=180
    )


def _eff(**overrides):
    kw = dict(
        raw_on=False,
        last_command_action="heat",
        tank_at_target=False,
        now=1000.0,
        last_heat_command_at=0.0,
        last_on_at=0.0,
        start_grace_s=120.0,
        defrost_grace_s=600.0,
    )
    kw.update(overrides)
    return hd.effective_compressor_running(**kw)


def test_effective_running_confirmed_by_raw_sensor():
    assert _eff(raw_on=True)


def test_effective_running_commanded_off_is_immediate():
    # A commanded stop is taken at face value even while the off-edge sensor lag still reads on,
    # so the planner won't price a continue->restart spurious cycle.
    assert not _eff(last_command_action="off", raw_on=True)


def test_effective_running_start_lag_after_heat_command():
    # Compressor started ~instantly; Tuya sensor still reads off inside the start grace.
    assert _eff(raw_on=False, last_heat_command_at=950.0, now=1000.0)  # 50s < 120s
    assert not _eff(raw_on=False, last_heat_command_at=800.0, now=1000.0)  # 200s > 120s


def test_effective_running_defrost_pause_below_target():
    # Brief off while below target with heat still commanded => transient defrost pause.
    assert _eff(raw_on=False, last_on_at=900.0, now=1000.0)  # off 100s < 600s
    assert not _eff(raw_on=False, last_on_at=300.0, now=1000.0)  # off 700s > 600s


def test_effective_running_at_target_is_a_genuine_stop():
    # At setpoint the unit's own thermostat stops it: not a pause, even within the defrost window.
    assert not _eff(raw_on=False, tank_at_target=True, last_on_at=950.0, now=1000.0)


def test_mark_target_reached_is_level_based(tmp_path):
    from datetime import datetime, timezone

    cfg = _config()
    cfg["hwc"]["daemon"]["state_file"] = str(tmp_path / "state.json")
    d = hd.HwcDaemon.__new__(hd.HwcDaemon)
    d.config = cfg
    d.last_reached_target_at = None

    # A single at/above-target observation latches even with no upward-crossing edge — this is
    # the false-negative fix (daemon restart while hot / dropped crossing event).
    d._mark_target_reached(61.0, datetime(2026, 6, 24, 4, 0, tzinfo=timezone.utc))
    first = d.last_reached_target_at
    assert first is not None

    # Idempotent within the same local day (no re-latch on later readings).
    d._mark_target_reached(62.0, datetime(2026, 6, 24, 5, 0, tzinfo=timezone.utc))
    assert d.last_reached_target_at == first

    # Below target does nothing.
    d.last_reached_target_at = None
    d._mark_target_reached(59.0, datetime(2026, 6, 24, 6, 0, tzinfo=timezone.utc))
    assert d.last_reached_target_at is None


def _soc_daemon(tmp_path, soc=None):
    cfg = _config()
    cfg["hwc"]["daemon"]["state_file"] = str(tmp_path / "state.json")
    cfg["hwc"]["dp_planner"] = {"soc_model": True}
    d = hd.HwcDaemon.__new__(hd.HwcDaemon)
    d.config = cfg
    d.last_reached_target_at = None
    d.soc = soc
    return d


def test_soc_tracker_disabled_returns_none(tmp_path):
    d = _soc_daemon(tmp_path)
    d.config["hwc"]["dp_planner"] = {}  # flag off
    assert d._update_soc_tracker(probe_c=54.0, heating=False) is None


def test_soc_tracker_seeds_and_persists(tmp_path):
    d = _soc_daemon(tmp_path)  # soc=None ⇒ first call seeds
    seed = d._update_soc_tracker(probe_c=54.0, heating=False)
    assert seed is not None
    v, t = seed
    assert v == 0.5 and t == 54.0          # conservative seed, T_hot≈probe
    assert d.soc["v_hot"] == 0.5 and "updated_at" in d.soc
    # persisted and reloadable
    reloaded = hd.HwcDaemon.__new__(hd.HwcDaemon)
    reloaded.config = d.config
    assert reloaded._load_state()["soc"]["v_hot"] == 0.5


def test_soc_tracker_top_watermark_snaps_full(tmp_path):
    import time as _t
    d = _soc_daemon(tmp_path, soc={"v_hot": 0.3, "t_hot": 55.0, "updated_at": _t.time() - 60})
    v, t = d._update_soc_tracker(probe_c=60.0, heating=True)  # probe at target ⇒ full
    assert v == 1.0 and t >= 60.0


def test_soc_state_save_preserves_target_reached(tmp_path):
    # The two persisted facts coexist: writing the SoC state must not drop last_reached_target_at.
    d = _soc_daemon(tmp_path)
    d.last_reached_target_at = "2026-06-26T04:00:00+00:00"
    d._update_soc_tracker(probe_c=54.0, heating=False)
    assert d._load_state()["last_reached_target_at"] == "2026-06-26T04:00:00+00:00"


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
