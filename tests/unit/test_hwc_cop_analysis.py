import hwc_cop_analysis as hca


def test_series_query_can_read_aggregate_weather_without_entity_tag():
    q = hca._series_query("humidity_adelaide", days=12, field="mean_value", rp="rp_30m")

    assert q == 'SELECT "mean_value" FROM "rp_30m"."humidity_adelaide" WHERE time> now()-12d'


def test_series_query_keeps_entity_filter_for_raw_ha_sensors():
    q = hca._series_query("sensor__temperature", eid="heat_pump_temperature", days=3)

    assert q == (
        'SELECT "value" FROM "sensor__temperature" '
        "WHERE entity_id='heat_pump_temperature' AND time> now()-3d"
    )
