import numpy as np
import pandas as pd
import pytest

from production_contract import ForecastContractError, rearrange_quantile_family, validate_apf, validate_forecast_family


def family(family_name="price", points=144):
    names = ("price_p30", "price", "price_p70") if family_name == "price" else ("load", "load_p65", "load_p75")
    index = pd.date_range("2026-08-10T00:00:00Z", periods=points, freq="30min")
    return {name: pd.DataFrame({name: np.arange(points, dtype=float)}, index=index) for name in names}


def test_complete_price_family_passes():
    validate_forecast_family(family(), "price")


@pytest.mark.parametrize("mutator", [
    lambda data: data.pop("price_p70"),
    lambda data: data["price"].iloc.__setitem__((0, 0), np.nan),
    lambda data: data["price_p30"].iloc.__setitem__((0, 0), 2_000),
])
def test_invalid_price_family_fails(mutator):
    data = family()
    mutator(data)
    with pytest.raises(ForecastContractError):
        validate_forecast_family(data, "price")


def test_load_rejects_negative_values_and_crossing():
    data = family("load")
    data["load_p65"].iloc[0, 0] = -1
    with pytest.raises(ForecastContractError):
        validate_forecast_family(data, "load")

    data = family("load")
    data["load_p75"].iloc[0, 0] = -1
    with pytest.raises(ForecastContractError):
        validate_forecast_family(data, "load")


def test_apf_staleness_is_reported():
    index = pd.date_range("2026-08-10T00:00:00Z", periods=2, freq="30min")
    with pytest.raises(ForecastContractError, match="stale"):
        validate_apf(pd.DataFrame({"price": [1, 2]}, index=index), now="2026-08-10T04:00:00Z", max_age_minutes=60)


@pytest.mark.parametrize("bad_index", [
    pd.DatetimeIndex(["2026-08-10T00:00:00Z", "2026-08-10T00:30:00Z", "2026-08-10T00:30:00Z"]),
    pd.DatetimeIndex(["2026-08-10T00:00:00Z", "2026-08-10T01:00:00Z"]),
])
def test_apf_duplicate_or_gap_fails(bad_index):
    with pytest.raises(ForecastContractError):
        validate_apf(pd.DataFrame({"price": range(len(bad_index))}, index=bad_index), now="2026-08-10T00:01:00Z")


@pytest.mark.parametrize("points", [143, 145])
def test_family_wrong_point_count_fails(points):
    with pytest.raises(ForecastContractError):
        validate_forecast_family(family(points=points), "price")


def test_load_quantile_crossing_fails():
    data = family("load")
    data["load_p65"].iloc[0, 0] = 0
    data["load"].iloc[0, 0] = 1
    with pytest.raises(ForecastContractError, match="cross"):
        validate_forecast_family(data, "load")


def test_shared_rearrangement_makes_load_smoke_and_runtime_policy_identical():
    data = family("load")
    data["load"].iloc[0, 0] = 3
    data["load_p65"].iloc[0, 0] = 1
    data["load_p75"].iloc[0, 0] = 2
    ordered = rearrange_quantile_family(data, "load", ("load", "load_p65", "load_p75"))
    validate_forecast_family(ordered, "load")
    assert [ordered[key].iloc[0, 0] for key in ("load", "load_p65", "load_p75")] == [1, 2, 3]
