import pytest

from eval.price_source_contracts import (
    format_source_banner,
    get_source_contract,
    require_apf_backed,
)


def test_amber_apf_lgbm_contract_is_apf_backed():
    contract = get_source_contract("amber_apf_lgbm")

    assert contract.apf_backed is True
    assert "price_forecast_log.csv" in contract.artifact
    assert "model_name='price'" in contract.artifact


def test_apf_free_sources_are_not_allowed_for_apf_extrapolation():
    with pytest.raises(ValueError, match="APF-free"):
        require_apf_backed("lgbm_strategic")


def test_banner_warns_for_suspended_apf_free_source():
    banner = format_source_banner("lgbm_strategic")

    assert "APF-free" in banner
    assert "WARNING" in banner
    assert "suspended" in banner


def test_tactical_tier1_contract_is_suspended_apf_free():
    contract = get_source_contract("p5min_tactical")

    assert contract.apf_backed is False
    assert "suspended" in contract.status
    assert "0-60 min" in contract.horizon
