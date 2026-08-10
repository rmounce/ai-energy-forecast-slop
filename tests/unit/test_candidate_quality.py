import pytest

from candidate_quality import evaluate_eligibility


def passing(family):
    base = {
        "structural": True,
        "inference": True,
        "primary_regressions": [0.05],
    }
    if family == "price":
        base["bias_worsening_mwh"] = [10.0]
    else:
        base["p65_coverage"] = 0.55
    return base


@pytest.mark.parametrize("family", ["price", "load"])
def test_threshold_boundaries_are_eligible(family):
    assert evaluate_eligibility(family, passing(family), comparable=True)["eligible_for_manual_promotion"]


def test_price_threshold_overage_is_ineligible():
    metrics = passing("price")
    metrics["bias_worsening_mwh"] = [10.0001]
    result = evaluate_eligibility("price", metrics, comparable=True)
    assert not result["eligible_for_manual_promotion"]


def test_load_coverage_upper_boundary_and_overage():
    metrics = passing("load")
    metrics["p65_coverage"] = 0.85
    assert evaluate_eligibility("load", metrics, comparable=True)["eligible_for_manual_promotion"]
    metrics["p65_coverage"] = 0.8501
    assert not evaluate_eligibility("load", metrics, comparable=True)["eligible_for_manual_promotion"]


def test_missing_comparable_evidence_fails_closed():
    result = evaluate_eligibility("price", passing("price"), comparable=False)
    assert not result["eligible_for_manual_promotion"]
