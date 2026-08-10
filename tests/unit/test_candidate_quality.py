import pytest

from candidate_quality import derive_screening_metrics, evaluate_eligibility


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


def test_screening_metrics_are_derived_from_bucket_components():
    payload = {
        "provenance": {"command": "eval.py", "rows_file": "rows.parquet"},
        "row_count": 10,
        "comparable": True,
        "buckets": [{"candidate_primary": 10.5, "incumbent_primary": 10.0,
                     "candidate_bias_mwh": 4, "incumbent_bias_mwh": 1}],
    }
    result = derive_screening_metrics(payload)
    assert result["primary_regressions"] == [pytest.approx(0.05)]
    assert result["bias_worsening_mwh"] == [pytest.approx(3)]


def test_screening_rejects_bare_summary_claims():
    with pytest.raises(ValueError, match="missing fields"):
        derive_screening_metrics({"primary_regressions": [0]})
