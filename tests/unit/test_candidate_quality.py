import hashlib
import numpy as np
import pandas as pd
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


def write_rows(tmp_path, family="price"):
    horizons = [1.0, 20.0, 36.0, 60.0] if family == "price" else [1.0, 30.0, 60.0]
    issue = pd.Timestamp("2026-08-01T00:00:00Z")
    rows = []
    quantiles = ("p30", "p50", "p70") if family == "price" else ("p50", "p65", "p75")
    for position, horizon in enumerate(horizons):
        actual = 100.0 + position
        row = {
            "forecast_issue_time": issue.isoformat(),
            "forecast_target_time": (issue + pd.Timedelta(hours=horizon)).isoformat(),
            "actual": actual,
        }
        for side, offset in (("candidate", 1.0), ("incumbent", 2.0)):
            for q_position, quantile in enumerate(quantiles):
                row[f"{side}_{quantile}"] = actual + offset + q_position
        rows.append(row)
    path = tmp_path / f"{family}_rows.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor = {
        "schema_version": 1,
        "family": family,
        "units": "$/MWh" if family == "price" else "W",
        "row_count": len(rows),
        "provenance": {
            "command": "fixed-eval-command",
            "rows_file": path.name,
            "rows_sha256": digest,
        },
    }
    return path, descriptor


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


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "bad"])
def test_non_finite_or_invalid_metric_fails_closed(bad):
    metrics = passing("price")
    metrics["primary_regressions"] = [bad]
    assert not evaluate_eligibility("price", metrics, comparable=True)["eligible_for_manual_promotion"]


def test_screening_metrics_are_derived_from_hashed_identical_rows(tmp_path):
    _, descriptor = write_rows(tmp_path, "price")
    result = derive_screening_metrics(descriptor, family="price", base_dir=tmp_path)
    assert [bucket["bucket"] for bucket in result["buckets"]] == [
        "0-16.5h", "16.5-28h", "28-48h", "48-72h",
    ]
    assert result["row_count"] == 4
    assert result["comparable"] is True
    assert result["primary_regressions"] == [pytest.approx(-1 / 3)] * 4
    assert all("candidate_pinball" in bucket for bucket in result["buckets"])


def test_screening_derives_load_p65_coverage(tmp_path):
    path, descriptor = write_rows(tmp_path, "load")
    rows = pd.read_csv(path)
    rows.loc[0, "candidate_p50"] = rows.loc[0, "actual"] - 2
    rows.loc[0, "candidate_p65"] = rows.loc[0, "actual"] - 1
    rows.to_csv(path, index=False)
    descriptor["provenance"]["rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = derive_screening_metrics(descriptor, family="load", base_dir=tmp_path)
    assert result["p65_coverage"] == pytest.approx(2 / 3)


def test_screening_rejects_hash_mismatch_and_nan_rows(tmp_path):
    path, descriptor = write_rows(tmp_path, "price")
    descriptor["provenance"]["rows_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        derive_screening_metrics(descriptor, family="price", base_dir=tmp_path)

    _, descriptor = write_rows(tmp_path, "price")
    rows = pd.read_csv(path)
    rows.loc[0, "candidate_p50"] = np.nan
    rows.to_csv(path, index=False)
    descriptor["provenance"]["rows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="finite"):
        derive_screening_metrics(descriptor, family="price", base_dir=tmp_path)


def test_descriptor_itself_cannot_claim_summary_metrics(tmp_path):
    _, descriptor = write_rows(tmp_path, "price")
    descriptor["primary_regressions"] = [-999]
    result = derive_screening_metrics(descriptor, family="price", base_dir=tmp_path)
    assert result["primary_regressions"] != [-999]
