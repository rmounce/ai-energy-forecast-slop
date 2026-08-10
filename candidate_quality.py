"""Deterministic candidate screening rules shared by reports and tests."""

from __future__ import annotations

PRICE_BIAS_MAX_WORSENING = 10.0
MAX_PRIMARY_REGRESSION = 0.05
LOAD_P65_COVERAGE_MIN = 0.55
LOAD_P65_COVERAGE_MAX = 0.85


def derive_screening_metrics(payload: dict) -> dict:
    """Derive comparison values from bucket-level components, never bare claims."""
    required = {"provenance", "row_count", "buckets", "comparable"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"screening evidence missing fields: {', '.join(sorted(missing))}")
    if not isinstance(payload["provenance"], dict) or not payload["provenance"].get("command"):
        raise ValueError("screening provenance.command is required")
    if not isinstance(payload["row_count"], int) or payload["row_count"] <= 0:
        raise ValueError("screening row_count must be positive")
    buckets = payload["buckets"]
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("screening buckets are required")
    regressions = []
    bias_worsening = []
    for bucket in buckets:
        for key in ("candidate_primary", "incumbent_primary"):
            if key not in bucket:
                raise ValueError(f"screening bucket missing {key}")
        incumbent = float(bucket["incumbent_primary"])
        candidate = float(bucket["candidate_primary"])
        if incumbent == 0:
            raise ValueError("screening incumbent primary metric cannot be zero")
        regressions.append((candidate - incumbent) / abs(incumbent))
        if "candidate_bias_mwh" in bucket and "incumbent_bias_mwh" in bucket:
            bias_worsening.append(abs(float(bucket["candidate_bias_mwh"])) - abs(float(bucket["incumbent_bias_mwh"])))
    result = dict(payload)
    result["primary_regressions"] = regressions
    if bias_worsening:
        result["bias_worsening_mwh"] = bias_worsening
    return result


def evaluate_eligibility(family: str, metrics: dict | None, *, comparable: bool) -> dict:
    """Return machine-readable screening decisions; missing evidence is a failure."""
    metrics = metrics or {}
    reasons: list[str] = []
    checks = {"structural": bool(metrics.get("structural", False)), "inference": bool(metrics.get("inference", False))}
    if not checks["structural"]:
        reasons.append("structural checks did not pass")
    if not checks["inference"]:
        reasons.append("inference smoke did not pass")
    if not comparable:
        reasons.append("no identical-row incumbent comparison evidence")

    regressions = metrics.get("primary_regressions", [])
    if not regressions:
        reasons.append("primary metric evidence is missing")
    elif any(value > MAX_PRIMARY_REGRESSION for value in regressions):
        reasons.append("a primary metric regresses by more than 5%")

    if family == "price":
        biases = metrics.get("bias_worsening_mwh", [])
        if not biases:
            reasons.append("price horizon-bucket bias evidence is missing")
        elif any(value > PRICE_BIAS_MAX_WORSENING for value in biases):
            reasons.append("price horizon-bucket absolute bias worsens by more than $10/MWh")
    elif family == "load":
        coverage = metrics.get("p65_coverage")
        if coverage is None:
            reasons.append("load p65 coverage evidence is missing")
        elif not LOAD_P65_COVERAGE_MIN <= coverage <= LOAD_P65_COVERAGE_MAX:
            reasons.append("load p65 empirical coverage is outside 0.55–0.85")
    else:
        raise ValueError(f"unknown family: {family}")
    return {
        "eligible_for_manual_promotion": not reasons,
        "thresholds": {
            "max_primary_regression": MAX_PRIMARY_REGRESSION,
            "price_max_bias_worsening_mwh": PRICE_BIAS_MAX_WORSENING,
            "load_p65_coverage": [LOAD_P65_COVERAGE_MIN, LOAD_P65_COVERAGE_MAX],
        },
        "eligibility_reasons": reasons,
        "checks": checks,
    }
