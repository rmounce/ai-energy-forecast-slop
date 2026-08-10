"""Deterministic, row-derived candidate screening and eligibility rules."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_BIAS_MAX_WORSENING = 10.0
MAX_PRIMARY_REGRESSION = 0.05
LOAD_P65_COVERAGE_MIN = 0.55
LOAD_P65_COVERAGE_MAX = 0.85
SCREENING_SCHEMA_VERSION = 1

FAMILY_SCREENING = {
    "price": {
        "units": "$/MWh",
        "quantiles": (("p30", 0.30), ("p50", 0.50), ("p70", 0.70)),
        "primary": "p50",
        "buckets": (
            ("0-16.5h", 0.0, 16.5),
            ("16.5-28h", 16.5, 28.0),
            ("28-48h", 28.0, 48.0),
            ("48-72h", 48.0, 72.0),
        ),
    },
    "load": {
        "units": "W",
        "quantiles": (("p50", 0.50), ("p65", 0.65), ("p75", 0.75)),
        "primary": "p50",
        "buckets": (
            ("0-24h", 0.0, 24.0),
            ("24-48h", 24.0, 48.0),
            ("48-72h", 48.0, 72.0),
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("screening rows_file must be CSV or Parquet")


def _pinball(actual: np.ndarray, prediction: np.ndarray, alpha: float) -> float:
    error = actual - prediction
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def _finite(values) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _timezone_aware(values) -> bool:
    try:
        return all(pd.Timestamp(value).tzinfo is not None for value in values)
    except (TypeError, ValueError):
        return False


def derive_screening_metrics(
    payload: dict,
    *,
    family: str,
    base_dir: str | Path = ".",
) -> dict:
    """Load identified rows and derive every promotion metric on identical rows."""
    if family not in FAMILY_SCREENING:
        raise ValueError(f"unknown family: {family}")
    required = {"schema_version", "family", "units", "provenance", "row_count"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"screening evidence missing fields: {', '.join(sorted(missing))}")
    if payload["schema_version"] != SCREENING_SCHEMA_VERSION:
        raise ValueError("unsupported screening schema_version")
    if payload["family"] != family:
        raise ValueError("screening family does not match requested family")
    spec = FAMILY_SCREENING[family]
    if payload["units"] != spec["units"]:
        raise ValueError(f"screening units must be {spec['units']}")
    if not isinstance(payload["row_count"], int) or isinstance(payload["row_count"], bool) or payload["row_count"] <= 0:
        raise ValueError("screening row_count must be a positive integer")

    provenance = payload["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("screening provenance must be an object")
    for key in ("command", "rows_file", "rows_sha256"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"screening provenance.{key} is required")
    rows_path = Path(provenance["rows_file"])
    if not rows_path.is_absolute():
        rows_path = Path(base_dir) / rows_path
    rows_path = rows_path.resolve()
    if not rows_path.is_file():
        raise ValueError(f"screening rows_file does not exist: {rows_path}")
    observed_hash = _sha256(rows_path)
    if observed_hash.lower() != provenance["rows_sha256"].lower():
        raise ValueError("screening rows_file SHA-256 mismatch")

    rows = _load_rows(rows_path)
    if len(rows) != payload["row_count"]:
        raise ValueError(f"screening row_count mismatch: descriptor={payload['row_count']}, rows={len(rows)}")
    quantile_names = [name for name, _ in spec["quantiles"]]
    value_columns = ["actual"] + [
        f"{side}_{quantile}"
        for side in ("candidate", "incumbent")
        for quantile in quantile_names
    ]
    required_columns = {"forecast_issue_time", "forecast_target_time", *value_columns}
    missing_columns = required_columns - set(rows.columns)
    if missing_columns:
        raise ValueError(f"screening rows missing columns: {', '.join(sorted(missing_columns))}")

    if not _timezone_aware(rows["forecast_issue_time"]) or not _timezone_aware(rows["forecast_target_time"]):
        raise ValueError("screening issue/target timestamps must be timezone-aware")
    issue = pd.to_datetime(rows["forecast_issue_time"], utc=True, errors="coerce")
    target = pd.to_datetime(rows["forecast_target_time"], utc=True, errors="coerce")
    if issue.isna().any() or target.isna().any():
        raise ValueError("screening issue/target timestamps must be valid UTC timestamps")
    identity = pd.MultiIndex.from_arrays([issue, target])
    if identity.has_duplicates:
        raise ValueError("screening issue/target row identities must be unique")
    horizon_hours = (target - issue).dt.total_seconds().to_numpy(dtype=float) / 3600.0
    if not np.isfinite(horizon_hours).all() or (horizon_hours < 0).any() or (horizon_hours > 72).any():
        raise ValueError("screening horizons must be within 0-72 hours")
    if not np.isclose(horizon_hours * 2.0, np.round(horizon_hours * 2.0)).all():
        raise ValueError("screening horizons must align to 30-minute intervals")

    numeric = rows[value_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("screening values must all be finite numbers")
    if family == "load" and (numeric.to_numpy(dtype=float) < 0).any():
        raise ValueError("load screening values must be non-negative")
    for side in ("candidate", "incumbent"):
        quantile_matrix = numeric[
            [f"{side}_{name}" for name in quantile_names]
        ].to_numpy(dtype=float)
        if (np.diff(quantile_matrix, axis=1) < 0).any():
            raise ValueError(f"screening {side} quantiles must be monotonically ordered")
    actual = numeric["actual"].to_numpy(dtype=float)

    bucket_results = []
    primary_regressions = []
    bias_worsening = []
    for position, (label, lower, upper) in enumerate(spec["buckets"]):
        upper_mask = horizon_hours <= upper if position == len(spec["buckets"]) - 1 else horizon_hours < upper
        mask = (horizon_hours >= lower) & upper_mask
        if not mask.any():
            raise ValueError(f"screening bucket {label} has no rows")
        bucket_actual = actual[mask]
        result = {
            "bucket": label,
            "horizon_start_hours": lower,
            "horizon_end_hours": upper,
            "row_count": int(mask.sum()),
        }
        for side in ("candidate", "incumbent"):
            primary = numeric[f"{side}_{spec['primary']}"].to_numpy(dtype=float)[mask]
            result[f"{side}_primary"] = float(np.mean(np.abs(primary - bucket_actual)))
            result[f"{side}_bias"] = float(np.mean(primary - bucket_actual))
            result[f"{side}_pinball"] = {
                name: _pinball(
                    bucket_actual,
                    numeric[f"{side}_{name}"].to_numpy(dtype=float)[mask],
                    alpha,
                )
                for name, alpha in spec["quantiles"]
            }
            result[f"{side}_coverage"] = {
                name: float(np.mean(bucket_actual <= numeric[f"{side}_{name}"].to_numpy(dtype=float)[mask]))
                for name, _ in spec["quantiles"]
            }
        incumbent_primary = result["incumbent_primary"]
        if incumbent_primary <= 0:
            raise ValueError(f"screening bucket {label} incumbent primary metric must be positive")
        primary_regressions.append((result["candidate_primary"] - incumbent_primary) / incumbent_primary)
        if family == "price":
            result["candidate_bias_mwh"] = result["candidate_bias"]
            result["incumbent_bias_mwh"] = result["incumbent_bias"]
            bias_worsening.append(abs(result["candidate_bias"]) - abs(result["incumbent_bias"]))
        bucket_results.append(result)

    derived = {
        "schema_version": SCREENING_SCHEMA_VERSION,
        "family": family,
        "units": spec["units"],
        "provenance": {
            **provenance,
            "rows_file": str(rows_path),
            "rows_sha256": observed_hash,
        },
        "row_count": len(rows),
        "comparable": True,
        "evaluation_issue_start_utc": issue.min().isoformat(),
        "evaluation_issue_end_utc": issue.max().isoformat(),
        "evaluation_target_start_utc": target.min().isoformat(),
        "evaluation_target_end_utc": target.max().isoformat(),
        "buckets": bucket_results,
        "primary_regressions": primary_regressions,
    }
    if family == "price":
        derived["bias_worsening_mwh"] = bias_worsening
    else:
        derived["p65_coverage"] = float(np.mean(actual <= numeric["candidate_p65"].to_numpy(dtype=float)))
    return derived


def evaluate_eligibility(family: str, metrics: dict | None, *, comparable: bool) -> dict:
    """Return machine-readable screening decisions; invalid evidence fails closed."""
    metrics = metrics or {}
    reasons: list[str] = []
    checks = {"structural": metrics.get("structural") is True, "inference": metrics.get("inference") is True}
    if not checks["structural"]:
        reasons.append("structural checks did not pass")
    if not checks["inference"]:
        reasons.append("inference smoke did not pass")
    if comparable is not True:
        reasons.append("no identical-row incumbent comparison evidence")

    regressions = metrics.get("primary_regressions", [])
    if not regressions or not _finite(regressions):
        reasons.append("primary metric evidence is missing or non-finite")
    elif any(float(value) > MAX_PRIMARY_REGRESSION for value in regressions):
        reasons.append("a primary metric regresses by more than 5%")

    if family == "price":
        biases = metrics.get("bias_worsening_mwh", [])
        if not biases or not _finite(biases):
            reasons.append("price horizon-bucket bias evidence is missing or non-finite")
        elif any(float(value) > PRICE_BIAS_MAX_WORSENING for value in biases):
            reasons.append("price horizon-bucket absolute bias worsens by more than $10/MWh")
    elif family == "load":
        coverage = metrics.get("p65_coverage")
        if coverage is None or not _finite([coverage]):
            reasons.append("load p65 coverage evidence is missing or non-finite")
        elif not LOAD_P65_COVERAGE_MIN <= float(coverage) <= LOAD_P65_COVERAGE_MAX:
            reasons.append("load p65 empirical coverage is outside 0.55-0.85")
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
