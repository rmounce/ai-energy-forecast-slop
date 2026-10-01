"""Small, dependency-light contracts for production forecast boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd


class ForecastContractError(ValueError):
    """Raised when a forecast surface cannot be safely used or published."""


@dataclass(frozen=True)
class PredictionInputs:
    """Already acquired frames; resident shadow must validate before admission."""
    future_sources: Mapping[str, pd.DataFrame]
    historical_df: pd.DataFrame
    forecast_start: datetime


@dataclass(frozen=True)
class PredictionOutcome:
    family: str
    source: str
    model_bundle_id: str
    forecasts: Mapping[str, pd.DataFrame]
    apf_age_minutes: float | None = None
    publication_result: str = "not_requested"
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))

    @property
    def point_counts(self) -> dict[str, int]:
        return {key: len(frame) for key, frame in self.forecasts.items()}


def _expected(family: str) -> tuple[tuple[str, ...], int, bool]:
    if family == "price":
        return ("price_p30", "price", "price_p70"), 144, False
    if family == "load":
        return ("load", "load_p65", "load_p75"), 144, True
    raise ForecastContractError(f"unknown forecast family: {family}")


def _utc(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def validate_forecast_family(
    forecasts: Mapping[str, pd.DataFrame],
    family: str,
    *,
    expected_start: pd.Timestamp | None = None,
    tolerance_minutes: int = 0,
) -> None:
    """Validate one complete, publishable 72-hour quantile family."""
    keys, points, non_negative = _expected(family)
    missing = [key for key in keys if key not in forecasts or forecasts[key] is None]
    if missing:
        raise ForecastContractError(f"{family}: missing quantiles: {', '.join(missing)}")

    frames: dict[str, pd.DataFrame] = {}
    for key in keys:
        frame = forecasts[key]
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] < 1:
            raise ForecastContractError(f"{family}/{key}: empty or invalid frame")
        if len(frame) != points:
            raise ForecastContractError(f"{family}/{key}: expected {points} points, got {len(frame)}")
        index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        if index.isna().any() or not index.is_unique or not index.is_monotonic_increasing:
            raise ForecastContractError(f"{family}/{key}: index must be UTC-aware, sorted, and unique")
        if len(index) > 1 and not (index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=30)).all():
            raise ForecastContractError(f"{family}/{key}: index is not contiguous at 30-minute resolution")
        if expected_start is not None:
            delta = abs((index[0] - _utc(expected_start)).total_seconds() / 60)
            if delta > tolerance_minutes:
                raise ForecastContractError(f"{family}/{key}: unexpected first target {index[0].isoformat()}")
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ForecastContractError(f"{family}/{key}: values must be finite")
        if non_negative and (values < 0).any():
            raise ForecastContractError(f"{family}/{key}: values must be non-negative")
        frames[key] = pd.DataFrame({key: values}, index=index)

    values = np.column_stack([frames[key].iloc[:, 0].to_numpy() for key in keys])
    if (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
        raise ForecastContractError(f"{family}: quantiles cross")


def rearrange_quantile_family(forecasts: Mapping[str, pd.DataFrame], family: str, quantile_order: tuple[str, ...] | None = None) -> dict[str, pd.DataFrame]:
    """Apply the production monotonic-rearrangement policy to a family."""
    keys, _, _ = _expected(family)
    ordered = tuple(quantile_order or keys)
    if set(ordered) != set(keys):
        raise ForecastContractError(f"{family}: quantile rearrangement keys do not match contract")
    matrix = np.column_stack([forecasts[key].iloc[:, 0].to_numpy(dtype=float) for key in ordered])
    matrix = np.sort(matrix, axis=1)
    return {
        key: pd.DataFrame({key: matrix[:, position]}, index=forecasts[key].index)
        for position, key in enumerate(ordered)
    }


def validate_apf(
    frame: pd.DataFrame,
    *,
    now: pd.Timestamp | None = None,
    max_age_minutes: float = 180.0,
) -> float:
    """Validate an aggregated Amber APF and return its observed age in minutes."""
    if frame is None or frame.empty:
        raise ForecastContractError("Amber APF is empty")
    index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    if index.isna().any() or not index.is_unique or not index.is_monotonic_increasing:
        raise ForecastContractError("Amber APF index must be UTC-aware, sorted, and unique")
    if len(index) > 1 and not (index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=30)).all():
        raise ForecastContractError("Amber APF is not contiguous at 30-minute resolution")
    if not np.isfinite(pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(dtype=float)).all():
        raise ForecastContractError("Amber APF contains non-finite values")
    observed_now = pd.Timestamp.now(tz="UTC") if now is None else _utc(now)
    # APF data is a future surface; freshness is measured from its first
    # interval, not its final horizon timestamp.
    age = (observed_now - index[0]).total_seconds() / 60.0
    if age > max_age_minutes:
        raise ForecastContractError(f"Amber APF is stale: age_minutes={age:.1f}, limit={max_age_minutes:.1f}")
    return age
