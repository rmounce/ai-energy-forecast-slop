"""Canonical price forecast source contracts for evaluation scripts.

The eval tree has several 72h price paths with similar names. Keep source
lineage here so scripts can print and check what they are actually scoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PriceSourceContract:
    label: str
    artifact: str
    resolution: str
    horizon: str
    apf_backed: bool
    status: str
    lineage: str
    use_for: str
    avoid_for: str

    @property
    def apf_status(self) -> str:
        return "APF-backed" if self.apf_backed else "APF-free"


SOURCE_CONTRACTS: dict[str, PriceSourceContract] = {
    "amber_apf_lgbm": PriceSourceContract(
        label="amber_apf_lgbm",
        artifact="price_forecast_log.csv rows with model_name='price'",
        resolution="30 min",
        horizon="0-72h, 144 steps",
        apf_backed=True,
        status="production incumbent / active APF extrapolation baseline",
        lineage=(
            "Amber commercial APF seeds the near horizon; the incumbent price "
            "LightGBM extrapolates the remaining tail to 72h."
        ),
        use_for=(
            "Questions about the current production APF extrapolation path, "
            "including tail residual correction and STPASA feature value."
        ),
        avoid_for="APF-free model replacement claims.",
    ),
    "pd_direct": PriceSourceContract(
        label="pd_direct",
        artifact="pd_direct_forecast_log.csv / pd_direct sensors",
        resolution="30 min",
        horizon="0-72h, 144 steps",
        apf_backed=False,
        status=(
            "suspended; retained for reference, but not currently trusted as a "
            "replacement path"
        ),
        lineage="APF-free strategic curve from AEMO pre-dispatch/seven-day inputs.",
        use_for="Historical comparisons or explicit APF-free revival work.",
        avoid_for="Evidence about APF extrapolation improvements.",
    ),
    "p5min_tactical": PriceSourceContract(
        label="p5min_tactical",
        artifact="p5min_forecast_log.csv / sensor.ai_p5min_price_forecast*",
        resolution="5 min",
        horizon="0-60 min, 12 steps",
        apf_backed=False,
        status=(
            "suspended tactical Tier 1 experiment; retained for reference, "
            "but not currently used by production EMHASS"
        ),
        lineage="AEMO P5MIN plus recent dispatch-price features into tactical LightGBM.",
        use_for="Explicit tactical-price revival work only.",
        avoid_for="Current production APF extrapolation evaluation.",
    ),
    "model_a_hybrid": PriceSourceContract(
        label="model_a_hybrid",
        artifact="retro_tier1_forecasts.pkl + retro_tft_forecasts.pkl",
        resolution="5 min tactical prefix plus 30 min strategic tail",
        horizon="0-72h stitched curve",
        apf_backed=False,
        status=(
            "suspended; retained for reference, but not currently trusted as a "
            "replacement path"
        ),
        lineage="Tier 1 tactical LightGBM prefix stitched to TFT strategic q50 tail.",
        use_for="Historical hybrid/TFT eval references or explicit revival work.",
        avoid_for="Evidence about APF extrapolation improvements.",
    ),
    "lgbm_strategic": PriceSourceContract(
        label="lgbm_strategic",
        artifact="retro_lgbm_strategic_forecasts.pkl",
        resolution="30 min",
        horizon="0-72h, 144 steps",
        apf_backed=False,
        status=(
            "suspended APF-free experiment; generated artifacts are not APF "
            "extrapolation evidence"
        ),
        lineage=(
            "Retrospective strategic LightGBM from AEMO/actual/history features, "
            "with optional STPASA tail features."
        ),
        use_for="Explicit APF-free strategic model experiments only.",
        avoid_for="APF extrapolation evaluation, APF tail residual correction.",
    ),
}


def get_source_contract(label: str) -> PriceSourceContract:
    try:
        return SOURCE_CONTRACTS[label]
    except KeyError as exc:
        known = ", ".join(sorted(SOURCE_CONTRACTS))
        raise KeyError(f"unknown price source '{label}'. Known sources: {known}") from exc


def format_source_banner(label: str, *, prefix: str = "Evaluating") -> str:
    contract = get_source_contract(label)
    lines = [
        f"{prefix}: {contract.label}",
        f"  source status: {contract.apf_status}; {contract.status}",
        f"  artifact: {contract.artifact}",
        f"  resolution/horizon: {contract.resolution}; {contract.horizon}",
        f"  lineage: {contract.lineage}",
    ]
    if not contract.apf_backed:
        lines.append(f"  WARNING: {contract.label} is APF-free; {contract.avoid_for}")
    return "\n".join(lines)


def require_apf_backed(label: str) -> PriceSourceContract:
    contract = get_source_contract(label)
    if not contract.apf_backed:
        raise ValueError(
            f"{label} is APF-free and cannot answer APF extrapolation questions: "
            f"{contract.avoid_for}"
        )
    return contract
