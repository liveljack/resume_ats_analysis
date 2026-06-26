"""Layer 4: Weighted fusion of rule / ML / LLM signals.

When an upstream signal is unavailable (e.g. no LLM API key, or ML disabled),
its weight is redistributed proportionally across the remaining available
signals — so the system always produces a calibrated blended score rather than
silently dropping a component.

Weights (config.FUSION_QUALITY / FUSION_MATCH) are initial values; calibrate
against the anchor set (see src/eval/calibration.py).
"""
from __future__ import annotations

from src.utils.config import FUSION_MATCH, FUSION_QUALITY, normalize_weights
from src.utils.schema import DimensionScore, MatchResult, QualityResult


def _blend(signals: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean over available signals, redistributing missing weights."""
    available = {k: v for k, v in signals.items() if v is not None and weights.get(k, 0) > 0}
    if not available:
        return 0.0
    w = normalize_weights({k: weights[k] for k in available})
    total = sum(available[k] * w[k] for k in available)
    return round(max(1.0, min(10.0, total)), 2)


def fuse_quality(
    rule_score: float,
    rule_dims: list[DimensionScore],
    llm_score: float | None,
    llm_dims: list[DimensionScore] | None = None,
    llm_reasons: list[str] | None = None,
) -> QualityResult:
    blended = _blend({"rule": rule_score, "llm": llm_score}, FUSION_QUALITY)
    dims = list(rule_dims)
    if llm_dims:
        dims.extend(llm_dims)
    reasons = list(llm_reasons or [])
    return QualityResult(
        score=blended,
        dimensions=dims,
        rule_score=round(rule_score, 2),
        llm_score=round(llm_score, 2) if llm_score is not None else None,
        reasons=reasons,
    )


def fuse_match(
    rule_score: float,
    ml_score: float | None,
    llm_score: float | None,
    rule_dims: list[DimensionScore] | None = None,
    llm_dims: list[DimensionScore] | None = None,
    llm_reasons: list[str] | None = None,
) -> MatchResult:
    blended = _blend(
        {"rule": rule_score, "ml": ml_score, "llm": llm_score}, FUSION_MATCH
    )
    dims = list(rule_dims or [])
    if llm_dims:
        dims.extend(llm_dims)
    return MatchResult(
        score=blended,
        dimensions=dims,
        rule_score=round(rule_score, 2),
        ml_score=round(ml_score, 2) if ml_score is not None else None,
        llm_score=round(llm_score, 2) if llm_score is not None else None,
        reasons=list(llm_reasons or []),
    )
