"""Pydantic schemas for parsed resumes, scores, and final results."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ResumeSections(BaseModel):
    """Structured sections extracted from a resume."""

    name: str = ""
    contact: str = ""
    summary: str = ""
    experience: str = ""
    education: str = ""
    skills: str = ""
    raw_text: str = ""
    language: Literal["zh", "en", "mixed"] = "en"
    # detected section headings present (for completeness check)
    sections_found: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    """Structured JD."""

    raw_text: str = ""
    requirements: str = ""
    responsibilities: str = ""
    language: Literal["zh", "en", "mixed"] = "en"


class DimensionScore(BaseModel):
    """A single rubric dimension's score."""

    name: str
    score: float  # 0-10
    weight: float
    reason: str = ""


class QualityResult(BaseModel):
    """Resume quality scoring output."""

    score: float  # 1-10 final
    dimensions: list[DimensionScore] = Field(default_factory=list)
    rule_score: float = 0.0
    llm_score: float | None = None
    reasons: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    """Resume-JD match scoring output."""

    score: float  # 1-10 final
    dimensions: list[DimensionScore] = Field(default_factory=list)
    rule_score: float = 0.0
    ml_score: float | None = None
    llm_score: float | None = None
    reasons: list[str] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Full pipeline result for one resume (+optional JD)."""

    resume_id: str = ""
    quality: QualityResult
    match: MatchResult | None = None
    passed: bool | None = None  # binary 达标 judgment when thresholds applied
