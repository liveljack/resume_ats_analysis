"""End-to-end pipeline: parse → rules → ML match → LLM judge → fusion.

``analyze_resume`` runs all layers for a single resume (+ optional JD).
Layers degrade gracefully: missing LLM key or heavy ML deps simply drop that
signal, and the fusion layer redistributes weights.
"""
from __future__ import annotations

from pathlib import Path

from src.features.keyword_coverage import extract_jd_keywords, keyword_coverage
from src.features.quality_rules import score_quality_rules
from src.fusion.fusion import fuse_match, fuse_quality
from src.judging.llm_judge import judge_match, judge_quality
from src.matching.matcher import MatchScorer
from src.parsing.parser import extract_text, parse_jd, parse_resume
from src.utils.config import MATCH_PASS_THRESHOLD, QUALITY_PASS_THRESHOLD
from src.utils.schema import AnalysisResult, MatchResult, QualityResult, ResumeSections

# Lazily-shared match scorer (loading an embedding model is expensive).
_match_scorer: MatchScorer | None = None


def _get_match_scorer() -> MatchScorer:
    global _match_scorer
    if _match_scorer is None:
        _match_scorer = MatchScorer()
    return _match_scorer


def analyze_text(
    resume_text: str,
    jd_text: str | None = None,
    resume_id: str = "",
    use_ml: bool = True,
    use_llm: bool = True,
) -> AnalysisResult:
    """Analyze a resume (and optionally a JD) from raw text."""
    sections = parse_resume(resume_text, resume_id=resume_id)

    # --- Layer 1: rules ---
    rule_quality, rule_q_dims = score_quality_rules(sections)

    # --- Layer 3: LLM judge (quality) ---
    llm_q_out = judge_quality(resume_text) if use_llm else None
    llm_q_score = llm_q_out.score if llm_q_out else None

    quality = fuse_quality(
        rule_score=rule_quality,
        rule_dims=rule_q_dims,
        llm_score=llm_q_score,
        llm_dims=llm_q_out.dimensions if llm_q_out else None,
        llm_reasons=llm_q_out.reasons if llm_q_out else None,
    )

    # --- Match (only if JD provided) ---
    match: MatchResult | None = None
    passed: bool | None = None
    if jd_text:
        jd = parse_jd(jd_text)
        keywords = extract_jd_keywords(jd)
        kw_score, matched, missing = keyword_coverage(sections, keywords)

        # Layer 2: ML semantic match.
        # Compare the resume's *relevant* content (skills + experience) against the
        # JD's *requirements + responsibilities* rather than full text — this focuses
        # the similarity on signal and substantially improves the tfidf fallback.
        resume_focus = " ".join([sections.skills, sections.experience]).strip() or resume_text
        jd_focus = " ".join([jd.requirements, jd.responsibilities]).strip() or jd_text
        ml_score: float | None = None
        if use_ml:
            try:
                scorer = _get_match_scorer()
                ml_score, _sim, _backend = scorer.score(resume_focus, jd_focus)
            except Exception:
                ml_score = None

        # Layer 3: LLM judge (match)
        llm_m_out = judge_match(resume_text, jd_text) if use_llm else None
        llm_m_score = llm_m_out.score if llm_m_out else None

        from src.utils.schema import DimensionScore
        kw_dim = [DimensionScore(
            name="keyword_coverage", score=kw_score, weight=0.0,
            reason=f"matched {len(matched)}/{len(keywords)}; missing={missing[:5]}",
        )]

        match = fuse_match(
            rule_score=kw_score,
            ml_score=ml_score,
            llm_score=llm_m_score,
            rule_dims=kw_dim,
            llm_dims=llm_m_out.dimensions if llm_m_out else None,
            llm_reasons=llm_m_out.reasons if llm_m_out else None,
        )

        passed = bool(quality.score >= QUALITY_PASS_THRESHOLD
                      and match.score >= MATCH_PASS_THRESHOLD)

    return AnalysisResult(
        resume_id=resume_id or sections.name or "unknown",
        quality=quality,
        match=match,
        passed=passed,
    )


def analyze_file(
    resume_path: str | Path,
    jd_path: str | Path | None = None,
    use_ml: bool = True,
    use_llm: bool = True,
) -> AnalysisResult:
    """Analyze resume (and optional JD) from file paths (.txt or .pdf)."""
    resume_text = extract_text(resume_path)
    jd_text = extract_text(jd_path) if jd_path else None
    resume_id = Path(resume_path).stem
    return analyze_text(resume_text, jd_text, resume_id=resume_id,
                        use_ml=use_ml, use_llm=use_llm)
