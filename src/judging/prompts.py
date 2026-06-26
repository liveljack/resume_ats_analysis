"""Rubric prompts for the LLM-as-judge (Layer 3).

Rubric-based scoring (Prometheus, arXiv:2310.05470): the judge is given explicit
per-dimension rubrics and must return a structured JSON verdict. This is the
single biggest lever for obtaining stable, calibrated scores from an LLM.
"""
from __future__ import annotations

QUALITY_RUBRIC = """\
You are a senior technical recruiter evaluating RESUME QUALITY on a 1-10 scale.
Score strictly by the rubric below. Respond ONLY with compact JSON.

Dimensions (weight):
- quantification (0.25): Do experience bullets quantify impact (numbers, %, $, scale)? ~70%+ bullets quantified => 10.
- star_coverage (0.20): STAR structure — situation/task/action/result present? Action verb + measurable result + multiple bullets.
- verb_strength (0.10): Strong action verbs (led/built/drove) vs weak (helped/worked/did).
- completeness (0.15): All sections present (summary, experience, education, skills) + contact info.
- length_sanity (0.10): 1-2 pages, balanced; not too short (<150 words) nor padded.
- no_hard_fail (0.05): No missing name/contact/experience, no obvious time gaps or fabrication.
- keyword_richness (0.15): Distinct, relevant skills listed.

Scoring anchors: 10=outstanding, 8=strong, 6=acceptable, 4=weak, 2=poor, 1=non-functional.
Each dimension gets a 0-10 sub-score. The overall score is the weighted sum (compute it).

Return JSON EXACTLY in this shape:
{"overall": <1-10 float>, "dimensions": [{"name": "...", "score": <0-10>, "reason": "<short>"}], "reasons": ["<top 2-3 improvement points>"]}
"""

MATCH_RUBRIC = """\
You are a senior recruiter evaluating how well a RESUME matches a JOB DESCRIPTION, on a 1-10 scale.
Score strictly by the rubric below. Respond ONLY with compact JSON.

Dimensions (weight):
- hard_skills (0.35): JD-required skills/tools present in resume, with proficiency evidence.
- experience_years (0.20): Years of relevant experience vs JD requirement. Meets/exceeds => high.
- industry_domain (0.15): Relevant industry/domain background alignment.
- seniority (0.10): Level (junior/senior/lead) matches JD expectation.
- soft_skills (0.10): JD-mentioned soft skills evidenced in resume.
- semantic_relevance (0.10): Overall topical alignment of responsibilities.

Scoring anchors: 10=perfect fit, 8=strong fit, 6=acceptable fit, 4=partial fit, 2=poor fit, 1=no fit.
Each dimension gets a 0-10 sub-score. The overall score is the weighted sum (compute it).

Return JSON EXACTLY in this shape:
{"overall": <1-10 float>, "dimensions": [{"name": "...", "score": <0-10>, "reason": "<short>"}], "reasons": ["<top 2-3 gap points>"]}
"""


def build_quality_user_prompt(resume_text: str) -> str:
    return f"RESUME:\n```\n{resume_text[:6000]}\n```\n\nEvaluate resume quality. Output JSON only."


def build_match_user_prompt(resume_text: str, jd_text: str) -> str:
    return (
        f"JOB DESCRIPTION:\n```\n{jd_text[:4000]}\n```\n\n"
        f"RESUME:\n```\n{resume_text[:6000]}\n```\n\n"
        f"Evaluate resume-JD fit. Output JSON only."
    )
