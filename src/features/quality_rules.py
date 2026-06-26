"""Layer 1 (rules): deterministic, explainable resume-quality features.

Each feature produces a 0-10 sub-score. A weighted sum (weights from the rubric
in design.md §7.1) yields ``rule_quality_score``. These scores anchor the LLM
judge and prevent drift on structural hard-failures (missing contact, empty
sections, etc.).
"""
from __future__ import annotations

import re

from src.utils.schema import DimensionScore, ResumeSections

# Strong vs weak action verbs (English). Chinese equivalents checked via keywords.
_STRONG_VERBS_EN = {
    "led", "built", "designed", "drove", "owned", "launched", "architected",
    "shipped", "delivered", "mentored", "optimized", "scaled", "deployed",
    "created", "developed", "implemented", "reduced", "improved", "increased",
}
_WEAK_VERBS_EN = {"helped", "worked", "did", "made", "assisted", "involved", "participated"}
_STRONG_VERBS_ZH = ["主导", "带领", "设计", "建设", "上线", "重构", "优化", "提升", "降低", "部署", "指导", "交付"]
_WEAK_VERBS_ZH = ["帮忙", "协助", "做过", "参与", "处理过"]

# Rubric weights (design.md §7.1). Must sum to 1.0.
QUALITY_WEIGHTS: dict[str, float] = {
    "quantification": 0.25,
    "star_coverage": 0.20,
    "verb_strength": 0.10,
    "keyword_richness": 0.15,
    "completeness": 0.15,
    "length_sanity": 0.10,
    "no_hard_fail": 0.05,
}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%|\$\s?\d|£|€|¥|\d{2,}|\d+\s?(?:x|k|m|b|亿|万)", re.I)
_BULLET_RE = re.compile(r"^\s*[-•*•▪]")


def _split_bullets(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if _BULLET_RE.match(ln) or ln.strip().startswith(("•", "·", "-"))]


def _score_quantification(exp: str) -> tuple[float, str]:
    if not exp.strip():
        return 0.0, "no experience section"
    bullets = _split_bullets(exp) or [ln for ln in exp.splitlines() if ln.strip()]
    if not bullets:
        return 0.0, "no bullet points"
    with_num = sum(1 for b in bullets if _NUMBER_RE.search(b))
    ratio = with_num / len(bullets)
    score = round(min(10.0, ratio * 14.0), 1)  # ~70% bullets quantified => ~10
    reason = f"{with_num}/{len(bullets)} bullets quantified ({ratio:.0%})"
    return score, reason


def _score_star(exp: str, lang: str = "en") -> tuple[float, str]:
    """STAR: situation/task/action/result. Proxy: has bullets with verbs + numbers."""
    if not exp.strip():
        return 0.0, "no experience section"
    bullets = _split_bullets(exp)
    if not bullets:
        return 2.0, "no bullet structure"
    has_action = any(re.search(r"\b[A-Za-z]+\b", b) for b in bullets)
    if lang in ("zh", "mixed"):
        has_action = has_action or any(v in exp for v in _STRONG_VERBS_ZH)
    has_result = any(_NUMBER_RE.search(b) for b in bullets)
    score = 0.0
    if has_action:
        score += 4.0
    if has_result:
        score += 4.0
    if len(bullets) >= 3:
        score += 2.0
    elif len(bullets) >= 1:
        score += 1.0
    return min(10.0, score), f"action={has_action}, result={has_result}, bullets={len(bullets)}"


def _score_verbs(exp: str, lang: str) -> tuple[float, str]:
    if not exp.strip():
        return 0.0, "no experience section"
    low = exp.lower()
    strong = sum(1 for v in _STRONG_VERBS_EN if re.search(rf"\b{v}\b", low))
    weak = sum(1 for v in _WEAK_VERBS_EN if re.search(rf"\b{v}\b", low))
    if lang in ("zh", "mixed"):
        strong += sum(1 for v in _STRONG_VERBS_ZH if v in exp)
        weak += sum(1 for v in _WEAK_VERBS_ZH if v in exp)
    if strong + weak == 0:
        return 3.0, "no action verbs detected"
    ratio = strong / (strong + weak)
    score = round(ratio * 10.0, 1)
    return score, f"strong={strong}, weak={weak}"


def _score_keyword_richness(skills: str) -> tuple[float, str]:
    if not skills.strip():
        return 0.0, "no skills section"
    tokens = re.split(r"[,，;；\s/|、]+", skills)
    tokens = [t for t in tokens if len(t.strip()) >= 2]
    # 10+ distinct skills => near full marks
    score = round(min(10.0, len(tokens) * 1.0), 1)
    return score, f"{len(tokens)} skill tokens"


def _score_completeness(sections: ResumeSections) -> tuple[float, str]:
    expected = ["summary", "experience", "education", "skills"]
    present = [s for s in expected if s in sections.sections_found]
    # contact completeness
    has_contact = bool(sections.contact) or "@" in sections.raw_text
    score = round((len(present) / len(expected)) * 8.0 + (2.0 if has_contact else 0.0), 1)
    missing = [s for s in expected if s not in sections.sections_found]
    reason = f"present={present}; contact={'yes' if has_contact else 'no'}; missing={missing}"
    return min(10.0, score), reason


def _score_length(text: str, lang: str = "en") -> tuple[float, str]:
    words = len(text.split())
    chars = len(text)
    cjk_chars = sum(1 for c in text if "一" <= c <= "鿿")
    # For Chinese, char count is the meaningful unit (whitespace split undercounts).
    if lang in ("zh", "mixed") and cjk_chars > 10:
        if cjk_chars < 80:
            return 2.0, f"too short ({cjk_chars} cjk chars)"
        if cjk_chars > 3000:
            return 5.0, f"overly long ({cjk_chars} cjk chars)"
        if 150 <= cjk_chars <= 1500:
            return 9.0, f"good length ({cjk_chars} cjk chars)"
        return 6.0, f"acceptable length ({cjk_chars} cjk chars)"
    # English: 200-900 words ideal.
    if words < 50 and chars < 200:
        return 2.0, f"too short ({words}w/{chars}c)"
    if words > 1500:
        return 5.0, f"overly long ({words}w)"
    if 200 <= words <= 900 or (300 <= chars <= 2500):
        return 9.0, f"good length ({words}w/{chars}c)"
    return 6.0, f"acceptable length ({words}w/{chars}c)"


def _score_no_hard_fail(sections: ResumeSections) -> tuple[float, str]:
    fails: list[str] = []
    if not sections.name.strip():
        fails.append("no_name")
    if not (sections.contact.strip() or "@" in sections.raw_text):
        fails.append("no_contact")
    if not sections.experience.strip():
        fails.append("no_experience")
    # time-gap / typo detection is intentionally left to the LLM judge.
    if not fails:
        return 10.0, "no structural hard failures"
    return 2.0, f"hard failures: {fails}"


def score_quality_rules(sections: ResumeSections) -> tuple[float, list[DimensionScore]]:
    """Return (overall 0-10 rule score, per-dimension breakdown)."""
    exp = sections.experience or sections.raw_text
    lang = sections.language

    raw: dict[str, tuple[float, str]] = {
        "quantification": _score_quantification(exp),
        "star_coverage": _score_star(exp, lang),
        "verb_strength": _score_verbs(exp, lang),
        "keyword_richness": _score_keyword_richness(sections.skills),
        "completeness": _score_completeness(sections),
        "length_sanity": _score_length(sections.raw_text, lang),
        "no_hard_fail": _score_no_hard_fail(sections),
    }

    dims: list[DimensionScore] = []
    total = 0.0
    for name, (sc, reason) in raw.items():
        w = QUALITY_WEIGHTS[name]
        dims.append(DimensionScore(name=name, score=sc, weight=w, reason=reason))
        total += sc * w

    return round(total, 2), dims
