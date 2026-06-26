"""Tests for Layer 1: rule-based quality scoring & keyword coverage."""
from src.features.keyword_coverage import extract_jd_keywords, keyword_coverage
from src.features.quality_rules import score_quality_rules
from src.parsing.parser import parse_jd, parse_resume


def _good_resume():
    return parse_resume(
        "Jane Doe\njane@email.com | +1-555-0100\n\n"
        "SUMMARY\nSenior engineer.\n\n"
        "EXPERIENCE\n- Led team of 5, lifted CTR 18% ($4.2M revenue)\n"
        "- Built pipeline serving 2M events/day\n\n"
        "EDUCATION\nB.S. CS, MIT\n\n"
        "SKILLS\nPython, PyTorch, Kubernetes, Spark, Docker, SQL\n"
    )


def _weak_resume():
    return parse_resume(
        "Bob\n\nEXPERIENCE\n- worked on stuff\n- helped with things\n\n"
        "SKILLS\nword, excel\n"
    )


def test_good_resume_scores_higher_than_weak():
    good_score, good_dims = score_quality_rules(_good_resume())
    weak_score, weak_dims = score_quality_rules(_weak_resume())
    assert good_score > weak_score, (good_score, weak_score)
    assert good_score >= 6.0
    assert weak_score <= 5.0


def test_quantification_detected():
    score, dims = score_quality_rules(_good_resume())
    qdim = next(d for d in dims if d.name == "quantification")
    assert qdim.score >= 8.0  # both bullets are quantified


def test_hard_fail_penalizes_missing_contact():
    # No email/phone => no_contact hard failure.
    r = parse_resume("Nobody\n\nEXPERIENCE\n- did something\n\nEDUCATION\ncollege\n\nSKILLS\nx\n")
    score, dims = score_quality_rules(r)
    hf = next(d for d in dims if d.name == "no_hard_fail")
    assert hf.score < 5.0


def test_keyword_coverage_matches_skills():
    jd = parse_jd(
        "Requirements:\n- Python\n- PyTorch\n- Kubernetes\n- Spark\n\n"
        "Responsibilities:\nBuild ML systems.\n"
    )
    kws = extract_jd_keywords(jd)
    assert "python" in kws or "pytorch" in kws
    score, matched, missing = keyword_coverage(_good_resume(), kws)
    assert score > 5.0
    assert "python" in matched


def test_keyword_coverage_weak_resume_low():
    jd = parse_jd("Requirements:\n- Python\n- PyTorch\n- Kubernetes\n- Spark\n- Docker\n")
    kws = extract_jd_keywords(jd)
    score, matched, missing = keyword_coverage(_weak_resume(), kws)
    assert score < 5.0
    assert len(missing) > 0
